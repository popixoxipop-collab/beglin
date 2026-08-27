// safetensors_load.c -- see safetensors_load.h for the format writeup and scope note.
//
// JSON scanner: hand-rolled, scoped to safetensors' exact header grammar (one level of
// nesting, plain-identifier keys, string/int/int-array values only) -- not a general JSON
// parser. Every read advances a bounds-checked cursor into the mmap'd header bytes; a
// malformed header is a FATAL with a specific reason, matching gguf_load.c's own doctrine.

#include "safetensors_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

typedef struct { const char *p, *end, *start; const char *path; } Cur;

static void cur_skip_ws(Cur *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}
static void cur_expect(Cur *c, char ch) {
    cur_skip_ws(c);
    if (c->p >= c->end || *c->p != ch) {
        fprintf(stderr, "FATAL: safetensors_load: %s: header JSON expected '%c' at offset %ld\n",
                c->path, ch, (long)(c->p - c->start));
        exit(1);
    }
    c->p++;
}
static int cur_peek_is(Cur *c, char ch) {
    cur_skip_ws(c);
    return c->p < c->end && *c->p == ch;
}

// Parses a JSON string literal (cursor already positioned at the opening '"'), returns a
// malloc'd NUL-terminated copy of its (unescaped) content. Handles the standard JSON escapes;
// \uXXXX is decoded to raw UTF-8 bytes for the BMP range (safetensors headers are
// machine-generated ASCII in every real file this session inspected -- full surrogate-pair
// handling is not needed for that grammar, and this FATALs rather than silently mis-decoding
// if one is ever encountered).
static char *cur_string(Cur *c) {
    cur_skip_ws(c);
    if (c->p >= c->end || *c->p != '"') {
        fprintf(stderr, "FATAL: safetensors_load: %s: header JSON expected string\n", c->path);
        exit(1);
    }
    c->p++;
    size_t cap = 64, len = 0;
    char *buf = malloc(cap);
    while (1) {
        if (c->p >= c->end) { fprintf(stderr, "FATAL: safetensors_load: %s: unterminated string\n", c->path); exit(1); }
        char ch = *c->p++;
        if (ch == '"') break;
        if (ch == '\\') {
            if (c->p >= c->end) { fprintf(stderr, "FATAL: safetensors_load: %s: unterminated escape\n", c->path); exit(1); }
            char esc = *c->p++;
            switch (esc) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'n': ch = '\n'; break;
                case 't': ch = '\t'; break;
                case 'r': ch = '\r'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'u':
                    fprintf(stderr, "FATAL: safetensors_load: %s: \\u escape not supported (not expected in a real header)\n", c->path);
                    exit(1);
                default:
                    fprintf(stderr, "FATAL: safetensors_load: %s: unknown string escape '\\%c'\n", c->path, esc);
                    exit(1);
            }
        }
        if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[len++] = ch;
    }
    buf[len] = '\0';
    return buf;
}

static uint64_t cur_u64(Cur *c) {
    cur_skip_ws(c);
    const char *start = c->p;
    int neg = 0;
    if (c->p < c->end && *c->p == '-') { neg = 1; c->p++; }  // data_offsets/shape are non-negative in practice; reject below
    if (c->p >= c->end || *c->p < '0' || *c->p > '9') {
        fprintf(stderr, "FATAL: safetensors_load: %s: header JSON expected integer\n", c->path);
        exit(1);
    }
    uint64_t v = 0;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') { v = v * 10 + (uint64_t)(*c->p - '0'); c->p++; }
    if (neg) {
        fprintf(stderr, "FATAL: safetensors_load: %s: negative integer not expected in shape/data_offsets\n", c->path);
        exit(1);
    }
    (void)start;
    return v;
}

static uint64_t *cur_u64_array(Cur *c, uint32_t *out_n) {
    cur_expect(c, '[');
    size_t cap = 8, n = 0;
    uint64_t *arr = malloc(cap * sizeof(uint64_t));
    if (!cur_peek_is(c, ']')) {
        while (1) {
            if (n >= cap) { cap *= 2; arr = realloc(arr, cap * sizeof(uint64_t)); }
            arr[n++] = cur_u64(c);
            cur_skip_ws(c);
            if (cur_peek_is(c, ',')) { c->p++; continue; }
            break;
        }
    }
    cur_expect(c, ']');
    *out_n = (uint32_t)n;
    return arr;
}

// Skips a JSON value of ANY type without materializing it (used for "__metadata__", whose
// value is a flat string->string object this parser doesn't need, and for any future header
// key not yet known -- forward-compatible rather than FATALing on an unrecognized key).
static void cur_skip_value(Cur *c) {
    cur_skip_ws(c);
    if (c->p >= c->end) { fprintf(stderr, "FATAL: safetensors_load: %s: unexpected end of header\n", c->path); exit(1); }
    char ch = *c->p;
    if (ch == '"') { free(cur_string(c)); return; }
    if (ch == '{') {
        c->p++;
        cur_skip_ws(c);
        if (!cur_peek_is(c, '}')) {
            while (1) {
                free(cur_string(c));
                cur_expect(c, ':');
                cur_skip_value(c);
                cur_skip_ws(c);
                if (cur_peek_is(c, ',')) { c->p++; continue; }
                break;
            }
        }
        cur_expect(c, '}');
        return;
    }
    if (ch == '[') {
        c->p++;
        cur_skip_ws(c);
        if (!cur_peek_is(c, ']')) {
            while (1) {
                cur_skip_value(c);
                cur_skip_ws(c);
                if (cur_peek_is(c, ',')) { c->p++; continue; }
                break;
            }
        }
        cur_expect(c, ']');
        return;
    }
    // number, true, false, null -- consume up to the next structural character
    while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' &&
           *c->p != ' ' && *c->p != '\t' && *c->p != '\n' && *c->p != '\r') c->p++;
}

static SafetensorsType parse_dtype(const char *s) {
    if (!strcmp(s, "F64")) return ST_TYPE_F64;
    if (!strcmp(s, "F32")) return ST_TYPE_F32;
    if (!strcmp(s, "F16")) return ST_TYPE_F16;
    if (!strcmp(s, "BF16")) return ST_TYPE_BF16;
    if (!strcmp(s, "I64")) return ST_TYPE_I64;
    if (!strcmp(s, "I32")) return ST_TYPE_I32;
    if (!strcmp(s, "I16")) return ST_TYPE_I16;
    if (!strcmp(s, "I8")) return ST_TYPE_I8;
    if (!strcmp(s, "U8")) return ST_TYPE_U8;
    if (!strcmp(s, "BOOL")) return ST_TYPE_BOOL;
    return ST_TYPE_UNKNOWN;
}

const char *safetensors_type_name(SafetensorsType t) {
    switch (t) {
        case ST_TYPE_F64: return "F64"; case ST_TYPE_F32: return "F32";
        case ST_TYPE_F16: return "F16"; case ST_TYPE_BF16: return "BF16";
        case ST_TYPE_I64: return "I64"; case ST_TYPE_I32: return "I32";
        case ST_TYPE_I16: return "I16"; case ST_TYPE_I8: return "I8";
        case ST_TYPE_U8: return "U8"; case ST_TYPE_BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

SafetensorsFile *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "FATAL: safetensors_load: could not open %s (%s)\n", path, strerror(errno)); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "FATAL: safetensors_load: fstat %s failed\n", path); exit(1); }
    if ((uint64_t)st.st_size < 8) {
        fprintf(stderr, "FATAL: safetensors_load: %s too small to hold a header length (%lld bytes)\n", path, (long long)st.st_size);
        exit(1);
    }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { fprintf(stderr, "FATAL: safetensors_load: mmap %s failed\n", path); exit(1); }

    SafetensorsFile *f = malloc(sizeof *f);
    f->fd = -1;  // already closed after mmap, matches gguf_load.c's own convention
    f->base = (uint8_t *)base;
    f->file_size = (size_t)st.st_size;

    uint64_t header_len;
    memcpy(&header_len, f->base, 8);  // little-endian host assumed, same as gguf_load.c
    if (header_len > f->file_size - 8) {
        fprintf(stderr, "FATAL: safetensors_load: %s: header_len=%llu extends past end of file (size=%zu)\n",
                path, (unsigned long long)header_len, f->file_size);
        exit(1);
    }
    f->data_start = 8 + header_len;

    Cur c = { (const char *)f->base + 8, (const char *)f->base + 8 + header_len, (const char *)f->base + 8, path };
    cur_expect(&c, '{');

    size_t cap = 64, n = 0;
    f->tensors = malloc(cap * sizeof(SafetensorsInfo));
    if (!cur_peek_is(&c, '}')) {
        while (1) {
            char *key = cur_string(&c);
            cur_expect(&c, ':');
            if (!strcmp(key, "__metadata__")) {
                free(key);
                cur_skip_value(&c);
            } else {
                cur_expect(&c, '{');
                char *dtype_str = NULL;
                uint64_t *shape = NULL; uint32_t shape_n = 0;
                uint64_t *offsets = NULL; uint32_t offsets_n = 0;
                if (!cur_peek_is(&c, '}')) {
                    while (1) {
                        char *field = cur_string(&c);
                        cur_expect(&c, ':');
                        if (!strcmp(field, "dtype")) { dtype_str = cur_string(&c); }
                        else if (!strcmp(field, "shape")) { free(shape); shape = cur_u64_array(&c, &shape_n); }
                        else if (!strcmp(field, "data_offsets")) { free(offsets); offsets = cur_u64_array(&c, &offsets_n); }
                        else { cur_skip_value(&c); }  // forward-compatible: ignore unknown fields
                        free(field);
                        cur_skip_ws(&c);
                        if (cur_peek_is(&c, ',')) { c.p++; continue; }
                        break;
                    }
                }
                cur_expect(&c, '}');
                if (!dtype_str || !offsets || offsets_n != 2) {
                    fprintf(stderr, "FATAL: safetensors_load: %s: tensor '%s' missing dtype or data_offsets\n", path, key);
                    exit(1);
                }
                if (shape_n > SAFETENSORS_MAX_DIMS) {
                    fprintf(stderr, "FATAL: safetensors_load: %s: tensor '%s' has %u dims (max %d)\n",
                            path, key, shape_n, SAFETENSORS_MAX_DIMS);
                    exit(1);
                }
                if (offsets[1] < offsets[0] || offsets[1] > (uint64_t)f->file_size - f->data_start) {
                    fprintf(stderr, "FATAL: safetensors_load: %s: tensor '%s' data_offsets [%llu,%llu] out of bounds (data section is %llu bytes)\n",
                            path, key, (unsigned long long)offsets[0], (unsigned long long)offsets[1],
                            (unsigned long long)(f->file_size - f->data_start));
                    exit(1);
                }
                if (n >= cap) { cap *= 2; f->tensors = realloc(f->tensors, cap * sizeof(SafetensorsInfo)); }
                SafetensorsInfo *t = &f->tensors[n++];
                t->name = key;  // ownership transferred, not freed below
                t->dtype = parse_dtype(dtype_str);
                if (t->dtype == ST_TYPE_UNKNOWN) {
                    fprintf(stderr, "FATAL: safetensors_load: %s: tensor '%s' has unrecognized dtype '%s'\n", path, key, dtype_str);
                    exit(1);
                }
                t->n_dims = shape_n;
                uint64_t nelem = 1;
                for (uint32_t d = 0; d < shape_n; d++) { t->shape[d] = shape[d]; nelem *= shape[d]; }
                for (uint32_t d = shape_n; d < SAFETENSORS_MAX_DIMS; d++) t->shape[d] = 1;
                t->n_elements = (shape_n == 0) ? 1 : nelem;  // safetensors allows 0-dim (scalar) tensors
                t->data_offset = f->data_start + offsets[0];
                t->n_bytes = offsets[1] - offsets[0];
                free(dtype_str); free(shape); free(offsets);
                key = NULL;  // ownership already transferred to t->name
            }
            cur_skip_ws(&c);
            if (cur_peek_is(&c, ',')) { c.p++; continue; }
            break;
        }
    }
    cur_expect(&c, '}');
    f->n_tensors = n;
    return f;
}

void safetensors_close(SafetensorsFile *f) {
    if (!f) return;
    for (uint64_t i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    munmap(f->base, f->file_size);
    free(f);
}

const SafetensorsInfo *safetensors_find_tensor(const SafetensorsFile *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++) if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}

const void *safetensors_tensor_data(const SafetensorsFile *f, const SafetensorsInfo *t) {
    return f->base + t->data_offset;
}
