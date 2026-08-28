// hf_config.c -- see hf_config.h for the format writeup and scope note.
//
// JSON scanner: hand-rolled, scoped to a flat top-level object -- not a general JSON parser.
// Every read advances a bounds-checked cursor into the loaded file bytes; a malformed document is
// a FATAL with a specific reason, matching gguf_load.c/safetensors_load.c's own doctrine.

#include "hf_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

// Internal per-entry type (distinct from the public HfValType in hf_config.h, which also has
// HF_TYPE_ABSENT -- not meaningful for a stored entry, only for a lookup miss).
typedef enum { HFV_STR, HFV_NUM, HFV_BOOL, HFV_NULL, HFV_OBJECT, HFV_ARRAY } HfEntryType;

typedef struct {
    char *key;          // malloc'd, NUL-terminated
    HfEntryType type;
    char *str_val;       // malloc'd, HFV_STR only
    double num_val;       // HFV_NUM only
    int bool_val;          // HFV_BOOL only
    HfConfig *obj_val;    // HFV_OBJECT only, and only when materialized (depth<=1); else NULL
} HfEntry;

struct HfConfig {
    HfEntry *entries;
    size_t n_entries;
    char *raw;    // malloc'd file contents, kept alive for the lifetime of str_val pointers... (unused,
                  // str_val is a separate copy, kept only so callers don't need to worry about it) --
                  // retained anyway as a defensive belt-and-suspenders in case of future changes.
};

typedef struct { const char *p, *end, *start; const char *path; } Cur;

static void cur_skip_ws(Cur *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}
static void cur_expect(Cur *c, char ch) {
    cur_skip_ws(c);
    if (c->p >= c->end || *c->p != ch) {
        fprintf(stderr, "FATAL: hf_config: %s: expected '%c' at offset %ld\n",
                c->path, ch, (long)(c->p - c->start));
        exit(1);
    }
    c->p++;
}
static int cur_peek_is(Cur *c, char ch) {
    cur_skip_ws(c);
    return c->p < c->end && *c->p == ch;
}

static char *cur_string(Cur *c) {
    cur_skip_ws(c);
    if (c->p >= c->end || *c->p != '"') {
        fprintf(stderr, "FATAL: hf_config: %s: expected string at offset %ld\n", c->path, (long)(c->p - c->start));
        exit(1);
    }
    c->p++;
    size_t cap = 64, len = 0;
    char *buf = malloc(cap);
    while (1) {
        if (c->p >= c->end) { fprintf(stderr, "FATAL: hf_config: %s: unterminated string\n", c->path); exit(1); }
        char ch = *c->p++;
        if (ch == '"') break;
        if (ch == '\\') {
            if (c->p >= c->end) { fprintf(stderr, "FATAL: hf_config: %s: unterminated escape\n", c->path); exit(1); }
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
                    fprintf(stderr, "FATAL: hf_config: %s: \\u escape not supported (not expected in a real config.json)\n", c->path);
                    exit(1);
                default:
                    fprintf(stderr, "FATAL: hf_config: %s: unknown string escape '\\%c'\n", c->path, esc);
                    exit(1);
            }
        }
        if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[len++] = ch;
    }
    buf[len] = '\0';
    return buf;
}

// Parses a JSON number (int or float, optional leading '-', optional exponent) via strtod --
// handles the full JSON number grammar (896, 1e-06, 1000000.0, -1) without hand-rolling exponent
// arithmetic.
static double cur_number(Cur *c) {
    cur_skip_ws(c);
    const char *start = c->p;
    if (c->p < c->end && *c->p == '-') c->p++;
    if (c->p >= c->end || *c->p < '0' || *c->p > '9') {
        fprintf(stderr, "FATAL: hf_config: %s: expected number at offset %ld\n", c->path, (long)(start - c->start));
        exit(1);
    }
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++;
    if (c->p < c->end && *c->p == '.') { c->p++; while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++; }
    if (c->p < c->end && (*c->p == 'e' || *c->p == 'E')) {
        c->p++;
        if (c->p < c->end && (*c->p == '+' || *c->p == '-')) c->p++;
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9') c->p++;
    }
    char *endp = NULL;
    double v = strtod(start, &endp);
    (void)endp;   // strtod may read past c->p only in the sense of its own internal scan; c->p is
                  // already correctly advanced by the loop above, which matches JSON's own grammar
    return v;
}

static int cur_bool(Cur *c) {
    cur_skip_ws(c);
    if (c->p + 4 <= c->end && !memcmp(c->p, "true", 4)) { c->p += 4; return 1; }
    if (c->p + 5 <= c->end && !memcmp(c->p, "false", 5)) { c->p += 5; return 0; }
    fprintf(stderr, "FATAL: hf_config: %s: expected true/false at offset %ld\n", c->path, (long)(c->p - c->start));
    exit(1);
}

// Skips a JSON value of ANY type without materializing it -- used for nested objects/arrays this
// project doesn't need (e.g. "architectures": [...], a present "rope_scaling": {...}).
static void cur_skip_value(Cur *c) {
    cur_skip_ws(c);
    if (c->p >= c->end) { fprintf(stderr, "FATAL: hf_config: %s: unexpected end of document\n", c->path); exit(1); }
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
    if (ch == 't' || ch == 'f') { cur_bool(c); return; }
    if (ch == 'n') {
        if (c->p + 4 <= c->end && !memcmp(c->p, "null", 4)) { c->p += 4; return; }
        fprintf(stderr, "FATAL: hf_config: %s: expected null at offset %ld\n", c->path, (long)(c->p - c->start));
        exit(1);
    }
    cur_number(c);
}

// Parses a `{ "key": value, ... }` object starting at c->p (which must be positioned at the
// opening '{'), consuming through the matching '}'. `depth` is the depth of THIS object being
// parsed (0 for the top-level document). A nested object value is materialized recursively via a
// nested parse_object() call only when its own depth (depth+1) is <=1 -- i.e. only direct
// children of the top-level document get their fields read; a grandchild object is still typed
// correctly (HFV_OBJECT) but its content is skipped, matching every real shape found in this
// project (rope_scaling/weight_map leaves are scalars/strings, never a further nested object).
// Returned HfConfig's `raw` is left NULL (it doesn't own a separate copy of the file bytes --
// the top-level caller, hf_config_open(), fills in the real owning buffer after this returns).
static HfConfig *parse_object(Cur *c, int depth) {
    cur_expect(c, '{');

    HfConfig *cfg = malloc(sizeof *cfg);
    cfg->raw = NULL;
    cfg->n_entries = 0;
    size_t cap = 16;
    cfg->entries = malloc(cap * sizeof(HfEntry));

    if (!cur_peek_is(c, '}')) {
        while (1) {
            char *key = cur_string(c);
            cur_expect(c, ':');
            cur_skip_ws(c);
            if (cfg->n_entries >= cap) { cap *= 2; cfg->entries = realloc(cfg->entries, cap * sizeof(HfEntry)); }
            HfEntry *e = &cfg->entries[cfg->n_entries++];
            e->key = key;
            e->str_val = NULL;
            e->obj_val = NULL;
            char ch = c->p < c->end ? *c->p : '\0';
            if (ch == '"') { e->type = HFV_STR; e->str_val = cur_string(c); }
            else if (ch == '{') {
                e->type = HFV_OBJECT;
                if (depth <= 0) e->obj_val = parse_object(c, depth + 1);
                else cur_skip_value(c);
            }
            else if (ch == '[') { e->type = HFV_ARRAY; cur_skip_value(c); }
            else if (ch == 't' || ch == 'f') { e->type = HFV_BOOL; e->bool_val = cur_bool(c); }
            else if (ch == 'n') {
                if (c->p + 4 <= c->end && !memcmp(c->p, "null", 4)) { c->p += 4; e->type = HFV_NULL; }
                else { fprintf(stderr, "FATAL: hf_config: %s: expected null at offset %ld\n", c->path, (long)(c->p - c->start)); exit(1); }
            }
            else { e->type = HFV_NUM; e->num_val = cur_number(c); }
            cur_skip_ws(c);
            if (cur_peek_is(c, ',')) { c->p++; continue; }
            break;
        }
    }
    cur_expect(c, '}');
    return cfg;
}

HfConfig *hf_config_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: hf_config: could not open %s (%s)\n", path, strerror(errno)); exit(1); }
    struct stat st;
    if (fstat(fileno(f), &st) != 0) { fprintf(stderr, "FATAL: hf_config: fstat %s failed\n", path); exit(1); }
    if (st.st_size <= 0) { fprintf(stderr, "FATAL: hf_config: %s is empty\n", path); exit(1); }
    char *raw = malloc((size_t)st.st_size + 1);
    size_t nread = fread(raw, 1, (size_t)st.st_size, f);
    fclose(f);
    if (nread != (size_t)st.st_size) {
        fprintf(stderr, "FATAL: hf_config: %s: short read (%zu of %lld bytes)\n", path, nread, (long long)st.st_size);
        exit(1);
    }
    raw[nread] = '\0';

    Cur c = { raw, raw + nread, raw, path };
    HfConfig *cfg = parse_object(&c, 0);
    cfg->raw = raw;
    return cfg;
}

void hf_config_close(HfConfig *c) {
    if (!c) return;
    for (size_t i = 0; i < c->n_entries; i++) {
        free(c->entries[i].key);
        free(c->entries[i].str_val);
        if (c->entries[i].type == HFV_OBJECT && c->entries[i].obj_val) hf_config_close(c->entries[i].obj_val);
    }
    free(c->entries);
    free(c->raw);   // NULL for nested (non-top-level) configs -- free(NULL) is a documented no-op
    free(c);
}

static const HfEntry *find(const HfConfig *c, const char *key) {
    for (size_t i = 0; i < c->n_entries; i++) if (!strcmp(c->entries[i].key, key)) return &c->entries[i];
    return NULL;
}

int hf_config_has_key(const HfConfig *c, const char *key) { return find(c, key) != NULL; }

HfValType hf_config_key_type(const HfConfig *c, const char *key) {
    const HfEntry *e = find(c, key);
    if (!e) return HF_TYPE_ABSENT;
    switch (e->type) {
        case HFV_STR:    return HF_TYPE_STR;
        case HFV_NUM:    return HF_TYPE_NUM;
        case HFV_BOOL:   return HF_TYPE_BOOL;
        case HFV_NULL:   return HF_TYPE_NULL;
        case HFV_OBJECT: return HF_TYPE_OBJECT;
        case HFV_ARRAY:  return HF_TYPE_ARRAY;
    }
    return HF_TYPE_ABSENT; // unreachable, silences -Wreturn-type on some compilers
}

size_t hf_config_n_entries(const HfConfig *c) { return c->n_entries; }
const char *hf_config_entry_key(const HfConfig *c, size_t i) { return c->entries[i].key; }

const HfConfig *hf_config_get_object(const HfConfig *c, const char *key) {
    const HfEntry *e = find(c, key);
    if (!e) return NULL;
    if (e->type != HFV_OBJECT) { fprintf(stderr, "FATAL: hf_config: key '%s' is not an object\n", key); exit(1); }
    if (!e->obj_val) {
        fprintf(stderr, "FATAL: hf_config: key '%s' is an object nested too deep to read (depth>=2 unsupported)\n", key);
        exit(1);
    }
    return e->obj_val;
}

int hf_config_get_i64(const HfConfig *c, const char *key, int64_t *out) {
    const HfEntry *e = find(c, key);
    if (!e) return 0;
    if (e->type != HFV_NUM) { fprintf(stderr, "FATAL: hf_config: key '%s' is not a number\n", key); exit(1); }
    *out = (int64_t)e->num_val;
    return 1;
}
int hf_config_get_f64(const HfConfig *c, const char *key, double *out) {
    const HfEntry *e = find(c, key);
    if (!e) return 0;
    if (e->type != HFV_NUM) { fprintf(stderr, "FATAL: hf_config: key '%s' is not a number\n", key); exit(1); }
    *out = e->num_val;
    return 1;
}
int hf_config_get_bool(const HfConfig *c, const char *key, int *out) {
    const HfEntry *e = find(c, key);
    if (!e) return 0;
    if (e->type != HFV_BOOL) { fprintf(stderr, "FATAL: hf_config: key '%s' is not a bool\n", key); exit(1); }
    *out = e->bool_val;
    return 1;
}
int hf_config_get_str(const HfConfig *c, const char *key, const char **out_ptr) {
    const HfEntry *e = find(c, key);
    if (!e) return 0;
    if (e->type != HFV_STR) { fprintf(stderr, "FATAL: hf_config: key '%s' is not a string\n", key); exit(1); }
    *out_ptr = e->str_val;
    return 1;
}
