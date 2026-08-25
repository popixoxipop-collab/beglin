// gguf_cache.c -- see gguf_cache.h. Format (all little-endian, host-native since this cache
// never leaves the machine that wrote it -- see the magic-string version check as the only
// portability guard this needs):
//
//   header (32 bytes): magic[8]="BEGLINC1", src_size u64, src_mtime u64 (seconds), n_tensors u32,
//                       reserved u32
//   directory[n_tensors]: GgufCacheEntry, packed, in file order
//   data section: each tensor's data, then its scales (if any), each 64-byte aligned from the
//                 start of the file (matches this codebase's existing alignment convention for
//                 K_Q4G64/SME2 buffers elsewhere)

#include "gguf_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_CACHE_MAGIC "BEGLINC1"

typedef struct {
    char magic[8];
    uint64_t src_size;
    uint64_t src_mtime;
    uint32_t n_tensors;
    uint32_t reserved;
} GgufCacheHeader;

struct GgufCacheFile {
    uint8_t *base;
    size_t size;
    const GgufCacheHeader *hdr;
    const GgufCacheEntry *dir;
};

static uint64_t align64(uint64_t x) { return (x + 63) & ~(uint64_t)63; }

int gguf_cache_is_valid(const char *cache_path, const char *src_gguf_path) {
    struct stat src_st;
    if (stat(src_gguf_path, &src_st) != 0) return 0;
    FILE *f = fopen(cache_path, "rb");
    if (!f) return 0;
    GgufCacheHeader h;
    size_t nread = fread(&h, sizeof h, 1, f);
    fclose(f);
    if (nread != 1) return 0;
    if (memcmp(h.magic, GGUF_CACHE_MAGIC, 8) != 0) return 0;
    return h.src_size == (uint64_t)src_st.st_size && h.src_mtime == (uint64_t)src_st.st_mtime;
}

GgufCacheFile *gguf_cache_open(const char *cache_path) {
    int fd = open(cache_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "FATAL: gguf_cache_open: could not open %s\n", cache_path); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "FATAL: gguf_cache_open: fstat %s failed\n", cache_path); exit(1); }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { fprintf(stderr, "FATAL: gguf_cache_open: mmap %s failed\n", cache_path); exit(1); }
    GgufCacheFile *c = malloc(sizeof *c);
    c->base = (uint8_t *)base;
    c->size = (size_t)st.st_size;
    c->hdr = (const GgufCacheHeader *)c->base;
    if (memcmp(c->hdr->magic, GGUF_CACHE_MAGIC, 8) != 0) {
        fprintf(stderr, "FATAL: gguf_cache_open: %s bad magic (caller should have checked gguf_cache_is_valid() first)\n", cache_path);
        exit(1);
    }
    c->dir = (const GgufCacheEntry *)(c->base + sizeof(GgufCacheHeader));
    return c;
}

uint32_t gguf_cache_count(const GgufCacheFile *c) { return c->hdr->n_tensors; }
const GgufCacheEntry *gguf_cache_entry(const GgufCacheFile *c, uint32_t i) { return &c->dir[i]; }
const uint8_t *gguf_cache_base(const GgufCacheFile *c) { return c->base; }

struct GgufCacheWriter {
    FILE *f;
    char *path;
    uint64_t src_size, src_mtime;
    uint32_t n_tensors, n_written;
    GgufCacheEntry *entries;
    uint64_t cursor;  // next-write absolute file offset, always 64-aligned
};

GgufCacheWriter *gguf_cache_writer_open(const char *cache_path, const char *src_gguf_path, uint32_t n_tensors) {
    struct stat src_st;
    if (stat(src_gguf_path, &src_st) != 0) {
        fprintf(stderr, "FATAL: gguf_cache_writer_open: stat %s failed\n", src_gguf_path); exit(1);
    }
    FILE *f = fopen(cache_path, "wb");
    if (!f) { fprintf(stderr, "FATAL: gguf_cache_writer_open: could not create %s\n", cache_path); exit(1); }
    GgufCacheWriter *w = malloc(sizeof *w);
    w->f = f;
    w->path = strdup(cache_path);
    w->src_size = (uint64_t)src_st.st_size;
    w->src_mtime = (uint64_t)src_st.st_mtime;
    w->n_tensors = n_tensors; w->n_written = 0;
    w->entries = calloc(n_tensors ? n_tensors : 1, sizeof(GgufCacheEntry));
    uint64_t dir_start = sizeof(GgufCacheHeader);
    uint64_t data_start = align64(dir_start + (uint64_t)n_tensors * sizeof(GgufCacheEntry));
    // Reserve header+directory space now (real content written in _close(), once every
    // entry's data offset is known) and seek straight to the aligned data start -- avoids
    // buffering the whole data section in memory for a second pass.
    if (fseeko(f, (off_t)data_start, SEEK_SET) != 0) {
        fprintf(stderr, "FATAL: gguf_cache_writer_open: fseek failed on %s\n", cache_path); exit(1);
    }
    w->cursor = data_start;
    return w;
}

static void write_aligned(GgufCacheWriter *w, const void *buf, uint64_t nbytes, uint64_t *out_off) {
    *out_off = w->cursor;
    if (nbytes) {
        size_t nw = fwrite(buf, 1, (size_t)nbytes, w->f);
        if (nw != nbytes) { fprintf(stderr, "FATAL: gguf_cache writer: short write\n"); exit(1); }
    }
    uint64_t padded = align64(nbytes);
    if (padded > nbytes) {
        static const uint8_t zeros[64] = {0};
        fwrite(zeros, 1, (size_t)(padded - nbytes), w->f);
    }
    w->cursor += padded;
}

void gguf_cache_writer_add(GgufCacheWriter *w, const char *name, int kind, int out, int in, int ng,
                            const void *data, uint64_t data_bytes,
                            const void *scales, uint64_t scales_bytes) {
    if (w->n_written >= w->n_tensors) { fprintf(stderr, "FATAL: gguf_cache writer: too many tensors added\n"); exit(1); }
    GgufCacheEntry *e = &w->entries[w->n_written++];
    snprintf(e->name, sizeof e->name, "%s", name);
    e->kind = kind; e->out = out; e->in = in; e->ng = ng;
    write_aligned(w, data, data_bytes, &e->data_offset);
    e->data_bytes = data_bytes;
    if (scales_bytes) write_aligned(w, scales, scales_bytes, &e->scales_offset);
    else e->scales_offset = 0;
    e->scales_bytes = scales_bytes;
}

void gguf_cache_writer_close(GgufCacheWriter *w) {
    if (w->n_written != w->n_tensors) {
        fprintf(stderr, "FATAL: gguf_cache writer: expected %u tensors, got %u\n", w->n_tensors, w->n_written);
        exit(1);
    }
    if (fseeko(w->f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "FATAL: gguf_cache writer: fseek to 0 failed\n"); exit(1);
    }
    GgufCacheHeader h;
    memcpy(h.magic, GGUF_CACHE_MAGIC, 8);
    h.src_size = w->src_size; h.src_mtime = w->src_mtime;
    h.n_tensors = w->n_tensors; h.reserved = 0;
    if (fwrite(&h, sizeof h, 1, w->f) != 1) { fprintf(stderr, "FATAL: gguf_cache writer: header write failed\n"); exit(1); }
    if (w->n_tensors && fwrite(w->entries, sizeof(GgufCacheEntry), w->n_tensors, w->f) != w->n_tensors) {
        fprintf(stderr, "FATAL: gguf_cache writer: directory write failed\n"); exit(1);
    }
    fclose(w->f);
    free(w->entries);
    free(w->path);
    free(w);
}
