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
//
// Security note (post-push review, 2026-08-25): this file is reached with a cache_path derived
// from QWEN_GGUF (an env var the user controls, but the RESULTING .beglin file is a NEW path
// this process creates/reads on the user's own machine -- unlike src_gguf_path, which the user
// explicitly points at and where following a symlink is the expected, wanted behavior, nothing
// here should have ever silently followed a pre-existing symlink planted at cache_path (e.g. a
// stale/shared tmp-style directory) into an unintended file. Every open of cache_path below now
// uses O_NOFOLLOW. Separately, this cache file's own header/directory content (n_tensors and
// each entry's byte offsets/lengths) is now validated against the actual mmap'd file size
// before any of it is used as pointer arithmetic in gguf_cache_open() -- a truncated or
// corrupted cache (crash mid-write, disk full, manual tampering) FATALs loudly instead of an
// out-of-bounds read past the mmap.

#include "gguf_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// Phase 4 sub-part 4: bumped C1->C2 for GgufCacheEntry's new `E` field -- a stale C1 cache
// written before this change fails the magic check in gguf_cache_is_valid() (returns 0, "no
// valid cache", same as "doesn't exist") and gets rebuilt, rather than being misread with a
// garbage/uninitialized E value.
#define GGUF_CACHE_MAGIC "BEGLINC2"

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
    if (stat(src_gguf_path, &src_st) != 0) return 0;  // src_gguf_path: following a symlink here is intended
    int fd = open(cache_path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return 0;  // missing, or a symlink (O_NOFOLLOW/ELOOP) -- either way, "no valid cache"
    GgufCacheHeader h;
    ssize_t nread = read(fd, &h, sizeof h);
    close(fd);
    if (nread != (ssize_t)sizeof h) return 0;
    if (memcmp(h.magic, GGUF_CACHE_MAGIC, 8) != 0) return 0;
    return h.src_size == (uint64_t)src_st.st_size && h.src_mtime == (uint64_t)src_st.st_mtime;
}

GgufCacheFile *gguf_cache_open(const char *cache_path) {
    int fd = open(cache_path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) { fprintf(stderr, "FATAL: gguf_cache_open: could not open %s (%s)\n", cache_path, strerror(errno)); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "FATAL: gguf_cache_open: fstat %s failed\n", cache_path); exit(1); }
    if ((uint64_t)st.st_size < sizeof(GgufCacheHeader)) {
        fprintf(stderr, "FATAL: gguf_cache_open: %s too small to hold a header (%lld bytes)\n", cache_path, (long long)st.st_size);
        exit(1);
    }
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
    // Bounds-validate the WHOLE file before any of it is trusted as pointer arithmetic --
    // n_tensors and every entry's offset/length come from the file itself, not from a source
    // this process already trusts the way it trusts its own live-transcode output.
    uint64_t n = c->hdr->n_tensors;
    uint64_t dir_bytes = n * (uint64_t)sizeof(GgufCacheEntry);   // n is uint32_t-derived; product fits u64 with wide margin
    uint64_t dir_end = sizeof(GgufCacheHeader) + dir_bytes;
    if (dir_end > (uint64_t)c->size) {
        fprintf(stderr, "FATAL: gguf_cache_open: %s directory (n_tensors=%llu) extends past end of file -- truncated or corrupt\n",
                cache_path, (unsigned long long)n);
        exit(1);
    }
    c->dir = (const GgufCacheEntry *)(c->base + sizeof(GgufCacheHeader));
    for (uint64_t i = 0; i < n; i++) {
        const GgufCacheEntry *e = &c->dir[i];
        // Split as "offset <= size" then "bytes <= size - offset" so this can't overflow even
        // for a maliciously/corruptly huge offset or bytes value.
        int data_ok   = e->data_offset   <= (uint64_t)c->size && e->data_bytes   <= (uint64_t)c->size - e->data_offset;
        int scales_ok = e->scales_offset <= (uint64_t)c->size && e->scales_bytes <= (uint64_t)c->size - e->scales_offset;
        if (!data_ok || !scales_ok) {
            fprintf(stderr, "FATAL: gguf_cache_open: %s entry %llu ('%.*s') has an out-of-bounds offset/length -- truncated or corrupt\n",
                    cache_path, (unsigned long long)i, (int)sizeof e->name, e->name);
            exit(1);
        }
    }
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
    // O_NOFOLLOW: refuse to write through a pre-existing symlink at cache_path (see file header
    // comment) -- if one exists, that's a stale/unexpected state, not something to silently
    // truncate-and-overwrite the symlink's target through.
    int fd = open(cache_path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0644);
    if (fd < 0) {
        fprintf(stderr, "FATAL: gguf_cache_writer_open: could not create %s (%s)\n", cache_path, strerror(errno));
        exit(1);
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) { fprintf(stderr, "FATAL: gguf_cache_writer_open: fdopen %s failed\n", cache_path); close(fd); exit(1); }
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

void gguf_cache_writer_add(GgufCacheWriter *w, const char *name, int kind, int out, int in, int ng, int E,
                            const void *data, uint64_t data_bytes,
                            const void *scales, uint64_t scales_bytes) {
    if (w->n_written >= w->n_tensors) { fprintf(stderr, "FATAL: gguf_cache writer: too many tensors added\n"); exit(1); }
    GgufCacheEntry *e = &w->entries[w->n_written++];
    snprintf(e->name, sizeof e->name, "%s", name);
    e->kind = kind; e->out = out; e->in = in; e->ng = ng; e->E = E;
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
