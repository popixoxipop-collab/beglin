// bpe_tokenizer.h -- Phase 6 (D-tok) real in-engine byte-level BPE tokenizer.
//
// Ports llama.cpp's own real algorithm (source read directly this session, not guessed):
// GPT2-style byte<->printable-unicode bijection (unicode_byte_to_utf8_map), a hand-coded
// per-architecture pre-tokenizer split function (unicode_regex_split_custom_qwen2 is the first
// ported; see D-tok-4/5 for the rest), and the classic rank-ordered greedy BPE merge
// (llm_tokenizer_bpe_session::tokenize -- symbols as a doubly-linked list, a min-heap of
// candidate merges ordered by (rank, left-position), lazy invalidation via a size check instead
// of queue restructuring).
//
// Depends on gguf_load.h (for GgufFile/GgufStr) and unicode_cpt_flags.h (Phase 2's codepoint
// classification table) -- both already own, separate translation units; this file follows the
// same convention.
//
// Vocab/merge lookups are string-keyed open-addressing hash tables (FNV-1a) built once at load
// time from the GGUF's own tokenizer.ggml.tokens/merges arrays (Phase 1) -- the GgufFile these
// point into MUST stay open for the tokenizer's whole lifetime (no copies are made of the
// vocab/merge string bytes, same zero-copy convention Phase 1's array accessors already use).

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_load.h"

typedef enum {
    BPE_PRETOK_QWEN2 = 0,   // D-tok-4: also covers qwen3moe/qwen3_moe (same real pre-type)
    BPE_PRETOK_LLAMA3 = 1,  // D-tok-5: real port of unicode_regex_split_custom_llama3
    BPE_PRETOK_GPT2 = 2,    // D-tok-5: real port of unicode_regex_split_custom_gpt2 -- also
                            // used for olmoe's "olmo" pre-type (no hand-coded llama.cpp
                            // reference exists for "olmo"; its regex_exprs is textually GPT2's
                            // pattern plus an explicit \s+(?!\S) alternative GPT2's hand-coded
                            // function already implements as its whitespace fallback -- oracle-
                            // verified empirically against llama-tokenize on a real OLMoE
                            // checkpoint, not assumed from the regex-text argument alone).
} BpePretokType;

typedef struct { const char *ptr; uint32_t len; uint32_t value; int used; } BpeHTEntry;
typedef struct { BpeHTEntry *entries; uint32_t cap; uint32_t count; } BpeStrIntMap;

// D-tok-5: a CONTROL/USER_DEFINED/UNKNOWN-type vocab entry (real GGUF token_type != NORMAL,
// e.g. "<|endoftext|>", or OLMoE's literal multi-space code-indent tokens "  "/"   ") is matched
// against the RAW input text as a literal substring BEFORE normal pretokenization+BPE runs --
// confirmed necessary, not assumed, by a real oracle mismatch this session (see RESULTS.md's
// D-tok-5): without this, OLMoE's own real tokenizer output for text containing a double-space
// run diverged from llama-tokenize's real output, because that "  " is its own literal vocab
// entry, not reachable via byte-mapped BPE merging at all.
typedef struct { const char *ptr; uint32_t len; int32_t id; } BpeSpecialTok;

typedef struct {
    const GgufStr *tokens;      // zero-copy, points into the GgufFile's mmap (Phase 1)
    uint64_t n_tokens;
    const GgufStr *merges;      // zero-copy, points into the GgufFile's mmap (Phase 1)
    uint64_t n_merges;
    BpeStrIntMap vocab_map;     // token string -> id (index into tokens[])
    BpeStrIntMap merge_map;     // "left right" string -> rank (index into merges[])
    BpePretokType pretok;
    int32_t bos_id, eos_id;     // -1 if not present in the GGUF
    BpeSpecialTok *special;     // token_type != NORMAL entries, for literal pre-scan matching
    uint32_t n_special;
} BpeVocab;

// Loads tokens/merges/special-ids from an already-open GgufFile (see gguf_load.h). Returns 1 on
// success, 0 if the required KVs are missing (e.g. not a GGUF with an embedded BPE vocab).
// `pretok` selects the hand-coded pre-tokenizer split function to use for encode().
int bpe_vocab_load(const GgufFile *f, BpePretokType pretok, BpeVocab *v);
void bpe_vocab_free(BpeVocab *v);

// encode(): UTF-8 text -> token ids. Returns the number of ids written (<= cap), or -1 if the
// output buffer was too small (matching load_ids()'s own cap-bounded contract, qwen_infer.c).
int bpe_encode(const BpeVocab *v, const char *text, size_t text_len, int32_t *out_ids, int cap);

// decode(): token ids -> UTF-8 text. Returns the number of bytes written (<= cap, NOT NUL-
// terminated -- caller adds a NUL if cap allows), or -1 if the output buffer was too small.
int bpe_decode(const BpeVocab *v, const int32_t *ids, int n_ids, char *out_buf, int cap);

#endif // BPE_TOKENIZER_H
