// bpe_tokenizer.c -- see bpe_tokenizer.h for scope/contract. Own translation unit.
//
// Ports llama.cpp's real algorithm (source fetched and read directly this session, not
// guessed): the GPT2-style byte<->printable-unicode bijection, the qwen2 hand-coded
// pre-tokenizer split (unicode_regex_split_custom_qwen2, src/unicode.cpp), and the classic
// rank-ordered greedy BPE merge (llm_tokenizer_bpe_session::tokenize, src/llama-vocab.cpp) --
// symbols as a doubly-linked list, candidate merges picked by lowest (rank, left-position),
// lazy invalidation via a size check instead of removing stale entries from the queue.

#include "bpe_tokenizer.h"
#include "unicode_cpt_flags.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal: a pre-tokenized "word" span, as (start, len) into a codepoint array. Not part of
// the public API -- bpe_encode() is the only entry point external code needs.
typedef struct { size_t start, len; } BpeSpan;

// Generated from llama.cpp's unicode_byte_to_utf8_map() logic (source fetched this
// session from src/unicode.cpp) -- GPT2-style byte<->printable-unicode bijection.
// byte b -> UTF-8 encoding of its mapped codepoint (1-2 bytes, since max mapped
// codepoint here is 256+67=323, which needs at most 2 UTF-8 bytes).
typedef struct { uint8_t b[3]; uint8_t n; } ByteToUtf8;
static const ByteToUtf8 BYTE_TO_UNICODE_UTF8[256] = {
  {{0xc4,0x80,0x00}, 2},
  {{0xc4,0x81,0x00}, 2},
  {{0xc4,0x82,0x00}, 2},
  {{0xc4,0x83,0x00}, 2},
  {{0xc4,0x84,0x00}, 2},
  {{0xc4,0x85,0x00}, 2},
  {{0xc4,0x86,0x00}, 2},
  {{0xc4,0x87,0x00}, 2},
  {{0xc4,0x88,0x00}, 2},
  {{0xc4,0x89,0x00}, 2},
  {{0xc4,0x8a,0x00}, 2},
  {{0xc4,0x8b,0x00}, 2},
  {{0xc4,0x8c,0x00}, 2},
  {{0xc4,0x8d,0x00}, 2},
  {{0xc4,0x8e,0x00}, 2},
  {{0xc4,0x8f,0x00}, 2},
  {{0xc4,0x90,0x00}, 2},
  {{0xc4,0x91,0x00}, 2},
  {{0xc4,0x92,0x00}, 2},
  {{0xc4,0x93,0x00}, 2},
  {{0xc4,0x94,0x00}, 2},
  {{0xc4,0x95,0x00}, 2},
  {{0xc4,0x96,0x00}, 2},
  {{0xc4,0x97,0x00}, 2},
  {{0xc4,0x98,0x00}, 2},
  {{0xc4,0x99,0x00}, 2},
  {{0xc4,0x9a,0x00}, 2},
  {{0xc4,0x9b,0x00}, 2},
  {{0xc4,0x9c,0x00}, 2},
  {{0xc4,0x9d,0x00}, 2},
  {{0xc4,0x9e,0x00}, 2},
  {{0xc4,0x9f,0x00}, 2},
  {{0xc4,0xa0,0x00}, 2},
  {{0x21,0x00,0x00}, 1},
  {{0x22,0x00,0x00}, 1},
  {{0x23,0x00,0x00}, 1},
  {{0x24,0x00,0x00}, 1},
  {{0x25,0x00,0x00}, 1},
  {{0x26,0x00,0x00}, 1},
  {{0x27,0x00,0x00}, 1},
  {{0x28,0x00,0x00}, 1},
  {{0x29,0x00,0x00}, 1},
  {{0x2a,0x00,0x00}, 1},
  {{0x2b,0x00,0x00}, 1},
  {{0x2c,0x00,0x00}, 1},
  {{0x2d,0x00,0x00}, 1},
  {{0x2e,0x00,0x00}, 1},
  {{0x2f,0x00,0x00}, 1},
  {{0x30,0x00,0x00}, 1},
  {{0x31,0x00,0x00}, 1},
  {{0x32,0x00,0x00}, 1},
  {{0x33,0x00,0x00}, 1},
  {{0x34,0x00,0x00}, 1},
  {{0x35,0x00,0x00}, 1},
  {{0x36,0x00,0x00}, 1},
  {{0x37,0x00,0x00}, 1},
  {{0x38,0x00,0x00}, 1},
  {{0x39,0x00,0x00}, 1},
  {{0x3a,0x00,0x00}, 1},
  {{0x3b,0x00,0x00}, 1},
  {{0x3c,0x00,0x00}, 1},
  {{0x3d,0x00,0x00}, 1},
  {{0x3e,0x00,0x00}, 1},
  {{0x3f,0x00,0x00}, 1},
  {{0x40,0x00,0x00}, 1},
  {{0x41,0x00,0x00}, 1},
  {{0x42,0x00,0x00}, 1},
  {{0x43,0x00,0x00}, 1},
  {{0x44,0x00,0x00}, 1},
  {{0x45,0x00,0x00}, 1},
  {{0x46,0x00,0x00}, 1},
  {{0x47,0x00,0x00}, 1},
  {{0x48,0x00,0x00}, 1},
  {{0x49,0x00,0x00}, 1},
  {{0x4a,0x00,0x00}, 1},
  {{0x4b,0x00,0x00}, 1},
  {{0x4c,0x00,0x00}, 1},
  {{0x4d,0x00,0x00}, 1},
  {{0x4e,0x00,0x00}, 1},
  {{0x4f,0x00,0x00}, 1},
  {{0x50,0x00,0x00}, 1},
  {{0x51,0x00,0x00}, 1},
  {{0x52,0x00,0x00}, 1},
  {{0x53,0x00,0x00}, 1},
  {{0x54,0x00,0x00}, 1},
  {{0x55,0x00,0x00}, 1},
  {{0x56,0x00,0x00}, 1},
  {{0x57,0x00,0x00}, 1},
  {{0x58,0x00,0x00}, 1},
  {{0x59,0x00,0x00}, 1},
  {{0x5a,0x00,0x00}, 1},
  {{0x5b,0x00,0x00}, 1},
  {{0x5c,0x00,0x00}, 1},
  {{0x5d,0x00,0x00}, 1},
  {{0x5e,0x00,0x00}, 1},
  {{0x5f,0x00,0x00}, 1},
  {{0x60,0x00,0x00}, 1},
  {{0x61,0x00,0x00}, 1},
  {{0x62,0x00,0x00}, 1},
  {{0x63,0x00,0x00}, 1},
  {{0x64,0x00,0x00}, 1},
  {{0x65,0x00,0x00}, 1},
  {{0x66,0x00,0x00}, 1},
  {{0x67,0x00,0x00}, 1},
  {{0x68,0x00,0x00}, 1},
  {{0x69,0x00,0x00}, 1},
  {{0x6a,0x00,0x00}, 1},
  {{0x6b,0x00,0x00}, 1},
  {{0x6c,0x00,0x00}, 1},
  {{0x6d,0x00,0x00}, 1},
  {{0x6e,0x00,0x00}, 1},
  {{0x6f,0x00,0x00}, 1},
  {{0x70,0x00,0x00}, 1},
  {{0x71,0x00,0x00}, 1},
  {{0x72,0x00,0x00}, 1},
  {{0x73,0x00,0x00}, 1},
  {{0x74,0x00,0x00}, 1},
  {{0x75,0x00,0x00}, 1},
  {{0x76,0x00,0x00}, 1},
  {{0x77,0x00,0x00}, 1},
  {{0x78,0x00,0x00}, 1},
  {{0x79,0x00,0x00}, 1},
  {{0x7a,0x00,0x00}, 1},
  {{0x7b,0x00,0x00}, 1},
  {{0x7c,0x00,0x00}, 1},
  {{0x7d,0x00,0x00}, 1},
  {{0x7e,0x00,0x00}, 1},
  {{0xc4,0xa1,0x00}, 2},
  {{0xc4,0xa2,0x00}, 2},
  {{0xc4,0xa3,0x00}, 2},
  {{0xc4,0xa4,0x00}, 2},
  {{0xc4,0xa5,0x00}, 2},
  {{0xc4,0xa6,0x00}, 2},
  {{0xc4,0xa7,0x00}, 2},
  {{0xc4,0xa8,0x00}, 2},
  {{0xc4,0xa9,0x00}, 2},
  {{0xc4,0xaa,0x00}, 2},
  {{0xc4,0xab,0x00}, 2},
  {{0xc4,0xac,0x00}, 2},
  {{0xc4,0xad,0x00}, 2},
  {{0xc4,0xae,0x00}, 2},
  {{0xc4,0xaf,0x00}, 2},
  {{0xc4,0xb0,0x00}, 2},
  {{0xc4,0xb1,0x00}, 2},
  {{0xc4,0xb2,0x00}, 2},
  {{0xc4,0xb3,0x00}, 2},
  {{0xc4,0xb4,0x00}, 2},
  {{0xc4,0xb5,0x00}, 2},
  {{0xc4,0xb6,0x00}, 2},
  {{0xc4,0xb7,0x00}, 2},
  {{0xc4,0xb8,0x00}, 2},
  {{0xc4,0xb9,0x00}, 2},
  {{0xc4,0xba,0x00}, 2},
  {{0xc4,0xbb,0x00}, 2},
  {{0xc4,0xbc,0x00}, 2},
  {{0xc4,0xbd,0x00}, 2},
  {{0xc4,0xbe,0x00}, 2},
  {{0xc4,0xbf,0x00}, 2},
  {{0xc5,0x80,0x00}, 2},
  {{0xc5,0x81,0x00}, 2},
  {{0xc5,0x82,0x00}, 2},
  {{0xc2,0xa1,0x00}, 2},
  {{0xc2,0xa2,0x00}, 2},
  {{0xc2,0xa3,0x00}, 2},
  {{0xc2,0xa4,0x00}, 2},
  {{0xc2,0xa5,0x00}, 2},
  {{0xc2,0xa6,0x00}, 2},
  {{0xc2,0xa7,0x00}, 2},
  {{0xc2,0xa8,0x00}, 2},
  {{0xc2,0xa9,0x00}, 2},
  {{0xc2,0xaa,0x00}, 2},
  {{0xc2,0xab,0x00}, 2},
  {{0xc2,0xac,0x00}, 2},
  {{0xc5,0x83,0x00}, 2},
  {{0xc2,0xae,0x00}, 2},
  {{0xc2,0xaf,0x00}, 2},
  {{0xc2,0xb0,0x00}, 2},
  {{0xc2,0xb1,0x00}, 2},
  {{0xc2,0xb2,0x00}, 2},
  {{0xc2,0xb3,0x00}, 2},
  {{0xc2,0xb4,0x00}, 2},
  {{0xc2,0xb5,0x00}, 2},
  {{0xc2,0xb6,0x00}, 2},
  {{0xc2,0xb7,0x00}, 2},
  {{0xc2,0xb8,0x00}, 2},
  {{0xc2,0xb9,0x00}, 2},
  {{0xc2,0xba,0x00}, 2},
  {{0xc2,0xbb,0x00}, 2},
  {{0xc2,0xbc,0x00}, 2},
  {{0xc2,0xbd,0x00}, 2},
  {{0xc2,0xbe,0x00}, 2},
  {{0xc2,0xbf,0x00}, 2},
  {{0xc3,0x80,0x00}, 2},
  {{0xc3,0x81,0x00}, 2},
  {{0xc3,0x82,0x00}, 2},
  {{0xc3,0x83,0x00}, 2},
  {{0xc3,0x84,0x00}, 2},
  {{0xc3,0x85,0x00}, 2},
  {{0xc3,0x86,0x00}, 2},
  {{0xc3,0x87,0x00}, 2},
  {{0xc3,0x88,0x00}, 2},
  {{0xc3,0x89,0x00}, 2},
  {{0xc3,0x8a,0x00}, 2},
  {{0xc3,0x8b,0x00}, 2},
  {{0xc3,0x8c,0x00}, 2},
  {{0xc3,0x8d,0x00}, 2},
  {{0xc3,0x8e,0x00}, 2},
  {{0xc3,0x8f,0x00}, 2},
  {{0xc3,0x90,0x00}, 2},
  {{0xc3,0x91,0x00}, 2},
  {{0xc3,0x92,0x00}, 2},
  {{0xc3,0x93,0x00}, 2},
  {{0xc3,0x94,0x00}, 2},
  {{0xc3,0x95,0x00}, 2},
  {{0xc3,0x96,0x00}, 2},
  {{0xc3,0x97,0x00}, 2},
  {{0xc3,0x98,0x00}, 2},
  {{0xc3,0x99,0x00}, 2},
  {{0xc3,0x9a,0x00}, 2},
  {{0xc3,0x9b,0x00}, 2},
  {{0xc3,0x9c,0x00}, 2},
  {{0xc3,0x9d,0x00}, 2},
  {{0xc3,0x9e,0x00}, 2},
  {{0xc3,0x9f,0x00}, 2},
  {{0xc3,0xa0,0x00}, 2},
  {{0xc3,0xa1,0x00}, 2},
  {{0xc3,0xa2,0x00}, 2},
  {{0xc3,0xa3,0x00}, 2},
  {{0xc3,0xa4,0x00}, 2},
  {{0xc3,0xa5,0x00}, 2},
  {{0xc3,0xa6,0x00}, 2},
  {{0xc3,0xa7,0x00}, 2},
  {{0xc3,0xa8,0x00}, 2},
  {{0xc3,0xa9,0x00}, 2},
  {{0xc3,0xaa,0x00}, 2},
  {{0xc3,0xab,0x00}, 2},
  {{0xc3,0xac,0x00}, 2},
  {{0xc3,0xad,0x00}, 2},
  {{0xc3,0xae,0x00}, 2},
  {{0xc3,0xaf,0x00}, 2},
  {{0xc3,0xb0,0x00}, 2},
  {{0xc3,0xb1,0x00}, 2},
  {{0xc3,0xb2,0x00}, 2},
  {{0xc3,0xb3,0x00}, 2},
  {{0xc3,0xb4,0x00}, 2},
  {{0xc3,0xb5,0x00}, 2},
  {{0xc3,0xb6,0x00}, 2},
  {{0xc3,0xb7,0x00}, 2},
  {{0xc3,0xb8,0x00}, 2},
  {{0xc3,0xb9,0x00}, 2},
  {{0xc3,0xba,0x00}, 2},
  {{0xc3,0xbb,0x00}, 2},
  {{0xc3,0xbc,0x00}, 2},
  {{0xc3,0xbd,0x00}, 2},
  {{0xc3,0xbe,0x00}, 2},
  {{0xc3,0xbf,0x00}, 2},
};

// Reverse: mapped-codepoint -> original byte, for decode(). Key is the codepoint value
// (max 323), stored as a sparse array sized 512 with 0xFF sentinel for "unused".
#define UNICODE_TO_BYTE_TABLE_SIZE 512
static const uint16_t UNICODE_TO_BYTE[UNICODE_TO_BYTE_TABLE_SIZE] = {
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,0x0029,0x002a,0x002b,0x002c,0x002d,0x002e,0x002f,
  0x0030,0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,0x0039,0x003a,0x003b,0x003c,0x003d,0x003e,0x003f,
  0x0040,0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,0x0049,0x004a,0x004b,0x004c,0x004d,0x004e,0x004f,
  0x0050,0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,0x0059,0x005a,0x005b,0x005c,0x005d,0x005e,0x005f,
  0x0060,0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,0x0069,0x006a,0x006b,0x006c,0x006d,0x006e,0x006f,
  0x0070,0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,0x0079,0x007a,0x007b,0x007c,0x007d,0x007e,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0x00a1,0x00a2,0x00a3,0x00a4,0x00a5,0x00a6,0x00a7,0x00a8,0x00a9,0x00aa,0x00ab,0x00ac,0xffff,0x00ae,0x00af,
  0x00b0,0x00b1,0x00b2,0x00b3,0x00b4,0x00b5,0x00b6,0x00b7,0x00b8,0x00b9,0x00ba,0x00bb,0x00bc,0x00bd,0x00be,0x00bf,
  0x00c0,0x00c1,0x00c2,0x00c3,0x00c4,0x00c5,0x00c6,0x00c7,0x00c8,0x00c9,0x00ca,0x00cb,0x00cc,0x00cd,0x00ce,0x00cf,
  0x00d0,0x00d1,0x00d2,0x00d3,0x00d4,0x00d5,0x00d6,0x00d7,0x00d8,0x00d9,0x00da,0x00db,0x00dc,0x00dd,0x00de,0x00df,
  0x00e0,0x00e1,0x00e2,0x00e3,0x00e4,0x00e5,0x00e6,0x00e7,0x00e8,0x00e9,0x00ea,0x00eb,0x00ec,0x00ed,0x00ee,0x00ef,
  0x00f0,0x00f1,0x00f2,0x00f3,0x00f4,0x00f5,0x00f6,0x00f7,0x00f8,0x00f9,0x00fa,0x00fb,0x00fc,0x00fd,0x00fe,0x00ff,
  0x0000,0x0001,0x0002,0x0003,0x0004,0x0005,0x0006,0x0007,0x0008,0x0009,0x000a,0x000b,0x000c,0x000d,0x000e,0x000f,
  0x0010,0x0011,0x0012,0x0013,0x0014,0x0015,0x0016,0x0017,0x0018,0x0019,0x001a,0x001b,0x001c,0x001d,0x001e,0x001f,
  0x0020,0x007f,0x0080,0x0081,0x0082,0x0083,0x0084,0x0085,0x0086,0x0087,0x0088,0x0089,0x008a,0x008b,0x008c,0x008d,
  0x008e,0x008f,0x0090,0x0091,0x0092,0x0093,0x0094,0x0095,0x0096,0x0097,0x0098,0x0099,0x009a,0x009b,0x009c,0x009d,
  0x009e,0x009f,0x00a0,0x00ad,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
  0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,0xffff,
};

// ---- String-keyed hash table (open addressing, FNV-1a, linear probing) ----
// Keys are {ptr,len} views into GgufFile-owned memory (Phase 1's zero-copy string arrays) --
// no key bytes are ever copied.

static uint64_t bpe_fnv1a(const char *p, uint32_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = 0; i < n; i++) { h ^= (unsigned char)p[i]; h *= 1099511628211ULL; }
    return h;
}

static void str_int_map_init(BpeStrIntMap *m, uint32_t expected_n) {
    uint32_t cap = 16;
    while (cap < expected_n * 2u + 1u) cap <<= 1;
    m->entries = calloc(cap, sizeof(BpeHTEntry));
    m->cap = cap;
    m->count = 0;
}

static void str_int_map_put(BpeStrIntMap *m, const char *ptr, uint32_t len, uint32_t value) {
    uint64_t h = bpe_fnv1a(ptr, len);
    uint32_t idx = (uint32_t)(h & (m->cap - 1));
    while (m->entries[idx].used) {
        if (m->entries[idx].len == len && memcmp(m->entries[idx].ptr, ptr, len) == 0) return; // first-wins on dup key
        idx = (idx + 1) & (m->cap - 1);
    }
    m->entries[idx].ptr = ptr;
    m->entries[idx].len = len;
    m->entries[idx].value = value;
    m->entries[idx].used = 1;
    m->count++;
}

static int str_int_map_get(const BpeStrIntMap *m, const char *ptr, uint32_t len, uint32_t *out_value) {
    if (m->cap == 0) return 0;
    uint64_t h = bpe_fnv1a(ptr, len);
    uint32_t idx = (uint32_t)(h & (m->cap - 1));
    while (m->entries[idx].used) {
        if (m->entries[idx].len == len && memcmp(m->entries[idx].ptr, ptr, len) == 0) {
            *out_value = m->entries[idx].value;
            return 1;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return 0;
}

// ---- UTF-8 codec (standard algorithm; unicode_len_utf8's real lookup table confirmed via
// direct source read of llama.cpp's src/unicode.cpp this session) ----

static uint32_t utf8_decode_one(const char *s, size_t n, size_t *pos) {
    static const int LEN_LUT[16] = {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4};
    unsigned char b0 = (unsigned char)s[*pos];
    int len = LEN_LUT[b0 >> 4];
    if (*pos + (size_t)len > n) { (*pos)++; return 0xFFFD; } // truncated -- replacement char, matches llama.cpp's own catch-and-replace
    uint32_t cpt;
    if (len == 1) {
        cpt = b0;
    } else if (len == 2) {
        cpt = ((uint32_t)(b0 & 0x1F) << 6) | ((unsigned char)s[*pos+1] & 0x3F);
    } else if (len == 3) {
        cpt = ((uint32_t)(b0 & 0x0F) << 12) | (((unsigned char)s[*pos+1] & 0x3F) << 6) | ((unsigned char)s[*pos+2] & 0x3F);
    } else {
        cpt = ((uint32_t)(b0 & 0x07) << 18) | (((unsigned char)s[*pos+1] & 0x3F) << 12) |
              (((unsigned char)s[*pos+2] & 0x3F) << 6) | ((unsigned char)s[*pos+3] & 0x3F);
    }
    *pos += (size_t)len;
    return cpt;
}

// ASCII-only lowercase, NOT llama.cpp's full Unicode unicode_tolower() table. Justified: the
// only call site (qwen2_split_span's 's/'t/'re/'ve/'m/'ll/'d contraction check) only ever
// compares the result against ASCII literals ('s','t','m','d','r','v','l','e) -- for any
// non-ASCII input, llama.cpp's real unicode_tolower() would also fail to equal those ASCII
// literals (its table only maps codepoints to OTHER codepoints, never TO an ASCII letter from
// a non-ASCII one), so an ASCII-only tolower is numerically identical for this specific
// comparison, not an approximation of it.
static uint32_t ascii_tolower(uint32_t cpt) {
    if (cpt >= 'A' && cpt <= 'Z') return cpt + 32;
    return cpt;
}

// ---- qwen2 pre-tokenizer split (ported verbatim from unicode_regex_split_custom_qwen2,
// src/unicode.cpp, fetched this session -- also used for qwen3moe/qwen3_moe, D-tok-0's real
// GGUF header read confirmed they share the exact same tokenizer.ggml.pre="qwen2" value) ----

static int qwen2_split_span(const uint32_t *cpts, size_t offset_ini, size_t offset_end,
                             BpeSpan *out_spans, int cap) {
    int n_out = 0;
    size_t prev_end = offset_ini;

    #define GET_CPT(p) (((p) >= offset_ini && (p) < offset_end) ? cpts[p] : 0xFFFFFFFFu)
    #define GET_FLAGS(p) (((p) >= offset_ini && (p) < offset_end) ? unicode_cpt_flags(cpts[p]) : 0)
    #define ADD_TOKEN(end_pos) do { \
        size_t _end = (size_t)(end_pos); \
        size_t _len = _end - prev_end; \
        if (_len > 0) { \
            if (n_out >= cap) return -1; \
            out_spans[n_out].start = prev_end; \
            out_spans[n_out].len = _len; \
            n_out++; \
        } \
        prev_end = _end; \
        pos = _end; \
    } while (0)

    for (size_t pos = offset_ini; pos < offset_end; ) {
        uint32_t cpt = GET_CPT(pos);
        uint16_t flags = GET_FLAGS(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < offset_end) {
            uint32_t c1 = ascii_tolower(GET_CPT(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                ADD_TOKEN(pos + 2);
                continue;
            }
            if (pos + 2 < offset_end) {
                uint32_t c2 = ascii_tolower(GET_CPT(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    ADD_TOKEN(pos + 3);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || (flags & UCPT_NUMBER))) {
            if ((flags & UCPT_LETTER) || (GET_FLAGS(pos + 1) & UCPT_LETTER)) {
                pos++;
                while (GET_FLAGS(pos) & UCPT_LETTER) pos++;
                ADD_TOKEN(pos);
                continue;
            }
        }

        // regex: \p{N}
        if (flags & UCPT_NUMBER) {
            pos++;
            ADD_TOKEN(pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
        {
            uint16_t flags2 = (cpt == ' ') ? GET_FLAGS(pos + 1) : flags;
            if (!(flags2 & (UCPT_WHITESPACE | UCPT_LETTER | UCPT_NUMBER)) && flags != 0) {
                if (cpt == ' ') pos++;
                while (!(flags2 & (UCPT_WHITESPACE | UCPT_LETTER | UCPT_NUMBER)) && flags2 != 0) {
                    pos++;
                    flags2 = GET_FLAGS(pos);
                }
                uint32_t cpt2 = GET_CPT(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    pos++;
                    cpt2 = GET_CPT(pos);
                }
                ADD_TOKEN(pos);
                continue;
            }
        }

        {
            size_t num_ws = 0;
            size_t last_end_rn = 0;
            while (GET_FLAGS(pos + num_ws) & UCPT_WHITESPACE) {
                uint32_t cpt2 = GET_CPT(pos + num_ws);
                if (cpt2 == '\r' || cpt2 == '\n') last_end_rn = pos + num_ws + 1;
                num_ws++;
            }

            // regex: \s*[\r\n]+
            if (last_end_rn > 0) {
                ADD_TOKEN(last_end_rn);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_ws > 1 && GET_CPT(pos + num_ws) != 0xFFFFFFFFu) {
                ADD_TOKEN(pos + num_ws - 1);
                continue;
            }

            // regex: \s+
            if (num_ws > 0) {
                ADD_TOKEN(pos + num_ws);
                continue;
            }
        }

        // no matches
        ADD_TOKEN(pos + 1);
    }

    #undef GET_CPT
    #undef GET_FLAGS
    #undef ADD_TOKEN
    return n_out;
}

// ---- BPE merge core (ported from llm_tokenizer_bpe_session::tokenize, src/llama-vocab.cpp) ----
//
// O(n^2) best-candidate scan per merge step rather than a true priority queue -- deliberate
// simplification, not a guess: real pre-tokenized "words" from qwen2_split_span are short (a
// single word/number/punctuation-run/whitespace-run), so n stays small in practice and a heap's
// extra bookkeeping buys nothing measurable here. Correctness-first per this project's own
// established sequencing (same precedent as D-metal-6 scoping GPU throughput out until
// correctness was settled) -- if real-prompt profiling later shows this matters, swap for a
// binary min-heap without changing the merge semantics.

#define BPE_MAX_SYMBOLS 8192

typedef struct { int prev, next; const char *text; size_t n; } BpeSymbol;
typedef struct { int left, right, rank; size_t size; int alive; } BpeCandidate;

static int bpe_find_rank(const BpeVocab *v, const char *lt, size_t ln, const char *rt, size_t rn) {
    char buf[512];
    if (ln + 1 + rn >= sizeof(buf)) return -1;
    memcpy(buf, lt, ln);
    buf[ln] = ' ';
    memcpy(buf + ln + 1, rt, rn);
    uint32_t rank;
    if (str_int_map_get(&v->merge_map, buf, (uint32_t)(ln + 1 + rn), &rank)) return (int)rank;
    return -1;
}

static int bpe_merge_word(const BpeVocab *v, const char *word, size_t word_len,
                           int32_t *out_ids, int cap) {
    static _Thread_local BpeSymbol syms[BPE_MAX_SYMBOLS];
    static _Thread_local BpeCandidate cand[BPE_MAX_SYMBOLS * 2];
    int n_syms = 0;

    {
        size_t i = 0;
        while (i < word_len) {
            if (n_syms >= BPE_MAX_SYMBOLS) return -1;
            unsigned char b0 = (unsigned char)word[i];
            size_t clen = 1;
            if ((b0 & 0xE0) == 0xC0) clen = 2;
            else if ((b0 & 0xF0) == 0xE0) clen = 3;
            else if ((b0 & 0xF8) == 0xF0) clen = 4;
            if (i + clen > word_len) clen = word_len - i;
            syms[n_syms].text = word + i;
            syms[n_syms].n = clen;
            syms[n_syms].prev = n_syms - 1;
            syms[n_syms].next = (i + clen < word_len) ? n_syms + 1 : -1;
            n_syms++;
            i += clen;
        }
    }
    if (n_syms == 0) return 0;

    int n_cand = 0;
    #define TRY_ADD_BIGRAM(l, r) do { \
        int _l = (l), _r = (r); \
        if (_l >= 0 && _r >= 0) { \
            int rk = bpe_find_rank(v, syms[_l].text, syms[_l].n, syms[_r].text, syms[_r].n); \
            if (rk >= 0) { \
                if (n_cand >= BPE_MAX_SYMBOLS * 2) return -1; \
                cand[n_cand].left = _l; cand[n_cand].right = _r; cand[n_cand].rank = rk; \
                cand[n_cand].size = syms[_l].n + syms[_r].n; cand[n_cand].alive = 1; \
                n_cand++; \
            } \
        } \
    } while (0)

    for (int i = 0; i + 1 < n_syms; i++) TRY_ADD_BIGRAM(i, i + 1);

    for (;;) {
        int best = -1;
        for (int i = 0; i < n_cand; i++) {
            if (!cand[i].alive) continue;
            if (best < 0 || cand[i].rank < cand[best].rank ||
                (cand[i].rank == cand[best].rank && cand[i].left < cand[best].left)) {
                best = i;
            }
        }
        if (best < 0) break;
        cand[best].alive = 0;

        int l = cand[best].left, r = cand[best].right;
        if (syms[l].n == 0 || syms[r].n == 0) continue;
        if (syms[l].n + syms[r].n != cand[best].size) continue;

        syms[l].n += syms[r].n;
        syms[r].n = 0;
        syms[l].next = syms[r].next;
        if (syms[r].next >= 0) syms[syms[r].next].prev = l;

        TRY_ADD_BIGRAM(syms[l].prev, l);
        TRY_ADD_BIGRAM(l, syms[l].next);
    }
    #undef TRY_ADD_BIGRAM

    int n_out = 0;
    for (int i = 0; i < n_syms; i++) {
        if (syms[i].n == 0) continue;
        uint32_t id;
        if (!str_int_map_get(&v->vocab_map, syms[i].text, (uint32_t)syms[i].n, &id)) {
            fprintf(stderr, "FATAL: bpe_tokenizer: merged piece '%.*s' not found in vocab -- "
                "merges/vocab tables are inconsistent\n", (int)syms[i].n, syms[i].text);
            exit(1);
        }
        if (n_out >= cap) return -1;
        out_ids[n_out++] = (int32_t)id;
    }
    return n_out;
}

// ---- Public API ----

int bpe_vocab_load(const GgufFile *f, BpePretokType pretok, BpeVocab *v) {
    memset(v, 0, sizeof(*v));
    if (!gguf_kv_str_array(f, "tokenizer.ggml.tokens", &v->tokens, &v->n_tokens)) return 0;
    if (!gguf_kv_str_array(f, "tokenizer.ggml.merges", &v->merges, &v->n_merges)) {
        v->merges = NULL;
        v->n_merges = 0;
    }
    v->pretok = pretok;

    str_int_map_init(&v->vocab_map, (uint32_t)v->n_tokens);
    for (uint64_t i = 0; i < v->n_tokens; i++) {
        str_int_map_put(&v->vocab_map, v->tokens[i].ptr, (uint32_t)v->tokens[i].len, (uint32_t)i);
    }

    str_int_map_init(&v->merge_map, (uint32_t)v->n_merges);
    for (uint64_t i = 0; i < v->n_merges; i++) {
        str_int_map_put(&v->merge_map, v->merges[i].ptr, (uint32_t)v->merges[i].len, (uint32_t)i);
    }

    int64_t bos, eos;
    v->bos_id = gguf_kv_i64(f, "tokenizer.ggml.bos_token_id", &bos) ? (int32_t)bos : -1;
    v->eos_id = gguf_kv_i64(f, "tokenizer.ggml.eos_token_id", &eos) ? (int32_t)eos : -1;
    return 1;
}

void bpe_vocab_free(BpeVocab *v) {
    free(v->vocab_map.entries);
    free(v->merge_map.entries);
    memset(v, 0, sizeof(*v));
}

int bpe_encode(const BpeVocab *v, const char *text, size_t text_len, int32_t *out_ids, int cap) {
    #define BPE_MAX_CPTS 65536
    static _Thread_local uint32_t cpts_buf[BPE_MAX_CPTS];
    static _Thread_local size_t byte_off_buf[BPE_MAX_CPTS + 1];
    size_t n_cpts = 0;
    size_t pos = 0;
    while (pos < text_len) {
        if (n_cpts >= BPE_MAX_CPTS) return -1;
        byte_off_buf[n_cpts] = pos;
        cpts_buf[n_cpts] = utf8_decode_one(text, text_len, &pos);
        n_cpts++;
    }
    byte_off_buf[n_cpts] = text_len;

    #define BPE_MAX_SPANS 16384
    static _Thread_local BpeSpan spans[BPE_MAX_SPANS];
    int n_spans;
    if (v->pretok == BPE_PRETOK_QWEN2) {
        n_spans = qwen2_split_span(cpts_buf, 0, n_cpts, spans, BPE_MAX_SPANS);
    } else {
        return -1;
    }
    if (n_spans < 0) return -1;

    int n_out = 0;
    static _Thread_local char mapped_buf[BPE_MAX_SYMBOLS];
    for (int s = 0; s < n_spans; s++) {
        size_t byte_start = byte_off_buf[spans[s].start];
        size_t byte_end = byte_off_buf[spans[s].start + spans[s].len];
        size_t mlen = 0;
        for (size_t bi = byte_start; bi < byte_end; bi++) {
            unsigned char b = (unsigned char)text[bi];
            const ByteToUtf8 *bt = &BYTE_TO_UNICODE_UTF8[b];
            if (mlen + bt->n > sizeof(mapped_buf)) return -1;
            memcpy(mapped_buf + mlen, bt->b, bt->n);
            mlen += bt->n;
        }
        int written = bpe_merge_word(v, mapped_buf, mlen, out_ids + n_out, cap - n_out);
        if (written < 0) return -1;
        n_out += written;
    }
    #undef BPE_MAX_CPTS
    #undef BPE_MAX_SPANS
    return n_out;
}

int bpe_decode(const BpeVocab *v, const int32_t *ids, int n_ids, char *out_buf, int cap) {
    int out_len = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || (uint64_t)ids[i] >= v->n_tokens) {
            fprintf(stderr, "FATAL: bpe_tokenizer: decode id %d out of vocab range [0,%llu)\n",
                    ids[i], (unsigned long long)v->n_tokens);
            exit(1);
        }
        const GgufStr *tok = &v->tokens[ids[i]];
        size_t p = 0;
        while (p < tok->len) {
            size_t char_start = p;
            uint32_t cpt = utf8_decode_one(tok->ptr, tok->len, &p);
            size_t char_len = p - char_start;
            if (cpt < UNICODE_TO_BYTE_TABLE_SIZE && UNICODE_TO_BYTE[cpt] != 0xffff) {
                if ((size_t)out_len + 1 > (size_t)cap) return -1;
                out_buf[out_len++] = (char)UNICODE_TO_BYTE[cpt];
            } else {
                if ((size_t)out_len + char_len > (size_t)cap) return -1;
                memcpy(out_buf + out_len, tok->ptr + char_start, char_len);
                out_len += (int)char_len;
            }
        }
    }
    return out_len;
}
