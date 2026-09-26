# CPU/npm Build Verification

This document covers the npm source-build path only. It does not change the
engine's supported platform policy and does not publish a package.

## Problem reproduced

A cold install of the 0.4.0 tarball on macOS arm64 failed before any link step:

```text
qwen_infer.c:41:10: fatal error: 'bpe_tokenizer.h' file not found
```

The package archive and postinstall build list had drifted behind the engine
source. In particular, the tokenizer and GGUF writer translation units added to
the engine were not closed over by the npm build/package manifests.

## Source closure

The postinstall native build now compiles these additional plain translation
units:

```text
bpe_tokenizer.c
gguf_write.c
gguf_write_quants.c
```

The npm archive ships those sources plus their direct headers and the generated
Unicode classification header:

```text
bpe_tokenizer.c
bpe_tokenizer.h
unicode_cpt_flags.h
gguf_write.c
gguf_write.h
gguf_write_quants.c
gguf_write_quants.h
```

No MLX/Metal dependency is added to the CPU/npm path. The existing caller-plain
check for `qwen_infer.o` remains unchanged.

## Verification commands

```bash
npm run test:package-source-closure
npm run test:package-cold-install
npm run test:package
```

`test:package-source-closure` checks both the build manifest and the actual
`npm pack --dry-run --json` file list. It also rejects local build output such
as `.build/` and `bin/qwen_infer` from the archive.

`test:package-cold-install` creates a new temporary consumer, packs the current
checkout, installs that tarball with lifecycle scripts enabled, requires the JS
API, and fails if the native binary is absent even when npm itself returned
success. This distinction prevents a missing compiler or native-build skip from
being mistaken for a successful native install.

## Scope

The package version, license, production binaries, GPU code, and platform
support policy are intentionally unchanged.


## EOE verification — 2026-09-26

Execution base and implementation:

```text
BASE_SHA            a0b8ab2c90af70fa8095556033e605477ec186ae
implementation HEAD b5ef5c2193e77b492abe151b32718df6795d65f1
host                EOE
platform            darwin/arm64
Node                v26.7.0
clang               Apple clang 21.0.0
```

The base tarball failure was reproduced independently:

```text
archive_has_bpe_tokenizer_h=false
npm install exit=1
qwen_infer.c:41:10: fatal error: 'bpe_tokenizer.h' file not found
```

After the source-closure fix, two independent clean consumer installs both
compiled and linked the native binary successfully. The tarball SHA-256 was
identical in both runs:

```text
64fe22a544ff0159b904a449913331e22898400022df3d357c4a681e6d777176
```

The two native binaries had the same size (661656 bytes) but different SHA-256
values. Therefore this verification proves clean build/link success but does
**not** claim byte-for-byte reproducible native linking.

Additional gates passed:

```text
npm run test:package-source-closure  PASS
npm run test:package-failure-fixtures PASS (4 fixtures)
npm run test:package                PASS
npm run build                       PASS; caller-plain check OK
npm test                            PASS
git diff --check                    PASS
```

No npm publish or production binary replacement was performed.
