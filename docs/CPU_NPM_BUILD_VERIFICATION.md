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
