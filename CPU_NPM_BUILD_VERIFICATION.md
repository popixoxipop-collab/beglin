# CPU/npm build verification

Agent A validates the npm source-build path independently of the GPU/MLX
runtime. This work does not publish a package or replace a production binary.

## Reproduced baseline failure

A clean `npm pack` from the unmodified package omitted the tokenizer/write
sources and headers. Installing that tarball into an empty consumer failed at:

```text
qwen_infer.c:41:10: fatal error: 'bpe_tokenizer.h' file not found
```

After adding the three missing translation units, recursive source-closure
checking also found two transitive package omissions:

```text
mlx_moe.h
unicode_cpt_flags.h
```

## Fixed compile/package inputs

The CPU postinstall compile/link set now includes:

```text
bpe_tokenizer.c
gguf_write.c
gguf_write_quants.c
```

The npm package allowlist includes those sources, their direct headers,
`mlx_moe.h`, and `unicode_cpt_flags.h`.

## Verification contracts

`scripts/test-package-source-closure.js` starts from every native compile input,
recursively follows quoted local includes, and requires the complete closure to
exist in `package.json#files`. It contains a negative fixture that removes
`bpe_tokenizer.h` and proves the checker catches the historical regression.

`scripts/test-package-cold-install.js` creates a real tarball, installs it into
a new temporary consumer, rejects a native-build SKIP as success, checks a
non-empty installed `bin/qwen_infer`, loads the installed JS API, and records
tarball/binary SHA-256. A no-clang fixture is explicitly classified as SKIP.

## Commands

```bash
npm test
npm run test:package-cold
```

The integration owner must rerun both commands on the final certified source
SHA before the parallel batch is marked VERIFIED.
