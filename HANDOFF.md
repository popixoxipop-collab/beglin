# Agent A HANDOFF — CPU/npm build closure

Status: **VERIFIED**  
Branch: `agents/20260926/a-cpu-npm`  
BASE_SHA: `a0b8ab2c90af70fa8095556033e605477ec186ae`  
Implementation head before evidence commit: `b5ef5c2193e77b492abe151b32718df6795d65f1`

## What was fixed

The npm source-build path was not closed over the current engine sources. The A branch adds these plain translation units to `scripts/postinstall-build.js`:

```text
bpe_tokenizer.c
gguf_write.c
gguf_write_quants.c
```

The package allowlist now includes those sources plus their direct headers and `unicode_cpt_flags.h`.

No MLX/Metal dependency was added. Package version, license, production binary and platform support policy were not changed.

## BASE failure reproduced

A detached clone at the exact BASE_SHA was packed and installed into an empty consumer.

```text
archive_has_bpe_tokenizer_h=false
npm install exit=1
qwen_infer.c:41:10: fatal error: 'bpe_tokenizer.h' file not found
```

This distinguishes the real BASE failure from a static source-list observation.

## Fixed-branch real CPU verification

Environment:

```text
host  EOE
OS    darwin/arm64
Node  v26.7.0
clang Apple clang version 21.0.0 (clang-2100.3.34.2)
```

Two independent cold installs from newly packed tarballs both succeeded.

```text
tarball SHA256
64fe22a544ff0159b904a449913331e22898400022df3d357c4a681e6d777176

cold install #1 binary
84111fd4dd3f25b1f414f425b15111fc495dcf771a63a9959767fdff840af500
size 661656

cold install #2 binary
1af103b6c75a73e47f2ccb6c5a0ba7a155fef5734ea65e99795211a50254b0ab
size 661656
```

The equal-size binaries have different hashes, so A does **not** claim byte-for-byte deterministic native linking.

## Tests

```text
npm run test:package-source-closure    PASS
npm run test:package-failure-fixtures PASS (4/4)
npm run test:package-cold-install      PASS x2
npm run test:package                   PASS
npm run build                          PASS
caller-plain check                     PASS
npm test                               PASS
git diff --check                       PASS
```

Failure fixtures cover missing source/header, missing clang, unsupported platform skip behavior, and a link failure that must propagate as failure.

## Evidence

```text
evidence/agent-a/package-source-closure.json
evidence/agent-a/cold-install.json
evidence/agent-a/base-repro.json
evidence/agent-a/A_CPU_NPM_COLD_INSTALL_20260926.log
evidence/agent-a/A_CPU_NPM_BASE_REPRO_20260926.log
agent_result.json
```

## Integration guidance

A0 should integrate only this A branch's owned CPU/npm files and evidence. No changes are required in the GPU worktree.

Agent C can use:
- BASE_SHA above,
- fixed source commit,
- deterministic tarball SHA,
- actual cold-install binary SHAs,
- real CPU evidence level.

No npm publish and no production binary replacement were performed.
