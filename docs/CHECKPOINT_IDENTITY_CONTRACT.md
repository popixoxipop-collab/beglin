# Checkpoint Identity Verification Contract

Status: Agent B implementation slice
Scope: read-only checkpoint identity verification for XOX shadow/certification

## Identity v1

The final checkpoint SHA is intentionally compatible with the existing
gpu_shadow_materialize.py checkpoint-identity-v1 algorithm.

For a safetensors index, the identity manifest is:

- schema = checkpoint-identity-v1
- kind = safetensors-index
- index: filename, byte size, SHA-256 of the exact index bytes
- shards: sorted distinct weight_map shard names, byte size, SHA-256

The final checkpoint_sha256 is the SHA-256 of canonical JSON for that manifest
(sort_keys=true, compact separators, ASCII JSON).

For a single safetensors file the manifest contains that file's filename, size,
and SHA-256.

The verification envelope and file stat metadata are not part of the v1
identity digest.

## Verification v1

tools/checkpoint_identity.py performs at least two complete passes.

Each pass:

1. resolves the checkpoint under an approved read-only root;
2. reads and parses the exact index bytes;
3. rejects absolute or parent-traversal shard names;
4. resolves every distinct referenced shard;
5. rejects shard targets outside the approved root;
6. hashes each file by streaming chunks;
7. compares dev/inode/size/mtime/ctime before and after each hash;
8. re-reads the index after all shards;
9. emits the v1 identity manifest.

The verifier requires the identity manifest, SHA, and file observations to agree
between complete passes. A changed input is a verification failure rather than
a new silently accepted identity.

This does not claim to defend against an adversary capable of changing a file
and restoring all observable metadata between reads. Certification should use
immutable or change-frozen model inputs.

## Symlink policy

The existing shadow materializer historically followed shard symlinks.

The independent verifier is stricter: a symlink may resolve only inside the
explicit approved checkpoint root. A cache layout whose shards resolve outside
that root must provide a broader, explicitly approved read-only root rather than
silently escaping the root.

## XOX detached verifier

tools/checkpoint_identity_xox.py fixes the certified input to:

- checkpoint:
  /Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json
- approved root:
  /Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors
- output root:
  /Users/xox/vdsp_shadow_runs/checkpoint_identity

It does not read Supabase credentials, production policy, candidate history, or
runtime process environments.

Intended Tailnet exact argv only:

    ["python3","tools/checkpoint_identity_xox.py"]
    ["python3","tools/checkpoint_identity_xox.py","--status"]

The worker mode is started locally by the launcher and should not be exposed as
a Tailnet command.

## Evidence levels

Synthetic/unit PASS proves the algorithm contract only.

Agent B may report actual checkpoint identity VERIFIED only after the XOX
detached verifier reaches COMPLETE and the persisted result confirms two full
passes over the actual index and all referenced shards.

Do not infer actual verification from checkpoint_sha256="auto" appearing in a
config or from a shadow cycle that stopped at dedupe before materialization.
