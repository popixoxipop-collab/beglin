# GPU Production Shadow Mode

This layer observes production evidence but never mutates production precision
state. It is intended to be applied on top of the real-device-certified XOX GPU
precision head described in `HANDOFF_GPU_TRACK_2026-09-25_PART2.md`.

## Important stacking rule

The GitHub `gpu-precision-g1-g3` branch is behind the certified XOX worktree.
The certified local head includes:

```text
215a5f9 feat(gpu): add real end-to-end GPU precision autopilot
```

Do **not** merge PR #3 as a substitute for synchronizing that certified local
history. Treat `gpu-shadow-mode` as a staging/cherry-pick unit and apply its
shadow-only commits on top of the certified local head.

## Pipeline

```text
production Supabase evidence (GET only)
  -> gpu_shadow_discovery.py
  -> READY candidate queue
  -> gpu_shadow_materialize.py
  -> one local candidate spec
  -> gpu_shadow_pipeline.py
  -> gpu_shadow_runner.py
  -> gpu_autopilot.py
  -> scratch-only control directory
  -> SHADOW_* result
```

Only one READY target is executed per cycle. Other READY targets are recorded as
deferred.

## Safety properties

- No live promotion file is written.
- No live demotion file is written.
- No production GPU txn/ACK path is inherited by the child.
- Supabase credentials are removed from the child autopilot environment.
- Discovery performs GET-only reads and does not log decisions back to Supabase.
- Shadow output/control roots may not overlap the engine checkout or configured
  production/control roots in either direction.
- `ADMITTED` from the child is reported only as `SHADOW_ADMITTED`.
- No automatic production expansion or promotion exists in this layer.
- Source replay files must already exist in an explicit local read-only mirror;
  the shadow tools do not SSH/scp from bob and do not write to bob.

## Read-only evidence credentials

Set only the two variables required for discovery. Do not shell-source bob
`.env`.

```bash
export QWEN_SUPABASE_URL='<read endpoint>'
export QWEN_SUPABASE_KEY='<read credential>'
```

These are used by the discovery process and scrubbed before the child GPU
autopilot is launched.

## Local replay mirror

The provenance table contains source paths captured on the discovery host.
Mirror only the required manifest/token files into a local XOX read-only
directory using your existing controlled file-transfer process.

Example mapping:

```text
source: /Users/bob/vdsp_p5_pre
local:  /Users/xox/vdsp_shadow_mirror/vdsp_p5_pre
```

The materializer validates the mapping and refuses missing files. It derives the
prompt length from the actual raw int32 token file size and hashes the actual
GPU binary.

## Example config

```json
{
  "model": "deepseek-v2-lite",
  "limit": 100,
  "shadow_root": "/Users/xox/vdsp_shadow_runs",
  "cwd": "/Users/xox/vdsp-engine-gpu-precision",
  "autopilot": "/Users/xox/vdsp-engine-gpu-precision/tools/gpu_autopilot.py",
  "binary": "/Users/xox/vdsp-engine-gpu-precision/build-gpu-precision/qwen_infer_gpu",
  "checkpoint_sha256": "<64-hex checkpoint identity>",
  "moe_base": "/Users/xox/vdsp_local_data/moe_base_deepseek",
  "safetensors": "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json",
  "path_maps": [
    "/Users/bob/vdsp_p5_pre=/Users/xox/vdsp_shadow_mirror/vdsp_p5_pre"
  ],
  "g6_repeats": 64,
  "timeout": 3600,
  "forbidden_roots": [
    "/private/tmp/qng64_ctl"
  ]
}
```

Use a shadow root and mirror outside the engine checkout.

## Run one cycle

From the certified XOX GPU worktree:

```bash
cd /Users/xox/vdsp-engine-gpu-precision
PYTHONPATH=tools python3 tools/gpu_shadow_pipeline.py \
  --config /path/to/gpu-shadow-config.json
```

Expected top-level outcomes:

```text
NO_READY_CANDIDATE
SHADOW_CYCLE_COMPLETE
SHADOW_PIPELINE_ERROR
```

Inside a completed cycle, the child result remains shadow-scoped:

```text
SHADOW_ADMITTED
SHADOW_REJECTED
SHADOW_ROLLBACK_OR_REGRESSION
SHADOW_ERROR
SHADOW_COMPLETED_UNCLASSIFIED
```

None of these statuses authorizes a production precision mutation.

## Artifacts

A cycle persists under `shadow_root`:

```text
discovery.json
inputs/<candidate_id>/
  candidate_spec.json
  g4_manifest.txt
  g6_manifest.txt
executions/runs/<run_id>/
  shadow_input.json
  stdout.log
  stderr.log
  shadow_result.json
last_cycle.json
```

The underlying certified autopilot also writes its own evidence underneath the
scratch control root supplied by the shadow runner.

## Promotion to a real production write path

Not part of this PR.

A future production-writing controller must be a separate explicit change with
its own review and enable switch. Passing shadow mode alone is not permission to
turn on GPU auto-promotion.
