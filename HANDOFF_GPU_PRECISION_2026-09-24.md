# HANDOFF — CPU/GPU Local Quantization Closed Loop

작성 시각: 2026-09-24 KST  
저장소: `popixoxipop-collab/beglin`  
목적: 새 ChatGPT 채팅에서 `@Tailnet Commander EOE Probe`로 XOX/BOB에 재접속한 뒤 GPU precision-control 구현을 즉시 이어가기 위한 상태 전달 문서.

## 1. 한 줄 요약

CPU 쪽 국소 정밀도 autopilot은 실제 production promotion까지 동작 중이고, GPU/MLX 쪽은 **실기기 없이 구현 가능한 G0~G5 control-plane 골격을 모두 미리 구현해 CI까지 통과**했다. 남은 핵심은 Apple Silicon 실기기에서 새 C++ MLX snapshot/restore 및 drain/sync/epoch 동작을 실제 컴파일·검증하는 것이다.

---

## 2. Production 현재 상태 — BOB

현재 production binary:

```text
/Users/bob/vdsp_m4_bench/qwen_infer_prod
SHA256:
3cac67f0ee9f01319027b266dee4ac2cebf23cc99bc797b0327d3aea0df3469a
```

현재 live promotion file:

```text
/private/tmp/qng64_ctl/promotion_nq_live.txt

kv_a_proj_with_mqa 13 7
shared_down_proj 4 6
shared_up_proj 3 5
```

현재 production logical policy hash:

```text
2675b825af7168c5da7af8119b6593b6e1feb96065abae27098a13c4b4c03d6d
```

마지막 확인 당시 demotion/quarantine 파일:

```text
/private/tmp/qng64_ctl/demotion_nq_live.txt
MISSING
```

중요: 이 handoff 작성 과정에서 production runtime/config는 변경하지 않았다.

---

## 3. CPU autopilot에서 이미 확정된 결과

### 실제 유지 중인 신규 promotion

`shared_down_proj/L4 n=6`

검증 흐름:

```text
fresh attribution
→ qNg64 real ladder
→ n=5 FAIL / n=6 PASS / n=7 PASS
→ actual serving-path n=6 PASS
→ live promotion
→ P4 POST
→ attribution 1 → 0
→ healthy
→ demotion 0
```

즉 L4 n=6은 현재 production에 유지 중이다.

### 차단된 후보

`shared_gate_proj/L14`

```text
new production preimage에서도:
n=5 FAIL
n=6 FAIL
n=7 FAIL
→ LIVE_LADDER_UNSAFE
```

`shared_down_proj/L26`

대표 historical real event:

```text
p17 / pos9
372 → 1

qNg64:
n=5 FAIL
n=6 PASS
n=7 FAIL

→ suffix-closed safe n 없음
→ safe_n=None
→ 배포 금지
```

---

## 4. Attribution provenance 구조

이미 main에 들어간 핵심 기능:

- `moe_attribution_provenance`
- `tools/attribution_provenance.py`
- `tools/autopilot_real_sweep.py`
- planner action:
  - `NEEDS_REAL_SWEEP`
  - `NEEDS_ATTRIBUTION_PROVENANCE`
  - `SKIP_UNSAFE`

목표:

```text
event_count만 기억
→ corpus 재스캔
```

이 아니라

```text
DB provenance
→ manifest / req / pos / orig / corrected 즉시 복구
→ isolated replay
→ qNg64 n=5/6/7
→ live preflight
```

로 가는 구조다.

main의 관련 commit:

```text
a0b8ab2 feat(autopilot): persist attribution provenance
```

---

## 5. GPU/MLX 계획서

계획서:

```text
PLAN_cpu_gpu_local_quantization_closed_loop_2026-09-24.md
```

원래 XOX 경로:

```text
/Users/xox/vdsp-engine/PLAN_cpu_gpu_local_quantization_closed_loop_2026-09-24.md
```

핵심 단계:

```text
G0 capability + safety audit
G1 backend/evidence context 분리
G2 GPU binding snapshot/restore
G3 drain/apply/rollback state machine
G4 GPU A/B/R preflight
G5 observer v3
G6 제한적 canary
```

---

## 6. GitHub 브랜치 / PR 상태

### PR #1 — G0

브랜치:

```text
gpu-precision-g0
```

head:

```text
373a4998b5b5bcda81a99eb61524f3eb9f2f9563
```

PR:

```text
#1 G0: harden GPU precision control prerequisites
```

CI:

```text
GitHub Actions run 35905018316
SUCCESS
```

G0에서 구현한 것:

- GPU stale qNg64 binding 제거
- `mlx_gpu_binding_kind()`
- telemetry API key/event key shadowing 수정
- complete-line JSONL checkpoint
- durable outbox
- push 실패 시 offset 전진 금지
- CPU live-preflight를 명시적으로 CPU-only fail-closed
- worker binary SHA/size/host/arch 기록
- controller git HEAD를 worker engine commit처럼 기록하지 않음
- req/pos exact REAL FLIP matching
- `tools/backend_capabilities.py`

### PR #2 — G1~G5 offline control-plane

브랜치:

```text
gpu-precision-g1-g3
```

base:

```text
gpu-precision-g0
```

head:

```text
74b2864451f2de9d7d117096a81d50b8268a9e32
```

PR:

```text
#2 G1-G5: prebuild backend-scoped precision control plane
```

CI:

```text
GitHub Actions run 35908644777
SUCCESS
```

---

## 7. PR #2에서 미리 구현해 둔 것

### G1 — Execution Context / Evidence v3

추가 파일:

```text
tools/precision_context.py
tools/precision_evidence_v3.py
tools/precision_planner_v3.py
supabase_migration_precision_context_v3.sql
```

context에 포함:

```text
model/checkpoint/tokenizer/base artifact
backend
device fingerprint
binary SHA256
build manifest
kernel revision
execution mode
runtime config
quant format
group size
correction mode
```

규칙:

- CPU PASS → GPU admission 불가
- 다른 binary PASS → 현재 binary admission 불가
- 다른 device/context PASS → 불가
- old policy preimage PASS → stale
- old epoch PASS → stale
- full candidate policy hash가 다르면 불가

### G2 — GPU binding snapshot/restore primitive

`mlx_moe.cpp/.h` 추가 API:

```c
mlx_gpu_binding_kind()
mlx_gpu_snapshot_binding()
mlx_gpu_restore_binding_snapshot()
mlx_gpu_drop_binding_snapshot()
mlx_gpu_binding_snapshot_count()
```

중요:

restore API 자체가 GPU quiescence를 만드는 것은 아니다.

호출 전 반드시:

```text
pause admission
→ drain requests
→ synchronize MLX/Metal
→ restore
```

가 완료됐다는 전제다.

snapshot은 실제 active representation 하나만 존재할 때만 성공한다.

### G3 — Durable transaction state machine

파일:

```text
tools/backend_adapters.py
tools/precision_transactions.py
tools/precision_control_state.py
```

상태:

```text
REQUESTED
→ ADMISSION_PAUSED
→ DRAINED
→ GPU_SYNCED
→ SNAPSHOT_DURABLE
→ VERIFYING
→ APPLIED
```

실패:

```text
ROLLBACK_REQUESTED
→ ROLLBACK_APPLIED
```

rollback도 실패:

```text
FAILED_ISOLATED
```

지원:

- context CAS
- policy hash CAS
- epoch CAS
- duplicate txn idempotency
- stale command reject
- durable desired/applied/quarantine
- restart 후 drift reconciliation

### G4 — A/B/R preflight skeleton

파일:

```text
tools/autopilot_preflight_v3.py
```

구조:

```text
A = 현재 승인 GPU policy / correction OFF
B = candidate GPU policy / correction OFF
R = immutable reference
```

검사:

- backend/context exact match
- correction OFF
- finite logits
- baseline original token 재현
- candidate policy 실제 적용 hash
- candidate token == reference token
- reference token == recorded corrected token
- isolated candidate worker가 끝난 뒤 baseline policy로 복원

현재 `MlxMetalBackendAdapter`는 의도적으로 fail-closed다.

즉 실기기 인증 전에는 실제 GPU promotion mutation이 켜지지 않는다.

### G5 — Observer v3

파일:

```text
tools/autopilot_observer_v3.py
```

기록 분모:

```text
requests_completed
tokens_evaluated
near_tie_events
reference_checks_attempted
effective_attribution_checks
attribution_hits
run_errors
backend
context_hash
policy_hash
weight_epoch
```

판정:

```text
CANARY_PASS
INCONCLUSIVE
INCONCLUSIVE_NO_OBSERVABILITY
INCONCLUSIVE_CONTEXT_MISMATCH
REGRESSION_DETECTED
```

중요:

```text
attribution_hits = 0
```

만으로 healthy 처리하지 않는다.

`effective_attribution_checks=0`이면 반드시:

```text
INCONCLUSIVE_NO_OBSERVABILITY
```

이다.

---

## 8. Telemetry idempotency 준비

추가 migration:

```text
supabase_migration_telemetry_ingest_idempotency.sql
```

목표:

```text
DB request 성공
→ local outbox done 쓰기 전에 process crash
→ 재실행
```

되어도 중복 event / duplicate event_count increment가 발생하지 않도록 함.

방식:

- source JSONL path + byte offset + line bytes
- deterministic `ingest_id`
- event는 ingest_id UNIQUE upsert
- attribution aggregate는 `increment_role_precision_idempotent()`
- ingest claim에 성공한 최초 1회만 event_count 증가

중요:

이 migration은 아직 live DB에 적용하지 않았다.

---

## 9. 아직 live 적용하지 않은 것

다음은 코드만 준비됐고 실제 운영에는 적용하지 않았다.

```text
supabase_migration_precision_context_v3.sql
supabase_migration_telemetry_ingest_idempotency.sql
```

그리고:

- PR #1 / #2 모두 아직 main merge하지 않음
- production qwen binary 교체 안 함
- GPU auto-promotion 안 켬
- GPU control dir 새 구조 live 사용 안 함

---

## 10. 새 채팅에서 Tailnet Commander 재개 방법

이 대화에서는 호출 시:

```text
FORBIDDEN: This conversation does not support developer MCPs
```

가 나왔다.

새 채팅에서:

```text
@Tailnet Commander EOE Probe
list_hosts 실행해
```

먼저 호출.

정상이면 예상 host:

```text
eoe
xox
bob
alienware
```

이후 XOX/BOB 상태 확인.

권장 첫 호출:

```text
list_hosts
host_status(xox)
host_status(bob)
```

그 다음 XOX repo 확인:

```text
cd /Users/xox/vdsp-engine
git status --short --branch
git fetch origin
git log --oneline --decorate -8
```

주의:

기존 unrelated local 변경을 건드리지 말 것:

```text
.claude/memory/*
README.md
graphify-out/
NCLT 관련 파일들
기타 동시 작업 파일
```

---

## 11. 실기기 연결 후 가장 먼저 할 일

### Step 1 — branch 동기화

XOX:

```text
git fetch origin
git checkout gpu-precision-g1-g3
```

단, local unrelated changes가 있으면 절대 강제 checkout/reset 금지.

필요하면 별도 worktree 권장.

### Step 2 — Apple/MLX compile gate

검증 대상:

```text
mlx_moe.cpp
mlx_moe.h
```

새 API가 실제 installed MLX revision에서 compile되는지 확인.

특히:

```text
std::optional<QTensor>
std::optional<DTensor>
std::optional<QNg64Tensor>
```

가 현재 MLX array copy/lifetime 규약에서 안전한지 확인.

### Step 3 — capability artifact

실제 GPU binary에서:

```text
tools/backend_capabilities.py
```

실행.

확정할 것:

- mlx_gpu_available symbol 존재
- 실제 binary SHA
- architecture
- GPU backend compiled
- n=5/n=6 native MLX
- n=7 custom Metal

### Step 4 — binding transition tests

최소:

```text
4 → 7 → 4
6 → 7 → 6
7 → 16 → 7
7 → 4
7 → 6
```

각 단계에서:

```text
mlx_gpu_binding_kind(name)
```

가 requested representation과 정확히 일치해야 함.

stale map 없어야 함.

### Step 5 — snapshot/restore runtime

실제 GPU bind 후:

```text
snapshot
→ candidate rebind
→ query kind/bits
→ restore snapshot
→ query kind/bits
```

100회 반복은 이후 soak.

처음에는 단일 tensor synthetic fixture로 검증.

### Step 6 — 실제 drain/synchronize API 연결

현재 transaction state machine의 mock:

```text
pause_admission()
drain()
synchronize()
snapshot()
apply_policy()
restore()
resume_admission()
```

를 실제 qwen/MLX online scheduler에 연결.

절대 registry restore부터 먼저 production에서 호출하지 말 것.

### Step 7 — KV / lazy graph epoch 검증

확인할 것:

- 이전 weight epoch로 만든 KV cache를 새 epoch request가 재사용하지 않음
- pending MLX lazy graph가 old binding을 잡고 있지 않음
- restore 이후 stale compiled graph가 old candidate를 계속 쓰지 않음

### Step 8 — 실제 GPU A/B/R preflight

후보 우선순위는 현재 CPU에서 성공한 target을 GPU에서 **새로** 시험:

```text
shared_down_proj/L4 n=6
shared_up_proj/L3 n=5
kv_a_proj_with_mqa/L13 n=7
```

CPU PASS를 GPU PASS로 복사하면 안 됨.

### Step 9 — failure drill

강제로 실패시켜:

```text
candidate mismatch
OOM
sync failure
restore failure
wrong applied policy
stale epoch
```

각각:

```text
rollback 또는 FAILED_ISOLATED
```

로 가는지 확인.

---

## 12. merge 전략

권장:

1. PR #1 G0를 Apple compile/runtime G0 gate 통과 후 merge
2. PR #2는 그 다음 rebase/base 자동 이동 확인
3. v3 migrations는 코드 merge와 별개로 staging 검증 후 live 적용
4. GPU auto-promotion은 마지막에 opt-in

현재는 둘 다 merge하지 않는 것이 안전하다.

---

## 13. 현재 가장 중요한 원칙

- CPU PASS != GPU PASS
- 소스에 GPU 코드가 있음 != production binary가 GPU capable
- requested policy != applied policy
- attribution 0 != healthy
- higher n != automatically safe
- GPU restore primitive != safe hot-demotion
- drain + sync + epoch/KV boundary가 있어야 runtime rollback
- DB evidence는 backend/context/binary/policy/epoch가 모두 맞아야 재사용

---

## 14. 다음 세션의 완료 목표

실기기 세션에서는 최소 여기까지 끝내면 된다.

```text
[ ] PR #1 branch Apple compile PASS
[ ] actual GPU binary capability artifact
[ ] 4→7→4 runtime PASS
[ ] 6→7→6 runtime PASS
[ ] 7→16→7 runtime PASS
[ ] snapshot/restore synthetic PASS
[ ] real scheduler drain/sync hook 위치 확인
[ ] KV/graph epoch strategy 확정
```

그 다음에 G3 actual adapter 연결 → G4 real GPU preflight → G5 real observer → G6 canary 순으로 진행.

---

## 15. 참고 PR / CI

```text
PR #1
G0: harden GPU precision control prerequisites

branch:
gpu-precision-g0

head:
373a4998b5b5bcda81a99eb61524f3eb9f2f9563

CI:
35905018316 SUCCESS
```

```text
PR #2
G1-G5: prebuild backend-scoped precision control plane

branch:
gpu-precision-g1-g3

base:
gpu-precision-g0

head:
74b2864451f2de9d7d117096a81d50b8268a9e32

CI:
35908644777 SUCCESS
```

---

## 16. Tailnet Commander가 계속 안 될 경우

새 채팅에서도:

```text
FORBIDDEN: This conversation does not support developer MCPs
```

가 나오면 해당 채팅 역시 Developer MCP 실행 비지원 세션이다.

이 경우:

- GitHub connector로 branch/PR/code 작업은 계속 가능
- 실제 XOX/BOB compile/runtime 검증은 Tailnet Commander가 실행 가능한 새 채팅에서 해야 함
- production 변경은 그 전까지 하지 말 것

