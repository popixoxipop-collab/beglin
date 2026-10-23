# BEGLIN Qv20 — 새로운 학습/평가 분리, 원본 C calibration 1차 완료 및 D50 접근 장애 핸드오프

**일자:** 2026-10-24 (KST)  
**Repo/branch:** `popixoxipop-collab/beglin` / `research/qv20-prompt-aware-equal-byte-20261024`  
**현실적 판정:** `SOURCE_ONLY_SPLIT_FROZEN_PASS / ORIGINAL_QWEN28_SOURCE_TRAIN32_PASS / SECOND_RANDOM73_CALIBRATION_FAILED_FILE_ACCESS / DYNAMIC_POLICY_NOT_FROZEN / HELDOUT_NOT_EVALUATED / NO_QUALITY_GAIN_CLAIM`.

## 연구 구분 및 기존 연구와의 차이

2025 NeurIPS DP-LLM(https://arxiv.org/abs/2508.06041)은 입력 기반 런타임 레이어 정밀도 선택을, 2026 MixQuant(https://arxiv.org/abs/2607.23047)은 정밀도 간 upstream 상호작용을 연구한다. BEGLIN Qv20의 1차 범위는 **Qv19 실제 Apple CPU/Qwen28 프로세스 안전 잠금 경로에서 두 역할·packed weight byte budget이 같은 28n8 마스크를 요청 토큰만으로 선택하는 작은 실험**이다. 이는 arbitrary per-layer/per-token bit assignment가 아니다.

- 후보 0: Qv8 `source_qv8`, 원래 projection packed bytes `765493248` (28 n8/168 Q4).
- 후보 1: role count와 byte budget이 같은 seed73 `random73`, projection packed bytes `765493248`.
- 두 후보의 전체 프로세스 RAM은 동등하지 않다. 실제 동적 전환에 쓰이는 대체 inactive packed/scales cache는 추가 약 90MiB/96MiB 제한을 가진다. 정적 마스크와 같은 projection packed bytes만으로 RAM Pareto 이득을 주장할 수 없다.
- 프롬프트 feature만 사용하는 미리 고정된 ridge/winsor training `tools/qv20_fit_prompt_policy.py`: `lambda=6`, NLL delta clip ±2, selector margin ±0.10, 32문항 leave-one-out CV. 정적 최고 평균 NLL보다 CV gain ≥0.03이고 각 마스크 선택 ≥4개일 때만 동적 정책 승인. 실패 시 **학습용 최고 고정 마스크로 fallback**. 이 모든 조건은 평가 결과 확인 전 GitHub에 작성/확인했다.
- GPU Metal, conditional four-choice NLL, 새로운 모델 패밀리 정확도 이득은 아직 평가되지 않았다.

## 입력 데이터 사전봉인

EOE owner-only `/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024`에서 원본 Arrow/Tokenizer로 3개 코호트를 만들고 **새 모델 점수 실행 전 SHA 봉인**.

| 구분 | 샘플 | 독립성 | SHA-256 |
|---|---:|---|---|
| ARC-Challenge TRAIN calibration | 32 = short16+long16 | 모델 선택용, 검증 성적에 포함하지 않음 | `6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28` |
| ARC-Challenge VALIDATION | 24 = short12+long12 | Qv14/16 기존 76 input ID 전부 제외 | `b0a27e22f0c5769f49c3dc95d2200feef9f06747349abef55bd82c29d13e2119` |
| OpenBookQA VALIDATION | 24 = short12+long12 | Qv13 OpenBookQA TEST 128과 split 분리 | `2f80539f7eca065e8dd5effcc1c320e2fbcf74f7e6af0e1e2816a8fe72aeb204` |

공개 사전등록 해시 `2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a`; GitHub `docs/research/BEGLIN_QV20_PREEXEC_ARC_TRAIN_ARC_VALIDATION_OB_VALIDATION_20261024.json`. 80/80 frozen input ID/prefix/gold token bytes all three preparation modes 검사, cross cohort question hash 중복 0. 원본 문항/토큰/정답은 D50에 보존하고 GitHub에는 업로드하지 않았다.

## 원본 Qwen28 CPU 보정 중 완료된 실증

- 소스 마스크 원본 C 작업 `job_192e789b14812afb4fb95b2c090b4581` → **SUCCEEDED exit0**.
- 정확히 32개의 ARC TRAIN source frozen 문항으로, 원본 Qwen28 Q4/n8 C 바이너리 `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221` 사용.
- 정확도 **19/32**, mean full-vocab gold answer next-token NLL **1.53042149**, wall ~445.87초, peak RSS ~4638.42MiB.
- public baseline receipt SHA `4696d663f66ae7939553b4e66c748e4cab4e10395d4e70a07e49bd8b94e0e746` GitHub에 확인 후 저장.
- 무작위73 보정 작업 `job_cbdfe07861a2fbd0a616c3af9325fa07`: **FAILED exit1 after 97.94s**, 실제 모델 출력 로그 쓰기 시 `PermissionError: [Errno 1] Operation not permitted`. 출력을 완료한 유효한 연구 결과/영수증이 **없으므로** 이를 성공으로 집계하거나 NLL을 추정하지 않는다.
- 그 직후 `/Volumes/D50` 루트 `os.listdir`, 해당 코호트 `os.listdir` 및 `Tailnet EOE read_file(@fs...)` 모두 **EPERM**. 기존 source 성공 로그도 이제 접근 거부. 파일 소유자 eoe(UID501)와 모드0700은 맞다.
- `df -h /Volumes/D50`, `diskutil info /Volumes/D50` 읽기 전용 조회로 볼륨은 **mounted, APFS, read-write, unlocked, 465GiB available**. 용량 부족이나 physical unmount 확인되지 않음. macOS 외장 볼륨/앱 개인정보 보호 권한(TCC) 또는 다른 호스트 접근정책을 점검해야 한다. 정확한 주체는 지금 단정 불가. `sudo`, 재마운트, sudo chmod/ACL 우회 **실행하지 않음**.

## 실행 가능한 코드와 사전 테스트

- `tools/qv20_freeze_equalbyte_cohorts.py` — 원본 Archer/OB 데이터 SHA 및 old validation IDs 제외하고 나머지 분리. PASS.
- `tools/qv20_actual_static_native.py` — 원본 Apple Qwen28 C 단일 마스크 모델 점수 및 부정확/허위 입력 차단. source train32 PASS.
- `tools/qv20_fit_prompt_policy.py` — train32 2-arm real C 결과가 모두 없으면 policy fit **fail-closed**. 두 번째 arm 누락으로 정상 차단 검증 완료. Holdout 원문/점수는 fitter가 읽지 않는다.
- `tools/qv20_build_prompt_selector_native.py` — Qv19 기존 모델 lifetime mutex/NLL 코드와 original 23 objects를 보존한 derived C. **합성 예시 policy만을 사용한 실제 Apple arm64 clang native 컴파일 PASS**, 소스 SHA `b92669103c245673b8bedb71095165e12fa1fe296c1489a337e17d9b3a317720`, exe SHA `21e8742cec7f7492bce88adb80c21eb7b632680fdf304dc47e06413a6130b1e9`. 이 예시 코드는 **synthetic 정책의 컴파일 확인**이지 실제 최적화나 품질 증거가 아니다.
- `tools/qv20_native_dynamic_heldout.py`, `tools/qv20_independent_heldout_audit.py` — 동적 실행 조건 및 48 paired stratified bootstrap, Holm/정답 차이 독립 감사 코드를 **heldout 점수 이전** GitHub에 등록.
- `tools/qv20_original_Q4_heldout_control.py` — 더 작은 packed Q4 비용 `736985088`의 reference scorer. 아직 실행 안 됨.
- `tools/qv20_actual_static_native_retry_empty_failed_dir.py` — random73의 실패 output 디렉터리가 이미 만들어졌으므로 **권한 복원 후 해당 폴더가 비었을 때만 재실행하도록 차단 조건 추가**. 원본 failed job/디렉터리를 삭제하거나 덮어쓰지 않음.

## 기존 점수에서의 탐색적 음성 결과 (Qv20 새 데이터 아님)

Qv14에서 이미 점수 계산이 완료된 이전 ARC validation 24문항으로, 사전 고정된 Qv20 ridge 선택규칙을 **기계적 sanity check**로만 적용한 결과:
- 고정 Qv8 1.64932349 mean NLL.
- 고정 random73 1.61290878 mean NLL.
- Qv20 정책 구조를 LOOCV로 적용하면 2.11796237 mean NLL, best static 대비 **−0.50505358 손실 증가**.
이 값은 과거 이미 본 표본의 음성 결과이므로 새로운 Qv20 calibration, heldout 또는 정확도 향상 증거가 아니다. 새 train32가 CV gate 실패하면 고정 정책 fallback하도록 설계했다.

## 다음 정확한 작업 게이트

1. **EOE의 D50 정상 파일 접근 권한을 관리자/소유자 승인 경로로 복구**. `ls /Volumes/D50`, owner-only `Qv20_PREEXEC_PUBLIC.json` read, `native_original_C_static/cal_arc_train/static_random` empty check 모두 정상이어야 한다. sudo/ACL/다른 계정으로 우회 금지.
2. 리소스 계획으로 EOE CPU≥2core/RAM≥5.5GB 적격성을 다시 확인. `git pull --ff-only`로 Qv20 branch의 `tools/qv20_actual_static_native_retry_empty_failed_dir.py` 가져온 후, source-locked **same 32 ARC TRAIN random73** 실행을 새 job/idempotency로 제출. 실패 폴더에 기존 파일이 있으면 즉시 중단/대사.
3. source/random 모두 원본 C 영수증 확인 뒤 `qv20_fit_prompt_policy.py`를 **한 번만** 실행. LOOCV 정책 승인 또는 정적 fallback을 봉인하고 public policy SHA를 GitHub 업로드. heldout before-policy scores 절대 사용 금지.
4. Qv20 real builder 실행, binary SHA 기록, Qv17 signed pack와 Qv19 원본 lifetime mutex binding 검증.
5. 정책 동결 **이후** ARC VALIDATION24, OB VALIDATION24 × 동일 packed budget static source/random, dynamic Qv20 및 Q4 저비용 control 실기기 호출. `qv20_independent_heldout_audit.py`는 원본 C loss/정답 per-case와 동적 선택 policy consistency부터 확인하고, dataset-stratified bootstrap/paired accuracy/Holm p로 비교. 동적 정책이 static fallback이면 **성능 개선 미입증**이라고 명시. 동일 projection bytes와 **추가 캐시 메모리**는 분리 계산.
6. 별도 Metal GPU 경로 및 실제 소프트웨어 서비스/주문 병렬 안전성은 Qv20의 CPU 연구 판정과 혼동하지 않는다.

**운영 불변:** Qv19 실제 원본 Qwen28 CPU mutex 연구 증거, original Qv17 signed cache, BEGLIN main, Tailnet immutable release `g6-0.9.0-operator.900387` 전부 유지. 프로덕션 자동 승격 없음.

**요약:** Qv20 정책 설계/시드·분할/코드 구조는 진행됐고, 원본 CPU calibration 32문항의 Qv8 1arm은 실증했다. 두 번째 source-matched arm이 D50 볼륨 접근 거부로 실패해 **Qv20 실제 정확도 기반 선택 정책의 성능은 아직 평가 불가**이다.
