# BEGLIN HCP P18 — EOE/D50 격리 평가 환경 이관 핸드오프

작성: 2026-10-15 KST. 연구 전용. **운영 정책 A/릴리스 불변**, 정책 B 자동 승격 금지.

## 확인된 실제 성과

- P18 frozen 원문은행 480문항 SHA-256 `c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df`. 선행 별도 XOX GPU/BF16 실측에서 BF16·A·B paired full-vocab logits 질문 **6/480**, gold 8/673, native A/B float32 vectors **16/1346**. 전체 통계 비열등성 **미인증**. task full generation, latency three-trial 480계약은 미완료.
- XOX SSD 잔여 약 **20.03GiB**(기준 보호선 20GiB), 쓰기 누적 중단. 원본 P17/운영 모델/캐시 삭제·복구 변경 없음.
- EOE D50 스토리지 풀이 약 660GiB 이상 남아 있어 연구 모델을 복제하는 새 실행 루트로 적합. EOE 내 사용자 승인 운영 환경과 별도 디렉터리.
- 원본 소스와 수학적 구현을 바꾸지 않고 **입력 문맥 상한만 112→1024** 확장한 BF16 교사 코드를 EOE용 독립 파일로 작성·원본 소스 대비 클래스 코드 차이 검사 PASS:
  - `hcp_p18_eoe_d50_original_bf16_core.py` SHA `295ce59380f83162d20f619bf9d0cccc961f0c1484d16f5031fdc0040bb6bb78`.
  - `hcp_p18_eoe_d50_original_bf16_480_unit.py` SHA `6cfebe610ca4d69160d9d0967da1a0423eea4d1265726c488e21dfff6740274b`.
  - EOE Python3.11 Torch 2.13.0 / Transformers 5.15.0 / Safetensors 0.8.0 / MLX 0.32.1 설치됨; P17 원본 Python 소스 SHA `40dd5550...`, `46510526...` 재확인. 새 BF16 실행기는 원본 체크포인트 index/config/modeling SHA, 5291개의 BF16 source tensor values, RAM/RSS/swap/D50 공간 및 480 bank를 fail-closed 검증.
- EOE 연구용 MLX C/CPP 소스: 원래 XOX P18 long-source `qwen_infer.c` SHA `fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa`에서 **원시 logits 저장 디렉터리 리터럴만 변경**. 새 EOE C source SHA `f5fab593e588d5a5b34528cc99c518fc2f894186ab46cb50bca5dc1b2e59df5e`; MLX 원본 SHA `a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b`.
- 실제 EOE 격리 `cmake` GPU 바이너리 빌드 **exit 0 PASS**, SHA `4b22f80d6368fd8653abcf2b4f3e4d9e1c5fdff8e50ef598d942bbfc1774f176`. 별도 `mlx_binding_transition_test` **컴파일/링크 PASS**. 테스트 실행은 안전 검사에서 차단되어 PASS로 기록하지 않음.
- EOE original BF16 `--index 0 --preflight-only` 시험: **불완전한 모델 소스를 거부(exit 1)**. 모델이 아직 복제 완료되지 않았기 때문이므로 진짜 BF16 forward PASS 주장 금지.

## 진행 중 복제 프로세스

- EOE process `proc_edd7dafc6a64ccb31f4b644910df0399`: 원본 XOX DeepSeek-V2-Lite BF16 safetensors (원본 디렉터리 약29GiB, shard 4개) → `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/teacher_checkpoint_source_copy/`. `rsync -av --partial --stats` SSH 읽기 전용.
- EOE process `proc_6223c668c2043cb5d772868c894b1a51`: XOX quantized DeepSeek MoE base (원본 약9.2GiB) → `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/`.
- 마지막 확인에서 두 process **RUNNING**, 실제 데이터 복사 진행 중. 최종 `rsync` 종료 코드·모델 SHA 교차 검증 **미완료**. 부분/임시 .safetensors 파일을 모델로 사용하지 말 것.
- 절대로 두 process를 중복 시작하지 말 것. `process_output`으로 ID를 재조회하고 터미널 exit 0을 확인해야 함.
- 완료 후 EOE checkpoint index SHA `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`, config SHA `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`, modeling SHA `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`. `TensorStore` 원본 5291 BF16 tensor 가치 manifest SHA `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`는 실행 후 대조.

## 재개 절차

1. 위 process ID 두 개의 종료 상태·stderr·exit code 확인. 실패 시 partial files 보호, 원래 XOX 데이터 무수정.
2. D50 source model index/shards/weights를 원본 XOX 파일과 hash/size로 검증. `hcp_p18_eoe_d50_original_bf16_480_unit.py --index 0 --preflight-only`를 통과시킨다.
3. EOE 모델 실기기 BF16 small pilot 원본 frozen MMLU-000의 전체 original 27L·5291 tensor logits 계산 후 XOX 원본 pilot NLL 1.017819917과 비교한다. PyTorch/장치 버전 차이는 별도 기록하고 값 일치를 맹목적으로 주장하지 않는다.
4. Native A/B는 EOE 독립 MLX runtime/원래 Qwen_MoE_BASE/원본 safetensors source와 673 prefix hash allowlist를 모두 고정한 별도 연구용 driver의 negative tests/실제 102400 logits를 검증해야 하며, **현재 아직 모델 실행 미검증**.
5. 검증된 다음 단위 실행만 허용. 480×BF16/A/B source NLL 후에도 독립 full-generation task scoring / warm latency 3× / frozen 10,000 bootstrap/Bonferroni pass를 충족하기 전에는 B 운영 승격 금지.

## 보안·운영 인계

- XOX 실 운영 stage SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, PIDs `35481,35493` 이전 검증 이후 불변.
- Tailnet 현재/실행 릴리스 `g6-0.9.0-operator.900350`, converged=true, active_operation=null.
- HF-cache/운영 P17/모델 가중치 사용자 보존 파일 수정·삭제 없음. XOX와 EOE 사이 이동은 **복사 전용**.
- Private original test text/token arrays, original model weights, raw GPU .f32/teacher .npy **GitHub 올리지 말 것**.
- 상태: **EOE native-build PASS / EOE original BF16 code parity PASS / both-model full copy NOT YET VERIFIED / EOE native A/B actual inference NOT YET VERIFIED / full P18 quality BLOCKED**.
