# Beglin P18 EOE/D50 전수 평가 이관: 실제 빌드 + 엄격한 모델 파일 검증 (2026-10-15 KST)

## 운영·품질 현재 판정

- 연구: P18 frozen external 480문항 / 673 gold-token positions.
- 실제 원본 BF16 + XOX Metal 정책 A/B 원시 전체 vocab logits 검증 질문: **6/480**, Gold 8/673, 102400-f32 A/B 총 16/1346.
- 추가 XOX BF16-only 긴 문맥 931+1 위치 source forward는 확인했지만, A/B 페어 6문항에 넣지 않음.
- P18 완전 생성 정답률/3회 고정 workload warm latency/독립480 stratified bootstrap은 미완료. **정책 B 비열등성 미인증**, incumbent A 운영 유지.
- Tailnet Operator last verified: `g6-0.9.0-operator.900350`, current=live=runtime, converged=true, active_operation=null. XOX serving C SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, PID 35481/35493, 유지.
- XOX SSD reserve 20 GiB 대비 초과 가용 용량 last measured approx **0.05GiB**. XOX에서 추가 BF16/native raw accumulation **정지**. 이전 scratch delete/attestation upload는 platform safety checker에 의해 차단된 상태이므로, 다른 제어면으로 우회하지 않음.

## EOE/D50로 포팅한 실제 소스 및 증명

Main root: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`.

| 작업 | 검증과 SHA |
|---|---|
| 원본 P17 BF16 27L/5291T source (그대로 사용) | `40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3` |
| EOE/D50 원본 BF16 core | `295ce59380f83162d20f619bf9d0cccc961f0c1484d16f5031fdc0040bb6bb78` |
| EOE/D50 실기기 BF16 per-question unit | `070d3ecdc58c02d5c85e380f94d3f42d1389896651a1688be2226d7561db9190` |
| P18 원래 XOX native C 673 allowed SHA source | `fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa` |
| EOE native C source — D50 capture path literal만 변경 | `f5fab593e588d5a5b34528cc99c518fc2f894186ab46cb50bca5dc1b2e59df5e` |
| MLX source 변경 없음 | `a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b` |
| EOE AppleClang MLX/Metal isolated CMake binary (compile+link exit0) | `4b22f80d6368fd8653abcf2b4f3e4d9e1c5fdff8e50ef598d942bbfc1774f176` |
| EOE P17 native P9 A/B adapter per-question | `9174aa5ccfd28f7f0500c17b452c66e9686e9fe761f3ac28d5608475a9d4e9ed` |
| 원본 모델 2종의 XOX/D50 각 파일 SHA 재계산·출처 봉인기 | `06a8dfcdf3f5b48935179a52f22d9fd405f9ab0b85fe93acfa7615bce7c95ffd` |
| 개발용 203토큰 모델 A/B의 XOX↔EOE 한 토큰 GPU parity probe | `c2ea938031887065c057ede90bcee590e9de09fa2c54bfc7ecd5bc876c0884d2` |
| 480문항 read-only next-source/resume status | `c80cb18c5f2b2ed0b3fa25b4dde0ac5bda9bdd30cfd740191e3c157fbe58f33f` |

**실측 구분:** EOE CMake configure/compile/link = PASS. Original source parity and P9 policy A/B hashes = static PASS. EOE GPU forward 및 원본 BF16 full model forward = **아직 미실행**, 모델 복사 미완료와 SHA-256 검증 영수증 부재로 모든 driver가 fail-closed. 개발 전용 203 소스는 테스트 입력 480문항과 별개이며, 실제 비교 시 정책 A/B 출력 SHA가 기존 XOX와 일치하는지 검사함.

## Model transfer (실제 실행 중, 미완료)

- Original BF16 DeepSeek-V2-Lite checkpoint: 원본 `/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors` → D50 `teacher_checkpoint_source_copy/`.
  - 원본 BF16 4 shard 정확한 합계 31,413,626,576 bytes + metadata, 출처 index SHA `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
  - EOE managed transfer process: `proc_edd7dafc6a64ccb31f4b644910df0399`, last observed RUNNING.
- Original quantized MoE based model: 원본 `/Users/xox/vdsp_local_data/moe_base_deepseek` → D50 `native_moe_base_source_copy/`.
  - original 5 files exact sum 9,828,513,716 bytes, XOX model files untouched.
  - EOE managed transfer process: `proc_6223c668c2043cb5d772868c894b1a51`, last observed RUNNING.
- Latest observed D50 copies: BF16 ~6.0GiB / quantized MoE ~4.6GiB. **진행량 수치는 파일 복제 완료 증명이 아님.**
- 소스 보존: rsync -a --partial, destination D50, XOX read only. D50 protected storage pool allocatable initially 약666GiB; transfer work EOE resource plan eligible.
- 원본 BF16 tensor values 5291 hash manifest `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`, 매 실제 BF16 forward에서 재확인.

## 입장 게이트 / 실제 검사 결과

- `/opt/homebrew/bin/python3.11 hcp_p18_eoe_d50_model_copy_seal.py --inspect` → 정확한 파일 크기·누락 확인; 마지막 검사 결과 `P18_D50_MODEL_TRANSFER_INCOMPLETE`.
- **두 rsync process 모두 EXITED 0을 확인한 뒤에만** `--seal` 실행 허용. 이 호출은 XOX/D50 sha256 각 파일 동등성을 검증해 immutable BF16/MoE provenance receipts 두 개를 생성. 소스 크기·SHA 미일치 시 fail-closed.
- `hcp_p18_eoe_d50_native_AB_unit.py --index 3` → source provenance receipt가 없어서 GPU 시작 전 fail-closed 검증 완료.
- `hcp_p18_eoe_d50_original_bf16_unit.py --index 3` → BF16 shard/index incomplete 시 실행 전 fail-closed 검증 완료.
- `hcp_p18_eoe_dev203_real_metal_AB_parity.py` → BF16 source receipt 미존재로 실제 GPU 시작 전 fail-closed 검증 완료.
- EOE source AST/SHA 4/4 PASS, EOE original P17 decoder equation code unchanged except 112→1024 window and source path PASS, native C hash allowlist 673/673 verified, original P9 A/B hashes differed and source path rehost static PASS.
- `hcp_p18_eoe_480_readonly_admission.py` → `P18_EOE_MODELS_PENDING_ORIGINAL_SHA_COPY`, next pre-registered question index 3 `p18-mmlu-003`, remaining question-NLL pairs 474/480.

## 정확한 재개 순서

1. EOE process_output for `proc_edd7dafc6a64ccb31f4b644910df0399` and `proc_6223c668c2043cb5d772868c894b1a51`, both EXITED 0; keep partial-copy states on any failure.
2. EOE run `/opt/homebrew/bin/python3.11 .../hcp_p18_eoe_d50_model_copy_seal.py --inspect` then `--seal` and SHA receipts ensure original BF16+native base complete.
3. Run dev-only 203 A/B EOE Metal parity (not 480 test). Verify XOX unmodified route, ACK/epoch, token SHA. If mismatch, stop; do not tune heldout.
4. Run `hcp_p18_eoe_d50_original_bf16_unit.py --index 3` once, and native `hcp_p18_eoe_d50_native_AB_unit.py --index 3` once. Independently audit D50 original BF16 numpy and native C worker SHA/readback. Repeat only first pre-frozen unscored items in order.
5. Later implement full finite original teacher task greedy 8/32/128, native model A/B complete output and warm matched triple trials, 480-question prereg gate and only then research PASS/BLOCKED. No automatic live promotion.
6. **Do not upload D50 source BF16 checkpoint files, quantized native model raw, private test prompts/tokens, .f32/.npy, user secrets to GitHub.** Commit only code plus aggregate/hand-off docs.

## Verdict

**P18 EOE source rehost / Metal compile / input SHA authority / model-copy fail-closed admission PASS. Model transfer incomplete; EOE actual inference not executed. Final research quality gate BLOCKED; incumbent policy A stays live.**

## 병렬 연구 소스 일치 확인

- 최신 연구 브랜치에서 별도 EOE BF16 480문항 실행기 `tools/hcp_p18_eoe_d50_original_bf16_480_unit.py` SHA `6cfebe610ca4d69160d9d0967da1a0423eea4d1265726c488e21dfff6740274b` 및 original 1024 BF16 core가 이미 커밋되어 있음을 확인함. 별도 구현과 내용 충돌로 덮어쓰지 않음.
- 본 문서의 `hcp_p18_eoe_d50_original_bf16_unit.py` SHA `070d3ecdc58c02d5c85e380f94d3f42d1389896651a1688be2226d7561db9190`는 **연구용 대체 구현**으로 별도 파일에 보존. 모델 복제 완료 후 dev-only 검증·원본 NLL 재현성 확인 전 어느 쪽도 운영 인증의 근거가 아님.
- EOE 네이티브 D50 capture-path-only manifest SHA `61f424fe7a2340929238c6b4569613f5ab32b7a663c488a76744935f30711870`; 실제 보존 본체 SHA는 상기 C sha로 재확인.
