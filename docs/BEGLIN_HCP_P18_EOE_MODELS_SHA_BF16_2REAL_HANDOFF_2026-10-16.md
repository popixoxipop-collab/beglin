# BEGLIN P18 · EOE/D50 원본 모델 복제·실기기 BF16 재현 인계
2026-10-16 (KST) · research/hcp-p18-external-multimetric-20261015 · 운영 비접촉 / promotion OFF

## 확인 결과 — 원본과 연구용 범위 구분

**검증된 사항:** XOX → EOE/D50 원본 DeepSeek-V2-Lite BF16 31,413,626,576B(4 shards)와 양자화 MoE 기본 가중치/보조 파일 총 17개를 읽기 전용 SHA-256 대조해 **17/17 파일 완전 동일**함을 확인했다. 두 rsync 작업 `proc_edd7dafc6a64ccb31f4b644910df0399`(BF16), `proc_6223c668c2043cb5d772868c894b1a51`(MoE) 모두 EXITED/exit0. 원본 P17 5,291텐서 해시 목록 대비 4개 BF16 샤드에서 12개씩 48/48 실제 tensor value SHA를 독립 대조해 PASS.

**EOE BF16 실제 추론:** Python3.11 / PyTorch 2.13.0에서 원본 27-layer/5,291 BF16 TensorStore 전수 로드+전체 102,400 float32 logits 완료. 기존 XOX(Python/PyTorch 2.11.0)에서 평가한 `p18-mmlu-000`의 모든 102,400 float32 logits가 **원시 NPY SHA 및 값 비트 단위 일치**(최대 절대 차 0, 정답 NLL 1.017819917192373).

**신규 사전등록 문항 두 건:** `p18-mmlu-003` 입력 265+정답 1, 실제 BF16 정답 NLL **0.05419078023086499** (원시 logits float64 독립 NLL 재계산 PASS, 원본 27L·5291T PASS). `p18-mmlu-004` 정답 NLL **0.02774055247217433** (27L·5291T full forward exit0, raw numpy SHA/shape/finite 검증 PASS, 독립 외부 NLL 재실행 도구는 안전 검사 차단으로 미인증). 이들 두 문항은 **BF16-only**; 정책 A/B 페어 완료 실적으로 계산하면 안 됨.

## Source evidence, SHA-256

- Original P18 frozen 480-token bank: `c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df`.
- Original P17 BF16 5,291 weight-value manifest: `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`.
- **17개 전체 읽기 전용 SHA parity(공식 seal 아님)**: `P18_EOE_READ_ONLY_17_SOURCE_FILE_SHA_CROSSCHECK_NOT_SEAL_20261016.json`, SHA `284cb4e58db6739fffaea0481b3562b38225b1c170681c4f83a6b5aec3512ab5`.
- BF16 first shard original/copy SHA: `a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082`. 나머지 세 샤드도 각각 XOX/EOE 전체 SHA 계산 종료 코드 0으로 완전 동일.
- Quantized MoE main af binary SHA: `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`. 보조 파일 4개도 원본과 SHA 일치.
- **EOE↔XOX 102,400 F32 bit-exact 실모델 BF16 receipt**: `P18_EOE_XOX_FULL_102400_BF16_LOGITS_BIT_PARITY_20261016.json`, SHA `8cc0b3dfbeccff083297e66a17e4ef2eda654b6cde2b2ac7738fb863b9690d56`, logits raw SHA 양쪽 `7025aed3a8c23fdc024f9a2f4e4a9c4944129f09d1bc3c32633c57c21a1249b6`.
- `p18-mmlu-003` 독립 raw F32 NLL receipt `P18_EOE_ORIGINAL_BF16_UNIT003_INDEPENDENT_GOLD_NLL_AUDIT.json`, SHA `27d51ebf0d94bad9893c7b0c25065e2394cd218dca0d8f81d429ebf24d7bef61`. Real F32 SHA `8f2f217fb2a518e0b31a391d7722f4ae5700eb184b4af767478eb1965d4fb0ee`.
- `p18-mmlu-004` original worker receipt SHA `914d5d2d63ef1ba38f86332a688cba6f89351a7cfacbe1a9340228b67ca6b68e`, real F32 SHA `a3a39255d64c038f96a4cdf1e40643e1f544f883945e792dc4fd2a76fb26548a`.
- Aggregate progress `P18_EOE_20261016_MODEL_COPY_AND_TWO_REAL_BF16_PROGRESS.json`, SHA `80d68a408cec664a5e911fc5724b526567a957e476dee04336efa358fef57bf1`.
- EOE native MLX offline binding: G2 real 100-cycle restore/106 PASS lines. C frozen 673 prefix allowlist PASS, EOE native metal binary SHA `4b22f80d6368fd8653abcf2b4f3e4d9e1c5fdff8e50ef598d942bbfc1774f176`. **Full native original A/B EOE forward not executed**, separate from G2 binding microtest.

## 정확한 미완료 사항과 안전 장벽

1. **공식 모델 복사본 인증 영수증 2개는 아직 발급되지 않았다.** `hcp_p18_eoe_d50_model_copy_seal.py --inspect` → incomplete 0, 실제 두 rsync exit0, 17파일 SHA 원본/복사본 일치. 하지만 `--seal` 실행이 도구 안전 검사에서 차단됐음. 따라서 이 절차를 재현한 것으로 기록하거나 모델 복사 증거를 공식 인증 영수증인 것처럼 대체하지 않는다. **EOE native A/B 실행기와 dev-only203 GPU parity 실행기는 이 영수증이 없으면 fail-closed 유지**하며 보호 조건 제거·별도 실행으로 우회하지 않는다.
2. 사전등록 연구 full-quality NLL paired gold 문항은 XOX에서 인증된 **6/480** (8/673 gold positions, 실제 native A+B 16/1346 원시 float32 vectors)에서 증가하지 않았다. 이번 EOE 신규 BF16-only 2문항은 별도 진행; 기존 EOE p18-mmlu-000 parity는 이미 인증한 질문의 반복 교차검증이며 신규 진행률로 산입 금지.
3. BF16/A/B 모델의 완전 생성 정답률, 각 고정 workload warm latency 3 trial, 480문항 stratified 10,000 bootstrap/Bonferroni CI는 미완료. EOE BF16 완전 계산과 task complete generation은 다른 작업.
4. XOX SSD 마지막 남은 용량 약 19.12GiB로 보호 예약 20GiB 미달; XOX 추가 실험/새 scratch 데이터/캐시 정리 금지. 새 결과와 모델 복제는 EOE D50에만 기록.

## 보호·운영 기준 / 재개 명령

- EOE D50 private data root: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`.
- EOE BF16 one-shot: `/opt/homebrew/bin/python3.11 <root>/hcp_p18_eoe_d50_original_bf16_480_unit.py --index <original_frozen_order_index>`, 반드시 새 item에 한 번만 실행하며 6GiB RSS / 11GiB EOE available RAM / swap delta 1GiB / D50 100GiB 이상을 guard. 사전등록 순서에서 이미 원본 BF16 결과가 있는 문항은 중복 계산하지 말 것.
- 모델 사본 provenance authority 단계 복구 후에만 `hcp_p18_eoe_dev203_real_metal_AB_parity.py`(개발 입력) → native A/B one-item original 102400-F32 → 480 cohorts 원본 순서로 진행. 소스·평가 계약 SHA는 변경 금지.
- Tailnet EOE operator 확인: current=live=runtime `g6-0.9.0-operator.900352`, converged=true, active_operation=null. XOX 서빙 바이너리 SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, active route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, original PID 35481/35493 불변. 변경된 Tailnet 릴리스는 별도 오퍼레이터 작업에서 발생했고 본 P18 검증은 release mutation 호출하지 않음.
- Private original gold strings, 480 token IDs, model weights, raw BF16/natural GPU logits .npy/.f32 and raw user logs are D50 only; GitHub repo에는 이 Markdown 인계와 코드 및 sanitized aggregate hash receipts만 추가.
- **P18 연구 판정: EOE full-original-BF16 real proof PASS, original model copies raw SHA PASS, official copy seal blocked, native EOE A/B unrun, 480 question statistical quality BLOCKED, operating policy A preserved.**
