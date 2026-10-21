# BEGLIN Q: 196개 텐서 국소 혼합 양자화 실기기 인계 (2026-10-21)

**결론:** 실제 Apple Silicon BEGLIN CPU에서 196개 투영 텐서 중 28개만 qNg64 8비트로 승격하는 혼합 정책을 구현·검증했다. 그러나 가중치 복원 오차로 선택한 정책은 동일 비용 무작위 대조보다 실제 PPL이 나빴다. 최종 Q 가설은 미판정이다.

## 데이터·엔진 고정

- 모델 Qwen/Qwen2.5-1.5B-Instruct. 공식 HF revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, EOE MLX BF16 revision 4ae77cb209f06199b8df1c94e21ff341332a3a89.
- EOE cached BF16 checkpoint 전체 파일 SHA-256: `073400b311d049b6379b8f3f7ad5a0258268031dd009b99389cfd4030144850a` (3,087,467,067바이트). L0 q_proj와 L0 input_norm의 원본 BF16 바이트가 Alienware 공식 HF checkpoint와 동일함을 확인했지만 338개 텐서 전체의 바이트 동일성까지 별도 증명한 것은 아니다.
- BEGLIN 원본 qwen_infer.c SHA-256: `52c8f5fd683b5ef0726e4738ab4b441624f54c807a3e71fb9d932cb79938d112`. SHA 고정 원본의 별도 연구 복사본에서만 mixed 로더 추가·컴파일. main, 운영, XOX 작업 트리, 모델 체크포인트 불변.
- 28개 계층 × attention q/k/v/o + MLP gate/up/down = 총 196개 원본 BF16 투영 가중치. 동일한 151,936 vocabulary, tokenizer, 두 개 공개 합성 chat window, 임베딩 공유 FP32 출력 헤드, 모든 비투영 가중치 F32를 유지한다. **BF16 원본 가중치를 FP32로 확장한 연산 기준이지 BF16 산술 런타임은 아니다.**

## 오차 프록시 기반 선정 (성능 측정 이전 고정)

- 사전 고정 규칙: 각 텐서를 원본 BEGLIN `gguf_quantize_q4g64_error_feedback`(packed nibble)과 `qng64_quantize_f32(n8)`(bitplane)로 독립 압축·복원하고, `relative_weight_error_q4² − relative_weight_error_n8²`이 큰 계층 4개씩을 역할별 선정. **이것은 가중치 Frobenius 오차 프록시이며 실제 downstream 민감도 측정이 아니다.**
- 역할별 4계층, 총 28개 n8 + 168개 q4. NumPy PCG64 seed=73 무작위 대조도 각 역할 4개를 뽑아 **정확히 같은 텐서 종류·저장 비용**을 사용한다. 모델 정답, PPL, 최종 홀드아웃은 선택에 사용하지 않았다.
- 선정된 프록시 레이어: q_proj 6/14/15/20; k_proj 5/12/26/27; v_proj 22/23/24/25; o_proj 6/8/16/21; gate_proj 18/19/21/22; up_proj 16/17/18/19; down_proj 12/14/19/27.
- 고정 scanner SHA-256: `720de694838d3a85156afae3732974f3867db1c8ce4957e55778da77724a0d8a`, 정책 파일 SHA: 프록시 `f65f50421cd9e9bac32bc780805942d68c3d503075ce962ce25315fdb5914cb9`, 역할 일치 무작위 `72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5`.

## 196개 투영 가중치의 논리적 저장 비용

| 정밀도 | n8 | packed4 | 가중치 저장량(스케일 포함) |
|---|---:|---:|---:|
| BF16 원본 | 해당 없음 | 해당 없음 | 2,620 MB |
| 전체 n8 | 196 | 0 | 1,392 MB |
| 프록시 혼합 | 28 | 168 | **831 MB** |
| 무작위 혼합 | 28 | 168 | **831 MB** |
| 전체 packed4 | 0 | 196 | 737 MB |

프록시 정책의 196개 텐서 평균 정규화 가중치 제곱오차는 0.0219004, 동일 비용 무작위는 0.0223673였다. **프록시는 가중치 복원 오차를 2.09% 적게 만들었으나 아래 실제 언어모델 손실에는 오히려 불리했다.**

## 실제 전체 28계층 BEGLIN 결과

| 투영 정밀도 | teacher-forced PPL (21토큰) | 기준 대비 상대 NLL 증가 | 실제 디코드 tok/s | PPL 프로세스 최대 RSS |
|---|---:|---:|---:|---:|
| 원본 BF16 소스 F32 | **1,988.57** | 기준 | 4.744 | 9.27 GB |
| 전체 qNg64 n8 | 2,148.11 | +1.02% | 0.190 | 5.49 GB |
| **역할 맞춤 무작위 28개 n8** | **3,411.23** | **+7.11%** | **1.233** | 4.92 GB |
| 전체 packed Q4G64 | 4,649.32 | +11.18% | 13.287 | 4.83 GB |
| **오차 프록시 상위 28개 n8** | **4,747.70** | **+11.46%** | **1.234** | 4.92 GB |

- 모든 정책은 동일한 원본 BF16 가중치와 입력 토큰 ID 파일을 사용했다. 원본 F32, 전체 n8, 전체 Q4 baseline은 이미 게시된 같은 실기기 영수증에서 재조회했다. 혼합 후보의 모델 로딩·실제 바인딩 수·원본 전체 vocabulary 로짓 생성, PPL 21토큰과 벤치마크를 새로 실행해 모두 EXITED/0 확인.
- 같은 첫 공개 합성 입력의 FP32 기준 KL: 전체 n8 0.000080, 무작위 혼합 0.0561, 프록시 혼합 0.0973, 전체 Q4 0.2450. 최상위 토큰 9707은 모두 같았지만 로짓 분포는 달랐다.
- 동일 비용 무작위 혼합은 프록시 혼합보다 PPL이 약 28% 낮다. 프록시 혼합은 **전체 Q4보다 PPL이 악화**됐으며, 따라서 비트폭 승격이 전체 품질을 항상 개선한다는 단조 가정도 이 작은 사례에서 불성립.
- 속도는 native BEGLIN `bench 1`: 10-token prefill + 8 warmup + 1 timed token, 3회 반복 중 **최속값(best-of-3)**. 중앙값이나 sustained throughput이 아니다. Q4는 6개 CPU worker, n8은 현재 scalar kernel이므로 속도 차이에 SIMD·스레드 영향이 있다. BEGLIN Metal GPU, 실제 3계열 task AUROC, 전력량 측정 아님.
- 실제 소규모 21토큰 PPL 로그에 `[ppl] fp32` 문자열이 나오지만 이는 레거시 g_int4 출력 표시다. mixed 실체는 로더의 28×qNg64/168×Q4G64 텐서 카운트로 검증했다.

## 검증 증거 및 재현

- 실기기 스캔 `proc_a7d376931d5be015e5fff4eeee9c2f2c`; mixed 컴파일 `proc_7aa4d90e249024983b6f9b22b538465e`; 두 혼합 로짓 `proc_fae9aaa72c2728ff333f8baf626b01d3`, `proc_9a369e6852b2ad4bee796fcbad617d87`; PPL `proc_b566089c4ff123c0ac944e03f7650102`, `proc_3e47ecd48eb4ec42cc3c60299f45302c`; 디코드 `proc_29a036b8feb060badfe9bf7f1e92c6a9`, `proc_51727976cd9f1f8e374c2a0d48d1eed0`. **8개 전부 EXITED/0**.
- 신규 역할·bitplane·packed nibble 검증 단위시험 `tests/research/test_q_native_scan_196.py` **6/6 PASS**. Python 독립 감사에서 196개 고유 텐서·정확히 4/역할·정확히 같은 비용·동일 모델/입력·5개 로짓 SHA·KL·저장 크기 확인.
- 실험 구현: `research/oml_acs/q_native_scan_196.py`, `research/oml_acs/q_full_model_mixed_lab.py`.
- 공개 집계 영수증: `docs/research/OML_ACS_Q_196_REAL_MIXED_NATIVE_EXECUTION_20261021.json`; 기존 baseline: `docs/research/OML_ACS_Q_NATIVE_FULL_MODEL_PPL_MEMORY_REAL_20261021.json`, `docs/research/OML_ACS_Q_NATIVE_FULL_MODEL_DECODE_THREADS_20261021.json`.

## 연구 결론·다음 게이트

**Q_FINAL NOT_READY.** 소스 텐서별 가중치 오차만 최소화하는 것은 실제 순차 LLM의 손실을 예측하는 충분조건이 아니다. 개선의 다음 단위는 (1) source-only calibration에서 *실제 layer output/logits/NLL* 변화를 측정하는 기능적 민감도 추정, (2) nonmonotonic bit-width 선택과 실제 형식별 커널 비용을 포함하는 혼합 최적화, (3) source/selection/evaluation 데이터를 분리한 새로운 독립 평가, (4) 필요 시 native n8 scalar를 CPU SIMD/Apple Metal로 최적화 후 동일 정책 재평가다. 원본 V6에서 소비한 24개 문항 ID의 권위 있는 출처·SHA는 아직 없으므로 최종 IID/OOD 평가를 열지 않는다. 자동 승격, production/main 변경 없음.
