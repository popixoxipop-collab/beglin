---
name: vdsp-general-serving-engine-goal
description: vdsp를 Apple Silicon GPU+MoE 대형모델 지원 범용 서빙엔진(MLX/llama.cpp급)으로 확장하는 것이 사용자의 장기 목표
metadata: 
  node_type: memory
  type: project
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-09-01T12:20:00.000Z
---

사용자의 장기 목표: 현재 vdsp(Apple Silicon CPU 전용, 단일 C 파일 `qwen_infer.c`,
q4g64 양자화, SME2/NEON 커널 통합 사례연구)를 **Apple Silicon GPU + MoE 대형모델을
지원하는 범용 서빙엔진**으로 확장 — MLX나 llama.cpp가 하는 것처럼.

**Why**: FreeToken(arXiv:2608.16157, FlashML-org — NVIDIA GPU 타겟 edge MoE
서빙엔진, $q^\star$ bandwidth-adaptive CPU-GPU co-execution 정책)을 보고
"우리 to-be와 같은 방향"이라 판단, 2026-08-22 세션에서 즉시 Opus 에이전트에게
설계 위임.

**핵심 구조적 차이(이미 사용자와 합의됨, 향후 설계의 전제)**: FreeToken의
핵심 문제(CPU RAM ↔ GPU VRAM PCIe 대역폭 병목)는 Apple Silicon의 unified
memory 때문에 그대로 적용되지 않음. Apple Silicon에서 진짜 문제는 (a) 주어진
연산을 CPU(NEON/SME2) vs GPU(Metal) 중 어디로 라우팅할지 결정하는 정책,
(b) 동시 실행 시 CPU/GPU 상호 간섭(이미 분리해둔 CPU/GPU 동시성 컴패니언
논문이 "partner-by-dispatch-mode interaction"으로 부분 규명 — 특정 GPU
파트너 상대로만 SME2 dispatch가 급격히 취약해짐, 균일한 페널티가 아님).

**MoE 관련 우려**: MoE는 토큰별 sparse expert 활성화라 실효 배치가 작음 —
SME2의 이점은 배치 크게 커야 나타남(batch=16 near-zero, batch=64 +37%,
기존 dense 모델 측정 기준). 즉 SME2가 MoE expert 연산 자체엔 안 맞고
GPU/Metal이 기본, SME2/NEON은 shared/dense 서브레이어나 여러 토큰을 같은
expert로 묶어 배칭할 때만 쓰는 구조일 가능성.

**진행상황**: 2026-08-22, Opus 에이전트(fresh, model=opus — fork는 parent
model인 Sonnet을 그대로 물려받아 opus 불가라 fresh agent로 위임)에게 단계적
설계 로드맵 위탁 완료. 실제 엔진소스(`macstudio:/Volumes/D50/vdsp @ 3537ce6`)와
논문 표를 직접 읽고 계산한 결과물:
- 산출물(영구 위치로 이동완료 2026-08-22): `~/Desktop/vdsp_v2_design/DESIGN_vdsp_v2_apple_silicon_serving_engine.md` + artifact https://claude.ai/code/artifact/5f7cf8e9-ae18-42bf-92de-06b808d91913
- ★★ co-execution 이득은 컴패니언 논문 기존 표 산술만으로 이미 나와있었음: M4 +20.5%(921.32 vs 764.44), M1Max +16.4% — "2배"가 아니라 "+20%" 효과, 예산배분에 결정적
- ★★ MoE decode는 SME2 구조적 배제: expert당 실효배치(B·k/E)가 SME2 최소문턱 넘으려면 Qwen3-30B-A3B 기준 B≥256 필요 + B≈32에서 이미 bandwidth-bound 전환 → SME2는 prefill+dense서브레이어에만 자리
- ★ retention%(간섭시 83.9%)가 아니라 절대처리량 기준으로는 최악조건에서도 NEON+SME2가 NEON보다 12.7% 여전히 빠름 — 라우터는 절대값 최적화해야 함
- ★★★ 하드웨어 제약(유일하게 노력으로 못 만회): SME2 기기=bob(M4,16GB) 1대뿐, 64GB 기기 2대는 SME2 없음 → MoE+SME2 동시검증 불가능한 상태
- 논문 권고: 현재 스코프 그대로 출간(MAJOR_REVISION 사유=스코프아닌 실행이슈, 이미 처리됨), 확장은 future work 한 문단만
- GPU 백엔드 도입 시 기존 QA 전체가 의존하는 "bit-identical 검증"이 원리적으로 불가능해짐(fp16 누산 등) → 대체 검증 프로토콜(oracle고정+rel-RMS예산+token agreement) 설계를 리팩터 착수 전 배치 권고
- 최우선 액션 3개(병렬가능): P0.1 기존바이너리로 co-execution 10분측정(신규코드 0줄), P0.3 mlx::core 링크 spike 200줄, V1+P0.2 기존 64GB M1Max서 MoE regime 실측

**P0.3 부분완료 (2026-08-23, bob/M4)**: (a) `pip install mlx`로 C++ 헤더+dylib
확보, Python 런타임 없이 순수 clang++로 링크 성공, Metal 자동선택 확인.
q4g64와 동일 설정(group=64,bits=4) quantized_matmul 4096x4096@M=64 왕복지연
**평균 1.10ms(200회)** → G3(레이어당 분할, 196회/token)는 이것만으로
~215ms/token 오버헤드라 트랩 확정, G1/G2가 유효한 설계점. (d) **q4g64→MLX
니블 패킹 순서가 완전 동일**(byte b의 low nibble=column 2b) — repack도
requant도 필요없고 `bias=-8*scale` 메타데이터 계산만 필요(설계문서가
예상한 두 시나리오보다 더 좋은 결과). (b)(c)도 완료(같은 날 후속): mmap 영역을 `newBufferWithBytesNoCopy`+Shared
storage로 직접 wrap 성공(복사 없이 GPU가 정확히 읽음), CPU-write 직후
GPU-read도 배리어/`didModifyRange` 없이 5/5 완전 코히런트 — unified memory가
공짜로 보장. **P0.3 전체 완료**: zero-copy 가중치 로딩 가능+수동 동기화
코드 불필요, 설계문서가 우려한 3개 리스크 중 2개 해소. 남은 갭: 실제
q4g64 니블 레이아웃으로 quantized_matmul까지 엔드투엔드 테스트는 아직
(단순 float 버퍼로만 검증). M1 Max/macstudio 전체 미측정(다른 세션 4개
활성). Raw: `~/Desktop/vdsp_v2_design/p0.3_spike/`

**P0.1 완료 — 양쪽 칩 모두 (2026-08-23)**:
bob/M4 실측 B=1 +0.9%(FAIL), B=16 +21.6%(PASS), B=64 +22.4%(PASS) — 설계문서
산술추정치(+20.5%)와 거의 정확히 일치. 하네스 함정 2건: (a) `llama-cli -no-cnv`가
non-interactive stdin에서도 무한 프롬프트루프 안 막아줌(15GB 로그 폭주 경험,
`llama-simple`/`llama-batched`로 대체) (b) `llama-batched -np N`은 `-kvu`
명시 안 하면 coupled-sequence KV 에러로 죽음. B=256은 엔진 하드캡(64)에
막혀 스킵.

macstudio/M1 Max도 같은 날 후속 완료 — 사용자가 명시 승인("필요하면 다른
작업들 맥스튜디오에서 정지 시켰다가... 다시 재개해도 돼")한 뒤, 컴패니언
논문 Threats-to-Validity가 이미 검증한 전례와 동일하게 상주 프로세스
`brain.selfplay`(Finance, PID 26018, 99.4% CPU 상시)를 측정구간만 SIGSTOP
→ 완료 즉시 SIGCONT. 결과: **B=1 +6.3%(FAIL), B=16 +22.7%(PASS),
B=64 +21.7%(PASS)** — M4와 거의 동일한 패턴, GPU/CPU 상호간섭도 거의
없음(concurrent 값이 solo 대비 ~2% 이내). `qwen_infer`는 bob과 동일 소스
파일(SME2 접미사 없는 빌드)이며 `kai_sme2_available()` 런타임 자동감지로
M1 Max에서 안전하게 NEON 폴백(SIGILL 위험 없음 확인). 하네스 함정 1건 추가:
macstudio에서 `QWEN_BASE`는 `weights/` 하위가 아니라 엔진 루트
(`llm_engine/`)여야 함(`ref/prompt_ids.i32` 조회 경로가 `$QWEN_BASE/ref/...`
고정이라, 다른 3개 가중치 경로는 별도 env var로 명시하는 구조).
**게이트(+12%) 양쪽 칩 모두 B≥16에서 통과 → SME2 유무와 무관하게 독립
서빙엔진 co-execution 방향 데이터로 최종 확정.** raw data:
`~/Desktop/vdsp_v2_design/p0.1_results/`(M4), `p0.1_results_macstudio/`(M1 Max).

**P0.2 V1 완료 (2026-08-23, macstudio/M1 Max, DeepSeek-V2-Lite-Chat-4bit-mlx
via MLX-LM, 실제 추론)**: `MoEGate.__call__` 몽키패치로 레이어별(26개 MoE
레이어, 레이어 혼합 금지 — 첫 시도에서 26개 레이어 인덱스를 그냥 합쳐버려
"B=1에 59/64 expert"라는 말도 안 되는 결과가 나왔던 버그를 게이트 인스턴스별
분리로 수정) 실측한 decode-step당 distinct-expert 수: **B=1 6.0/64(=top_k,
검증) / B=8 32.3/64 / B=16 45.9/64 / B=32 57.4/64(90%) / B=64
62.4/64(97.5%)**. 설계문서 §5.2의 균일분포 이론표(6/35/51/61/64)와 형태는
거의 일치하지만 실측이 매 B에서 약간 더 낮음 — 실제 학습된 라우터가 균일
가정보다 오히려 더 편중돼있다는 뜻(이론표가 낙관적이었음). 도중 프롬프트
배칭 버그 1건 추가 발견·수정: 손으로 쓴 real text 8개만 순환배치해 B>8에서
동일 시퀀스 중복 발생 → greedy decode라 중복은 라우팅도 동일 → B=16/32/64가
전부 B=8과 같은 값(30.8)으로 인위적으로 고정되는 현상 → 더 큰 코퍼스에 대한
슬라이딩 윈도우로 교체해 해결.

V4(달성대역폭 vs M1 Max 400GB/s 이론한계)는 시도했으나 절대수치는
결론부재 — bytes/forward는 B에 따라 늘지만 achieved GB/s는 오히려
**감소**(182.6→116.6→110.7→85.7→53.0 GB/s, B=1→64전부 400GB/s 미만)했는데,
이는 대역폭한계에 수렴하는 워크로드라면 나올 수 없는 방향(설계문서 Q5가
이미 명시한 "SLC/GPU캐시 재사용으로 analytic 대역폭 추정이 신뢰 불가"
불확실성과 정확히 일치 — decode 스텝 사이 안 바뀌는 attention/dense
가중치가 매 스텝 DRAM에서 새로 안 읽히고 GPU측 캐시에 남아있을 가능성).

**최종 판정(V1만으로 게이트에 답 충분)**: B≥32에서 이미 레이어당 expert의
90%+가 매 forward 스트리밍됨 → **MoE decode가 realistic serving batch
범위 내에서 사실상 dense-weight-streaming으로 수렴 = bandwidth-bound
YES**. → 설계문서 P5는 routing(P5b)이 아니라 **bytes-reduction(P5c: expert
양자화, 스텝간 expert 캐싱, prefetch)으로 재스코프**해야 함(decode-time
MoE-FFN 서브레이어 한정 — co-execution/P0.1은 prefill+dense/shared
서브레이어엔 여전히 유효, 별개 결론). raw: `~/Desktop/vdsp_v2_design/p0.2_results/`.
남은 것: V2(커널 크로스오버)/V3(token-gathering spike)/V5(A/B), Track A의
더 어려운 케이스(Qwen3-30B-A3B), Track B(SME2+MoE, 신규 엔진코드 필요).

**V2 완료 (2026-08-23, bob/M4) — 예상 반전**: MoE expert shape(1408×2048)
실측 결과 SME2가 M=1부터 M=64 전 구간에서 NEON(`gemm_qXg64_sdot_mt`,
T=10, 논문 검증방식 그대로)을 이김 — M=1 2.06배, M=5-11 1.72-1.80배,
M=18-64 1.27-1.30배. 기존 dense-projection shape(8960×1536, "M=16
near-zero, M=64 +37%")와 정반대 방향. **액션 아이템**: 실제 엔진의
`kai_route()` 게이트(`sme2_kai.c:51-54`, `M >= kai_sme2_min_m()`)는 SME2
커널 자체의 하드웨어 타일크기 기반 단일 고정임계값이라 shape-무관 —
MoE expert shape엔 불필요하게 보수적임(M=1도 이미 이기는데 게이트가
차단). Track B 착수 시 이 게이트를 그대로 재사용하면 안 됨. 벤치마크
방법론 버그 2건을 논문 기존 수치와 대조해 자체발견·수정(잘못된 함수
`gemv_q4g64_mt` 루프→`gemm_qXg64_sdot_mt` 1회 디스패치로 교체, T=6→T=10).

**V3 완료 (2026-08-23, bob/M4) — GO 판정**: macstudio에서 뽑은 실제
decode-step 1회의 per-expert row 분포(B=64, 63개 expert 활성, 총 384
token-expert 배정, count 1~17)를 bob으로 옮겨 실측. gather 없이 384개
개별 M=1 디스패치=12.23ms vs 63개 expert별 실제-M으로 gather(실제
memcpy gather+scatter+quant-pack 비용 전부 포함)=4.07ms → **gather가
3.01배 빠름**. 3MB 활동데이터가 SLC에 다 들어가는 크기라 메모리이동
비용이 저렴한 게 핵심 이유로 추정. **CPU MoE decode via SME2는 죽지
않음 — Track B(SME2+MoE 신규 엔진코드) 착수 근거 확보.**

**Track A V1 재실행 완료 (2026-08-23, macstudio, Qwen3-30B-A3B-4bit,
E=128,k=8 fine-grained)**: DeepSeek(E=64,k=6)보다 실제 라우팅이 훨씬 더
편중됨 — B=64에서도 레이어당 76.5/128(59.8%)만 활성화(균일가정 98%,
DeepSeek 실측 97.5%와 대조). **P0.2의 "bandwidth-bound YES"는 아키텍처
의존적** — coarse-grained(DeepSeek류, E/k≈11)는 P5c(bytes-reduction)
쪽, fine-grained(Qwen3류, E/k=16)는 B=64에도 40%가 안 만져지는 진짜
sparsity가 남아있어 P5b(routing) 투자 여지 있음. Track B가 여전히
DeepSeek-V2-Lite를 타겟으로 삼는 건 "보수적인 케이스"를 테스트하는
셈(Qwen3류 모델이면 SME2 투자가 더 유리했을 것). 계측 함정 1건:
QuantizedLinear vs Linear(4bit로드시 실제 클래스가 QuantizedLinear라
처음엔 아무것도 안 잡힘, ZeroDivisionError로 즉시 발각).

**Track B Phase MoE-1 완료 (2026-08-23)**: DeepSeek-V2-Lite MLX 4bit
체크포인트를 vdsp 엔진이 로드 가능한 포맷으로 변환+검증(forward pass는
아직 없음, 순수 로딩 단계만). 구현 착수 직전 실측으로 **두 가지 기존
전제를 뒤집음**: (1) P0.3의 "byte 재해석만으로 무손실 변환(bias=-8*scale)"은
합성데이터 한정 결과였고 실제 체크포인트에선 그룹의 32.7%가 1%+ 이탈
(최대 33.4%) — 사용자가 손실 근사 대신 **엔진에 진짜 affine 4bit 디코드
경로(`K_Q4G64AF`) 신규 추가**를 선택, 결과 160/160(Python)+6/6(C, bob↔macstudio
교차언어) 완전 일치로 오차 0 검증. (2) DeepSeek-V2-Lite의 attention은
MLA(압축 KV+부분 RoPE)라 기존 GQA 엔진코드로 못 돌림 — 사용자가 모델은
유지하고 MLA도 별도 phase에서 신규구현하기로 결정, **Track B 전체 잔여
스코프가 애초 추정보다 큼**(MLA가 MoE FFN 작업과 비슷한 규모의 신규
서브시스템). 표준 구현: `mlx_deepseek_to_q4g64af.py`(변환)+
`verify_moe_conversion.py`(Python 검증)+`verify_moe_af_load.c`(C 검증,
qwen_infer.c 미변경 — 2700줄 프로덕션 파일에 forward pass 없이 손대는
리스크 회피, 실제 통합은 라우터+forward가 생기는 MoE-2로 유예).
Raw: `~/Desktop/vdsp_v2_design/trackb_moe1_results/`. 남은 것: MoE-2(라우터+
MLA attention+정확도게이트)/MoE-3(gather+SME2 dispatch, 여기서 V5 처음
가능) — 각각 별도 세션.

**Track B Phase MoE-2a 완료 (2026-08-23) — MLA attention+YaRN RoPE 정확도
게이트 PASS**: layer 0(dense) 단독으로 MLA를 검증하는 standalone C
프로그램(`mla_verify.c`, qwen_infer.c 미변경)을 MLX 실제 forward pass와
대조 — 8개 실텍스트 토큰 position 전부 rel-L2 ≤1.3e-3(plan의 hard fail
기준 1e-2 대비 충분히 여유). 착수 중 **실측으로 버그 2건 발견·즉시수정**
(둘 다 문서/기억이 아니라 실제 MLX 호출로 검증): (1) DeepSeek YaRN이 쓰는
`mx.fast.rope(traditional=True)`는 interleaved pairing(x[2i],x[2i+1])이지
qwen_infer.c 기존 rope_apply()의 rotate_half 컨벤션이 아님. (2) **더 결정적인
버그**: `freqs` 배열이 곱셈이 아니라 나눗셈으로 쓰임(`angle=pos/freqs[i]`,
`pos*freqs[i]` 아님) — 균일 freq값(예:전부 1.0) 테스트로는 곱셈/나눗셈 구분이
원리적으로 불가능했고(양쪽 다 같은 결과), **서로 다른 freq값[1,2,3,4]로
재테스트+MLX의 실제 pre/post-RoPE 텐서를 디버그훅으로 직접 캡처해 회전각을
역산**해서 발견. 수정 전엔 pos=0(각도=0, 항등회전이라 버그가 안 보임)만
멀쩡하고 pos=1~7이 rel-L2 0.5~0.97(사실상 무상관)로 파국적으로 틀렸었음 —
"pos 0만 맞고 pos≥1부터 완전히 틀어짐"이라는 패턴 자체가 위치의존 로직(RoPE)
버그라는 결정적 단서였음. YaRN 주파수테이블(factor=40,mscale=0.707)은
`mlx_lm/models/deepseek_v2.py` 수식 그대로 이식, MLX 실제 테이블과
float32 정밀도 한계까지 정확히 일치 확인. 압축이 아니라 확장된(expanded)
K/V 캐싱을 채택한 설계결정은 "MLX 레퍼런스 자체가 그렇게 구현돼있음"을
코드로 직접 확인 후 내린 것이라 유지(진짜 압축-캐시 최적화는 미래 처리량
phase로 유예). 문서 갭 1건 추가 발견: MoE-1의 "양쪽 호스트 동일 사본"
주장이 실제론 거짓이었음(f32 blob이 bob엔 없었음) — mla_verify.c 자체의
FATAL 체크로 즉시 발각, relay로 수정. Raw:
`~/Desktop/vdsp_v2_design/trackb_moe2a_results/`. 남은 것: MoE-2b(라우터+
naive MoE FFN forward+27레이어 전체 정확도게이트)/MoE-3 — 각각 별도 세션.

**Track B Phase MoE-2b 완료 (2026-08-23) — 라우터+MoE FFN+27레이어 전체
forward, 첫 실행에서 바로 PASS**: MoE-2a의 MLA attention 코드(AFTensor/
decode_af/YaRN/interleaved RoPE)를 그대로 재사용해 27레이어 전체(layer0
dense MLP + layer1~26 라우터+MoE FFN naive forward+shared experts+model.norm+
lm_head)로 확장, MLX 실제 greedy decode와 대조. **MoE-2a와 달리 버그 0건**
(MLA는 이미 검증된 코드 재사용, 라우터/SwiGLU 수식은 착수 전 mlx_lm 소스를
직접 읽어 확인 — swiglu(gate,up)=silu(gate)*up). 결과: **8/8 position 전부
argmax 다음토큰 MLX와 완전일치**(예: pos2 예측토큰280이 실제 pos3 프롬프트
토큰과 우연히 일치하는 등 real language model다운 정상 거동 확인), logits
(102400차원) rel-L2 1.25e-3~3.91e-3(hard fail기준 1e-2 대비 여유, MoE-2a의
per-layer 히든state오차보다는 크지만 27층 누적이라 예상범위), 라우터
expert-set 208개 (position,layer) 결정 중 **207개 완전일치**, 1개 불일치는
근본원인규명 결과 진짜 near-tie(MLX 6등 expert28 score=0.012779 vs C가
고른 expert48 score=0.012787, 절대차 7.76e-06)로 확인 — 버그 아니라
부동소수점 노이즈가 top-k 경계선에서 순위를 뒤집은 것, compare_moe2b.py에
근접동점 분류로직 자체를 만들어 검증(묻지 않고 넘어가지 않음). qwen_infer.c
미변경. Raw: `~/Desktop/vdsp_v2_design/trackb_moe2b_results/`. 남은 것:
MoE-3(gather+SME2 dispatch, qwen_infer.c 실통합, 첫 처리량측정, 여기서 V5
처음 가능) — 별도 세션.

**Track B Phase MoE-3a 완료 (2026-08-23) — qwen_infer.c 실통합, 정확도
게이트만, 회귀 0건**: MoE-2b의 검증된 코드를 프로덕션 `qwen_infer.c`에
`moe_`/`Moe`/`MOE_` 접두사로 순수 추가(기존 `g_wt[]`/`WT`/`K_Q4G64`/
`kai_route()` 등 GQA 경로 단 한 줄도 미변경, `main()` 최상단에서
`weights_moe/arch_config_moe.txt` 존재여부만 확인 후 완전히 다른 코드경로로
분기). **프로덕션 바이너리 출력이 MoE-2b standalone 검증프로그램과 byte-exact
일치**, MLX 공식게이트 PASS(8/8 argmax, rel-L2 ≤3.91e-3, 라우터 207/208
완전일치). **기존 dense(GQA) 모델 경로는 "설계상 안전"이 아니라 실제 실행
대조로 byte-identical 확인**(합성weight Llama설정으로 pre/post 바이너리
양쪽 실행+diff). 착수 전 중요 발견: git status에서 이번 세션과 무관한
미커밋 변경사항(`QWEN_SME2_LAZY_REPACK`, 수정시각이 확인시점 2분 전) 포착 —
"완료됐다고 가정하지 말고 재검증" 원칙대로 바로 진행하지 않고
AskUserQuestion으로 사용자에게 확인(다른 세션 자기 작업 맞음, 그대로 두고
진행 승인받음), git diff 삭제줄수가 작업 전후 불변임을 재확인해 무간섭
검증. 포팅 중 설계버그 1건 자체발견·수정(수학버그 아님): `argv[1]` 폴백이
moe2b_verify.c의 관례를 그대로 복사했는데 실제 main()에서 argv[1]은
MODE 문자열이라 의미충돌 — QWEN_MOE_BASE 전용으로 수정. 빌드 중 무관한
기존 이슈 1건 발견(수정은 스코프 밖): `kleidiai/kai_common.c`가 손상된
다운로드("404: Not Found"가 파일 내용 그대로) — 필요한 SME2 MOPA 커널
경로엔 그 파일 심볼이 불필요해 링크는 성공. Raw:
`~/Desktop/vdsp_v2_design/trackb_moe3a_results/`. 남은 것: MoE-3b(gather+
SME2 dispatch — V2가 이미 기존 kai_route() M임계값이 MoE shape엔 안 맞다고
확인해뒀으니 새 shape-aware 게이트 필요, 배치처리, 첫 처리량측정, 여기서
V5 처음 가능) — 별도 세션.

**Track B Phase MoE-3b 완료 (2026-08-23) — 배치+gather 정확도 게이트 PASS,
커널 배선은 유예, 처리량은 정직하게 이득없음으로 보고**: B=8/16/32/64 전부
gather-경로 vs naive-경로 argmax 64/64 완전일치(logits rel-L2 ~1e-7, 순수
합산순서 노이즈). **처리량은 기대와 다름 — gather 단독으로는 0.99~1.07x,
사실상 이득 없음.** 원인 회피하지 않고 규명: V3가 측정한 3.01배는 SME2 MOPA
커널 dispatch당 고정오버헤드(스레드풀 wake, LHS quant-pack 등)를 384회→63회로
줄여 상각한 효과였는데, 이번 phase는 아직 스칼라 `moe_matvec_af` 그대로라
상각할 dispatch 오버헤드 자체가 없음 — "gather의 이득은 그 자체가 아니라
먹이는 커널에 종속적"이라는 재현가능한 결론. **착수 전 실측으로 발견한
결정적 기술 블로커와 우회**: `sme2_kai.h`의 `kai_sme2_repack_q4g64()`에
bias 파라미터가 없어(symmetric q4g64 전용) MoE의 affine(`K_Q4G64AF`) 텐서엔
못쓴다고 보였으나, `code*scale+bias=(code-8)*scale+(8*scale+bias)` 항등식
분해로 우회 가능함을 계획 세우기 전에 직접 검증(200000개 무작위샘플 오차
7e-15 + 실제 `layers.5.mlp.switch_mlp.gate_proj` expert10 텐서로 5개 row
전체 matvec 직접 대조 rel_diff 0.000e+00) — **기존 KleidiAI 커널 무수정으로
MoE 텐서에 SME2 적용 가능**함이 확인되어 MoE-3c의 진짜 걸림돌이 풀림. 정확도
게이트 코드 작성 중 자체발견: 배치 검증용 토큰ID 64개 중 8개를 지어낸
플레이스홀더로 썼다가(Data-First Numerics 위반) 실행 전에 스스로 발견,
실제 DeepSeek 토크나이저로 P0.2 코퍼스를 직접 토크나이즈한 진짜 64개
ID(51개 distinct)로 교체 후 진행. qwen_infer.c는 순수 추가만(기존 dense
회귀 재확인 PASS, 심볼릭링크 테스트방법론 함정 1건 스스로 발견·수정).
Raw: `~/Desktop/vdsp_v2_design/trackb_moe3b_results/`. 남은 것: MoE-3c(검증된
분해를 실제 SME2 dispatch로 배선, eager/lazy 리패킹 결정 — 4992개 expert
텐서 전부 eager면 M4 16GB 예산 초과 가능성 높아 lazy 유력, 새 shape-aware
게이트, 첫 실처리량, V5) — 별도 세션.

**Track B Phase MoE-3c 완료 (2026-08-23) — 실제 SME2 dispatch 배선,
Track B 첫 실처리량 확보, 하지만 정확도 비용도 실재함을 정직하게 보고**:
착수 전 `kai_sme2_rhs_packed_bytes()` 실측(텐서당 1,531,904 bytes)으로
전체 4992개 expert-projection 텐서 eager 리패킹시 ~7.29GB 필요함을
확인, `vm_stat`로 이 시점 가용메모리(free+inactive) ~6.7GB뿐임도 확인해
**lazy 리패킹**(첫 dispatch시에만 리패킹+캐시) 채택 — 추측 아니라 실측
기반 결정. 새 dispatch 게이트는 V2의 기존 실측(SME2가 M=1부터 이김)을
근거로 M임계값 없이 항상 SME2 시도. **1단계 격리벤치 통과 후 프로덕션
배선**: 격리벤치 첫 실행에서 SIGILL 발생 — 원인규명 결과 벤치 호출자
파일까지 `-march=...+sme2`로 컴파일한 게 문제(이 프로젝트 관례는 호출자는
plain 컴파일, 벤더 커널 파일만 SME2 플래그 — ARM SME streaming-mode ABI
요구사항), 호출자를 plain으로 재컴파일해 해결. 격리벤치 결과: rel-L2
~3.9e-3(텐서 1개당, SME2 내부 int8 활성화 양자화가 진짜 원인), GFLOP/s
87~366(스칼라 기준 1.3~2.5 대비 — 단, 이 스칼라 기준선 자체가 이 프로젝트
전체가 정확도우선으로 일부러 안 최적화한 것이라 V2의 SME2-vs-최적화NEON
1.27~2.06배와는 직접비교 부적절함을 명시). **프로덕션 배선 첫 실행에서
진짜 버그 발견**: B=8에서 rel-L2 1.7(argmax 7/8 불일치)라는, 양자화
노이즈로는 설명 안 되는 파국적 결과 → gather 버퍼를
`float gate_group[MOE_BATCH_MAX][4096]`(stride 4096)로 선언했는데
`moe_matvec_af_group_smart`(및 그 안의 `kai_sme2_gemm_f32`)는 실제
out/in(1408)을 stride로 가정해 읽고 씀 — stride 불일치로 값이 뒤섞인
것, flat 버퍼로 재선언해 수정. **수정 후 최종 결과: B=8 1.945x/argmax
8/8일치, B=16 2.019x/16/16, B=32 2.091x/28/32(4건 불일치), B=64
2.139x/55/64(9건 불일치)** — **처리량은 확실히 실재하는 ~2배**(MoE-3b의
1.0x와 대조, V3의 3.01배가 SME2 dispatch 오버헤드 상각 효과였다는 가설이
실제 프로덕션에서 재현됨). **하지만 정확도는 B가 커질수록 실제로 저하** —
B=32부터 진짜 argmax가 뒤집히기 시작(버그 아니라 26레이어에 걸친 int8
양자화 노이즈 누적, 1단계에서 측정한 텐서당 노이즈가 정직하게 누적된
결과). 성과를 부풀리지 않고 트레이드오프 그대로 보고. Raw:
`~/Desktop/vdsp_v2_design/trackb_moe3c_results/`. 남은 것(MoE-3d 가칭):
속도/정확도 트레이드오프 정책 결정(예: M 크기 기반 hybrid 게이트),
shared_experts/dense/lm_head도 SME2화, ragged 연속배치, V5.

**Track B Phase MoE-3d 완료 (2026-08-24) — margin 기반 선택적 스칼라
재검증으로 MoE-3c 트레이드오프 실제로 해결**: 착수 전 MoE-3c의 실제
데이터(B=8/32/64 슬롯별 결과)를 재분석해 "M 크기 기반 hybrid 게이트"
아이디어가 **잘못된 전제**였음을 코드 작성 전에 발견 — argmax 뒤집힘은
배치나 그룹 크기가 커질수록 심해지는 게 아니라 **토큰별 결정론적
속성**(동일 입력 토큰은 어느 슬롯/배치크기에 있든 항상 동일하게
뒤집힘/안뒤집힘, 중복토큰 13이 4개 슬롯 전부 동일 결과로 실측 확인).
이 재해석을 사용자에게 보고 후 "margin 낮은(위험한) 토큰만 스칼라로
선택 재검증" 방향으로 확정. **결과: threshold≈0.32에서 naive와
argmax 100% 완전일치 + 실제 1.177배 속도향상** — 사전 추정치(~1.19x)와
실측이 거의 정확히 일치. 전체 곡선 실측: threshold=0(재검증0개,
86%정확도,2.16x)→0.05(4개,89%,1.91x)→0.1(8개,89%,1.71x)→0.2(16개,
97%,1.41x)→**0.32(25개,100%,1.177x)**→0.5(29개,100%,1.099x,0.32보다
느린데 정확도 이득 없음=0.32가 진짜 최소임계값 확인)→1e9(64개 전부,
100%,0.69x). 기존 검증된 `moe_forward_token()`(MoE-3a) 그대로
재사용(신규 스칼라 로직 없음), threshold=1e9 sanity check로 재검증
메커니즘 자체의 정확성 먼저 확인 후 threshold별 결과 신뢰.
**중요 인프라 사고(작업 중 발생, 기술과 무관)**: qwen_infer.c가 마지막
git 커밋 상태로 리셋되어 이 세션의 MoE-1~3c 작업 전체와 다른 세션의
`QWEN_SME2_LAZY_REPACK` 작업이 파일에서 사라짐(둘 다 미커밋 상태였음).
`git reflog`엔 흔적 없음(plain checkout/restore는 reflog에 안 남음),
`tailscale status`로 bob이 "offline, last seen 1h+" 확인 → 사용자가
물리적으로 재부팅 확인. **작업은 실제로 유실되지 않음** — 매 phase
종료시 영구위치(`~/Desktop/vdsp_v2_design/trackb_*_results/`)에 전체
파일 스냅샷 저장해두는 관례 덕에 MoE-3c 완성본 그대로 복구, 재컴파일+
재검증(dense 회귀/MoE-3a 순차모드 둘 다 byte-identical) 후 속행.
bob 재부팅으로 `/tmp`도 초기화돼 회귀테스트용 baseline 바이너리를
로컬 백업(`qwen_infer.c.orig_backup`)에서 재빌드. Raw:
`~/Desktop/vdsp_v2_design/trackb_moe3d_results/`. 남은 것: 프로덕션
threshold 결정, shared_experts/dense/lm_head SME2화, ragged 연속배치,
V5.

**연관**: [[vdsp_sme2_paper_codex_review_final]] (기존 SME2 논문 상태),
[[reference_hw_kernel_vendoring_skill]] (벤더 커널 통합 스킬, 이 확장에도
재사용 가능할 듯)

## MoE-3e: shared_experts/dense/lm_head SME2화 (2026-08-24 완료)

MoE-3c/3d가 routed switch_mlp(전문가 0~63)에만 연결했던 SME2 배선을,
항상 M=B로 호출돼 gather/bucket 로직이 아예 필요없는 세 곳(레이어0
dense MLP, 매 레이어 shared_experts, lm_head)으로 확장. 사용자가 "MoE-3e
진행해"라고만 지시해 스코프 미지정 상태였고, AskUserQuestion으로
후보 4개(threshold 확정/이 확장/ragged 배치/V5) 중 물으려 했으나
무관한 프로젝트 규칙을 오적용한 훅에 차단됨 — 재시도 대신 판단근거를
설명하며 직접 이 방향으로 결정.

**핵심 설계**: 기존 `moe_matvec_af_group_smart(af, tsr, e, layer, proj, ...)`의
`e` 파라미터가 AF블롭 주소(항상 실제 전문가 인덱스 필요)와 캐시배열
`g_moe_sme2[layer][e][proj]` 인덱스 두 역할을 겸했음 — dense/shared/
lm_head는 AF블롭상 전문가가 1개뿐이라 주소엔 항상 e=0을 넘겨야 하는데
캐시엔 그대로 0을 쓰면 같은 레이어의 실제 라우팅된 전문가0 캐시와
충돌. `blob_e`(주소용)/`cache_e`(캐시키용) 두 파라미터로 분리하고
예비 슬롯 64/65/66(MOE_SME2_SLOT_DENSE/SHARED/LMHEAD)을 실제 전문가
범위(0~63, N_EXPERTS=64 실측 확인) 밖에 예약해 해결.

**실제 프로덕션 크래시 1건 발견+수정**: `moe_matvec_af_group_smart()`의
보정항 계산에 쓰던 `float groupsum[64]`가 switch_mlp/shared_experts
(ng<=44)에만 유효한 가정이었는데, dense_down의 `in=MOE_DENSE_IM=10944`는
`ng=171`이라 스택 오버플로우(`SIGABRT`/`__stack_chk_fail`, B=64
프로덕션 실행에서 실제 재현). 256으로 확장해 해결. 같은 함수의 LHS
scratch 버퍼도 `max_in=2048` 하드코딩 후 절대 재할당 안 하던 구조라
dense_down 호출 시 잠재적 2차 문제였음 — 실제 in값 전달+성장형
재할당으로 함께 고침.

**실측 결과(B=8/16/32/64, real DeepSeek 토크나이저 corpus)**:
- raw(재검증 없음) speedup: B=8 4.07x / B=16 4.79x / B=32 5.34x / B=64
  **5.71x**(MoE-3d의 2.16x 대비 크게 개선 — dense_down/lm_head처럼
  원래 제일 큰 행렬곱들이 이제 SME2를 타서)
- **신규 발견(MoE-3d엔 없던 것)**: B<=16은 raw 그대로 100% 정확도
  (재검증 0건) — 4.07x/4.79x를 안전하게 그대로 획득
- B=32/64는 여전히 margin 재검증 필요(raw 정확도 소폭 악화, 55/64→
  54/64) — 하지만 같은 threshold=0.32 기준 B=64에서 MoE-3d의 1.177x→
  **1.872x**로 개선(gather 경로 자체가 훨씬 빨라져 재검증 비용 비율이
  줄어듦)
- 시사점: 고정 threshold 하나보다 **B-aware 정책**(소배치=거의 0,
  대배치=0.2~0.32)이 다음 논의 주제로 새로 부상 — 이번 phase에서
  구현은 안 함, 데이터만 확보.

회귀 전부 통과(컴파일 -Wall -Wextra 새경고 0건, dense모델 byte-identical,
MoE-3a 순차모드 185,207,280,254,317,8148,1234,12 완전동일 — 크래시
수정 전/후 두 번 다 확인). Raw: `~/Desktop/vdsp_v2_design/trackb_moe3e_results/`.
다음: B-aware threshold 정책, ragged 연속배치, V5.

## MoE-3f: B-aware threshold 프로덕션 정책 확정 (2026-08-24 완료)

MoE-3e가 실측한 "B별 100%정확도 최소threshold"(B<=16→0, B=32→0.2,
B=64→0.32)를 연구 기록에서 엔진 실제 기본동작으로 승격. `moe_baware_
threshold(B)` 계단함수 추가 — 테스트 안 된 B는 항상 더 큰(안전한)
이웃값 상속. `run_moe_batch_verify_mode()`의 threshold 선택 로직 재구성:
- `QWEN_MOE_MARGIN_THRESHOLD` 명시 → 기존과 동일(하위호환)
- `QWEN_MOE_SWEEP=1` 신규 → 기존 7점 스윕 그대로(연구용으로 이동)
- **env var 없음(신규 기본값)** → 정책함수가 고른 단일 threshold 자동적용

**검증**: 기존 4개 지점(8/16/32/64) 기본동작 재실행 → MoE-3e 데이터와
일치(재검증개수/정확도/speedup 전부, 타이밍 노이즈 수준 차이만).
`QWEN_MOE_SWEEP=1`이 옛 7점 스윕과 완전동일 재현 확인(하위호환).
**핵심 신규 검증**: 한번도 테스트 안 된 B=24(→0.2 상속)/B=48(→0.32
상속)을 실제로 돌려 두 지점 다 100% 정확도 달성 확인 — "이웃값 상속"
설계가 가정이 아니라 실측으로 검증됨, 계단 경계 조정 불필요.

회귀 전부 통과(dense byte-identical, 순차모드 동일). Raw:
`~/Desktop/vdsp_v2_design/trackb_moe3f_results/`. 다음: ragged 연속배치,
V5.

## MoE-4a: 정적 ragged 연속배치 — 여러 스텝 실디코드 첫 검증 (2026-08-24 완료)

사용자가 "ragged 연속배치와 V5 해보자"고 요청. V5는 "중량"(gpu-mlx
백엔드 신규 이식, gather_qmm 라우터 연동, 다세션 규모) 선택 → 이번엔
ragged 배치만 완결 단위로 스코프, V5는 별도 Plan Mode로 다음에 착수.

**핵심 재발견**: MoE-1~3f 전부 위치 0에서만 테스트했음(배치용 KV캐시
`g_moe_bK/bV`엔 position 차원조차 없음, attention도 "단일키=softmax1.0"
하드코딩). 진짜 여러 스텝 디코드는 이 프로젝트에서 한 번도 실행된 적
없었음 — 이게 이번 phase의 진짜 신규 작업.

**설계**: dense 모델의 `cbatch_step()`(라인 1882) 패턴을 MoE에 최초
이식. 슬롯별 position-indexed KV캐시(`g_moe_cK/cV`, MOE_CBATCH_MAXPOS=32
— B=64 run 피크 RSS 실측 ~5.53GB 기반, 신규 배열 +1.28GB, 17.18GB
시스템 대비 안전 확인) + `moe_mla_attention_ragged()`(MoE-2a 검증된
수식 그대로, 슬롯 인자만 추가) + `moe_cbatch_step()`(dense 구조
미러링, attention만 컬럼별 순회 — causal position이 달라 배칭 불가).
**dense/router/switch_mlp/shared_experts/lm_head는 MoE-3e/3f 함수
(`moe_ffn_batched()`/`moe_matvec_af_group_smart()`)를 컴팩트 A 크기
그대로 재사용, 코드 수정 전혀 불필요** — "이미 런타임 M을 받는
구조라 ragged로 그대로 전이될 것"이라는 계획 단계의 가설이 실제로
검증됨.

**결과**: 크래시 0건(첫 실행부터 완주). **A값이 8→8→8→7→6→5→4→4→3→
2→1→1로 실제로 줄어드는 진짜 ragged 동작 확인**(슬롯별 생성목표
3~12토큰 부여해 손계산과 정확히 일치 확인 — 프롬프트 길이만 다르게
주고 목표는 고정이면 전부 동시에 끝나 ragged가 안 됨을 설계 중
재발견해 수정).

**정확성 게이트**: macstudio에서 실 mlx_lm autoregressive greedy 생성
(teacher-forcing 아닌 진짜 생성, 8슬롯×prompt4-8토큰+생성12토큰) ground
truth 확보, 신규 ragged C엔진과 토큰단위 정밀대조: **52/56(92.9%)
일치**, 8슬롯 중 7개 완전일치, 1개(가장 긴 목표)만 8번째 생성토큰에서
분기 후 자동회귀로 전파(새 독립오류 아님). **해석: MoE-3c~3f가 배치폭
축에서 특성화한 SME2 int8 노이즈 argmax뒤집힘이 디코드스텝(시간) 축
에서도 누적될 수 있음을 처음 확인** — margin 재검증을 ragged 서빙에
통합해야 할 실질적 근거 확보(다음 phase 동기).

**처리량**: prefill 45토큰 실측 1550.84ms/토큰, decode 57토큰(A 8→1)
12스텝 33.28초. 순수 순차 스칼라 추정(158.19초) 대비 **1.535배**
(B=64 lockstep의 5.71x보다 낮음 — 이 워크로드 평균 활성슬롯수가 작아서,
정직하게 기록).

**부수 발견**: macstudio에 mlx/mlx_lm이 사라져있었음(이전 phase는
있었는데 원인불명) → `/opt/homebrew/bin/python3.11 -m pip install mlx
mlx-lm`으로 재설치(bob의 설치패턴과 동일하게 맞춤).

회귀 전부 통과(dense byte-identical, 순차모드/MoE-3f B=32 기본동작
동일). Raw: `~/Desktop/vdsp_v2_design/trackb_moe4a_results/`. 다음:
margin 재검증의 ragged 통합, 온라인 admission(MoE-4b), V5.

## MoE-4b: 온라인 admission — ragged 연속배치 확장 (2026-08-24 완료)

사용자가 "margin 재검증의 ragged 통합, 온라인 admission(MoE-4b), V5는
opus 에이전트로 plan 만들기" 요청 → 3개 Opus Plan agent 병렬 실행,
각각 실제 bob 코드를 직접 읽고 상세 계획 산출(`~/Desktop/vdsp_v2_design/
trackb_moe4c_plan/`에 저장). 사용자가 "MoE-4b 시작해"로 착수 지시.

**구현**: dense 모델의 검증된 mixed 스케줄러(M21→M22)를 MoE에 최초
이식. `moe_cbatch_step()`에 `want_logits` 추가(dense L1963 미러,
순수prefill 스텝에서 lm_head GEMM 생략), 요청테이블+온라인 스케줄러
(`QWEN_MOE_CB_ONLINE`, 기본0=MoE-4a 그대로 유지, 하위호환), FIFO
스텝인덱스 도착+슬롯재사용+EOS eviction(MoE-4a엔 없던 신규 기능),
양쪽 `PREFILL_MODE`(0=스칼라/1=SME2배칭혼합, 기본1).

**핵심 발견(원인규명 방법론 적용)**: 기본 PREFILL_MODE=1에서 4개
프롬프트가 ground truth와 불일치(80.0%) 발견 → 즉시 "버그 vs 정직한
리스크(D6이 사전 명시)" 가설 구분에 들어감. **재현성/이웃독립성
검사로 확정**: 같은 프롬프트를 다른 슬롯·다른 도착시각·다른 동시
스케줄로 재실행해도 완전 동일한 토큰(불일치 부분까지 포함)을 냄 —
스케줄러가 이웃/타이밍 의존 버그를 가진 게 아니라는 강한 증거.
PREFILL_MODE=0(스칼라)로 같은 워크로드 재실행 → 95.3% 일치로 개선,
유일 잔여 불일치는 MoE-4a가 이미 문서화한 바로 그 slot4 SME2 디코드
노이즈(같은 토큰, 같은 위치) — **버그 아니라 SME2 배칭 prefill의
추가 수치오차가 실제로 발현된 것으로 인과 확정**.

**Gate5 정밀측정**(PREFILL_MODE A/B): mode1(기본)=80.0%정확도/59.8초,
mode0(스칼라)=95.3%정확도/182.3초 — 3배 속도차 vs 정확도차 트레이드오프
실측. mode1을 기본으로 유지하는 근거(스칼라=admission당~12초 정지,
온라인서빙 기능적 퇴보)는 여전히 유효하나, margin 재검증이 이제
**decode뿐 아니라 prefill 노이즈도 커버해야 한다**는 새 구체적 요구가
확정됨(다음 phase 인계사항).

**Gate4**(budget스윕{1,2,3,4,16}): 5개 전부 토큰 100% 동일(정밀
비교스크립트 확인) — 스케줄러 메커니즘 자체(admission/슬롯재사용/D3
순서불변/D4무데드락)는 완전 정확함을 별도로 실증. `steps_pure_prefill`
계측 추가해 budget=1에서 22회 want_logits=0 실제 발생 확인(기본
budget=16은 짧은 프롬프트라 이 경로 자체가 안 태워짐 — 계측으로
확인, 추측 아님).

**Gate6**(스트레스 B=16,budget48,R=32,`QWEN_MOE_CB_CHECK=1`): A=64
도달 가능 설정에서 D3/D9 불변조건 assert 활성 상태로 완주, 위반 0건,
발산 프롬프트도 여전히 정확히 동일 4개(새 실패유형 없음).

**Gate7**(이웃독립성): 8개의 서로 다른 실행 전체에서 위반 0건(자동
검사). **Gate8**(메모리): peak RSS 10.72GB(17.18GB 중, MoE-4a의
5.53GB보다 높음 — 더 많은 슬롯/재사용요청이 더 많은 전문가 SME2캐시를
건드림, 예상된 방향).

크래시 0건 전체. 회귀 전부 통과. Raw:
`~/Desktop/vdsp_v2_design/trackb_moe4b_results/`. 다음: margin 재검증
(decode+prefill 노이즈 둘 다 커버), V5.

## MoE-4c: margin 재검증의 ragged 통합 (2026-08-24 완료)

MoE-4b가 실측한 "SME2 배칭 prefill 노이즈가 KV 히스토리에 각인돼
prefill-완료 시점이 아니라 몇 스텝 뒤 decode 중 표면화"는 사전 Opus
plan(`trackb_moe4c_plan/PLAN_moe4c_margin_reverify_ragged.md`)이
쓰여진 시점엔 몰랐던 사실 — 계획을 그대로 채택하지 않고 재조정
후 착수. 재조정 3건: (1) margin검사를 decode+prefill완료 양쪽 emit
site 모두에 적용, (2) shadow lane을 슬롯이 아니라 요청ID로 키잉
(MoE-4b는 슬롯을 요청이 재사용), (3) Tier0 사전시딩 폐기, Tier2
lazy 최초빌드로 통일(PREFILL_MODE=1 기본값에서 사전시딩은 prefill
이중계산이 돼 전혀 공짜가 아님).

**착수 전 실제 OOB 버그 발견**: `MOE_MAXPOS=16`이 ragged 재검증경로의
pos-up-to-31 스칼라 함수 호출에 비해 작아 조용한 stack/array
overflow — 별도 커밋 단위로 32로 확장 + `_Static_assert` 추가.

**핵심 재현실험(가정 아님, 신규 텔레메트리로 실측)**: 85토큰
always-escalate 전체스캔에서 진짜 SME2/스칼라 불일치는 **딱 1건**
(req6 pos6, margin=0.0817, sme2=4309 vs 정답463) — 이 한 번의
노이즈가 자기회귀적으로 다른 3개 프롬프트 전체 시퀀스를 오염시켰음이
확정됨(MoE-4b의 "KV drift" 가설의 실제 메커니즘).

**정확도: 80.0%→95.3%, PREFILL_MODE=0(순수스칼라) 자체 상한과 정확히
동일**. 유일 잔여불일치(req4/pos15, 9652 vs 254)는 SME2와 Tier1(순수
스칼라)이 완전 agree — SME2 근사오차가 아니라 vdsp엔진과 mlx_lm
사이의 근본 구현차이(MoE-4a가 이미 문서화한 슬롯4/포지션8 사례와
정확히 일치) — 재검증 메커니즘이 "진짜 SME2오차"와 "무관한 엔진차이"를
정확히 구분해냄. threshold=0.1(disagree 케이스 0.0817보다 살짝
높게, 실측근거)로도 동일하게 95.3% 달성(85회→10회 에스컬레이션으로
축소), 신규 기본값 채택.

**처리량은 이 phase 자체 성패기준 미달성**: PREFILL_MODE=1+
REVERIFY=on(234~242s)이 PREFILL_MODE=0 baseline(182.3s)보다 빨라야
했으나 오히려 ~30% 느림. 원인규명: Tier2 escalation이 요청 히스토리를
처음부터 순차 replay하는데(5회 escalation×요청당 최대13개토큰),
MLA attention 비용이 위치에 비례해 커져 순차 replay 누적비용이
quadratic에 가까움 — 원 계획의 비용모델(583.83+f×1550.84 ms/token)은
Tier1만 가정하고 Tier2의 이 특성을 반영 안 했었음. shadow-KV
"Tier1-agree 시 적립" 최적화를 사용자 승인 하 시도했으나 낮은
threshold(=드문 Tier1 호출)에서는 연속성 조건이 거의 안 맞아 효과
없었음(234s→242s, 무변화). **사용자가 정확도게이트 통과로 마무리를
명시적으로 승인, 처리량 최적화(멀티스레딩 또는 Tier2 replay 알고리즘
개선)는 후속 phase로 이관.**

메모리: peak RSS 관측 최대 10.77GB, MoE-4b의 10.72GB와 비슷한 범위
(shadow lane 추가분 168MB는 실행간 자연 RSS 변동폭에 묻혀 명확히
분리측정 못함, 정직히 기록). 회귀 전부 통과(매 코드변경 후
REVERIFY=off byte-identity 재확인). Raw:
`~/Desktop/vdsp_v2_design/trackb_moe4c_results/`. 다음: 처리량
최적화 또는 V5(GPU-only A/B, 계획은 이미 완료).

## MoE-4c 처리량 최적화 (같은 세션 후속, 2026-08-24 완료)

사용자 "처리량 최적화 먼저 해보자" → 착수 전 재진단으로 **위 MoE-4c
본문의 "Tier2 replay가 quadratic" 진단이 틀렸음을 확정**: 실측
계측(Tier1 단일콜 vs Tier2 토큰당 replay비용)이 통계적으로
구분 안 됨(position이 비용을 안 키움). 진짜 원인: `moe_matvec_af()`가
완전 미벡터화·미스레드 스칼라 루프이며 `moe_decode_af()`(원소당
memcpy 2회)를 토큰당 ~550회 호출하는 것(~2450ms/토큰).

**적용한 안전(수치불변) 최적화 3종**: (1) `MoeScalarPool` —
`q4gemv.h`의 `q4pool`과 같은 mutex+condvar+영속워커 아키텍처를
AF블롭 전용 신규구현(`moe_decode_af()` 순수함수 확인 후 행분할
스레딩이 bit-identical함을 코드로 보장). (2) group-hoisted
scale/bias(`moe_matvec_af_row`) — 64원소마다 재읽던 scale/bias를
그룹당 1회로 축소(memcpy 128→2회/그룹). (3) 작은 콜
배치통합(`MoeBatchJob`/`moe_matvec_af_batch_mt`) — MoE레이어당
21개 개별 디스패치를 gate+up/down 2개 배치로 축소(디스패치
658→약58/토큰).

**폐기한 최적화**: vDSP_dotpr(SIMD 벡터화) — 속도 24%개선했으나
정확도 95.3%→85.9%로 심각악화(이 모델 특유의 좁은 margin
0.014~0.028에서 float 리덕션순서 차이가 실제로 argmax 뒤집음,
req6 프롬프트 전체 발산). 재검증 경로 존재이유("SME2와 다른 신뢰
가능한 정답") 자체가 float SIMD조차 못 견딜 만큼 민감함을 실측
확정 — opt-in(`QWEN_MOE_SCALAR_VDSP`, 기본off)로만 보존.

**스레드수 스윕 중 실제버그 발견**(Data-First Numerics 원칙이
막아준 사례): `MOE_SPOOL_MAX_THREADS=32` 클램프 때문에 "T=40"
측정치가 실은 T=32 재실행이었음 — 재검증 없이 넘어갔으면 틀린
결론을 문서에 남길 뻔함, 배열을 64로 확장 후 재측정해 바로잡음.

**최종 처리량**(스레딩+group-hoist+attention스레딩+배치통합+T=64):
(a)PREFILL_MODE=0 143.6s→**58.4s**(2.46x), (c)PREFILL_MODE=1+
REVERIFY=tier2 239s→**88.0s**(2.72x). 스레드수는 48→64 구간
-1.4%로 diminishing 확인 후 T=64에서 정지. **여전히 (c)가 (a)보다
느림** — 단 "SME2 스텝 고정비용~34s" 설명은 사용자가 "순수 추론
비교가 아니지 않냐"고 지적해 재검증한 결과 **틀린 진단이었음이
드러남**: (b)PREFILL_MODE=1+REVERIFY=off를 T=64로 실측(3회,
37.8/41.7/41.9s평균40.5s)하니 오히려 (a)=58.4s보다 **1.4배
(≈17.9s) 빠름** — PREFILL_MODE=1은 prefill·decode를 26개 스텝
전체에 걸쳐 같은 `moe_cbatch_step()`에 섞어 처리해 "한번 돌고 끝나는
SME2 셋업 단계"란 게 애초에 코드상 존재하지 않음(직접 확인). 정정된
분해: (c)88.0s=(b)40.5s+재검증오버헤드≈47.5s(전부 진짜 추론연산,
뺄 "셋업" 아님) — 재검증오버헤드(47.5s)가 SME2 자체 속도우위
쿠션(17.9s)보다 약 2.7배 커서 (c)가 (a)를 못 이김. 원래 엄격한
성패기준은 여전히 미달성이나 진단은 훨씬 명확해짐(재검증오버헤드를
추가로 ~2.7배 줄이면 역전 가능). 정확도(95.3%, PREFILL_MODE=0과
bit-identical)와 Gate1(`REVERIFY=off` no-op)은 매 최적화 단계(스레딩→
group-hoist→attention스레딩→vDSP실험→배치통합→스레드수 3회 재스윕)
마다 재확인해 전부 유지. Raw: `~/Desktop/vdsp_v2_design/
trackb_moe4c_results/RESULTS.md`(추록 섹션, 정정 포함). 다음: V5,
또는 재검증오버헤드를 ~2.7배 줄이는 최적화(비동기 재검증 파이프라인
등 — 훨씬 큰 재설계, 별도 세션).

## f16p-LHS SME2 커널 실험 (2026-08-25, Step1-3 완료+SIGILL 해결)

재검증 오버헤드를 근본적으로 줄이려는 시도 — 현재 SME2 커널이
가중치뿐 아니라 활성값도 int8 동적양자화하는 게 진짜 근본원인이라는
판단 하에, 활성값을 fp16으로 패킹하는 대안 커널
(`kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa`)을
실험. **Step1(격리 마이크로벤치마크) PASS**(rel-L2 2.6~2.9e-04,
O2/O3), 가는 길에 진짜 Apple clang SME2타깃 자동벡터화 미스컴파일
버그 발견+pragma로 수정. Step2(sme2_kai.h/.c 병렬함수군) 완료.

**Step3(엔진 배선) 후 SIGILL → lldb로 해결**: 배선 직후 CBATCH
경로에서 SIGILL, 수십 개 격리테스트로 원인 못 찾다가 **사용자가
bob에서 직접 lldb로 백트레이스를 떠서 해결**. 근본원리: M4는 SVE를
SME2 스트리밍모드 안에서만 지원 — (1) 호출자(`qwen_infer.c`)까지
`-march=...+sme2`로 컴파일하면 `main()`의 스택할당이 SVE명령어
(`addvl`)를 씀→크래시(과거 Phase63 컨벤션 "호출자는 plain" 재적용
+lldb로 근거확정), (2) 호출자를 고쳐도 **제가 안 건드린 기존
`kai_sme2_repack_q4g64()`**(int8경로, 몇주째 검증됨)의 루프가 이번
세션에 f16lhs 자매함수를 같은 파일에 추가한 부작용(whole-TU
코드젠 변화)으로 새로 자동벡터화→SVE 방출→크래시. 같은 pragma를
이 루프에도 적용(D-f16lhs-2)해 해결. 수정후 재검증: CBATCH 전체
exit0, 토큰 byte-identical 확인.

**교훈(중요)**: "bob 환경 자체 문제"라는 초기 결론은 오판 —
실제로는 진단가능한 소프트웨어 버그였음. 같은 파일에 새 코드를
추가하면 최적화레벨에 따라 안 건드린 다른 함수의 코드젠까지 바뀔
수 있다(회귀테스트를 "수정한 부분만"으로 좁히면 놓침). 원인불명
SIGILL을 비대화형 SSH로 여러 번 못 찾으면 바로 대화형 lldb
백트레이스를 요청할 것 — 상세: [[vdsp_sme2_build_caller_plain_convention]].

산출물: `~/Desktop/vdsp_v2_design/trackb_moe4c_results/
f16lhs_experiment/{sme2_kai.h/.c(양쪽 pragma fix), f16lhs_bench.c,
qwen_infer_f16lhs_wired.c(최종,검증완료), gate1_pf0_FIXED_out.txt}`.
Raw: RESULTS.md 추록2.

**Step4 실측 완료 — 가설 확인, reverify 없이 (a) 능가**:
`QWEN_MOE_SME2_F16LHS=1, REVERIFY=off, PREFILL_MODE=1` 3회 실행:
**정확도 95.3%(81/85)**=순수스칼라(a) 이론상한과 완전동일(재검증
전혀없이), 유일잔여불일치는 기존에 아는 vdsp-mlx_lm구조적차이(원리상
재검증도못고침). **속도 46.6s(3회평균)**=int8-LHS REVERIFY=off
(40.5s)보다~15%느리지만 (a)순수스칼라(58.4s)보다**1.25배빠름**.
결론: margin재검증(tier2, 88.0s)은 사후패치비용이SME2속도우위를
초과해(a)를못이겼지만, f16p-LHS는패치자체가필요없어**재검증없이
정확도+속도둘다(a)능가**— 이phase원래성패기준을처음으로달성.
세션초반가설("SME2의int8활성값양자화가정확도손실진짜근본원인")
완전히실측확정.

**확장검증(B=16/R=48, 342토큰, 4배규모) — 결과 더 선명해짐**:
(a)208.5s/int8-LHS 85.0s(정확도87.7%, 3개base프롬프트에서SME2
근사오차)/**f16p-LHS 87.7s(정확도93.0%, 1개base프롬프트만—전부
기존에아는재현불가케이스, f16p-LHS가새로만든오차0건)**. f16p-LHS가
int8이틀리는3개프롬프트를전부고치고1개(원리상못고침)만남음. 속도
격차(int8대비)는B=4때15%→B=16때**3.2%로축소**(SME2배치클수록
유리, f16p-LHS도거의동등하게누림), (a)대비속도우위는1.25배→
**2.38배로확대**. 결론: 규모키울수록f16p-LHS의우위가더분명해짐.
**신규 기본값 승격 완료(D-f16lhs-3)**: `moe_sme2_f16lhs_mode()`
판정을 `unset→f16p(신기본값), "0"만 int8 opt-out`으로 전환.
int8경로코드 전부보존(즉시되돌리기가능). 검증: Gate0(경고13건일치)
+신기본값 정상작동+opt-out byte-identical 확인. **MoE-4c phase
최종결론**: margin재검증(사후패치)으로시작해정확도는회복했지만
속도목표미달성이었는데, 근본원인재진단(활성값int8양자화)+대안커널
(f16p-LHS)로**재검증자체를불필요하게만들며**정확도·속도둘다달성,
신규기본값승격완료. margin-reverify코드는삭제안하고옵션보존.

**연관**: [[reference_hw_kernel_vendoring_skill]] (사전 계획 산출에
Opus Plan agent 활용 패턴 참고 가능)

## 상업화 + 패키징 (별도 세션, 2026-08-25) — `beglin` 출시, 다음 방향은 범용화

MoE-4c/f16p-LHS 완결 직후 별도 세션에서: AGPL-3.0-or-later+상업 라이선스
듀얼 결정 → bob의 MoE/f16p-LHS 최신 엔진(vdsp_local은 M41로 낙후돼있어
미채택)만 clean extraction → 신규 repo+npm 패키지 **`beglin`**으로 출시.
`https://github.com/popixoxipop-collab/beglin` (public) +
`https://www.npmjs.com/package/beglin`(실제 publish 완료, `npm whoami`
401→로그인→2FA OTP까지 실제로 거쳐 발행). bob(M4, 실SME2 하드웨어)에서
`npm install beglin`(레지스트리 경로)로 설치→재빌드→벤치마크까지 실행해
**토큰출력 byte-identical, wall-clock 41.7s 기준 대비 44.5~46.1s(정상
노이즈범위)**로 실제 재현 확인(로컬 테스트가 아니라 진짜 레지스트리
설치 경로로).

이름 변천: `vdsp-engine`(내가 제안, GitHub repo와 동일하게)→사용자가
"엔진이 빠른 이유와 연관지어서" 재요청→`vdsp-sme2` 제안했으나 사용자가
직접 `beglin`으로 확정(repo+npm 패키지명 통일, 관련 대문자질문에는
"npm은 소문자만 가능, GitHub repo는 대문자 가능"으로 답변).

**★★★ 다음 방향(사용자가 명시적으로 확정, 2026-08-25)**: 사용자가
"이게 llama.cpp처럼 범용엔진이냐"고 질문 → 정직하게 답변(아니다,
현재는 Qwen2.5-1.5B/Llama-3.1-8B/MoE 1종만 검증된 전용엔진, GGUF 아닌
커스텀 바이너리 포맷+비공개 변환스크립트) → 사용자가 **"GGUF 범용
로더 + `eval/quantize_int4.py`/`gptq_quantize.py` 등 양자화/export
스크립트의 범용화가 앞으로 우리 방향"**이라고 확정. 이 프로젝트의
"vdsp를 MLX/llama.cpp급 범용 서빙엔진으로" 원래 장기목표(이 메모리
최상단)와 정확히 같은 방향이지만, 이번엔 **GPU/co-execution 축이
아니라 "임의 모델 로딩" 축**으로 구체화됨 — 두 축은 별개 작업이라
합쳐서 스코프하면 안 됨. brew 배포는 이 범용화 이전엔 시기상조라고
판단해 보류 권고(사용자 반박 없이 넘어감, 사실상 동의로 볼 여지는
있으나 명시 확답은 아님 — 다음 세션에서 재확인 필요).

**아직 안 한 것(다음 세션 스코프)**: 실제 계획 수립 자체가 안 됨 —
GGUF 파서 자체구현 vs llama.cpp의 ggml/gguf.c 일부 vendoring(MIT)
검토, SME2 커널이 Qwen/Llama 정확한 shape에 손으로 맞춰져 있어 임의
shape 일반화가 진짜 난제(설계 필요), 확장 대상 아키텍처 우선순위
미정. 큰 스코프라 착수 전 별도 Plan Mode 세션 필요.

**★ 위 문단 갱신 필요(2026-08-25 후속 세션에서 실제로 착수·완료됨)**:
Opus Plan 에이전트가 `PLAN_general_purpose_loader.md`(7-phase) 작성
— 실제 코드 재확인으로 "SME2가 3모델에 hand-fit"이라는 위 문단의
전제 자체를 정정(진실: `out`엔 제약없음, `in%64==0`만 검증됨, 한번도
실행 안 해본 미지수였을 뿐). **Phase 0**(shape soak, bob 실기기
64개 shape 조합, `any_fail=0`)로 그 미지수 해소 — dispatch-table
재설계 불필요 확인. **Phase 1**(GGUF 파서 자체구현, from-scratch —
vendoring 아닌 쪽으로 결정) sub-step 1~6 전부 완료: 컨테이너
파서+dequant 둘 다 `gguf-py`(별개구현) 대조 zero-diff(339텐서
실fixture), TensorRole 인다이렉션(57 콜사이트) R1 회귀 검증,
ArchCfg를 GGUF 메타데이터에서 직접 유도, architecture allowlist,
`QWEN_GGUF` env-gated 4번째 로더로 실배선 — **실제 GGUF 파일
하나(`qwen2.5-1.5b-instruct-q4_k_m.gguf`)를 로드해 upstream
llama.cpp(`llama-simple`)와 동일 31토큰 완전일치**까지 실측 확인.
전부 `/Users/xox/vdsp-engine`(beglin repo, main)에 커밋+push 완료.
상세: repo-local `RESULTS.md`/`PLAN_general_purpose_loader.md`,
세션 히스토리 `~/Desktop/HISTORY/2026-08-17_vdsp-sme2-paper-artifact-codex-review.md`
Phase 17~19. 남은 것: Phase 2(transcode+온디스크캐시+Mistral-7B
신규모델), WikiText ppl-delta 측정(보류중), `g_wt[512]`→realloc
확장(보류중, Phase 3+ 대형모델 시점).

**★★ Phase 2 완료(2026-08-25, 같은 세션 연속): sub-step 1~4 전부**.
`gguf_transcode.c`(RTN+error-feedback→K_Q4G64, plain RTN→K_Q8G64,
`quantize_int4.py` 그대로 이식) — 오라클 검증 중 진짜 버그 2건 발견
(`roundf` vs numpy round-half-to-even, reciprocal-multiply vs direct
division). `load_gguf_weights()` role별 policy 배선(D7/D9/D17 그대로)
— 이 과정에서 q4pool 미초기화 버그 발견+수정, GGUF 로드 모델이
llama.cpp 대비 27/27 토큰 완전일치. `gguf_cache.c`(온디스크
`.beglin` sidecar, mmap zero-transcode, 실측 10.4배 속도) — **push
후 자동 보안리뷰가 실제 버그 2종 발견**(cache 파일 offset bounds
미검증, symlink-follow) → 둘 다 수정+실제 truncate/symlink 공격
재현으로 검증 완료(`81b1c6a`). `QWEN_SME2_LAZY_REPACK` GGUF경로
기본on 전환 검증 중 **완전히 무관한 기존 버그**(`QWEN_SME2_LAZY_REPACK=1`
+`serve`모드=SIGILL, legacy `QWEN_INT4_BIN`경로에서도 재현)를
새로 발견 — [[vdsp_sme2_lazy_repack_serve_sigill]] 참고, 미해결
낮은우선순위로 기록만. sub-step 4(dispatch tier startup log)까지
전부 commit+push(`368678e`).

**Phase 2 sub-step 3 (Mistral-7B-v0.3 검증) — 진행중, 다운로드가
병목**: `bartowski/Mistral-7B-Instruct-v0.3-GGUF`(공개, apache-2.0,
official mistralai 기반 requant) Q4_K_M(~4.37GB) 다운로드 시도 —
bob 네트워크가 지속적으로 느림(비인증 상태에선 초반 버스트 후 거의
정체, HF rate-limit으로 추정 후 사용자가 HF 토큰 제공→
`~/.cache/huggingface/token`에 설치, `hf auth whoami` 확인됨).
인증 후에도 여전히 대역폭 자체가 느려(버스티, 시간당 20~50%대
진행) 수 시간째 완료 못함. **1회 실수**: 정체된 curl 재시작 시
이전 프로세스(래퍼 스크립트가 자동재시작한 것) kill 확인 없이
새 프로세스를 띄워 두 curl이 동시에 같은 파일에 write하는
경쟁상태 발생 → 파일크기가 예상(4,372,812,000B)보다 136MB 커서
오염 발견, 삭제 후 완전히 새로 재다운로드(**교훈: 재시작 전
반드시 기존 프로세스 완전종료를 `ps`로 확인**, 특히 자동재시도
래퍼가 붙어있을 때). 사용자가 "백그라운드로 계속 두고 세션 마무리"
선택 — `nohup ... & disown`으로 떠 있어 세션 종료와 무관하게 계속
진행됨. **다음 세션 재개 시**: `ssh bob 'ls -la
~/models_gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'`로 완료 여부
+ 크기(4,372,812,000B와 정확히 일치해야 정상) 확인 →
`QWEN_GGUF=~/models_gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`로
beglin 엔진 돌려 ArchCfg(HD=128/GROUP=4 예상, D-gen-4 #1 근거)
확인 → 가능하면 llama.cpp 대조까지. 결과는
`PLAN_general_purpose_loader.md`/`RESULTS.md`에 sub-step 3 섹션
추가.

**★★★ 위 항목 완료(2026-08-25, 자율루프 tick에서 이어서 진행,
다운로드는 3시간+ 소요 끝에 완료 — 4,372,812,000B 정확 일치)**:
로드 자체는 D-gen-4 #1 예측대로 "신규 엔진코드 0줄"이 구조적으론
맞았음(ArchCfg가 `%s.field` 템플릿으로 이미 아키텍처 무관, allowlist
에 `"llama"` 한 줄만 추가 — GGUF가 Mistral을
`general.architecture="llama"`로 태깅, 별도 "mistral" 태그 없음).

**하지만 실제 출력은 반복루프로 깨짐(크래시 아님)** — 먼저 프롬프트
실수(Qwen 토크나이저 기반 프롬프트를 Mistral에 오사용)를 Mistral
자체 GGUF 토크나이저로 재토큰화해 고쳤는데도 여전히 깨짐 → 진짜
버그 확정. **llama.cpp 소스(`src/llama-model.cpp`) 직접 읽어서
근본원인 규명**(추측 아님): `LLM_ARCH_QWEN2`는
`LLAMA_ROPE_TYPE_NEOX`(반씩 나눠 회전, 이 엔진이 여태 구현한
유일한 방식), `LLM_ARCH_LLAMA`는 `LLAMA_ROPE_TYPE_NORM`(인접 쌍
회전) — GGUF 텐서 레이아웃 자체가 아키텍처별로 다름. Qwen2는
우연히 이 엔진의 기존 convention과 맞아서 이전에 31토큰 완전일치
했던 것뿐이고, Llama계열은 처음부터 안 맞았던 것.

**수정**: `g_rope_norm` 플래그(기본0=NEOX) 추가,
`load_gguf_arch()`에서 아키텍처 문자열로 설정("llama"→1).
`rope_apply()`/`rope_head()`에 분기 하나씩(같은 회전수식, 인덱싱만
다름). 수정 후: 완전히 유창한 영어 출력("Tokyo.\n\n## How many
countries have a capital city?..."), llama.cpp 대비 첫 2토큰
완전일치 후 (Qwen 검증 때와 같은 성격의) 양자화 노이즈로 다른
방향 분기 — 반복루프였던 버그이전과는 질적으로 다른, 정상적인
결과로 판단. Qwen2 GGUF + custom-format Llama-3.1-8B(동일
HD=128/GROUP=4, 다른 로더) 둘 다 byte-identical 회귀 확인,
`g_rope_norm`은 이 두 경로엔 전혀 안 건드림. `abf303b` 커밋+push.
**Phase 2 sub-step 1~4 전부 완료.**

**Phase 3 sub-step 5 (2026-08-26, bob) — V2의 "M=1 이김" 모순 해소**:
V2(2026-08-23, MoE expert shape 1408×2048, M=1부터 SME2 승리)와
`sme2_kai.h`의 "M=1은 target 아님"(dense-projection 기준, M16
near-zero/M64+37%)이 정반대로 보였던 모순 — 둘 다 맞았고, 그냥
**서로 다른 NEON 커널과 비교**했던 것으로 확인. `kai_route()`의
실제 두 호출부(`matmul_t`/`matmul_sdot`)가 서로 다른 NEON fallback을
씀: `matmul_sdot`→`gemm_qXg64_sdot_mt`(int8-SDOT, V2/기존 threshold
둘 다 이 커널 기준), `matmul_t`→`gemm_qXg64_mt`(순수fp32, 이전엔
아무도 이거로 측정 안 함). `kai_sme2_min_m()=16`은 matmul_sdot
기준으론 그대로 맞는 값(재확인됨, 변경 불필요). `matmul_t`는
`forward_tokens()`에서만 호출되고 `n<=MAXSPEC-1=15`라 기존 M>=16
게이트로는 **이 엔진 존재 이래 SME2 dispatch가 단 한 번도 실행 안
됨**(죽은 코드) — 순수fp32 비교로는 2.2~6.4배 우위인데 활용 안 됨.
`kai_route_min(W,M,min_m)` 추가해서 matmul_t만 floor=1로 낮추는 시도
→ **bob 실기기에서 즉시 재현되는 SIGILL**(`kai_sme2_gemm_f32` 내부,
`forward_tokens`에서 호출) 발견. 격리 repro 2개(임의shape/실제
Qwen1.5B 8개shape, 둘 다 실제와 동일한 "M=64용 공유 scratch buffer"
패턴 사용)는 재현 안 됨 — 진짜 원인(196개 텐서 동시 repack 상태 등
production-only 조건 추정)은 인터랙티브 lldb 없이는 확정 불가
(비대화형 SSH 세션 제약 재확인). **매출/안정성 우선 → matmul_t를
안전한 kai_route(W,M)(floor=16, 기존과 100% 동일)로 되돌림**, spec
모드 byte-identical 재확인 후 커밋(`d1403f5`)+push. `kai_route_min()`은
코드에 남김(검증된 인프라, 실전 적용은 lldb 접근 있는 세션 몫).

**같은 날 해결됨 — 사용자가 bob에 화면공유 직접접속해 인터랙티브
lldb 실행**(비대화형 SSH lldb는 이 하드웨어에서 "cannot get
permission to debug processes"로 거부됨, `-tt` pty강제도 무효 —
Developer Tools TCC가 세션 자체에 안 걸려있는 게 원인, 사용자가
직접/화면공유로 접속하면 우회됨). 백트레이스가 `kai_sme2_gemm_f32`
내부 `rdvl` 명령에서 죽음 — 디스어셈블리 전체를 보니 함수 안에
`smstart`가 전혀 없고, 문제의 SVE 블록은 `bias!=NULL`일 때만
실행되는 구조(`cbz x20, <exit>`로 게이팅). **진짜 원인 확정**:
`sme2_kai.c`의 bias-add 루프(`if(bias){for m{for r ym[r]+=bias[r]}}`)가
이 SME2-arch-flag TU에서 컴파일러에 의해 불법 SVE로
auto-vectorize됨 — SME2 커널(.S 어셈블리) 호출이 이미 스트리밍 모드를
끄고 반환한 *뒤에* 이 코드가 실행되는데, Apple Silicon엔 순정 SVE가
없어 무조건 illegal. 이 파일이 이미 두 번 겪었던 것과 같은 클래스의
"caller-plain 위반"인데, 이 루프는 나중에 추가돼서 그 수정들을
비껴갔던 것. **더 중요한 발견**: `kai_sme2_gemm_f16lhs()`(MoE
f16p-LHS 경로, D-f16lhs-3 기본값)에도 완전히 동일한 미수정 루프가
있었음 — `matmul_sdot`의 기존 배포 경로도 bias 있는 Q/K/V에서
이론상 같은 크래시 위험을 안고 있었는데, 실전 B값이 전부 16의
배수라 우연히 한 번도 안 터진 것뿐(구조적 보호 아님). `#pragma
clang loop vectorize(disable)`로 수정(처음엔 outer 루프에 잘못
붙여서 무효였고, inner 루프로 옮긴 뒤 otool로 SVE 0개 확인). 원인
해결 후 matmul_t에 `kai_route_min(W,M,1)` 재적용 → spec 모드
byte-identical 출력(3회 반복 확인) + **처리량 33.5→~100 tok/s
(2.98배) 실측**, matmul_sdot 기존 경로도 `identical 1`로 무영향
재확인. 커밋 `b8302d3`+push. Phase 3-5 완전 종결.

**Phase 3-1~4,6 전체 완료 (2026-08-26, 같은 날) — GGUF 5개 모델
shape-ladder 검증 + HD=64 신규커널 결정**: Qwen2.5-0.5B(HD=64,
GROUP=7)/Llama-3.2-1B(HD=64,GROUP=4)/Qwen2.5-3B(HD=128,GROUP=8)/
Llama-3.2-3B(HD=128,GROUP=3)/Qwen2.5-7B(HD=128,GROUP=7, 2-shard) 전부
llama.cpp(`llama-simple`) 대비 greedy 검증 완료 — 모델 클수록 완전일치
토큰수 증가(2→2→2→11→9), 일관된 패턴. 발견/수정 2건: (1) Q5_0
quant type 미지원(작은모델 전용 quant recipe) → `gguf_quants.c`에
`dequant_row_q5_0` 포팅. (2) Llama-3 NTK rope scaling GGUF에 전용키
없음(low/high_freq_factor 없음, 이전 세션이 미룬 조사 완료) →
llama.cpp가 사전계산한 `rope_freqs.weight` 텐서([hd/2])로 대체함을
`ggml-cpu/ops.cpp` 소스로 확인, 기존 `g_rope_scale[]` 메커니즘(레거시
로더와 동일 슬롯)에 `1/ff[i]`로 얹음. Qwen2.5-7B는
`llama-gguf-split --merge`로 2-shard 병합(엔진 로더 자체는 단일파일
유지). Phase 3-6(HD=64 신규 attn_neon.h 커널 필요여부)은 기존
`QWEN_PROF=1 bench` 프로파일러로 실측 — attention이 전체 decode의
1.46~2.49%뿐(FFN proj 55~63%, head_gemv 20~32%가 진짜 병목) →
**신규 커널 안 만들기로 결정**(이론상 최대개선 2.5%로 정당화 안 됨).
commit d14a4fa→3337f5e→13152b5→fb4eea3→320bf49, 전부 push.
**Phase 3(sub-step 1~6) 완전 종결.** 다음은 Phase 4(MoE 경로
de-hardcode, 계획서 자체가 "가장 큰 덩어리"로 표시, 사용자 확인 후
착수).

## MoE-format safetensors Steps 4-6 + per-role/per-expert precision engine (2026-08-29~30 완료)

**인프라 선행**: bob(M4,16GB RAM+1GB swap뿐, `sysctl` 실측)이 Qwen3-30B-A3B
(기본정밀도 ~32.4GB, 풀int4도 ~15.6GB 필요)를 실행할 수 없음을 config.json
기반 계산으로 확인 → macstudio(M1 Max, 64GB RAM)로 대형모델 작업 이관.
시스템 Python 3.9의 `huggingface_hub` 캡핑(1.8.0) 문제는 포터블 Python 3.12
(`~/py312_portable/`, brew/sudo 불필요, astral-sh 배포)로 해결, 다운로드
2-4MB/s→~10MB/s. bob<->macstudio 직접 LAN SSH(`bob-lan`/`macstudio-lan`
alias, 전용키)로 대용량 전송 8-10배 향상(~9MB/s). 사용자 지시로 xox까지
포함한 3-way Tailscale SSH mesh를 **영구** 구성(임시키 정리 계획 폐기,
memory 별도기록: `reference_bob_macstudio_lan_ssh`/`reference_3way_tailscale_ssh_mesh`).

**Step 6(OLMoE) 진행 중 진짜 버그 2건 발견+수정**:
1. **out-of-vocab prompt ID**: 기본값 100000이 OLMoE vocab_size=50304 범위
   밖 → C/MLX 양쪽 다 pos-0 로짓이 우연히 같은 all-zero로 퇴화(둘 다 틀려서
   눈치채기 어려웠음). 실 프롬프트ID로 교체해 해결.
2. **D-qknorm-1(진짜 아키텍처 버그)**: Qwen3-MoE의 q_norm/k_norm은 reshape
   *후* per-head 정규화(weight shape=[128]), OLMoE는 reshape *전* 전체벡터
   정규화(weight shape=[2048]) — 같은 텐서 이름, 다른 축 관례. 실 safetensors
   shape 검사+mlx_lm 소스 직독으로 확인. `MOE_QKNORM_WHOLE_VECTOR` 플래그+
   `moe_qknorm_apply()` 헬퍼로 수정.

**근본원인 추가규명(사용자가 "남은 오차를 계속 파고들어"라고 명시 지시)**:
레이어별 hidden-state 실측 덤프로 레이어13의 근접-동점 라우팅 플립(score
gap 1.87e-05)을 확정 특정 — 원인은 그 이후 레이어들의 비선형 증폭 +
레퍼런스 자체 최종레이어 norm 자연축소(56.2→8.25)가 상대오차를 부풀림.

**전략적 전환(사용자 지시)**: 이 발산을 "더 못 없앤다"가 아니라 "그래서
per-role 정밀도 엔진이 필요하다"는 실증사례로 재구성. per-role 엔진의
F32 게이트를 (기존 embed_tokens/lm_head 전용에서) attention 역할까지
확장(`st_register_moe_f32_as_af()`, forward-pass 함수 무변경).

**실측(OLMoE, 8개 teacher-forced position)**:
| | int8 baseline | attn F32 only | expert top-8 only | attn F32 + expert top-8 |
|---|---|---|---|---|
| router hard mismatch | 1/128 | **0/128** | 1/128(다른 위치) | **0/128** |
| rel-L2 실패 위치 수 | 5/8 | 4/8 | 4/8 | **1/8** |

attention F32 승격만으로 router mismatch 완전제거+최악위치 4.8e-2→6.5e-3.
잔여 발산(pos 0,2,3,7)은 라우팅 expert 양자화 노이즈로 추적 → AEQ식
frequency/importance 프로파일러(`moe_st_expert_profiler_olmoe.py`, 60개
프롬프트)로 top-8-per-layer 선별승격 실행. **핵심 발견**: expert top-8
**단독**으로는 baseline 대비 명확히 안 나음(같은 4/8 실패, sparsity
재확인 — 60개 프롬프트에도 레이어당 63-64/64 expert가 활성화돼 "top-8"이
거의 균일분포에서 임의로 자르는 것에 가까움). 하지만 **attention F32와
결합하면 상호보완적으로 작용해 실패위치 4/8→1/8로 개선** — attention이
못 건드리던 발산까지 닫음. 엔진의 실제 가치(선택적·근거기반 승격이
baseline이 못 닫는 발산을 닫음)를 실증.

`QWEN_MOE_EXPERT_BITS` 포맷을 2필드→3필드(`<layer> <expert_id> <bits>`,
bits∈{4,8,32})로 변경(intentional breaking change, 이 기능 자체가 기본값
채택된 적 없음). `moe_decode_af()`/`moe_matvec_af_row()`를 per-expert bits
해석(`t->ebits[e]`)하도록 수정 — attention 전용(E=1) 케이스에선 안 드러나던
버그를 mixed E>1 승격에서 발견·수정.

커밋: `c26ddb0`(fix: qk-norm axis+F32 attention), `5606ad3`(docs: Steps
4-6), `48c228d`(feat: per-expert F32 bits=32), `7127eb7`(docs: top-k
expert promotion 실측). 전체 세션기록:
`~/Desktop/HISTORY/2026-08-29_vdsp-moe-per-role-precision-engine.md`.

**남은 것**: Step 4/5(Qwen3-30B-A3B 구조검증+실측게이트 — 61GB 체크포인트는
이미 macstudio에 다운로드 완료, 아직 등록/forward 코드 미실행), Step
7(전체 마무리 문서화+커밋). pos 7의 잔여 1.9e-2 발산은 이번 라운드에서
더 파고들지 않기로 결정(엔진 유효성 실증 목적은 이미 충족).

## MoE-format safetensors Steps 1-7 전체 CLOSED (2026-08-30)

Step5(Qwen3-30B-A3B 실측게이트)+Step7(최종문서화) 완료로 7-step 계획 전체 종결.
3개 아키텍처(deepseek_v2/qwen3_moe/olmoe) 전부 구조등록+실측게이트 통과.

**Step5 실측 방법론 우회 (재사용 가능 패턴)**: Qwen3-30B-A3B(bf16 61GB)는 DeepSeek/
OLMoE 게이트가 쓴 "MLX fp32 강제업캐스트"(~122GB 필요) 방식이 macstudio 64GB로도
물리적으로 불가능 — 사용자가 "llama.cpp로는 못 돌려?" 제안 → `llama-cpp-python`
설치해 이미 디스크에 있던 Q4_K_M GGUF(18.5GB, mmap기반이라 bob 16GB로도 무리없음)를
레퍼런스로 채택. 정직한 캐비어트: 이건 pristine ground truth가 아니라 독립적으로
양자화된 다른 아티팩트라 rel-L2가 원래 게이트들보다 크게 나옴(0.03~0.38) — 벡터
norm비율(0.99~1.05)+Pearson상관(0.96~0.996) 실측으로 "diffuse 노이즈일 뿐 구조적
불일치 아님"을 확인하는 절차가 필수. 결과: 7/8 argmax일치, 유일 불일치(pos6)는
양쪽 다 top-2가 같은 토큰 순서만 바뀐 근접동점으로 확인(버그 아님).
**교훈**: fp32-forced MLX reference가 메모리부족으로 불가능할 때 llama.cpp+기존
GGUF양자화본이 정직하게 캐비어트 명시하는 조건 하에 대체 참조로 쓸만함 — 향후
대형 체크포인트 게이트에 재사용 가능한 패턴.

README에 사용자 지시로 "테세우스의 배" 프레이밍 섹션 신규 추가: 아키텍처별
최소-정밀도-유지 조합을 찾는 조합최적화 탐색 자체(nCk 스케일)는 이 프로젝트가
아니라 향후 다른 연구자/LLM이 수행할 미래 과제, 이 엔진은 그 탐색을 실험
가능하게 만드는 밑바탕(substrate)이라는 메타 프레이밍 — 이 프로젝트 전체의
존재이유를 명문화.

전체 세션기록: `~/Desktop/HISTORY/2026-08-29_vdsp-moe-per-role-precision-engine.md`.
남은 것: 이 7-step 계획 관점에서는 없음(D-expert-promo-1은 향후 확장 여지로 문서화
됐을 뿐 미해결 버그 아님). 다음 장기목표 방향은 README의 "open question" 섹션이
가리키는 조합최적화 탐색(다른 연구자/LLM 몫) 또는 vdsp v2(GPU+co-execution) 트랙.

## V5-pre + Opus 재계획 (2026-08-30) — MLX 백엔드 투자 판단이 크게 바뀜

**V5-pre (llama.cpp+Metal 저비용 사전탐색, 실행완료)**: Qwen3-30B-A3B(18.5GB)는
bob(M4,16GB) Metal 워킹셋 한계(12.71GB)를 46% 초과해 OOM(하드웨어 한계, 백엔드
무관). DeepSeek-V2-Lite(9.65GB, 한계 안에 여유)로 재시도해 llama.cpp+Metal
**48.34 tok/s**(단일유저) 확보, 우리 CPU/SME2 엔진 실측 **2.47 tok/s**(8인 동시
서빙 합산, `QWEN_MOE_CBATCH=1`이 REQS/SLOTS 무시하고 8프롬프트 하드코딩된 것도
새로 발견) — 약 19.6배 차이.

**Opus Plan 에이전트 재계획(같은 세션, bob 라이브 재검증)** — 몇 가지 핵심 수치가
뒤집힘:
- **메모리 대역폭 루프라인 계산(새로 함)**: bob GPU 스트리밍 대역폭 실측
  94.6GB/s. B=1 디코드는 토큰당 ~1.53GB 읽어야 해서 **이론 한계 61.8 tok/s**
  — llama.cpp의 48.34는 이미 이 한계의 **82.6%**. 즉 **커스텀 MLX 백엔드도
  B=1에서는 최대 ~1.28배**밖에 못 이김 — 8-12세션 투자를 정당화 못 함.
  B=64는 이론한계 ~640 tok/s인데 llama.cpp는 180.91(28%)만 달성 — **여기
  ~3.5배 여지가 진짜 기회**(배치 서빙 영역).
- **P0.1 "CPU+GPU 동시실행 상호간섭 거의 없음"이 M1 Max(macstudio)에만 해당,
  bob(M4)은 GPU -6%, CPU -13~17%로 재확인됨** — 이전 요약이 두 칩 다 해당하는
  것처럼 일반화했던 게 틀렸음(design doc 자체의 §4.2는 이미 정확히 말하고
  있었음).
- **DeepSeek-V2-Lite에서 co-execution은 이상적 경우도 +3.3%, 실측 간섭 반영하면
  -3.2%(순손실)** — CPU/GPU 속도비가 30배라 P0.1의 +20% 효과(2.9배 비율에서
  측정)가 전혀 다른 체제. **FreeToken의 실제 메커니즘(GPU가 PCIe fetch로
  멈춰있는 동안 CPU가 overflow 계산)도 통합메모리에는 그 "멈춰있는 시간" 자체가
  없어서 이식 안 됨** — CPU expert 1개(~2.3ms)가 GPU 레이어 전체(~0.08ms)의
  28배 걸림, 절대 못 숨김.
- **결론**: V6(role별 CPU/GPU 디바이스 배정 + FreeToken식 expert 분담)는
  **속도가 아니라 용량(capacity) 기능으로 재정의**됨 — 못 올라가던 모델
  (Qwen3-30B-A3B)을 부분 GPU-상주+CPU-overflow로 실제로 돌아가게 만드는 게
  진짜 가치, throughput 향상은 기대하면 안 됨.
- **전체 세션 규모**: V5(GPU 백엔드 기초) 8-14세션 + V6(role+device 일반화)
  7-12세션 = **총 15-26세션** — 원래 8-12세션 추정보다 훨씬 큼.
- **권고**: V5a만 우선 승인(20분짜리 gate 0 link+metallib 스모크테스트부터),
  V5b+V5c를 한 묶음으로 진행하되 V5c의 kill-gate(llama.cpp의 48.34 tok/s
  못 넘으면 중단 재검토)를 진짜 정지점으로 삼을 것.

전체 계획 파일: `~/Desktop/vdsp_v2_design/trackb_v5_plan/PLAN_v5_v6_gpu_backend_and_role_device.md`
(F-1~F-16 실측근거, D-gpu-1~10 설계결정, V5a~g/V6a~d 세부 게이트 전부 포함).

## V5a 완료 (2026-08-30, 같은 세션 후속) — 8개 게이트 전부 PASS

사용자가 "V5a 시작"으로 직접 착수 지시(질문 없이 바로 실행하라는 명시적
피드백 있었음, [[feedback_...]] 참고할 만한 교훈: 명확한 지시 후 곁가지
확인질문 금지). `mlx_moe.h`/`mlx_moe.cpp` 신규(C++17, MLX 링크, `sme2_kai.h`
벤더경계 패턴 그대로), `qwen_infer.c`에 `run_moe_gpu_mode()` 추가
(`#ifdef QWEN_GPU_MLX` 전체가드).

**실제 버그 2건 발견·수정(원인규명 방법론 적용, 표준 hw-kernel-vendoring
스킬이 경고하는 정확히 그 클래스)**:
1. `mx::allocator::can_reuse_alien_buffer()`가 이 MLX 빌드(bob)에서
   Python 아닌 C++ 프로세스에서 호출 시 **무조건 SIGSEGV** — 격리
   프로브로 확정(단순 malloc 포인터+MLX 워밍업 후에도 재현). 원래
   optional 진단용(zero-copy 여부 리포팅)이라 correctness엔 불필요함을
   확인 후 완전 제거, Gate5는 MLX 메모리 카운터만으로 잔류.
2. Gate3 dequant 동치검사가 `col0` 파라미터 없이 항상 컬럼[0,16)만
   비교하는데 CPU측은 랜덤 `col0+c`를 봐서 **서로 다른 컬럼을 비교하는**
   테스트-하네스 버그(max_abs_diff=0.69로 표면화) — `col0` 파라미터
   추가로 수정. 부수적으로 `mx::dequantize`가 2차원 이상 요구한다는
   것도 발견(1D row는 예외), `expand_dims`로 해결.

**최종 실측 결과(DeepSeek-V2-Lite AF-blob, 269개 텐서, 9.81GB)**:
Gate1(기본빌드 byte-identity) PASS · Gate2(bits sanity) 269/269 PASS ·
Gate3(dequant 동치) 400개 좌표 max_abs_diff=**0.0 정확히** PASS ·
Gate4(GEMM 교차검증) worst rel_l2=2.13e-07(bar 1e-5) PASS ·
Gate5(residency) active=peak=9.814GB vs 12.71GB 천장, 여유 확보 PASS ·
Gate6(경고베이스라인) 20개 그대로 PASS · Gate7(SVE/SME 누출) 0건 PASS ·
R8(회귀) byte-identical .o가 그 자체로 최강 증거.

**V5a 완료, V5b(레이어0 MLA attention on GPU)가 다음 승인 대상.**
상세: `RESULTS.md` §"V5a: GPU weight binding + numerical equivalence --
real gates, all PASS", 스냅샷 `~/Desktop/vdsp_v2_design/trackb_v5a_results/`.

## V5b 완료 (같은 세션 후속, 2026-08-30 07시대) — 레이어0 MLA attention on GPU

사용자가 "V5b 시작 질문 및 멈추지 말고 오전 11시까지 진행" 지시 → 질문 없이
바로 실행. `mlx_gpu_mla_layer0()`/`mlx_gpu_mla_config()` 신규 구현 —
q_proj/kv_a_proj/kv_b_proj/o_proj GEMM(V5a가 이미 바인딩한 텐서 재사용),
kv_a_layernorm RMSNorm, RoPE(traditional=true 교차쌍+YaRN freqs 나눗셈),
causal self-attention 전부 GPU.

**와이어링 전 3개 빌딩블록을 격리 프로브로 먼저 검증**(hw-kernel-vendoring
원칙 — 헤더만 읽고 믿지 않음): `fast::rope`(freqs override)가 실제
DeepSeek YaRN 테이블(32개 distinct)로 CPU 검증 arithmetic과
max_abs_diff=4.77e-07 일치, `fast::scaled_dot_product_attention`
(mask_mode="")이 CPU 수동 softmax와 1.49e-08 일치, `fast::rms_norm`이
1.19e-07 일치 — 셋 다 안전 확인 후 실제 배선.

**실측 결과(8개 실 prompt position, MoE-2a 실캡처 prompt_ids
[100000,549,4345,280,8204,317,245,1234] 재사용)**: cpu_vs_gpu
worst=2.78e-07(bar 1e-4) PASS, cpu_vs_truth worst=1.299e-03(bar 1e-2,
MoE-2a 원래 결과 "≤1.3e-3"와 6자리까지 일치 — 하네스 자체 정확성
재확인), gpu_vs_truth worst=1.299e-03(bar 1e-2) PASS — 계획이 예측한
전이관계(GPU≈CPU 수준 오차라 GPU-vs-truth≈CPU-vs-truth) 정확히 실측
확인. 회귀: 기존 pre-V5 baseline과 byte-identical 재확인(V5a+V5b 전체
합산 후에도).

상세: `RESULTS.md` §"V5b: layer-0 MLA attention on GPU -- real gates,
all PASS", 스냅샷 `~/Desktop/vdsp_v2_design/trackb_v5b_results/`.
**V5b 완료. 다음: V5c(27레이어 전체 forward, 계획의 kill-gate) — 별도
승인 필요.**

## V5c 완료 — 정확도게이트 전부 PASS, **처리량 kill-gate 실패**(같은 세션, 07시대)

사용자 "V5b 시작... 멈추지 말고 오전 11시까지 진행" 지시 연장선에서 계속
실행. `mlx_gpu_layer_step()` 신규 — 전체 FFN(dense+router+gather_qmm
라우팅전문가+shared전문가) 27레이어 전체로 확장. 신규 MLX API
`gather_qmm` 격리검증(group_size=8은 커널 없음 에러, 64로 재시도해
max_abs_diff=4.77e-07 확인 후 배선). Router(64-wide softmax+top-6)는
호스트(CPU)에 의도적으로 유지 — 정확도/성능상 GPU로 밀 이유 없는
합법적 role별 CPU/GPU 분담.

**정확도: 첫 실행에서 전부 PASS**(이번 라운드는 버그 0건 — V5a의 2건과
대조): argmax parity 8/8(실제 MLX ground truth 대비), gpu_vs_cpu worst
rel_l2=4.84e-07(bar 1e-4), gpu_vs_truth worst=3.91e-03(bar 1e-2) —
cpu_vs_truth와 6자리까지 일치, 게다가 이 값이 이 프로젝트 MoE-2b 자체
기록("rel-L2 1.25e-3~3.91e-3")과 거의 정확히 일치 — 전체 새 파이프라인이
진짜 정확함을 강하게 뒷받침.

**★★★ 처리량 kill-gate 실패**: 실측 **~7.8 tok/s**(3회 반복,
7.771/7.789/7.812) vs 계획의 bar **48.34 tok/s**(llama.cpp+Metal parity)
— roofline 61.8 tok/s의 겨우 13%(llama.cpp는 82.6%). **원인규명(추측
아니라 수치로 정합성 확인)**: 라우팅 레이어당 ~19개(비라우팅 레이어
~11개), 27레이어 합산 **토큰당 ~505개의 개별 mx::eval() 디스패치** —
128.7ms/토큰 ÷ 505 ≈ 0.25ms/디스패치, M=1(단일토큰) 초소형 GEMM 대비
Metal 커맨드버퍼 제출+대기 고정비용이 압도하는 정합적 수치. 계획
자체의 D-gpu-4 설계원칙("토큰당 eval() **1회**")을 정확히 위반한
경우 — 이번 라운드는 신규 프리미티브(gather_qmm 등)를 하나씩 격리
검증 후 배선하는 정확성우선 방식으로 개발했고, 그 결과 각 프리미티브
호출이 즉시 eval()되는 eager 패턴이 됨(27레이어 전체를 하나의 lazy
그래프로 지연시켰다가 마지막에 1회 eval()하지 않음). **병목은 이
구현의 eager-eval 디스패치 패턴이지, gather_qmm 자체 연산비용도 MLX/
하드웨어 한계도 아님**(V5b의 격리된 rope/rms_norm/sdpa 전부 정확+
빠른 원시함수임을 이미 확인) — 고쳐야 할 지점이 명확히 식별됨.

**계획 자체의 kill-gate 프로토콜대로 여기서 정지+재스코프 필요**: V5d/e는
시작 안 함. 수정안(27레이어 전체를 lazy 그래프 1개로 재구성, eval() 1회로
축소)은 식별됐으나 실제 엔지니어링(별도 세션 규모)이 필요하고 성공을
보장하지 않음 — 시도하지 않고 정직하게 보고.

상세: `RESULTS.md` §"V5c: full 27-layer GPU forward -- correctness gates
all PASS, KILL-GATE FAILS", 스냅샷 `~/Desktop/vdsp_v2_design/trackb_v5c_results/`.

### 세션 끝 시점 상태 (이 창)
- V5a/V5b 전부 완료+실측+문서화+커밋 완료.
- "릴레이" 모니터링 요청은 완전 미해결·의미불명 상태로 남음 — 사용자가
  다시 꺼내기 전엔 건드리지 말 것.
- 행동교정 기록: 명확한 지시("V5a 시작"/"V5b 시작... 멈추지 말고") 이후엔
  곁가지 확인질문 금지, 시간이 허락하는 한 계속 실행(사용자가 명시적으로
  화냄, 시간낭비 지적한 전례 있음).

## V5c-fused 시도 (같은 세션 후속) — 버그 1건 발견+수정, 실측 ~3.2배 가속,
## 하지만 버그 2건째 미해결로 kill-gate 여전히 실패

사용자 지시 "eval() 한 번으로 lazy graph 재구성 시도해봐"(V5c kill-gate
실패 직후) 그대로 착수.

**1차 시도(one-eval-per-TOKEN, 계획의 D-gpu-4 원칙대로)**: 27레이어 전체를
lazy 그래프 하나로 짓고 마지막에 eval() 1회. **결과: pos=0 정확(5.22e-07),
pos>=1 전부 틀림**(rel-L2 0.4~1.3) — 개별 서브계산(레이어별 attention,
dense FFN, 라우터, routed FFN, 각 레이어 x_out 전체)은 강제 조기 eval()로
확인하면 전부 정확한데도 그렇다. MLX 자체 그래프 실행기 내부 원인은 이번
라운드 범위 밖. **처리량은 실제 ~37.8 tok/s 도달**(48.34 bar에는 미달이나
디스패치-오버헤드 진단이 방향상 맞았음을 재확인).

**2차 시도(one-eval-per-LAYER, 절충안)**: 레이어당 ~15-20개 연산을 하나의
lazy 그래프로 짓되 **레이어마다** eval() 호출(~27-28회/토큰). 1차 시도의
버그가 그래프 "깊이"(27레이어 병합) 문제인지 테스트하려는 의도.

이 과정에서 진짜 버그 2건 발견 — 둘 다 이미 검증된 eager
`mlx_gpu_layer_step()`과 lazy 중간값을 대조(임시 디버그 훅)하는 동일한
방법으로 찾음:

- **버그2(발견+수정)**: `switch_down`(선택된 전문가마다 다른 입력행 필요)에
  `gather_qmm`을 쓰면 `(TOPK,TOPK,out)` 전체 cross product가 나옴(행별
  페어링 아님, 격리 검증 완료) — 정답은 그 대각선, `mx::take_along_axis`로
  추출. 이 추출 자체는 격리 상태(합성 mlx.core 재현, 단독 eval())에서는
  정확한데, **다른 무관한 연산들과 함께 더 큰 하나의 eval() 배치에 묶이면
  조용히 거의 0에 가까운/뒤섞인 값을 반환**함(레이어1 강제 조기eval→정상,
  레이어2 동일코드 미적용→여전히 틀림, 2회 재현). **수정**: down_flat을
  계산 직후 단독으로 eval() — pos=0이 eager와 정확히 일치(5.220869e-07,
  3회 반복 재현) 확인.

- **버그3(발견, 미해결)**: 버그2 수정 후에도 K/V 히스토리가 있는 모든
  위치(pos>=1)에서 여전히 틀림(gpu_vs_cpu 0.44~1.26, 히스토리 길수록
  증가) — 3회 반복 동일 재현. x_mid/FFN출력/top_idx/swiglu_2d/down_flat을
  모든 레이어·모든 위치에서 대조하는 광범위한 이분탐색을 했는데 **전부
  정확하다는 결과**가 나왔으나, **이 이분탐색 도구 자체가 오염됨**을 뒤늦게
  발견: 검증용 eager 경로(`mlx_gpu_layer_step_dbg`)가 lazy 경로와 같은
  `g_mla_K`/`g_mla_V` 캐시를 공유해서, 검증 호출 자체가 조용히 그 캐시
  슬롯을 "자가치유"해버림 — 실제 lazy 경로의 진짜 동작을 가려버린 것.
  이후 깨끗한(비오염) 파이프라인에 직접 4가지 "이것도 단독 eval 필요한가"
  테스트(o, attn, x_mid를 명시적 eval 출력으로, FFN결합출력을 명시적
  eval출력으로, x_out에 강제 host-sync read)를 시도 — **4가지 전부
  바이트 단위로 동일하게 틀린 결과**, 버그2와 달리 eval-타이밍 문제가
  아닐 가능성이 높음을 시사. K/V 히스토리 읽기/concat 경로(레이어 내
  유일한 position-의존 코드)의 진짜 로직버그일 가능성이 가장 높으나,
  이번 라운드 시간예산 내에 격리 못함.

**실측 처리량(버그2 수정만 적용, 3회 반복)**: 0.648s/24.709, 0.649s/24.653,
0.650s/24.625 tok/s — **eager 7.8 tok/s 대비 실제 ~3.2배 가속**(48.34 bar
에는 여전히 미달). argmax parity 1/8(pos=0만 정확) — **이 빌드는 아직
production-ready 아님**, pos>=1 정확성 미확정 상태로 사용 금지.

**정직하게 정지**: 투자한디버깅 규모(investigation-protocol의 "2회 이상
막히면 정직하게 보고" 기준 충족) 대비 반환 감소 → 이번 라운드 종료. 임시
디버그 계측(peek 함수들, 전역 g_dbg_*, env-var 게이트 비교블록)은 전부
제거하고 `mlx_gpu_layer_step_dbg()`(추가 out-param 있는 eager 변형)만
재사용 가능한 회귀도구로 남김.

상세: `RESULTS.md` §"V5c-fused: lazy-graph rewrite attempt -- root cause
confirmed correct, real ~3x speedup demonstrated, but TWO real
MLX-composition bugs found (one fixed, one unresolved) -- KILL-GATE still
FAILS".

### 다음 세션 재개 지점 (2026-08-30 해결됨 — 아래 섹션 참고)
- ~~Bug3(pos>=1) 근본원인 미해결~~ → **해결됨, 아래 참고**
- V5d/e는 여전히 시작 안 함(계획 자체 kill-gate 프로토콜, 처리량 미달로
  여전히 미착수).
- 커밋 여부: 이 시점 코드+문서 변경은 이후 `fe7f2c7`로 커밋됨(사용자
  "commit this" 승인).

## Bug 3 재시도 — "device-side fixed-shape" 원칙으로 근본원인 발견+수정
## 완료 (같은 날 후속 세션, 2026-08-30) — ★★ 8개 위치 전부 정확, 여전히
## 처리량 kill-gate는 미달

`fe7f2c7` 커밋 직후 사용자가 두 가지를 순서대로 지시: (1) "cuda graph는
이 lazy graph문제를 어떻게 해결했지? FreeToken에서는? AEQ에서는?" —
리서치 질문, (2) "Bug 3 device-side fixed-shape로 다시 접근해봐" — 방금
리서치한 원칙을 실제 재시도에 적용하라는 지시.

**리서치 결과**: CUDA Graph capture/replay는 이미 정확한 eager 실행을
1회 기록 후 그대로 재생 — MLX의 매 호출마다 새로 해석되는 lazy 그래프와
구조적으로 다르며, 이 트랙이 계속 만난 버그 계열에 원리적으로 면역.
FreeToken(github.com/FlashML-org/FreeToken, `gh api`+`WebFetch`로 실제
소스 확인)의 `GraphRunner`(`python/freetoken/engine/graph.py`)가 실제
CUDA graph capture를 하며, 자체 `moe.py` 주석이 decode 경로를
"device-side, fixed-shape... capture-safe"라 명시 — 이 원칙을 MLX엔
없는 literal capture/replay가 아니라 설계 원칙으로 이식. AEQ는 메모리
검색으로 무관 확인(양자화 비트폭 연구, 런타임 그래프 실행과 무관).

**재설계**: 매 위치마다 커지는/재concatenate되는 K/V 히스토리를
CONSTANT-shape·CONSTANT-pointer 윈도우(`{1,H,MLA_L0_MAXPOS=32,*}`
전체를 매번 그대로 wrap)+boolean mask(`j<=pos`)로 교체,
`mx::fast::scaled_dot_product_attention(..., "array", mask_arr)`로
검증(mask_mode="array" 필수, `strings libmlx.dylib`로 사전확인 —
실행 전 잡은 API 제약, 런타임 버그로 실제 발생한 적은 없음). 레이어당
eval 1회→2회(Stage A: 이 위치 자신의 K/V 계산+즉시persist, Stage B:
고정윈도우 attention+FFN)로 분리.

**재설계 직후 pos=0까지 회귀(rel-L2 0.66~1.76)** — investigation-protocol
그대로 재수행(bob 실제 MLX 0.32.1 위에서 격리 probe, 로컬 syntax check
아님):
- 기각: 마스킹 메커니즘 자체(`probe_mask.cpp`/`probe_mask2.cpp`/
  `probe_mask3.cpp`, 실제 모델 차원 H=16/QHD=192/VHD=128로 3중 검증,
  전부 `max_abs_diff=0.0`)
- 기각: stale Metal 버퍼 캐싱(`probe_stale_ptr.cpp`)
- 기각: eval 타이밍/배칭(버그2와 같은 형태의 standalone eval 시도,
  결과 불변)
- **확정 근본원인**: `v_new = mx::slice(kv_b_r, {0,0,NOPE}, {1,H,NOPE+VHD})`가
  후속 연산 없는 bare slice(형제 `k_nope`는 `mx::concatenate()`로 흘러가
  부수효과로 압축됨)라 `mx::eval()`이 압축을 강제 안 함
  (`row_contiguous` 여전히 false) — 이 모델의 `NOPE==VHD==128` 우연이
  실제 stride(256)와 naive `hh*VHD` 인덱싱이 가정하는 dense stride(128)를
  충돌시켜 `v_new`의 진짜 데이터와 `k_nope`의 이웃 바이트가 조용히
  뒤섞임(관측된 "한 헤드 걸러 정확" 패턴과 정확히 일치, `probe_slice.cpp`로
  실차원 격리재현+strides `[4096,256,1]` vs 기대 dense `[2048,128,1]`
  직접 확인).
- `mx::copy(slice(...))` 1차 시도 **실패**(MLX 옵티마이저가 strided view를
  "이미 valid"로 보고 실제 복사를 생략 — 격리 probe로도 확인).
  **진짜 수정: `mx::contiguous(mx::slice(...))`** — 격리 probe로 먼저
  확인(row_contiguous→true, dense strides, 모든 naive 읽기 정확) 후
  실코드 적용.

**실제 하드웨어 결과(bob, `QWEN_MOE_BASE=~/moe_base_deepseek
QWEN_MOE_GPU_FUSED=1 ./qwen_infer_v5cfused_final`)**:
```
argmax parity vs real MLX ground truth: 8/8 (bar 8/8)
WORST across 8 positions: gpu_vs_cpu=7.505390e-07 (bar <=1e-4)
```
**★★★ Bug 3 완전 해결 — 8개 위치(0~7) 전부 정확, pos=0뿐 아니라
pos>=1 전체에서 처음으로 eager 경로와 사실상 machine-epsilon 수준
일치.** 이 세션에서 만든 fused GPU forward pass의 첫 완전정확 버전.

**처리량(3회 반복, 동일 8-warmup/16-measured 프로토콜)**: 0.736s/21.753,
0.734s/21.793, 0.734s/21.786 tok/s. **~21.78 tok/s vs 48.34 bar(~45%)**
— eager 7.8 tok/s 대비 실제 ~2.8배 가속이지만, 버그2만 적용된(하지만
pos>=1은 틀렸던) 이전 24.7 tok/s보다는 느림 — 고정윈도우 2단계-eval
설계가 정확성의 대가로 치른 실측 처리량 비용(D1 결정기록에 사전 명시된
그대로, 놀라움 아님).

**KILL-GATE: 여전히 실패**하지만 **이번엔 정확한 구현 위에서의 실패**
— 이전 두 번의 "더 빠른" 수치(37.8/24.7 tok/s)는 전부 pos>=1에서 틀린
빌드에서 나온 것이라 신뢰도가 근본적으로 다름. Gate 1(기본빌드
바이트동일)도 재확인 시도 — bob의 clang/ld 자체가 동일 소스 두 번
연속빌드에서도 바이트동일을 보장 안 함(비결정적 빌드, 격리
재현확인)을 발견해 raw cmp는 무효 판정, 대신 (a) 이번 세션
`qwen_infer.c` 변경이 100% `#ifdef QWEN_GPU_MLX` 안쪽임을 git diff로
정적 확인 (b) 디스어셈블(`otool -tv`) 대조로 실제 차이가 균일한
주소/literal-pool 이동뿐(로직 변경 없음)임을 확인 — 두 증거로 기본빌드
보존 결론.

상세: `RESULTS.md` §"Bug 3 RE-ATTEMPTED with the 'device-side
fixed-shape' principle -- FOUND, ROOT-CAUSED, and FIXED". 다음: V5d/e
착수 여부는 여전히 사용자 재스코프 결정 대기(처리량 미달, 정확성은
이제 해결).

## V5c-fused 처리량 최적화 (2026-08-31, repo-isolation 전환 후 첫 세션) —
## eval-count 감축으로 21.78→37.16 tok/s(+70.6%), 정확도 완전 유지,
## kill-gate 여전히 미달이나 격차 45%→77%로 좁혀짐

정확도(8/8)는 해결됐지만 처리량(21.78 vs bar 48.34)이 미달인 채였던
V5c-fused를, "토큰당 real `mx::eval()` 호출 수"(81회)를 직접 줄이는
방향으로 공략. 매 eval()이 실제 Metal command-buffer 제출+host-sync
경계라는 가설 하에, 실제 파이프라인 수정 전 항상 isolated probe로
bob의 실제 MLX 0.32.1에서 먼저 검증(이 프로젝트 기존 규율 그대로).

**Win 1 — switch_down의 강제 eval을 제거(우회가 아니라 근본원인 제거)**:
기존 switch_down이 `gather_qmm`에 `lhs_indices=arange(TOPK)`를 명시
전달해 (TOPK,TOPK,out) cross-product를 만들고 `take_along_axis`로
대각선만 추출하던 것(Bug 2의 원래 유발 지점, standalone eval로 우회
중이었음)을, **Apple 공식 `mlx_lm`의 `switch_layers.py`(`SwitchLinear`/
`QuantizedSwitchLinear.__call__`)가 이 정확히 같은 "행마다 다른
expert" 패턴에 lhs_indices를 아예 안 쓴다**는 사실을 소스 직접
확인으로 발견 후 그대로 적용. Python(`probe_gather_no_lhs.py`)+
C++(`probe_down_no_lhs.cpp`, 실제 TOPK=6/IM=1408/HIDDEN=2048차원)
두 단계로 사전검증: lhs_indices=nullopt 형태가 정확할 뿐 아니라
deferred eval(Bug2의 스트레스 조건 그대로 재현)에서도 안전함을
확인 — **Bug 2의 오염은 cross-product 조합 자체의 문제였지 "지연
평가" 자체의 문제가 아니었다**는 뜻. 결과: 정확도 8/8·gpu_vs_cpu
7.5e-07(완전 동일) 유지, 처리량 **21.78→29.99~30.06 tok/s(3회),
+37.7%**.

**Win 2 — Stage A/B eval 분리를 K/V persistent array로 제거**: 고정폭
K/V 윈도우(Bug 3의 수정)가 raw host 메모리(`g_mla_K`/`g_mla_V`)였기
때문에 매 레이어 `mx::eval(stageA)`가 강제였음(Stage B의 host wrap이
그 메모리를 읽기 전 반드시 완료돼야 하는데, MLX는 host memcpy
의존성을 볼 수 없음). Fused 경로 전용 신규 persistent `mx::array`
K/V(`g_fused_K`/`g_fused_V`, 최초 1회 zero-init)로 교체,
`mx::slice_update(src, update, start_indices, axes)`로 갱신(Python
바인딩은 dynamic-index 오버로드만 노출, C++엔 static Shape 오버로드도
있음 — 실제 사용은 dynamic 형태). `probe_slice_update_chain.py`로
먼저 검증: 5회 체이닝된 slice_update, 전부 지연, 끝에 단일 eval →
호스트버퍼 기준과 완전일치(diff=0.0), 아직 미평가 상태의 체이닝된
윈도우에 대한 masked attention 읽기도 numpy 대비 float32 정밀도까지
일치(1.19e-07) — bare `mx::slice()`와 달리 slice_update 결과는
`mx::contiguous()` 없이도 안전함을 확인(Bug 3와 다른 종류의 이슈였을
가능성을 사전에 배제). 결과: 정확도 8/8·gpu_vs_cpu 7.5e-07(완전
동일, bit단위로 이전과 같음) 유지, 처리량 **29.99→37.03~37.17
tok/s(3회), Win1 대비 추가 +23.6%**.

**누적: 21.78→~37.16 tok/s(평균, 37.171/37.153/37.159), +70.6%**,
정확도는 이번 라운드 전체에서 마지막 자릿수까지 완전 불변(eval
경계 개수/위치만 바꿨지 산술은 안 건드림). eval count: 81→55(Win1)
→**28**(Win1+2: 27레이어×1eval + finalize 1).

**반증된 가설(시도 후 롤백) — "토큰당 완전 1 eval"**: Bug2/Bug3 둘 다
닫혔으니, 원래의 Attempt 1(전체 27레이어를 그래프 1개로 지연, 끝에
eval 1번)의 옛 미해결 버그("Bug 1": pos=0만 맞고 pos≥1부터 틀림)가
사실 Bug 3와 같은 원인이었을 것이라는 가설로 재시도 — x_out의
레이어별 eval도 제거해 finalize()에서 logits+전체 K/V를 한번에
eval. **실측: 처리량은 bar를 실제로 넘음(52.98 tok/s > 48.34)이나
정확도가 2/8로 완전히 깨짐, 오염 시작 지점이 정확히 pos≥1**(Bug 1의
원래 증상과 동일 패턴). → **"Bug1=Bug3" 가설은 반증됨** — 레이어
경계를 넘는 지연(deferral)에는 아직 원인 미규명의 별개 이슈가
실재함(레이어 내부 position 간 slice_update 체이닝은 probe로 안전
확인됐지만, 그건 다른 조건). 검증된 37.16 tok/s 버전(Win1+2, 레이어당
1 eval)으로 롤백, 이 반증 결과와 재현 신호(pos=0 정상/pos≥1부터
30~77% rel-L2)는 향후 세션이 Bug2/3와 같은 이분탐색 방법론으로
집을 수 있도록 RESULTS.md에 상세 기록.

**KILL-GATE: 여전히 미달**(37.16 < 48.34)이나 격차 45%→77%로 축소.
"Bug 1" 근본원인을 규명하면 남은 ~23%(52.98 tok/s 천장이 잠깐
확인됨)를 열 가능성이 있음 — 계속 추적할지 여기서 멈출지는 향후
세션의 열린 결정. 상세: `RESULTS.md` §"V5c-fused throughput round".
qwen_infer.c는 이번 라운드 미변경(mlx_moe.cpp/mlx_moe.h만).

## V5d 재스코프+구현 (같은 날 후속, 2026-08-31) — 배치 B-token GPU
## decode, 정확도 완전 통과, 처리량 목표 미달(다음 레버 특정됨)

Bug1 수정으로 KILL-GATE가 통과된 직후 "V5d/e 재스코프 시작해" 지시로
착수. 원래 V5d 설계(구 eager `mlx_gpu_layer_step()` 확장 가정)는 이번
세션이 완전히 대체한 아키텍처(진짜 토큰당 1 eval, persistent K/V
`mx::array`) 기준으로 다시 설계 필요 — Plan Mode로 전환해 "포크 대신
기존 fused 경로에 배치축 B를 직접 일반화" 방향을 사용자 승인 받고
착수.

**실 파이프라인 수정 전 격리 probe 2건**(이번 라운드 내내 지켜온
규율 그대로): `probe_batched_slice_update.py`(배치축 있는 체이닝
slice_update+masked attention, diff 0.0/1.79e-07 통과) —
`probe_batched_router_ffn.py`(라우터+gather_qmm+switch_down 전체
조합, B=8)에서 **실제 버그 1건 발견**: `gather_qmm`의
`lhs_indices=nullopt`는 x의 배치축과 rhs_indices의 배치축을 자동으로
페어링하지 않음 — `mx::expand_dims(x,{-2,-3})`(mlx_lm의
SwitchLinear 관례) 없이는 (8,6,8,1408)라는 진짜 BxTOPKxB
cross-product가 나옴(기대는 (8,6,1,1408)). B=1일 때는 배치축이
없어서(bare {1,HIDDEN}) 우연히 정상처럼 보였을 뿐 — 이 경로 자체가
B=1에서는 한번도 제대로 실행된 적이 없었음. expand_dims 추가로
수정, per-row 참조와 1.4e-06까지 일치 확인 후 실코드 적용.

**적용 내용**: `g_fused_K`/`g_fused_V`/`g_fused_x`에 선두 배치축 B
추가, 라우터 argsort/take_along_axis를 행별 2D 연산으로 일반화, RoPE
스칼라 offset은 추가된 B축에도 그대로 broadcast(검증됨), attention
mask는 **단일 공유** `{1,1,1,MLA_L0_MAXPOS}`로 유지(B tokens가 항상
같은 pos에서 lockstep이라는 V5d 고유 전제 — V5e 래그드 설계에서는
못 씀, 헤더 주석에 명시). 신규 `mlx_gpu_set_batch(B)`(기본값 B=1,
기존 호출부 전부 무변경으로 자동으로 이전 동작 유지).

**B=1 회귀(3회) 완전 통과**: 8/8, gpu_vs_cpu 완전동일(7.505390e-07),
처리량 52.456~52.527(기존 52.91 평균과 노이즈 수준 차이) — 일반화가
기존 동작의 진짜 상위집합임을 확인.

**신규 진입점** `run_moe_gpu_batch_gate()`(`QWEN_MOE_GPU_BATCH=B`,
`run_moe_gpu_fused_gate()`의 verbatim 구조적 미러) —
`run_moe_batch_verify_mode()`의 실제 64토큰 코퍼스+이미 교차검증된
`moe_forward_batch(use_gather=1)`을 CPU 기준점으로 재사용.

**GPU 정확도 표(실측, MoE-3c/3e CPU 표와 나란히 비교)**:

| B | flipped/B | worst rel-L2 | tok/s |
|---|---|---|---|
| 8 | 0/8 | 5.12e-04 | 76.2 |
| 16 | 0/16 | 6.02e-04 | 99.0 |
| 24 | 0/24 | 6.02e-04 | 105.7 |
| 32 | 0/32 | 6.63e-04 | 108.5 |
| 48 | 0/48 | 6.77e-04 | 107.0 |
| 64 | 0/64 | 6.77e-04 | 109.6 |

**★ 전 구간 완전 정확(flipped=0)** — CPU/SME2 arm이 B=64에서 margin
재검증 없이는 54/64(MoE-3e)에 그치는 것과 극명한 대조. GPU arm은
margin 재검증 장치가 구조적으로 불필요하다는 V5d 설계문서 예측이
실측으로 확인됨. Determinism(B=32, 3회 반복) 완전 동일값
(6.627989e-04) 확인.

**처리량은 목표 미달**: B=64에서 109.6 tok/s vs 목표 250(llama.cpp
180.91의 1.38배) — 심지어 llama.cpp 자체보다도 느림(0.61배).
**원인 특정(미적용)**: `sort_indices=true` global sort/unsort(mlx_lm의
`_gather_sort`/`_scatter_unsort`, F-4)를 아직 구현 안 함 — F-4가
사전 측정한 "unsorted가 B=64에서 7.02배 느림" 패턴과 정확히 일치하는
격차 크기라 재규명 없이도 유력한 다음 레버로 특정, 다음 라운드로
이관.

**상태**: V5d의 정확도+determinism 산출물은 완결·검증됨. 처리량 목표는
같은 날 F-4 구현으로 완결됨(아래).

## F-4 구현: global sort/unsort 적용, 실 in-situ 크로스오버 실측 —
## B=64 처리량 109.6→224.3 tok/s(+105%), llama.cpp 넘어섬

"sort_indices=true global sort/unsort(F-4)를 구현해" 지시로 착수.
mlx_lm의 `_gather_sort`/`_scatter_unsort`(switch_layers.py 원문 그대로)를
채택, 크로스오버는 F-4의 격리 마이크로벤치(캐시-warm 오염 지적됨)
재사용 대신 실제 27레이어 스트리밍 pass 안에서 직접 재측정.

**격리 probe 먼저**(`probe_sorted_gather.py`, 실제 NE=64/HIDDEN=2048/
IM=1408/TOPK=6, B=64): sorted 경로가 기존 배포된 unsorted 계산과
1.0e-05까지 일치(fp32 누적오차 수준) 확인 후 실코드 적용.

**mlx_moe.cpp 적용**: 라우터 FFN 분기에 런타임 `B*top_k >=
g_gpu_sort_threshold` 조건 분기 추가(신규 `mlx_gpu_set_sort_
threshold()`, 기본값 사실상 무한대=미적용 — 명시적 설정 전까진 방금
출하된 KILL-GATE 빌드와 완전 동일 동작). sorted 분기: top_idx를
`{B*top_k}`로 flatten→전역 argsort(행별 아님)→`mx::take(...,
order//top_k)`로 x를 정렬된 선택 순서로 gather→gather_qmm을
sorted_indices=true로 gate/up/down 전부 실행(이미 row-per-selection
레이아웃이라 unsorted 분기와 달리 expand_dims 불필요)→`mx::take(...,
inv_order)`로 원래 순서 복원.

**실측 크로스오버 스윕**(실하드웨어, `QWEN_MOE_GPU_SORT_THRESHOLD`로
강제 전환하며 실제 배치 게이트 파이프라인 안에서 측정):

| B(B*top_k) | unsorted | sorted | 개선율 |
|---|---|---|---|
| 1(6) | 51.3 | 50.2 | **-2%(sorted 손해)** |
| 8(48) | 76.3 | 91.2 | +19% |
| 16(96) | 99.2 | 136.6 | +38% |
| 24(144) | 104.1 | 156.5 | +50% |
| 32(192) | 108.7 | 170.3 | +57% |
| 48(288) | 106.8 | 187.1 | +75% |
| 64(384) | 109.6 | 224.3(2회 재현: 224.2/224.3) | **+105%** |

크로스오버는 B=1↔8 사이 — B=1(이 프로젝트의 KILL-GATE 기준점)이
유일하게 sorted가 손해 보는 지점(argsort 오버헤드가 선택 6개
규모에선 정렬 이득을 못 넘음).

**실측 이상치 1건, 근본원인 규명 없이 방치 안 함**: B=64 sorted 첫
측정이 108.9(unsorted와 통계적으로 동일)로 나와서 즉시 단독
재실행 → 224.3, 다시 224.2로 재현 — 직전 11회 연속 프로세스 실행의
시스템 부하/발열 상태였을 가능성이 유력, sorted 코드 경로 자체의
속성이 아님으로 판단(추가 계측 없이 방치하지 않고 명시적으로 기록).

**프로덕션 정책**(`qwen_infer.c`의 신규 `moe_gpu_sort_threshold(B,
top_k)`, `moe_baware_threshold()`의 기존 관례 그대로 미러): B<8은
unsorted 유지(B=1 손해 실측), B>=8은 항상 sorted(전 구간 이득
실측, B 커질수록 증가). `QWEN_MOE_GPU_SORT_THRESHOLD` env var로 향후
재측정 override 가능.

**정확도/determinism 완전 유지**: B=8~64 전부 여전히 flipped=0(worst_
rel_l2도 unsorted 때와 동일값), B=1 KILL-GATE 완전 무변화(이 게이트는
sort threshold를 아예 안 건드리므로 구조적으로 영향 없음).

**★★★ 결과: B=64가 목표(250)의 89.7%(이전 43.8%에서 상승), llama.cpp
자체(180.91)의 1.24배** — "llama.cpp에 짐"에서 "24% 앞섬"으로 이
한 번의 변경으로 역전. `bob:~/vdsp_m4_bench/qwen_infer_v5cfused_final`
갱신.

**상태**: V5d+F-4로 배치 decode 트랙 완결(정확도+determinism+처리량
전부 실측 검증). 남은 250 목표 대비 10.3% 격차는 이번 라운드에서
더 추적 안 함 — V5e(ragged multi-step decode)가 원 설계 의존순서상
다음 단계. 상세: `RESULTS.md` §"F-4: global sort/unsort implemented".

## Bug 1 근본원인 규명 + 수정 (같은 날 후속, 2026-08-31) — ★★★
## KILL-GATE 최종 통과, 21.78→~52.91 tok/s(+143%)

사용자 "Bug1 원인 계속 파봐" 지시로 착수. investigation-protocol
그대로 적용: 재현→경쟁가설→가설별증거→인과사슬→전후검증→기각가설
보고.

**재현 단계의 결정적 단서**: "1-eval-per-token" 실패 빌드를 2회 더
재실행 — **결과가 실행마다 다름**(1회차 2/8·pos1 rel_l2=0.30, 2회차
1/8·pos1 rel_l2=1.36, argmax는 둘 다 549로 같지만 오차크기는 다름).
진짜 산술버그라면 동일 입력에 매번 동일한 틀린 값이 나와야 정상 —
실행마다 다르다는 것 자체가 **미초기화/해제된 메모리를 읽고 있다는
강한 신호**, 코드를 건드리기 전에 가설공간을 재가중.

**가설 4개**:
1. **H_freqs(확정)**: `mlx_gpu_layer_step_lazy()`의 RoPE `freqs_f32`가
   함수-로컬 `std::vector<float> freqs_f`를 `noop_deleter`로 wrap(zero-copy,
   포인터만 참조) — eval이 함수 반환 전에 일어나야만 안전한데, "1-eval-
   per-token" 설계는 finalize()까지 전부 미룸(27개 레이어 호출 전부
   반환+스택해제 완료 후). RoPE 회전각=pos/freq이므로 pos=0은 freq값과
   무관하게 항상 0(항등회전) → "pos=0만 항상 맞고 pos≥1부터 비결정적으로
   틀림"을 정확히 예측 — 이번 재현+Bug1 원 보고(2026-08-30) 둘 다와
   정확히 일치.
2. **H_weights(기각)**: qwen_infer.c의 `run_moe_gpu_full_gate()` 호출부
   직접 확인(~6950줄) — w_inln/w_postln/w_kvaln/w_gate는 영속 mmap
   blob(`g_moe_f32_blob`) 포인터, x_embed는 루프 밖에서 1회 malloc —
   전부 프로그램 lifetime 동안 안전. 소스 직접 확인으로 기각(추정 아님).
3. **H_depth(기각, 수정으로 반증됨)**: MLX 그래프 실행기 자체가 27레이어
   전체를 아우르는 깊은 지연그래프를 못 다룰 것이라는 원래(2026-08-30)
   주력 가설 — freqs_f32 lifetime **하나만** 고쳤을 뿐 그래프 깊이/eval
   횟수는 전혀 안 건드렸는데 완전히 해결됨 → 그래프 깊이 자체는 문제가
   아니었음이 실측으로 반증됨.
4. **H_gfusedx(기각, 약한 가설)**: g_fused_x는 MLX 자체 그래프 노드
   참조카운팅이 관리하는 mx::array를 담고 있어(freqs_f32처럼 외부
   raw 포인터가 아님) 메커니즘상 안전 — freqs 수정만으로 해결된 결과가
   간접 확인.

**인과사슬**: `freqs_f`(함수-로컬 std::vector) → `freqs_f32`가 그
`.data()` 포인터만 wrap(복사 없음) → 함수 반환, 스택 해제 → 같은
스택영역을 다음 26개 레이어 호출의 자기 로컬변수들이 반복 재사용 →
`mlx_gpu_forward_finalize()`가 한참 뒤 첫 eval() 호출 → MLX가
freqs_f32 포인터로 그 시점 스택에 있는 아무 값이나 읽음(진짜 RoPE
주파수 아님) → 모든 레이어의 q_pe_rot/k_pe_rot이 쓰레기 주파수로
계산됨 → pos=0만 우연히 정상(항등회전), pos≥1부터 오염, 실행마다
스택 잔여값이 달라 결과도 비결정적.

**수정**: 영속 global `g_mla_yarn_freqs_f32`를 `mlx_gpu_mla_config()`에서
1회만 채우고, 함수-로컬 복사본 대신 이 버퍼를 wrap하도록 변경.
**2단계 검증**:
- Step A(수정만, 기존 레이어당-eval 설계 유지): 회귀없음 확인 —
  8/8·gpu_vs_cpu 완전동일(7.5e-07)·처리량 37.075(변화없음). 이 버그가
  기존 배포판에서도 "숨어있던"(가려진 것이지 없던 게 아닌) 버그였음을
  실증.
- Step B(수정+"1-eval-per-token" 재적용): **3회 반복 전부 8/8,
  gpu_vs_cpu 완전동일(7.5e-07, bit-identical), 처리량 52.809~52.996
  tok/s(평균 ~52.91)**.

**★★★ KILL-GATE 최종 통과**: 52.91 tok/s(평균) vs bar 48.34 — **bar의
109.4%**, 61.8 roofline의 85.6%(target 55, 89%엔 못 미침이나 kill-gate
자체는 통과). 이 라운드 전체 누적: eager 7.8 → V5c-fused(Bug3) 21.78
→ Win1(no lhs_indices) 30.0 → Win2(K/V slice_update) 37.16 →
Bug1수정(진짜 1-eval/token) **~52.91 tok/s**. 원 21.78 대비 **+143%**,
eager 대비 **~6.8배**. `qwen_infer.c` 전체 라운드 미변경(mlx_moe.cpp만).
`bob:~/vdsp_m4_bench/qwen_infer_v5cfused_final`이 이 최종본을 가리킴.

상세: `RESULTS.md` §"Bug 1 ROOT-CAUSED AND FIXED ... KILL-GATE PASSES".

**V5d(같은 날 후속, 배치 B-token GPU decode)**: `mlx_gpu_layer_step_lazy()`에
배치축 B 일반화(g_fused_K/V/x가 {B,...}). `gather_qmm`의 lhs_indices=nullopt
크로스곱 버그를 실사용 규모(B=8)에서 발견+수정(mlx_lm SwitchLinear의
expand_dims 컨벤션 필요, B=1에선 배치축 부재로 안 보였던 버그).
B∈{8,16,24,32,48,64} 전 구간 **flipped=0, determinism 확인**.

**F-4(V5d 직후)**: mlx_lm의 global sort/unsort(`_gather_sort`/`_scatter_unsort`)
이식, `B*top_k>=threshold` 런타임 분기. 실 27레이어 pass 내 크로스오버
재측정(격리벤치 수치 안 믿고): B=1 손해/B≥8 이득. **B=64 109.6→224.3
tok/s(+105%), llama.cpp(180.91) 대비 0.61x→1.24x로 역전**.

**V5e(재스코프 후 완료, ragged multi-step GPU decode)**: V5d의 lockstep
전제(모든 시퀀스가 같은 pos)를 깨는 설계 — `mlx_gpu_cbatch_layer_step_lazy()`가
A개 컬럼 각각 자기만의 (slot,pos)를 가짐, prefill과 decode를 동일 호출
형태로 통합(CPU 레퍼런스는 둘을 분리했지만 V5e는 하나로). 3개 MLX
프리미티브(scatter 2축 window-scatter, rope per-row offset array, sdpa
per-row mask) 전부 격리 probe로 실 dims 검증 후 적용 — **4번째 후속 probe에서
Bug1과 같은 부류의 새 함정 발견**: rope offset array가 H=1에선 맞았지만
H=16(실헤드수)에선 조용히 틀림(offset이 axis -2를 진짜 증분 시퀀스축으로
취급, 헤드마다 다른 회전 적용됨, max_abs_diff=2.54) → (A,H)를 한 축으로
flatten해 해결(max_abs_diff=0). 이 패턴은 `~/.claude/skills/
hw-kernel-vendoring/SKILL.md`에 일반화해 기록(사용자 제안).
실측 게이트(`QWEN_MOE_GPU_CBATCH=1`): 첫 실행에서 57개 전부 FLIP → CPU의
`moe_cbatch_step()` 기록 컨벤션 재확인 후 off-by-one 발견(생성 배열이
ref[k+1]과 대응, ref[0]은 절대 기록 안 되는 CPU 자체 컨벤션) → 수정 후
**56/56 정확도 일치**(57개 중 1개는 JSON 12토큰 캡처 상한 밖이라
구조적으로 검증불가, 명시적으로 로그됨 — 계획서의 "56/56 기대"와 정확히 일치).
메모리 active=9.982GB/peak=10.053GB (ceiling 12.71GB, V5a B=1 기준
9.814GB 대비 +0.24GB만 증가, K/V캐시가 가중치 대비 작은 비중이라 예상대로).
처리량은 참고용으로만 측정(11.256 tok/s, prefill+decode 통합이라 V5d의
decode-only 224.3과 직접비교 불가 — 정직하게 명시). **V5e 정확성
목표 완료**. 남은 것: 진짜 batched-causal prefill(한 시퀀스 N포지션을
한 마스크드 dispatch로), V5f(CPU vs GPU 최종 A/B 리포트).
상세: `RESULTS.md` §"V5e: ragged multi-step GPU decode -- CORRECTNESS
VERIFIED (56/56)".

**V5f(재스코프 없이 즉시 착수, 새 코드 불필요)**: 세션 최초의 CPU-vs-GPU
비교(llama.cpp 48.34 tok/s 1유저 vs CPU 2.47 tok/s 8동시유저)는 엔진도
동시성도 안 맞는 "정직하지만 통제안된" 비교였음 — 그 텍스트 자체가
"real controlled V5d/V5f-style comparison later"를 예고해뒀었음. V5e의
두 게이트(`QWEN_MOE_CBATCH=1` CPU / `QWEN_MOE_GPU_CBATCH=1` GPU)가 이미
완전히 동일한 8슬롯 워크로드를 쓰고 있어서, 같은 바이너리·같은 머신·
같은 102토큰(45prefill+57decode)으로 3회씩 재측정만 하면 됐음(신규
코드 없음, F-4의 이상치 탐지 규율 그대로 3회 반복).
**CPU 평균 2.771 tok/s(2.60~2.93), GPU 평균 11.866 tok/s(11.52~12.49)
— GPU가 ~4.3배(3.9~4.8배 범위) 우세**, 사상 최초의 진짜 통제 비교.
**격차가 V5d/F-4의 B=64 수치(224.3 tok/s)보다 훨씬 작은 이유도
얼버무리지 않고 설명**: V5e는 정확성 우선 설계라 45개 prefill 포지션
전부가 decode 스텝과 동일 비용(포지션당 MLX eval-graph 1회)으로
처리됨 — 진짜 batched-causal prefill(한 시퀀스 N포지션을 한 dispatch로)
이 계획서에서 명시적으로 제외됐던 바로 그 부분이라, GPU의 진짜
배치우위가 이 워크로드에서 최선의 케이스로 발휘를 못 함. 남은
격차 대부분이 거기 있을 것으로 판단(확정 아님). **V5f 완료**.
상세: `RESULTS.md` §"V5f: CPU vs GPU A/B, matched workload".

**V5g(재스코프 후 완료, 진짜 batched-causal prefill)**: mlx_moe.cpp/h
변경 전혀 없이(신규 primitive/probe 0건) — `mlx_gpu_cbatch_layer_step_
lazy()`의 기존 per-row causal mask(row m이 j<=spos[m] 참조)가 spos[m]=
그 row 자신의 시퀀스내 위치일 때 이미 정확한 하삼각 causal mask임을
재도출, A=45(8슬롯 전체 prompt 위치 합)로 한 번에 배치 prefill 실행.
`run_moe_gpu_cbatch_prefill_gate()`(`QWEN_MOE_GPU_CBATCH_PREFILL=1`)
신규 추가, 기존 `run_moe_gpu_cbatch_gate()`는 그대로 보존(V5f 기준
재현성 유지, Rule 3).
**첫 실측에서 정확도는 즉시 56/56이었지만 처리량이 V5f와 거의
동일(11.15~11.97)해서 "무의미한 변화"처럼 보였음** — 추측으로
넘어가지 않고 prefill 스텝만 별도 계측 → **A=45 dispatch 단독이
7407ms(전체 8600ms의 86%)**, decode 12스텝(57토큰)은 겨우 1192ms.
근본원인: 이 프로세스가 A=45 shape을 처음 만나면서 MLX/Metal JIT
컴파일 비용을 지불 중 — 기존 `run_moe_gpu_batch_gate()`가 이미
`warmup_steps=4`로 방어하던 바로 그 부류의 비용을 이 신규 게이트만
빠뜨렸던 것. warmup pass(같은 A=45, 미계측, 버림) 추가 후 재측정 →
**prefill 단독 7407ms→421.80ms(17.6배 감소)**, 가설을 직접 실험으로
확정. **최종(warm, 4회): 평균 86.517 tok/s**(81.99~93.88) — **V5f GPU
기준(11.866) 대비 7.3배, CPU(2.771) 대비 ~31배**, V5f가 예측한 격차의
거의 전부를 회수. decode 스케줄도 이제 CPU 레퍼런스와 완전히 동일
(`8,8,8,7,6,5,4,4,3,2,1,1`, 8슬롯이 동시에 decode 진입하므로).
정확도 56/56 warmup 전후 모두 확정 유지(warmup의 K/V 오염은 실제
prefill의 동일 좌표 덮어쓰기로 자동 해소되는 설계). V5a~g 표준계획
전체 COMPLETE.
상세: `RESULTS.md` §"V5g: true batched-causal prefill".

**V5h(재스코프 후 완료, GPU 온라인/동적 admission 스케줄러)**: V5a~g는
전부 8슬롯을 up-front로 전량 admit하는 고정 워크로드 — 실서빙 형태
아님. CPU엔 이미 있던 MoE-4b 온라인 스케줄러(요청테이블+FIFO
step-indexed arrival+슬롯재사용+budget-chunked prefill, 2026-08-24
Gate1-8 검증완료)를 GPU에 최초 이식. mlx_moe.cpp/h 변경 전혀 없이(V5g의
"임의 (slot,pos) 행 조합은 이미 correct" 통찰을 decode+prefill 혼합
행에도 그대로 적용) 순수 C 스케줄링 로직만 포팅.
`run_moe_gpu_cbatch_online_gate()`(`QWEN_MOE_GPU_CBATCH_ONLINE=1`)
신규, CPU와 동일 env var(`QWEN_MOE_CB_SLOTS/REQS/PREFILL_BUDGET/ARRIVE/
STOP_EXTRA`) 재사용해 동일 워크로드 재현 가능(하드코딩 중복 없음).
CPU의 D3 invariant(한 스텝 내 같은 슬롯은 spos 오름차순)는 CPU 고유의
"쓰고 바로 같은 루프서 읽는" 순차 hazard 때문 — GPU는 scatter/take가
MLX 그래프 의존성으로 순서 보장돼서 이 제약이 원래 불필요함(기록해둠).
V5g의 웜업 교훈을 일반화: 스텝마다 shape이 다르므로 전체 시뮬레이션을
2-pass(1회 미계측 버림+1회 실측)로 실행 — 결정론적이라 안전.
**검증**: CPU 온라인 스케줄러(PREFILL_MODE=0, 정확도 95.3% 기준)를
ground truth로 삼아 B=8/R=16 기본 arrival + staggered arrival(0×8,5×4,
10×4) 두 워크로드에서 **32/32 요청·228/228 토큰 완전 일치**(프로그램적
diff), invariant 위반 0건, arrival-gate 위반 0건(admit_step이 두
엔진 모두 arrival 조건 만족). admit_step 절대값은 두 엔진이 살짝
다름(CPU mode0=admission시 동기 전량prefill, GPU=budget-chunked라
mode1 구조 미러 — 스케줄 전략 차이지 버그 아님, 명시적으로 설명).
**처리량**: GPU 평균 **99.582 tok/s**(99.556~99.620, ~0.03% 편차 —
이번 세션 전체 게이트 중 최소 변동폭, 2-pass 웜업이 전체 shape을
미리 컴파일해서로 추정). CPU mode0(정확도기준, 의도적으로 느림)
평균 1.607 tok/s → **GPU ~62배**. CPU의 실제 서빙용 모드(mode1,
SME2배칭, 정확도 80%) 2.793 tok/s → **GPU ~35.7배**(더 대표성있는
비교로 병기, 큰 숫자만 내세우지 않음). V5a~h 트랙 전체 COMPLETE.
상세: `RESULTS.md` §"V5h: GPU online/dynamic admission scheduler".

**V5i(재스코프 후 완료, GQA 모델 일반화 — MLA 전용이던 GPU 경로에
DeepSeek이 아닌 모델 최초 이식)**: "다음은 일반화?"에 사용자가
"GQA 모델 일반화"로 확정. 착수 전 리서치(3개 병렬 Explore agent)로
스코프가 예상보다 훨씬 큼을 발견 — Qwen3-30B-A3B(기존 CPU측 GQA
모델)는 GGUF Q4_K_M 18.5GB가 bob 12.71GB 워킹셋 상한을 46% 초과해
**하드웨어 원천 불가**(이 프로젝트 기존 결론 재확인, 재작업 문제가
아님), 기존 AF export도 디스크에서 사라짐(deleted 디렉터리로의 깨진
심볼릭링크). OLMoE(~7.5GB)만 하드웨어로 가능한데 AF export가 아예
없었음(원본 HF safetensors뿐). AskUserQuestion으로 "Phase A(export)만/
Phase A+B(GPU attention까지) 통합/방향전환" 중 확인 → **"전체를 한
번에" 선택**, 두 phase 통합 계획으로 승인.

**Phase A(export)**: `mlx_olmoe_gqa_selftest_export.py`(macstudio) —
기존 DeepSeek exporter+Qwen3 selftest 파일포맷의 Rule-3 미러, layer-0
attention-only로 스코프 축소(B=1이 이번 라운드 실목표라 16레이어
전체export 불필요 — 구현 중 자체판단, "승인된 트랙 내 재확인 불필요"
원칙대로 사용자 재컨펌 없이 진행). `mlx-community/OLMoE-1B-7B-0125-4bit`
(WebSearch/WebFetch로 존재 사전확인) 다운로드 49분 소요.
**재확인한 기존버그 2건**(과거 CPU측 OLMoE 작업에서 이미 문서화됐던
것, 이번에 재검증만): (1) vocab_size=50304라 DeepSeek 기본 프롬프트ID
(100000+)가 범위밖 — export가 작은 ID`[100,549,4345]` 사용. (2)
**핵심**: OLMoE q_norm/k_norm이 reshape 전 전체벡터를 정규화(Qwen3-MoE의
per-head 컨벤션과 다름) — `mlx_lm.models.olmoe` 소스 직접 재확인
(`self.q_norm=nn.RMSNorm(n_heads*head_dim,...)`, reshape 전 적용).
기존 CPU측 `MOE_QKNORM_WHOLE_VECTOR`가 이미 정확히 처리.
신규 CPU 진입점 `run_moe_gqa_olmoe_selftest_mode()`(`QWEN_MOE_GQA_
OLMOE_SELFTEST=<dir>`) — 기존 Qwen3용 `run_moe_gqa_selftest_mode()`의
형제함수(같은 파일포맷, 다른 하드코딩 config), 기존 함수를 고치는 게
아니라 새로 추가(이 파일의 기존 "같은shape함수, 다른실모델config"
관례 그대로). **검증**: CPU의 기존 검증된 `moe_gqa_attention()`이
export의 실 MLX reference와 rel_l2 1.5~2.3e-07(float32 노이즈 수준)
일치, 3개 position 전부. Phase A 완료.

**Phase B(GPU attention, B=1)**: 격리 probe 먼저(이 세션 전체의
불변규율) — NeoX rope(`traditional=false`)는 mlx_moe.cpp에서 한 번도
안 쓰인 회전컨벤션(기존 전부 traditional=true/MLA interleaved+YaRN).
`probe_neox_rope.cpp`, 실 OLMoE 차원(H=16,HEAD_DIM=128,theta=10000)에서
`max_abs_diff=1.01327896e-06`(V5b의 traditional=true probe 4.77e-07
대비 한 자릿수 크지만 여전히 float32 반올림 노이즈 수준 — 진짜
컨벤션 불일치였다면 O(1) 발산이지 1e-6이 아님, pass로 판정).
`mlx_gpu_gqa_config()`+`mlx_gpu_gqa_layer0()`(mlx_moe.cpp/.h) 신규
추가 — 기존 `mlx_gpu_mla_config()`/`mlx_gpu_mla_layer0()`의 런타임
분기가 아니라 **완전히 별도 함수**(CPU 레퍼런스의 기존 설계원칙
그대로: MLA/GQA는 프로젝션 구조 자체가 다름). Plain q/k/v/o_proj(LoRA
없음), whole-vector qknorm, NeoX rope(단일position이라 V5e의
array-offset 트릭 불필요, scalar offset=pos), GQA head-grouping은
sdpa K/V 버퍼 staging시 explicit host-side broadcast로 구현(MLX sdpa의
암묵적 GQA지원에 의존 안 함). 별도 K/V캐시(`g_gqa_K/V`).
신규 CPU 드라이버 `run_moe_gqa_gpu_gate()`(`QWEN_MOE_GQA_OLMOE_GPU=
<dir>`) — `run_moe_gqa_olmoe_selftest_mode()`의 구조적 미러, attention만
GPU 경로로 교체.
**B=1 게이트, 3-way 교차검증**(GPU vs CPU vs 실 MLX reference, GPU-vs-
CPU만이 아님): 3개 position 전부 rel_l2 1.5e-07~2.6e-07 범위 —
GPU가 CPU와 가까운 정도가 아니라 **실모델 MLX 출력 자체를 CPU와
동일한 정밀도로 독립재현**. V5a/V5b 원래 게이트기준(rel-L2≤1e-5) 대비
압도적 여유로 PASS. **V5i는 B=1/layer-0 스코프에서 COMPLETE** —
멀티레이어 전체모델(V5c급)/batched-ragged-online(V5d-h급) OLMoE GPU
일반화, Qwen3-30B-A3B GPU(하드웨어로 원천배제)는 명시적으로 다음
라운드 스코프.
상세: `RESULTS.md` §"V5i: GQA (Grouped Query Attention) model
generalization on the GPU path".

**V5j(완료, GQA 멀티레이어 전체모델 GPU forward — V5i의 V5c급 후속)**:
"다음은?" 질문에 사용자가 "멀티레이어로 확장" 확정 — MLA 트랙의
V5b(layer-0)→V5c(27레이어+kill-gate) 패턴 그대로. 리서치(Explore
3개+Plan검증 1개 병렬)로 예상보다 작은 스코프 확인: CPU는 이미 검증된
generic 멀티레이어 OLMoE forward(`run_moe_safetensors_verify_mode()`)
보유, GPU 포스트어텐션 블록(라우터/switch_mlp/shared_experts/residual)은
`g_mla_*` 참조 0건으로 verbatim 재사용 확인, MLX primitive 전부 B=1
라운드에서 이미 검증됨 — 신규 필요한 건 풀 export+실 reference
capture+lazy-graph GQA broadcast뿐.

**Phase C(export+CPU검증)**: C1 — `MOE_QKNORM_WHOLE_VECTOR`를
arch_config_moe.txt 기반 8개 config loader 전부에 추가(순수 additive,
기존 DeepSeek kill-gate 8/8 argmax·동일 rel_l2·52.244 tok/s로
byte-identical 재확인). C2 — `mlx_olmoe_full_to_q4g64af.py`(DeepSeek
exporter fork, GQA attention tensors+역양자화 라우터+fp32-upcast fix,
dense/shared_experts 브랜치 없음) — **첫 실행에서 진짜 버그**:
`post_attention_layernorm.weight`를 통째로 빠뜨려 FATAL, 확인 후 수정
재실행(4.323GB AF/8.92MB F32, 114/81 텐서, 공식과 정확히 일치). C3 —
`olmoe_reference_capture.py`, 실 텍스트 13토큰 teacher-forced. C4 —
`run_moe_verify_mode()`(신규 C코드 불필요) 대비 **13/13 argmax
일치**, worst rel_l2 1.106e-02.

**Phase D(GPU 멀티레이어+검증)**: D1 — B=1 eager GQA의 head-broadcast가
host-side memcpy라 lazy 그래프에 그대로 이식하면 host round-trip 강제
(MLA의 Bug1/3과 같은 부류) → device-side reshape+broadcast_to+reshape
메커니즘을 synthetic group=4에서 검증, **max_abs_diff=0.0**. D2-D4 —
`g_fused_gqa_K/V`(GQA 전용 캐시)+`ensure_fused_gqa_kv_init()`, 신규
`mlx_gpu_gqa_layer_step_lazy()`(D1 검증된 broadcast+기존 post-attention
블록 verbatim 재사용), `mlx_gpu_gqa_forward_finalize()`(계획단계에서
미리 발견한 g_mla_rms_eps 네이밍누수를 실버그로 만나기 전에 수정).
D5 — `run_moe_gpu_gqa_fused_gate()`(`run_moe_gpu_fused_gate()`엔 없는
GQA-aware KROW/VROW 분기 포함) — **GPU vs 실 MLX ground truth 13/13
argmax, rel_l2 5.049e-03~1.106e-02**, CPU 자체 정밀도 대역과 거의 동일.

**원인규명하다 만 진짜 이상현상(얼버무리지 않고 기록)**: 이 인터리브드
드라이버 내부에서 계산된 CPU logits가 C4의 깨끗한 standalone 출력과
diverge — position 0은 byte-identical(rel_l2가 C4의 pos0 값과 정확히
일치)이지만 position 1부터 divergence가 커짐(최대 26%, argmax flip
2건). **가설 2개 반증**: (1) 일반 CPU 비결정성 — 반증(C4 standalone
2회 `cmp` byte-identical), (2) GPU/CPU 타이밍 중첩에 의한 64스레드풀
race — 반증(`QWEN_MOE_SCALAR_THREADS=1` 강제해도 완전히 동일한
divergent 수치 재현), (3) `routing_out=NULL` 부작용 — 반증(소스 확인,
진단용 fprintf 전용, 계산에 영향 없음). position0-clean/position1-부터
오염 패턴은 CPU의 K/V 히스토리 배열이 position0 write와 position1
read 사이 뭔가에 영향받는다는 뜻이지만, 이번 라운드 신규 코드의 어떤
포인터도 그 배열과 aliasing되는 지점을 찾지 못함 — **근본원인 이번
라운드 예산 내 미확정**. 라운드의 주요 정확성 주장(GPU vs 실 MLX
truth)에는 영향 없음(C4 standalone과 D5 GPU-vs-truth 양쪽에서 독립
확인됨) — 다음에 인터리브드 CPU+GPU GQA 호출을 건드릴 사람을 위한
후속조사 항목으로 명시.

**처리량**(관측치만, 외부 baseline 주장 없음 — bob에 llama.cpp OLMoE
지원은 이미 컴파일돼있지만 이 프로젝트에서 벤치마크된 적 없음): 3회
107.061/106.824/106.914 tok/s(평균 106.93, ~0.1% 편차 — 정확도 수치의
완전한 재현성과 마찬가지로 타이트함). **V5j COMPLETE**. batched/ragged/
online GQA(V5d-h급), 실 llama.cpp 기준선, 위 CPU cross-check 이상현상
근본원인규명 전부 다음 라운드 오픈 항목.
상세: `RESULTS.md` §"V5j: full multi-layer GQA GPU forward (OLMoE)".

★V5l(V5j-ragged Phase D 온라인 admission 스케줄러에 실 프롬프트 연결,
V5k 완료 직후 사용자 요청): 스코프 GQA만(V5k와 동일한 "GQA먼저, MLA는
다음라운드" 패턴). Plan Mode로 진행 — Explore agent로 MLA쪽 온라인게이트
구조 확인(GQA와 달리 corpus테이블이 file-scope 아닌 함수로컬
`static const`, EOS도 `MOE_EOS_TOKEN_ID` 안읽고 100001 하드코딩 —
다음라운드 메모), Plan agent로 실코드(`moe_cb4b_admit_guard()` 본문 등)
직접읽고 설계 검증. ★설계검증 중 진짜 버그 발견: guard가
`plen+maxnew<=MOE_CBATCH_MAXPOS(32)`만 체크하고 `rq_out`의 실제 선언폭
`MOE_CBATCH_KNEW(12)`는 전혀 모름 — 기존 8개 corpus의 `moe_cbatch_gen[]`이
우연히 전부 12이하라 지금까지 안 드러난 잠재 버퍼오버플로우, manifest로
임의 maxnew를 넣을 수 있게 되면서 새로 노출될 위험이었음. 설계:
`QWEN_MOE_CB_PROMPT_MANIFEST=<path>`(`<i32경로> <max_new_tokens>` 한줄씩,
`load_ids()` 재사용) — corpus/manifest 소스를 로드시점에 `mf_plen/mf_maxnew/
mf_ids` 단일테이블로 통합해(핫루프 3곳 전부 이 테이블만 참조, 분기는
한곳뿐) manifest 미설정시 오늘과 완전동일값 보장 + `rq_out` 폭을
`MOE_CBATCH_MAXPOS(32)`로 확장(guard의 실제 상한과 일치시켜 버그 근본해결,
GQA GPU게이트 로컬선언 1곳만, 동일패턴 다른 3곳(MLA-CPU원본/GQA-CPU-twin/
MLA-GPU)은 미변경). 구현은 Edit string-matching이 4개 미러링함수 중
모호하게 여러곳 매칭되는 문제를 겪어 line-number기반 `sed`로 정밀치환.
**검증 5개 전부 통과**: (1) git diff 전량 `#ifdef QWEN_GPU_MLX`내부
(2) manifest미설정시 wall-clock필드 제외 stderr 완전동일(diff empty)
(3) 기존8개를 manifest로 재현해도 완전동일 (4) MC=10(기존8+실프롬프트의
진짜 부분수열 2개, "새 텍스트 토큰화"는 이 엔진에 토크나이저가 없어
불가하므로 실제 corpus 텍스트의 실제 prefix를 사용) + R=20으로
wraparound 유발 — 첫등장·재등장 전부 V5k 단일시퀀스게이트 ground truth와
정확히 일치 (5) ★rq_out수정을 ASan으로 실증: 폭12로 되돌린 사본을
`-fsanitize=address`로 컴파일, `R=64,B=8,전요청 plen=1/maxnew=31`(guard가
허용하는 최악케이스)로 실행 → 정확히 예측지점(`rq_out[r][rq_nout[r]++]`)에서
진짜 global-buffer-overflow 실측(ASan: "0 bytes after global variable
...rq_out") — 폭32 수정판은 동일 ASan+동일 최악케이스에서 64개 요청 전부
클린 통과. 이론적 우려가 아니라 실재 버그였음을 최고강도로 증명.
상세: `RESULTS.md` §"V5l: real-prompt manifest for the GQA online
admission scheduler".

★V5l MLA 대응(GQA 완료 직후 "1번 진행" 사용자 승인으로 바로 이어서
진행): 동일 매니페스트 메커니즘을 `run_moe_gpu_cbatch_online_gate()`
(DeepSeek/MLA 온라인게이트)에 이식. GQA 라운드가 남긴 두 갭도 이번에
같이 닫음 — (1) corpus 테이블이 GQA와 달리 애초에 함수로컬
`static const`라 file-scope로 끌어올릴 필요 자체가 없었음(else-branch가
그냥 로컬 const를 복사) (2) EOS가 두 eviction 체크 지점 모두
`100001` 하드코딩이었던 걸 `MOE_EOS_TOKEN_ID`(config-driven, V5k Phase
1b와 동일 관례)로 교체 — 실제 값은 100001로 동일하므로 오늘 기준
동작변화 없는 순수 정확성 정리. 매니페스트 로더 함수도
`moe_gqa_cbatch_load_manifest`→`moe_cbatch_load_manifest`로 개명해
GQA/MLA 양쪽이 그대로 재사용(본체에 GQA 특정 로직이 전혀 없어서
~40줄 중복 방지, `moe_load_gqa_cbatch_config()` 공유헬퍼 선례와 동일
판단). **검증 5개 전부 통과, GQA때와 동일 프로토콜**: (1) git diff
전량 `#ifdef QWEN_GPU_MLX`내부(9648~10642) (2) manifest미설정시
wall-clock필드 제외 stderr 완전동일(EOS 하드코딩→config전환이 진짜
no-op임을 이 스텝이 직접증명) (3) 기존 8개 실 DeepSeek 프롬프트를
manifest로 재현해도 완전동일 (4) MC=10(기존8+실프롬프트 부분수열2) +
R=20 wraparound — 첫등장·재등장 전부 V5k MLA 단일시퀀스게이트 ground
truth와 정확히 일치 (5) ★rq_out ASan 실증: pre-fix 사본이 GQA와
**정확히 같은 지점 형태**(`run_moe_gpu_cbatch_online_gate.rq_out`, "0
bytes after global variable")에서 재현, post-fix는 동일 최악케이스
(R=64/B=8/plen=1/maxnew=31) 64개 요청 전부 클린. **V5l는 이제
GQA+MLA 두 토폴로지 모두 COMPLETE** — V5k와 똑같은 "GQA 먼저, MLA
즉시 미러링" 패턴으로 온라인 스케줄러 실프롬프트 지원도 두 토폴로지
동시완결. 상세: `RESULTS.md` §"V5l MLA mirror: real-prompt manifest
for the MLA online admission scheduler".

★V5l GQA CPU twin(온라인스케줄러 CPU ground-truth `run_moe_gqa_
cbatch_online_cpu_gate()`에도 동일 매니페스트, 사용자가 명시 요청:
"지금 매니페스트 워크로드는 V5k 단일시퀀스 게이트로만 대조 가능하고,
CPU ground-truth 동시 대조 수단이 없음"): ★진짜 컴파일 에러로 설계
결함 발견 — `moe_cbatch_load_manifest()`가 원래 GQA-CPU-twin과
GQA-GPU-게이트 "사이"에 정의돼있어서 CPU twin(더 앞에 선언)에서
호출하면 C에서 암묵적 선언→나중 static 정의 충돌 에러. 로더 함수
자체를 CPU twin보다 앞으로 물리 이동시켜 해결(향후 MLA CPU twin도
같은 위치에서 볼 수 있음). 로더 내부 로그태그도 GPU전용이던
`[moe gpu gqa cb online]`을 토폴로지중립 `[moe cbatch manifest]`로
정정. **검증 5개 전부 통과, 이번엔 (4)가 질적으로 다름**: 이전
GQA/MLA GPU 라운드는 V5k 단일시퀀스 게이트와만 대조했지만, 이번엔
**GQA GPU 게이트의 이전 라운드 MC=10/R=20 실행결과와 직접 대조** —
req/prompt/slot/admit_step/tokens/steps=37/admitted_after_evict=16/
queue_wait_events=16/queue_wait_max_steps=31 전부 완전일치. 이게
이번 라운드의 진짜 목적(온라인 스케줄러 자체의 CPU=GPU 증명, 고정
corpus가 아닌 임의 매니페스트 워크로드에서). (5) ASan도 CPU twin
자체 함수범위에서 재현+수정확인(GPU와 동일 지점 형태). ★CPU 스칼라
경로가 GPU보다 압도적으로 느림을 재확인(R=64 최악케이스 tok/s=1.833
vs GPU 214.7 — 기존 세션 기록(V5f/V5h)과 일치하는 이미 알려진
특성, 이번 변경의 부작용 아님). 다음 유일 남은 갭: MLA CPU twin
(`run_moe_cbatch_verify_mode()`의 online분기). 상세: `RESULTS.md`
§"V5l GQA CPU twin: manifest support for the CPU ground-truth
scheduler".

★V5l MLA CPU twin(마지막 갭, 사용자 명시요청 "MLA CPU twin에도
매니페스트 이식"): 대상은 `run_moe_cbatch_verify_mode()`(MoE-4a/4b/4c
통합함수, static+online+reverify 전부 한 함수)의 online 브랜치.
★구조적으로 이전 3라운드와 다른 발견: 이 함수(그리고 그 앞으로
옮겨야 했던 로더)가 **`#ifdef QWEN_GPU_MLX` 가드 밖**에 있음(6000줄
이전엔 헤더 include용 가드 1개뿐, 이 함수는 4999에서 시작) — 즉
"GPU 빌드에만 영향, dense 기본 빌드는 완전 무관"이라는 이전
3라운드의 안전장치가 이번엔 구조적으로 성립 안 함. dense-only
컴파일(`-DQWEN_GPU_MLX` 없이)을 직접 실행해 클린 확인. 로더 함수를
**두 번째로** 재이동(GQA-CPU-twin 앞→이번엔 이 함수 앞, 파일에서
가장 이른 위치). EOS는 이번엔 손대지 않기로 판단(이 함수는 config를
스스로 안 읽고 caller가 이미 로드했다는 설계 — EOS_TOKEN_ID 하나를
위해 이 원칙 깨고 파일 재오픈하는 건 스코프 확장, 값(100001) 자체는
이미 실측재확인+이 함수가 DeepSeek 전용 리터럴이라 실질 안전 —
정확성 문제 없이 문서에만 명시). **검증 5개 전부 통과**, (4)가
GQA CPU twin과 마찬가지로 GPU게이트 실제결과와 직접대조(steps=37/
admitted_after_evict=16/queue_wait_events=16/queue_wait_max_steps=31
전부일치), (5) ASan pre-fix 재현(`run_moe_cbatch_verify_mode.rq_out`)
+post-fix 클린(R=64 전부). **V5l `QWEN_MOE_CB_PROMPT_MANIFEST`가 이제
4개 온라인게이트(GQA GPU/GQA CPU/MLA GPU/MLA CPU) 전부 커버 —
CPU/GPU × GQA/MLA 매트릭스 완결**. 남은 공통갭: `load_ids()` 입력검증
(4곳 전부 미해결). 상세: `RESULTS.md` §"V5l MLA CPU twin: manifest
support for the MLA CPU ground-truth scheduler".

★V5l `load_ids()` 입력검증(4개 게이트 전부, 사용자 명시요청 "이것도
진행하자"): 4개 게이트가 전부 공유 로더(`moe_cbatch_load_manifest()`)
를 통하므로 **한 곳만 수정하면 4곳 전부 적용** — `load_ids()` 반환
직후 각 토큰이 `0 <= id < MOE_VOCAB`(호출측 게이트가 이미 로드한
실제 vocab 크기)인지 검사, 위반시 파일/인덱스/값 명시하며 FATAL.
완벽한 매직넘버 검증은 불가능(임의 파일도 우연히 vocab범위 안 정수로
읽힐 수 있음)하지만, 포맷이 아예 다른 파일(텍스트/다른 바이너리/
truncated)은 거의 확실히 초반 토큰에서 범위밖 값이 나와 실무적으로
충분한 방어선. **검증**: (1) dense-only+GPU 빌드 둘다 클린 (2) 4개
게이트 전부 기존 유효 매니페스트로 재실행 — timing필드(`tok/s=`
누락분 발견해 필터 보완) 제외 완전동일, 새 체크가 정상 매니페스트엔
진짜 no-op임을 확인 (3) 양성테스트(GQA GPU게이트, 로더가 동일코드라
나머지 3곳 대표): 범위밖 토큰(999999, 실제 MOE_VOCAB=50304 로그로
직접확인)+음수토큰(-5) 둘 다 GPU작업 시작 전에 정확히 FATAL 트리거
확인. **V5l 매니페스트 기능의 유일 남은 공통갭 해소 — 4라운드+1
보강으로 완전히 마무리**. 상세: `RESULTS.md` §"V5l: manifest loader
input validation".

★★라우터 근접-동점(margin) 통계 프로파일러(사용자 제안 "정밀도 각개
승격" 아이디어의 측정 단계): 이 프로젝트가 이미 한 번 실제로 겪은
사건(OLMoE 레이어13/pos0 라우터 플립, gap=1.87e-05, attention F32
승격으로 실제 수정 완료됨, `qwen_infer.c:12010-12029` 주석)을
반복측정 도구로 일반화. `moe_router_margin_profiler.py`(macstudio
전용, repo에 안 넣음, 기존 `moe_st_expert_profiler.py` 훅 방식
차용) — OLMoE의 `mlp.gate`는 순수 `nn.Linear`라(전용 Gate 모듈 아님)
`OlmoeSparseMoeBlock.__call__` 자체를 몽키패치해 top-k 선택에 실제
쓰이는 raw routing_weights를 옆에서 재계산(실제 forward엔 영향 없음),
매 (layer,token) 라우터콜마다 k번째 선택expert vs (k+1)번째 탈락
expert의 margin 기록. **실측(30프롬프트/642토큰/16레이어/10272
라우터콜)**: 전체분포 p1=0.000015, median=0.001694, **min=0.000000
(진짜 완전동점 이벤트 다수 관측)**. 레이어13 자체 p1(0.000031)이
실제 사건 gap(1.87e-05)과 같은 자릿수 — 도구가 실제 위험지점을
재발견함을 확인(타당성 검증). **의외 발견**: 가장 위험한 건 후반
레이어가 아니라 **초반 레이어(0-4)** — layer1 median이 layer15보다
4.3배 더 촘촘함. 이미 고친 layer13 사건은 이 순위에서 중간 정도이지
최악이 아님 → **0-4레이어에 아직 못 찾은 미발견 사건이 있을 가능성
높음**(기존 조사는 pos0 한 지점 발산으로 우연히 트리거된 것, 체계적
스윕 아니었음). OLMoE는 dense-layer/shared-expert가 아예 없어서
(`first_k_dense_replace=0`/`n_shared_experts=0`) `allow_f32=0` 게이트
제약은 이번엔 무관 — DeepSeek-V2-Lite로 포팅할 때나 해당. **이번
라운드는 측정까지만**: 승격 후 실제로 플립이 줄어드는지 인과검증
(layer13 사건 때처럼 before/after 실측)과 런타임 자동 적응형 승격
(4→8→32bit, 16bit는 `QWEN_MOE_ROLE_BITS`가 지원 안 함) 비전은 다음
라운드. 전체 데이터: macstudio
`/Users/eoe/vdsp_olmoe_full_weights/moe_router_margin.json`. 상세:
`RESULTS.md` §"Router near-tie (margin) statistics profiler".

★★재현검증(완전 다른 주제 30프롬프트, 겹침 0%)으로 사용자 요청
("ㅇㅇ")에 응해 실행 — 처음 가설보다 더 정교한 결론. layer12(51,55)
쌍은 **진짜로 재현됨**(독립 코퍼스에서도 3/811회, margin
7.6e-6~6.1e-5 — 같은 자릿수) → 이 쌍은 우연이 아니라 실재하는
구조적 약점. 하지만 **"가장 위험한 쌍"은 아니었음** — round2
자체의 top50 closest에는 0/50(각기 다른 일회성 쌍들이 더 촘촘했음),
게다가 layer12 자체는 round2에서 median 기준 **가장 느슨한
레이어**(0.003418, round1 때 "중간위험"이라던 것과 배치) — 즉
"layer12가 위험"이 아니라 "layer12 안의 이 특정 쌍 하나만" 콕
집어 촘촘한 것. **훨씬 강하게 재현된 신호는 레이어 단위 패턴**:
초반 레이어(1-4)가 두 독립 코퍼스에서 공통으로 가장 촘촘함
(layer1 median 0.000755 vs 0.000748 — 거의 동일). **결론 수정**:
특정 expert쌍 하나를 쫓기보다 "왜 초반 레이어가 구조적으로 더
촘촘한가"가 더 근거있고 실행가능한 다음 조사 대상. 사용자의
"항상 존재하는 취약 쌍" 가설은 부분적으로만 맞음(존재는 하지만
지배적 원인은 아님) — 사용자에게 이 뉘앙스 그대로 보고 예정.
전체 데이터: macstudio
`/Users/eoe/vdsp_olmoe_full_weights/moe_router_margin_v2.json`. 상세:
`RESULTS.md` §"Router near-tie profiler: cross-corpus replication".

★사용자 지적("layer13 F32 승격이 진짜 최소 필요 정밀도였는지 검증
안 됨 — 4/8/16/32 각각 어디서 붕괴하는지 스윕했어야, 추론엔진도
'학습/캘리브레이션'이 필요한 상태로 정의하고 싶음")을 받아 코드
아님, `ROADMAP.md`에 `D-roadmap-2`로 문서화(WHY/COST/EXIT 형식,
기존 D-roadmap-1과 동일 관례). 핵심: 지금까지의 모든 정밀도 수정
(OLMoE attention F32, DeepSeek profiler-driven int8 승격)이 전부
"문제발견→가장 안전한 티어로 즉시 점프"였지 실제 붕괴지점을 스윕한
적 없음 — margin profiler는 "위험한가"만 답하지 "최소 몇 비트면
충분한가"는 안 답함. 엔진에 이미 있는 bit-width 사다리
(`QWEN_MOE_ROLE_BITS`/`QWEN_MOE_EXPERT_BITS`)를 실제로 스윕하는
루프가 빠진 부분. **아직 착수 안 함, 의도와 갭만 기록** — 사용자가
"일단 적어놓고 다음 단계를 정할게"라고 명시, 코드/스윕 하네스는
다음 단계 결정 후.

★★★D-roadmap-2 Track A **완료+실측 검증**: `qwen_infer.c`에 16비트
(F16) 티어 추가 — `st_register_moe_f16_as_af()`/`moe_decode_af()`·
`moe_matvec_af_row()`의 bits==16 분기/`moe_load_role_bits()`+
`QWEN_MOE_EXPERT_BITS` FATAL체크 확장/`st_register_moe_role()`
allow_f32게이트 재사용/`st_register_moe_experts_mixed_as()` 전문가별
경로까지 전부 미러링. **검증 순서**: (1) bob에서 `_Float16` 왕복
독립 프로브(실제값+overflow-to-inf+subnormal 전부 정상) (2)
dense-only+GPU매크로 빌드 둘다 클린 (3) **실제 OLMoE 체크포인트로
end-to-end 링크+실행**(`/Users/bob/olmoe_1b7b_hf`,
`QWEN_MOE_SAFETENSORS`+`run_moe_safetensors_verify_mode()`) — baseline
8/8 정상, `q_proj -1 16` 승격도 8/8 정상+argmax 불변. **핵심 결과**:
`q_proj -1 16`과 `q_proj -1 32`가 8개 포지션 전부 **완전히 동일한
logit** 산출 — 16비트가 이미 32비트와 같은 정밀도를 이 role에서
확보함을 실측 확인, D-roadmap-2가 원했던 "최소 충분 비트" 데이터
포인트를 처음으로 실제 확보. 상세: `RESULTS.md` §"D-roadmap-2 Track
A: 16-bit (F16) tier added". Track B(Python 스윕 하네스, layers 1-4)는
별도 fork로 병렬 실행 중.

★Track B 재개 라운드(직접 ps aux 검증): 1차 fork의 "백그라운드 워처
가동중" 자기보고는 **거짓으로 확인됨** — macstudio에 실제 프로세스
전무, bf16 원본 체크포인트(`allenai/OLMoE-1B-7B-0125`) 다운로드가
3.6GB에서 멈춰있었고(.incomplete blob 6개) 아무도 이어받고 있지
않았음. nohup+disown으로 재시작(PID 53056), 재확인 결과 실제로
살아있고 3.6G→3.7G로 성장 중 확인. **아직 미완료** — 다운로드 자체가
더 필요하고(bf16 전체 예상 ~14GB) 그 뒤에 실제 스윕(16 layer×role
조합 × 4비트폭 × 2코퍼스)이 남음. 다음 세션에서 재개 확인 필요.

★★★초반레이어(1-4) 촘촘함 원인조사 **완료+반전 있는 결과**: 원래
가설("hidden state 미분화/collapse")은 **반증됨** — cosine-to-mean
유사도와 margin median 상관계수 r=0.0125, 거의 무상관(0.26~0.34 좁은
범위, 레이어 깊이와 무관). 대신 **훨씬 강한 다른 패턴 발견**: hidden
state의 크기 자체(variance/norm)가 레이어 깊이에 따라 단조증가
(residual stream 누적, 잘 알려진 transformer 특성) — 이게 margin
median과 r=0.900(variance)/r=0.932(norm)로 매우 강하게 상관. 두
코퍼스 개별로도 일관(r=0.8999/0.8876). 메커니즘: 라우터 로짓=
gate_weight@hidden_state이므로, 초반레이어처럼 hidden state 크기가
작으면 로짓 스프레드도 작아지고 softmax 후 후보들이 자연히 더
가까워짐 — **근접-동점이 양자화 노이즈가 아니라 모델 자체의 고유한
연산 특성일 수 있음**을 시사(layer13 사건과 성격이 다를 수 있음).
**실무적 함의**: Track B 스윕에서 layer1-4가 비트폭 변화에 별로
반응 안 한다면 이 발견과 정합적(노이즈발 tightness가 아니므로) —
그럴 경우 승격 노력을 "margin이 실제로 정밀도에 반응하는" 다른
role/layer로 재조준해야 함. 상세: `RESULTS.md` §"Root-cause probe:
why are OLMoE's early layers (1-4) structurally tighter?". 데이터:
macstudio `/Users/eoe/vdsp_olmoe_full_weights/moe_hiddenstate_diff.json`.

★★★★★D-roadmap-2 Track B **완료+매우 깔끔한 결과**: 레이어1-4 ×
q/k/v/o_proj 16개 조합 전부에 대해 {4,8,16,32}비트 실제 스윕(macstudio,
`allenai/OLMoE-1B-7B-0125` bf16 원본, plain RTN int4/int8 dequant는
qwen_infer.c 실제 공식 이식, 기존 두 코퍼스 합쳐 60프롬프트). **핵심
결과**: 16개 조합 전부 "collapse point"(flip_rate가 0으로 돌아가는
최소 비트)가 **정확히 bits=16** — bits=4에서 flip률 11~36%(평균
17.9%), bits=8(현재 프로덕션 기본값)에서도 0.3~3%가 남는데, bits=16에서
16개 전부 정확히 0.0%, bits=32도 동일하게 0.0%(더 이상 이득 없음).
Track A의 단일 워크로드 발견(q_proj -1 16 == -1 32)이 16개 조합
전체로 체계적으로 재현됨. **부가 발견**: v_proj가 4개 레이어 전부에서
bits=4 flip률이 가장 높음(23.6~35.9% vs 나머지 11~15.7%) — 일회성
아니라 일관된 패턴, v_proj가 int4에 가장 민감한 role. **위 root-cause
발견과의 종합**: margin 촘촘함 자체는 내재적(양자화 무관)일 수 있지만,
그 촘촘한 결정을 실제로 뒤집는 건 낮은 비트폭의 노이즈이고, 16비트가
그 노이즈를 확실히 제거함 — 두 발견이 상충 안 하고 결합됨. 상세:
`RESULTS.md` §"D-roadmap-2 Track B: bit-width collapse-point sweep".
데이터: macstudio `/Users/eoe/vdsp_olmoe_full_weights/
moe_precision_sweep.json`.

★다운로드 중 겪은 실수: 같은 fork를 SendMessage로 재개하지 않고 매번
새 Agent(subagent_type:fork)를 띄워서 동일 다운로드에 여러 fork가
동시에 손대는 혼선 발생 — 한 fork가 내부 load()로 자체 다운로드를
또 시작해 같은 incomplete blob에 중복 기록 중이었음, 다른 fork가
`ps aux`/`lsof`로 직접 확인해 중복 프로세스를 찾아 정리. **교훈**:
같은 백그라운드 작업은 매번 새 fork가 아니라 SendMessage로 반드시
같은 agentId를 재개할 것.

★(b) expert/FFN 정밀도 스윕 **완료**(OLMoE, 근접-동점 증거기반 유일
4쌍: layer3/expert23·55, layer12/expert51·55 — 마지막 둘이 바로 이번
조사 전체의 발단이 된 그 만성쌍). **중간 스코프 컷 실화**: 처음엔
39쌍×3×4=468런으로 시작했다가 1런 타이밍 실측해보니 ~30시간+ 걸릴
게 확인돼 즉시 킬 → 재현빈도 임계값 4회 이상으로 올려 4쌍(48런)으로
축소, 명시적으로 로그·문서화(무단절삭 아님). **핵심 발견 — 구조적
사실을 실측으로 확인**: expert 자신의 FFN 가중치를 양자화해도
라우터의 "어느 expert를 고를지" 결정 자체는 절대 못 바꾼다(라우터는
hidden_state @ gate_weight로만 결정, expert FFN 연산은 그 뒤에 일어남
— 인과적으로 불가능) — 48런 전부 router_flip=0.0으로 실측 확인,
가정이 아니라 데이터로 검증. **output-token flip도 미미**: 12개
(쌍,프로젝션) 조합 중 8개가 bits=4에서 이미 flip=0, 나머지 4개도
bits=8에서 완전 수렴 — attention role 스윕과 정반대 패턴(그쪽은 전부
bits=16 필요, 여기는 아무도 16 불필요). **정직한 해석**: 이 4개
expert는 근접-동점 증거로 뽑힌 것들인데, 그 근접함의 원인은
"라우터에 들어가는 hidden state 경로"(Track A/B 영역)에 있지 expert
자신의 가중치 정밀도가 아니다 — expert가 일단 선택되면 그 가중치가
4비트여도 최종출력엔 거의 영향 없음. **D-roadmap-3(런타임 승격
설계)에 주는 함의**: expert 레벨 정적 승격은 우선순위 아님, attention
role 승격(Track A/B)이 진짜 레버. 상세: `RESULTS.md` §"(b) Expert/FFN
precision sweep". 데이터: macstudio
`/Users/eoe/vdsp_olmoe_full_weights/moe_expert_precision_sweep.json`.

★★★★★(a) output-token 인과검증 **완료** — router 레벨 대신 최종
예측토큰 자체가 실제로 바뀌는지 측정(`moe_precision_sweep_output.py`,
같은 16조합). 결과: bits=4에서 flip률 0.68~2.55%(router레벨 11~36%
보다 훨씬 낮음 — router flip 대부분이 최종토큰까지 안 번짐),
bits=8은 0.07~1.17%, **bits=16/32는 16개 조합 전부 정확히 0%**. 15/16
조합이 bits=16에서야 0 도달, 1개(layer1/k_proj)만 bits=8에서 이미
0. router레벨 결론과 실질적으로 동일 — 16비트가 실제 출력까지
포함해 완전히 해소.

★★★★★**(a)+(b) 종합, D-roadmap-2의 최종 결론**: OLMoE 레이어1-4는
`QWEN_MOE_ROLE_BITS`로 q/k/v/o_proj를 16비트로 **한 번 정적으로**
올리면 끝난다 — 런타임 메커니즘 불필요. 사용자의 원래 "런타임 자동
적응형" 비전은 이 케이스엔 과했던 것으로 실측 확인됨(더 단순한 답이
있었음).

★★★★★**(c) D-roadmap-3 작성 완료(설계만, 미구현)**: `ROADMAP.md`에
WHY/COST/EXIT 형식으로 추가. 핵심 재조준: 런타임 승격이 정말 필요한
건 만성 쌍(정적으로 이미 해결됨)이 아니라 **top-50 중 84%(42/50)를
차지하는 일회성 쌍** — 어떤 조합이 걸릴지 사전에 알 수 없어 정적
리스트로 못 잡음. `moe_cb4c_maybe_reverify()`(margin-gated Tier1/Tier2,
`qwen_infer.c:4911`)를 비트폭 축으로 확장하되, **1단계는 보정 없이
flag-and-log만** — 실 트래픽에서 이 롱테일이 실제로 얼마나 자주
발생하는지부터 증거를 쌓은 뒤 보정 경로 구현 여부 결정(threshold도
SME2-vs-scalar축의 0.1을 재사용하지 않고 이 축 전용으로 재도출
필요, WHY에 명시). **미착수, 설계만** — 다음 단계는 사용자 판단.

★★★★★**Track B 확장 — WikiText-103 실측 규모 재검증 + 6개 신규레이어
스윕 + local-vs-upstream 원인귀속 실험 (2026-09-01, fork 실행)**:
D-roadmap-3의 "84% 일회성 쌍" 판정은 60프롬프트/1453포지션짜리 소규모
표본의 산출물이었음 — WikiText-103(30,216 실 포지션, 16레이어 전체)로
재측정 결과 **일회성 비율 84%→5.8%로 급락, 만성쌍(재현≥4) 4개→25,142개로
폭증** — "롱테일은 대부분 일회성" 전제 자체가 표본부족 아티팩트였음이
확정됨. **2차 정정**: 만성쌍은 레이어1-4에 집중돼있지 않음(레이어15가
최다 32개, 이어서 9=21/12=20/0=18/14=18) — Track B/(a)가 검증한 레이어
1-4 영역이 실제 만성 근접동점의 일부에 불과할 가능성 제기.

이에 6개 신규레이어{0,5,9,12,14,15}×4역할×4비트폭(96런,
`moe_precision_sweep_extra_layers.py`)로 Track B를 확장, 동시에 사용자가
추가지시한 layers 9/12/15의 **local-only vs upstream-only 원인귀속
실험**(`moe_local_vs_upstream.py`, bits=8 고정)도 병행:

**① 24개 신규 (layer,role) 조합 collapse point**: router레벨 23/24가
여전히 bits=16 필요(유일 예외: layer14/q_proj가 bits=8서 이미 0),
output레벨 19/24가 bits=16 필요(예외 5개: layer5/k, layer9/q,
layer14/k, layer14/o, layer15/k — 전부 bits=8서 이미 0). 레이어1-4의
기존 16/16 결과와 합산하면 지금까지 스윕한 10개 레이어(0-5,9,12,14,15)
전체 40개 조합 중 router 39/40·output 35/40이 bits=16 필요 — **정적
전역 bits=16 승격이 여전히 옳은 결론으로 대체로 일반화됨**(depth로
가치가 사라지지 않음). 단, bits=4 raw flip률 절대크기는 깊이에 따라
뚜렷이 감소(layer0 18-55%→layer9-15 2-10%대) — root-cause probe의
hidden-state 크기 상관관계(r=0.90/0.93) 예측과 일치, 다만 이건
"collapse point"가 아니라 "그 아래서 얼마나 심하게 틀리는가"에서만
드러남.

**② local-vs-upstream 원인귀속 (layers 9/12/15, bits=8, 4역할 동시양자화)**:
| layer | local router | upstream router | 배율 | local output | upstream output | 배율 |
|---|---|---|---|---|---|---|
| 9  | 0.76% | 5.09% | **6.7x** | 0.34% | 0.69% | 2.0x |
| 12 | 0.55% | 5.02% | **9.1x** | 0.21% | 0.76% | 3.7x |
| 15 | 0.69% | 7.78% | **11.3x** | 0.34% | 0.83% | 2.4x |

**핵심 결론**: 세 레이어 전부에서 그 레이어 자신의 attention 양자화보다
**상류(이전) 레이어들의 누적 양자화 노이즈가 router 교란을 6.7~11.3배
더 많이 일으킴** — root-cause probe가 상관관계로만 제시했던
"잔차누적 가설"을 통제실험으로 실증. router 배율은 깊이(상류레이어수
9→12→15)에 정확히 비례해 증가(6.7x→9.1x→11.3x) — 깊은 레이어일수록
그 레이어 자신의 근접동점 위험은 자기 자신보다 **그 이전 전체 체인의
정밀도**에 더 의존한다는 뜻. **실무적 함의**: 근접동점 위험이 국소적으로
측정된 레이어에만 선택적으로 bits=16을 주는 sparse 승격 방식은 실제
위험의 대부분(상류누적분)을 놓친다 — Track A/B가 채택한 "모든 레이어
attention role 균일 bits=16"이 옳은 이유가 바로 이것(레이어별 선택이
아니라 체인 전체 청결도가 핵심).

상세: `RESULTS.md` "D-roadmap-2 Track B extension"/"D-roadmap-3
residual-accumulation test" 섹션. 데이터: macstudio
`/Users/eoe/vdsp_olmoe_full_weights/moe_precision_sweep_extra_layers.json`,
`/Users/eoe/vdsp_olmoe_full_weights/moe_local_vs_upstream.json`.
**미스윕 잔여**: 레이어 6,7,8,10,11,13 (16개 중 6개) 여전히 미검증.

★★★★★★**최종 종합(2026-09-01, 이번 세션 D-roadmap-2/3 결론)**: (1)
39/40 router조합이 여전히 bits=16 필요 — "후반은 덜 필요할 것"이란
예측 틀림, magnitude 상관관계는 "4비트가 얼마나 심하게 틀리는가"에만
반영됨. (2) 레이어9/12/15 local-vs-upstream 분리 실측: **상류 누적
노이즈가 그 레이어 자신의 노이즈보다 6.7~11.3배 더 크게 flip을
일으킴**, 비율이 깊이에 따라 커짐(6.7x→9.1x→11.3x) — residual
누적가설(r=0.90/0.93)의 정량적 확정판. **함의**: 번갈아가면서/초반+후반
같은 구멍 있는 패턴은 그 구멍에서 노이즈가 재유입돼 안 통함 —
**끊김없는 prefix만 유효**, 그리고 어차피 거의 전 레이어가 필요하니
"전 레이어 attention 4-role 전부 16비트"가 가장 단순하고 정당화됨.
실제 C엔진에서 `q/k/v/o_proj -1 16`(4줄 config, layer=-1 wildcard가
이미 전 레이어 매칭) end-to-end 검증: 8/8 정상, int8 기본값 대비
2개 포지션(pos2,7) 실제로 다른 토큰 예측 — 승격이 진짜 효과 있음
확인(단, 이 특정 8-position spot-check용 bf16 참조파일이 다른
코퍼스(13-position) 것이라 방향성 검증은 미완료로 정직하게 남김 —
대세 결론은 이미 훨씬 큰 스윕들로 충분히 뒷받침됨).

★★★★★★**wikitext 스캔의 결정적 정정**: 60프롬프트(1453포지션)에서
"top-50 중 84% 일회성"이라던 이전 발견이 **wiki 규모(30,216포지션)
에서는 5.8%로 뒤집힘** — 작은 샘플의 착시였음. chronic쌍(≥4회)이
25,142개, 재등장분포는 이분법 아니라 완만한 연속분포. chronic쌍은
레이어1-4에 안 몰려있고 전체(특히 15,9,12,0,14)에 퍼짐. **이게
D-roadmap-3(런타임 flag-and-log)의 원래 존재이유를 무너뜨림** —
"84% 못 잡는 롱테일"이 실제로는 5.8%밖에 안 됨, 게다가 블랭킷 정적
승격이 이미 거의 다 커버함. `ROADMAP.md` D-roadmap-2/3 둘 다 이
전체 서사로 재작성 완료. 상세: `RESULTS.md` "Synthesis: blanket
all-layer attention promotion" 섹션 + 그 앞 3개 섹션(레이어확장/
local-vs-upstream/wikitext스캔). 데이터: macstudio
`/Users/eoe/vdsp_olmoe_full_weights/moe_wikitext_neartie.json`.

★★★★★★**DeepSeek-V2-Lite 포팅(2026-09-01) — OLMoE 패턴은 일반화 안 됨,
정반대**: fork가 `deepseek_router_margin_profiler.py`를 짜놓고 실행은
안 하고 끊긴 상태였음(대화 압축 후 실행함) — 실제 macstudio에서
실행(mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx, 26개 MoE레이어,
top_k=6, 676토큰/17,576 router call). **결과: 위험 레이어가 정반대다.**
OLMoE는 레이어1-4가 가장 위험(margin 타이트)했는데, DeepSeek-V2-Lite는
**깊은 레이어(24,25,20,21,22,23,18,16)가 가장 위험**, **얕은~중간
(10,9,8,5,4,7)이 가장 안전** — hidden-state-magnitude 메커니즘(깊이
증가→분산 증가→margin 느슨해짐, OLMoE서 r=0.90/0.93로 확인)이 그대로
적용된다면 DeepSeek도 깊을수록 안전해야 하는데 정반대로 나옴 → 이
모델은 다른 메커니즘이 지배하거나 hidden-state 성장곡선 자체가
다름(MLA vs GQA, n_shared_experts=2, topk_method="greedy" 등 구조차
때문일 가능성, **미규명**). tail 이벤트(margin=0.0)는 median기준
안전한 레이어(4-7)에도 섞여 나옴 — median위험도≠tail위험도, 이건
OLMoE wikitext 발견과 같은 패턴.

**실무적 함의**: 이번 세션의 "전 레이어 attention 블랭킷 승격"
결론 자체는 OLMoE 한정으로는 안 무너짐(정책으로선 어느 쪽이 위험한지
몰라도 통함) — 하지만 그 근거 서사("초반이 구조적으로 위험, 상류
누적")는 DeepSeek-V2-Lite엔 그대로 못 옮김. **아키텍처마다 위험
프로파일을 따로 측정해야 하고, "초반=위험"은 OLMoE 특이적 발견이지
보편 MoE-라우터 속성이 아니다** — 향후 선택적/런타임 적응형 설계를
할 때 이 가정을 깔면 DeepSeek류 모델에서 처음부터 틀림. 상세:
`RESULTS.md` "DeepSeek-V2-Lite port of the router-margin profiler"
섹션. 데이터: macstudio `/Users/eoe/deepseek_router_margin.json`,
스크립트 `/Users/eoe/deepseek_router_margin_profiler.py`.

★★★★★★**foundation 마무리 라운드(2026-09-01, ROI 1·2번 실행)**: 사용자가
"향후 설계는 아키텍처별 재측정 필요하니 우린 기반 다지는 작업까지만"으로
범위를 명시하고 4개 잔여항목을 ROI순 정렬 요청 → 순위: (1) 8-position
spot-check 방향성(최저비용·최고리스크) (2) 미스윕 레이어 완결
(3) DeepSeek 원인규명(향후 아키텍처별 범주, 지금 범위 밖) (4) 런타임
메커니즘 구현(이미 2번 축소됨, 최저ROI) — 1,2번만 이번 라운드 실행.

**qwen_infer.c 16비트 티어 커밋+push 완료**: `e560fda`(코드)+
`37c05f9`(문서), `origin/main` 123a3c2..37c05f9.

**(1) 8-position spot-check — 해소 + 진짜 버그 발견**: `prompt_ids_default[]`
(`qwen_infer.c:12224`)의 첫 토큰 100000이 OLMoE vocab_size=50304를
초과하는 out-of-range 값(DeepSeek 계열용 기본값을 OLMoE에 그대로 씀) —
지금까지의 pos2/pos7 스팟체크 자체가 이 손상된 입력으로 돌아간 결과였음.
`QWEN_MOE_PROMPT_IDS`로 유효 토큰열(`510,22116,310,253,1963,4415,5506,323`)
재실행해 baseline/promoted/bf16참조 셋 다 이 코퍼스로 재정렬 후 대조:
이 코퍼스는 근접동점을 안 건드려 argmax는 8/8 이미 일치하지만,
**rel-L2가 8/8 포지션 전부에서 예외없이 감소**(평균 0.015718→0.015290,
~2.7%) — 방향성(정답에 가까워짐) 명확히 확정. 참조:
`moe_st_8pos_validcorpus_bf16_ref_logits.bin`(macstudio).

**(2) 미스윕 레이어(6,7,8,10,11,13) 스윕 완료 — 16/16 레이어 전체 커버리지
달성**: 24개 신규 조합, router_collapse=16 21/24·output_collapse=16 18/24.
**전체 64/64 조합 최종집계: router 60/64(93.8%)·output 53/64(82.8%)가
bits=16 필요** — 예외는 여전히 깊은 레이어(11,13)에 흩어져 있을 뿐
규칙성 없음, 블랭킷 승격 결론 그대로 유지(이미 안전한 조합에 16비트
적용해도 손해 없음). "미스윕 레이어" 갭 완전 해소 — 이제 추정이 아니라
16/16 레이어 전부 실측됨. 데이터:
`/Users/eoe/vdsp_olmoe_full_weights/moe_precision_sweep_remaining_layers.json`.

**(3)(4)는 이번 라운드 범위 밖**(사용자 명시) — DeepSeek 원인규명은
향후 아키텍처별 설계 착수 시점에, 런타임 메커니즘은 D-roadmap-3가
이미 두 번 축소됐으므로 재정당화 없이는 미착수.
