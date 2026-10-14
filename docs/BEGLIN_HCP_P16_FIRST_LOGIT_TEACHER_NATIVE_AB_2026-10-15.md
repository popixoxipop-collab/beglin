# Beglin HCP P16 — Original BF16/FP32 full teacher vs native quantized A/B logits

작성: 2026-10-15 KST
판정: **동일 입력 첫 토큰 full-vocabulary logits 실기기 수집·독립 비교 PASS / 품질 비열등성 및 운영 승격 BLOCKED**

저장소: `popixoxipop-collab/beglin` (Private)
연구 브랜치: `research/hcp-p16-native-vs-fullteacher-logits-20261015`
상위 P15 commit: `fb8e4bea935f7f0b3511e1fac3b87c1bf54f604d`
네이티브 실행 전 연구 프로토콜·토큰 사전등록 commit: `f2ed4b49541db2cdedc2103fcf1aade09a9997c7`

## 1. 무엇을 실제로 입증했나

P15에서 본래 DeepSeek-V2-Lite 전체 원본 BF16 체크포인트(27계층·5,291텐서)를 계층 스트리밍 방식으로 모두 forward하고, 동일한 두 문장에 대해 BF16 또는 FP32 연산으로 각각 102,400개 full-vocabulary logits를 얻었다. P16에서는 마침내 **정확히 같은 7토큰·9토큰 원본 tokenizer 입력을 실제 Beglin GPU persistent 네이티브 모델 정책 A/B에 제공**, 첫 생성 토큰의 원본 `gpu_logits` 102,400개(float32)를 **그리디 argmax 직전 마지막 prefill 위치에서 직접 추출**했다.

- XOX 연구용 네이티브 단일 MLX GPU 워커에서 총 **8개 요청 / 8개 실제 409,600-byte full logits 캡처 성공**, native job `job_ab44abfd88296628249efa77fa249dbb` SUCCEEDED(exit 0).
- 두 원본 입력 각각 정책 A/B를 **2회씩** 실행했다. 동일 입력·동일 정책은 전환 epoch가 달라도 **logits SHA-256까지 바이트 완전 일치**.
- 실제 네이티브 초기 정책 A(steady)와 **REBIND_SET 전환 6건**, 그 외 policy steady 2건. 모든 원본 정책 SHA·epoch·native ACK를 검증했다. 캡처된 logit argmax가 반환된 실제 첫 생성 토큰 8/8과 일치한다.
- EOE에서 C 코드가 기록한 SHA, 실제 C worker 출력 로그, input token ID SHA, 전체 float32 배열, 두 P15 교사 `.npy` 4개 SHA를 독립으로 다시 비교한 결과 **4개 유일한 (입력, 정책) 셀/8개 캡처 전수 검증 통과**.
- 국소 두 텐서(공유 전문가 L3 `shared_up_proj`, L26 `shared_down_proj`)만 서로 다른 5·6비트 A/B로 전환하며 다른 AF 양자화 레이어도 여전히 존재한다. 따라서 BF16 teacher와의 차이 전체를 **이 두 텐서의 국소 양자화에 인과 귀속해서는 안 된다.**

**핵심 결론:** `전체 교사 logits 미실행`이라는 구 P14 blocker는 P15에서 해소됐고, `교사와 네이티브 A/B 동일 입력 full logits 직접 비교 미실행`이라는 P15 blocker도 P16에서 해소됐다. 그러나 품질 비열등성 검증과 운영 승격은 **아직 FAIL 아닌 BLOCKED(판정 표본·기준 부족)**.

## 2. 소스·입력·실험 보호 계약

원본 모델 index SHA: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.

원본 P15 frozen token manifest SHA: `c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b`. P16 frozen 2-입력 참조 SHA: `0ea208ca6b6bf9fa404fd291679e164f63217c91b1d6e15756d331261998c36e`.

원본 P5 C native CB128 source SHA: `bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612`. 동일 P9 MLX KV128 source SHA: `ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f`. P16 **관측만** 추가한 격리 C source SHA: `aedb3f915c532b7e454515598b6d9836a4962e23391e2e304980cde64bc7c1af`. 실제 컴파일 실기기 바이너리 SHA: `3d0f19b64125dbffb3d80e26ea07eff8d5b2ab80d3aa1d83bc9f59dd729ef9c5`.

P16 C 계측은 마지막 prefill 위치를 확인하고 pass=1, 요청 1건, 생성 max_new=1, 정확히 7·9토큰 및 두 frozen input SHA만 허용한다. `BEGLIN_P16_LOGITS_DIR`가 설정되고 지정된 **해당 실험의 XOX 소유 루트**와 일치해야 데이터가 기록된다. 중복 파일·symlink·비정상/NaN logits·중복 캡처·잘못된 입력은 fail-closed(`_exit(117)`). 원본 CB128 C를 계측 코드 제거로 **바이트 단위 복원**할 수 있으며 수식·MLX·양자화·그리디 선택·ACK·롤백을 변경하지 않았다.

각 `.f32` 파일은 little-endian float32 × 102,400 = **409,600 bytes**. C `CommonCrypto SHA256`와 Python `hashlib`를 독립 대조했으며, 내부 `worker.log` 원본도 EOE 비공개 아카이브에서 보존했다. SHA 캡처의 오버헤드를 포함한 관측 시간으로 성능 개선을 주장하지 않는다.

## 3. 실제 BF16 / FP32 teacher 분포 비교

입력 0: `The result of seven times eight is` (7토큰). 입력 1: `Complete the sentence: The capital of France is` (9토큰). 원본 두 문장은 정답형 외부 holdout이 아닌 사전고정 연구용 엔지니어링 입력이다.

### JS divergence — 낮을수록 교사 확률분포에 가깝다 (nats)

| 입력 | 전체 교사 연산 | 정책 A | 정책 B |
|---|---|---:|---:|
| 수학 문장 | BF16 CPU | 0.026654 | **0.022806** |
| 수학 문장 | FP32 CPU | 0.023366 | **0.020571** |
| 프랑스 수도 문장 | BF16 CPU | **0.020205** | 0.027373 |
| 프랑스 수도 문장 | FP32 CPU | **0.020322** | 0.026953 |

### KL(teacher || native) — 낮을수록 교사에 가깝다 (nats)

| 입력 | 전체 교사 연산 | 정책 A | 정책 B |
|---|---|---:|---:|
| 수학 문장 | BF16 | 0.118398 | **0.101695** |
| 수학 문장 | FP32 | 0.104694 | **0.092246** |
| 프랑스 수도 문장 | BF16 | **0.076958** | 0.102489 |
| 프랑스 수도 문장 | FP32 | **0.077986** | 0.102116 |

위 두 독립 분포 지표는 **입력별 승자가 반대**다. 수학 문장에서는 B가 A보다 teacher 확률분포에 가까웠고, 프랑스 수도 문장에서는 A가 가까웠다.

### 첫 생성 토큰

| 입력 | BF16 전체 교사 | FP32 전체 교사 | 네이티브 A | 네이티브 B |
|---|---|---|---|---|
| 수학 문장 | 토큰 ID 207 (공백) | ID 15092 (` fifty`) | 15092 | 15092 |
| 프랑스 수도 문장 | ID 51536 (` _____`) | 51536 | 51536 | 51536 |

수학 문장의 native A/B는 교사 FP32의 argmax와 같고 BF16의 argmax와 다르다. 프랑스 수도 문장은 네 종류 모두 같은 첫 토큰을 선택한다. 그러나 `_____ ` 토큰은 해당 문제의 **정답 표지가 아니다**. Top-1 매칭을 정답률로 환산해서는 안 된다.

### 전체 logits 원본값 차이

BF16 교사 대비 상대 L2: 수학 A 0.481050, B 0.474589 / 프랑스 수도 A 0.137368, B 0.142250. 차이가 꽤 커서 단순히 `동일 분포`나 `BF16과 정밀도 전환 근사 일치`라고 말할 수 없다. AF 기본 양자화, 누적 rounding, kernel attention/RoPE/LM head 등의 기여분을 세분화한 독립 ablation이 필요한 부분이다.

## 4. 원본 증거·실기기 ACK

사전등록 순서: A0→B0→A1→B1→B0→A0→B1→A1.

- 실기기 P16 결과 원본 SHA: `902452b285faf48927445d8b8df95f20666fc6d0367de7aa3a96b6ffe858efc1`.
- EOE 독립 전체 분포 감사 결과 SHA: `8d665b07d3015e13a81db81e0126a00d4b0602f0ac90d5e83de8578e090f32fb`.
- 원본 C native log SHA: `0aacee9a80a8345fbbed787d74f8ddbc6eced750c7a3fc94722033d05d647099` (EOE private).
- 최초 실행 XOX 연구용 native job `job_ab44abfd88296628249efa77fa249dbb`: SUCCEEDED, exit 0.
- P16 계측 C에서 P9 원본으로 되돌리는 역함수, 허용 토큰 SHA, 중복·NaN·범위·마지막 prefill 위치: 경계 테스트 **15/15 PASS**, AppleClang GPU 경로 `-fsyntax-only` exit 0(기존 cblas deprecation warnings만 있음).
- 원본 full native/teacher 배열 변조, source/ACK/정밀도 hash 위조, teacher 원본 값 변조, 양자화 품질 비열등성 거짓 주장 차단: 신규 **24/24 PASS**.
- EOE P8~P16 15개 회귀군 **292/292 PASS**. XOX 원본 P15 상주형↔스트리밍 수치동등성 재검증 **18/18 PASS**, job `job_b36c04822b918f32e27443d0abc2f158`. **이번 턴 전체 재검증 310/310 PASS**. (이 숫자는 서로 다른 범위·시대의 회귀 테스트 합계이며 통계적 인증 횟수가 아님.)

8개의 원본 C `.f32` logits(각 409,600 bytes), 네이티브 admission/epoch/ACK 메타, P16 원본 사전등록 및 EOE 전체 비교 결과는 저장소 `experiments/hcp_p16/evidence/native_xox_20261015/`에 SHA 기준으로 보존한다. P15 각 BF16/FP32 teacher logits `.npy`와 5,291텐서 원본 출처 digest는 이전 P15 연구 증거를 그대로 사용한다. 원본 가중치 파일은 GitHub에 복사하지 않았다.

## 5. 운영 보존과 강제 차단

원본 stage SHA: `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`. 원본 active-route SHA: `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`. 기존 stage PID: `35481,35493`. P16 전후 원본 서빙·라우트를 변경하지 않았고, 실험용 GPU 워커만 시작하고 종료했다. 새 원본 BF16 모델 GPU 적재, main merge 또는 Tailnet operator 활성화를 실행하지 않았다.

**아직 완성되지 않은 인증:**
1. 정적 AF/A/B 및 캐시 동적 경로에 대한 **독립 미사용 holdout의 인간 또는 객관적 정답 레이블, NLL/perplexity, 여러 토큰 전체 생성, 표본/CI와 품질 비열등성 마진**.
2. 본 실험에서 동일 입력 반복은 바이트 일치하지만 새 실제 데이터·장기 모델 교체·다중 워커 정책 경합·cold boot·호스트 크래시의 exactly-once 원래 요청 처리를 대표하지 않는다.
3. 관측된 큰 logit L2 차이의 원인이 단순 국소 2개 양자화 레이어 때문인지, 나머지 AF 가중치·계층 누적·MLX FPU dtype·소프트맥스 구현 때문인지를 ablation으로 분리하지 않았다.
4. 이 실험의 `quality_noninferiority_certified=false`, `automatic_live_promotion=false` 유지. 어떠한 연구 브랜치도 자동 운영 승격 대상이 아니다.

**최종 P16 판정:** P15 전체 BF16·FP32 원본 교사와 실제 P9 국소 A/B의 동일 입력 전체 logits 직접 비교 성공. **정밀도 정책 B가 전역적으로 A보다 낫다는 증거는 없음.** 다음 P17은 실제 unseen corpus·다중 토큰 손실·나머지 AF 양자화 오차를 분해하고 사전등록한 비열등성 심사를 실행한다.
