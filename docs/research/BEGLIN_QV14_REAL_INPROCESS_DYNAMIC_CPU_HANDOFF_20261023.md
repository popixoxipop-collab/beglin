# BEGLIN Qv14 — 실제 동일 프로세스 국소 정밀도 전환 CPU 검증 및 다음 최적화 게이트

**연구일**: 2026-10-23 (Asia/Seoul)  
**저장소 / 연구 브랜치**: `popixoxipop-collab/beglin` / `research/qv14-inprocess-dynamic-20261023`  
**최종 판정**: `IN_PROCESS_CPU_Q4_N8_SWITCH_REAL_NATIVE_PASS / STATIC_NUMERIC_PARITY_28_OF_28_PASS / SWITCH_COST_PROHIBITIVE / ACCURACY_GAIN_NOT_ESTABLISHED / METAL_NOT_TESTED / AUTO_PROMOTION_OFF`

## 1. 이전 Qv13과 실제 차이

Qv13은 Qwen2.5-1.5B-Instruct의 총 196개 projection 텐서 중 28개를 시작 시 n8로 로드한 서로 다른 **고정 마스크별 개별 프로세스**의 성능을 측정한 연구다. Qv13의 해당 방식으로는 요청 사이 실제 정밀도 전환을 주장할 수 없었다.

Qv14의 이 작업은 원본 Qv5 mixed C 로더를 외부에서 복제하고, 인접한 두 요청이 서로 다른 마스크를 요구하면 **살아 있는 같은 엔진의 실제 WT(가중치) 객체를 기존 Q4G64↔qNg64(n8)로 변환**한다. 원본 BF16 safetensors를 재읽어 계층별 텐서를 재양자화하고 성공 후 기존 packed/scales 버퍼를 해제한다. 전체 Q4/전체 n8 모델 2벌을 미리 RAM에 올려놓고 포인터만 전환하는 방식이 **아니다**. 한 텐서의 F32 임시 변환·신 packed 버퍼가 추가 공간을 차지한다.

- 원본 C 소스 SHA256: `375b3dfb02ac2c0fc90f3f500718578fe62d38cdcac541099fb896797470dbeb`.
- 기존 정적 mixed C 바이너리 SHA256: `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221`.
- 분리 빌드 v2 바이너리 SHA256: `0955e6d5b0caf4c5f947b1f740c566d224dcd0b47a656b5760de720bf944ac90`.
- 별도 로그형식 보정 v3 바이너리 SHA256: `f9ad286158e98e1597ab2f2e97f22d1aa0ce5fb46e7b617ec66a0921a955f7b4`.
- v3 C 소스 SHA256: `07a69e0a67cac0c448d6c7f6ec23cc506d95cf9a3bfa905acbb1d41abbbd7702`.
- v2→v3 C 패치 차이는 **실험 로그의 줄바꿈 문자를 바로잡은 정확히 한 구간**이다. 성능 커널, 텐서 변환 계산, 정책 선택 함수는 변경하지 않았다.
- 원본 BEGLIN Q4/Apple NEON n8 구성 및 링크 오브젝트 23개를 다시 사용했고, 모델 원본/BF16 체크포인트/원본 mixed binary는 변경하지 않았다.

## 2. 동적 정책과 메모리 계약

이 실험에서 동적 요청 선택은 **휴리스틱**이다. 모델 로짓·정답·실패 여부를 보고 최적의 마스크를 선택하는 알고리즘은 **아직 없다**.

- 입력 prefix 길이 **85토큰 미만이면 Qv8 고정 28n8**.
- 입력 prefix 길이 **85토큰 이상이면 무작위 seed73 고정 28n8**.
- 비교 두 마스크 모두 projection packed 가중치 **765,493,248바이트**, 동일한 n8 28개와 역할별 분포를 가진다. 요청마다 역할·계층 정밀도 상태를 확인한다.
- 정밀도 변환은 실행 중인 모델의 기존 WT 객체에 적용되며, 변경할 텐서를 **먼저 n8→Q4로 내리고, 그다음 Q4→n8로 올려** 가중치 예산을 크게 넘기지 않도록 구현했다.
- 조건에 맞는 프로파일이 연속되면 재양자화를 건너뛰는 코드가 있다. 초깃값은 반드시 Qv8 original mask와 일치하도록 검사한다.
- 실제 RAM 사용량은 별도 측정한다. 임시 F32 전개와 새 packed 버퍼 때문에 **전환 직후의 기기 RSS**는 정적 실행보다 높을 수 있다.

## 3. 1차 원본 ARC validation 4문항

기존 Qv13 test와 다른 ARC-Challenge validation에서 입력 선택·정답 토큰·마스크를 **추론 전에 SHA로 봉인**했다.

- private 입력 봉인 SHA256: `e12ba4ae07dba6476c4452357ce77b32f2a4c7ba64c5c1bb013e74cf1245e032`.
- public manifest SHA256: `83fb76b032f628a4bf1b4b3d00471a635e010bb40f8ae27cf7b50030131c4dd5`.
- 문항 입력 길이: **72, 103, 70, 90**토큰. 기대 정밀도: Qv8, random73, Qv8, random73.
- 하나의 28계층 C 프로세스에서 **실제 프로파일 전환 3회**, 회당 **40개**, 총 **120회 텐서 재양자화**.
- 각 요청의 28 n8/168 Q4 역할 실증, 평균 전환 시간 약 2.39초, 누적 전환 시간 **7.181초**.
- 해당 정적 원본 C Qv8 및 random73를 별도로 실행한 뒤 문항별 정답 NLL **4/4 정확히 동일**, 정답 여부 4/4 동일. 4문항 모두 정답.
- **기록할 실패:** 초기 v2 C 실행은 정상 종료해 점수까지 출력했으나 C 디버그 형식에 리터럴 `\\n`이 들어가 Python 래퍼의 앵커 기반 로그 파싱이 실패했고, 따라서 **원래 Tailnet job 상태는 FAILED**. 별도의 독립 파서는 **원본 로그 SHA를 보존하고 줄바꿈 구분만 인식**해서 네이티브 원본 결과를 검산했다. v3에서 이 로그 형식만 고친 후 별도 새 입력을 검증했다. 실패를 소급해 job 자체 SUCCEEDED로 바꾸지 않는다.
- 공개 결과: `docs/research/BEGLIN_QV14_INPROCESS_SWITCH_NUMERIC_PARITY_NATIVE_PUBLIC_20261023.json`.

## 4. 신규 24문항 확대 실측: v3 native C

이전 4문항을 제외한 ARC-Challenge validation 문항에서 **SHA 우선 순서로 짧은 12건/긴 12건**을 고정해 교대로 평가했다. 더 나은 마스크를 선별하기 위해 이 신규 24개 출력·정답률을 보지 않았다.

- Private 입력 SHA256: `e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6`.
- Public 입력 SHA256: `6cdd0679a8b36512a4f13651a1229bd8ee91b0f979fdd8c193c04dff69ed79ba`.
- 총 24개 요청. 같은 살아있는 C 프로세스에서 **23회 프로파일 전환**, **920회 실제 WT 정밀도 재양자화**, 누적 switch time **55.003343초**, 즉 전환당 약 **2.39145초**.
- 매 요청마다 28개 n8 / 168개 Q4 실행 정밀도 상태 검사 PASS.
- **모든 24문항에서 동적 결과를 그 문항에 실제 선택된 원본 정적 마스크의 C 결과와 비교**: 최대 정답 NLL 절댓값 차이 **0**, 정답 여부 일치 24/24. 동적 프로파일이 단순 로깅이 아니라 원래 정적 정밀도 계산을 재현한다는 강한 실험 증거다.

| 조건 | 정답/24 | 평균 full-vocab gold-token NLL ↓ | 프로세스 wall | peak process RSS |
|---|---:|---:|---:|---:|
| **동적 원본 Q4↔n8** | **17** | **1.590553** | **186.410초** | **4,696.45 MiB** |
| 정적 Qv8 28n8 | 16 | 1.649323 | 131.658초 | 4,638.91 MiB |
| 정적 무작위73 28n8 | 16 | 1.612909 | 130.908초 | 4,638.91 MiB |

동적 실행은 고정 Qv8 실행보다 wall **54.753초 증가(약 1.416배)**했고, **직접 계측한 switch 55.003초**와 거의 일치한다. 실제 최대 프로세스 RSS도 **57.547 MiB** 더 높다. 이 차이를 조용한 지속 추론 벤치마크 TTFT/TPOT로 과장하지 않는다. CPU 부하·워밍업/모델 로딩 등은 별도로 표준화하지 않았다.

동적 전환은 정답 1개 개선으로 보였지만, 정적 Qv8 대비 평균 full-vocab gold NLL 차이 +0.05877, 15,000회 paired question bootstrap 95% 구간 약 **[−0.727, +0.733]**이며, 정적 random73 대비 +0.02236, 구간 **[−0.301, +0.324]**로 모두 0 포함한다. 정확도 대응 McNemar exact 양측 p값도 두 비교 모두 **1.0**. 24건에서는 *정확도·품질 개선 우위는 미입증*이다.

## 5. 독립 근거와 코드 파일

- 4개 request 재현 코드: `tools/qv14_freeze_dynamic_validation4.py`, `tools/qv14_build_inprocess_native.py`, `tools/qv14_native_dynamic_eval.py`, `tools/qv14_independent_native_audit.py`.
- 24개 확장 실증 코드: `tools/qv14_freeze_validation24.py`, `tools/qv14_build_inprocess_native_v3.py`, `tools/qv14_native_eval24.py`, `tools/qv14_replication24_independent_audit.py`.
- 공개 4문항 결과 SHA256: `6fadafb684a63c0c8046fc8dfeef60710dbd0a4dbb67860daef76ec093008e10`.
- 공개 24문항 결과 SHA256: `f3d3fb4c5f38887b8e78f78e93d67a3abc38fa09f22ff38ed9547ac3a0193378`.
- 실제 C 실행, static 비교 및 로그/영수증은 EOE의 `/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023`에 owner-only로 보존. GitHub에는 **원문 질문/토큰/문항별 원본 NLL을 공개하지 않음**.
- 24문항 첫 네이티브 작업은 Python 래퍼의 사전실행 잠금 키 `rows` vs `exact_input_rows` 불일치로 모델 forward **이전** FAIL. 입력·마스크 자체를 변경하지 않고 래퍼만 고쳤으며, 72개 문항·정밀도 입력 파일 전수 SHA 검증 후 새 idempotency-key로 재실행했다. 이 이력도 삭제하지 않는다.

## 6. 현재 과학적 결론과 다음 수행할 작업

**기능 측면:** 실제 Apple Silicon CPU에서 **프로세스 유지형 텐서별 정밀도 변환을 검증했다**. 고정된 마스크 파일을 따로 로드해 실행한 것과 달리, 실제 live `WT` packed/scales가 Q4/n8 사이 전환된다. 전환 후 정적 원본 C와 문항별 수치가 정확히 일치한다.

**최적화 측면:** 실패/미입증. 전환 하나가 40개 텐서에 약 **2.39초** 필요해 매 요청 마스크를 바꾸면 비용이 매우 크다. 이 입력에서는 wall 약 42% 증가, RSS 추가 약 58MiB. 예제 정책은 prefix 길이 임곗값에 따른 단순 휴리스틱으로, 학습·최적화된 정책이 아니다. 24건의 정답률 +1은 통계적 증거가 아니다.

**다음 성능 연구 기준:**
1. **실제 메모리·latency 예산형 선택 게이트:** 전환 예상 비용, input/hardware-conditioned precision utility, 전환 빈도, 최악 지연, 재사용 구간을 포함한 정책 목표함수 설계. Qv13 test 256개와 Qv14 4/24 평가 문항은 해당 정책을 학습하는 데 재사용하지 않는다.
2. **비트플레인/패킹 알고리즘 자체 개선:** BF16→F32→Q4/n8 전체 양자화의 약 2.39초 전환을 줄이는 증분 압축/온디맨드 캐시를 격리 구현하되, 캐시된 다른 전체 packed 모델의 RAM 중복 상주를 숨기지 않는다. 복잡한 런타임 모드 및 재시작 후 메모리 중복을 독립 측정한다.
3. **독립 학습/검증/평가 및 품질:** 새 데이터셋에서 정책을 훈련용/평가용으로 엄격히 분리해, 4선택 조건부 NLL·정답률·full-vocab NLL·worst-tail·출력 교정·요청별 대사 검사를 수행한다.
4. **CPU/Metal 별도 판정:** 본 검증은 원본 CPU C/NEON 경로다. Metal GPU 실행의 가중치 residency, bitplane 전환, kernel correctness/latency는 별도 코드·실기기로 검증 전까지 미실증.
5. **운영 경계:** 원본 C/체크포인트, main, immutable Tailnet 운영 릴리스, auto-promotion은 이번 실험에서 변경하지 않았다. 산업용 서빙 준비도는 별개.

**한 줄 판정:** Qv14에서 사용자 입력별 **실제 live 국소 양자화 전환 능력은 구현·실증**했다. 하지만 첫 버전은 지연·메모리 오버헤드가 커서 **최적 할당 엔진이나 최종 성능 향상으로 승격할 수 없다.**
