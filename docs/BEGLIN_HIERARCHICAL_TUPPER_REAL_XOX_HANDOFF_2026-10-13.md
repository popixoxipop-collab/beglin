# Beglin 실가중치 계층형 Tupper Micro-Expert 검증 보고서

- 기준일: 2026-10-13
- 저장소: `popixoxipop-collab/beglin`
- 연구 브랜치: `research/hierarchical-tupper-moe-20261013`
- 원본 `main` 및 운영 서빙 변경: 없음
- 검증 범위: 실제 DeepSeek-V2-Lite BF16 Safetensors, MoE layer 1의 8/64 routed experts, `hidden=2048`, `intermediate=1408`, `gate/up/down`을 함께 평가

## 판정

**현재 공통기저+Q2 압축 가설: 실가중치 검증에서 기각(현 구현 및 실험 범위).**

Micro-Expert로 분할해 모든 채널을 합하는 연산은 원본 SwiGLU의 수학적 항등 분해이며, 일반화한 Tupper `N×M` 활성 마스크는 무손실 비트 인코딩으로 동작한다. 하지만 이러한 성질은 Expert 간 공통 저랭크 법칙의 존재, 압축 효과, 희소 선택의 정확도를 보장하지 않는다.

## 실제 XOX 실험 및 근거

1. 8 Experts 전체 행 실가중치: [GitHub Actions #38390892514](https://github.com/popixoxipop-collab/beglin/actions/runs/38390892514), SUCCESS.
2. 64 Experts 전체 행 실가중치: [GitHub Actions #38391147377](https://github.com/popixoxipop-collab/beglin/actions/runs/38391147377), SUCCESS. [아티팩트](https://github.com/popixoxipop-collab/beglin/actions/runs/38391147377/artifacts/11809475198). 검증된 파일 4개의 SHA256은 실행 영수증과 일치함.
3. 64 Experts, 학습 Expert 48개/평가 Expert 16개, rank 2/8/16 대조: [GitHub Actions #38391901218](https://github.com/popixoxipop-collab/beglin/actions/runs/38391901218), SUCCESS (다만 해당 시도는 업로드 경로 불일치로 아티팩트가 보존되지 않음). 수정한 재실행: [GitHub Actions #38392195248](https://github.com/popixoxipop-collab/beglin/actions/runs/38392195248), **SUCCESS, 아티팩트 5파일 다운로드 및 영수증 SHA-256 4/4 일치 검증 완료**. [재실행 아티팩트](https://github.com/popixoxipop-collab/beglin/actions/runs/38392195248/artifacts/11809316885).

### 최초 64-Expert 평가 (train=16, heldout=48, rank=2)

| Micro 채널 | 공유 평균 대비 예측 MSE 설명률 | shared+Q2 / Q4 가중치 제곱 오차 | shared+Q2 / Q4 추정 저장량 | 50% 채널 유지 출력 상대 L2 |
|---:|---:|---:|---:|---:|
| 16 | -0.00443% | 43.86× | 0.7232× | 68.36% |
| 32 | -0.00011% | 43.86× | 0.7230× | 69.12% |
| 64 | +0.00244% | 43.86× | 0.7229× | 62.83% |
| 128 | +0.00337% | 43.85× | 0.7228× | 65.34% |

압축 측면에서는 27.7%의 추정 바이트 감소가 보이지만 오차가 44배가량 커졌다. 이는 제품에 사용할 수 있는 품질-비용 트레이드오프가 아니다.

### rank ablation (train=48, heldout=16)

| rank | 채널 | MSE 설명률 | shared+Q2 / Q4 오차 | shared+Q2 / Q4 저장량 |
|---:|---:|---:|---:|---:|
| 2 | 16 | -0.00939% | 42.05× | 0.7232× |
| 2 | 64 | -0.00161% | 42.05× | 0.7229× |
| 8 | 16 | -0.03452% | 37.75× | 1.0567× |
| 8 | 64 | -0.00740% | 37.76× | 1.0563× |
| 16 | 16 | -0.06998% | 32.12× | 1.5014× |
| 16 | 64 | -0.01483% | 32.14× | 1.5008× |

rank 증가로 Q2 잔차 오차는 일부 감소하나 저장량이 오히려 증가하고 홀드아웃 MSE 설명률은 계속 0 이하를 보인다. rank 부족만으로 실패를 설명하기 어렵다.

## 무결성 및 제한

- XOX 호스트 신원 검증 및 해당 BF16 safetensors index의 SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- rank ablation에서는 해당 64개 Expert가 포함된 **선택된 shard 1개** (`model-00001-of-000004.safetensors`, 8,594,887,408바이트)의 내용 전체를 스트리밍 SHA-256으로 읽고 시작/끝 파일 메타데이터 일치를 확인해 PASS. 그 SHA-256은 `a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082`. **전체 4개 shard의 전체 체크포인트는 검증하지 않았음**.
- 출력 희소화 입력은 무작위 합성 hidden vector. 실제 모델 실행의 hidden activation, router top-k, logits, perplexity, task metric은 아직 측정하지 않음.
- 6개의 Macro-Expert 선택은 실측 라우터 출력이 아니라 고정된 expert ID 0~5를 사용. 활성 채널 비율은 계산비용 비례를 가정한 이론값이며 실제 연산시간 또는 전력 절약이 아님.
- Q2/Q4는 수치 proxy이며 Beglin 고유 qNg64의 에러 피드백·비트 평면 인코딩 및 GPU 커널 검증이 아님.
- Expert별 뉴런 순열 정렬을 실시하지 않음. 함수 수준에서 공통성이 존재하더라도 좌표계가 달라 가중치 저랭크 모델에서 발견되지 않을 가능성이 있음.
- 미관측 Expert는 약 20%의 좌표로 계수를 보정하고 나머지 좌표에서 평가했으며, **Expert 전체를 보지 않고 예측한 zero-shot 결과는 아님**.
- Tupper 일반화는 NxM 마스크를 하나의 수로 표현하는 무손실 부호화일 뿐, 수식 자체가 추가 압축을 만들지 않음.

## 다음 실험 결정

1. **뉴런 순열 정렬 가설 시험**: `gate/up` 행과 `down` 열을 반드시 동반 재배열하고, 학습 좌표와 평가 좌표를 분리한 대조군(원본 순서/정렬/무작위 순서) 비교. 원본 SwiGLU 출력 항등성 PASS가 필수.
2. **입력 활성값 기반 시험**: 기존 승인된 오프라인 자료에서 얻은 실제 layer-1 hidden vectors만 사용. 없으면 그 사실을 `BLOCKED`로 남기고 합성 activation으로 대체하지 않음.
3. **잔차 압축 기준 강화**: 동일 텐서에서 원본 Beglin qNg64(n=2..15)와 공통기저+qNg64를 총 바이트·복원 오차·연산비용으로 비교. 비단조 오차 때문에 bit 폭별 전수 검사.
4. **동적 micro routing은 별도 연구**: 원본 FFN의 일부 채널을 생략하면 현재 50% 설정에서 출력 상대 L2가 62.8~69.1%이므로 승격 금지. 토큰별 게이트 학습·출력 재생성·라우터 안정성·실제 품질 기준 PASS 후에만 실행.
5. **운영 정책**: `main` 불변, live 자동승격 OFF, XOX read-only/isolated runner 유지.

현재 결과는 **압축 엔진 성능 개선 성공**이 아니라, 기존 가설을 실가중치에서 반증한 소중한 대조 실험이다.

## 아티팩트·재현 명세

- 분석 소스 커밋(수정본): `d34e06a7fbd016d596815470ab1bbd3a051778c1`.
- 정량 실험: `experiments/hierarchical_tupper_moe/hierarchical_moe_probe.py`.
- 실가중치 읽기: `experiments/hierarchical_tupper_moe/real_checkpoint_reader.py`.
- XOX 비접촉 실험: `.github/workflows/beglin-hierarchical-real-xox.yml`.
- Rank 변화 + shard 해시: `.github/workflows/beglin-hierarchical-rank-ablation-xox.yml`.
- rank-대조 아티팩트 ZIP SHA-256(다운로드본): `8ce7522b75c504e2e46194b510f5e2fd60abaf8c57dd898540ca50f855436cfa`.
- 64 Expert 원본 실험 아티팩트 ZIP SHA-256(다운로드본)은 GitHub actions 패키지 내용 해시와 별도로 기록; 각 JSON 파일 SHA는 영수증을 재검증함.
- 임의의 64개 Expert 전체를 복원한 4-bit/2-bit 하이브리드 방식의 품질을 주장하지 않으며, 현재는 **투영 가중치 수치 오차 및 합성 hidden-vector 실험**에 한정함.