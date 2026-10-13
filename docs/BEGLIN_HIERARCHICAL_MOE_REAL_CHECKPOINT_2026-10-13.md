# Beglin 계층형 Tupper Micro-Expert — XOX 실제 체크포인트 검증 결과 (2026-10-13)

## 판정

**REAL CHECKPOINT / READ-ONLY WEIGHT EXPERIMENT: EXECUTED.**
**COMPRESSION AND SPARSE ROUTING QUALITY: FAIL for the tested unaligned, post-hoc baselines.**
**PRETRAINED TOKEN-LEVEL QUALITY: NOT EVALUATED.**
**PRODUCTION PROMOTION: NOT AUTHORIZED.**

요청은 임의 106×17 격자가 아닌 모델 차원 유래의 N×M 격자와 Expert 내부 채널 Assembly를 DeepSeek-V2-Lite 실제 가중치에 적용하는 것이다. 22개의 64-neuron parent와 4개의 16-neuron child로 분해할 수 있다. 모든 child를 더하면 기존 SwiGLU와 같은 함수이나, child를 생략할 때 정확도는 보존되지 않는다.

## 실기기 및 체크포인트

- XOX GitHub Actions self-hosted runner: hostname `XOX.local`, account `xox`, labels `[self-hosted,xox,macOS,ARM64,virtual-character]`.
- XOX 로컬 Safetensors BF16 인덱스 SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- **전체 4개 shard 반복 SHA 인증 PASS**: `checkpoint-verification-v1`, 2 passes, manifest checkpoint identity `1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898`.
- 실제 분석: MoE layer 1, routed Experts 0..63 (64개), hidden 2048, intermediate 1408, `gate/up/down` 실가중치 합계 553,648,128 scalars, 중간행 1408/1408 분석.
- 사전학습 엔진에서 캡처된 토큰 활성값이 아니라 임의 Gaussian hidden vectors로 local SwiGLU 출력 오차를 측정했다. 따라서 **perplexity/로그잇/token-match/실제 Router flip에 대한 주장은 금지**한다.
- qNg64 실제 bit-plane 또는 Metal kernels가 아닌 단순 Q2/Q4 수치 proxy를 사용했다.

## 실행 및 증거

| 목적 | GitHub Actions Run | 결론 |
|---|---|---|
| XOX 8-Expert 읽기 / 전체 행 | [38390892514](https://github.com/popixoxipop-collab/beglin/actions/runs/38390892514) | SUCCESS |
| XOX 64-Expert 전체 | [38391147377](https://github.com/popixoxipop-collab/beglin/actions/runs/38391147377) | SUCCESS |
| 체크포인트 전체 4 shard 이중 SHA | [38391574999](https://github.com/popixoxipop-collab/beglin/actions/runs/38391574999) | SUCCESS |
| Train-48, holdout-16 / rank 2·8·16 | [38391901218](https://github.com/popixoxipop-collab/beglin/actions/runs/38391901218) | SUCCESS |
| 별도 연구 브랜치 calibration-greedy | [38392063275](https://github.com/popixoxipop-collab/beglin/actions/runs/38392063275) | SUCCESS |

위 run들은 업무용 운영 라우터 설정·quant promotion을 변경하지 않는, 체크포인트 읽기 전용 연구 코드에 속한다. 체크포인트 파일 자체의 쓰기 작업은 수행하지 않았다.

## 공유 수학 기저: 실제 64 Expert 실측

| Micro 채널 폭 | rank-2 설명력 (공유 평균 대비 미관측 좌표 MSE 개선) | hybrid/Q4 예상 저장량 | hybrid Q2 잔차/Q4 가중치 제곱오차 |
|---|---:|---:|---:|
| 16 | -0.0044% | 72.32% | 43.86× |
| 32 | -0.0001% | 72.30% | 43.86× |
| 64 | +0.0024% | 72.29% | 43.86× |
| 128 | +0.0034% | 72.28% | 43.85× |

저장량이 적다는 사실만으로 품질 인증이 되지 않는다. **이 미정렬 rank-2 + Q2 proxy는 품질 실패**다. 별도 Train-48 / holdout-16, rank 2·8·16 ablation에서도 설명력 개선은 거의 0이며 rank 8/16은 저장량을 오히려 늘렸다. 이는 현재 parameterization이 부적합하다는 증거지, 가중치에 어떠한 공통 구조도 존재하지 않는다는 증명은 아니다. Expert별 뉴런의 합동 순열과 기능상 등가 정렬을 아직 적용하지 않았다.

## 50% sparse micro-routing 문제: 실제 64 Expert weights

`f_e(h)=Σ_g f_{e,g}(h)` 이므로 K개를 생략한 상대 출력 오차는 본질적으로 `||Σ_{g∉S} f_{e,g}(h)||/||Σ_g f_{e,g}(h)||`에 따른다. 뉴런을 단순히 묶는 것 자체로 항들이 중복되지는 않는다.

| 채널 폭 | 정적 출력 크기순 | Calibration residual-greedy | 테스트 입력 정보 사용 (비실용) |
|---|---:|---:|---:|
| 16 | 68.36% | 68.57% | 36.05% |
| 32 | 69.12% | 69.85% | 43.24% |
| 64 | 62.83% | 62.71% | 50.43% |
| 128 | 65.34% | 63.23% | 50.29% |

- 16-channel 25%, 50%, 75% 유지 시 static 상대 L2 = 87.83%, 68.36%, 47.73%.
- Greedy는 보정용 단일 랜덤 activation에서 각 채널의 출력이 아니라 **남은 출력 잔차 감소량**을 최대화하도록 고른다. 검증용 랜덤 activation에는 최적화를 적용하지 않는다. **결과는 대체로 비슷하거나 악화**되어, 단순 정적 보정만으로 해결되지 않았다.
- 테스트 입력을 본 후 전체 채널을 이미 계산해 중요도를 정하는 방식은 실행 FLOPs를 절감하지 못하고, 전체 출력 최적 subset을 보장하지도 않는다. 이전 보고서에 남은 `oracle_test_relative_l2_upper_bound`라는 키 이름은 지나치게 강한 표현이므로 데이터 이용 방식에 맞게 *test-informed magnitude baseline*으로 해석해야 한다.
- 기여가 거의 독립적이고 에너지가 비슷하면 비활성 비율 1-ρ에 대한 자연스러운 상대 L2 규모는 대략 `sqrt(1-ρ)`. 50% 유지 시 약 70.7%이며 관찰된 63~69%와 규모가 비슷하다. 이는 메커니즘에 관한 해석이지 모델 출력의 독립성 자체를 통계적으로 검증한 결과는 아니다.

## 정확한 다음 작업과 승인 경계

1. **실제 활성값 수집**: 동일 SHA 체크포인트에서 추론용 이산 prompt/토큰에 대한 1번 MoE 레이어 입력 `h`, 실 Top-K Expert ID/확률 및 teacher 출력; 보정/검증 prompt 완전 분리. 별도 shadow worker에서만 수행하고 운영 서비스는 그대로 둔다.
2. **등가 뉴런 정렬**: Expert 간 `gate`/ `up` 행과 `down` 열을 함께 순열한 뒤 동일한 기능을 유지하는지 점검. 미정렬/정렬 비교와 실제 holdout Expert 검증; 기저 rank는 별도 sweep.
3. **학습 가능한 sparse router**: Parent64→child16 계층에서 토큰별 가벼운 gating과 teacher distillation 또는 저비용 잔차 보정. 총 router 비용·보정 비용이 절약한 matmul 비용보다 작아야 하며 토큰 품질은 독립 평가.
4. **qNg64 원본 커널 비교**: 해시 고정 실제 checkpoint에 대해 정확한 qNg64 압축/복원, CPU/Metal parity, 실제 RSS·Latency·throughput, routing flip, token agreement, ppl 측정. Q2/Q4 proxy 통계로 승격 금지.
5. **증거 기반 promotion**: 테스트를 통과한 후보만 artifact immutable + exact source acceptance 후 별도 승인. 현재 `production_write_allowed=false` 유지.

## 상태

- **완료**: XOX remote execution via self-hosted Actions, 실가중치 8→64, channel-group sweep, 체크포인트 전체 SHA(2회), rank ablation, calibration greedy 실험.
- **미완료**: 실제 activation / tokenizer prompt replay / 64 Expert의 true routed weighted output / post-training gating / native qNg64 / GPU speed / 운영 적용.
- **정리**: 일반화된 터퍼 비트 격자는 가변 `N×M` 라우팅 마스크 인코딩이며, 가중치 압축 성능 자체를 제공하지 않는다.

이 문서는 현재 확인된 실패 결과를 감추지 않고 다음 실험의 반증 기준으로 보존한다.
