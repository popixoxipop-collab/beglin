---
name: vdsp-smallgicp-neon-omp-port
description: vdsp의 NEON/GICP 경험을 small_gicp Mac 포팅 + 5종 커널 NEON 구현 + libomp 활성화로 실측 증명한 작업 (KAIST 심현철 랩 관심사 직결)
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f0b2ea6-a78c-4f22-b665-47f910037aae
  modified: 2026-08-24T04:40:02.983Z
---

Mac Studio(M1 Max, 8P+2E)에서 `small_gicp`(Kenji Koide, AIST)를 브루 없이 순수 pip 격리 툴체인(cmake+ninja venv)으로 빌드·검증하고, KAIST 심현철 교수 논문(Enhancing State Estimator for Autonomous Racing)의 5종 커널 공분산 추정(RBF/Gaussian/Polynomial/HI/Laplacian)을 ARM NEON으로 직접 구현·검증·벤치마크했다. 두 작업 모두 golden-oracle 검증(스칼라 레퍼런스 대조, 알려진 rigid transform 복원) 통과 후에만 속도 측정.

**결과**:
- small_gicp self-test: 5° Z회전+평행이동 알려진 변환 복원, 회전오차 0.0016°/변위오차 0.25mm — PASS
- libomp: 이미 Homebrew Cellar에 keg-only로 존재(`/opt/homebrew/Cellar/libomp/22.1.8`) → 소스 빌드도 `brew link`도 없이 CMake 플래그로 직접 경로 지정. `otool -L`로 실링크 확인, self-test 재검증(수치 완전 동일) 후 20만 포인트 GICP A/B 측정: **941.82ms → 202.10ms, 4.66배**
- NEON 5커널: 전부 스칼라 대비 오차 ~1e-16(배정밀도 반올림 수준)로 PASS. 속도향상은 1.12~1.55x — exp/pow 등 초월함수는 NEON 네이티브 명령이 없어 거리/내적/min-sum만 벡터화하고 초월함수는 레인별 scalar libm 호출로 처리(직접 벡터화 exp 근사는 검증 리스크 때문에 의도적으로 안 함). 초월함수가 아예 없는 HI 커널만 진짜 end-to-end 벡터화돼 1.55x로 확실히 앞서는 패턴 확인 — 설계 선택과 결과가 일치.

**후속(같은 세션)**: SLEEF 실소스(`sleefsimddp.c`의 `xexp()`) 직접 확인해 Cody-Waite 범위축소+10차다항식 구조 파악, SLEEF 자체가 macOS+AArch64를 공식 미지원("❌")임도 확인. 대안으로 Apple Accelerate `vvexp`/`vvpow`(vForce)를 이웃 k=20개 배치로 묶어 호출하는 3번째 경로 구현·실측: **pow 기반 Polynomial 커널만 확실히 이김(1.13x→1.45x), exp 기반 3개(RBF/Gaussian/Laplacian)는 배치 오버헤드가 이득을 상쇄해 거의 무의미하거나 RBF는 오히려 근소하게 느려짐(1.07x)** — "vForce가 항상 낫다"가 아니라 "초월함수 자체가 무거울 때만(pow) 배치가 이긴다"는 게 실측 결론. ★이전 턴에 "vForce가 정답"이라 단정했던 걸 실측이 정정한 사례.

**후속2(정확도 비교, 같은 세션)**: 논문 Table V(RMSE 비교)를 흉내내려 small_gicp의 `PointCloud::cov(i)`(직접 주입 가능 확인) + `KdTree::knn_search` + `align(target,source,target_tree,...)`(내부 공분산 재추정 안 함, 커스텀 covs 그대로 사용) 조합으로 실제 GICP 등록 정확도 실험. 평면구조 있는 합성 "방"(바닥+벽2, N=4800)에 노이즈 스윕.

★★ **1차 결론(고노이즈서 커널가중이 baseline보다 훨씬 덜 강건하다)은 정정됨 — D7에서 뒤집힘.** 저노이즈(σ=0.01/0.02) 차이없음, 중노이즈(σ=0.08) 커널이 baseline보다 2배 정확까지는 유효. 고노이즈(σ=0.15)서 4/5커널 0/8붕괴를 처음엔 "커널가중=분포이탈에 덜 강건한 통계적 트레이드오프"로 결론냈으나, small_gicp 소스(`util/normal_estimation.hpp`의 `CovarianceSetter`) 직접 확인 결과 **baseline은 실제 계산된 고유값을 아예 안 쓰고 고정값(1e-3,1.0,1.0)으로 강제치환**(GICP논문 표준 평면정규화, 조건수 항상 정확히 1000)하는데, 내가 짠 5커널은 이 정규화 없이 원본 가중공분산을 그대로 썼던 게 원인이었음. 원본 조건수를 직접 측정(`kernel_weight_diagnostic.cpp` D7 확장)하니 노이즈 0.02→0.15서 ~110→~3.5로 실제 붕괴(거의 등방성) 확인 — baseline은 데이터와 무관하게 항상 1000이라 안 무너짐. **같은 고정 고유값 정규화를 커널가중 축(고유벡터)에 얹어 재실험하니 4/5커널 붕괴가 전부 사라지고 baseline과 대등한 수준으로 복귀**(Gaussian/Laplacian 8/8, baseline 7/8보다 오히려 나음). 즉 원인은 "커널가중의 본질적 약점"이 아니라 "내 구현이 baseline에 있는 표준 정규화 스텝을 빠뜨림"이었음.
- 정규화 적용 후 잔여차이: Polynomial만 유일하게 baseline보다 뚜렷이 나쁨(6/8, 0.602° vs baseline 0.490°) — 논문의 실제 발견(Polynomial 압도적으로나쁨, rmse=164.93 vs 나머지~10-11)과 방향은 일치하나 N=8trial이라 표준편차(~0.2)가 평균차(~0.1)보다커서 통계적 확정은 아님.
- **교훈**: 1차 결론을 메모리에 못박기 전에 메커니즘까지 규명해야 함(가설→반증→재가설 2단계 필요했음). CLAUDE.md 세션이력 규칙("완료/확인됨 주장은 가설로 취급, 재검증")이 내 자신의 직전 턴 결론에도 그대로 적용된 실사례.

**후속3(N 늘려 통계적 확정, 같은 세션)**: rng가 trial마다 커널 6개 전부 동일 seed(1000+trial)로 재시작 — 이미 paired 설계였음을 활용해 baseline 대비 paired t-test 구현(수동, N≥30이면 |t|>1.96→p<.05 정규근사). N=8→30→100 스윕:
- N=8: Polynomial만확실히나쁨(6/8,0.602° vs baseline 0.490°)처럼보임
- N=30: 정반대로 RBF/Gaussian/Laplacian이 회전오차서 baseline보다 **유의미하게 낫다**(p<.01~.05), Polynomial/HI는 변위오차서 유의미하게나쁨(p<.05~.01) — 처음과 다른 패턴
- **N=100(최종): 전부 n.s.(유의미한차이없음).** 효과크기(mean_diff)도 N커질수록 0으로수렴(RBF회전오차 -0.132→-0.034) — 실효과 확정이아니라 소표본 허위양성이었다는 통계학적 시그니처.
★★★ **최종 확정 결론**: 정규화(D7) 적용 후, 이 합성 "방" 데이터셋에서는 5커널 중 어느 것도 baseline 대비 통계적으로 유의미한 정확도 차이 없음(N=100, paired t-test). 논문의 극적차이(Polynomial 15배 나쁨)는 이 단순 합성 평면 셋업에 없는 실제 LiDAR 특성(점밀도불균일/동역학/거리별노이즈)에 기인할 가능성. N=8→30→100 3단계 전부 다른 결론을 냈다가 최종수렴한 과정 자체가 "성급한 결론을 데이터로 스스로 걸러낸" 좋은 방법론적 사례.

**후속4(실측 KITTI LiDAR, 같은 세션)**: HeLiPR(SNU김아영랩)은 신청폼+승인대기 필요해 보류, 대신 Academic Torrents의 **KITTI raw drive `2011_09_29_drive_0071_sync`(4.34GB, 등록불필요, CC-BY-NC-SA)**를 libtorrent(pip venv 격리설치)로 확보 — Velodyne 1059프레임(1.9GB)+OXTS(GPS/IMU) pose. 포맷 전부 pykitti(utiasSTARS/pykitti) 실소스로 검증(추측안함): `.bin`=float32[x,y,z,reflectance], OXTS 30필드(앞6개=lat,lon,alt,roll,pitch,yaw), pose=Mercator투영+Rz·Ry·Rx. IMU→Velodyne 외부파라미터는 **2011_09_26 캘리브레이션 실측값으로 근사**(2011_09_29 전용 파일은 공개미러 못찾음, 한계로 명시). leaf_size=1.0m(논문 파라미터 재현), gap=5프레임, small_gicp `voxelgrid_sampling`+`align()` 재사용.
- N=3(예비): 방향성이 논문과 일치하는 조짐(Laplacian/RBF/Gaussian이 baseline보다 낮은오차, Polynomial/HI가 높음) — 합성데이터 N=8→30→100 함정 반복 안하려 즉시 N=100으로 확인
- **N=100(최종, paired t-test): 합성데이터와 정반대로 실제로 유의미한 차이 확인.** 회전오차: RBF(-0.042,p<.01)/Gaussian(-0.041,p<.01)/Laplacian(-0.043,p<.01)이 baseline(0.113°) 대비 **약37% 더 정확**, Polynomial(-0.002)/HI(-0.001)는 baseline과 차이없음(n.s.). 변위오차는 전부n.s.이나 RBF/Gaussian/Laplacian표준편차가 평균보다커서(outlier trial의존 추정) 결론보류.
★★★ **최종 결론**: 합성 평면데이터(N=100,전부n.s.)에선 안 보이던 커널선택 효과가 **실제 센서데이터(N=100,p<.01)에선 통계적으로 확실히 존재** — "실측 데이터의 어떤 특성이 차이를만드는가"라는 원래 질문에 실증적 답 확보. 논문 순위(Laplacian<RBF<HI<Gaussian≪Polynomial)와 부분일치(Laplacian/RBF가 좋다는건 맞음)하나 완전일치는아님(내 결과는 HI가 Polynomial과 묶여 "개선없음" 그룹, 논문은 HI가 Gaussian보다 나음) — 캘리브레이션 근사+leaf_size외 미기재 GICP하이퍼파라미터 차이가 원인일 수 있음, 정직하게 "재현 아님, 부분적 방향일치"로 취급.

**후속5(변위오차 outlier 원인규명, 같은 세션)**: N=100 결과에서 RBF/Gaussian/Laplacian만 변위오차 표준편차가 평균보다 큰 게 미해결로 남아있었음. per-trial dump로 원인 추적:
- worst 변위오차 trial이 baseline/Polynomial/HI는 {51,48,52,50,67}(전부 <0.64m)인데 RBF/Gaussian/Laplacian만 별도로 {44,86,87,88}에서 **2.7~4.1m 파국적 실패**가 따로 존재 — 3개 가설 순차검증: ①급회전(OXTS yaw거의일정)기각 ②성긴포인트(raw count 정상범위)기각 ③along-track축퇴(회전오차는0.09~0.22°정상인데변위만튐)→**확인**. GICP전형적실패모드(반복/자기유사구조에서진행방향변위만국소최적해에갇힘)의서명과정확히일치.
- ★해석: baseline의 고정형(1e-3,1,1) 정규화 공분산은 국소방향에 과확신하지 않아 이 함정을 피하는데, 데이터적응형 커널가중(특히 exp기반 거리커널)은 반복구조에서 확신이 오히려 독으로 작용해 이 특정 구간에서만 빠지는 것으로 추정(시각화 없이 스칼라 지표만으론 이 이상 확정 어려움, 정직한 한계로 명시).
- 평균 비교(37%↓,p<.01) 자체는 여전히 유효 — 이번 결과는 "왜 변위오차 분산이 컸는가"에 대한 완결된 인과사슬 규명(가설 2개 반증+1개 확인)이지 앞선 결론을 뒤집는 게 아님.

**Why**: [[project_grad_school_prep_2026]] 참조 — 사용자가 "자율주행 쪽 SLAM/EKF 연장선"으로 대학원 방향을 명확히 하면서, KAIST 심현철 교수 논문에 나온 "GICP 스캔매칭 CUDA 가속 + 5종 커널 공분산 비교"에 흥미를 느낌. vdsp에서 이미 검증한 "golden-oracle 먼저, 그다음 벡터화/가속" 방법론을 완전히 다른 도메인(LLM 추론 → 포인트클라우드 등록)에 그대로 적용해서, 연구계획서/CV에 "관심 분야와 실제로 맞닿은 기술 경험"이라는 직접 증거로 쓸 수 있는 결과물이 됨.

**How to apply**: 연구계획서(`~/Desktop/research_statement_content_draft.md`)나 심현철 랩 컨택 메일 작성 시 이 결과를 구체적 근거로 인용할 것. 코드 위치: `~/projects/small_gicp/`(build=OMP켜짐/build_noomp=꺼짐 둘 다 보존), `~/projects/kernel_covariance_neon.cpp`, `~/projects/small_gicp_selftest.cpp` — 전부 Mac Studio, 아직 git 추적 안 됨(로컬 파일만). 필요시 정식 repo화 고려.
