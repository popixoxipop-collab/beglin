# BEGLIN P18 — 개발용 공유 BF16 실기기 승인 게이트 (2026-10-16)

## 현재 판정

**READY_FOR_REAL_DEVONLY_VALIDATION; REAL_27L_SHARED_PARITY_UNRUN**.

이 문서에서 실제 구현/검증한 사실과 아직 수행하지 못한 작업을 구분한다.

- EOE Tailnet 원격 연결은 한 차례 복구되었으며, 당시 operator `g6-0.9.0-operator.900352`, converged=true, active_operation=null이었다. 이후 간헐적 transport timeout/HOST_UNREACHABLE 발생.
- EOE/D50에 사용자 제공 QA 패키지의 **원본 바이트 SHA 완전 일치**를 확인하고 배치했다:
  - `hcp_p18_shared_bf16_gold_window_v2.py`: SHA-256 `c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f`.
  - `hcp_p18_devonly_shared_bf16_gold_v2_qualify.py`: SHA-256 `5b50cd59f7b056177a5ef696e8603299c722081ae00f6d21ba634d15af2620a1`.
- 예비 SHA/AST 검사 PASS. QA 패키지 4개 합성·정적 테스트 묶음 **45/45 재실행 PASS**. 실제 DeepSeek 모델의 공유 가중치 batch forward는 **실행되지 않았음**.
- EOE 재개용 독립 staged verifier 준비: `hcp_p18_devonly_staged_shared_bf16_parity.py`, SHA `cf5b20128bfa93d151d24eb1969397fa9e1def35025c13f295b392a953f6f7ff`. *정적 검증만 완료*, 런타임/실모델 PASS를 주장하지 않는다.
- EOE CPU: 10 논리 코어에 1분 load 11.13→14.80; `resource_plan` CPU 2/1/0.5 모두 **insufficient_free_cpu**, free_after_job CPU=0. 자원 강행/관리 프로세스 종료 금지.
- XOX 여유 디스크 약 19.3GiB로 20GiB 보호선 미달. XOX는 절대 대체 실험 출력 저장소로 사용하지 않는다.

## 재개 시 정확히 수행할 27계층 검증

D50 canonical root:
`/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`

**SOURCE SCOPE**: SHA `b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c`로 봉인된 기존 203토큰 development_only_not_480_test 입력만. `prompt_length=101,198`/pseudo-gold `3,5`; 원본 480 시험 문항이나 답변 금지.

- EOE resource_plan이 CPU2 RAM9GiB와 D50 reserve를 승인한 경우에만 실행. Guard: peak RSS 9GiB, new swap 1GiB, host available RAM 최소11GiB, D50 100GiB 최소 가용.
- 이력 손실 위험을 줄이기 위해 4개 1회성 source stage를 순서대로 실행. 각 stage 완료 후 SHA 봉인 receipt와 `.npy`를 D50에 보존. 서로 중복 실행하지 않는다.

```
/opt/homebrew/bin/python3.11 <root>/hcp_p18_devonly_staged_shared_bf16_parity.py --stage standalone_short
/opt/homebrew/bin/python3.11 <root>/hcp_p18_devonly_staged_shared_bf16_parity.py --stage standalone_long
/opt/homebrew/bin/python3.11 <root>/hcp_p18_devonly_staged_shared_bf16_parity.py --stage shared_short_long
/opt/homebrew/bin/python3.11 <root>/hcp_p18_devonly_staged_shared_bf16_parity.py --stage shared_long_short
/opt/homebrew/bin/python3.11 <root>/hcp_p18_devonly_staged_shared_bf16_parity.py --stage audit
```

여기서 `<root>`는 상단의 canonical D50 경로로 대체. 마지막 audit은 모든 original full-vocab F32 logits에서 **모든 8 gold positions × 102,400 logits × 정·역순**의 비트 단위 동일성 및 measured speedup을 비교해야 함. 어느 한 값이라도 다르면 batch 사용 불가. 이 실행 스크립트 자체도 실기기 시험 전 개발용으로 간주하며, 실제 stage 첫 실행 시 추가 read-only source contract 점검이 필요.

## 본 프로젝트 품질 현황(변경 없음)

- XOX BF16+native A/B paired gold: **6/480문항** (8/673 source gold positions, 16/1346 A+B raw float32 vectors).
- EOE BF16-only 추가 4문항으로 중복 제거된 원본 BF16 출처 **10/480문항** (선행 완료 기준).
- 본 공유 배치 실험은 기존 480문항 평가 결과, task accuracy, latency 및 Bonferroni/cluster bootstrap을 대신할 수 없다.
- EOE 공식 복제 가중치 승격 영수증은 별도 fail-closed gate이고 생성 미완료. Native A/B production promotion=false, incumbent A 유지.

**결론:** 45/45 synthetic/static PASS, dev-only real-model source and resumable staged QA prepared, **CPU admission BLOCKED**, original 27-layer bit-exact shared validation UNRUN, performance gain UNMEASURED. 운영 변경 없음.
