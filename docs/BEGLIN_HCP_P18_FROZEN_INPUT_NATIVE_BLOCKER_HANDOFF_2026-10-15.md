# BEGLIN P18 — 외부 480문항 입력 봉인 및 네이티브 차단 핸드오프
2026-10-15 · popixoxipop-collab/beglin · branch research/hcp-p18-external-multimetric-20261015

## 판정
P18_PREREGISTERED_INPUTS_AND_CONTRACT_PASS_NATIVE_EVAL_BLOCKED
P18 원본 BF16 교사 추론 0회, A/B 새 네이티브 추론 0회, 정답률/지연시간 실측 0회.
운영 정책 A 유지, B 승격 차단. 본 작업에서는 Tailnet release 활성화, 운영 라우트·바이너리 변경 없음.
확인 당시 current/live/runtime = g6-0.9.0-operator.900350, converged=true, active_operation=null.

## 검증한 사실

1. XOX 원본 DeepSeek-V2-Lite model.safetensors.index.json SHA256 = d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf. XOX tokenizer.json SHA256 = 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669. D50의 3개 토크나이저 파일은 원본과 해시 일치.
2. P17 봉인된 12문항: prompt/full-context/gold 모든 token ID 12/12 재현. 영수증 SHA = 3b7f99193e26c7bc6bca6c9cf8252256b880ed97f66e15d94836fa06ee0ad5f9.
3. P18 외부 MMLU/ARC-Challenge/GSM8K test 각160, 총480: 0-shot BPE 경계 480/480 안정, empty gold=0. 0-shot 검사 SHA d7442fde2f9df6eaea8b36a821d7ec10069829819f58d5fe80230123e665012e.
4. 별도 공개 dev/validation/train 분할 5개씩 총15개 few-shot 데모를 고정, test 문항과 질문 문구의 정규화 해시로 중복 차단. MMLU dev parquet SHA 2b19bde1ed8ca6b482fb283abc90e8e0d9d228947029c0b91795d64b28b3bc3f, ARC validation SHA 395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05, GSM8K train SHA ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d. dev manifest SHA 2d9f6e63fe49cfe646493f0e3e79870ff2eca34adac5081b3bb8fb305b28d836.
5. Seed 20261015, 0/3/5-shot과 8/32/128 토큰 길이 교차 배정. 최종 frozen 480 입력의 토큰 배열 SHA c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df. 프롬프트 길이 38~931, 금본 정답 길이 1~6 토큰, 가장 긴 프롬프트+생성 cap = 963.
6. 독립 감사가 15개 dev 원본과 480개 test 원본의 레코드/정답/토큰 배열을 재계산하여 전수 PASS. 감사 SHA 4e43064669116c9e4c670f969e230c0ddb29a1535f9b191c89f68f9be0a91ea1.
7. 품질 게이트용 private cohort SHA 19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65. frozen 통계 프로토콜 SHA 77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c.
8. 초기 gate 구현 오류: 데이터셋 Git 커밋 40자리 SHA-1을 64자리 Parquet SHA-256으로 검사하던 결함 발견·수정. 수정 게이트 SHA 237d6ce68c714f8e5098bf212796bde965abc669f3fc03147eb50ffd9bba18de, 테스트 SHA 61d40e514b9ab007bcc4ea74d1ad74185407a107e71904df8495831c25b848b9. 실제 frozen 계약 PASS, 변조·누락 9/9 거부, 합성 QA 25/25 PASS. 계약 영수증 SHA 80d39b127ef6b9606ea6066cca5d4af5fcf52e6ed60bfe3dfb9c0d5c273090c4. 초기 오프라인 QA는 구 구현 검사이므로 최신 구현 QA와 혼동하지 않는다.

## 실제 XOX 런타임 한계

XOX의 P17 native C SHA d724af1eb97ee715e206188fa120547f4cc1f18f55a3c37938b6b5a4ce9986d2, MLX SHA ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f 확인.
- C MOE_CBATCH_MAXPOS = 128
- C MOE_MAXPOS = 160
- MLX MLA_L0_MAXPOS = 128
- 기존 128 캐시 및 P18 계획 생성 cap을 함께 수용할 수 있는 문항은 82/480.
- 최대 필요 위치는 963이므로 4096까지 무조건 늘릴 필요 없이 1024위치 scratch 연구 확장을 목표로 할 수 있음. 다만 C/MLX cache position 차원이 128→1024로 8배 증가하며, 캐시 allocator 및 14GiB RSS watchdog 등 전수 검증 전 매크로만 수정하는 위험한 승격을 금지.
- XOX SSD free 약17.671GiB, 보호 reserve20GiB 대비 약2.329GiB 부족. 현재 read-only admission 영수증 SHA c891f6fedef4cb520955328e97f498d729066faa3e2befb181c841fb067b1d4b.
- XOX Hugging Face hub 캐시 약48GiB 이상에 금융/음성/추론 관련 다른 작업 모델 포함. 보존·사용 여부 확인 없이 삭제하지 않음. P17 전용 scratch 약72MiB로 삭제해도 보호 예약 회복에 불충분.
- P17 native 원본 qwen_infer.c와 mlx_moe.cpp를 D50 native_p1024_original 폴더에 원본 SHA 보존한 읽기 전용 참고용으로 복사. 실서비스에 영향 없음.

## 통계 계약

B−A NLL 질문 단위 층화 부트스트랩 단측 상한 <=+0.05 nats/token; B−A 정답률 단측 하한 >=−0.05; paired B/A warm latency geomean ratio 단측 상한 <=1.10; 경험적 p95 ratio <=1.15. 10,000회 seed20261015, Bonferroni 가족 오류율 0.05. BF16 positive-control score floors MMLU30%, ARC30%, GSM8K10%. 480개 질문 단위, 각 평가셋160개. P18 gate PASS라도 운영 승격 자동화는 항상 false.

## 데이터 경로 / 금지 범위

EOE D50 작업 디렉터리: /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015
원문 문항/정답, 토큰 ID, Parquet, 토크나이저 파일은 D50 private 보존, GitHub 업로드 금지. GitHub에는 코드/검증 리포트/집계 영수증/frozen 계약만 저장. P17 원시 logits 및 운영 라우트·바이너리·릴리스 불변.

## 실제 추론 전 다음 작업

1. 1024위치 scratch-only 네이티브 C/MLX 확장 설계, 동적/희소 K/V 메모리 또는 프로파일별로 할당하는 경로 검토. C allocation과 MLX fused KV 및 14GiB RSS 상한을 고려하여 128/159/160/963/1023/1024 위치 양·음성, 원본 128 parity, 실제 Metal ACK를 확인.
2. XOX 보호 SSD reserve20GiB를 넘길 수 있도록 소유권·사용 참조·SHA가 확인된 별도 캐시만 정책적 아카이브. 다른 작업 모델, P17 sealed 증거 삭제 금지.
3. 독립 BF16/네이티브 원시 full-vocab logits 감사, 정답 채점, 고정 work latency audit 구현·검증 후 실제 480회 paired 측정.
4. 채점 결과를 사전 고정된 frozen 프로토콜과 SHA에만 투입해 최종 연구 판정. 정책 B 운영 승격은 별도 사람 승인 이전 금지.

결론: P18 데이터 및 평가 계약 PASS. native 1024문맥과 XOX SSD 보호 게이트 BLOCKED. 모델 실측 0회.
