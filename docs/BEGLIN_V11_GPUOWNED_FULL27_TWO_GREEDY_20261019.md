# BEGLIN v11 — 실모델 27계층 GPU 소유권 비교 (2026-10-19)

**판정: 격리 연구 PASS · 독립 감사 PASS · 운영 승격 금지.** EOE의 SHA 고정 DeepSeek-V2-Lite BF16 원본 체크포인트, qNg64 n7/group64 정밀도 정책, 실제 Apple Silicon C++/Metal 환경에서 검증했습니다. XOX 서빙, EOE 운영 릴리스, 자동 GPU 승격은 변경하지 않았습니다.

## 구현 및 실행 범위

실제 DeepSeek-V2-Lite 27계층에서 prefill과 연속 두 번의 KV 디코딩을 수행했습니다. 모델이 직접 선택하고 입력한 토큰은 **429 → 4281**이며, 그다음 예측은 **6230**입니다(6230은 실제로 다시 입력하지 않았음). 기존 검증된 단일-fused Metal MLA 경로는 BF16 latent 512차원 + 회전된 RoPE K 64차원의 영속 캐시를 사용합니다. 원본 BF16 attention/로그잇 전체 재계산은 독립 수치 비교 경로로만 실행했습니다.

바뀐 부분은 범용 qNg64 선형 가중치의 **GPU 레지스트리 소유권**입니다. v10은 MLX GPU 텐서 외에 C++ 자체 CPU 압축 복사본을 보존했습니다. v11은 uint8 압축 평면·FP32 스케일을 GPU 계산 결과로 명시적으로 materialize/eval한 뒤 C++ 임시 호스트 벡터를 해제하고 GPU 압축 텐서만 C++ 레지스트리에 보관합니다.

**중요: Python 원본 모델의 압축 가중치 배열과 8GiB CPU SHA 압축 캐시는 여전히 존재**합니다. 전체 모델 가중치가 GPU에 상주하거나 프로세스가 GPU 메모리만 쓰는 것은 아닙니다.

## 동일 실모델/동일 8GiB CPU 캐시 정책의 실측

| 연구 프로토콜 기준 | v10 CPU+GPU 이중 복사 | v11 GPU-computed C++ 레지스트리 |
| --- | ---: | ---: |
| 실제 모델 prefill / 첫 디코딩 / 둘째 디코딩 계층 | 27 / 27 / 27 | **27 / 27 / 27** |
| 실제 선택·입력 토큰 | 429, 4281 | **동일** |
| 다음 예측(미입력) | 6230 | **동일** |
| 두 번째 토큰 이후 캐시 vs 원본 비캐시 logits 상대 L2 | 0.00862746973559613 | **동일** |
| 범용 GPU qNg64 모델 역할 | 2,367개 | **2,367개** |
| 실제 범용 Metal GEMV | 8,289회 | **8,289회** |
| GPU 역할 등록 / 적중 / 교체 | 6,723 / 1,566 / 6,629 | **동일** |
| 원본 C 양자화 / SHA 적중으로 생략 | 2,394 / 4,332 | **동일** |
| CPU 압축 캐시 최대 실제 저장량 | 6,794,035,200바이트 | **동일** |
| GPU fused MLA kv_b 27개 압축 가중치 | 53,084,160바이트 | **동일** |
| 7토큰·27계층 영속 latent KV 텐서 | 217,728바이트 | **동일** |
| 전체 연구 실행 시간 | 463.668초 | **464.119초** |
| 최대 프로세스 RSS | 9,315,319,808바이트 (8.68GiB) | **9,133,473,792바이트 (8.51GiB)** |
| C++ GPU 레지스트리의 추가 CPU 압축 복사본 | 있음 | **없음** |
| 일반 GPU 핸들·fused MLA 핸들 해제 | PASS | **PASS** |

두 단일 실행의 전체 시간 차이는 +0.452초(+0.10%)로 유의미한 처리량 향상 근거가 아닙니다. 관측 RSS는 **181,846,016바이트(173.4MiB, 1.95%) 감소**했습니다. 그러나 Python/PyTorch/MLX 메모리 할당·캐시의 실행 간 편차를 배제하지 않았으므로 특정 코드 변경의 보편적 절감량을 보증할 수 없습니다. 영속 MLA KV 자체의 88.75% 저장량 절감과 전 프로세스 메모리 절감률을 혼동하지 마십시오.

이 연구 프로토콜은 모델 원본 BF16 파일을 여전히 계층별로 읽고, 참조 전체 시퀀스 재계산도 포함하므로 정상 서빙 TTFT, TPOT 또는 TPS 기준이 아닙니다.

## 모델/코드 권위 및 독립 감사

| 검증 근거 | SHA-256 |
| --- | --- |
| 원본 BF16 safetensors index | d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf |
| v10 동일 정책 27계층·2토큰 원본 실측 영수증 | 79cd095f4dfe69503a85fb7f37ba69495452067b11659d15864e24568a6a55ee |
| **v11 27계층·2토큰 실제 실행 영수증** | **8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745** |
| **v11 SHA/수치/캐시 독립 감사 영수증** | **d4d05d13fe398ec01d1dc4f7e97b05805fcff8dce59d7f5f7664ab2003763132** |
| EOE 전용 실모델 실행기 원본 | defae1ecb7254e5716b87a0d5b0c99964b0d61d727c708f2942df5010c64eb46 |
| EOE 전용 독립 감사 원본 | ad64bb7b85080bfb786f9775986d5475fcd18707c54fe095c4989f5f5f0839f1 |
| 공개 가능한 v11 C++ GPU-computed 연산 소스 | f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201 |
| v11 네이티브 Metal 라이브러리(EOE 비공개) | a7796a2d96e37150a3b6e0291e8c92acae0a7134deff1a0ea4f7fc0516c653b6 |
| 범용 qNg64 GPU 역할 LRU Python 어댑터 | 32f5eff3308ed0d48ab900ad24994a7a2892fbefcfcec51c1026d4b7e7271e7b |
| 원본 C++/네이티브 C 수치 비교 래퍼 | 0b77dbfab0537d9fd090541b499e29163cddbdae8cb3a4153c6a8dfe09261eff |

독립 감사 최종 반환값:
**PASS_REAL_FULL27_TWO_GREEDY_DUAL_GPU_V11_COMPUTED_GPU_OWNERSHIP_INDEPENDENT**

독립 검증은 세 단계 각각의 27계층 실행, 원본 토큰/로짓 권위, qNg64 압축 커널/라이브러리 SHA, 원본 C++와 네이티브 C 수치 오라클, 256MiB·96핸들 범용 GPU 캐시 한도, 8GiB CPU SHA 메모 및 모든 GPU 핸들 해제를 검사했습니다.

## 재현 코드와 미완료 조건

연구용 portable 디렉터리: research/n7_multi_projection_resident_v1/

- n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp: GPU-computed 텐서 소유권 개선 커널 (운영 후보 아님)
- research_n7_device_lru_cache_v3_GPUonly_candidate.py: SHA 고정 qNg64 역할 캐시 및 제한된 GPU LRU
- research_n7_device_delegate_v1.py: 원본 native C++/C 기준과 모델 역할별 결과 대조
- native_mla_gpu_resident_decode_adapter_v1.py: 원본 MLA kv_b의 별도 GPU 상주·latent KV 캐시 연결 어댑터 (같은 연구 계열의 latent_mla_kv_source_cache_v3.py에 의존)
- 기존 build_and_test_gpuowned_v11_candidate.sh: 실제 Mac/MLX에서 세 독립 프로세스 재현; 합성 시험 3/3 PASS.

원본 체크포인트, 비공개 EOE 경로, 원본 모델 텐서, 은닉값, 라우터가 선택한 전문가 ID, 프롬프트·로그잇 배열 및 비밀키는 GitHub에 올리지 않습니다.

**남은 실서비스 검증**: 임의 동시 실행 중 핸들 해제, 메모리 압박 및 장시간 유출, OS sleep/resume, 다른 MLX/Metal 버전, 여러 독립 프롬프트의 장문 생성과 품질 홀드아웃, 완전 사전양자화된 모델 로더, 실제 TTFT·TPOT p50/p95, 총 CPU+GPU RSS. 단일 실험 결과만으로 운영 승격하지 말고 v10 이중 백업을 fallback으로 보존하십시오.

**배포/병합: 금지. 자동 GPU/live 승격: OFF.**
