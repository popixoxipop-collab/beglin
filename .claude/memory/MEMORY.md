# vdsp-engine Memory

08-31: Finance/AEQ 등과 같은 급의 repo-isolation으로 전환 — 이전엔 전역
memory(`~/.claude/projects/-Users-xox/memory/`)에 있던 vdsp_*.md 11개를
전부 여기로 이전. 신규 vdsp-engine 관련 메모리는 앞으로 여기 저장할 것
(전역 저장 시 `repo-isolation-guard.py`가 차단+리다이렉트).

## 연구/조사 결과
- [turboquant_lut_router_sensitivity](turboquant_lut_router_sensitivity.md) — TurboQuant/PolarQuant+Lloyd-Max 코드북: 가중치 재구성은 -8~12% 개선(저장공간 0), 그러나 MoE top-k 라우팅 민감도로 실제 forward-pass는 5/8 argmax(버그 아님, MoE-2b/4c와 동일 메커니즘)

## 장기 목표 / 진행중 트랙
- [vdsp_general_serving_engine_goal](vdsp_general_serving_engine_goal.md) — ★★★ 장기목표: vdsp→Apple Silicon GPU+MoE 서빙엔진(MLX/llama.cpp급). MoE-format safetensors 7단계 CLOSED. V5a/V5b/V5c 정확도게이트 PASS. V5c-fused: Bug1/2/3 전부 근본원인 발견+완전해결 — **KILL-GATE 최종 통과**(2026-08-31, 21.78→~52.91 tok/s, bar 48.34의 109%). V5d(배치 B-token GPU decode): B∈{8..64} 전 구간 **flipped=0**. F-4: global sort/unsort → **B=64 224.3 tok/s, llama.cpp의 1.24배**. V5e(ragged multi-step, prefill+decode 통합설계): 4번째 probe에서 Bug1과 같은 부류의 새 함정(rope offset이 H=1엔 맞고 H=16엔 조용히 틀림) 발견+수정, off-by-one 발견+수정 거쳐 **정확도 56/56 완료**, 메모리 10.05GB(ceiling 12.71GB) 확인. 패턴은 hw-kernel-vendoring 스킬에 일반화 기록. V5f(CPU vs GPU 최종리포트)는 미착수

## 과거 완료/참고 트랙
- [vdsp_kleidiai_sme2_padding_mimicry](vdsp_kleidiai_sme2_padding_mimicry.md) — ARM KleidiAI SME2 프로덕션 통합 SHIPPED. Qwen B=64 +37%, Llama B=16 +26%, 지는 케이스 없음. M1Max가 M4보다 범용성능 우세(SME2 발동구간에서만 M4 역전)
- [vdsp_smallgicp_neon_omp_port](vdsp_smallgicp_neon_omp_port.md) — small_gicp Mac 포팅 + NEON 커널 5종 + libomp, KITTI N=100 회전오차 37%↓ 유의
- [vdsp_complex_pitfalls](vdsp_complex_pitfalls.md) — Apple vDSP 복소 커널 실측 함정 10개(M11~M41) + 판별기준, baseline ppl 11.5759
- [vdsp_m46_python_pipeline_and_d50_incident](vdsp_m46_python_pipeline_and_d50_incident.md) — M46(Python export/quantize/layout 일반화)+M47(Llama-3.1-8B RTN ppl 실측) 완료, RTN갭 67% 해소. 세션중 D50 실데이터 덮어쓰기 사고+복구 포함
- [vdsp_gptq_smoothquant_math_and_status](vdsp_gptq_smoothquant_math_and_status.md) — GPTQ+SmoothQuant 파이프라인 수학원리 + M48(Llama golden gate) 완료현황, 엔진/Python ppl 격차 미해결
- [vdsp_sme2_build_caller_plain_convention](vdsp_sme2_build_caller_plain_convention.md) — SME2 빌드 시 호출자는 항상 plain(-march 없이) 컴파일해야 하는 컨벤션, M4는 SVE를 SME2 스트리밍모드 안에서만 지원(lldb로 확정)
- [vdsp_sme2_lazy_repack_serve_sigill](vdsp_sme2_lazy_repack_serve_sigill.md) — `QWEN_SME2_LAZY_REPACK=1`+serve모드 조합 SIGILL, GGUF무관 기존코드에서도 재현(미해결, 우선순위 낮음)
- [vdsp_sme2_paper_codex_review_final](vdsp_sme2_paper_codex_review_final.md) — vdsp SME2 arXiv 논문 동료검증 이력, Codex 10라운드 후에도 독립 리뷰가 §5.1 수치모순(3~5배) 새로 발견, 실측 재검증으로 수정완료. "충분검증" 판단은 잠정적이라는 교훈
- [vdsp_m42_complex_quant_no_go](vdsp_m42_complex_quant_no_go.md) — 브랜치 m42-complex-w2의 복소 widely-linear W2 양자화 결론=NO-GO, 미머지 방치. ★main의 M42(구조적 일반화, 별개 작업)와 이름만 같음 주의
- [vdsp_m42_structural_gen_qwen_score_spec](vdsp_m42_structural_gen_qwen_score_spec.md) — main에 머지된 M42(qwen_score.c/qwen_spec.c 구조적 일반화, M41 후속), 완료·독립검증·반영. m42-complex-w2의 동명 M42(NO-GO)와 다른 작업
