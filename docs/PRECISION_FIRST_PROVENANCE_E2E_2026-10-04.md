# First provenance-backed precision E2E proof

Implementation source: f52e674cc3f9833bc67628c4cfe20a21154bb276.

A fresh capture of the original P5 L3 risk fixture preserved the exact raw replay input. On the same fixture, shared_up/L3 n6 emitted [55222,372], n5 emitted [55222,1], and returning to n6 emitted [55222,372]. Raw token SHA: 9a0dfec8bb5d047acb340a7d61fa2467624f2f2b5b95398e2a3d60211357aab7.

The first P9 attempt exposed a real orchestration bug: shadow applied only the changed L3 target and omitted the resident baseline shared_down/L26 n5 policy. That produced three false G4 rejections. A direct full-policy cold-start probe proved L3 n5 + L26 n5 emits the corrected token 1.

The fix makes gpu_autopilot baseline-aware: PRE runs the full baseline policy and POST changes exactly the selected target while retaining all unchanged resident precision rows. Isolated PRE/POST workers no longer compare their independent startup epoch counters; exact baseline/requested/applied policy hashes remain mandatory.

Exact-source repeated P9 shadow after the fix:
- /Users/xox/vdsp_serving/p9-l3-full-context-exact-f52e674.json
- SHA-256 cc9e03e040473c52d95ead51a05df9ae9dfb86743edc34ff813adf5f9ea6ab03
- 3/3 SHADOW_ADMITTED
- certification status MANUAL_REVIEW_CANDIDATE

P10 dry-run materialization:
- /Users/xox/vdsp_serving/precision-e2e-first-candidate-f52e674/result.json
- SHA-256 9b7ffe4059388f47553c75994f074edceef17a84956e81feea7b8df3ef806431
- G4 verdict SHA-256 339d13bbfbaa95572938f1fcee21cb2f009200d6eed9892e336310674c1081f7
- G6 canary SHA-256 c439ed8e9316432391211083af972eda4400812f950b1a8afec2bed247801188
- status READY_FOR_RUNTIME_PREIMAGE_MATERIALIZATION
- production_cutover_allowed=false
- trusted approval created=false
- production cutover performed=false

Regression: focused baseline-context/P9 tests 26/26 PASS; GPU suite 231/231 PASS; Certified successor regression run 37132981272 SUCCESS on exact implementation head f52e674cc3f9833bc67628c4cfe20a21154bb276.
