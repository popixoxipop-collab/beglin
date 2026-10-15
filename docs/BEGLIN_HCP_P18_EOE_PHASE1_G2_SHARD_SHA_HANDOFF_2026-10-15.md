# BEGLIN P18 EOE/D50 — G2 binding, original BF16 first-shard SHA and transfer checkpoint
2026-10-15 KST · research only · operating policy A unchanged

## Real EOE native security and MLX test (PASS)
- P18 480 frozen questions / 673 gold prefixes exact source SHA: c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df
- XOX source C: fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa
- EOE C: f5fab593e588d5a5b34528cc99c518fc2f894186ab46cb50bca5dc1b2e59df5e
- Verified that EOE source differs from original XOX long-context C source ONLY by a single raw capture destination path literal. All 673 source hash allowlists match original frozen bank exactly.
- EOE isolated MLX Metal C/C++ build qwen_infer_gpu SHA: 4b22f80d6368fd8653abcf2b4f3e4d9e1c5fdff8e50ef598d942bbfc1774f176
- Actually executed mlx_binding_transition_test exit code 0; 106 PASS lines, including 4→5→4, 4→6→4, 4→7→4, 7→16→7 and the G2 100-cycle restore soak.
- Actual QA source script SHA: f1f1bce25eabbf06e31b815f4871546b39be1851ac36eb874d2712f7847b544a
- Test log SHA: c1d628903c6727aaed3df4527da52484689b60acf214c4c97776a85a9bf97cdc
- Source/real-MLX QA receipt SHA: feac45c7c3ca83dfa71ec5ef50ee19db791436c05cf027da47250aff9082254a

## Real BF16 original checkpoint validation (PARTIAL, verified first shard only)
- XOX original model-00001-of-000004.safetensors exact size 8,594,887,408 bytes.
- Original XOX full file SHA and copied EOE D50 full file SHA each read independently, exit 0 and match: a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082.
- Further deterministic 12/12 original BF16 tensor-value hashes from the first shard match the sealed P17 5,291 tensor-value manifest SHA 88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda.
- P18 original checkpoint metadata index SHA d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf; config SHA f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d; original modeling SHA 7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c.
- Original BF16 shape: 27 decoder layers, 5,291 source tensors, 4 shards, vocab 102400.
- First-shard evidence receipt SHA f1acf0b489529623975de4ce665ce042e745fe877c8c713eb3bcba8d2c02e2f7.
- Three BF16 shards and whole quantized MoE model are NOT YET certified. Do not treat partial model transfers as complete.

## Running immutable XOX read-only → EOE D50 transfers
- BF16 rsync process proc_edd7dafc6a64ccb31f4b644910df0399.
- Quantized MoE rsync process proc_6223c668c2043cb5d772868c894b1a51.
- EOE D50 root: /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015.
- Do not restart or duplicate transfers. Confirm both process outputs EXITED 0; inspect exact model filenames and sizes with hcp_p18_eoe_d50_model_copy_seal.py --inspect, then call --seal after completion to verify EACH file vs XOX source SHA and create original BF16/MOE immutable source receipts.
- After both receipts, first perform development-only 203 token EOE vs original XOX policy A/B Metal parity, then original BF16 27L full forward and native A/B 102400-F32 logits, and independent raw audit for preregistered next index 3. Do not tune frozen held-out inputs.

## Quality and operating gates
- So far 6/480 real original BF16 and native A/B paired questions from XOX, 8/673 gold positions and 16/1346 native A+B full-vocabulary vectors. EOE actual full-model inference remains UNRUN until exact model transfers complete and pass SHA seal.
- Task complete generation 0/480 certified; warm fixed-work latency 3-trials 0/480; 480-question bootstrap Bonferroni research gate remains BLOCKED.
- Current/live/runtime release g6-0.9.0-operator.900350, converged true, active operation null; original XOX serving stage SHA 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9, route SHA c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47 and PIDs 35481, 35493 unchanged. No production changes; policy A incumbent, policy B auto-promotion OFF.
- Never commit private gold prompt text/token IDs, raw model weights, raw logits or secret env values to GitHub. Only code, hash receipts, summarized docs.
