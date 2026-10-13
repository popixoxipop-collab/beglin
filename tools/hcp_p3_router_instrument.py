#!/usr/bin/env python3
"""Patch only a per-run scratch C source copy at the actual post-full-softmax rank-6/7 stage.

Never patch production or source checkout. Capture all 26 routed layers for
DeepSeek-V2-Lite, before MOE_NORM_TOPK_PROB renormalizes the selected weights.
"""
from __future__ import annotations
import argparse,hashlib,pathlib,re

HOOK = r'''
            // P3 RESEARCH ONLY: complete-softmax rank-K and first unselected
            // score. Runs AFTER TopK selection and BEFORE TopK normalization.
            if (getenv("BEGLIN_P3_ROUTER_BOUNDARY")) {
                const char *boundary_file = getenv("BEGLIN_P3_ROUTER_BOUNDARY");
                if (MOE_N_EXPERTS != 64 || MOE_TOP_K != 6)
                    { fprintf(stderr,"P3_BAD_ROUTE_TOPOLOGY\n"); exit(2); }
                float kth_score = 2.0f, next_score = -1.0f;
                int kth_id = -1, next_id = -1;
                for (int ee = 0; ee < MOE_N_EXPERTS; ee++) {
                    int is_selected = 0;
                    for (int kk = 0; kk < MOE_TOP_K; kk++)
                        if (top_idx[kk] == ee) { is_selected = 1; break; }
                    float score = router_scores[ee];
                    if (!isfinite(score) || score < 0.0f || score > 1.00001f)
                        { fprintf(stderr,"P3_INVALID_FULL_SOFTMAX\n"); exit(2); }
                    if (is_selected) {
                        if (score < kth_score) { kth_score = score; kth_id = ee; }
                    } else {
                        if (score > next_score) { next_score = score; next_id = ee; }
                    }
                }
                if (kth_id < 0 || next_id < 0 || kth_score < next_score - 1e-7f)
                    { fprintf(stderr,"P3_RANK_CONTRACT_FAILED\n"); exit(2); }
                double ent = 0.0, probability_sum = 0.0;
                for (int ee = 0; ee < MOE_N_EXPERTS; ee++) {
                    double probability = router_scores[ee];
                    probability_sum += probability;
                    if (probability > 0.) ent -= probability * log(probability);
                }
                if (fabs(probability_sum-1.0)>1e-3)
                    { fprintf(stderr,"P3_BAD_SOFTMAX_NORM\n"); exit(2); }
                float gap = kth_score-next_score;
                float relative_gap = gap/fmaxf(kth_score, 1e-30f);
                float log_ratio = logf(fmaxf(kth_score,1e-30f)) -
                                  logf(fmaxf(next_score,1e-30f));
                FILE *evidence = fopen(boundary_file,"ab");
                if (!evidence) { fprintf(stderr,"P3_CAPTURE_OPEN_FAILED\n"); exit(2); }
                int w = fprintf(evidence,
                    "pos %d layer %d kth %d runnerup %d p6 %.9g p7 %.9g relgap %.9g logratio %.9g entropy %.9g\n",
                    pos, l, kth_id, next_id, kth_score, next_score,
                    relative_gap, log_ratio, (float)(ent/log((double)MOE_N_EXPERTS)));
                if (w < 10 || fclose(evidence) != 0)
                    { fprintf(stderr,"P3_CAPTURE_WRITE_FAILED\n"); exit(2); }
            }
'''

FFN_ANCHOR = ("        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);\n"
              "        moe_rmsnorm(x, w_postln, h2, MOE_HIDDEN);")
H2_HOOK = r'''
          // P3 scratch-only pre-FFN vector capture; exactly layer 1.
          if (l == 1 && pos >= 0 && pos < 40) {
              const char *cap = getenv("BEGLIN_CAPTURE_PRE_FFN_INPUT");
              if (cap && cap[0]) {
                  FILE *fp = fopen(cap, "ab");
                  if (!fp) { fprintf(stderr,"CAPTURE_OPEN_FAIL\n"); exit(2); }
                  int hdr[3] = {pos, l, MOE_HIDDEN};
                  if (fwrite(hdr,sizeof(int),3,fp)!=3 ||
                      fwrite(h2,sizeof(float),MOE_HIDDEN,fp)!=(size_t)MOE_HIDDEN) {
                      fclose(fp); fprintf(stderr,"CAPTURE_WRITE_FAIL\n"); exit(2);
                  }
                  if (fclose(fp)!=0) exit(2);
              }
          }
'''

def patch(source:str)->str:
    if "BEGLIN_P3_ROUTER_BOUNDARY" in source:
        raise ValueError("original source already instrumented")
    h2_sites=[m.start() for m in re.finditer(re.escape(FFN_ANCHOR),source)]
    correct=[i for i in h2_sites
             if "float *mlp_out = g_mft_mlp_out;" in source[i+len(FFN_ANCHOR):i+len(FFN_ANCHOR)+160]
             and "moe_attention(af, t, l, pos, h, x)" in source[max(0,i-5000):i]]
    if len(correct)!=1:
        raise ValueError(f"BLOCKED h2 anchor qualified={len(correct)}")
    at=correct[0]+len(FFN_ANCHOR)
    source=source[:at]+"\n"+H2_HOOK+source[at:]
    anchor=("            int *top_idx = g_mft_top_idx;\n"
            "            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);\n"
            "            if (MOE_NORM_TOPK_PROB) moe_topk_renorm(router_scores, top_idx, MOE_TOP_K);")
    if source.count(anchor)!=1:
        raise ValueError(f"BLOCKED post-softmax and pre-renormalization site count={source.count(anchor)}")
    return source.replace(anchor,
        "            int *top_idx = g_mft_top_idx;\n"
        "            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);\n"
        +HOOK+"\n"
        "            if (MOE_NORM_TOPK_PROB) moe_topk_renorm(router_scores, top_idx, MOE_TOP_K);")

def instrument(root:pathlib.Path):
    c=root/"qwen_infer.c";cmake=root/"CMakeLists.txt"
    orig=c.read_text();updated=patch(orig)
    cc=cmake.read_text()
    anchor="add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n"
    if cc.count(anchor)!=1:raise ValueError("BLOCKED qng64 build anchor")
    c.write_text(updated)
    cmake.write_text(cc.replace(anchor,anchor+"  qng64_dense.c\n"))
    print("P3_SOURCE_PATCHED",{"original_sha256":hashlib.sha256(orig.encode()).hexdigest(),
        "scratch_sha256":hashlib.sha256(updated.encode()).hexdigest(),
        "router_hook_count":updated.count("BEGLIN_P3_ROUTER_BOUNDARY"),
        "ffn_hook_count":updated.count("BEGLIN_CAPTURE_PRE_FFN_INPUT")})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch-source",type=pathlib.Path,required=True)
    a=p.parse_args();instrument(a.scratch_source)
