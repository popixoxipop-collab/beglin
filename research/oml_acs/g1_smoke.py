"""G1 technical smoke on one cached BF16 Qwen model (NOT final evaluation).

Eight ad hoc engineering anchors, source contrast pairs, a single held-out
classification question and three greedy continuations. No public dataset
claims and no cross-family steering claims.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from hf_adapter import ModelSpec, PinnedHF
from acs_core import AnchorProjector, source_axis_direction, candidate_probs

REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" / REV

ANCHORS = [
    "Describe how rainfall works.",
    "Explain a famous historical discovery.",
    "What does photosynthesis mean?",
    "Compare the integers six and nine.",
    "Write a short poem about a sunrise.",
    "Explain why plants need sunlight.",
    "Describe a simple recipe for bread.",
    "What is the role of a library?",
]
POS = ["A surprisingly excellent and moving film.",
       "That restaurant served a wonderful dinner.",
       "The product worked perfectly and I loved it."]
NEG = ["A painfully boring and disappointing film.",
       "That restaurant served a terrible dinner.",
       "The product failed badly and I hated it."]

def main():
    start = time.perf_counter()
    if not (PATH / "model.safetensors").is_file():
        raise RuntimeError("pinned cached BF16 weights not found, no download permitted")
    spec = ModelSpec(MODEL_ID, REV, 28, 14)
    model = PinnedHF(spec, local_snapshot_path=str(PATH), device="cuda")
    print("STAGE model_loaded", round(time.perf_counter()-start, 2), flush=True)
    ah = np.stack([model.last_token(x) for x in ANCHORS])
    pos = np.stack([model.last_token(x) for x in POS])
    neg = np.stack([model.last_token(x) for x in NEG])
    proj = AnchorProjector.fit(ah)
    axis = source_axis_direction(pos, neg)
    canonical = proj.project(axis, treat_as_point=False)
    reconstructed = proj.backproject(canonical)
    print("STAGE real_activations", ah.shape, round(time.perf_counter()-start, 2), flush=True)
    q = ("Read the review and choose one of the exact words 'Positive' or 'Negative'. "
         "Review: The service was awful. Label:")
    lp_positive = model.candidate_log_likelihood(q, "Positive")
    lp_negative = model.candidate_log_likelihood(q, "Negative")
    probs = candidate_probs([lp_positive,lp_negative])
    print("STAGE common_candidate_scores", probs.tolist(), flush=True)
    generation_prompt = "Please write one brief sentence describing a typical movie."
    out_base = model.generate(generation_prompt,max_new_tokens=14)
    out_axis = model.generate(generation_prompt,direction=reconstructed,alpha=3,max_new_tokens=14)
    rng = np.random.default_rng(42)
    out_random = model.generate(generation_prompt,direction=rng.normal(size=proj.hidden_dim),
                                alpha=3,max_new_tokens=14)
    torch.cuda.synchronize()
    rec = {
        "schema":"oml-acs-g1-bf16-single-model-smoke/1",
        "verdict":"TECHNICAL_SMOKE_ONLY",
        "model":MODEL_ID,"revision":REV,"precision":"BF16",
        "layer_index_zero_based":14,"gpu":torch.cuda.get_device_name(0),
        "anchors_kind":"synthetic_engineering_not_HELM",
        "activation_matrix_shape":list(map(int,ah.shape)),
        "source_pairs_engineering_only":3,
        "canonical_coordinate_dim":int(canonical.shape[0]),
        "axis_cosine_after_anchor_reconstruction":float(axis@reconstructed),
        "candidate_question_sha256":hashlib.sha256(q.encode()).hexdigest(),
        "candidate_log_likelihoods":{"Positive":lp_positive,"Negative":lp_negative},
        "candidate_normalized_probs":probs.tolist(),
        "generation_controls":{"no_intervention":out_base,
                               "self_axis_alpha_3":out_axis,
                               "random_axis_alpha_3":out_random},
        "peak_cuda_mb":round(torch.cuda.max_memory_allocated()/2**20,2),
        "elapsed_seconds":round(time.perf_counter()-start,2),
        "scientific_endpoints":{"cross_family":"NOT_TESTED","detection_AUROC":"NOT_TESTED",
                                "answer_combination_NLL":"NOT_TESTED",
                                "causal_behavior_change":"NOT_ESTABLISHED"},
    }
    if not np.isfinite(canonical).all() or not np.isfinite(probs).all():
        raise RuntimeError("nonfinite source or candidate scores")
    print("G1_SMOKE_RECEIPT "+json.dumps(rec,ensure_ascii=False,sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
