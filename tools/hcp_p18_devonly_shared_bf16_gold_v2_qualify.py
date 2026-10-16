#!/usr/bin/env python3
"""UNEXECUTED EOE/D50 REAL MODEL QUALIFICATION for experimental P18 v2.

Only the immutable 203-token DEVELOPMENT-ONLY source is used. Two independent
original full 27-layer/5291-tensor BF16 gold-window forward passes (3 and 5
pseudo gold tokens), compared to the proposed once-only streamed two-sequence
shared version. No frozen 480 test question is loaded. Any raw float32 mismatch
rejects shared version; default P17 original single-item evaluator remains the
only qualified reference. Never modifies production or promotes policy B.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
import pathlib
import sys
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import psutil
import torch

ROOT = pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
P17 = pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/tools")
MODEL = ROOT / "teacher_checkpoint_source_copy"
DEV = ROOT / "p18_devonly_203_hardware_input_PRIVATE.json"
DEST = ROOT / "P18_DEVONLY_V2_REAL_ORIGINAL_BF16_SHARED_GOLD_PARITY"
sys.path[:0] = [str(P17), str(ROOT)]
from hcp_p15_streamed_teacher import TensorStore, original_classes
from hcp_p18_eoe_d50_original_bf16_core import P18LongGoldWindowTeacher
from hcp_p18_shared_bf16_gold_window_v2 import forward_two_original_gold_windows

PIN = {
    "dev": "b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c",
    "index": "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
    "config": "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
    "implementation": "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c",
    "teacher": "40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3",
    "streamer": "46510526dbaf21884bca30a85ad3d1131099ccf8a1d41063064c1b1fd6683f56",
    "P18_core": "295ce59380f83162d20f619bf9d0cccc961f0c1484d16f5031fdc0040bb6bb78",
    # This SHA is of the reviewed v2 source shipped alongside this script.
    "v2_core": "c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f",
    "original_tensor_values": "88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda",
    "tokenizer": "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
}


def sha(path):
    p = pathlib.Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError("Original pinned source file missing or symlinked")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokhash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


class Guard:
    def __init__(self):
        self.signal = threading.Event()
        self.peak_rss = 0
        self.start_swap = psutil.swap_memory().used
        self.max_new_swap = 0

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        self.max_new_swap = max(self.max_new_swap, max(0, psutil.swap_memory().used - self.start_swap))
        if self.peak_rss > 9 * 1024**3:
            return "RSS exceeds 9GiB research cap"
        if psutil.virtual_memory().available < 11 * 1024**3:
            return "EOE available RAM below 11GiB"
        if self.max_new_swap > 1024**3:
            return "New swap growth exceeds 1GiB"
        if psutil.disk_usage(str(ROOT)).free < 100 * 1024**3:
            return "D50 reserved headroom below 100GiB"
        return None

    def _monitor(self):
        while not self.signal.wait(0.4):
            failure = self.check()
            if failure:
                print("P18_REAL_ORIGINAL_DEVONLY_GOLD_BATCH_RESOURCE_ABORT", failure, flush=True)
                os._exit(94)

    def __enter__(self):
        failure = self.check()
        if failure:
            raise RuntimeError(failure)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, *_):
        self.signal.set()
        self.thread.join(timeout=3)
        if kind is None and self.check():
            raise RuntimeError("EOE research resource terminal failure")


def original_once(tokens, prompt_length, classes):
    store = TensorStore(MODEL)
    teacher = P18LongGoldWindowTeacher(store, classes, dtype=torch.bfloat16)
    begun = time.monotonic()
    with torch.inference_mode():
        z = teacher.forward_gold(torch.tensor([tokens], dtype=torch.int64),
                                 prompt_length, len(tokens) - prompt_length)
    sec = time.monotonic() - begun
    if len(teacher.layer_records) != 27 or len(store.loaded_names) != 5291:
        raise ValueError("Real original 27L/5291T BF16 source incomplete")
    value_sha = hashlib.sha256((json.dumps(store.value_hashes, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    if value_sha != PIN["original_tensor_values"]:
        raise ValueError("Original P17 source tensor values differ")
    result = z.numpy().copy()
    if (result.dtype != np.float32 or result.shape != (len(tokens) - prompt_length, 102400)
            or not np.isfinite(result).all()):
        raise ValueError("Original standalone BF16 full-vocabulary reference absent")
    del store, teacher, z
    gc.collect()
    return result, sec


def main():
    if not ROOT.is_dir() or ROOT.is_symlink() or DEST.exists():
        raise RuntimeError("Invalid or already used EOE dev-only scope")
    for path, expected in (
        (DEV, PIN["dev"]),
        (MODEL / "model.safetensors.index.json", PIN["index"]),
        (MODEL / "config.json", PIN["config"]),
        (MODEL / "modeling_deepseek.py", PIN["implementation"]),
        (MODEL / "tokenizer.json", PIN["tokenizer"]),
        (P17 / "hcp_p17_teacher_gold.py", PIN["teacher"]),
        (P17 / "hcp_p15_streamed_teacher.py", PIN["streamer"]),
        (ROOT / "hcp_p18_eoe_d50_original_bf16_core.py", PIN["P18_core"]),
        (ROOT / "hcp_p18_shared_bf16_gold_window_v2.py", PIN["v2_core"]),
    ):
        if sha(path) != expected:
            raise RuntimeError("Original D50/source SHA changed: " + path.name)
    dev = json.loads(DEV.read_text())
    raw = dev["prompt_ids"]
    if (dev.get("source_role") != "development_only_not_480_test"
            or dev.get("gold_answer_not_scored") is not True
            or len(raw) != 203 or dev.get("prompt_token_count") != 203
            or tokhash(raw) != dev.get("prompt_token_sha256")
            or dev.get("tokenizer_sha256") != PIN["tokenizer"]):
        raise ValueError("Test/holdout or modified development source forbidden")
    # Deterministic synthetic gold *from dev tokens alone*, no held-out source.
    rows = [raw[:101] + raw[101:104], raw[:198] + raw[198:203]]
    prompts = [101, 198]
    if [len(x) for x in rows] != [104, 203] or [len(x)-p for x,p in zip(rows,prompts)] != [3,5]:
        raise ValueError("Frozen dev-only pseudo-gold windows drifted")
    if psutil.virtual_memory().available < 17 * 1024**3:
        raise RuntimeError("EOE RAM <17GiB")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        os.nice(10)
    except PermissionError:
        pass
    classes, model_fns = original_classes(MODEL / "modeling_deepseek.py")
    begun = time.monotonic()
    with Guard() as guard:
        references = [original_once(tokens, prompt, classes) for tokens, prompt in zip(rows, prompts)]
        teacher = P18LongGoldWindowTeacher(TensorStore(MODEL), classes, dtype=torch.bfloat16)
        shared_start = time.monotonic()
        batched = forward_two_original_gold_windows(teacher, rows, prompts, max_length=1024)
        shared_sec = time.monotonic() - shared_start
        values_sha = hashlib.sha256((json.dumps(teacher.store.value_hashes,sort_keys=True,indent=2) + "\n").encode()).hexdigest()
        if values_sha != PIN["original_tensor_values"] or len(teacher.store.loaded_names) != 5291:
            raise RuntimeError("Shared original 5291-tensor SHA or source count differs")
        # Dynamic RoPE buffers may depend on the previously processed length.
        # Independently check long-then-short order with a *fresh* source stream.
        reverse_teacher = P18LongGoldWindowTeacher(TensorStore(MODEL), classes, dtype=torch.bfloat16)
        reverse_start = time.monotonic()
        reverse = forward_two_original_gold_windows(reverse_teacher, rows[::-1], prompts[::-1], max_length=1024)
        reverse_seconds = time.monotonic() - reverse_start
        reverse_valhash = hashlib.sha256((json.dumps(reverse_teacher.store.value_hashes,sort_keys=True,indent=2) + "\n").encode()).hexdigest()
        if reverse_valhash != PIN["original_tensor_values"] or len(reverse_teacher.store.loaded_names) != 5291:
            raise RuntimeError("Reverse original BF16 source read integrity failed")
        verified = []
        for ids, prompt, ref, shared in zip(rows, prompts, references, batched):
            expected, control_seconds = ref
            actual = shared.detach().cpu().numpy()
            reverse_actual = reverse[len(rows) - 1 - len(verified)].detach().cpu().numpy()
            equal = (actual.shape == expected.shape and actual.dtype == np.float32
                     and np.array_equal(actual, expected) and np.array_equal(reverse_actual, expected))
            difference = max(float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
                             float(np.max(np.abs(reverse_actual.astype(np.float64) - expected.astype(np.float64)))))
            verified.append({
                "source_role": "development_only_not_480_test",
                "input_length": len(ids), "prompt_length": prompt,
                "gold_positions": len(ids)-prompt,
                "original_standalone_seconds": control_seconds,
                "shared_original_full_logits_SHA256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "standalone_original_full_logits_SHA256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "full_float32_bit_equal_both_orders": bool(equal),
                "maximum_absolute_logits_difference": difference,
            })
        qualified = all(item["full_float32_bit_equal_both_orders"] for item in verified)
        DEST.mkdir(mode=0o700)
        receipt = {
            "schema": "beglin-hcp-p18-Eoe-27L-5291T-two-dev-gold-batched-parity-v2",
            "status": "P18_DEVONLY_REAL_SHARED_BF16_27L_5291T_GOLD_PARITY_PASS" if qualified else "P18_DEVONLY_SHARED_BF16_GOLD_PARITY_FAIL_DO_NOT_USE",
            "original_source_role": "development_only_not_480_test",
            "pinned_source_SHA256": PIN,
            "P17_original_27_layers": 27,
            "source_TensorStore_5291_single_read": True,
            "comparison": verified,
            "baseline_two_single_forward_seconds": sum(ref[1] for ref in references),
            "shared_one_tensor_stream_seconds": shared_sec,
            "reverse_shared_tensor_stream_seconds": reverse_seconds,
            "reverse_order_real_model_checked": True,
            "overall_research_wall_seconds": time.monotonic() - begun,
            "EOE_peak_RSS_bytes": guard.peak_rss,
            "EOE_new_swap_bytes": guard.max_new_swap,
            "frozen_480_quality_certified": False,
            "native_A_B_forward_authorized": False,
            "automatic_live_promotion": False,
            "production_touched": False,
        }
        proof=DEST/"devonly_gold_parity_receipt.json"
        with proof.open("x") as f:
            json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status": receipt["status"], "gold_positions": [3,5],
            "bit_equal": qualified, "full_vocab": 102400,
            "baseline_seconds": receipt["baseline_two_single_forward_seconds"],
            "shared_seconds": shared_sec,"receipt_SHA256": sha(proof)}, sort_keys=True),flush=True)
        if not qualified:
            raise RuntimeError("Original full logits dev bit-exact parity FAILED; leave v1 reference as authority")

if __name__ == "__main__":
    main()
