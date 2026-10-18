#!/usr/bin/env python3
"""Research-only source-SHA-bound overlay to EOE original v11 two-token driver.

DO NOT upload/copy the private original DeepSeek experimental driver/model.
This overlay verifies its exact SHA and patches ONLY the on-the-fly qNg64
constructor, fixed-role prepare path, independent host memo counter gate and
scope-accurate receipt fields. It never edits that script on disk.

The old model script still loads original BF16 layers as a SEPARATE reference.
The 5,181-role sealed prequant packed paths are the actual Metal code/scale
inputs for packed GPU model linears. No production serving or TTFT assertion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ORIGINAL_SHA="defae1ecb7254e5716b87a0d5b0c99964b0d61d727c708f2942df5010c64eb46"
RESEARCH_DIR=Path(__file__).resolve().parent


def gate(cond,msg):
    if not cond:
        raise RuntimeError("BEGLIN_SEALED_ORIGINAL_V11_OVERLAY_FAIL_CLOSED:"+msg)


def replace_exact(source,old,new,label):
    n=source.count(old)
    gate(n==1, "original model source no longer matches "+label+": "+str(n))
    return source.replace(old,new,1)


def make_modified_source(original):
    raw=Path(original).read_bytes()
    gate(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,
         "original full27 v11 private source changed: refuse to patch")
    text=raw.decode("utf-8")
    text=replace_exact(
        text,
        "import research_n7_source_pack_memo_v1 as host_n7_memo",
        "import research_n7_source_pack_memo_v1 as host_n7_memo\n"
        "from prequant_full27_execution_adapter_v1 import SealedPrequantFull27",
        "import sealed C++ module adapter")
    text=replace_exact(
        text,
        "    report=ROOT/f\"real_CPP_Metal_GPUowned_v11_dual_hostmemo8_first{max_layers}_v1.json\"",
        "    report=Path(os.environ['BEGLIN_V11_PREQUANT_RECEIPT_DIR'])/"
        "f\"real_CPP_Metal_V11_SEALED_PREQUANT_first{max_layers}_v1.json\"",
        "independent immutable receipt path")
    text=replace_exact(
        text,
        "    native=packed.exp.base.Native()\n",
        "    native=packed.exp.base.Native()\n"
        "    prequant_hooks=SealedPrequantFull27("
        "os.environ['BEGLIN_V11_PREQUANT_ROOT'],"
        "os.environ['BEGLIN_V11_PREQUANT_MANIFEST_SHA'],"
        "known,cpp_class,packed)\n"
        "    packed.prepare_weights=prequant_hooks.prepare_weights\n",
        "installed full27 sealed packed producer (no fresh native quantization)")
    text=replace_exact(
        text,
        "                    W=np.ascontiguousarray(original.float().numpy(),dtype=np.float32)\n"
        "                    mod=cpp_class(None,native,W,7)",
        "                    mod=prequant_hooks.module(key,native,cpp_class,"
        "expected_shape=tuple(original.shape))",
        "dynamically chosen MoE expert model packed weights")
    text=replace_exact(
        text,
        '             "original_native_C_quantize_executions"]>0',
        '             "original_native_C_quantize_executions"]==0',
        "no dynamic original C quantization host memo gate")
    text=replace_exact(
        text,
        '"PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_GPUOWNED_AND_HOST_SHA_MEMO"',
        '"PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_PREQUANT_SEALED_5181"',
        "actual sampled 2 token result scope")

    # These fields were true of the old host-memo runtime, not of this new
    # exact sealed source path; refuse to preserve now-misleading claims.
    text=replace_exact(
        text,
        '"host_original_n7_C_quantization_elided_on_SHA_content_hit":True',
        '"host_original_n7_C_quantization_elided_on_SHA_content_hit":False',
        "scope of no in-flight quantization")
    text=replace_exact(
        text,
        '"GPU_roles_source_checkpoint_streamed_and_requantized_each_decoder_step":True',
        '"GPU_roles_source_checkpoint_streamed_and_requantized_each_decoder_step":False',
        "source weight invocation contract")
    text=replace_exact(
        text,
        '"all_other_model_linear_weights_still_repacked_per_step":True',
        '"all_other_model_linear_weights_still_repacked_per_step":False',
        "all other n7 role packed data now read from disk")

    text=replace_exact(
        text,
        '    with report.open("x",encoding="utf-8") as fd:',
        '    gate(prequant_hooks.loads>0 and prequant_hooks.quantize_calls_during_decode==0,\n'
        '         "expected actual sealed mmap native GPU model roles")\n'
        '    receipt_body["schema"]="beglin.v11.two-greedy-27layer-sealed-n7-prequant/1"\n'
        '    receipt_body["v11_sealed_model_loader"]=prequant_hooks.summary()\n'
        '    receipt_body["original_private_runner_sha256"]="'+ORIGINAL_SHA+'"\n'
        '    receipt_body["v11_native_C_quantize_during_inference"]=0\n'
        '    receipt_body["original_model_BF16_reference_reads_not_eliminated"]=True\n'
        '    receipt_body["full27_prequant_frozen_manifest_complete"]=True\n'
        '    with report.open("x",encoding="utf-8") as fd:',
        "append scoped module selection and real source authority metrics")
    gate(text != raw.decode("utf-8"),"patch produced identical source")
    return text


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original-driver",required=True)
    ap.add_argument("--sealed-root",required=True)
    ap.add_argument("--manifest-sha",required=True)
    ap.add_argument("--receipts-dir",required=True)
    ap.add_argument("--max-layers",type=int,choices=(1,2,27),required=True)
    args=ap.parse_args()
    gate(Path(args.receipts_dir).is_dir() and
         not Path(args.receipts_dir).is_symlink(),
         "receipts directory must exist and be isolated")
    gate(Path(args.original_driver).is_file() and
         not Path(args.original_driver).is_symlink(),
         "private original full27 source unavailable")
    gate(Path(args.sealed_root).is_dir() and
         not Path(args.sealed_root).is_symlink(),
         "sealed model root unavailable")
    gate(len(args.manifest_sha)==64 and all(x in "0123456789abcdef"
         for x in args.manifest_sha), "expected prequant manifest SHA invalid")
    os.environ["BEGLIN_V11_PREQUANT_ROOT"]=str(Path(args.sealed_root).resolve())
    os.environ["BEGLIN_V11_PREQUANT_MANIFEST_SHA"]=args.manifest_sha
    os.environ["BEGLIN_V11_PREQUANT_RECEIPT_DIR"]=str(Path(args.receipts_dir).resolve())
    sys.path.insert(0,str(RESEARCH_DIR))
    sys.path.insert(0,str(Path(args.original_driver).parent))
    script=make_modified_source(args.original_driver)
    scope={"__name__":"beglin_v11_original_full27_sealed_overlay",
           "__file__":str(Path(args.original_driver).resolve()),
           "__package__":None}
    exec(compile(script,str(args.original_driver), "exec"),scope)
    return scope["main"](args.max_layers)


if __name__=="__main__":
    raise SystemExit(main())
