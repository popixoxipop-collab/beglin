#!/usr/bin/env python3
"""JE01 only 16 already-fixed CALIBRATION prompt next-gold-token IDs (not test).

This is a one-token teacher-forcing PILOT only; it is not the preregistered
full-target NLL or Bayesian candidate comparison. The 16 input IDs and native
input byte hashes were frozen from the old32/new384 *unlabeled length-only*
real calibration parity experiment before gold continuations were read.
The actual WikiText public train pages MUST match every original frozen source
page SHA + x-revision. No test/OOD data or candidate outcomes are accessed.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import struct
import sys

import je01_data_freeze as data

SCHEMA="beglin-je01-original-cal16-next-gold-token-identity/1"
BASE_SHA="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
TOK_SHA="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
SELECTED_IDS=(
"01ab288093faa4f727151e78dd0074f3814e7270dc7084b51ef3e8a23cd1ce59",
"64ea8925ced5b315f63bd12b813cb2c83e347e425674e291d892eb00865c3435",
"0fc1a065e7859c01a3f4d9962b5f1084a7f4fdc53bffdf409c4750e123a47f5c",
"02c1025cf405fc1a80f0b39e5748f8f246fda9be1bdf9db7bf7c0bf179c907a8",
"136de45ba61bf8457cc3fb849dceedc1c7f57634caa4444662aa946fb9417ead",
"2d69fca898825e235ffad28af59d5ef02a0d4e4b72b2e658c6fa085fe67ad481",
"618fbf4b630383405ba4f686ac3bc4004f94e2d1eee6653ac8b6654c47d871b5",
"0b2bcfd3c8921860eeb1709eaa6dd78e95d73958aac3ee6f5b6ee117edd63a90",
"0fc87948017613d2d93780527570795263397cad8780b66a672b86e498a6741f",
"8d6bc0df0fc0dccc29ef3223c0856266565edff7603dc6fea3741913c50195e0",
"ec65a781e0629ae0b600f23ef480256b0c8852619a7494c6d23035e0efb07d65",
"2068195b24794c13a6817beae53eaf7dbe31ba3692f5ad04a69fd4d05a0439a8",
"28df487f8c28af2aec12c01e3cc2c4f58f435e0ae4105ce2b73c23c3fc8f2a84",
"693baeab5fe549d21bdebb6be4e1d29f03dfa9a54a622ca249fb45b9aeb75359",
"8367cdb60778593d06c5eab2e5b0146d81197f0bf7b040016b4519ba07c8bbc2",
"ac7a4e77a69784194ee43b16c5cc7ff4e876f4d2f60a4eebc8679ec8971fe8b6")
NATIVE_INPUT_SHAS=(
"a3a0960188d5ea59963ddd0c2601ed186202ba7879bc73bb821efaf5c5faec88",
"efb3f13c5143703a8cc36d344bc8b2722901c921cbbcde857da5dcae1dd93c1b",
"b46c0969bc13d2ef432f9bcb02a613f6913accb2154360f90e2e0a59da6204cf",
"b341580aea2f9adff14ed2ad0d1b0d8e8ce4336dcdb03836ce3ff4f40cae285b",
"0d2e90409e39056602044a4f3e911dc384865b6fb4f91d15113adba59f079537",
"51fc0bd1f2de60845f45ccee546760599b8859542b4b8fdf26807ee998f49cc5",
"04325b0aa87f73573061dd59de4e3385f0885c0e8190385bb3d11bffc5abe964",
"48743a76bdb0d91eed7fc1286e683ca7fcf6d5b1f23e98fb9860d3edeb72d0e8",
"e861f4a07497ba4b345a604fbb58dea472740360af6c52d20adcc040d6ac4ebb",
"2aedfdc4ec23a43df461240c894189802c010792d981fde0f31cc19aaee9975f",
"2f53d7afd4eed2292d47f6fce7a510cbb28f7210e5ae81e9f0830c68cd15ca6d",
"d92ae9b34a12640ce83b612fed210f0f91cbe4a920992e5a5fd56ed3ca9c4b69",
"b5da4dc8ef60627ab85906091dcd50930bc62090740e2c9a9bc2972c06977629",
"86e9edf83f68728d79ea8150d39d87fcf30b9225510ca160b38a3ee665115a75",
"9c20e27077e480c412bae36cb6b23743478723c987cfb66975625301f6e675a6",
"25537250d8a832ad46f938bf5b956c97519fc81b1d2f69ae54ec055624bd87d9")
# Native input shas above are independently checked against the already
# immutable calibration16_receipt.json; the remaining list must be exact
# before executing. No labels ever used to create SELECTED_IDS.

def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
             ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def run(frozen_path,tokenizer_path):
    from tokenizers import Tokenizer
    frozen=json.loads(Path(frozen_path).read_text())
    if frozen.get("manifest_sha256")!=BASE_SHA or data.digest({k:v for k,v in frozen.items() if k!="manifest_sha256"})!=BASE_SHA:
        raise ValueError("JE01_FROZEN_SOURCE_MANIFEST_DRIFT")
    if hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()!=TOK_SHA:
        raise ValueError("JE01_ORIGINAL_TOKENIZER_SHA_DRIFT")
    if len(SELECTED_IDS)!=16 or len(NATIVE_INPUT_SHAS)!=16 or len(set(SELECTED_IDS))!=16:
        raise ValueError("JE01_FIXED_SIXTEEN_SELECTION_DRIFT")
    source=data.SOURCES[0]
    assert source[0]=="calibration"
    checked=frozen["sources"]["calibration"]["page_sha256_by_offset"]
    allrows={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(data.request_rows,source,offset):offset for offset in data.planned_offsets(source)}
        for task in as_completed(fs):
            pg=task.result()
            off=fs[task]
            if pg["page_sha256"]!=checked[str(off)]:
                raise ValueError("JE01_ORIGINAL_SOURCE_PAGE_SHA_DRIFT")
            for item in pg["row_records"]:
                allrows[item["row_idx"]]=item["row"]
    frozenselected={r["source_row_id_sha256"]:r for r in frozen["cohorts"]["calibration"]["records"]}
    tok=Tokenizer.from_file(str(tokenizer_path))
    out=[]
    for k,(identity,expected_input_sha) in enumerate(zip(SELECTED_IDS,NATIVE_INPUT_SHAS)):
        original=frozenselected.get(identity)
        if not original:raise ValueError("JE01_SELECTED_CALIBRATION_INPUT_NOT_IN_FROZEN_MANIFEST")
        text=data.normal(allrows[original["row_idx"]]["text"])[:900]
        boundary=text.rfind(" ",0,min(len(text)-32,int(len(text)*.70)))
        if boundary<64:raise ValueError("JE01_WIKITEXT_BOUNDARY_DRIFT")
        prefix=text[:boundary]
        gold=text[boundary:].strip()
        if data.digest(prefix)!=original["prompt_sha256"] or data.digest(gold)!=original["target_sha256"]:
            raise ValueError("JE01_PUBLIC_GOLD_SOURCE_SHA_DRIFT")
        prompt_ids=tok.encode(prefix,add_special_tokens=False).ids
        raw=struct.pack("<"+"i"*(len(prompt_ids)+1),100000,*prompt_ids)
        if hashlib.sha256(raw).hexdigest()!=expected_input_sha:
            raise ValueError("JE01_ACTUAL_NATIVE_PROMPT_SHAS_DONT_MATCH:"+str(k))
        fullids=tok.encode(prefix+" "+gold,add_special_tokens=False).ids
        matched=fullids[:len(prompt_ids)]==prompt_ids
        first=fullids[len(prompt_ids)] if matched and len(fullids)>len(prompt_ids) else None
        if first is not None and not 0<=first<102400:raise ValueError("JE01_GOLD_TOKEN_INVALID")
        out.append({"request":k,"input_id":identity,
                    "native_prompt_sha256":expected_input_sha,
                    "prompt_tokens_including_bos":len(prompt_ids)+1,
                    "prompt_sha256":original["prompt_sha256"],
                    "target_sha256":original["target_sha256"],
                    "full_sequence_tokenizer_preserves_prompt_prefix":matched,
                    "first_public_gold_token_id":first,
                    "full_sequence_token_count_including_bos":len(fullids)+1})
    report={"schema":SCHEMA,
      "source_frozen_manifest_sha256":BASE_SHA,
      "original_tokenizer_sha256":TOK_SHA,
      "dataset_split":"wikitext-103-raw-v1 train CALIBRATION ONLY",
      "no_heldout_or_OOD_labels_accessed":True,
      "prior_fixed_input_count":16,
      "valid_first_gold_token_targets":sum(z["first_public_gold_token_id"] is not None for z in out),
      "prompt_prefix_boundary_mismatch_count":sum(not z["full_sequence_tokenizer_preserves_prompt_prefix"] for z in out),
      "source_target_text_not_published":True,
      "records":out,
      "original_pretrained_NLL":"NOT_SCORED_BY_THIS_SCRIPT",
      "native_BEGLIN_weight_mutation":False,
      "production_promotion":False}
    report["receipt_sha256"]=sha(report)
    return report

if __name__=="__main__":
    if len(sys.argv)!=3:raise SystemExit("usage: script frozen.json original_tokenizer.json")
    try:print(json.dumps(run(sys.argv[1],sys.argv[2]),sort_keys=True,separators=(",",":")))
    except BaseException as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:250]}))
        raise SystemExit(2)
