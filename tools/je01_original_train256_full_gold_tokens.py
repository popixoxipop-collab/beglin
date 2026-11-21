#!/usr/bin/env python3
"""JE01 original DeepSeek WikiText TRAIN ONLY: 256 full gold target token IDs.

Use only after freezing real original 256 route-based 8 Expert pilot slots.
Never query same-source validation or HellaSwag OOD labels, and never re-select
samples based on NLL. Emits SHA-bound token IDs, NOT raw public text. This does
not run a model and cannot establish any native NLL/precision improvement.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,pathlib,struct
import je01_data_freeze as data

SCHEMA="beglin-je01-original-deepseek-wikitext256-train-full-target-tokens/1"
FROZEN_SHA="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
TOKENIZER_SHA="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
CONFIG_SHA="f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
TRAIN_IDS_SHA="3d6cc45aa2e64da2f137bce24bc3a2f6df9869e0e2a351bf4979f601304719b5"
TOTAL_TRAIN=256
BOS=100000
CONTEXT=384

def canonical(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()

def reconstruct_train(frozen_path,tokenizer_path,config_path):
    from tokenizers import Tokenizer
    path=pathlib.Path(frozen_path);tokenfile=pathlib.Path(tokenizer_path)
    configfile=pathlib.Path(config_path)
    if hashlib.sha256(tokenfile.read_bytes()).hexdigest()!=TOKENIZER_SHA:
        raise ValueError("JE01_ORIGINAL_TOKENIZER_SHA_DRIFT")
    if hashlib.sha256(configfile.read_bytes()).hexdigest()!=CONFIG_SHA:
        raise ValueError("JE01_ORIGINAL_MODEL_CONFIG_SHA_DRIFT")
    config=json.loads(configfile.read_text())
    if config["model_type"]!="deepseek_v2" or config["bos_token_id"]!=BOS or \
       config["vocab_size"]!=102400 or config["num_hidden_layers"]!=27:
        raise ValueError("JE01_MODEL_CONFIG_ARCHITECTURE_DRIFT")
    frozen=json.loads(path.read_text())
    sig=frozen.get("manifest_sha256")
    if sig!=FROZEN_SHA or canonical({k:v for k,v in frozen.items() if k!="manifest_sha256"})!=sig:
        raise ValueError("JE01_FROZEN_SOURCE_MANIFEST_SHA_DRIFT")
    if frozen["gold_labels_used_in_sampling"] or \
       frozen["cohorts"]["calibration"]["count"]!=TOTAL_TRAIN:
        raise ValueError("JE01_REAL_TRAIN_SELECTION_OR_COUNT_DRIFT")
    # Only calibration TRAIN page retrieval, never the two heldout sources.
    source=data.SOURCES[0]
    if source[0]!="calibration" or source[2]!="wikitext-103-raw-v1" or \
       source[3]!="train":
        raise ValueError("JE01_DATA_SOURCE_NOT_TRAIN")
    source_pages=frozen["sources"]["calibration"]["page_sha256_by_offset"]
    offsets=data.planned_offsets(source)
    rows={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(data.request_rows,source,off):off for off in offsets}
        for future in as_completed(fs):
            page=future.result();offset=fs[future]
            if page["page_sha256"]!=source_pages[str(offset)]:
                raise ValueError("JE01_TRAIN_WINDOW_FULL_SHA_CHANGED:"+str(offset))
            for record in page["row_records"]:
                index=record["row_idx"]
                if index in rows and rows[index]!=record["row"]:
                    raise ValueError("JE01_SAME_TRAIN_INDEX_CONFLICT")
                rows[index]=record["row"]
    tok=Tokenizer.from_file(str(tokenfile))
    all_prompt_ids=[];samples=[]
    for n,item in enumerate(frozen["cohorts"]["calibration"]["records"]):
        raw=data.normal(rows[item["row_idx"]]["text"])[:900]
        limit=min(len(raw)-32,int(len(raw)*.70))
        boundary=raw.rfind(" ",0,limit)
        if boundary<64:raise ValueError("JE01_TRAIN_PREFIX_SPLIT_DRIFT")
        prompt=raw[:boundary]
        target=raw[boundary:].strip()
        if data.digest(prompt)!=item["prompt_sha256"] or \
           data.digest(target)!=item["target_sha256"]:
            raise ValueError("JE01_ORIGINAL_TRAIN_PROMPT_OR_GOLD_SHA_DRIFT")
        ids=tok.encode(prompt,add_special_tokens=False).ids
        all_prompt_ids.append(ids)
        full=tok.encode(prompt+" "+target,add_special_tokens=False).ids
        if not ids or len(full)+1>CONTEXT or len(full)<=len(ids):
            raise ValueError("JE01_ORIGINAL_TRAIN_FULL_CONTEXT_DRIFT")
        aligned=full[:len(ids)]==ids
        if aligned:
            target_ids=full[len(ids):]
            if not target_ids or any(v<0 or v>=102400 for v in target_ids):
                raise ValueError("JE01_INVALID_GOLD_TOKEN_ID")
        else:target_ids=None
        original_bytes=struct.pack("<"+"i"*(len(ids)+1),BOS,*ids)
        sample={"row":n,
             "input_id":item["source_row_id_sha256"],
             "source_row_idx":item["row_idx"],
             "native_BOS_prompt_sha256":hashlib.sha256(original_bytes).hexdigest(),
             "prompt_sha256":item["prompt_sha256"],
             "target_text_sha256":item["target_sha256"],
             "original_BOS_prompt_positions":len(ids)+1,
             "model_full_BOS_sequence_positions":len(full)+1,
             "tokenizer_prefix_exact":aligned,
             "gold_continuation_token_ids":target_ids,
             "gold_token_count":len(target_ids) if aligned else None}
        samples.append(sample)
    if data.digest(all_prompt_ids)!=TRAIN_IDS_SHA:
        raise ValueError("JE01_REAL_TRAIN_256_TOKENIZER_IDS_CHANGED")
    if len(samples)!=TOTAL_TRAIN or len(set(s["input_id"] for s in samples))!=TOTAL_TRAIN:
        raise ValueError("JE01_TRAIN_256_IDS_DUPLICATE_OR_MISSING")
    usable=sum(s["tokenizer_prefix_exact"] for s in samples)
    result={
      "schema":SCHEMA,
      "status":"TRAIN256_FULL_GOLD_SOURCE_READY" if usable==256 else "TRAIN256_ALIGNMENT_PARTIAL_HOLD",
      "source_frozen_manifest_sha256":FROZEN_SHA,
      "original_model_config_sha256":CONFIG_SHA,
      "original_model_tokenizer_sha256":TOKENIZER_SHA,
      "original_train256_token_ids_sha256":TRAIN_IDS_SHA,
      "original_256_source_rows":256,
      "aligned_to_original_prompt_prefix_count":usable,
      "unscoreable_original_prefix_boundary_count":256-usable,
      "retrieved_dataset_splits":["wikitext-103-raw-v1 train"],
      "same_source_validation_and_ood_labels_used":False,
      "all_rows_preserved_no_resampling":True,
      "original_pretrained_384_full_sequence_inference":"NOT_RUN",
      "real_gold_full_target_NLL":"NOT_MEASURED",
      "Bayesian_Expert_tier_quality":"NOT_MEASURED",
      "production_weights_or_route_modified":False,
      "cases":samples}
    result["receipt_sha256"]=canonical(result)
    return result

if __name__=="__main__":
    import argparse,sys
    a=argparse.ArgumentParser()
    a.add_argument("--original-frozen-manifest",required=True)
    a.add_argument("--original-tokenizer-json",required=True)
    a.add_argument("--original-config-json",required=True)
    args=a.parse_args()
    try:
        proof=reconstruct_train(args.original_frozen_manifest,args.original_tokenizer_json,args.original_config_json)
        print(json.dumps(proof,sort_keys=True,separators=(",",":"),allow_nan=False))
    except BaseException as error:
        print(json.dumps({"status":"BLOCKED","error_type":type(error).__name__,
                          "reason":str(error)[:250]},sort_keys=True))
        sys.exit(2)
