#!/usr/bin/env python3
"""P17 resume source-pinned full BF16 teacher items 01..11 after sealed pilot 00.

NEVER overwrites item00 or any existing evidence. One original 27-layer teacher
per child process, <=6GiB RSS with in-child memory watchdog. Stops on failure.
"""
import hashlib,json,os,pathlib,psutil,shutil,subprocess,sys,time
ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
FROZEN="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
TEACHER="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
STAGE="/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu"
ROUTE="/Users/xox/vdsp_serving/active-route.json"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def live():
    ps=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=8)
    return (sha(STAGE),sha(ROUTE),tuple(sorted(int(s.split()[0]) for s in ps.splitlines() if STAGE in s)))
def checked(root,index,expected):
    item=root["items"][index]
    dir=ROOT/"private_teacher"
    p=dir/(item["id"]+".json")
    raw=dir/(item["id"]+".bf16_logits.npy")
    source=dir/(item["id"]+".source_hashes.json")
    if any(not x.is_file() or x.is_symlink() for x in (p,raw,source)):
        raise RuntimeError("P17 missing actual teacher evidence "+item["id"])
    z=json.loads(p.read_text())
    if (z["status"]!="FULL_BF16_DEEPSEEK_27L_5291T_GOLD_LOGIT_WINDOW_COMPLETE" or
        z["decoder_layers_forwarded"]!=27 or
        z["original_source_tensors_exactly_once"]!=5291 or
        z["original_full_vocab"]!=102400 or
        z["gold_answer_token_count"]!=item["expected_gold_token_count"] or
        z["input_token_sha256"]!=item["prompt_token_sha256"] or
        z["logits_NPY_sha256"]!=sha(raw) or
        z["source_hashes_manifest_sha256"]!=sha(source) or
        z["production_before"]!=z["production_after"] or
        z["whole_model_teacher_executed"] is not True):
        raise RuntimeError("P17 fraudulent/partial teacher receipt "+item["id"])
    return {"id":item["id"],"NLL":z["mean_gold_nll_nats"],"gold":item["expected_gold_token_count"],
            "teacher_receipt_SHA256":sha(p),"original_logits_SHA256":sha(raw),
            "original_weight_SHA256":z["source_hashes_manifest_sha256"]}
def main():
    if ROOT.is_symlink() or ROOT.parent.name!=".tailnet-tmp" or not ROOT.is_dir():
        raise RuntimeError("P17 unrelated filesystem root")
    frozen=ROOT/"input/P17_FROZEN_TOKENS.json";teacher=ROOT/"src/tools/hcp_p17_teacher_gold.py"
    if sha(frozen)!=FROZEN or sha(teacher)!=TEACHER:
        raise RuntimeError("P17 teacher source or inputs do not match preregistration")
    items=json.loads(frozen.read_text())
    if items["question_count"]!=12 or items["gold_token_prefix_count"]!=31:
        raise RuntimeError("P17 full cohort not 12 questions /31 gold tokens")
    original=live()
    if original!=(STAGE_SHA,ROUTE_SHA,(35481,35493)):
        raise RuntimeError("P17 immutable original stage/route/PIDs changed")
    records=[checked(items,0,None)]
    for index in range(1,12):
        item=items["items"][index]
        log=ROOT/"private_teacher"/(item["id"]+".run.log")
        if any(x.exists() for x in (ROOT/"private_teacher"/(item["id"]+".json"),
            ROOT/"private_teacher"/(item["id"]+".bf16_logits.npy"),log)):
            raise RuntimeError("P17 prior output exists: refuse overwrite at "+item["id"])
        if shutil.disk_usage(ROOT).free<20_100_000_000 or psutil.virtual_memory().available<8*1024**3:
            raise RuntimeError("P17 protected CPU memory or disk gate blocked further inference")
        now=time.monotonic()
        with log.open("x") as f:
            proc=subprocess.run(["/Users/xox/.venv-vllm-metal/bin/python3","-u",str(teacher),"--index",str(index)],
                 cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,timeout=360,
                 env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
                    "MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1",
                    "TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
        if proc.returncode!=0:
            print("P17_ORIGINAL_BF16_TEACHER_CHILD_FAILED",json.dumps({"item":item["id"],"code":proc.returncode,
                  "log_SHA256":sha(log),"completed_before_failure":len(records)}),flush=True)
            raise RuntimeError("P17 actual teacher one-item child did not complete")
        record=checked(items,index,None)
        records.append(record)
        if live()!=original:raise RuntimeError("P17 immutable original stage changed during teacher")
        print("P17_REAL_BF16_27L_TEACHER",json.dumps({"completed":len(records),"of":12,"item":record["id"],
              "mean_gold_NLL":round(record["NLL"],5),"gold_tokens":record["gold"],
              "runtime_seconds":round(time.monotonic()-now,1)}),flush=True)
    if len(records)!=12 or sum(x["gold"] for x in records)!=31:
        raise RuntimeError("P17 incomplete gold cohort")
    if len({x["original_weight_SHA256"] for x in records})!=1:
        raise RuntimeError("Original BF16 source checkpoint weights drifted between 12 forwards")
    receipt=ROOT/"evidence/P17_FULL_12_BF16_TEACHERS.json"
    with receipt.open("x") as f:json.dump({"schema":"beglin-hcp-p17-12-original-whole-BF16-gold-teachers-v1",
        "status":"TWELVE_REAL_27_LAYER_5291_TENSOR_BF16_FORWARD_COMPLETE",
        "frozen_source_sha256":FROZEN,"items":records,
        "whole_model_gold_token_full_logit_positions":31,"full_BF16_forwards":12,
        "same_5291_original_source_tensor_hashes":True,
        "external_independent_dataset":False,
        "quality_noninferiority_certified":False,"production_touched":False,
        "automatic_promotion":False},f,indent=2,sort_keys=True);f.write("\n")
    print("P17_REAL_FULL_TEACHER_COHORT_COMPLETE",json.dumps({"teacher_runs":12,"gold_tokens":31,"receipt_SHA256":sha(receipt)}),flush=True)
if __name__=="__main__":main()
