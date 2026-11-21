#!/usr/bin/env python3
"""JE01 research-only: validate original 512 prompt/256 train-gold artifacts;
prepare exactly 16 SHA-attested teacher-forcing source inputs for native AF.
Never read independent heldout/OOD gold. No production model changes.
"""
import argparse,hashlib,json,os,struct
from pathlib import Path

INPUT_SHA="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
GOLD_SHA="355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7"
BASE_SHA="f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41"
NLL=2.46035653719724
PREFIX=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                        allow_nan=False).encode()).hexdigest()
def load(p,expected):
    if p.is_symlink() or not p.is_file():raise ValueError("JE01_ORIGINAL_SOURCE_MISSING")
    d=json.loads(p.read_text())
    signed=d.pop("receipt_sha256",None)
    if signed!=expected or canonical(d)!=expected:
        raise ValueError("JE01_ORIGINAL_SOURCE_RECEIPT_SHA_MISMATCH:"+p.name)
    return d
def root(src):
    choices=[src,src/"je01_inputs"]
    matched=[p for p in choices if p.is_dir() and not p.is_symlink() and
             (p/"JE01_NATIVE_384_INPUTS_RECEIPT.json").is_file()]
    if len(matched)!=1:raise ValueError("JE01_UNIQUE_NATIVE_INPUT_ROOT_REQUIRED")
    return matched[0]
def run(inputs,gold_file,baseline_file,target):
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise ValueError("JE01_NEW_TARGET_MUST_NOT_EXIST")
    input_root=root(inputs)
    original=load(input_root/"JE01_NATIVE_384_INPUTS_RECEIPT.json",INPUT_SHA)
    gold=load(gold_file,GOLD_SHA)
    base=load(baseline_file,BASE_SHA)
    if original["total_input_records"]!=512 or original["bos_id"]!=100000 or \
       original["native_context_positions"]!=384 or \
       original["source_manifest_sha256"]!="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147":
        raise ValueError("JE01_FROZEN_REAL512_PROMPT_SOURCE_DRIFT")
    if gold["status"]!="TRAIN256_FULL_GOLD_SOURCE_READY" or \
       gold["original_256_source_rows"]!=256 or \
       gold["aligned_to_original_prompt_prefix_count"]!=256 or \
       gold["same_source_validation_and_ood_labels_used"] is not False:
        raise ValueError("JE01_REAL_TRAIN256_GOLD_SOURCE_DRIFT")
    if base["status"]!="REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS" or \
       abs(base["mean_gold_token_NLL"]-NLL)>1e-13 or \
       base["gold_full_target_tokens_scored"]!=11024 or \
       base["source_gold_sha256"]!=GOLD_SHA:
        raise ValueError("JE01_ORIGINAL_AF_NLL_BASELINE_CHANGED")
    prompts=original["cohorts"]["calibration"]["file_receipts"]
    cases=gold["cases"]
    reference=base["per_case_real_NLL"]
    if len(prompts)!=len(cases)!=len(reference)!=256:
        raise ValueError("JE01_ORIGINAL_256_COUNT_DRIFT")
    rows=[]
    for i,(p,c,r) in enumerate(zip(prompts,cases,reference)):
        f=input_root/"calibration"/(p["input_id"]+".bin")
        if f.parent!=input_root/"calibration" or f.is_symlink() or not f.is_file():
            raise ValueError("JE01_SOURCE_PROMPT_BIN_MISSING")
        raw=f.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=p["native_int32_sha256"] or \
           len(raw)!=p["native_int32_bytes"] or len(raw)%4 or \
           p["native_int32_sha256"]!=c["native_BOS_prompt_sha256"] or \
           p["input_id"]!=c["input_id"] or r["source_id"]!=p["input_id"] or \
           c["row"]!=i or r["row"]!=i:
            raise ValueError("JE01_TRAIN_PROMPT_SOURCE_IDENTITY_FAIL:"+str(i))
        toks=list(struct.unpack("<"+"i"*(len(raw)//4),raw))
        gold_ids=c["gold_continuation_token_ids"]
        if toks[0]!=100000 or len(toks)!=c["original_BOS_prompt_positions"] or \
           len(gold_ids)!=c["gold_token_count"] or \
           len(toks)+len(gold_ids)!=c["model_full_BOS_sequence_positions"] or \
           len(toks)+len(gold_ids)>384 or r["tokens"]!=len(gold_ids) or \
           not all(type(x) is int and 0<=x<102400 for x in gold_ids):
            raise ValueError("JE01_GOLD_NATIVE_384_ALIGNMENT_FAIL:"+str(i))
        rows.append((p["input_id"],toks,gold_ids,r["first_gold_nll"]))
    if sum(len(x[2]) for x in rows)!=11024:raise ValueError("JE01_11024_TARGET_COUNT_MISMATCH")
    target.mkdir(mode=0o700)
    inp=target/"inputs";inp.mkdir(mode=0o700)
    chunk=target/"chunk00";chunk.mkdir(mode=0o700)
    metadata=bytearray(PREFIX)
    manifest=[]
    evidence=[]
    for i,(id,toks,ids,first) in enumerate(rows[:16]):
        data=struct.pack("<"+"i"*(len(toks)+len(ids)-1),*(toks+ids[:-1]))
        path=inp/(id+".bin")
        with path.open("xb") as fd:
            fd.write(data);fd.flush();os.fsync(fd.fileno())
        if any(ch.isspace() for ch in str(path)) or len(str(path))>900:
            raise ValueError("JE01_UNSAFE_PORTABLE_PATH")
        manifest.append(str(path)+" 1\n")
        metadata.extend(struct.pack("<II",len(toks),len(ids)))
        metadata.extend(struct.pack("<384i",*(ids+[-1]*(384-len(ids)))))
        evidence.append({"row":i,"input_id":id,"raw_teacherforce_sha256":hashlib.sha256(data).hexdigest(),
                         "gold_tokens":len(ids),"first_gold_NLL_original":first})
    m=chunk/"manifest.txt";m.write_text("".join(manifest))
    g=chunk/"gold_meta.bin";g.write_bytes(metadata)
    receipt={"schema":"beglin-je01-eoe-train16-native-tf-source/1",
             "status":"TRAIN256_ALL_SOURCE_SHA_PASS_AND_PILOT16_NATIVE_INPUT_CREATED",
             "original_fulltarget_train256_baseline_NLL":NLL,
             "source_512_sha256":INPUT_SHA,"source_gold_sha256":GOLD_SHA,
             "source_original_AF_NLL_sha256":BASE_SHA,
             "pilot_cases":16,"pilot_gold_tokens":sum(x["gold_tokens"] for x in evidence),
             "pilot_evidence":evidence,"manifest_sha256":hashlib.sha256(m.read_bytes()).hexdigest(),
             "meta_sha256":hashlib.sha256(metadata).hexdigest(),
             "independent_128_128_labels_used":False,
             "real_model_forward":"NOT_RUN","candidate_gold_NLL":"NOT_MEASURED",
             "production_modified":False}
    receipt["receipt_sha256"]=canonical(receipt)
    (target/"JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return receipt
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--native-input-root",type=Path,required=True)
    p.add_argument("--gold-json",type=Path,required=True)
    p.add_argument("--baseline-json",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        d=run(a.native_input_root,a.gold_json,a.baseline_json,a.output)
        print(json.dumps({"status":d["status"],"pilot16_gold_tokens":d["pilot_gold_tokens"],
                          "receipt_sha256":d["receipt_sha256"]},sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}));raise SystemExit(2)
