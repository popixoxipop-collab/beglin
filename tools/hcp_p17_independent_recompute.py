#!/usr/bin/env python3
"""P17 independent evidence re-calculation from original XOX files.

Reads true original 27-layer BF16 teacher gold-logit .npy windows, each of the
62 native P9 C-origin A/B .f32 vectors, their C worker log, and 24 native greedy
request receipts. Computes 31 gold-token NLLs per policy/teacher, gold-prefix
accuracy, full-vocab KL/JS without substituting any model approximation.
Consumes existing immutable source only, writes new audited score/summary.
Does not run inference, change weights, alter live runtime or promote anything.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,pathlib,re
import numpy as np
from hcp_p17_quality_gate import gate,FROZEN_SHA

MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
P17_CAPTURE_LINE=re.compile(r"^P17_NATIVE_GOLD_PREFIX_LOGITS rid=([0-9a-f]{32}) inputsha=([0-9a-f]{64}) plen=([0-9]+) pos=([0-9]+) next=([0-9]+) vocab=(102400) bytes=(409600) floatsha=([0-9a-f]{64})$")

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def source(path):
    path=pathlib.Path(path)
    if not path.is_file() or path.is_symlink():raise ValueError("P17 source missing or symlink "+str(path))
    return json.loads(path.read_text())
def tokenhash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()

def probs(x):
    z=np.asarray(x,dtype=np.float64)
    if z.shape!=(102400,) or not np.isfinite(z).all():raise ValueError("not finite full original vocabulary logit")
    t=np.exp(z-z.max())
    den=t.sum()
    if not np.isfinite(den) or den<=0:raise ValueError("invalid native/teacher softmax")
    return t/den
def mean_gold_nll(x,target):
    p=np.asarray(x,dtype=np.float64)
    if p.shape!=(102400,) or not np.isfinite(p).all() or type(target)!=int or not 0<=target<102400:
        raise ValueError("invalid original teacher/native logits/target alignment")
    m=p.max()
    return float(m+math.log(float(np.exp(p-m).sum()))-float(p[target]))
def kl(p,q):
    mask=p>0
    if np.any(mask & (q==0)):raise ValueError("native zero probability under positive original teacher")
    return float(np.sum(p[mask]*np.log(p[mask]/q[mask])))
def js(p,q):
    m=(p+q)/2
    return .5*(kl(p,m)+kl(q,m))

def validate(xox_root):
    root=pathlib.Path(xox_root)
    if (root.is_symlink() or not root.is_dir() or
        root.name!="hcp-p17-unseen-multitoken-20261015" or
        root.parent.name!=".tailnet-tmp"):
        raise ValueError("P17 root not the exact run-owned scratch")
    frozen=root/"input/P17_FROZEN_TOKENS.json"
    if sha(frozen)!=FROZEN_SHA:raise ValueError("P17 original question token manifest changed")
    items=source(frozen)
    if items["question_count"]!=12 or items["gold_token_prefix_count"]!=31:
        raise ValueError("P17 gold question unit count drift")
    preflight=root/"preflight.json"
    if preflight.is_file():
        p=source(preflight)
        if (p.get("original_stage_SHA256")!=STAGE_SHA or
            p.get("original_route_SHA256")!=ROUTE_SHA or
            p.get("production_touched") is not False):
            raise ValueError("P17 source preflight or original serving drift")
    forced=source(root/"evidence/P17_NATIVE_FORCED_PREFIX_FULL_LOGITS.json")
    greedy=source(root/"evidence/P17_NATIVE_GREEDY_EIGHT_TOKEN_RESULTS.json")
    teacher_cohort=source(root/"evidence/P17_FULL_12_BF16_TEACHERS.json")
    if (teacher_cohort.get("status")!="TWELVE_REAL_27_LAYER_5291_TENSOR_BF16_FORWARD_COMPLETE" or
        teacher_cohort.get("full_BF16_forwards")!=12 or
        teacher_cohort.get("whole_model_gold_token_full_logit_positions")!=31 or
        teacher_cohort.get("frozen_source_sha256")!=FROZEN_SHA or
        teacher_cohort.get("production_touched") is not False):
        raise ValueError("P17 original BF16 full-model 12-question cohort incomplete")
    live={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,"stage_pids":[35481,35493]}
    for name,r,count in (("prefix",forced,62),("greedy",greedy,24)):
        if (r.get("schema")!="beglin-hcp-p17-real-native-BF16-gold-prefix-full-logits-v1" or
            r.get("status")!="P17_REAL_NATIVE_ALL_REQUESTS_COMPLETE" or
            r.get("phase")!=name or r.get("native_requests")!=count or
            r.get("source_frozen_sha256")!=FROZEN_SHA or
            r.get("native_MLX_source_SHA256")!=MLX_SHA or
            r.get("original_live_before")!=live or r.get("original_live_after")!=live or
            r.get("production_touched") is not False or
            r.get("quality_noninferiority_certified") is not False or
            len(r.get("records",[]))!=count):
            raise ValueError("P17 actual native source "+name+" not complete/registered")
    log=root/"native/capture-worker/worker.log"
    if not log.is_file() or log.is_symlink():raise ValueError("P17 original C event witness missing")
    traces={}
    for line in log.read_text(errors="replace").splitlines():
        if "P17_NATIVE_GOLD_PREFIX_LOGITS" not in line:continue
        m=P17_CAPTURE_LINE.fullmatch(line.strip())
        if not m:raise ValueError("P17 malformed actual C native logits witness")
        rid,tok,plen,pos,gen,_,_,digest=m.groups()
        if rid in traces:raise ValueError("P17 duplicate actual C request-id capture")
        traces[rid]={"token_sha256":tok,"len":int(plen),"pos":int(pos),
                     "generated":int(gen),"raw_SHA256":digest}
    if len(traces)!=62:raise ValueError("P17 wrong number of actual native C captures")
    prefix={}
    last_policy="A";last_epoch=None;worker_pid=None
    for row in forced["records"]:
        key=(row.get("question_id"),row.get("step"),row.get("policy"))
        if key in prefix:raise ValueError("P17 duplicate question/gold position/policy")
        question=next((x for x in items["items"] if x["id"]==row["question_id"]),None)
        if question is None:raise ValueError("P17 source question ID not frozen")
        step=row["step"];policy=row["policy"]
        if policy not in ("A","B") or not isinstance(step,int) or not 0<=step<question["expected_gold_token_count"]:
            raise ValueError("P17 unknown input source gold prefix")
        gold=question["forced_prefixes"][step]
        rid=row.get("raw_float_logits_filename","")
        if not isinstance(rid,str) or not rid.endswith(".f32"):
            raise ValueError("P17 no native original C float array")
        rid=rid[:-4]
        trace=traces.get(rid)
        if trace is None:raise ValueError("P17 missing specific original C capture")
        real=root/"capture"/(rid+".f32")
        if real.is_symlink() or not real.is_file() or real.stat().st_size!=409600:
            raise ValueError("P17 original C native float raw array missing/wrong")
        if (sha(real)!=row["native_full_logits_sha256"] or
            sha(real)!=trace["raw_SHA256"] or
            trace["token_sha256"]!=gold["prefix_token_sha256"] or
            trace["len"]!=gold["prefix_len"] or
            trace["pos"]!=gold["prefix_len"]-1 or
            row["source_frozen_prefix_sha256"]!=gold["prefix_token_sha256"] or
            row["input_sha256"]!=gold["prefix_token_sha256"] or
            row["gold_next_token_id"]!=gold["gold_next_token_id"] or
            trace["generated"]!=row["generated_token_ids"][0] or
            len(row["generated_token_ids"])!=1):
            raise ValueError("P17 original native C full vocab/prefix/target/argmax mismatch")
        if row["changed_policy"]!=(last_policy!=policy):
            raise ValueError("P17 original precision switch ordering inconsistent")
        if (last_epoch is not None and
            row["precision_epoch"]!=last_epoch+int(row["changed_policy"])):
            raise ValueError("P17 native precision epoch not monotonic")
        if row["changed_policy"] and row["terminal_ack_status"]!="REBIND_SET_APPLIED":
            raise ValueError("P17 native genuine terminal REBIND_SET ACK missing")
        last_epoch=row["precision_epoch"];last_policy=policy
        if worker_pid is None:worker_pid=row["worker_pid"]
        if row["worker_pid"]!=worker_pid:
            raise ValueError("P17 native worker changed PID mid-evaluation")
        z=np.fromfile(real,dtype="<f4")
        if z.shape!=(102400,) or not np.isfinite(z).all() or int(np.argmax(z))!=row["generated_token_ids"][0]:
            raise ValueError("P17 actual raw greedy logit provenance invalid")
        prefix[key]={"vector":z,"floatsha":sha(real),"row":row}
    greedy_map={}
    for row in greedy["records"]:
        key=(row.get("question_id"),row.get("policy"))
        if key in greedy_map:raise ValueError("P17 duplicate native generated sample")
        question=next((x for x in items["items"] if x["id"]==row["question_id"]),None)
        if question is None or row["policy"] not in ("A","B"):
            raise ValueError("P17 native answer prompt missing source question")
        generated=row["generated_token_ids"]
        if (row["input_sha256"]!=question["prompt_token_sha256"] or
            row["source_frozen_prefix_sha256"]!=question["prompt_token_sha256"] or
            not isinstance(generated,list) or not 1<=len(generated)<=8 or
            any(type(x)!=int or x<0 or x>=102400 for x in generated)):
            raise ValueError("P17 greedy native exact source generation invalid")
        gold=question["gold_answer_ids"]
        if row["strict_gold_token_prefix_match"]!=(len(generated)>=len(gold) and generated[:len(gold)]==gold):
            raise ValueError("P17 native generated strict gold prefix claim false")
        greedy_map[key]=row
    if len(prefix)!=62 or len(greedy_map)!=24:raise ValueError("P17 real source native coverage incomplete")
    data={"schema":"beglin-hcp-p17-independent-real-teacher-native-question-scores-v1",
          "frozen_token_sha256":FROZEN_SHA,
          "verified_actual_BF16_gold_logit_positions":31,
          "verified_actual_native_gold_logit_positions":62,
          "verified_native_real_greedy_generations":24,
          "independent_actual_logits_source_audit_passed":True,
          "production_touched":False,"records":[]}
    source_weight_shas=set()
    kljs=[]
    for item in items["items"]:
        qid=item["id"];n=item["expected_gold_token_count"]
        teachdoc=source(root/"private_teacher"/(qid+".json"))
        weights=root/"private_teacher"/(qid+".source_hashes.json")
        goldfile=root/"private_teacher"/(qid+".bf16_logits.npy")
        if (not goldfile.is_file() or goldfile.is_symlink() or not weights.is_file() or weights.is_symlink() or
            teachdoc["status"]!="FULL_BF16_DEEPSEEK_27L_5291T_GOLD_LOGIT_WINDOW_COMPLETE" or
            teachdoc["decoder_layers_forwarded"]!=27 or
            teachdoc["original_source_tensors_exactly_once"]!=5291 or
            teachdoc["input_token_sha256"]!=item["prompt_token_sha256"] or
            teachdoc["gold_answer_token_sha256"]!=item["gold_answer_sha256"] or
            teachdoc["gold_answer_token_count"]!=n or
            teachdoc["reference_arithmetic"]!="BF16" or
            teachdoc["whole_model_teacher_executed"] is not True or
            teachdoc["production_touched"] is not False or
            teachdoc["source_hashes_manifest_sha256"]!=sha(weights) or
            teachdoc["logits_NPY_sha256"]!=sha(goldfile)):
            raise ValueError("P17 original full BF16 teacher logit window missing/untrusted "+qid)
        if len(json.loads(weights.read_text()))!=5291:
            raise ValueError("P17 original BF16 checkpoint tensor hash list not 5291")
        source_weight_shas.add(sha(weights))
        t=np.load(goldfile,allow_pickle=False)
        if t.shape!=(n,102400) or t.dtype!=np.float32 or not np.isfinite(t).all():
            raise ValueError("P17 original whole model BF16 teacher full logits malformed")
        teacher_nll=[]
        cells={}
        for policy in ("A","B"):
            losses=[];digests=[]
            for step,target in enumerate(item["gold_answer_ids"]):
                actual=prefix[(qid,step,policy)]
                z=actual["vector"]
                teacher=t[step]
                tnll=mean_gold_nll(teacher,int(target))
                nnll=mean_gold_nll(z,int(target))
                if policy=="A":
                    if not math.isclose(tnll,float(teachdoc["gold_nll_nats"][step]),rel_tol=1e-8,abs_tol=1e-7):
                        raise ValueError("P17 independently recomputed original teacher gold NLL mismatch")
                    teacher_nll.append(tnll)
                losses.append(nnll);digests.append(actual["floatsha"])
                pt=probs(teacher);pn=probs(z)
                kljs.append({"item_id":qid,"gold_step":step,"policy":policy,
                    "KL_teacher_to_native_nats":kl(pt,pn),
                    "JS_divergence_nats":js(pt,pn),
                    "teacher_selected_argmax":int(np.argmax(teacher)),
                    "native_selected_argmax":int(np.argmax(z)),
                    "teacher_reference_gold_nll":tnll,"native_gold_nll":nnll})
            chain=json.dumps(digests,separators=(",",":"),sort_keys=False).encode()
            cells[policy]={"gold_token_nll_nats":losses,
                "native_greedy_token_ids":greedy_map[(qid,policy)]["generated_token_ids"],
                "real_source_logits_manifest_sha256":hashlib.sha256(chain).hexdigest()}
        data["records"].append({"item_id":qid,"domain":item["domain"],
            "gold_answer_sha256":item["gold_answer_sha256"],"gold_token_count":n,
            "original_BF16_teacher_gold_nll_nats":teacher_nll,
            **cells})
    if len(source_weight_shas)!=1:raise ValueError("P17 actual whole-model BF16 source weights drifted")
    result=gate(root/"input/P17_FROZEN_TOKENS.json",data)
    result["independent_real_full_vocab_KL_JS_per_prefix"]=kljs
    result["source_teacher_cohort_receipt_SHA256"]=sha(root/"evidence/P17_FULL_12_BF16_TEACHERS.json")
    result["source_native_prefix_receipt_SHA256"]=sha(root/"evidence/P17_NATIVE_FORCED_PREFIX_FULL_LOGITS.json")
    result["source_native_greedy_receipt_SHA256"]=sha(root/"evidence/P17_NATIVE_GREEDY_EIGHT_TOKEN_RESULTS.json")
    result["source_common_BF16_checkpoint_tensor_hash_manifest_SHA256"]=next(iter(source_weight_shas))
    result["source_capture_worker_log_SHA256"]=sha(log)
    result["independent_teacher_actual_logits_columns"]=31*102400
    result["independent_native_actual_logits_columns"]=62*102400
    result["native_exact_original_C_capture_events"]=62
    result["raw_logs_and_checkpoint_original_files_not_renamed_or_mutated"]=True
    return data,result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--verified-score-records",required=True)
    ap.add_argument("--output",required=True)
    x=ap.parse_args()
    inputdata,report=validate(x.root)
    for path,document in ((x.verified_score_records,inputdata),(x.output,report)):
        with pathlib.Path(path).open("x") as f:json.dump(document,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print("P17_INDEPENDENT_REAL_SOURCE_A_B_QUALITY_RECALCULATED",json.dumps({
       "status":report["status"],"full_teacher_gold_positions":31,
       "genuine_native_full_logit_prefixes":62,
       "real_native_greedy_generations":24,
       "bootstrap_NLL":report["B_minus_A_nll_question_bootstrap"],
       "bootstrap_prefix":report["B_minus_A_prefix_accuracy_question_bootstrap"],
       "production_touched":False,
       "quality_noninferiority_certified":False
       },sort_keys=True))
