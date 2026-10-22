#!/usr/bin/env python3
"""Independent source-only Qv6 24/64/128 validation; no optimizer imports."""
import hashlib,json,math,os,pathlib
import numpy as np
from scipy.stats import spearmanr
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/fisher_196_train_private_receipt.json")
RAW=LAB/"fisher_24_64_128_source_private.json"
ARC=LAB/"source_arc_128_frozen.json"
CROSS=LAB/"openbookqa_val48_frozen.json"
PUB=ROOT/"docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
OUT=LAB/"independent_128_original_Fisher_source_audit.json"
PRE=ROOT/"configs/research/BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def secure(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("original private source not sealed: "+str(p))
    return p.read_bytes()
def check():
    if sha(PRE.read_bytes())!="2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b":
        raise RuntimeError("Qv6 prereg changed")
    if sha(secure(ARC))!="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2" or sha(secure(CROSS))!="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0":
        raise RuntimeError("source/held frozen data changed")
    oldbytes=secure(OLD);raw=secure(RAW)
    if sha(oldbytes)!="6d87dfef842fcce932cdb85c9a42bc3ede380493917ae608663e2072f8143417":
        raise RuntimeError("original Qv5 24 gradients changed")
    model=json.loads(raw);original=json.loads(oldbytes);source=json.loads(secure(ARC))
    pub=json.loads(PUB.read_text())
    if pub["source_private_full_receipt_SHA256"]!=sha(raw) or model["actually_executed_new_BF16_backward_items"]!=104 or model["source_gradients"]!=128:
        raise RuntimeError("private/public Fisher source authority mismatch")
    oldrows=original["source_score_rows"]
    assert len(oldrows)==196
    measures4=[];measures8=[];digest=[]
    for ix in range(24,128):
        f=LAB/"additional_original_BF16_gradients"/f"gradient_{ix:03d}.json"
        b=secure(f)
        if sha(b)!=model["new_gradient_checkpoint_SHA256s"][ix-24]:
            raise RuntimeError("original BF16 per-item append-only SHA differs")
        c=json.loads(b)
        q=source["all_source_questions"][ix]
        if c["source_case_index"]!=ix or c["prompt_token_sha256"]!=q["input_sha256"] or len(c["q4_squared_directions"])!=196:
            raise RuntimeError("per-case calibration context changed")
        digest.append(c)
    for ix,row in enumerate(oldrows):
        m4=np.asarray(list(row["per_source_item_q4_sq"])+[d["q4_squared_directions"][ix] for d in digest],dtype=np.float64)
        m8=np.asarray(list(row["per_source_item_n8_sq"])+[d["n8_squared_directions"][ix] for d in digest],dtype=np.float64)
        if m4.shape!=(128,) or m8.shape!=(128,) or not np.isfinite(m4).all() or not np.isfinite(m8).all():
            raise RuntimeError("nonfinite original BF16 196x128 directional Fisher")
        field=f"{row['role']}:{row['layer']}"
        if not np.array_equal(m4,np.asarray(model["score_rows_196"][field]["q4_squared"])) or not np.array_equal(m8,np.asarray(model["score_rows_196"][field]["n8_squared"])):
            raise RuntimeError("score_rows do not match source per-question receipts")
        measures4.append(m4);measures8.append(m8)
    diff=np.asarray(measures4)-np.asarray(measures8)
    info={}
    roles=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
    for n in (24,64,128):
        v=diff[:,:n]; mean=v.mean(axis=1)
        rho=float(spearmanr(v[:,::2].mean(axis=1),v[:,1::2].mean(axis=1)).statistic)
        for i,r in enumerate(roles):
            actual=np.asarray(pub["quality_upgrade_by_role_layer_per_size"][str(n)][r])
            if not np.allclose(actual,mean[i*28:(i+1)*28],rtol=0,atol=1e-10):
                raise RuntimeError("public 196 rank score arithmetic differs "+str(n))
        if abs(rho-pub["nested_split_stats"][str(n)]["odd_even_sample_Fisher_rank_spearman"])>1e-12:
            raise RuntimeError("original BF16 source odd/even Spearman changed")
        info[str(n)]={"source_odd_even_spearman":rho,"negative_gain_tensors":int((mean<0).sum())}
    for a,b in ((24,64),(24,128),(64,128)):
        x=diff[:,:a].mean(axis=1);y=diff[:,:b].mean(axis=1)
        rho=float(spearmanr(x,y).statistic)
        if abs(rho-pub["nested_pairwise_rank_stability"][f"{a}_{b}"]["full_196_spearman"])>1e-12:
            raise RuntimeError("source 24/64/128 rank overlap not independently confirmed")
    return {"schema":"BEGLIN-Qv6-independent-original-BF16-nested-24-64-128-audit/1",
      "status":"PASS_ALL_196_TENSOR_SCORES_128_SOURCE_INPUTS",
      "private_main_source_receipt_SHA256":sha(raw),
      "source_104_case_receipts_independently_reparsed":104,
      "old_24_receipt_sha256":sha(oldbytes),
      "scores":info,"no_cross_validation_output_used":True,
      "no_operational_mutations":True}

def run():
    if OUT.exists():raise FileExistsError("independent source audit exists")
    report=check()
    data=(json.dumps(report,sort_keys=True,indent=2)+"\n").encode()
    os.umask(0o077)
    with OUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("QV6_INDEPENDENT_ORIGINAL_FISHER_128_PASS",json.dumps({"SHA":sha(data),**report["scores"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
