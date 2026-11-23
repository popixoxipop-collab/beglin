#!/usr/bin/env python3
"""JE01 independent native original AF4 vs true BF16 Chat vs NIG 3/5 bit attribution.

Reuses previous SHA-attested exact 24-role AF4-vs-BF16 report. Rehashes full
original native AF4 and all 4 original Chat BF16 shards again, then audits the
actual sealed 43,253,760-byte NIG native bitplanes loaded in the 27-layer real
NLL experiment. Exactly 8 Expert × 3 roles and a wrong-Expert negative control.
No forward, no dataset/OOD labels, no operating model changes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path("/Volumes/D50/TailnetCommander/work/je01_affine4_bf16_source_parity_20261122_v1")
NIG_ROOT=Path("/Volumes/D50/TailnetCommander/work/je01_eoe_real_nig_gold16_20261122_v1")
SOURCE_DIGEST="be513c64e6af0a91613abdc34f1051a666023870561c143e4ad231e93a6e9233"
STAGE_DIGEST="736ca28a433a30bfba31e3d09720d07a91ce5e72b4aa9edcc2f0a8a9c60bc5ea"
NIG_DIGEST="dab4f50b6233688d789facf37a7fc94e22b0f4aa33298c36e5f69d89063db8cd"
GOLD_NIG_DIGEST="880f0cae56288adaa8e29645779c03e94cd002e178a06f9e31e284ee7b405ec5"
GOLD_OFF_DIGEST="958a04f40ca3229df55e74af1175b9c4bb20de96b4c83703641890c3e91208c5"
SCHEMA="beglin-je01-original-af4-chat-bf16-nig-q3q5-realweight-source-quantization-attribution/1"

sys.path.insert(0,str(ROOT))
import je01_original_af4_vs_chat_bf16_selected24_weight_audit_v2 as base

def signed(p,digest):
    if not p.is_file() or p.is_symlink():
        raise ValueError("JE01_PREVIOUS_SEALED_EVIDENCE_MISSING:"+str(p))
    r=json.loads(p.read_text())
    sha=r.pop("receipt_sha256",None)
    if sha!=digest or base.canonical(r)!=digest:
        raise ValueError("JE01_PREVIOUS_EVIDENCE_DIGEST_MISMATCH:"+p.name)
    return r

def real_nig_source():
    d=signed(NIG_ROOT/"JE01_REAL_NIG_EOE_SOURCE_STAGE.json",STAGE_DIGEST)
    if d["status"]!="JE01_EOE_REAL_NIG8_SOURCE_AND_GOLD_NATIVE_BINARY_SHA_STAGED_PASS" or \
       d["source_NIG_SHA256"]!=NIG_DIGEST or d["full_43MB_byte_size"]!=43253760 or \
       d["real_role_count"]!=24 or d["real_low3_roles"]!=6 or d["production_modified"]:
        raise ValueError("JE01_ORIGINAL_NIG_STAGE_RECORD_MISMATCH")
    p=Path(d["bundle_path"])
    t=Path(d["tsv_path"])
    if p.resolve().parent!=NIG_ROOT/"weights_single" or \
       t.resolve().parent!=NIG_ROOT/"tsv" or \
       p.is_symlink() or t.is_symlink() or not p.is_file() or not t.is_file() or \
       p.stat().st_size!=43253760 or base.sha_file(p)!=NIG_DIGEST:
        raise ValueError("JE01_ORIGINAL_REAL_NIG_PAYLOAD_SHA_OR_SAFE_PATH_DRIFT")
    blob=p.read_bytes()
    rows={}
    spans=[]
    for line in t.read_text().splitlines():
        v=line.split("\t")
        if len(v)!=10:raise ValueError("JE01_NIG_TSV_10_FIELD_CONTRACT")
        l,e,role,bits,rowsize,colsize,po,pe,so,se=v
        l,e,bits,rowsize,colsize,po,pe,so,se=map(int,(l,e,bits,rowsize,colsize,po,pe,so,se))
        key=(l,e,role)
        if key in rows or (l,e) not in base.FROZEN_SLOTS or role not in base.ROLES or \
           (rowsize,colsize)!=base.SHAPES[role] or bits not in (3,5) or \
           pe-po!=rowsize*colsize*bits//8 or se-so!=rowsize*(colsize//64)*4 or \
           not 0<=po<pe<=43253760 or not 0<=so<se<=43253760:
            raise ValueError("JE01_NIG_NATIVE_REAL_ROLE_TSV_INVALID:"+str(key))
        rows[key]=(bits,rowsize,colsize,po,pe,so,se)
        spans.extend(((po,pe),(so,se)))
    expected={(l,e,r) for l,e in base.FROZEN_SLOTS for r in base.ROLES}
    if set(rows)!=expected:raise ValueError("JE01_NIG_SELECTED_EXPERT_24_ROLES_CHANGED")
    if sorted(spans)[0][0]!=0 or sorted(spans)[-1][1]!=43253760 or \
       any(x[1]!=y[0] for x,y in zip(sorted(spans)[:-1],sorted(spans)[1:])):
        raise ValueError("JE01_NATIVE_REAL_NIG_ALL_48_RANGE_CONTIGUITY_INVALID")
    if sum(z[0]==3 for z in rows.values())!=6:
        raise ValueError("JE01_NIG_REAL_LOW_BIT_ROLE_COUNT_CHANGED")
    return blob,rows

def dequant_native_qn(blob,desc):
    bits,rows,cols,pb,pe,sb,se=desc
    groups=cols//64
    packed=np.frombuffer(blob,dtype=np.uint8,count=pe-pb,offset=pb)
    scale=np.frombuffer(blob,dtype="<f4",count=(se-sb)//4,offset=sb)
    if scale.size!=rows*groups or packed.size!=rows*cols*bits//8 or \
       not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError("JE01_SOURCE_NATIVE_Q3Q5_PACK_SCALE_INVALID")
    planes=np.unpackbits(packed.reshape(rows,groups,bits,8),axis=-1,bitorder="little")
    value=np.zeros((rows,groups,64),dtype=np.int16)
    for b in range(bits):
        value |= planes[:,:,b,:].astype(np.int16)<<b
    w=((value-(1<<(bits-1)))*scale.reshape(rows,groups,1)).reshape(rows,cols)
    if not np.isfinite(w).all():raise ValueError("JE01_NATIVE_QN_DEQUANT_NAN")
    return w.astype(np.float32,copy=False)

def square_error(a,ref):
    x=a.astype(np.float64,copy=False);y=ref.astype(np.float64,copy=False)
    delta=x-y
    return float(np.sum(delta*delta)),float(np.sum(y*y)),float(np.max(np.abs(delta)))

def run(out):
    if out.exists() or out.is_symlink() or out.parent!=ROOT:
        raise ValueError("JE01_SOURCE_DIAGNOSTIC_FRESH_REPORT_REQUIRED")
    prev=signed(ROOT/"all24_roles.json",SOURCE_DIGEST)
    if prev["status"]!="REAL_SOURCE_AF4_VS_CHAT_BF16_ALL_24_ROLE_DEQUANT_COMPARE_COMPLETE" or \
       prev["role_comparison_count"]!=24:
        raise ValueError("JE01_POSITIVE_REFERENCE_24ROLE_NOT_FULL")
    gold=signed(NIG_ROOT/"nig"/"JE01_EOE_NIG_16_GOLD_NLL.json",GOLD_NIG_DIGEST)
    off=signed(NIG_ROOT/"off"/"JE01_EOE_NIG_16_GOLD_NLL.json",GOLD_OFF_DIGEST)
    if gold["status"]!="ACTUAL_REAL_ORIGINAL27_Q35_NIG_PILOT16_MEASURED" or \
       off["status"]!="PASS_ORIGINAL27_NIG_BINARY_NO_OVERLAY_GOLD_PARITY" or \
       gold["gold_tokens"]!=656 or off["gold_tokens"]!=656 or \
       gold["mode"]!="nig" or off["mode"]!="off" or \
       gold["true_source_heldout_ood_labels_accessed"] or \
       off["true_source_heldout_ood_labels_accessed"]:
        raise ValueError("JE01_GOLD16_NIG_OFF_REAL_EVIDENCE_DRIFT")
    bundle,roles=real_nig_source()
    afpath=base.safe_file(base.AF_ROOT,"deepseek_moe_af.bin",base.AF_FILE_SHA,9814343680,rehash=True)
    layout=base.read_af_layout()
    wmap,shards,headers=base.read_bf16_headers()
    summary={"3":{"af_to_BF16_sse":0.0,"NIG_to_BF16_sse":0.0,"NIG_to_AF4_sse":0.0,"bf16_energy":0.0,
                  "AF4_energy":0.0,"coefficients":0,"roles":0},
             "5":{"af_to_BF16_sse":0.0,"NIG_to_BF16_sse":0.0,"NIG_to_AF4_sse":0.0,"bf16_energy":0.0,
                  "AF4_energy":0.0,"coefficients":0,"roles":0}}
    per=[];groupSums={n:0.0 for n in ("af_to_BF16_sse","NIG_to_BF16_sse","NIG_to_AF4_sse","bf16_energy","AF4_energy")}
    for i,(layer,expert) in enumerate(base.FROZEN_SLOTS):
      for role in base.ROLES:
        original,q4code,scales,bias,source=base.affine4_original(afpath,layout,layer,expert,role)
        bf16,tensor=base.bf16_selected(wmap,shards,headers,layer,expert,role)
        item=roles[(layer,expert,role)]
        bit=item[0]
        nig=dequant_native_qn(bundle,item)
        orig_sse,bf16_E,orig_max=square_error(original,bf16)
        nig_sse,_,nig_max=square_error(nig,bf16)
        diff_sse,af_E,diff_max=square_error(nig,original)
        row={"layer":layer,"expert":expert,"role":role,"nig_real_bits":bit,
             "coefficients":int(bf16.size),
             "native_AF4_role_packed_sha":source["af_packed_sha256"],
             "true_original_Chat_BF16_tensor_sha":tensor["original_bf16_tensor_sha256"],
             "original_AF4_relative_RMSE_to_BF16":math.sqrt(orig_sse/bf16_E),
             "new_NIG_relative_RMSE_to_BF16":math.sqrt(nig_sse/bf16_E),
             "new_NIG_relative_RMSE_to_original_AF4":math.sqrt(diff_sse/af_E),
             "NIG_minus_AF4_BF16_squared_error":nig_sse-orig_sse,
             "AF4_vs_BF16_original_q4_scale_requant_code_agreement":
               base.stats(original,bf16,q4code,scales,bias)["original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias"],
             "max_abs_AF4_minus_BF16":orig_max,
             "max_abs_NIG_minus_BF16":nig_max,
             "max_abs_NIG_minus_AF4":diff_max}
        per.append(row)
        bucket=summary[str(bit)]
        bucket["af_to_BF16_sse"]+=orig_sse
        bucket["NIG_to_BF16_sse"]+=nig_sse
        bucket["NIG_to_AF4_sse"]+=diff_sse
        bucket["bf16_energy"]+=bf16_E
        bucket["AF4_energy"]+=af_E
        bucket["coefficients"]+=int(bf16.size)
        bucket["roles"]+=1
        for metric in groupSums:
            groupSums[metric]+={"af_to_BF16_sse":orig_sse,
              "NIG_to_BF16_sse":nig_sse,"NIG_to_AF4_sse":diff_sse,
              "bf16_energy":bf16_E,"AF4_energy":af_E}[metric]
        print(json.dumps({"progress":len(per),"layer":layer,"expert":expert,"role":role,"nig_bits":bit,
          "original_AF4_BF16_RMSE":row["original_AF4_relative_RMSE_to_BF16"],
          "NIG_BF16_RMSE":row["new_NIG_relative_RMSE_to_BF16"],
          "NIG_AF4_RMSE":row["new_NIG_relative_RMSE_to_original_AF4"]},sort_keys=True),flush=True)
    if len(per)!=24 or summary["3"]["roles"]!=6 or summary["5"]["roles"]!=18:
        raise ValueError("JE01_REAL_NIG_24ROLE_BINARY_DECODE_NOT_COMPLETE")
    agg={}
    for bit,v in summary.items():
        agg[bit]={"roles":v["roles"],"coefficients":v["coefficients"],
          "AF4_relative_RMSE_to_BF16":math.sqrt(v["af_to_BF16_sse"]/v["bf16_energy"]),
          "NIG_relative_RMSE_to_BF16":math.sqrt(v["NIG_to_BF16_sse"]/v["bf16_energy"]),
          "NIG_relative_RMSE_to_original_AF4":math.sqrt(v["NIG_to_AF4_sse"]/v["AF4_energy"]),
          "delta_NIG_minus_AF4_weight_squared_error_ratio_to_BF16":
            (v["NIG_to_BF16_sse"]-v["af_to_BF16_sse"])/v["bf16_energy"]}
    aggregate={"coefficients":sum(v["coefficients"] for v in summary.values()),
       "original_AF4_RMSE_to_BF16":math.sqrt(groupSums["af_to_BF16_sse"]/groupSums["bf16_energy"]),
       "NIG_RMSE_to_BF16":math.sqrt(groupSums["NIG_to_BF16_sse"]/groupSums["bf16_energy"]),
       "NIG_RMSE_to_original_AF4":math.sqrt(groupSums["NIG_to_AF4_sse"]/groupSums["AF4_energy"]),
       "NIG_minus_AF4_weight_squared_error_ratio_to_BF16":
          (groupSums["NIG_to_BF16_sse"]-groupSums["af_to_BF16_sse"])/groupSums["bf16_energy"]}
    # Sanity negative control: wrong source expert in SAME layer, same role.
    layer,expert,role=21,61,"gate_proj"
    a,nibs,scale,bias,_=base.affine4_original(afpath,layout,layer,expert,role)
    correct,_=base.bf16_selected(wmap,shards,headers,layer,expert,role)
    wrong,_=base.bf16_selected(wmap,shards,headers,layer,60,role)
    ctrl_correct=base.stats(a,correct,nibs,scale,bias)
    ctrl_wrong=base.stats(a,wrong,nibs,scale,bias)
    negative={"original_AF4":{"layer":21,"expert":61,"role":role},
      "true_source_correct_expert_id":61,
      "decoy_different_expert_same_layer_id":60,
      "correct_same_source_weight_corr":ctrl_correct["pearson_weight_correlation"],
      "decoy_wrong_source_weight_corr":ctrl_wrong["pearson_weight_correlation"],
      "correct_same_source_nibble_agreement":ctrl_correct["original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias"],
      "decoy_wrong_source_nibble_agreement":ctrl_wrong["original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias"],
      "negative_control_bf16_wrong_expert_labels_used":False}
    result={"schema":SCHEMA,"status":"REAL_ORIGINAL_AF4_VS_SAME_CHAT_BF16_NIG_Q3Q5_24_ROLE_ERROR_ATTRIBUTION_COMPLETE",
      "true_original_9point8GB_AF4_sha256":base.AF_FILE_SHA,
      "true_original_BF16_Chat_four_shard_fullsha_rechecked":True,
      "previous_original_AF4_vs_BF16_24role_receipt":SOURCE_DIGEST,
      "actual_NIG43MB_native_BF16_Chat_source_sha256":NIG_DIGEST,
      "original_AF4_encoding":"AFFINE4_PER_GROUP_NIBBLE_SCALE_AND_SIGNED_BIAS",
      "NIG_encoding":"SYMMETRIC_N3_OR_N5_BITPLANES_PER_GROUP_SCALE_ZERO_CENTERED",
      "gold_token_656_real_NLL_original":off["actual_native_mean_gold_NLL"],
      "gold_token_656_real_NLL_NIG":gold["actual_native_mean_gold_NLL"],
      "actual_NIG_minus_AF4_real_656gold_NLL":gold["paired_mean_NLL_candidate_minus_original"],
      "full_gold_256_retested":False,"n_ood_labels_used":0,
      "summary_per_NIG_role_precision":agg,
      "aggregate_actual_weight_error":aggregate,
      "per_role":per,"wrong_expert_negative_control":negative,
      "same_source_checkpoint_bitwise_proof":"NOT_APPLICABLE_Q4_LOSSY",
      "exact_original_4bit_encoder_equivalence":"NOT_PROVEN",
      "original_transformer_model_forward_this_job":"NOT_RUN",
      "production_modified":False}
    result["receipt_sha256"]=base.canonical(result)
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        d=run(a.output)
        print(json.dumps({"status":d["status"],"receipt_sha256":d["receipt_sha256"],
              "aggregate":d["aggregate_actual_weight_error"],
              "by_bits":d["summary_per_NIG_role_precision"],
              "negative_control":d["wrong_expert_negative_control"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
              "reason":str(e)[:350]},sort_keys=True),flush=True)
        raise SystemExit(2)
