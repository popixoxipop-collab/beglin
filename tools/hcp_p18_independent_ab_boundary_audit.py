#!/usr/bin/env python3
"""P18 independent *smoke-only* receipt+GPU log audit, with no quality certification.
XOX authoritative files were copied to EOE D50; hashes pinned before this script.
"""
from __future__ import annotations
import hashlib,json,pathlib,re
R=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
E=R/"native1024_real_evidence"
BIN="12a50d4282596ebcf4297f28694698e97b6f4fac11ebe2117bf1701f94dd4591"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
DEV203="b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c"
DEV962="7780310965c84ac74203657bce486acee763c3d53854a088e8d37b06ceb0c3e5"
SAMPLES=[
 ("203_B","p18_devonly203_B_native_gpu_receipt.json","09bd83bbb68ac1f090c0c1713a770fb0a266110c66c487fd016860c08feb75f9","p18_devonly203_B_worker.log","0a31c21ac9d270ede1e7199adcd1aee8f721df4000b18197565ef35c8c06bf56",203,"B",DEV203),
 ("962_A","p18_devonly962_A_native_gpu_receipt.json","b79482ce30743713c57484e1543869aa2a2d27393dec99fce05d1619aa39b7c8","p18_devonly962_A_worker.log","1053349894e2661e5e3f40636aabaed6558c2979e2d050db3ed60142aaaab068",962,"A",DEV962),
 ("962_B","p18_devonly962_B_native_gpu_receipt.json","09d636a46938e5b5a1b9f8366276036713fa11c7bc9dee2ca4f06fd0bca347a5","p18_devonly962_B_worker.log","ec2c120df37c4baf2074c4b42e54c34ea0204715ee2083a05648728d29b26203",962,"B",DEV962)
]
def sha(path):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError("Unexpected source file type")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def snapshot_is_unchanged(snapshot):
 return snapshot=={"stage_sha256":STAGE,"route_sha256":ROUTE,
                    "stage_pids":[35481,35493],"scratch_pids":[]}
def audit():
 obs=[]
 for name,rec_path,rec_sha,log_path,log_sha,prompt_len,policy,input_sha in SAMPLES:
  rec=E/rec_path; log=E/log_path
  if sha(rec)!=rec_sha or sha(log)!=log_sha:raise RuntimeError("Untrusted copy/raw SHA: "+name)
  d=json.loads(rec.read_text())
  if(d.get("candidate_binary_sha256")!=BIN or d.get("input_dev_SHA256")!=input_sha
      or d.get("production_touched") is not False
      or d.get("P18_noninferiority_certified") is not False
      or d.get("automatic_live_promotion") is not False
      or d.get("external_test_questions_used_in_this_probe")!=0
      or not snapshot_is_unchanged(d.get("original_stage_before"))
      or not snapshot_is_unchanged(d.get("original_stage_after"))):
   raise RuntimeError("P18 smoke source/prod boundary mismatch: "+name)
  o=d.get("observation",{})
  if(o.get("input_tokens")!=prompt_len or o.get("native_generated_tokens")!=1
     or o.get("finite_logits") is not True
     or type(o.get("rss_bytes"))!=int or not (0<o["rss_bytes"]<14*1024**3)
     or type(o.get("runtime_ms"))not in (int,float) or not (0<o["runtime_ms"]<240000)
     or not isinstance(o.get("ack_sha256"),str) or len(o["ack_sha256"])!=64):
   raise RuntimeError("Native output/ACK/RSS evidence malformed: "+name)
  if policy=="B":
   if(o.get("precision_transition_status")!="REBIND_SET_APPLIED"
       or o.get("precision_change_count")!=2
       or o.get("precision_after_epoch")!=o.get("precision_before_epoch",-99)+1
       or d.get("experimental_policy")!="B"):
    raise RuntimeError("B transition/epoch contract missing: "+name)
  elif d.get("policy")!="A":raise RuntimeError("A policy identity invalid")
  lines=log.read_text(errors="replace")
  success=re.findall(r"GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off finite_logits=1 logits_checked=(\d+) requests=1",lines)
  produced=re.findall(r"req 0 prompt 0 slot 0 .* nout 1 tokens: (\d+)",lines)
  if (len(success)!=1 or int(success[0])!=prompt_len*102400
      or len(produced)!=1):
   raise RuntimeError("Actual native trace missing/malformed: "+name)
  generated=int(produced[0]);hash_token=hashlib.sha256(str(generated).encode()).hexdigest()
  if hash_token != o.get("generated_token_sha256"):
   raise RuntimeError("Receipt generated token hash disagrees with actual GPU log: "+name)
  if policy=="B" and not re.search(r"REBIND_SET_APPLIED epoch=3 changed_targets=2",lines):
   raise RuntimeError("Missing actual B precision ACK in GPU log")
  obs.append({"sample":name,"policy":policy,"prompt_tokens":prompt_len,"generated":1,
    "generated_token_sha256":hash_token,"num_logits_finite_checked":int(success[0]),
    "rss_bytes":o["rss_bytes"],"total_runtime_ms":o["runtime_ms"],
    "receipt_sha256":rec_sha,"log_sha256":log_sha,"native_ack_sha256":o["ack_sha256"]})
 for length in (203,962):
  a=next((q for q in obs if q["sample"]==f"{length}_A"),None)
  b=next((q for q in obs if q["sample"]==f"{length}_B"),None)
  if length==962 and (not a or not b or a["generated_token_sha256"]!=b["generated_token_sha256"]):
   raise RuntimeError("962 policy A/B did not agree on a single token")
 prior=json.loads((E/"p18_devonly203_native_gpu_receipt.json").read_text())["observation"]
 if prior["generated_token_sha256"]!=obs[0]["generated_token_sha256"]:
  raise RuntimeError("203 policy A/B one-token agreement absent")
 out={"schema":"beglin-hcp-p18-a-b-203-963-independent-raw-log-audit-v1",
  "status":"P18_A_B_203_AND_963_NATIVE_GPU_SMOKES_VERIFIED_NOT_QUALITY",
  "audited_native_smokes":3,"included_prior_A_203":True,
  "observations":obs,"203_A_B_first_token_agree":True,"962_A_B_first_token_agree":True,
  "maximum_observed_input_tokens":962,"maximum_observed_total_position":963,
  "P18_all_480_BF16_teachers_executed":False,
  "P18_480_AB_full_logits_executed":False,
  "NLL_nondinferiority_certified":False,
  "production_touched":False,"automatic_live_promotion":False}
 output=R/"P18_NATIVE_AB_BOUNDARY_INDEPENDENT_AUDIT.json"
 if output.exists():raise FileExistsError("Refuse to overwrite sealed audit")
 with output.open("x") as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 print(json.dumps({"status":out["status"],"audited":3,"maximum_total_position":963,
   "receipt_sha256":sha(output)},sort_keys=True))
if __name__=="__main__":audit()
