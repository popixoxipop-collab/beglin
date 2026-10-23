#!/usr/bin/env python3
"""Qv16 original C static-random numerical reference ONLY for selected long 24 cases.
The exact 24 cases are indices 1,3,...47 in SHA-frozen validation48, chosen
before Qv16 native execution solely by prefix length >=85. Saves 50% CPU work
vs scoring random73 on 24 additional short cases the router never selects.
"""
import hashlib,json,math,os,pathlib,re,struct,subprocess,time,psutil
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
FROZEN=LAB/"independent_arc_validation48"
COHORT=LAB/"original_native_random_reference_chosen_long24"
RESULT=LAB/"native_static48_logs"
BIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed")
BW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BIN_SHA="f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"
PRIVATE_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
PUBLIC_SHA="ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a"
MASK_SHA="5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def shaf(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for part in iter(lambda:f.read(262144),b""):h.update(part)
 return h.hexdigest()
def need(cond,why):
 if not cond:raise RuntimeError("QV16_REFERENCE_FAIL_CLOSED_"+why)
def write(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def enc(a):return (json.dumps(a,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
def run():
 need(shaf(FROZEN/"frozen_inputs_private.json")==PRIVATE_SHA and shaf(FROZEN/"frozen_manifest_PUBLIC.json")==PUBLIC_SHA,"source_lock")
 need(shaf(BIN)==BIN_SHA,"original_unmodified_native_C_binary")
 need(shaf(FROZEN/"masks/random73.txt")==MASK_SHA,"original_random73")
 source=json.loads((FROZEN/"frozen_inputs_private.json").read_text())["requests"]
 indices=list(range(1,48,2))
 need(len(indices)==24 and all(source[i]["prefix_len"]>=85 for i in indices),"source_length_only_selection")
 if COHORT.exists():raise FileExistsError("source-frozen 24 reference already exists")
 os.umask(0o077);COHORT.mkdir(parents=True,mode=0o700)
 for j,i in enumerate(indices):
  row=source[i]
  ids=(FROZEN/"static_random/eval/ppl_work"/f"w{i}.i32").read_bytes()
  answer=(FROZEN/"static_random/eval/ppl_work"/f"w{i}.choices").read_bytes()
  need(ids==struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"]),"exact_pinned_input_"+str(i))
  need(answer==(" ".join(str(row["suffix_ids"][c]) for c in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n").encode(),"source_label_"+str(i))
  write(COHORT/"eval/ppl_work"/f"w{j}.i32",ids)
  write(COHORT/"eval/ppl_work"/f"w{j}.choices",answer)
  if j==0:write(COHORT/"prompt_ids.i32",ids)
 manifest={"schema":"BEGLIN-Qv16-static-original-C-selected-random73-only-long24/1",
 "source_frozen48_SHA256":PRIVATE_SHA,"original_mask_random73_sha256":MASK_SHA,
 "n":24,"original_48_indices":indices,"selection":"all 24 ODD indices of frozen alternation (long, random73); no logits or scores used"}
 write(COHORT/"SELECTION_MANIFEST_PRIVATE.json",enc(manifest))
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_QV","BEGLIN_Q_LAB_","BEGLIN_Q_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(COHORT),"QWEN_PROMPT":str(COHORT/"prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(FROZEN/"masks/random73.txt")})
 t=time.monotonic()
 p=subprocess.Popen([str(BIN),"masked_choice_qv5","24"],cwd=str(BW),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while p.poll() is None:
  try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.08)
 sout,serr=p.communicate();wall=time.monotonic()-t
 full=sout+b"\n"+serr;RESULT.mkdir(parents=True,exist_ok=True,mode=0o700)
 logfile=RESULT/"static_random_chosen24.log"
 write(logfile,full)
 if p.returncode:raise RuntimeError("original C failed "+str(p.returncode)+": "+serr.decode(errors="replace")[-1300:])
 txt=full.decode()
 rows=ROW.findall(txt);summ=TOTAL.findall(txt)
 need(len(rows)==24 and len(summ)==1,"original_C_rows")
 losses=[];correct=[]
 for j,(k,nll,ans,pred,prefix) in enumerate(rows):
  row=source[indices[j]]
  need(int(k)==j and int(prefix)==row["prefix_len"] and int(ans)=="ABCD".index(row["correct"]),"source_roundtrip_"+str(j))
  val=float(nll);need(math.isfinite(val) and val>=0,"NLLfinite")
  losses.append(val);correct.append(int(ans==pred))
 total,n,won,mean,ppl=map(float,summ[0])
 need(n==24 and won==sum(correct) and abs(total-sum(losses))<1e-7 and abs(mean-total/24)<1e-8,"original_C_total")
 need(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"original_mixed_roles")
 receipt={"schema":"BEGLIN-Qv16-Qwen28-original-C-static-random73-selected-long24/1",
 "status":"ORIGINAL_UNMODIFIED_QWEN28_C_24_SELECTED_LONG_INPUTS_REAL_SCORED",
 "source_Qv16_frozen48_SHA256":PRIVATE_SHA,"source_random_mask_SHA256":MASK_SHA,
 "original_C_binary_SHA256":BIN_SHA,"selection_indices":indices,
 "original_C_log_SHA256":shaf(logfile),"n":24,
 "NLL_original_selected_24":losses,"correct_original_selected_24":correct,
 "correct":sum(correct),"mean_gold_NLL":mean,"wall_seconds":wall,"peak_RSS_bytes":peak,
 "source_selection_uses_only_pre_frozen_length":True,
 "no_original_model_or_release_change":True}
 receipt_file=RESULT/"static_random_chosen24.json";write(receipt_file,enc(receipt))
 print("QV16_ORIGINAL_RANDOM24_STATIC",json.dumps({"correct":sum(correct),"n":24,
 "mean_goldNLL":mean,"wall_s":round(wall,3),
 "receipt_sha256":shaf(receipt_file),"log_sha256":shaf(logfile)},sort_keys=True),flush=True)
if __name__=="__main__":run()
