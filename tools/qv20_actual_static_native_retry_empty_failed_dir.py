#!/usr/bin/env python3
"""Qv20 genuine unchanged Apple Qwen28 ORIGINAL C reference scorer.
Runs exactly ONE frozen split, ONE role-and-byte-matched 28n8 mask in one native C
process. No adaptive policy, no use of heldout scores to fit policy.
The private original C logs and per-case gold NLL stay owner-only on EOE D50.
"""
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time
import psutil
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
EXE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
PINS={
 "native_original_static_exe":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221",
 "public_freeze":"2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a",
 "cal_arc_train":"6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28",
 "test_arc_validation":"b0a27e22f0c5769f49c3dc95d2200feef9f06747349abef55bd82c29d13e2119",
 "test_ob_validation":"2f80539f7eca065e8dd5effcc1c320e2fbcf74f7e6af0e1e2816a8fe72aeb204",
 "static_source":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "static_random":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
R_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
R_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def check(ok,why):
 if not ok:raise RuntimeError("QV20_ORIGINAL_C_STATIC_FAIL_"+why)
def encode(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
def output_new(p,content):
 with p.open("xb") as f:f.write(content);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def run(split,arm):
 check(split in ("cal_arc_train","test_arc_validation","test_ob_validation"),"split")
 check(arm in ("static_source","static_random"),"mask")
 check(sha(EXE)==PINS["native_original_static_exe"],"source_Qwen28_original_Q4N8_C")
 check(sha(ROOT/"Qv20_PREEXEC_PUBLIC.json")==PINS["public_freeze"],"Qv20_preexecution_lock")
 check(sha(ROOT/split/"frozen_inputs_private.json")==PINS[split],"split_prelock")
 check(sha(ROOT/"masks"/(arm+".txt"))==PINS[arm],"fixed_mask_equal_budget")
 d=json.loads((ROOT/split/"frozen_inputs_private.json").read_text())
 rows=d["selected_rows"];n=len(rows)
 check(n==(32 if split=="cal_arc_train" else 24),"frozen_split_size")
 for i,row in enumerate(rows):
  src=ROOT/split/arm/"eval/ppl_work"
  ids=struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"])
  choices=(" ".join(str(row["suffix_ids"][x]) for x in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n").encode()
  check((src/f"w{i}.i32").read_bytes()==ids,"frozen_input_"+str(i))
  check((src/f"w{i}.choices").read_bytes()==choices,"frozen_answer_"+str(i))
 folder=ROOT/"native_original_C_static"/split/arm
 os.umask(0o077)
 if folder.exists():
  # Qv20 random73 train32 first job FAILED while writing an output log because
  # macOS denied D50 access; repair must preserve the original directory, never
  # overwrite evidence. Retry only this EXACT case after owner re-authorizes.
  if split!="cal_arc_train" or arm!="static_random":
   raise FileExistsError("original C evaluation already exists: "+str(folder))
  if any(folder.iterdir()):
   raise RuntimeError("failed-job path is not empty: preserve evidence, do not retry")
 else:
  folder.mkdir(parents=True,exist_ok=False,mode=0o700)
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 env.update({
  "VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
  "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
  "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
  "QWEN_BASE":str(ROOT/split/arm),
  "QWEN_PROMPT":str(ROOT/split/arm/"prompt_ids.i32"),
  "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
  "BEGLIN_QV2_NATIVE_ASYNC":"0",
  "BEGLIN_Q_LAB_N8_ROLES_FILE":str(ROOT/"masks"/(arm+".txt"))})
 t=time.monotonic()
 process=subprocess.Popen([str(EXE),"masked_choice_qv5",str(n)],
   cwd=str(QWEN),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while process.poll() is None:
  try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.08)
 sout,serr=process.communicate();wall=time.monotonic()-t
 logfile=folder/"native_original_C_Qwen28.log"
 output_new(logfile,sout+b"\n"+serr)
 if process.returncode!=0:raise RuntimeError("original C terminated "+str(process.returncode)+": "+serr.decode(errors="replace")[-1800:])
 text=logfile.read_text()
 found=R_ROW.findall(text);summ=R_SUM.findall(text)
 check(len(found)==n and len(summ)==1,"actual_percase_C_log_rows")
 check(text.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"Qwen28_native_mix_weights")
 gold=[];correct=[]
 for i,(window,nll,ans,pred,prefix) in enumerate(found):
  row=rows[i];value=float(nll);right=int(ans==pred)
  check(int(window)==i and int(prefix)==row["prefix_len"] and int(ans)=="ABCD".index(row["correct"]), "native_token_id_"+str(i))
  check(math.isfinite(value) and value>=0,"original_goldNLL_valid_"+str(i))
  gold.append(value);correct.append(right)
 total,count,won,mean,ppl=map(float,summ[0])
 check(int(count)==n and int(won)==sum(correct) and abs(sum(gold)-total)<1e-7 and abs(mean-total/n)<1e-8,"native_model_metrics")
 receipt={"schema":"BEGLIN-Qv20-actual-original-Qwen28-C-single-static-mask-case-scores/1",
 "state":"ACTUAL_ORIGINAL_QWEN28_C_STATIC_"+arm.upper()+"_PASS",
 "cohort":split,"arm":arm,"n":n,
 "original_Qwen28_C_executable_sha256":PINS["native_original_static_exe"],
 "source_frozen_private_SHA256":PINS[split],"source_public_preexec_SHA256":PINS["public_freeze"],
 "real_native_original_C_log_SHA256":sha(logfile),
 "original_full_vocab_gold_nexttoken_NLL":gold,
 "original_choice_correct_each":correct,
 "correct":sum(correct),"mean_NLL":mean,
 "actual_native_process_wall_s":wall,"actual_native_peak_RSS_MiB":peak/1048576,
 "real_native_Q4_N8_weights_active":True,
 "no_model_kernel_or_operator_release_changes":True}
 priv=folder/"PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json";output_new(priv,encode(receipt))
 public={k:v for k,v in receipt.items() if k not in ("original_full_vocab_gold_nexttoken_NLL","original_choice_correct_each")}
 public["private_scores_receipt_SHA256"]=sha(priv)
 output_new(folder/"PUBLIC_AGGREGATE.json",encode(public))
 print("BEGLIN_QV20_ACTUAL_C_STATIC_ARM",json.dumps({"cohort":split,"arm":arm,"n":n,"correct":sum(correct),
  "NLL":round(mean,8),"native_wall_s":round(wall,3),"RSS_MiB":round(peak/1048576,2),
  "private_receipt_SHA256":sha(priv),"public_receipt_SHA256":sha(folder/"PUBLIC_AGGREGATE.json")},sort_keys=True),flush=True)
if __name__=="__main__":
 cli=argparse.ArgumentParser();cli.add_argument("--cohort",choices=["cal_arc_train","test_arc_validation","test_ob_validation"],required=True)
 cli.add_argument("--arm",choices=["static_source","static_random"],required=True)
 a=cli.parse_args();run(a.cohort,a.arm)
