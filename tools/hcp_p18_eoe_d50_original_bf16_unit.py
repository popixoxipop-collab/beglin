#!/usr/bin/env python3
"""P18 original 27L/5291T DeepSeek-V2-Lite BF16 gold-window teacher on EOE D50.

Zero XOX model writes. Runs ONLY an immutable 480-benchmark question, exact
original P17 torch decoder layer, P18 112→1024 window bound. Uses copied
original checkpoint after full shard SHA verification and verifies all 5291
source tensor *values* on every real teacher forward. Isolated append-only
D50 source. No terminal greedy accuracy or promotion is inferred.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,json,os,pathlib,psutil,resource,subprocess,sys,threading,time
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np,torch
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
P17=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/tools")
sys.path.insert(0,str(P17));sys.path.insert(0,str(BASE))
import hcp_p17_teacher_gold as og
from hcp_p15_streamed_teacher import TensorStore,original_classes
from hcp_p18_eoe_d50_original_bf16_core import P18LongGoldWindowTeacher
MODEL=BASE/"teacher_checkpoint_source_copy"
MANIFEST=BASE/"EOE_P18_BF16_MODEL_COPY_RECEIPT.json"
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
CORE="295ce59380f83162d20f619bf9d0cccc961f0c1484d16f5031fdc0040bb6bb78"
P17_SOURCE="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
CKPT="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
TENSOR_VALUES="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
MAX_RSS=7*1024**3
MIN_RAM=10*1024**3
MIN_D50_DISK=50*1024**3
MAX_NEW_SWAP=1024**3

def digest(path):
 p=pathlib.Path(path)
 if not p.is_file() or p.is_symlink():raise ValueError("Missing/symlink file: "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def token_sha(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def source_live():
 """Always observe, never edit, original serving on XOX."""
 root="/Users/xox/vdsp_serving"
 cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","xox","shasum","-a","256",
  root+"/persistent-stage/qwen_infer_gpu",root+"/active-route.json"]
 output=subprocess.check_output(cmd,text=True,timeout=18).strip().splitlines()
 if len(output)!=2:raise RuntimeError("XOX original native stage SHA check unavailable")
 chk=[line.split()[0] for line in output]
 psout=subprocess.check_output(["ssh","-o","BatchMode=yes","xox","ps","-axo","pid=,comm="],text=True,timeout=18)
 pids=sorted(int(line.split()[0]) for line in psout.splitlines()
             if "/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu" in line)
 identity={"stage_sha256":chk[0],"route_sha256":chk[1],"stage_pids":pids}
 if identity!={"stage_sha256":STAGE,"route_sha256":ROUTE,"stage_pids":[35481,35493]}:
  raise RuntimeError("XOX original serving mutated or unavailable")
 return identity
class Watch:
 def __init__(self):
  self.stop=threading.Event();self.peak=0
  self.start_swap=psutil.swap_memory().used;self.delta=0
 def check(self):
  mem=psutil.Process().memory_info().rss
  self.peak=max(self.peak,mem)
  self.delta=max(self.delta,max(0,psutil.swap_memory().used-self.start_swap))
  if mem>=MAX_RSS:return "RESEARCH_RSS_ABOVE_7GIB"
  if psutil.virtual_memory().available<MIN_RAM:return "EOE_HOST_AVAILABLE_RAM_BELOW_10GIB"
  if psutil.disk_usage(str(BASE)).free<MIN_D50_DISK:return "EOE_D50_FREE_BELOW_50GIB"
  if self.delta>=MAX_NEW_SWAP:return "NEW_SWAP_ABOVE_1GIB"
  return None
 def scan(self):
  while not self.stop.wait(.35):
   problem=self.check()
   if problem:
    print("P18_EOE_BF16_RESOURCE_ABORT",problem,flush=True)
    os._exit(92) # only this run-owned teacher process, never a service process
 def __enter__(self):
  reason=self.check()
  if reason:raise RuntimeError("EOE original teacher resource preflight: "+reason)
  self.thread=threading.Thread(target=self.scan,daemon=True);self.thread.start()
  return self
 def __exit__(self,*args):
  self.stop.set();self.thread.join(timeout=2)
  if args[0] is None and self.check():
   raise RuntimeError("EOE original teacher final resource guard: "+str(self.check()))
def preflight(index):
 if (not BASE.is_dir() or BASE.is_symlink() or
     not MODEL.is_dir() or MODEL.is_symlink() or
     not 0<=index<480 or digest(BASE/"p18_multishot_frozen_PRIVATE.json")!=FROZEN or
     digest(P17/"hcp_p17_teacher_gold.py")!=P17_SOURCE or
     digest(BASE/"hcp_p18_eoe_d50_original_bf16_core.py")!=CORE or
     digest(MODEL/"model.safetensors.index.json")!=CKPT):
  raise ValueError("EOE BF16 source/teacher/token bank or 27L original model identity drift")
 receipt=json.loads(MANIFEST.read_text())
 if (receipt.get("schema")!="beglin-hcp-p18-sha-verified-d50-original-BF16-model-copy-v1" or
     receipt.get("status")!="ORIGINAL_BF16_FOUR_SHARDS_COPY_SHA_VERIFIED"
     or receipt.get("original_source_index_sha256")!=CKPT or
     receipt.get("all_four_shards_source_digests_verified") is not True):
  raise RuntimeError("D50 original BF16 checkpoint copy is not yet SHA sealed")
 src=json.loads((BASE/"p18_multishot_frozen_PRIVATE.json").read_text())
 if len(src["items"])!=480 or sum(q["gold_token_count"] for q in src["items"])!=673:
  raise ValueError("EOE corpus no longer exact 480/673")
 q=src["items"][index];ids=q["prompt_ids"];gold=q["gold_ids"]
 if(not 1<=len(gold)<=8 or len(ids)+len(gold)>1024 or
    q["prompt_token_sha256"]!=token_sha(ids) or
    q["gold_token_sha256"]!=token_sha(gold) or
    q["prompt_token_count"]!=len(ids) or q["gold_token_count"]!=len(gold) or
    q["split"]!="test"):
  raise ValueError("EOE source frozen question/gold/token identity mismatch")
 before=source_live()
 if psutil.disk_usage(str(BASE)).free<MIN_D50_DISK or psutil.virtual_memory().available<16*1024**3:
  raise RuntimeError("EOE original BF16 resource preflight rejected")
 return q,before
def main(index):
 q,before=preflight(index)
 out=BASE/"eoe_480_stream"/("unit_%03d"%index)/"teacher"
 if out.parent.exists():raise FileExistsError("EOE original model result for this frozen unit already exists")
 out.mkdir(parents=True,mode=0o700)
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 started=time.monotonic()
 with Watch() as watchdog:
  classes,original_function_hashes=original_classes(MODEL/"modeling_deepseek.py")
  store=TensorStore(MODEL)
  model=P18LongGoldWindowTeacher(store,classes,dtype=torch.bfloat16,
   progress=lambda r:print("P18_EOE_REAL_ORIGINAL_BF16_LAYER",json.dumps(
    {"unit":index,"layer":r["layer"],"seconds":round(r["seconds"],3)},sort_keys=True),flush=True))
  logits=model.forward_gold(torch.tensor([q["prompt_ids"]+q["gold_ids"]],dtype=torch.int64),
                            len(q["prompt_ids"]),len(q["gold_ids"]))
  raw=logits.numpy()
  if(raw.dtype!=np.float32 or raw.shape!=(len(q["gold_ids"]),102400) or
     not np.isfinite(raw).all() or len(model.layer_records)!=27 or len(store.value_hashes)!=5291):
   raise RuntimeError("EOE whole original 27L/5291T BF16 gold F32 logits incomplete")
  hashtext=json.dumps(store.value_hashes,sort_keys=True,indent=2)+"\n"
  tensor_hash=hashlib.sha256(hashtext.encode()).hexdigest()
  if tensor_hash!=TENSOR_VALUES:raise RuntimeError("original 5291 BF16 tensor values differ from XOX source")
  x=raw.astype(np.float64);peak=x.max(axis=1)
  nll=peak+np.log(np.exp(x-peak[:,None]).sum(axis=1))-x[np.arange(len(q["gold_ids"])),q["gold_ids"]]
  if not (np.isfinite(nll).all() and (nll>=0).all() and (nll<150).all()):
   raise ValueError("Original BF16 gold NLL not real/finite")
  after=source_live()
  if after!=before:raise RuntimeError("XOX original serving mutated")
  npy=out/"original_BF16_full_vocab_gold.npy"
  with npy.open("xb") as f:np.save(f,raw,allow_pickle=False)
  record={
   "schema":"beglin-hcp-p18-original-d50-27L-BF16-teacher-frozen-unit-v1",
   "status":"P18_ORIGINAL_BF16_REAL_27L_5291T_ON_EOE_D50",
   "index":index,"item_id":q["item_id"],"dataset":q["dataset"],
   "frozen_480_source_sha256":FROZEN,
   "prompt_token_sha256":q["prompt_token_sha256"],
   "gold_token_sha256":q["gold_token_sha256"],
   "record_sha256":q["record_sha256"],
   "original_checkpoint_index_sha256":CKPT,
   "original_5291_tensor_values_manifest_sha256":tensor_hash,
   "original_decoder_layers":len(model.layer_records),
   "original_source_tensors_once":len(store.value_hashes),
   "real_BF16_gold_positions":len(q["gold_ids"]),
   "raw_F32_logits_shape":list(raw.shape),
   "raw_F32_logits_SHA256":digest(npy),
   "gold_NLL_nats":[float(y) for y in nll],
   "source_original_torch_function_hashes":original_function_hashes,
   "elapsed_seconds":time.monotonic()-started,
   "RSS_peak_bytes":watchdog.peak,"swap_delta_bytes":watchdog.delta,
   "original_XOX_serving_before":before,"original_XOX_serving_after":after,
   "native_A_B_evaluated":False,"task_accuracy_certified":False,
   "statistical_480_ninferiority_certified":False,
   "automatic_live_promotion":False,"production_touched":False}
  rec=out/"teacher_receipt.json"
  with rec.open("x") as f:json.dump(record,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_EOE_BF16_480_SINGLE_ITEM_COMPLETE",json.dumps({
   "index":index,"item_id":q["item_id"],"teacher_gold_mean_NLL":float(np.mean(nll)),
   "receipt_sha256":digest(rec),"source_logits_sha256":digest(npy),
   "RSS_GiB":round(watchdog.peak/2**30,3),"production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True)
 args=parser.parse_args()
 lock=BASE/"eoe_480_teacher_single_worker.lock"
 with lock.open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  main(args.index)
