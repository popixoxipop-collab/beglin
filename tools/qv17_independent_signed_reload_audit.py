#!/usr/bin/env python3
"""Independent Qv17 original Qwen28 C signed packed alternate weight-cache audit.
Read only: 83.5 MiB packed file, dev Ed25519 signature, original 3GiB BF16 model,
native raw C logs, Qv16 pinned 48 inputs and independent original C static outputs.
No model invocation, no signing, no editing originals. Emits aggregate evidence only.
"""
import argparse,hashlib,json,math,os,pathlib,re,struct,statistics,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
ROOT=pathlib.Path(__file__).resolve().parents[1]
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
MAN=Q17/"owner_only_artifacts/alternate40_Qv8_random73_manifest.json"
SIGN=Q17/"owner_only_artifacts/alternate40_Qv8_random73_manifest.sig"
PUB=ROOT/"docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt"
RUN=Q17/"native_executions/reload48"
EXE=Q17/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
NATIVE_C=Q17/"native_persisted_C_v2/qv17_original_Qwen28_persisted_cache.c"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
MODEL=SNAP/"model.safetensors"
LOCK=Q16/"independent_arc_validation48/frozen_inputs_private.json"
EXE_SHA="16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b"
CSHA="0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa"
PUB_ROOT_SHA="5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847"
MAN_SHA="491deb61dbc30fad08b4a5d819b270b64167d5551b422823ae0b431f4a5c6824"
PACK_SHA="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
LOCK_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
BIN_MODEL_SHA="073400b311d049b6379b8f3f7ad5a0258268031dd009b99389cfd4030144850a"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SHAPES={"q_proj":(1536,1536),"k_proj":(256,1536),"v_proj":(256,1536),
        "o_proj":(1536,1536),"gate_proj":(8960,1536),"up_proj":(8960,1536),"down_proj":(1536,8960)}
R_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
R_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
R_SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
R_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
R_SUMCACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
R_LOAD=re.compile(r"^\[QV17-PACK-LOAD\] entries (\d+) bytes (\d+) sha256_check_ms ([0-9eE.+-]+) total_read_ms ([0-9eE.+-]+) digest ([a-f0-9]{64})$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def verify(test,what):
 if not test:raise RuntimeError("QV17_INDEPENDENT_VERIFICATION_FAILED: "+what)
def record(path):return json.loads(path.read_text())
def check_pack(mask):
 hdr=struct.Struct("<8sIIQ");ent=struct.Struct("<HHHHIIQQ")
 role_src={tuple(x.split()) for x in (Q16/"independent_arc_validation48/masks/source_qv8.txt").read_text().splitlines()}
 role_rnd={tuple(x.split()) for x in (Q16/"independent_arc_validation48/masks/random73.txt").read_text().splitlines()}
 different=[(r,l) for r in ROLE_ORDER for l in range(28) if ((r,str(l)) in role_src)!=((r,str(l)) in role_rnd)]
 verify(len(different)==40,"40 different source vs alternate roles")
 with PACK.open("rb") as f:
  magic,version,n,payload=hdr.unpack(f.read(hdr.size))
  verify(magic==b"BEGQ17A1" and version==1 and n==40 and payload<=96*1048576,"header/memory budget")
  expected=sum(1 for _ in different)
  visited=[];actual_payload=0
  for role,layer in different:
   rid,l,bits,res,out,ins,nbytes,sbytes=ent.unpack(f.read(ent.size))
   target=8 if (role,str(layer)) in role_rnd else 4
   expout,expin=SHAPES[role]
   verify(rid==ROLE_ORDER.index(role) and l==layer and bits==target and res==0,"tensor role-layer-bits")
   verify((out,ins)==(expout,expin),"actual model tensor dimensions")
   verify(nbytes==out*ins*(2 if bits==8 else 1)//2,"target packed byte size")
   verify(sbytes==out*ins//64*4,"scales size")
   f.seek(nbytes+sbytes,1);actual_payload+=nbytes+sbytes
   visited.append([role,layer,target])
  verify(actual_payload==payload and f.tell()==PACK.stat().st_size and PACK.stat().st_size==hdr.size+ent.size*40+payload,
        "packed bytes/trailing payload")
 return {"n":expected,"payload_bytes":payload,"total_file_bytes":PACK.stat().st_size,
         "status":"verified 40 distinct role-layer bitwidths and exact sizes"}
def audit():
 for p,h in [(EXE,EXE_SHA),(NATIVE_C,CSHA),(PUB,PUB_ROOT_SHA),(MAN,MAN_SHA),(PACK,PACK_SHA),(LOCK,LOCK_SHA)]:
  verify(p.is_file() and sha(p)==h,"SHA pinned path "+str(p.name))
 signer_raw=bytes.fromhex(PUB.read_text().strip().split(":",1)[1])
 public=Ed25519PublicKey.from_public_bytes(signer_raw)
 public.verify(SIGN.read_bytes(),MAN.read_bytes())
 doc=record(MAN)
 verify(doc["packed_file_sha256"]==PACK_SHA and doc["native_Qv17_executable_sha256"]==EXE_SHA and
        doc["native_Qv17_C_source_sha256"]==CSHA and
        doc["BF16_original_model_file_sha256"]==BIN_MODEL_SHA,"signed manifest contents")
 verify(sha(MODEL)==BIN_MODEL_SHA,"original BF16 model complete SHA-256")
 layout=check_pack(doc)
 verify(layout["payload_bytes"]==doc["packed_layout"]["payload_bytes"],"signed payload")
 runlog=RUN/"real_original_C_run.log"
 runreceipt=RUN/"verified_original_C_receipt.json"
 verify(runlog.is_file() and runreceipt.is_file(),"native reload evidence")
 receipt=record(runreceipt)
 verify(receipt["mode"]=="reload" and receipt["n"]==48 and receipt["original_native_executable_sha256"]==EXE_SHA,
        "actual Qv17 protected runner")
 verify(receipt["actual_native_log_SHA256"]==sha(runlog),"raw native C log digest")
 verify(receipt["local_developer_signed_manifest_validation"]["status"]=="ED25519_LOCAL_DEVELOPER_SIGNATURE_AND_PACK_SHA256_VERIFIED",
        "trusted signed bootstrap verified before native process")
 txt=runlog.read_text()
 rows=R_ROW.findall(txt);summ=R_SUM.findall(txt);sw=R_SWITCH.findall(txt);cs=R_CACHE.findall(txt);ct=R_SUMCACHE.findall(txt);load=R_LOAD.findall(txt)
 verify(len(rows)==len(sw)==len(cs)==48 and len(summ)==len(ct)==len(load)==1,"original native C rows")
 verify(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"real original 28L mixed representation")
 entries,bytes_,sha_ms,load_ms,checksum=load[0]
 verify(int(entries)==40 and int(bytes_)==layout["payload_bytes"] and checksum==PACK_SHA,"native loader SHA and exact entries")
 verify(float(load_ms)>=float(sha_ms)>=0,"native C time accounting")
 cases=record(LOCK)["requests"]
 original=record(Q16/"native48/prewarm1/receipt_private.json")
 qv8=record(Q16/"native_static48_logs/static_source.json")
 rnd=record(Q16/"native_static48_logs/static_random_chosen24.json")
 verify(rnd["selection_indices"]==list(range(1,48,2)),"independent original C static random subset")
 val=[];correct=[];numeric=0;static=0
 for i,((index,nll,y,pred,length),(swidx,pref,mask,changed,elapsed),(cacheidx,hit,miss,drop,current,peak,cap)) in enumerate(zip(rows,sw,cs)):
  verify(int(index)==int(swidx)==int(cacheidx)==i and int(length)==int(pref)==cases[i]["prefix_len"],"native original frozen input index "+str(i))
  verify(int(y)=="ABCD".index(cases[i]["correct"]),"original correct letter")
  need="source_qv8" if int(length)<85 else "random73"
  verify(mask==need and int(changed)==(0 if i==0 else 40),"original CPU request precision switch")
  verify(int(cap)==96*1048576 and int(current)<=int(cap) and int(peak)<=int(cap),"strict cache cap")
  verify(int(hit)==(0 if i==0 else 40) and int(miss)==0 and int(drop)==0,"zero request requantization")
  gold=float(nll);right=int(y==pred)
  verify(math.isfinite(gold) and gold>=0,"NLL finite")
  verify(gold==original["gold_fullvocab_token_NLL"][i] and right==original["correct_each"][i],
         "original Qv16 unchanged native C bit-exact")
  if i%2==0:
   target=qv8["gold_full_vocab_NLL"][i];corr=qv8["correct_each"][i]
  else:
   target=rnd["NLL_original_selected_24"][(i-1)//2];corr=rnd["correct_original_selected_24"][(i-1)//2]
  verify(gold==target and right==corr,"selected original static C bit-exact")
  val.append(gold);correct.append(right);numeric+=1;static+=1
  import struct
  needed=struct.pack("<"+"i"*len(cases[i]["prefix_ids"]),*cases[i]["prefix_ids"])
  verify((Q16/"independent_arc_validation48/dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()==needed,
         "original source-locked input token bytes")
 _,N,correctcount,avg,ppl=map(float,summ[0])
 verify(int(N)==48 and correctcount==sum(correct) and abs(sum(val)/48-avg)<1e-8,"native C model score summary")
 hits,misses,drops,live,peak,cap=map(int,ct[0])
 verify(hits==47*40 and misses==0 and drops==0 and peak<=cap==96*1048576,"native real-time cache hit and hard cap")
 return {
 "schema":"BEGLIN-Qv17-independent-original-C-restart-signed-precision-cache-48/1",
 "status":"ED25519_LOCAL_DEV_SIGNED_ARTIFACT_RELOADED_REAL_NATIVE_48_STATIC_PARITY_PASS",
 "real_Qwen28_native_rows_reparsed":numeric,"real_original_C_static_reference_exact":static,
 "actual_cache_hit":hits,"recalc_during_requests":misses,"actual_tensors_repacked":47*40,
 "cached_inactive_peak_mib":peak/1048576,
 "native_pack_sha256":PACK_SHA,"signed_manifest_sha256":MAN_SHA,
 "full_source_BF16_sha256":BIN_MODEL_SHA,"pinned_Qv17_exe_sha256":EXE_SHA,
 "packed_cache_layout":layout,
 "native_C_log_sha256":sha(runlog),"native_receipt_sha256":sha(runreceipt),
 "native_C_pack_sha_verify_ms":float(sha_ms),"native_C_signed_pack_digest_verified_load_total_ms":float(load_ms),
 "bootstrap_signed_manifest_and_full_3GB_model_sha_verify_ms":receipt["local_developer_signed_manifest_validation"]["verifier_total_ms"],
 "wall_s":receipt["native_process_wall_seconds"],
 "peak_native_process_RSS_MiB":receipt["native_process_peak_RSS_MiB"],
 "correct":int(correctcount),"mean_gold_NLL":avg,
 "original_Qv16_96MiB_first_quant_prewarm_ms":2412.228,
 "original_Qv16_request_warm_all47_switch_ms":13.611,
 "strict_model_integrity_verified_before_native_spawn":True,
 "local_dev_signing_key_NOT_external_prod_attestation":True,
 "no_model_files_original_kernels_or_operating_release_changed":True,
 "limits":[
 "Measuring a NEW process reload, not physical power-off/disk-cache-cold reboot",
 "Full original model SHA256 adds its own ~1.5s security cost; native packed load time is a separate number",
 "Caller must use signed Python bootstrap; direct C caller has SHA256 expected-digest guard but no native Ed25519 verification",
 "Model checkpoint SHA and tokenizer/mask SHA are exact but model quality across new architectures is untested",
 "The 48 validation items are prior Qv16 experimental items; proving mechanism, not independent accuracy advantage",
 "No Metal GPU or concurrent requests; original Q4/n8 model performance still under separate optimization",
 "Packed artifact's inactive 83.5MiB is an additional residency beyond active model, transient peak up to ~90MiB"
 ],"no_private_prompt_token_gold_case_or_C_logits_export":True}
if __name__=="__main__":
 cli=argparse.ArgumentParser();cli.add_argument("--output",type=pathlib.Path)
 opts=cli.parse_args()
 result=audit();data=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
 if opts.output:
  if opts.output.exists():raise FileExistsError("audit report append-only")
  opts.output.parent.mkdir(parents=True,exist_ok=True)
  with opts.output.open("xb") as f:f.write(data)
  print("QV17_SECOND_INDEPENDENT_NATIVE_RELOAD_AUDIT_SAVED",str(opts.output),hashlib.sha256(data).hexdigest())
 else:print(data.decode())
