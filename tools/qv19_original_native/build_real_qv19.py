#!/usr/bin/env python3
"""Build a Qv19 Apple Qwen28 two-pthread EXCLUSIVE model lease native research binary.
Directly uses SHA-pinned Qv17 source, exact tested Qv19 lease C/bridge, and 23
unaltered original Q4/n8 object files. No original modification or promotion.
"""
import pathlib,hashlib,json,os,subprocess,difflib
REPO=pathlib.Path(__file__).resolve().parents[2]
SRC=REPO/"tools/qv19_original_native"
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023/native_persisted_C_v2")
ORIGINAL=Q17/"qv17_original_Qwen28_persisted_cache.c"
BASELINE=Q17/"BUILD_RECEIPT.json"
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
OUT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/build_v1")
SHA={
 "parent":"0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa",
 "score":"10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72",
 "lease_c":"5788e0f4d2afbf592f93e7ca2bc6e1cbe0b3f5ecdfcaace3742b157842ad3b93",
 "lease_h":"5fc29c9e25bf6148166f823eb1587c90ebe41d6cfe6d7a6e8e5f4d711398ec8d",
 "bridge":"aab974fa09b4ff65d2559aeaa1aff2bb9f03dcfb7743d51dbc11b41dd3363cd7"}
BRANCH='    } else if (!strcmp(mode,"masked_choice_qv5") || !strcmp(mode,"qv14_dynamic_choice")) {'
GOLD_START='            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);'
GOLD_END='        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]'
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def need(ok,reason):
 if not ok:raise RuntimeError("QV19_FAIL_CLOSED_"+reason)
def save(p,data):
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def cmd(label,argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=200)
 save(OUT/(label+".log"),(r.stdout+"\n"+r.stderr).encode())
 if r.returncode:raise RuntimeError("QV19_"+label+"_FAILED: "+r.stderr[-2800:])
def run():
 if OUT.exists():raise FileExistsError("Qv19 research build v1 exists; immutable")
 for n,p in (("parent",ORIGINAL),("lease_c",SRC/"qv19_request_lease.c"),
             ("lease_h",SRC/"qv19_request_lease.h"),("bridge",SRC/"qv19_original_c_bridge.inc")):
  need(p.is_file() and sha(p)==SHA[n],"source_SHA_"+n)
 old=ORIGINAL.read_text();bridge=(SRC/"qv19_original_c_bridge.inc").read_text()
 need(old.count(BRANCH)==1 and old.count("\nint main(")==1,"original_C_single_entry")
 need(old.count(GOLD_START)==1 and old.count(GOLD_END)==1,"gold_score_anchor_unique")
 a=old.index(GOLD_START);b=old.index(GOLD_END,a)
 need(hashlib.sha256(old[a:b].encode()).hexdigest()==SHA["score"],"parent_gold_scorer")
 need(bridge.count("qv19_run_request(")==1,"lease_bridge_entry_count")
 new=old.replace("\nint main(","\n"+bridge+"\nint main(",1)
 newbranch='''    } else if (!strcmp(mode,"qv19_guarded_choice")) {
        int requests=argc>2?atoi(argv[2]):-1;
        qv19_native_guarded_real_C(base,xtmp,logits,requests);
'''
 new=new.replace(BRANCH,newbranch+BRANCH,1)
 aa=new.index(GOLD_START);bb=new.index(GOLD_END,aa)
 need(new[aa:bb]==old[a:b],"parent_nll_score_modified")
 baseline=json.loads(BASELINE.read_text());entries=baseline.get("original_23_kernel_objects")
 need(isinstance(entries,list) and len(entries)==23,"original_23_reference_records")
 existing={}
 for e in entries:
  path=pathlib.Path(e["file"])
  need(path.is_file() and sha(path)==e["sha256"],"kernel_object_hash_"+path.name)
  existing[str(path.resolve())]=path
 qng=QWEN/"lab_local/qng64_n8_repacked_full_20261022"
 allfiles=[qng/x for x in ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 allfiles+=sorted(x for x in (QWEN/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 need(len(existing)==23 and len(allfiles)==23 and set(str(x.resolve()) for x in allfiles)==set(existing),"original_23_link_closure")
 os.umask(0o077);OUT.parent.mkdir(parents=True,exist_ok=True,mode=0o700);OUT.mkdir(mode=0o700)
 derived=OUT/"qv19_original_Qwen28_derived.c";save(derived,new.encode())
 for n in ("qv19_request_lease.c","qv19_request_lease.h"):
  save(OUT/n,(SRC/n).read_bytes())
 diff="\n".join(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile="pinned_Qv17",tofile="derived_Qv19",n=2))+"\n"
 save(OUT/"Qv17_to_Qv19.diff",diff.encode())
 cmd("compile_C",["clang","-O3","-w","-I",str(QWEN),"-I",str(OUT),"-c",str(derived),"-o",str(OUT/"main.o")])
 cmd("compile_lease",["clang","-std=c11","-O2","-w","-I",str(OUT),"-c",str(OUT/"qv19_request_lease.c"),"-o",str(OUT/"lease.o")])
 exe=OUT/"beglin_qv19_original_C_exclusive_lease"
 cmd("link",["clang","-O3",str(OUT/"main.o"),str(OUT/"lease.o")]+[str(x) for x in allfiles]+["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"])
 exe.chmod(0o700)
 with exe.open("rb") as f:magic=f.read(8)
 need(magic==bytes.fromhex("cffaedfe0c000001"),"not_Apple_arm64_MachO")
 receipt={"schema":"BEGLIN-Qv19-original-real-Apple-Qwen28-derived-two-thread-lease/1",
 "status":"REAL_APPLE_NATIVE_DERIVED_C_BINARY_COMPILED_NOT_EVALUATED",
 "original_qv17_C_sha256":SHA["parent"],"original_scoring_source_sha256":SHA["score"],
 "lease_source_sha256":SHA["lease_c"],"lease_header_sha256":SHA["lease_h"],
 "bridge_source_sha256":SHA["bridge"],"derived_C_sha256":sha(derived),
 "native_executable_sha256":sha(exe),
 "original_23_kernel_objects_sha256":{p.name:sha(p) for p in allfiles},
 "same_process_two_pthreads_with_exclusive_forward":True,"parallel_forward_proven":False,
 "original_checkpoint_model_and_operator_release_untouched":True}
 b=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
 save(OUT/"BUILD_RECEIPT.json",b)
 print("QV19_ACTUAL_APPLE_C_NATIVE_COMPILED",json.dumps({"original_C":SHA["parent"],
  "derived_C":receipt["derived_C_sha256"],"executable_sha256":receipt["native_executable_sha256"],
  "source_NLL_unchanged":True,"kernel_objects":23,"directory":str(OUT)},sort_keys=True),flush=True)
if __name__=="__main__":run()
