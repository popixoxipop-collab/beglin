#!/usr/bin/env python3
"""Build Qv15 paired-downgrade-then-upgrade precision cache variant.
Same original C inference and original Qv14 quantization; only cache swap ORDER
changes. Hard 96-MiB inactive buffer budget remains enforced. No original writes.
"""
import pathlib,subprocess,json,hashlib,os,difflib
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv15-bounded-cache-20261023")
BASE=LAB/"cached_original_C_native_build"
DEST=LAB/"cached_original_C_native_build_paired_v2"
OLD=BASE/"qv15_from_pinned_original_Qv14_v3.c"
SHA_OLD="0b1a069384d042b1b1164eb6ca05499000f21219e5692773476021edfeaafd79"
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(262144),b""):h.update(x)
 return h.hexdigest()
def store(path,data):
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 path.chmod(0o600)
REPL=r'''        /* PAIR EACH SAME-ROLE DOWNGRADE WITH AN UPGRADE:
           These two frozen masks have exactly equal per-role N8 cardinality.
           No intermediate live increase above 28 N8 slots, and at most ONE
           just-downgraded role's extra inactive cache bytes before its paired
           promotion releases another cached N8 tensor. This avoids the v1
           global-downgrade loop transient spike that evicted one hot slot. */
        for(int r=0;r<7;r++){
            int role=qv14_roles_id[r],down[28],up[28],nd=0,nu=0;
            for(int l=0;l<28;l++){
                char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
                WT *w=wt(name);
                int active8=(w->kind==K_QNG64&&w->bits==8);
                int want8=desired[role][l];
                if(active8&&!want8)down[nd++]=l;
                else if(!active8&&want8)up[nu++]=l;
            }
            if(nd!=nu){
                fprintf(stderr,"FATAL: Qv15 paired same-role static mask cardinality violated role %d down %d up %d\n",
                    role,nd,nu);exit(1);
            }
            for(int i=0;i<nd;i++){
                qv14_repack_one(role,down[i],0);changed++;
                qv14_repack_one(role,up[i],1);changed++;
            }
        }
'''
def main():
 if DEST.exists():raise FileExistsError("paired cached version already exists")
 if sha(OLD)!=SHA_OLD:raise RuntimeError("pinned original Qv15 cache C changed")
 s=OLD.read_text()
 start="        /* Downgrade before upgrade so resident packed bytes cannot exceed"
 end="        qv14_profile=target;"
 if s.count(start)!=1 or s.count(end)!=1:raise RuntimeError("C source anchor drift")
 idx=s.index(start);tail=s.index(end,idx)
 after=s[:idx]+REPL+s[tail:]
 before_lines=s.splitlines();after_lines=after.splitlines()
 diff=list(difflib.unified_diff(before_lines,after_lines,fromfile="cache_v1",tofile="paired_cache_v2"))
 assert sum(x.startswith("@@") for x in diff)==1
 os.umask(0o077);DEST.mkdir(parents=True,exist_ok=False,mode=0o700)
 code=DEST/"qv15_original_C_paired_cache_v2.c"
 store(code,after.encode())
 store(DEST/"v1_to_v2_source.diff",("\n".join(diff)+"\n").encode())
 obj=DEST/"qv15_paired_source.o";exe=DEST/"beglin_qv15_bounded_cache_paired_v2"
 compile=subprocess.run(["clang","-O3","-w","-I",str(QWEN),"-c",str(code),"-o",str(obj)],capture_output=True,text=True,timeout=180)
 store(DEST/"compile_log.txt",(compile.stdout+"\n"+compile.stderr).encode())
 if compile.returncode:raise RuntimeError("paired C compile failure "+compile.stderr[-2200:])
 objs=[QWEN/"lab_local/qng64_n8_repacked_full_20261022"/n for n in ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 others=sorted(x for x in (QWEN/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(others)!=20 or not all(o.is_file() for o in objs):raise RuntimeError("original 23 object link closure changed")
 link=subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in objs+others]+["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],capture_output=True,text=True,timeout=180)
 store(DEST/"link_log.txt",(link.stdout+"\n"+link.stderr).encode())
 if link.returncode:raise RuntimeError("paired C link failure "+link.stderr[-2200:])
 exe.chmod(0o700)
 receipt={"schema":"BEGLIN-Qv15-original-C-paired-WT-bounded-inactive-cache-build/2",
 "state":"REAL_C_COMPILED_BINARY_NOT_BENCHMARKED",
 "unchanged_native_kernel_objects":len(objs+others),
 "original_quant_cache_C_v1_SHA256":SHA_OLD,
 "derived_paired_C_SHA256":sha(code),"native_paired_executable_SHA256":sha(exe),
 "source_diff_SHA256":sha(DEST/"v1_to_v2_source.diff"),
 "one_hunk_source_diff":True,
 "only_semantic_change":"global downgrades then global upgrades -> same-role downgrade+upgrade pairs preserving active 28n8 and keeping inactive cache under 96 MiB",
 "unchanged_wt_packed_quant_math":True,
 "fixed_mask_role_cardinality_checked_at_runtime":True,
 "no_operator_release_or_original_weights_changed":True}
 store(DEST/"BUILD_RECEIPT.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
 print("QV15_PAIRED_C_NATIVE_BUILT",json.dumps({"executable":str(exe),"exe_sha256":sha(exe),"source_sha256":sha(code),"one_hunk_diff":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
