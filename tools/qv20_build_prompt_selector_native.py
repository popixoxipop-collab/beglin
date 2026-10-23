#!/usr/bin/env python3
"""Build derived Qv20 REAL original Apple Qwen28 C with low-capacity learned
prompt-token-only two-equal-byte mask selector. EXACT Qv19 original mutex path,
all original BF16,Q4/n8 kernels, sign file and forward NLL math are preserved.
Only changes Qv19 request role choice: n>=85 -> frozen calibration-only ridge.
FAIL CLOSED unless the policy has been frozen, pass/failed LOOCV recorded, and
trained ONLY on ARC train calibration. Never inspects heldout predictions/labels.
"""
import hashlib,json,math,os,pathlib,subprocess,difflib
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
SOURCE=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/build_v1/qv19_original_Qwen28_derived.c")
ORIGINAL_BUILD=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/build_v1/BUILD_RECEIPT.json")
PARENT_C_SHA="b3a2535c28cb276df66f0fc5ae677c0c0c4f996d34de3b3032907f3a5337fe32"
SCORE_SHA="10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72"
POLICY=ROOT/"policy/calibration_policy_private.json"
BASE_C=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv19-native-github-20261023/tools/qv19_original_native")
BUILD=ROOT/"native_prompt_router_Qwen28_Apple_C_v1"
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
PINS={"source_mask":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "random_mask":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde",
 "cal_lock":"6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def need(c,msg):
 if not c:raise RuntimeError("QV20_PROMPT_ROUTER_NATIVE_FAIL_CLOSED_"+msg)
def out(p,b):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def fmt(a):return ",".join(format(float(x),".17g") for x in a)
def choice_source(policy):
 accepted=int(policy["dynamic_policy_admitted_from_calibration"])
 fallback=int(policy["static_fallback_index"])
 coefs=policy["coefficients_intercept_and_slopes"]
 means=policy["feature_means"];scales=policy["feature_scales"]
 need(len(coefs)==6 and len(means)==len(scales)==5,"features_shape")
 need(all(math.isfinite(float(v)) for v in list(coefs)+means+scales),"finite_calibration_coeff")
 need(all(v>=.05 for v in scales) and fallback in (0,1),"nonzero_scale")
 return f'''
/* Qv20 frozen ARC-train-only source-token precision selector.
 * This function reads ONLY prompt IDs, never model gold answer or logits.
 * Candidate 0 Qv8 and candidate 1 random73 have IDENTICAL role count and
 * 765493248 packed projection weight bytes. Qv19 model lifetime lock follows.
 * A CALIBRATION-only LOOCV failing model falls back to the best static mask.
 */
static int qv20_source_only_choice(const int *ids,int n){{
    static const int dynamic_admitted={accepted};
    static const int fallback={fallback};
    static const double threshold={format(float(policy["profile_Qv8_if_predicted_delta_greater_than"]),".17g")};
    static const double mu[5]={{{fmt(means)}}};
    static const double scale[5]={{{fmt(scales)}}};
    static const double beta[6]={{{fmt(coefs)}}};
    if(!ids||n<18||n>208){{fprintf(stderr,"FATAL: Qv20 invalid prompt token length\\n");exit(3);}}
    if(!dynamic_admitted)return fallback;
    int unique=0,hi=0,repeated=0;
    double total=0;
    for(int i=0;i<n;i++){{
        int v=ids[i];
        if(v<0||v>=151936){{fprintf(stderr,"FATAL: Qv20 invalid token ID\\n");exit(3);}}
        total+=v;hi+=(v>=100000);
        if(i>0 && v==ids[i-1])repeated++;
        int found=0;
        for(int j=0;j<i;j++)if(ids[j]==v){{found=1;break;}}
        if(!found)unique++;
    }}
    double f[5]={{
        log((double)n/85.0),(double)unique/n,(double)hi/n,
        total/(151936.0*n),(double)repeated/(n-1)
    }};
    double pred=beta[0];
    for(int k=0;k<5;k++)pred+=beta[k+1]*(f[k]-mu[k])/scale[k];
    if(pred>threshold)return 0;
    if(pred<-threshold)return 1;
    return fallback;
}}
'''
def run():
 if BUILD.exists():raise FileExistsError("Qv20 native compiled project already exists; append-only")
 need(SOURCE.is_file() and sha(SOURCE)==PARENT_C_SHA,"exact_Qv19_real_C")
 need(POLICY.exists(),"calibration-only policy not frozen")
 d=json.loads(POLICY.read_text())
 need(d["training_split"].startswith("ARC-Challenge train") and
      d["calibration_input_SHA256"]==PINS["cal_lock"] and
      d["source_mask_sha256"]==PINS["source_mask"] and
      d["random_mask_sha256"]==PINS["random_mask"],"calibration_only_source_binding")
 raw=SOURCE.read_text()
 needle="call.target=(n>=85?1:0);call.tokens=ids;"
 need(raw.count(needle)==1,"only_original_native_selector_anchor")
 hook="static void *qv19_native_worker_run("
 need(raw.count(hook)==1,"native_model_lifetime_bridge_unique")
 modified=raw.replace(needle,"call.target=(qv20_policy_enabled ? qv20_source_only_choice(ids,n) : (n>=85?1:0));call.tokens=ids;",1)
 modified=modified.replace(hook,"static int qv20_policy_enabled=0;\n"+choice_source(d)+"\n"+hook,1)
 mode='} else if (!strcmp(mode,"qv19_guarded_choice")) {'
 need(modified.count(mode)==1,"original_native_command_selector_unique")
 # Isolated Qv20 native mode; same guarded function, but separate explicit opt-in.
 modified=modified.replace(mode,
   '} else if (!strcmp(mode,"qv20_prompt_aware_choice")) {\n'
   '        const char *v=getenv("BEGLIN_QV20_POLICY_SHA256");\n'
   '        if(!v || strcmp(v,"'+sha(POLICY)+'")){fprintf(stderr,"FATAL: Qv20 frozen policy SHA gate\\n");exit(4);}\n'
   '        qv20_policy_enabled=1;\n'
   '        int requests=argc>2?atoi(argv[2]):-1;\n'
   '        qv19_native_guarded_real_C(base,xtmp,logits,requests);\n'
   '        qv20_policy_enabled=0;\n'
   '    '+mode,1)
 a="            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);"
 b='        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]'
 ix=raw.index(a);iy=raw.index(b,ix)
 need(hashlib.sha256(raw[ix:iy].encode()).hexdigest()==SCORE_SHA,"old_C_NLL")
 jx=modified.index(a);jy=modified.index(b,jx)
 need(modified[jx:jy]==raw[ix:iy],"original_NLL_mathematical_code_modified")
 baseline=json.loads(ORIGINAL_BUILD.read_text())
 objs=baseline["original_23_kernel_objects_sha256"]
 original=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023/native_persisted_C_v2/BUILD_RECEIPT.json")
 entries=json.loads(original.read_text())["original_23_kernel_objects"]
 need(len(entries)==23 and all(sha(pathlib.Path(z["file"]))==z["sha256"] for z in entries),"orig_Q4_N8_kernel_bindings")
 OS=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/build_v1")
 need(sha(OS/"qv19_request_lease.c")==baseline["lease_source_sha256"],"original_Qv19_C_lease")
 os.umask(0o077);BUILD.mkdir(parents=True,mode=0o700)
 code=BUILD/"qv20_prompt_aware_Qwen28_derived.c";out(code,modified.encode())
 diff="\n".join(difflib.unified_diff(raw.splitlines(),modified.splitlines(),fromfile="Qv19_NATIVE",tofile="Qv20_ROUTER",n=2))+"\n"
 out(BUILD/"Qv19_to_Qv20.diff",diff.encode())
 import shutil
 shutil.copyfile(OS/"qv19_request_lease.c",BUILD/"qv19_request_lease.c")
 shutil.copyfile(OS/"qv19_request_lease.h",BUILD/"qv19_request_lease.h")
 obj=BUILD/"qv20.o"
 cmd=["clang","-O3","-w","-I",str(QWEN),"-I",str(BUILD),"-c",str(code),"-o",str(obj)]
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 out(BUILD/"compile.log",(result.stdout+"\n"+result.stderr).encode())
 if result.returncode:raise RuntimeError("Qv20 original C compile failed "+result.stderr[-3000:])
 leaseobj=BUILD/"lease.o"
 result=subprocess.run(["clang","-std=c11","-O2","-w","-I",str(BUILD),"-c",str(BUILD/"qv19_request_lease.c"),"-o",str(leaseobj)],capture_output=True,text=True,timeout=180)
 out(BUILD/"compile_lease.log",(result.stdout+"\n"+result.stderr).encode())
 if result.returncode:raise RuntimeError("original Qv19 lease compile failed "+result.stderr[-3000:])
 dest=BUILD/"beglin_qv20_2mask_source_only_ridge"
 components=[pathlib.Path(e["file"]) for e in entries]
 result=subprocess.run(["clang","-O3",str(obj),str(leaseobj)]+[str(x) for x in components]+
         ["-o",str(dest),"-framework","Accelerate","-lpthread","-lm"],capture_output=True,text=True,timeout=180)
 out(BUILD/"link.log",(result.stdout+"\n"+result.stderr).encode())
 if result.returncode:raise RuntimeError("Qv20 linking failed "+result.stderr[-2600:])
 dest.chmod(0o700)
 with dest.open("rb") as f:magic=f.read(8)
 need(magic==bytes.fromhex("cffaedfe0c000001"),"actual_Apple_MachO_ARM64")
 receipt={"schema":"BEGLIN-Qv20-original-Qwen28-C-bounded-prompt-only-ridge-mask-selector-build/1",
 "status":"ORIGINAL_C_PROMPT_ROUTER_APPLE_NATIVE_COMPILED__UNTESTED",
 "Qv19_original_derived_C_sha256":PARENT_C_SHA,
 "source_calibration_policy_SHA256":sha(POLICY),
 "Qv20_derived_C_sha256":sha(code),"Qv20_Apple_native_binary_SHA256":sha(dest),
 "unchanged_original_Qv19_lease_C_sha256":sha(BUILD/"qv19_request_lease.c"),
 "untouched_Qv17_NLL_score_source_SHA256":SCORE_SHA,
 "original_23_Q4_n8_kernel_object_SHA256":[{"name":p.name,"sha":sha(p)} for p in components],
 "source_only_input_predictor":True,"training_data_only":True,
 "dynamic_admitted_from_calibration":d["dynamic_policy_admitted_from_calibration"],
 "extra_inactive_alt_cache_96_MiB_is_not_same_total_RAM_as_fixed_mask":True,
 "no_production_release_change":True}
 out(BUILD/"BUILD_RECEIPT.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
 print("QV20_ACTUAL_PROMPT_ROUTER_NATIVE_COMPILED",json.dumps({"binary_sha256":sha(dest),"source_SHA256":sha(code),
 "cal_policy_SHA256":sha(POLICY),"dynamic_admitted":d["dynamic_policy_admitted_from_calibration"],
 "original_23_objects":len(components)},sort_keys=True),flush=True)
if __name__=="__main__":run()
