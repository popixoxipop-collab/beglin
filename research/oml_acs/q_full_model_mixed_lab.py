#!/usr/bin/env python3
"""Build isolated BEGLIN full-Qwen mixed q4 / qNg64 n8 model worker.

Injects a SHA-pinned opt-in research-only per-(role,layer) bit mask into a
COPY of original qwen_infer.c; never alters tracked native engine source.
Mixed policies were frozen from 196 source-only weight distortion values.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
from q_full_model_lab_prepare import check_source_and_patch, OFFICIAL_REV, MODEL_REV, MODEL, SOURCE

OLD=ROOT/"lab_local/q_full_model_20261021_v2"
SCAN=ROOT/"lab_local/q_tensor_proxy_20261021"
OUT=ROOT/"lab_local/q_full_model_mixed_20261021"
ORIGINAL_SHA="52c8f5fd683b5ef0726e4738ab4b441624f54c807a3e71fb9d932cb79938d112"
SCAN_SHA="720de694838d3a85156afae3732974f3867db1c8ce4957e55778da77724a0d8a"
ROLES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
PREP_MODELS=("proxy","random")

def digest(data:bytes):return hashlib.sha256(data).hexdigest()

MIXED_POLICY_CODE=r"""
    unsigned char q_lab_mix_n8[N_LAYER_ROLES][64]={{0}};
    if (q_lab && !strcmp(q_lab,"mixed")) {
        const char *spec=getenv("BEGLIN_Q_LAB_N8_ROLES_FILE");
        if (!spec || !*spec || g_cfg.nl!=28) {
            fprintf(stderr,"FATAL: mixed research policy requires explicitly supplied 28-layer mask\n");
            exit(1);
        }
        FILE *fp=fopen(spec,"r");
        if (!fp) { fprintf(stderr,"FATAL: missing mixed policy mask\n");exit(1); }
        const char *names[7]={{"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}};
        const int ids[7]={{ROLE_ATTN_Q,ROLE_ATTN_K,ROLE_ATTN_V,ROLE_ATTN_O,ROLE_MLP_GATE,ROLE_MLP_UP,ROLE_MLP_DOWN}};
        int count[7]={{0}};
        char role[40];int layer;
        while (fscanf(fp,"%39s %d",role,&layer)==2) {
            int j=-1;
            for(int i=0;i<7;i++)if(!strcmp(role,names[i])){j=i;break;}
            if (j<0 || layer<0 || layer>=g_cfg.nl || count[j]>=4 || q_lab_mix_n8[ids[j]][layer]) {
                fprintf(stderr,"FATAL: duplicate/invalid role-layer n8 promotion\n");
                fclose(fp);exit(1);
            }
            q_lab_mix_n8[ids[j]][layer]=1;
            count[j]++;
        }
        if (!feof(fp)) {fprintf(stderr,"FATAL: malformed mixed policy row\n");fclose(fp);exit(1);}
        fclose(fp);
        for(int i=0;i<7;i++) {
            if(count[i]!=4) {fprintf(stderr,"FATAL: expected 4 n8 projections per each role\n");exit(1);}
        }
        fprintf(stderr,"[Q-NATIVE-RESEARCH] validated mixed 28/196 n8 tensor role-layer mask\n");
    }
"""
# Translate the Python template into actual C source (not Python C escaping).
MIXED_POLICY_CODE=MIXED_POLICY_CODE.replace("{{","{").replace("}}","}")

def mix_source()->str:
    raw=SOURCE.read_bytes()
    if digest(raw)!=ORIGINAL_SHA:raise RuntimeError("source BEGLIN changed; abort")
    source=check_source_and_patch()
    pieces=[
      ('strcmp(q_lab,"q4"))','strcmp(q_lab,"q4") && strcmp(q_lab,"mixed"))'),
      ('must be f32, n8, or q4','must be f32, n8, q4, or mixed'),
      ('    int n_lab_n8=0;',MIXED_POLICY_CODE+"\n    int n_lab_n8=0;"),
      ('else if (q_lab && !strcmp(q_lab,"n8")) {','else if (q_lab && (!strcmp(q_lab,"n8") || (!strcmp(q_lab,"mixed") && q_lab_mix_n8[r][l]))) {'),
    ]
    for old,new in pieces:
        if source.count(old)!=1:
            raise RuntimeError("unexpected lab code patch site: "+old)
        source=source.replace(old,new,1)
    if source.count("q_lab_mix_n8")!=3:
        raise RuntimeError("mixed mask not correctly represented")
    return source

def write_private(path:Path,data:bytes):
    if path.exists():raise FileExistsError("do not overwrite "+str(path))
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)

def prepare():
    if OUT.exists():raise FileExistsError("mixed experiment already exists")
    src=mix_source()
    scan=SCAN/"q_196_proxy_results_private.json"
    if not scan.is_file() or digest(scan.read_bytes())!=SCAN_SHA or scan.stat().st_mode&0o777!=0o600:
        raise RuntimeError("pinned source-only distortion evidence missing/drift")
    evidence=json.loads(scan.read_text())
    if evidence["source_tensors"]!=196 or evidence["n8_per_role"]!=4:
        raise RuntimeError("wrong tensor budget")
    os.umask(0o077)
    OUT.mkdir(parents=True,mode=0o700)
    file=OUT/"qwen_infer_q_mixed_lab.c"
    write_private(file,src.encode())
    for mode in PREP_MODELS:
        p=SCAN/f"n8_{mode}_role_layer_policy.txt"
        if not p.is_file():raise RuntimeError("selection policy absent")
        rows=[tuple(z.split()) for z in p.read_text().splitlines()]
        if len(rows)!=28 or len(set(rows))!=28:raise RuntimeError("invalid policy size")
        for role in ROLES:
            if sum(row[0]==role for row in rows)!=4:
                raise RuntimeError("role quotas not satisfied")
        dest=OUT/f"policy_{mode}.txt"
        write_private(dest,p.read_bytes())
        for rootname in (f"mixed_{mode}",):
            folder=OUT/rootname
            folder.mkdir(mode=0o700)
            for sub in ("results","eval","eval/ppl_work"):
                (folder/sub).mkdir(exist_ok=True,mode=0o700)
            srcinput=OLD/"q4"
            for rel in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
                xx=srcinput/rel
                if not xx.is_file():raise RuntimeError("existing public synthetic prompt missing")
                write_private(folder/rel,xx.read_bytes())
    deps=[x for x in (ROOT/".build").glob("*.o") if x.name!="qwen_infer.o"]
    if len(deps)!=21:raise RuntimeError("missing original BEGLIN build objects")
    obj=OUT/"mixed_lab.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(file),"-o",str(obj)],check=True)
    binary=OUT/"beglin_q_mixed_lab"
    subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in sorted(deps)]+
                   ["-o",str(binary),"-framework","Accelerate","-lpthread"],check=True)
    binary.chmod(0o700)
    report={
      "schema":"beglin-Q-native-per-role-mixed-lab/1","status":"COMPILED_BEFORE_MIXED_EVAL",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "pinned_official_revision":OFFICIAL_REV,"MLX_cache_BF16_rev":MODEL_REV,
      "source_BEGLIN_original_sha256":ORIGINAL_SHA,
      "mixed_source_sha256":digest(file.read_bytes()),"binary_sha256":digest(binary.read_bytes()),
      "source_proxy_evidence_sha256":SCAN_SHA,
      "role_budgets":{role:4 for role in ROLES},
      "policy_sha256":{s:digest((OUT/f"policy_{s}.txt").read_bytes()) for s in PREP_MODELS},
      "runtime":"true BEGLIN 28-layer CPU forward, 196 projection tensors; fixed tied FP32 lm_head in both conditions",
      "condition":"mixed role-and-layer n8 mask applied only at safetensors source load, no posthoc target tuning",
      "original_test_split_used":False,"main_changed":False,"production_changed":False
    }
    write_private(OUT/"mixed_lab_source_receipt.json",(json.dumps(report,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_Q_MIXED_PREPARED",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    prepare()
