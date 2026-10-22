#!/usr/bin/env python3
"""Research-only instrumented native C token latency trace for TinyLlama.

Both q4 and 22n8 workers share actual BEGLIN native kernels and identical
Llama original model. Instrument 3 repetitions of decode-only benchmark to
retain *all timed per-token* forward+logit milliseconds, rather than merely
the fastest trial. Each trial prints mean/p50/p95 after timestamps finish.
Original sources and live serving are never modified.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
BEG=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
QV5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
QV7=ROOT/"lab_local/qv7_tinyllama_20261022"
BASE=QV5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin"
SOURCES={"q4":BASE/"qv5_base.c",
         "fisher_22n8":QV7/"native_mixed_22/qv7_tinyllama_original_qv5_mixed_copy.c"}
OUT=QV7/"native_latency_traces_instrumented"
def sha(b):return hashlib.sha256(b).hexdigest()
def seal(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def patch(src):
    marker='int main(int argc, char **argv) {'
    if src.count(marker)!=1:raise RuntimeError("not original BEGLIN C main")
    compare="""static int qv7_token_lat_compare(const void *a,const void *b){
    double x=*(const double *)a, y=*(const double *)b;
    return x<y ? -1 : x>y ? 1 : 0;
}
"""
    src=src.replace(marker,compare+marker,1)
    anchor='    } else if (!strcmp(mode,"bench")) {'
    start=src.index(anchor)
    stop=src.index('    } else if (!strcmp(mode,',start+len(anchor))
    original=src[start:stop]
    s=original
    head='        for(int rep=0;rep<3;rep++){'
    if s.count(head)!=1:raise RuntimeError("bench trial outer loop ambiguous")
    s=s.replace(head,'        double *qv7_trace_ms=calloc((size_t)3*N,sizeof(double));\n'
                       '        if(!qv7_trace_ms){fprintf(stderr,"FATAL: no latency buffers\\n");return 1;}\n'+head,1)
    record='                g_t_layers+=b-a; g_t_logits+=c-b; }'
    if s.count(record)!=1:raise RuntimeError("bench timer anchor ambiguous")
    s=s.replace(record,'                qv7_trace_ms[rep*N+g]=(c-a)*1000.0;\n'+record,1)
    output='        double ms=best/N*1e3;'
    if s.count(output)!=1:raise RuntimeError("bench post-timing anchor ambiguous")
    extra="""        for(int qv7_rep=0;qv7_rep<3;qv7_rep++){
            double qv7_total=0;
            for(int qv7_i=0;qv7_i<N;qv7_i++) qv7_total+=qv7_trace_ms[qv7_rep*N+qv7_i];
            qsort(qv7_trace_ms+qv7_rep*N,(size_t)N,sizeof(double),qv7_token_lat_compare);
            int qv7_m=(N-1)/2, qv7_p=(int)(0.95*(N-1)+0.999999);
            if(qv7_p>=N)qv7_p=N-1;
            fprintf(stderr,"[qv7_trace] rep=%d tokens=%d mean_ms=%.9f p50_ms=%.9f p95_ms=%.9f\\n",
                qv7_rep,N,qv7_total/N,qv7_trace_ms[qv7_rep*N+qv7_m],
                qv7_trace_ms[qv7_rep*N+qv7_p]);
        }
        free(qv7_trace_ms);
"""
    s=s.replace(output,extra+output,1)
    return src[:start]+s+src[stop:]
def main():
    if OUT.exists():raise FileExistsError("native trace binaries already built")
    os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
    report={}
    for mode,source in SOURCES.items():
        if not source.is_file():raise RuntimeError("source missing "+str(source))
        raw=source.read_bytes()
        # Exact original already audited Qv5 and Qv7 C, no prior instrumentation.
        if b"qv7_trace" in raw:raise RuntimeError("native source was previously instrumented")
        copied=patch(raw.decode()).encode()
        target=OUT/f"qv7_token_trace_{mode}.c"
        seal(target,copied)
        obj=OUT/f"qv7_token_trace_{mode}.o"
        exe=OUT/f"beglin_qv7_trace_{mode}"
        subprocess.run(["clang","-O3","-w","-I",str(BEG),"-c",str(target),"-o",str(obj)],check=True,timeout=100)
        objs=sorted((BEG/".build").glob("*.o"))
        if mode=="q4":
            links=[x for x in objs if x.name!="qwen_infer.o"]
            if len(links)!=21:raise RuntimeError("reference C q4 linker inputs unavailable")
        else:
            base=BEG/"lab_local/qng64_n8_repacked_full_20261022"
            links=[base/"qng64_original_aliased.o",base/"n8_repacked_simd.o",
                   base/"n8_repacked_dispatch.o"]+[x for x in objs if x.name not in ("qwen_infer.o","qng64_dense.o")]
            if len(links)!=23:raise RuntimeError("reference C qNg64-n8 linker inputs unavailable")
        subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in links]+
           ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True,timeout=100)
        exe.chmod(0o700)
        report[mode]={"original_source_SHA256":sha(raw),"instrumented_C_sha256":sha(copied),
          "native_executable_SHA256":sha(exe.read_bytes()),"compile_exit":0}
        print("QV7_NATIVE_TRACE_RESEARCH_INSTRUMENTED_BIN",mode,sha(exe.read_bytes()),flush=True)
    receipt={"schema":"BEGLIN-Qv7-actual-native-latency-ABBA-trace-instrumented/1",
     "status":"TWO_RESEARCH_ONLY_C_NATIVE_BINARIES_NEW_PER_TOKEN_TRACE",
     "model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     "modes":report,"added_3_trials_each_per_token_forward_and_logit_trace":True,
     "original_native_sources_never_replaced":True,
     "auto_runtime_promotion":False}
    b=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
    seal(OUT/"build_receipt.json",b)
    print("QV7_TOKEN_TRACE_BUILD_PASS",sha(b),flush=True)
if __name__=="__main__":main()
