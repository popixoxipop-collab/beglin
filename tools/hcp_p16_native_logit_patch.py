#!/usr/bin/env python3
"""P16 exact-source reversible first-new-token native full-logits observer.

Opt-in BEGLIN_P16_LOGITS_DIR ONLY in the XOX run-owned scratch capture dir.
It writes native float32 102400-element row at LAST PROMPT PREFILL position,
BEFORE greedy argmax is emitted, only pass=1 / one request / one new token.
Never alters original model math, quantization, policy ACK or output tokens.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib

C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
MODEL_VOCAB=102400
CAPTURE_DIR="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p16-native-logits-20261015/capture"
PROMPT0_SHA="f3b3029dbb5ecfe92bba4fd4922b6b4557b90aff65eaadc3e9bd34c485866ff4"
PROMPT1_SHA="ee9f8b6e784080ed0150ae0979b46051675e77234bc6661ef5b13e7bd5566238"

HEADER='#include "gguf_transcode.h"'
INCLUDES='\n/* P16_CAPTURE_INCLUDES_BEGIN */\n#include <CommonCrypto/CommonDigest.h>\n#include <fcntl.h>\n#include <unistd.h>\n/* P16_CAPTURE_INCLUDES_END */'
START="static int run_moe_gpu_cbatch_online_gate(int argc, char **argv) {"
END="\n// V5i Phase B: B=1 GQA GPU correctness gate."
COUNT_ANCHOR="        const int MCN = mf_n;"
COUNT_PATCH=COUNT_ANCHOR+"\n        int p16_capture_count = 0;"
ANCHOR="""                int am = moe_gpu_argmax_finite(lg, MOE_VOCAB, "moe gpu cb online", step, r,
                                      &competing_token, &margin);
                if (pass == 1) {
                    moe_gpu_neartie_maybe_log(
                        s, r, spos_arr[m], tok_arr[m], am, competing_token, margin, lg, A);
                }
                rq_out[r][rq_nout[r]++] = am; rq_t_first[r] = temit;"""
NEW="""                int am = moe_gpu_argmax_finite(lg, MOE_VOCAB, "moe gpu cb online", step, r,
                                      &competing_token, &margin);
                if (pass == 1 && persistent_mode &&
                    getenv("BEGLIN_P16_LOGITS_DIR")) {
                    if (p16_capture_count != 0 || R != 1 || r != 0 ||
                        rq_maxnew[r] != 1 || rq_nout[r] != 0 ||
                        (rq_plen[r] != 7 && rq_plen[r] != 9) ||
                        spos_arr[m] != rq_plen[r] - 1) {
                        fprintf(stderr,"P16_INVALID_NATIVE_FIRST_LOGIT_CAPTURE_CONTEXT\\n");
                        fflush(stderr);
                        _exit(117);
                    }
                    p16_write_native_logits(getenv("BEGLIN_P16_LOGITS_DIR"),
                        persistent_request_id,mf_ids[r%MCN],rq_plen[r],lg,am);
                    p16_capture_count++;
                }
                if (pass == 1) {
                    moe_gpu_neartie_maybe_log(
                        s, r, spos_arr[m], tok_arr[m], am, competing_token, margin, lg, A);
                }
                rq_out[r][rq_nout[r]++] = am; rq_t_first[r] = temit;"""
# Raw string literal keeps \n intact inside original C fprintf.
NEW=NEW.replace('CONTEXT\\\\n','CONTEXT\\n')
# A single backslash-n *text sequence* in C, never an actual source newline.
HELPER=rf"""/* P16_CAPTURE_HELPER_BEGIN */
static void p16_hex_hash(const uint8_t *data, size_t n, char out[65]) {{
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (!data || !n || n>10000000 || !CC_SHA256(data,(CC_LONG)n,digest)) _exit(117);
    for(int i=0;i<CC_SHA256_DIGEST_LENGTH;i++)
        snprintf(out+(size_t)(i*2),3,"%02x",(unsigned)digest[i]);
    out[64]='\0';
}}
static void p16_write_native_logits(const char *dir,const char *rid,
    const int *tokens,int plen,const float *logits,int generated) {{
    static const char allowed_dir[]="{CAPTURE_DIR}";
    if (!dir || strcmp(dir,allowed_dir)!=0 || !rid || strlen(rid)!=32 ||
        !tokens || !logits || (plen!=7 && plen!=9) ||
        MOE_VOCAB!=102400 || sizeof(float)!=4 || generated<0 || generated>=MOE_VOCAB) {{
        fprintf(stderr,"P16_CAPTURE_SCOPE_INVALID\n");fflush(stderr);_exit(117);
    }}
    for(int i=0;i<32;i++) {{
        if (!((rid[i]>='0' && rid[i]<='9') || (rid[i]>='a' && rid[i]<='f'))) _exit(117);
    }}
    char text[1024];
    size_t len=0;
    for(int i=0;i<plen;i++) {{
        if(tokens[i]<0 || tokens[i]>=MOE_VOCAB) _exit(117);
        int n=snprintf(text+len,sizeof(text)-len,"%s%d",i?",":"",tokens[i]);
        if(n<=0 || (size_t)n>=sizeof(text)-len) _exit(117);
        len+=(size_t)n;
    }}
    char tok_hash[65];
    p16_hex_hash((const uint8_t *)text,len,tok_hash);
    if (strcmp(tok_hash,"{PROMPT0_SHA}")!=0 &&
        strcmp(tok_hash,"{PROMPT1_SHA}")!=0) {{
        fprintf(stderr,"P16_UNFROZEN_TOKENS\n");fflush(stderr);_exit(117);
    }}
    for(int i=0;i<MOE_VOCAB;i++)
        if (!isfinite(logits[i])) {{fprintf(stderr,"P16_LOGIT_NONFINITE\n");_exit(117);}}
    char fname[1200];
    int z=snprintf(fname,sizeof(fname),"%s/%s.f32",dir,rid);
    if(z<=0 || (size_t)z>=sizeof(fname)) _exit(117);
    int fd=open(fname,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0) {{perror("P16_EXCLUSIVE_CAPTURE_OPEN");_exit(117);}}
    const size_t size=(size_t)MOE_VOCAB*sizeof(float);
    const uint8_t *ptr=(const uint8_t *)logits;
    size_t written=0;
    while(written<size) {{
        ssize_t n=write(fd,ptr+written,size-written);
        if(n<=0) {{fprintf(stderr,"P16_CAPTURE_WRITE_FAILURE\n");_exit(117);}}
        written+=(size_t)n;
    }}
    if(fsync(fd)!=0 || close(fd)!=0) {{
        fprintf(stderr,"P16_CAPTURE_FSYNC_FAILURE\n");_exit(117);
    }}
    char log_hash[65];
    p16_hex_hash(ptr,size,log_hash);
    fprintf(stderr,
      "P16_NATIVE_FULL_LOGITS rid=%s inputsha=%s plen=%d pos=%d next=%d vocab=%d bytes=%zu floatsha=%s\n",
       rid,tok_hash,plen,plen-1,generated,MOE_VOCAB,size,log_hash);
    fflush(stderr);
}}
/* P16_CAPTURE_HELPER_END */
"""

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def patch(original:bytes)->bytes:
    if sha(original)!=C128_SHA:raise ValueError("Only sealed P5 CB128 C is accepted")
    text=original.decode()
    if (text.count(HEADER)!=1 or text.count(START)!=1 or text.count(END)!=1 or
        "P16_CAPTURE" in text):raise ValueError("P16 source anchors ambiguous")
    start=text.index(START);end=text.index(END,start)
    body=text[start:end]
    if body.count(COUNT_ANCHOR)!=1 or body.count(ANCHOR)!=1:
        raise ValueError("No uniquely identified *last prefill* greedy logits source")
    body=body.replace(COUNT_ANCHOR,COUNT_PATCH,1).replace(ANCHOR,NEW,1)
    text=text[:start]+body+text[end:]
    text=text.replace(HEADER,HEADER+INCLUDES,1)
    text=text.replace(START,HELPER+START,1)
    result=text.encode()
    if strip(result)!=original:raise ValueError("P16 annotation not exactly reversible")
    return result

def strip(patched:bytes)->bytes:
    text=patched.decode()
    for old,new in ((HEADER+INCLUDES,HEADER),(HELPER+START,START),
                    (COUNT_PATCH,COUNT_ANCHOR),(NEW,ANCHOR)):
        if text.count(old)!=1:raise ValueError("P16 unexpected capture annotation in source")
        text=text.replace(old,new,1)
    if "P16_CAPTURE" in text or "P16_NATIVE_FULL_LOGITS" in text:
        raise ValueError("P16 capture marker survives stripping")
    return text.encode()

def run(source,out):
    src,dst=pathlib.Path(source),pathlib.Path(out)
    if not src.is_file() or src.is_symlink() or dst.exists():
        raise ValueError("Expected original C source and fresh run-owned output")
    orig=src.read_bytes();instrumented=patch(orig)
    with dst.open("xb") as f:f.write(instrumented)
    print("P16_OBSERVER_SOURCE_READY",json.dumps({
      "source_sha256":sha(orig),"observer_source_sha256":sha(instrumented),
      "reversible":strip(instrumented)==orig,
      "native_capture":CAPTURE_DIR,
      "entire_CB128_and_MLX_model_math_unchanged":True,
      "production_touched":False},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.source,a.output)
