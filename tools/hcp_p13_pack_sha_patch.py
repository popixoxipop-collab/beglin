#!/usr/bin/env python3
"""P13 reversible C-only source annotation to SHA actual native P9 packed buffers.

Research opt-in BEGLIN_P13_PACKED_SHA=1. Uses original C qNg64 cache both
initial startup-A hits and B cache stores; hashes packed codes and float32
scales without exporting weights, touching MLX math or intercepting ACK.
An unexpected source, shape or bits fails closed only when opt-in.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib

SOURCE_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
HEADER='#include "gguf_transcode.h"'
INCLUDE='\n/* P13_COMMONCRYPTO_BEGIN */\n#include <CommonCrypto/CommonDigest.h>\n/* P13_COMMONCRYPTO_END */'
START="static MoeAFTensor *moe_gpu_qng64_cached("
END="\nstatic int moe_gpu_rebind_apply_quiescent("
BEGIN="/* P13_TRACE_FUNCTION_BEGIN */"
END_MARK="/* P13_TRACE_FUNCTION_END */"
FLAG="BEGLIN_P13_PACKED_SHA"

HELPER=r"""/* P13_TRACE_FUNCTION_BEGIN */
static void p13_sha256_hex(const uint8_t *data,size_t n,char dest[65]) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    if (!data || n==0 || n>100000000u) _exit(121);
    CC_SHA256(data,(CC_LONG)n,hash);
    for (int i=0;i<CC_SHA256_DIGEST_LENGTH;i++)
        snprintf(dest+(size_t)(i*2),3,"%02x",(unsigned)hash[i]);
    dest[64]=0;
}
static void p13_native_sha(MoeAFTensor *qnt,MoeAttribRole role,int layer,int n,const char *phase) {
    const char *enabled=getenv("BEGLIN_P13_PACKED_SHA");
    if (!enabled || strcmp(enabled,"1")!=0) return;
    const int up=(role==MOE_ATTRIB_SHARED_UP && layer==3);
    const int down=(role==MOE_ATTRIB_SHARED_DOWN && layer==26);
    if (!up && !down) return;
    if (!qnt || !qnt->base || (n!=5 && n!=6) ||
        qnt->E!=1 || qnt->bits!=n ||
        (up && (qnt->out!=2816 || qnt->in!=2048)) ||
        (down && (qnt->out!=2048 || qnt->in!=2816)) ||
        qnt->ng!=qnt->in/64 ||
        qnt->packed_off<0 || qnt->scale_off<0 ||
        qnt->packed_bytes!=(long)((size_t)qnt->out*(size_t)qnt->ng*(size_t)n*8) ||
        qnt->scale_off<qnt->packed_off+qnt->packed_bytes ||
        qnt->scale_off>30000000L) {
        fprintf(stderr,"P13_NATIVE_HASH_CONTRACT_FAIL role=%d layer=%d bits=%d\n",
                (int)role,layer,n);
        fflush(stderr);
        _exit(121);
    }
    const size_t packsize=(size_t)qnt->packed_bytes;
    const size_t scalesize=(size_t)qnt->out*(size_t)qnt->ng*sizeof(float);
    char phex[65],shex[65];
    p13_sha256_hex(qnt->base+qnt->packed_off,packsize,phex);
    p13_sha256_hex(qnt->base+qnt->scale_off,scalesize,shex);
    fprintf(stderr,
      "P13_NATIVE_PACKED_SHA role=%s layer=%d bits=%d phase=%s packed=%s scales=%s packed_bytes=%zu scales_bytes=%zu\n",
       up?"shared_up":"shared_down",layer,n,phase,phex,shex,packsize,scalesize);
    fflush(stderr);
}
/* P13_TRACE_FUNCTION_END */
"""

CACHE_HIT='        return cached;'
CACHE_PATCH='        p13_native_sha(cached,role,layer,n,"hit");\n        return cached;'
CACHE_STORE='    if (qnt) moe_gpu_cache_store(role, layer, n, qnt);'
CACHE_STORE_PATCH='    if (qnt) moe_gpu_cache_store(role, layer, n, qnt);\n    if (qnt) p13_native_sha(qnt,role,layer,n,"store");'

def sha(data:bytes):return hashlib.sha256(data).hexdigest()

def patch(raw:bytes)->bytes:
    if sha(raw)!=SOURCE_SHA:raise ValueError("P13 only accepts the sealed P5 C CB128 source")
    text=raw.decode("utf-8")
    if text.count(HEADER)!=1 or text.count(START)!=1 or text.count(END)!=1 or BEGIN in text:
        raise ValueError("P13 ambiguous injection anchors")
    start=text.index(START);end=text.index(END,start)
    body=text[start:end]
    if body.count(CACHE_HIT)!=1 or body.count(CACHE_STORE)!=1:
        raise ValueError("P13 original active GPU precision cache path changed")
    body=body.replace(CACHE_HIT,CACHE_PATCH,1)
    body=body.replace(CACHE_STORE,CACHE_STORE_PATCH,1)
    text=text[:start]+body+text[end:]
    text=text.replace(HEADER,HEADER+INCLUDE,1)
    text=text.replace(START,HELPER+START,1)
    changed=text.encode()
    if strip(changed)!=raw:raise ValueError("P13 source modifications are not perfectly reversible")
    return changed

def strip(raw:bytes)->bytes:
    text=raw.decode("utf-8")
    pairs=[(HEADER+INCLUDE,HEADER),
       (HELPER+START,START),
       (CACHE_PATCH,CACHE_HIT),
       (CACHE_STORE_PATCH,CACHE_STORE)]
    for a,b in pairs:
        if text.count(a)!=1:
            raise ValueError("P13 unexpected non-reversible source")
        text=text.replace(a,b,1)
    if BEGIN in text or END_MARK in text or "P13_COMMONCRYPTO_BEGIN" in text:
        raise ValueError("P13 marker left in reverted source")
    return text.encode()

def run(source,output):
    a,b=pathlib.Path(source),pathlib.Path(output)
    if not a.is_file() or a.is_symlink() or b.exists():raise ValueError("P13 only accepts regular source and fresh output")
    orig=a.read_bytes();modified=patch(orig)
    with b.open("xb") as f:f.write(modified)
    print("P13_SOURCE_PATCHED",json.dumps({
       "sealed_original_C128_SHA256":sha(orig),
       "P13_annotated_C128_SHA256":sha(modified),
       "reversible":strip(modified)==orig,"opt_in_env":FLAG,
       "original_ack_and_MLX_functions_unchanged":True,
       "production_touched":False},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.source,a.output)
