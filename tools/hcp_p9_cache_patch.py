#!/usr/bin/env python3
"""P9: exact-source opt-in, 4-entry MLX qNg64 n5/n6 packed representation cache.

Only the two P5 precision-switch tensors L3/shared_up and L26/shared_down,
and only inside new run-owned compiled research sources. Original C/ACK and
GPU-fused KV behavior are unchanged. Fail-closed on unexpected source mutation.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib

P8_MLX_SHA="b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a"
MARKER="/* P9_PREPARED_CACHE"
ENV="BEGLIN_HCP_P9_PREPARED_CACHE"

def sha(content:bytes)->str:
    return hashlib.sha256(content).hexdigest()

INCLUDES='#include <unordered_map>\n#include <vector>\n'
EXTRA_INCLUDES='/* P9_PREPARED_CACHE_INCLUDES_BEGIN */\n#include <cstdlib>\n#include <limits>\n/* P9_PREPARED_CACHE_INCLUDES_END */\n'

GLOBAL_ANCHOR="static mx::array mixed_qng64_gemv_e0(const char *name, const mx::array &x);\n"
GLOBAL=r"""/* P9_PREPARED_CACHE_GLOBAL_BEGIN */
struct P9Prepared {
    const uint8_t *source_blob;  // only immutable run-owned model lifetime
    long packed_off, scale_off, E, out, in, ng;
    int bits;
    std::vector<uint8_t> source_codes;   // exact-byte guard against stale/reused memory
    std::vector<uint8_t> source_scales;  // exact-byte guard; current binding constructs fresh scales
    mx::array packed_w;                  // MLX-owned repacked codes, shared with snapshots
};
static std::unordered_map<std::string,P9Prepared> g_p9_prepared;
static size_t g_p9_retained_bytes = 0;
static constexpr size_t P9_MAX_BYTES = 256ull * 1024ull * 1024ull;
static constexpr size_t P9_MAX_ENTRY_BYTES = 64ull * 1024ull * 1024ull;
static constexpr size_t P9_MAX_ENTRIES = 4;

static bool p9_opted_in(void) {
    const char *e=std::getenv("BEGLIN_HCP_P9_PREPARED_CACHE");
    return e && std::strcmp(e,"1")==0;
}
static bool p9_target(const char *name, int bits, long E,long out,long in,long ng,
                      long packed_off,long scale_off) {
    if (!name || (bits!=5 && bits!=6) || E!=1 || out<=0 || out>32768 ||
        in<=0 || in>32768 || in%64 || ng!=in/64 || packed_off<0 || scale_off<0)
        return false;
    return std::strcmp(name,"model.layers.3.mlp.shared_experts.up_proj")==0 ||
           std::strcmp(name,"model.layers.26.mlp.shared_experts.down_proj")==0;
}
static std::string p9_key(const char *name,int bits) {
    return std::string(name)+"#"+std::to_string(bits);
}
/* P9_PREPARED_CACHE_GLOBAL_END */
"""
HIT_ANCHOR="            long group_stride = (long)bits * 8;           // bytes for ONE 64-element group"
HIT_CODE=r"""            /* P9_PREPARED_CACHE_HIT_BEGIN */
            const bool p9_active = p9_opted_in() &&
                p9_target(name,bits,E,out,in,ng,packed_off,scale_off);
            const std::string p9k = p9_active ? p9_key(name,bits) : "";
            const size_t p9_codes_size = (size_t)E*(size_t)out*(size_t)ng*(size_t)bits*8;
            const size_t p9_scales_size = (size_t)E*(size_t)out*(size_t)ng*sizeof(float);
            bool p9_cache_hit=false;
            if (p9_active) {
                auto cached=g_p9_prepared.find(p9k);
                if (cached!=g_p9_prepared.end()) {
                    const P9Prepared &s=cached->second;
                    bool same_identity = (s.source_blob==blob &&
                        s.packed_off==packed_off && s.scale_off==scale_off &&
                        s.E==E && s.out==out && s.in==in && s.ng==ng && s.bits==bits);
                    if (!same_identity || s.source_codes.size()!=p9_codes_size ||
                        s.source_scales.size()!=p9_scales_size ||
                        std::memcmp(s.source_codes.data(),blob+packed_off,p9_codes_size)!=0 ||
                        std::memcmp(s.source_scales.data(),blob+scale_off,p9_scales_size)!=0) {
                        fprintf(stderr,"P9_PREPARED_CACHE_SOURCE_MISMATCH name=%s bits=%d\n",name,bits);
                        return 0; // fail CLOSED, old binding/snapshot remains untouched
                    }
                    w=s.packed_w; // reference-counted MLX-owned packed buffer
                    p9_cache_hit=true;
                    fprintf(stderr,"P9_PREPARED_CACHE_HIT name=%s bits=%d entries=%zu retained=%zu\n",
                            name,bits,g_p9_prepared.size(),g_p9_retained_bytes);
                }
            }
            if (!p9_cache_hit) {
            /* P9_PREPARED_CACHE_HIT_END */
"""
MISS_ANCHOR='            int bias_code = 1 << (bits - 1);'
MISS_CODE=r"""            /* P9_PREPARED_CACHE_MISS_BEGIN */
            }
            if (p9_active && !p9_cache_hit) {
                // Install a bounded, exact-source, independent owned representation.
                // Rebuilding/verification failure never removes an existing binding.
                const size_t reserved=2*p9_codes_size+p9_scales_size;
                if (p9_codes_size<=P9_MAX_ENTRY_BYTES &&
                    reserved<=P9_MAX_BYTES &&
                    g_p9_prepared.size()<P9_MAX_ENTRIES &&
                    g_p9_retained_bytes<=P9_MAX_BYTES-reserved) {
                    try {
                        std::vector<uint8_t> codes(p9_codes_size), scales_copy(p9_scales_size);
                        std::memcpy(codes.data(),blob+packed_off,p9_codes_size);
                        std::memcpy(scales_copy.data(),blob+scale_off,p9_scales_size);
                        P9Prepared entry{blob,packed_off,scale_off,E,out,in,ng,bits,
                                         std::move(codes),std::move(scales_copy),w};
                        auto inserted=g_p9_prepared.emplace(p9k,std::move(entry));
                        if (inserted.second) {
                            g_p9_retained_bytes+=reserved;
                            fprintf(stderr,"P9_PREPARED_CACHE_STORE name=%s bits=%d entries=%zu retained=%zu\n",
                                    name,bits,g_p9_prepared.size(),g_p9_retained_bytes);
                        }
                    } catch (...) {
                        fprintf(stderr,"P9_PREPARED_CACHE_STORE_SKIPPED name=%s bits=%d\n",name,bits);
                    }
                } else {
                    fprintf(stderr,"P9_PREPARED_CACHE_BUDGET_SKIP name=%s bits=%d bytes=%zu\n",
                            name,bits,reserved);
                }
            }
            /* P9_PREPARED_CACHE_MISS_END */
"""
def patch(b:bytes)->bytes:
    s=b.decode("utf-8")
    if MARKER in s:raise ValueError("P9 double patch rejected")
    if s.count(INCLUDES)!=1 or s.count(GLOBAL_ANCHOR)!=1:
        raise ValueError("P9 source unexpected global/includes")
    if s.count(HIT_ANCHOR)!=1 or s.count(MISS_ANCHOR)!=1:
        raise ValueError("P9 source unexpected qNg64 path anchors")
    if not all(k in s for k in ('int mlx_gpu_restore_binding_snapshot(uint64_t snapshot_id)',
         'int mlx_gpu_reset_runtime_epoch(void)',
         'g_tensors.insert_or_assign(', 'mx::array(repacked,', 'mlx_gpu_bind_af(')):
        raise ValueError("P9 original MLX lifecycle/rollback missing")
    s=s.replace(INCLUDES,INCLUDES+EXTRA_INCLUDES,1)
    s=s.replace(GLOBAL_ANCHOR,GLOBAL+GLOBAL_ANCHOR,1)
    s=s.replace(HIT_ANCHOR,HIT_CODE+HIT_ANCHOR,1)
    s=s.replace(MISS_ANCHOR,MISS_CODE+MISS_ANCHOR,1)
    return s.encode("utf-8")

def strip(b:bytes)->bytes:
    s=b.decode("utf-8")
    pairs=[
        (INCLUDES+EXTRA_INCLUDES,INCLUDES),
        (GLOBAL+GLOBAL_ANCHOR,GLOBAL_ANCHOR),
        (HIT_CODE+HIT_ANCHOR,HIT_ANCHOR),
        (MISS_CODE+MISS_ANCHOR,MISS_ANCHOR)
    ]
    for old,new in pairs:
        if s.count(old)!=1:raise ValueError("P9 source cannot be reversed exactly")
        s=s.replace(old,new,1)
    if MARKER in s:raise ValueError("unrecognized P9 marker")
    return s.encode("utf-8")

def execute(source:pathlib.Path, dest:pathlib.Path):
    if source.is_symlink() or not source.is_file() or dest.exists():
        raise ValueError("P9 only runs on a regular source and new output")
    raw=source.read_bytes()
    # Git blob SHA from verified P5 C++ source is authoritative.
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob!=P8_MLX_SHA:raise ValueError("P9 input source is not the P5 sealed C++ blob")
    generated=patch(raw)
    if strip(generated)!=raw:raise ValueError("P9 is not a reversible source-only patch")
    with dest.open("xb") as f:f.write(generated)
    print("P9_MLX_SOURCE_PATCH",json.dumps({"original_blob_sha":blob,
      "original_sha256":sha(raw),"patched_sha256":sha(generated),
      "opt_in_env":ENV,"same_source_after_strip":True,
      "research_only":True,"production_touched":False},sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    a=ap.parse_args();execute(a.source,a.output)
