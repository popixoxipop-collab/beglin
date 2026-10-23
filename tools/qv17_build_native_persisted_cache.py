#!/usr/bin/env python3
"""BEGLIN Qv17: isolated, v16-pinned original C with disk-persistent 40 alternate
quantized tensor variants. Native checks exact SHA256 and internal layout before
loading, while Python bootstrap must also verify detached Ed25519-signed manifest.
No signed-asset trust claim for direct C bypass of the Python bootstrap.
"""
import pathlib,hashlib,os,subprocess,json,difflib
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work")
Q16=BASE/"beglin-qv16-cost-aware-prewarm-20261023"
PARENT=Q16/"derived_startup_prewarm_C_v1/qv16_from_original_Qv15_paired.c"
PIN="b05c2e8aac488f9accba44c61073a72f2103be6284b37aa5de0a500185995a9f"
Q17=BASE/"beglin-qv17-signed-packed-cache-20261023"
BUILD=Q17/"native_persisted_C_v2"
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for part in iter(lambda:f.read(262144),b""):h.update(part)
 return h.hexdigest()
def save(p,content):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(content);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
NATIVE= r'''
/* QV17 test-only raw packed/scales snapshot.
 * Native format is little-endian Apple arm64 specific, not portable.
 * Ed25519 signing is verified by the Python launcher with a pinned public key;
 * this native code enforces SHA256 + layout bounds to prevent loader bypass of
 * basic integrity or arbitrary role tensor substitutions.
 */
typedef struct __attribute__((packed)) {
    char magic[8]; uint32_t version; uint32_t count; uint64_t payload_bytes;
} Qv17Header;
typedef struct __attribute__((packed)) {
    uint16_t rid,layer,bits,reserved;
    uint32_t out,in; uint64_t packed_bytes,scales_bytes;
} Qv17Entry;
_Static_assert(sizeof(Qv17Header)==24,"Qv17Header layout");
_Static_assert(sizeof(Qv17Entry)==32,"Qv17Entry layout");
static void qv17_die(const char *s){
    fprintf(stderr,"FATAL: Qv17 packed cache %s\n",s);exit(1);
}
static size_t qv17_packed_size(const WT *w,int bits){
    return bits==8?qng64_packed_bytes(w->out,w->in,8):(size_t)w->out*(size_t)w->in/2;
}
static void qv17_fp_hash(FILE *f,char out[65]){
    unsigned char buff[131072];unsigned char dig[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX ctx;
    if(fseek(f,0,SEEK_SET)!=0||CC_SHA256_Init(&ctx)!=1)qv17_die("SHA init");
    for(;;){
        size_t n=fread(buff,1,sizeof(buff),f);
        if(n && CC_SHA256_Update(&ctx,buff,(CC_LONG)n)!=1)qv17_die("SHA update");
        if(n<sizeof(buff)){
            if(ferror(f))qv17_die("SHA file read");
            break;
        }
    }
    if(CC_SHA256_Final(dig,&ctx)!=1)qv17_die("SHA final");
    static const char alphabet[]="0123456789abcdef";
    for(int j=0;j<CC_SHA256_DIGEST_LENGTH;j++){
        out[j*2]=alphabet[dig[j]>>4];out[j*2+1]=alphabet[dig[j]&15];
    }
    out[64]=0;
    if(fseek(f,0,SEEK_SET)!=0)qv17_die("SHA rewind");
}
static void qv17_fwrite(const void *p,size_t n,FILE *f){
    if(n && fwrite(p,1,n,f)!=n)qv17_die("artifact short write");
}
static void qv17_fread(void *p,size_t n,FILE *f){
    if(n && fread(p,1,n,f)!=n)qv17_die("artifact short read");
}
static int qv17_valid_slot(int rid,int layer,const Qv15CacheSlot *slot,
                           const WT *wt_active){
    int different=qv14_mask_source[rid][layer]!=qv14_mask_random[rid][layer];
    if(!different)return 0;
    int want=qv14_mask_random[rid][layer]?8:4;
    if(!slot->packed || !slot->scales || slot->bits!=want ||
       slot->bytes!=qv15_variant_bytes(wt_active,want) ||
       wt_active->bits!=(qv14_mask_source[rid][layer]?8:4))
        qv17_die("not an exact cached alternate of Qv8 source mask");
    return 1;
}
static void qv17_pack_write(const char *path){
    if(qv15_max_bytes!=(size_t)96*1048576||qv16_precomputed_tensors!=40||qv16_prewarm_enabled!=1||
       qv15_misses!=0||qv15_hits!=0||
       qv15_drops!=0||qv15_cache_bytes==0)qv17_die("writer needs source-restored 40-cache prewarm");
    double start=qv14_clock_ms();
    int count=0;size_t payload=0;
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        int role=qv14_roles_id[r];Qv15CacheSlot *slot=&qv15_slots[role][l];
        char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
        WT *w=wt(name);
        if(qv17_valid_slot(role,l,slot,w)){
            count++;
            if(payload>qv15_max_bytes-slot->bytes)qv17_die("pack payload exceeds cap");
            payload+=slot->bytes;
        }
    }
    if(count!=40||payload!=qv15_cache_bytes)qv17_die("writer count or payload");
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0)qv17_die("cannot exclusively create owner-only packed file");
    FILE *f=fdopen(fd,"wb");if(!f)qv17_die("fdopen write");
    Qv17Header h;memcpy(h.magic,"BEGQ17A1",8);
    h.version=1;h.count=40;h.payload_bytes=(uint64_t)payload;
    qv17_fwrite(&h,sizeof h,f);
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        int role=qv14_roles_id[r];Qv15CacheSlot *slot=&qv15_slots[role][l];
        char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
        WT *w=wt(name);
        if(!qv17_valid_slot(role,l,slot,w))continue;
        Qv17Entry e={0};e.rid=(uint16_t)role;e.layer=(uint16_t)l;
        e.bits=(uint16_t)slot->bits;e.out=(uint32_t)w->out;e.in=(uint32_t)w->in;
        e.packed_bytes=(uint64_t)qv17_packed_size(w,slot->bits);
        e.scales_bytes=(uint64_t)w->out*(uint64_t)w->ng*4;
        if(e.packed_bytes+e.scales_bytes!=slot->bytes)qv17_die("writer variant byte mismatch");
        qv17_fwrite(&e,sizeof e,f);
        qv17_fwrite(slot->packed,(size_t)e.packed_bytes,f);
        qv17_fwrite(slot->scales,(size_t)e.scales_bytes,f);
    }
    if(fflush(f)!=0||fsync(fd)!=0)qv17_die("writer fsync");
    if(fclose(f)!=0)qv17_die("writer close");
    double elapsed=qv14_clock_ms()-start;
    fprintf(stderr,"[QV17-PACK-WRITE] entries %d payload_bytes %zu total_file_bytes %zu write_ms %.5f\n",
       count,payload,sizeof(Qv17Header)+(size_t)count*sizeof(Qv17Entry)+payload,elapsed);
}
static void qv17_pack_load(const char *path,const char *digest_expected){
    if(qv15_max_bytes!=(size_t)96*1048576||qv15_cache_bytes||
       qv15_hits||qv15_misses||qv15_drops)qv17_die("loader must start clean at Qv8 source");
    if(!digest_expected||strlen(digest_expected)!=64)qv17_die("trusted expected SHA256 is required");
    for(int k=0;k<64;k++)if(!((digest_expected[k]>='0'&&digest_expected[k]<='9')||
             (digest_expected[k]>='a'&&digest_expected[k]<='f')))qv17_die("bad expected lowercase hash");
    double start=qv14_clock_ms();
    int fd=open(path,O_RDONLY|O_NOFOLLOW);
    if(fd<0)qv17_die("cannot open nofollow packed artifact");
    struct stat st;
    if(fstat(fd,&st)!=0||!S_ISREG(st.st_mode)||(st.st_mode&077)!=0||st.st_size<sizeof(Qv17Header))
        qv17_die("not owner-only regular packed artifact");
    FILE *f=fdopen(fd,"rb");if(!f)qv17_die("fdopen read");
    char actual[65];qv17_fp_hash(f,actual);
    if(strcmp(actual,digest_expected))qv17_die("pinned packed SHA256 mismatch, do not serve");
    double sha_end=qv14_clock_ms();
    Qv17Header h;qv17_fread(&h,sizeof h,f);
    if(memcmp(h.magic,"BEGQ17A1",8)||h.version!=1||h.count!=40||
       h.payload_bytes>qv15_max_bytes||(uint64_t)st.st_size!=
        sizeof(Qv17Header)+(uint64_t)h.count*sizeof(Qv17Entry)+h.payload_bytes)
         qv17_die("packed header size/schema/cap mismatch");
    size_t bytes=0;int count=0;
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        int role=qv14_roles_id[r];
        if(qv14_mask_source[role][l]==qv14_mask_random[role][l])continue;
        Qv17Entry e;qv17_fread(&e,sizeof e,f);
        char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
        WT *w=wt(name);
        int target=qv14_mask_random[role][l]?8:4;
        if(e.reserved||e.rid!=role||e.layer!=l||e.bits!=target||
          (uint32_t)w->out!=e.out||(uint32_t)w->in!=e.in||
          e.packed_bytes!=qv17_packed_size(w,target)||
          e.scales_bytes!=(uint64_t)w->out*(uint64_t)w->ng*4||
          e.packed_bytes+e.scales_bytes!=qv15_variant_bytes(w,target)||
          e.packed_bytes>SIZE_MAX||e.scales_bytes>SIZE_MAX||
          e.packed_bytes+e.scales_bytes>qv15_max_bytes-bytes)
              qv17_die("wrong tensor index/precision/shape/cap in packed entry");
        Qv15CacheSlot *slot=&qv15_slots[role][l];
        if(slot->packed||slot->scales)qv17_die("duplicate cache entry");
        slot->packed=malloc((size_t)e.packed_bytes);
        slot->scales=malloc((size_t)e.scales_bytes);
        if(!slot->packed||!slot->scales)qv17_die("loader allocation");
        qv17_fread(slot->packed,(size_t)e.packed_bytes,f);
        qv17_fread(slot->scales,(size_t)e.scales_bytes,f);
        slot->bits=target;slot->bytes=(size_t)(e.packed_bytes+e.scales_bytes);
        bytes+=slot->bytes;count++;
    }
    if(fgetc(f)!=EOF||ferror(f)||fclose(f)!=0||count!=40||bytes!=h.payload_bytes)
        qv17_die("trailing bytes/short file/entry count");
    qv15_cache_bytes=bytes;qv15_peak_cache_bytes=bytes;
    qv14_check_state(qv14_mask_source);
    double elapsed=qv14_clock_ms()-start;
    fprintf(stderr,"[QV17-PACK-LOAD] entries %d bytes %zu sha256_check_ms %.5f total_read_ms %.5f digest %s\n",
      count,bytes,sha_end-start,elapsed,actual);
}
static void qv17_prepare_persistent_cache(void){
    const char *read=getenv("BEGLIN_QV17_PACK_LOAD");
    const char *write=getenv("BEGLIN_QV17_PACK_WRITE");
    if(read&&write)qv17_die("cannot write and read same run");
    if(read){
        if(getenv("BEGLIN_QV16_PREWARM")&&strcmp(getenv("BEGLIN_QV16_PREWARM"),"0"))
            qv17_die("persisted read must not run expensive prewarm");
        qv17_pack_load(read,getenv("BEGLIN_QV17_PACK_SHA256"));
        return;
    }
    qv16_precompute_startup();
    if(write){
        if(!getenv("BEGLIN_QV16_PREWARM")||
           strcmp(getenv("BEGLIN_QV16_PREWARM"),"1"))
              qv17_die("create requires original BF16 startup warmup");
        qv17_pack_write(write);
    }
}
'''
def main():
 if BUILD.exists():raise FileExistsError("never overwrite an already built Qv17 derived C binary")
 if sha(PARENT)!=PIN:raise RuntimeError("Qv16 proven native C source identity mismatch")
 old=PARENT.read_text()
 hook='static void qv14_prepare_profiles(void){'
 if old.count(hook)!=1:raise RuntimeError("original start hook ambiguous")
 new=old.replace(hook,NATIVE+"\n"+hook,1)
 before='    qv15_cache_init();\n    qv16_precompute_startup();\n    qv14_profile=0;'
 after='    qv15_cache_init();\n    qv17_prepare_persistent_cache();\n    qv14_profile=0;'
 if new.count(before)!=1:raise RuntimeError("original Qv16 warmup hook ambiguous")
 new=new.replace(before,after,1)
 inc='#include <Accelerate/Accelerate.h>'
 if new.count(inc)!=1:raise RuntimeError("original include anchor ambiguous")
 new=new.replace(inc,inc+'\n#include <CommonCrypto/CommonDigest.h>',1)
 score_start='            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);'
 score_end='        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]'
 a=old[old.index(score_start):old.index(score_end,old.index(score_start))]
 b=new[new.index(score_start):new.index(score_end,new.index(score_start))]
 if a!=b:raise RuntimeError("the original C logits & full vocabulary scorer has changed")
 os.umask(0o077);BUILD.mkdir(parents=True,mode=0o700)
 cp=BUILD/"qv17_original_Qwen28_persisted_cache.c"
 save(cp,new.encode())
 diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile="Qv16-pinned",tofile="Qv17-persisted",n=2))
 save(BUILD/"qv16_to_qv17.diff",("\n".join(diff)+"\n").encode())
 objs=[QW/"lab_local/qng64_n8_repacked_full_20261022"/x for x in
       ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 others=sorted(p for p in (QW/".build").glob("*.o") if p.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(others)!=20 or any(not p.is_file() for p in objs):raise RuntimeError("original kernel link object closure changed")
 buildobj=BUILD/"qv17_original.o"
 compile=subprocess.run(["clang","-O3","-w","-I",str(QW),"-c",str(cp),"-o",str(buildobj)],
                        capture_output=True,text=True,timeout=180)
 save(BUILD/"compile_log.txt",(compile.stdout+"\n"+compile.stderr).encode())
 if compile.returncode:raise RuntimeError("Qv17 original C native compile failed "+compile.stderr[-2800:])
 exe=BUILD/"beglin_qv17_signed_packed_cache"
 link=subprocess.run(["clang","-O3",str(buildobj)]+[str(p) for p in objs+others]+
                     ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],
                     capture_output=True,text=True,timeout=180)
 save(BUILD/"link_log.txt",(link.stdout+"\n"+link.stderr).encode())
 if link.returncode:raise RuntimeError("Qv17 native C linker failed "+link.stderr[-2800:])
 exe.chmod(0o700)
 receipt={
  "schema":"BEGLIN-Qv17-original-Qwen28-C-native-serialized-precision-cache/1",
  "status":"EXPERIMENTAL_NATIVE_BINARY_BUILT__SIGNED_RELOAD_NOT_YET_RUN",
  "pinned_Qv16_real_C_source_sha256":PIN,"derived_Qv17_C_SHA256":sha(cp),
  "derived_Qv17_executable_sha256":sha(exe),
  "source_diff_SHA256":sha(BUILD/"qv16_to_qv17.diff"),
  "original_Qv16_forward_and_gold_NLL_scorer_code_SHA256":
    hashlib.sha256(a.encode()).hexdigest(),
  "original_23_kernel_objects":[{"file":str(p),"sha256":sha(p)} for p in objs+others],
  "format_version":1,"format_magic":"BEGQ17A1",
  "Ed25519_signature_verification_location":"trusted Python bootstrap; C independently verifies expected artifact SHA256 and exact tensor role/shape/type/bytes",
  "security_note":"local development signing key does not prove external trusted release provenance",
  "additional_inactive_cache_hard_cap_MiB":96,
  "model_source_untouched":True,"Metal_GPU_unverified":True,"no_operator_release_change":True}
 save(BUILD/"BUILD_RECEIPT.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
 print("QV17_ORIGINAL_C_PERSISTENT_CACHE_COMPILED",json.dumps({"source_sha256":sha(cp),"native_binary_sha256":sha(exe),
 "source_original_Qv16_sha256":PIN,"kernel_objects":len(objs+others),
 "math_scorer_SHA256":receipt["original_Qv16_forward_and_gold_NLL_scorer_code_SHA256"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
