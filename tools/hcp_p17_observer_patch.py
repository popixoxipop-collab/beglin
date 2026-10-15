#!/usr/bin/env python3
"""P17 reversible exact-sha adaptation of proven P16 first-logit observer.

Use ORIGINAL P5 C CB128 to compile an opt-in 31-frozen-prefix allowlist for
teacher-forced gold logit NLL. The P16 existing C observer's math, final
prefill position, expected output geometry and ACK remain unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
import hcp_p16_native_logit_patch as base

FROZEN_SHA="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
C128_SHA=base.C128_SHA
OLD_DIR=base.CAPTURE_DIR
DIR="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015/capture"
NEW_LOG="P17_NATIVE_GOLD_PREFIX_LOGITS"
OLD_LOG="P16_NATIVE_FULL_LOGITS"
P16_MARK="BEGLIN_P16_LOGITS_DIR"
P17_MARK="BEGLIN_P17_LOGITS_DIR"
LEN_CHECK_OLD="(rq_plen[r] != 7 && rq_plen[r] != 9)"
LEN_CHECK_NEW="(rq_plen[r] < 20 || rq_plen[r] > 112)"
FUN_LEN_OLD="(plen!=7 && plen!=9)"
FUN_LEN_NEW="(plen<20 || plen>112)"
OLD_ALLOWED=f'''if (strcmp(tok_hash,"{base.PROMPT0_SHA}")!=0 &&
        strcmp(tok_hash,"{base.PROMPT1_SHA}")!=0)'''
NEW_ALLOWED="if (!p17_prefix_allowed(tok_hash))"
FUNCTION_HOOK="static void p16_write_native_logits("
HOOK_END="/* P16_CAPTURE_HELPER_END */"
def sha(b):return hashlib.sha256(b).hexdigest()

def allowlist(frozen):
    path=pathlib.Path(frozen)
    if path.is_symlink() or not path.is_file() or sha(path.read_bytes())!=FROZEN_SHA:
        raise ValueError("P17 freeze input SHA not preregistered")
    j=json.loads(path.read_text())
    if j.get("question_count")!=12 or j.get("gold_token_prefix_count")!=31:
        raise ValueError("P17 source corpus unexpected")
    hashes=[x["prefix_token_sha256"] for row in j["items"] for x in row["forced_prefixes"]]
    if len(hashes)!=31 or len(set(hashes))!=31:
        raise ValueError("P17 duplicated or missing source prefix SHA")
    if any(len(x)!=64 or any(ch not in "0123456789abcdef" for ch in x) for x in hashes):
        raise ValueError("P17 invalid source SHA string")
    lines=",\n".join('        "'+x+'"' for x in hashes)
    return ("""/* P17_FROZEN_ALLOWLIST_BEGIN */
static int p17_prefix_allowed(const char *hex) {
    static const char *const allowed[] = {
"""+lines+"""
    };
    if (!hex) return 0;
    for (size_t i=0;i<sizeof(allowed)/sizeof(allowed[0]);i++)
        if (strcmp(hex,allowed[i])==0) return 1;
    return 0;
}
/* P17_FROZEN_ALLOWLIST_END */
""")

def patch(original,frozen):
    if sha(original)!=C128_SHA:raise ValueError("P17 accepts only original P5 CB128")
    allowed=allowlist(frozen)
    p16=base.patch(original).decode("utf-8")
    swaps=[
        (OLD_DIR,DIR,1),
        (P16_MARK,P17_MARK,2),
        (LEN_CHECK_OLD,LEN_CHECK_NEW,1),
        (FUN_LEN_OLD,FUN_LEN_NEW,1),
        (OLD_ALLOWED,NEW_ALLOWED,1),
        (OLD_LOG,NEW_LOG,1)]
    s=p16
    for old,new,expect in swaps:
        if s.count(old)!=expect:
            raise ValueError(f"P17 canonical observer anchor changed: {old[:30]} matches {s.count(old)}")
        s=s.replace(old,new)
    if s.count(FUNCTION_HOOK)!=1:raise ValueError("P17 hook ambiguous")
    s=s.replace(FUNCTION_HOOK,allowed+FUNCTION_HOOK,1)
    result=s.encode()
    if strip(result,frozen)!=original:
        raise ValueError("P17 native source is not byte-for-byte reversible")
    return result

def strip(raw,frozen):
    allowed=allowlist(frozen)
    s=raw.decode()
    old_p16=base.patch(base.strip(base.patch(b'fake'))) if False else None
    if s.count(allowed)!=1:raise ValueError("P17 allowlist source modified")
    s=s.replace(allowed,"",1)
    for old,new,expect in reversed([
        (OLD_DIR,DIR,1),
        (P16_MARK,P17_MARK,2),
        (LEN_CHECK_OLD,LEN_CHECK_NEW,1),
        (FUN_LEN_OLD,FUN_LEN_NEW,1),
        (OLD_ALLOWED,NEW_ALLOWED,1),
        (OLD_LOG,NEW_LOG,1)]):
        if s.count(new)!=expect:raise ValueError("P17 source rollback anchor missing")
        s=s.replace(new,old)
    return base.strip(s.encode())

def run(source,frozen,out):
    a,b=pathlib.Path(source),pathlib.Path(out)
    if a.is_symlink() or not a.is_file() or b.exists():raise ValueError("P17 requires regular C source and fresh output")
    original=a.read_bytes();result=patch(original,frozen)
    with b.open("xb") as f:f.write(result)
    print("P17_NATIVE_FIRST_PREFILL_31_HASH_WHITELIST_READY",json.dumps({
      "source_SHA256":sha(original),"frozen_SHA256":FROZEN_SHA,
      "instrumented_SHA256":sha(result),"reversible":strip(result,frozen)==original,
      "captured_vocab":102400,"original_math_unchanged":True,
      "production_touched":False},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("source","frozen","output"):p.add_argument("--"+name,required=True)
    a=p.parse_args();run(a.source,a.frozen,a.output)
