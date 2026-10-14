#!/usr/bin/env python3
"""Scratch-only one-shot failure AFTER first qNg64 GPU bind, before 2nd.

Exercises existing exact REBIND_SET_FAILED rollback + following recovery ACK;
opted-in only under the native P9 fault-test worker. Never production.
"""
import argparse,hashlib,pathlib,json
P5_C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
START="static int moe_gpu_rebind_set_apply_quiescent(const MoeGpuTxnCommand *cmd) {"
END="\nstatic int moe_gpu_demotion_pending(void) {"
DECL_ANCHOR="    int bound_count = 0;\n"
ANCHOR="        bound_count++;\n    }\n\n    if (!mlx_gpu_reset_runtime_epoch()) {"
DECL="""    /* P9_FAULT_DECL_BEGIN */
    static int p9_fault_once = 0; /* confined to run-owned worker */
    /* P9_FAULT_DECL_END */
"""
INJECT="""        /* P9_FAULT_BEGIN */
        if (i == 0 && !p9_fault_once) {
            const char *f = getenv("BEGLIN_HCP_P9_FAULT_AFTER_FIRST_BIND_ONCE");
            if (f && strcmp(f,"1")==0) {
                p9_fault_once=1;
                fprintf(stderr, "P9_FAULT_AFTER_FIRST_BIND_ONCE txn=%s\\n",cmd->txn_id);
                goto rollback; /* original snapshot restore and terminal failure ACK */
            }
        }
        /* P9_FAULT_END */
"""
def digest(b):return hashlib.sha256(b).hexdigest()
def patch(raw:bytes)->bytes:
    if digest(raw)!=P5_C128_SHA:raise ValueError("wrong sealed P5 CB128 C source")
    text=raw.decode()
    if text.count(START)!=1 or text.count(END)!=1:raise ValueError("ambiguous txn source")
    start=text.index(START);end=text.index(END,start)
    body=text[start:end]
    if body.count(DECL_ANCHOR)!=1 or body.count(ANCHOR)!=1 or "goto rollback;" not in body:
        raise ValueError("missing exact txn rollback anchors")
    body=body.replace(DECL_ANCHOR,DECL_ANCHOR+DECL,1)
    body=body.replace(ANCHOR,ANCHOR.replace("        bound_count++;\n","        bound_count++;\n"+INJECT,1),1)
    out=(text[:start]+body+text[end:]).encode()
    if strip(out)!=raw:raise ValueError("fault injection changed additional source bytes")
    return out
def strip(raw:bytes)->bytes:
    s=raw.decode()
    for k in (DECL,INJECT):
        if s.count(k)!=1:raise ValueError("cannot reverse P9 fault source")
        s=s.replace(k,"",1)
    return s.encode()
def run(src,dest):
    src,dest=pathlib.Path(src),pathlib.Path(dest)
    if not src.is_file() or src.is_symlink() or dest.exists():raise ValueError("invalid owned scratch paths")
    r=patch(src.read_bytes())
    with dest.open("xb") as f:f.write(r)
    print(json.dumps({"C128_sha":P5_C128_SHA,"fault_sha":digest(r),
                      "exact_reverse":True,"production_touched":False,
                      "fault_is_optin_one_shot":True},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args();run(a.source,a.output)
