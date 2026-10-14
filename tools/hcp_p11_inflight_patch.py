#!/usr/bin/env python3
"""SHA-pinned, reversible P11 research-only crash windows in P5 CB128 C.

Never edits an existing source. Both windows are inside the 2-target atomic
REBIND_SET path; the second is after local epoch mutation but BEFORE durable ACK.
A marker is fsynced, then the isolated worker stays parked for bounded 15s.
If controller doesn't kill it, fail-closed _exit rather than serving a request.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
MARK="/* P11_EXACT_CRASH_"
HEADER="""/* P11_EXACT_CRASH_INCLUDE_BEGIN */
#include <fcntl.h>
#include <unistd.h>
/* P11_EXACT_CRASH_INCLUDE_END */
"""
FUNCTION="""/* P11_EXACT_CRASH_GATE_BEGIN */
static void p11_test_gate(const MoeGpuTxnCommand *cmd, const char *phase) {
    const char *target=getenv("BEGLIN_P11_CRASH_WINDOW");
    if (!target || strcmp(target, phase)!=0) return;
    const char *file=getenv("BEGLIN_P11_CRASH_MARKER");
    if (!cmd || !cmd->valid || !file || file[0]!='/' || strstr(file,"..") ||
        !strstr(file,"/hcp-p11-inflight-")) {
        fprintf(stderr,"P11_GUARD_REJECTED_INVALID_CRASH_MARKER\\n");
        _exit(91);
    }
    int fd=open(file,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if (fd<0) { fprintf(stderr,"P11_MARKER_CREATE_FAILED\\n"); _exit(91); }
    char out[220];
    int n=snprintf(out,sizeof out,"%s %s\\n",phase,cmd->txn_id);
    if (n<=0 || n>=(int)sizeof out || write(fd,out,(size_t)n)!=n ||
        fsync(fd)!=0 || close(fd)!=0) {
        fprintf(stderr,"P11_MARKER_DURABILITY_FAILED\\n");_exit(91);
    }
    fprintf(stderr,"P11_INFLIGHT_TEST_MARKER_PERSISTED phase=%s txn=%s\\n",phase,cmd->txn_id);
    fflush(stderr);
    sleep(15); /* caller must kill only THIS run-owned native process */
    fprintf(stderr,"P11_INFLIGHT_TEST_GATE_EXPIRED_FAIL_CLOSED\\n");
    fflush(stderr);
    _exit(92);
}
/* P11_EXACT_CRASH_GATE_END */
"""
CALL_ONE="""        /* P11_EXACT_CRASH_AFTER_FIRST_BIND_BEGIN */
        if (i==0) p11_test_gate(cmd, "first_bind");
        /* P11_EXACT_CRASH_AFTER_FIRST_BIND_END */
"""
CALL_TWO="""    /* P11_EXACT_CRASH_BEFORE_ACK_BEGIN */
    p11_test_gate(cmd, "before_ack");
    /* P11_EXACT_CRASH_BEFORE_ACK_END */
"""
START="static int moe_gpu_rebind_set_apply_quiescent(const MoeGpuTxnCommand *cmd) {"
END="\nstatic int moe_gpu_demotion_pending(void) {"
A="        bound_count++;\n"
B='    if (!moe_gpu_txn_mark_terminal(cmd, "REBIND_SET_APPLIED", cmd->target_count)) {'
def sha(b):return hashlib.sha256(b).hexdigest()
def patch(b:bytes)->bytes:
    if sha(b)!=C128_SHA:raise ValueError("only sealed P5 C128 source accepted")
    s=b.decode()
    if MARK in s or s.count(START)!=1 or s.count(END)!=1:raise ValueError("ambiguous or patched source")
    start=s.index(START);end=s.index(END,start)
    body=s[start:end]
    if body.count(A)!=1 or body.count(B)!=1:raise ValueError("crash anchors not unique")
    if not ("goto rollback;" in body and "mlx_gpu_restore_binding_snapshot" in body and
        "g_moe_gpu_weight_epoch++;" in body):
        raise ValueError("native original rollback protocol missing")
    body=body.replace(A,A+CALL_ONE,1).replace(B,CALL_TWO+B,1)
    s=s[:start]+body+s[end:]
    s=HEADER+s.replace(START,FUNCTION+START,1)
    out=s.encode()
    if strip(out)!=b:raise ValueError("unable to restore source exactly")
    return out
def strip(b:bytes)->bytes:
    s=b.decode()
    pairs=[(HEADER,""),(FUNCTION,""),(A+CALL_ONE,A),(CALL_TWO+B,B)]
    for old,new in pairs:
        if s.count(old)!=1:raise ValueError("unrecognized P11 source markers")
        s=s.replace(old,new,1)
    if MARK in s:raise ValueError("leftover injected P11 marker")
    return s.encode()
def run(source,output):
    source=pathlib.Path(source);output=pathlib.Path(output)
    if not source.is_file() or source.is_symlink() or output.exists():raise ValueError("expected regular input and new output")
    orig=source.read_bytes();changed=patch(orig)
    with output.open("xb") as f:f.write(changed)
    print("P11_ISOLATED_INFLIGHT_GATE",json.dumps({"source_SHA":sha(orig),
      "fault_source_SHA":sha(changed),"source_exactly_reversible":strip(changed)==orig,
      "windows":["first_bind","before_ack"],"optin":True,
      "production_touched":False},sort_keys=True),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--source",required=True)
    a.add_argument("--output",required=True)
    x=a.parse_args();run(x.source,x.output)
