"""Exact-source P5 CB128 native observer; no model math/ACK/rollback changes.

Produces a separate file, never edits source in place. Only a later storage-
approved isolated build may execute it. Not native profiling evidence itself.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import re

ORIGINAL = 'dd73b7c7f997aa66d0a2217f9b0f3ab74c5db28b8dba1b586d4f182352bbcee1'
CB128 = 'bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612'
START = 'static int moe_gpu_rebind_set_apply_quiescent(const MoeGpuTxnCommand *cmd) {'
END = '\nstatic int moe_gpu_demotion_pending(void) {'
MARK = re.compile(r'/\* P8_OBSERVER_BEGIN:[0-9]+ \*/.*?/\* P8_OBSERVER_END:[0-9]+ \*/\n', re.S)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def to_cb128(source):
    if sha(source) != ORIGINAL or source.count('#define MOE_CBATCH_MAXPOS 32') != 1:
        raise ValueError('original P5 source SHA/CB32 anchor mismatch')
    return source.replace('#define MOE_CBATCH_MAXPOS 32', '#define MOE_CBATCH_MAXPOS 128')


def strip_observer(source):
    return MARK.sub('', source)


def patch(source):
    if sha(source) != CB128:
        raise ValueError('only exact approved P5 CB128 source accepted')
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('ambiguous native transaction function')
    start, end = source.index(START), source.index(END)
    body = source[start:end]
    count = 0
    def mark(text):
        nonlocal count
        count += 1
        return '/* P8_OBSERVER_BEGIN:%d */\n%s\n/* P8_OBSERVER_END:%d */\n' % (count, text, count)
    def before(anchor, text):
        nonlocal body
        if body.count(anchor) != 1:
            raise ValueError('native observer anchor not unique: '+anchor[:80])
        body = body.replace(anchor, mark(text)+anchor)
    def after(anchor, text):
        nonlocal body
        if body.count(anchor) != 1:
            raise ValueError('native observer anchor not unique: '+anchor[:80])
        body = body.replace(anchor, anchor+mark(text))

    before('    MoeAFTensor *base_ptrs[MOE_GPU_TXN_MAX_TARGETS] = {0};', '''    const char *p8_dir = getenv("BEGLIN_HCP_P8_PROFILE_DIR");
    const int p8_on = p8_dir && p8_dir[0];
    double p8_entry = p8_on ? nowt() : 0.0;
    double p8_valid = 0.0, p8_sync = 0.0, p8_snap = 0.0;
    double p8_loop = 0.0, p8_reset = 0.0, p8_meta = 0.0, p8_ack = 0.0;
    double p8_cache = 0.0, p8_bind = 0.0, p8_target_t = 0.0;''')
    # This sync anchor is unique only before the rollback label.
    before('    if (!mlx_gpu_synchronize()) {\n        fprintf(stderr, "[moe gpu rebind set] FAIL sync txn=%s\\n", cmd->txn_id);',
           '    if (p8_on) p8_valid = nowt();')
    before('    for (int i = 0; i < cmd->target_count; i++) {\n        if (!mlx_gpu_snapshot_binding',
           '    if (p8_on) p8_sync = nowt();')
    before('    for (int i = 0; i < cmd->target_count; i++) {\n        const MoeGpuTxnTarget *t = &cmd->targets[i];\n        MoeAFTensor *qnt',
           '    if (p8_on) p8_snap = nowt();')
    before('        MoeAFTensor *qnt = moe_gpu_qng64_cached(',
           '        if (p8_on) p8_target_t = nowt();')
    before('        int ok = qnt && mlx_gpu_bind_af(',
           '        if (p8_on) { double t = nowt(); p8_cache += t - p8_target_t; p8_target_t = t; }')
    before('        int applied_bits = 0;',
           '        if (p8_on) p8_bind += nowt() - p8_target_t;')
    before('    if (!mlx_gpu_reset_runtime_epoch()) {',
           '    if (p8_on) p8_loop = nowt();')
    before('    for (int i = 0; i < cmd->target_count; i++) {\n        (void)mlx_gpu_drop_binding_snapshot(attempt_sids[i]);',
           '    if (p8_on) p8_reset = nowt();')
    before('    if (!moe_gpu_txn_mark_terminal(cmd, "REBIND_SET_APPLIED", cmd->target_count)) {',
           '    if (p8_on) p8_meta = nowt();')
    before('    fprintf(stderr,\n            "[moe gpu rebind set] COMMIT txn=%s targets=%d epoch=%llu\\n",', r'''    if (p8_on) {
        p8_ack = nowt();
        /* Caller must bind this directory to its new run-owned workspace. */
        char p8_path[1024];
        int p8_npath = snprintf(p8_path, sizeof p8_path, "%s/native_phases.jsonl", p8_dir);
        if (p8_dir[0] != '/' || strstr(p8_dir, "..") ||
            p8_npath < 0 || (size_t)p8_npath >= sizeof p8_path) {
            fprintf(stderr, "FATAL: P8 invalid isolated profile directory\n"); exit(1);
        }
        int p8_fd = open(p8_path, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0600);
        if (p8_fd < 0) { perror("P8 profile open"); exit(1); }
        char p8_buf[2048];
        int p8_n = snprintf(p8_buf, sizeof p8_buf,
            "{\"schema\":\"beglin-p8-native-phases-v1\",\"epoch\":%llu,\"targets\":%d,"
            "\"precheck_ms\":%.9f,\"synchronize_ms\":%.9f,\"snapshot_ms\":%.9f,"
            "\"qng64_cache_lookup_or_build_ms\":%.9f,\"mlx_bind_ms\":%.9f,"
            "\"bind_verify_and_loop_ms\":%.9f,\"epoch_reset_ms\":%.9f,"
            "\"metadata_commit_ms\":%.9f,\"durable_ack_ms\":%.9f,"
            "\"entry_to_ack_ms\":%.9f,\"profile_write_excluded\":true}\n",
            (unsigned long long)g_moe_gpu_weight_epoch, cmd->target_count,
            (p8_valid-p8_entry)*1000.0, (p8_sync-p8_valid)*1000.0,
            (p8_snap-p8_sync)*1000.0, p8_cache*1000.0, p8_bind*1000.0,
            (p8_loop-p8_snap-p8_cache-p8_bind)*1000.0,
            (p8_reset-p8_loop)*1000.0, (p8_meta-p8_reset)*1000.0,
            (p8_ack-p8_meta)*1000.0, (p8_ack-p8_entry)*1000.0);
        if (p8_n < 0 || (size_t)p8_n >= sizeof p8_buf ||
            write(p8_fd, p8_buf, (size_t)p8_n) != p8_n) {
            close(p8_fd); fprintf(stderr, "FATAL: P8 incomplete phase record\n"); exit(1);
        }
        if (close(p8_fd)) { perror("P8 profile close"); exit(1); }
    }''')
    altered = source[:start]+body+source[end:]
    # Header insertion is also reversible, and not dependent on prior includes.
    altered = mark('#include <fcntl.h>\n#include <unistd.h>')+altered
    if strip_observer(altered) != source:
        raise ValueError('observer changed original bytes beyond reversible inserts')
    if altered.count('REBIND_SET_APPLIED') != source.count('REBIND_SET_APPLIED'):
        raise ValueError('ACK contract changed')
    return altered


def validate_phases(row):
    fields = ('precheck_ms','synchronize_ms','snapshot_ms','qng64_cache_lookup_or_build_ms',
              'mlx_bind_ms','bind_verify_and_loop_ms','epoch_reset_ms','metadata_commit_ms','durable_ack_ms')
    import math
    if row.get('schema') != 'beglin-p8-native-phases-v1' or row.get('targets') != 2:
        raise ValueError('wrong profile contract')
    vals = [row.get(k) for k in fields+('entry_to_ack_ms',)]
    if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < -1e-6 for v in vals):
        raise ValueError('invalid phase timing')
    if abs(sum(vals[:-1])-vals[-1]) > 0.001:
        raise ValueError('phase sum does not reconcile with entry-to-ACK time')
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    args = ap.parse_args()
    if args.source.is_symlink() or args.output.exists():
        raise SystemExit('source must be regular; output must be a new file')
    original = args.source.read_text()
    base = to_cb128(original) if sha(original)==ORIGINAL else original
    result = patch(base)
    with args.output.open('x') as stream:
        stream.write(result)
    print(json.dumps({'original_sha256':sha(original),'cb128_sha256':sha(base),
        'observer_sha256':sha(result),'roundtrip_restores_P5':strip_observer(result)==base,
        'native_executed':False,'production_touched':False},sort_keys=True))
