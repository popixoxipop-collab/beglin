#!/usr/bin/env python3
"""Exact-P5-source, scratch-only MLA128 build plus opt-in full-vocab probe."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,subprocess
from hcp_p5_scratch_pair_capacity import patch_both

HELPER = r"""
// HCP P6 research only: read finalized logits; never changes logits or tokens.
// Native-endian u32 [magic, slot, position, vocab] then vocab float32 values.
static void hcp_p6_dump_logits(const float *logits, int count, int vocab,
                               const int *slots, const int *positions) {
    const char *path = getenv("BEGLIN_HCP_P6_LOGITS_PATH");
    if (!path || !path[0]) return;
    if (count < 1 || vocab < 1 || sizeof(float) != 4) {
        fprintf(stderr, "FATAL: P6 invalid logit capture shape\n"); exit(1);
    }
    FILE *f = fopen(path, "ab");
    if (!f) { fprintf(stderr, "FATAL: P6 logit capture open failed\n"); exit(1); }
    for (int m = 0; m < count; m++) {
        uint32_t header[4] = {0x50364c47u, (uint32_t)slots[m],
                             (uint32_t)positions[m], (uint32_t)vocab};
        if (fwrite(header, sizeof(uint32_t), 4, f) != 4 ||
            fwrite(logits + (size_t)m*vocab, sizeof(float), vocab, f) != (size_t)vocab) {
            fprintf(stderr, "FATAL: P6 incomplete logit capture\n"); fclose(f); exit(1);
        }
    }
    if (fclose(f) != 0) { fprintf(stderr, "FATAL: P6 logit capture close failed\n"); exit(1); }
}
"""
CALL = """            if (pass == 1)
                hcp_p6_dump_logits(gpu_logits, A, MOE_VOCAB, slot_arr, spos_arr);

"""

def patch_capture(source: str) -> str:
    anchor='            // 4. decode columns: emit + evict (EOS / stop_extra / maxnew / position cap).'
    if source.count(anchor) != 1:
        raise ValueError('MLA online dump anchor is not unique')
    if 'hcp_p6_dump_logits' in source:
        raise ValueError('capture patch is already applied')
    include='#include <errno.h>'
    if source.count(include) != 1: raise ValueError('source include anchor mismatch')
    source=source.replace(include,include+'\n#include <stdint.h>\n'+HELPER,1)
    return source.replace(anchor,CALL+anchor,1)

def build_source(repo: pathlib.Path, dest: pathlib.Path) -> dict:
    repo=repo.resolve(strict=True); dest=dest.resolve()
    if dest.exists() or dest.parent != repo.parent or dest.name != 'probe-src':
        raise ValueError('only a fresh probe-src sibling of the research checkout is allowed')
    if repo.parent.name != 'hcp-p6-af-logits-20261014-1104':
        raise ValueError('wrong explicitly owned research root')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    dest.mkdir()
    # Copy tracked source only; no model/data, existing build, or production tree.
    import shutil
    names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
    for name in filter(None,names):
        src=repo/name
        if src.is_symlink(): raise ValueError('tracked symlink is not supported')
        if src.is_file():
            out=dest/name; out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,out)
    bounds=patch_both(dest)
    c=dest/'qwen_infer.c'; c.write_text(patch_capture(c.read_text()))
    files={name:hashlib.sha256((dest/name).read_bytes()).hexdigest()
           for name in ['qwen_infer.c','mlx_moe.cpp','CMakeLists.txt']}
    rec={'schema':'beglin-hcp-p6-build-source-v1','source_commit':head,
         'p5_base_commit':'b5cf761089153a6326855f267f1381fadb3f414e',
         'bounds':bounds,'instrumented_source_sha256':files,
         'dump_default':'disabled','mathematics_and_ack_unchanged':True,
         'production_touched':False,'automatic_live_promotion':False}
    (repo.parent/'build-source.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n')
    print(json.dumps(rec,sort_keys=True))
    return rec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True)
    p.add_argument('--dest',type=pathlib.Path,required=True);a=p.parse_args()
    build_source(a.repo,a.dest)
