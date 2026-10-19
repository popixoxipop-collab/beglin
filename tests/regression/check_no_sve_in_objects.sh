#!/bin/bash
# Fail if any given object file contains SVE-style instructions (z/p registers, ptrue, whilelt, cnt[bhwd], index..z).
#
# Used by ctest (kleidi_neon_no_sve) on the KleidiAI objects that are plain NEON code.
#
# D-N8-guard: why this check exists.
#   KleidiAI's SME2 kernels legitimately contain SVE/SME instructions, but they only ever run inside a streaming-mode
#   region. The NEON pack functions (kai_lhs_*, kai_rhs_pack_*) run in normal mode. When they were compiled with
#   -march=armv9.2-a+sme2 the compiler vectorized loops with SVE (cntw, ptrue, index z..), and on Apple M4, which has no
#   non-streaming SVE, the first call SIGILLed (the int8-LHS path, QWEN_MOE_SME2_F16LHS=0).
#   WHY : a build flag silently changes codegen; only inspecting the produced objects catches it.
#   COST: macOS only (needs otool), reads each object once.
#   EXIT: replace otool with llvm-objdump when the build is made portable.
set -u
if [ "$#" -lt 1 ]; then echo "usage: $0 <object-file>..." >&2; exit 2; fi
command -v otool > /dev/null 2>&1 || { echo "otool not found" >&2; exit 2; }
PAT='\bz[0-9]+\.|\bp[0-9]+\.|\bptrue\b|\bwhilelt\b|\bcnt[bhwd]\b|\bindex\b[^;]*\bz[0-9]'
bad=0
for o in "$@"; do
  if [ ! -f "$o" ]; then echo "missing object: $o" >&2; exit 2; fi
  n=$(otool -tV "$o" 2> /dev/null | /usr/bin/grep -a -c -E "$PAT")
  if [ "$n" -gt 0 ]; then
    echo "FAIL: $(basename "$o"): $n SVE-style instruction(s)"
    otool -tV "$o" 2> /dev/null | /usr/bin/grep -a -E "$PAT" | head -3 | sed 's/^/        /'
    bad=1
  else
    echo "ok:   $(basename "$o")"
  fi
done
exit $bad
