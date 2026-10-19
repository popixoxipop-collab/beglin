#!/usr/bin/env bash
# BEGLIN n7/group64 multiple matrix GPU-resident research-only Apple Metal test.
# No checkpoint access, account/network calls, remote runtime or deployment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BEGLIN_REPO_ROOT:-$(cd "${HERE}/../.." && pwd)}"
PY="${BEGLIN_PYTHON:-python3}"
CXX="${CXX:-clang++}"
CC="${CC:-clang}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo 'BLOCKED: Apple Silicon + Metal device is required' >&2
  exit 3
fi
test -f "${REPO_ROOT}/qng64_dense.c"
test -f "${REPO_ROOT}/qng64_dense.h"
MLX_PARENT="$("${PY}" -c 'from pathlib import Path; import mlx.core as m; print(Path(m.__file__).resolve().parent)')"
if [[ ! -f "${MLX_PARENT}/lib/libmlx.dylib" || ! -d "${MLX_PARENT}/include/mlx" ]]; then
  echo 'BLOCKED: installed MLX C++ headers/dylib not found' >&2
  exit 3
fi
mkdir -p "${HERE}/build"
# Use unique per-run QA evidence so CI reruns never overwrite receipts.
RUN_OUTPUT_DIR="$(mktemp -d "${HERE}/build/qa-run.XXXXXXXX")"
export BEGLIN_TEST_RECEIPTS_DIR="${RUN_OUTPUT_DIR}"

"${CC}" -std=c11 -O2 -fPIC -dynamiclib -Wall -Wextra -Werror \
  -I"${REPO_ROOT}" \
  "${REPO_ROOT}/qng64_dense.c" \
  -o "${HERE}/build/libbeglin_qng64_reference.dylib"

"${CXX}" -std=c++20 -O2 -fPIC -dynamiclib \
 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-deprecated-copy \
 "-I${MLX_PARENT}/include" "-I${HERE}" \
 "${HERE}/n7_multi_projection_gpu_residency_v10_256mb.cpp" \
 "-L${MLX_PARENT}/lib" -lmlx \
 "-Wl,-rpath,${MLX_PARENT}/lib" \
 -o "${HERE}/build/libbeglin_n7_multi_projection_GPU_resident_v10_256mb.dylib"

"${PY}" -m py_compile "${HERE}/test_multi_gpu_n7_v10_portable.py"
"${PY}" -u "${HERE}/test_multi_gpu_n7_v10_portable.py"

"${PY}" -m py_compile "${HERE}/research_n7_source_pack_memo_v1.py" "${HERE}/test_n7_host_packed_memo_portable_v1.py"
"${PY}" -u "${HERE}/test_n7_host_packed_memo_portable_v1.py"

"$PY" -m py_compile "$HERE/estimate_n7_host_cache_budget_v1.py" "$HERE/test_estimate_n7_host_cache_budget_v1.py"
"$PY" -u "$HERE/test_estimate_n7_host_cache_budget_v1.py"
