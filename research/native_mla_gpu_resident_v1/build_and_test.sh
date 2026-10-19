#!/usr/bin/env bash
# BEGLIN portable research-only Apple Silicon GPU-resident qNg64 MLA test.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BEGLIN_REPO_ROOT:-$(cd "${HERE}/../.." && pwd)}"
PY="${BEGLIN_PYTHON:-python3}"
CXX="${CXX:-clang++}"
CC="${CC:-clang}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo 'BLOCKED: Apple Silicon Metal required' >&2
  exit 3
fi
test -f "${REPO_ROOT}/qng64_dense.c"
test -f "${REPO_ROOT}/qng64_dense.h"
MLX_PARENT="$("${PY}" -c 'from pathlib import Path; import mlx.core as m; print(Path(m.__file__).resolve().parent)')"
if [[ ! -f "${MLX_PARENT}/lib/libmlx.dylib" || ! -d "${MLX_PARENT}/include/mlx" ]]; then
  echo 'BLOCKED: compatible MLX C++ headers/library missing' >&2
  exit 3
fi
mkdir -p "${HERE}/build"

"${CC}" -std=c11 -O2 -fPIC -dynamiclib -Wall -Wextra -Werror \
  -I"${REPO_ROOT}" "${REPO_ROOT}/qng64_dense.c" \
  -o "${HERE}/build/libbeglin_qng64_reference.dylib"

COMMON=(
 -std=c++20 -O2 -fPIC -dynamiclib
 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-deprecated-copy
 "-I${MLX_PARENT}/include" "-I${HERE}"
 "-L${MLX_PARENT}/lib" -lmlx
 "-Wl,-rpath,${MLX_PARENT}/lib"
)
"${CXX}" "${COMMON[@]}" \
  "${HERE}/mla_qng64_metal_single_fused_v1.cpp" \
  -o "${HERE}/build/libbeglin_mla_qng64_single_fused_v1.dylib"
"${CXX}" "${COMMON[@]}" \
  "${HERE}/mla_qng64_resident_weight_handle_v1.cpp" \
  -o "${HERE}/build/libbeglin_mla_qng64_resident_weight_v1.dylib"

"${PY}" -m py_compile "${HERE}/test_qng64_resident_gpu_portable_v1.py"
"${PY}" -u "${HERE}/test_qng64_resident_gpu_portable_v1.py"
