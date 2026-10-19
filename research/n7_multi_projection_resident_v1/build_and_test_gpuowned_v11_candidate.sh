#!/usr/bin/env bash
# Research-only BEGLIN original n7/g64 GPU computed-ownership cold-process QA.
# Requires physical Apple Silicon Metal. Do NOT merge into serving, ship
# model tensors or treat candidate memory ownership as production-certified.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BEGLIN_REPO_ROOT:-$(cd "${HERE}/../.." && pwd)}"
PY="${BEGLIN_PYTHON:-python3}"
CC="${CC:-clang}"
CXX="${CXX:-clang++}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo 'BLOCKED: physical Apple Silicon Metal required' >&2
  exit 3
fi
if [[ ! -f "${REPO_ROOT}/qng64_dense.c" || ! -f "${REPO_ROOT}/qng64_dense.h" ]]; then
  echo 'BLOCKED: original BEGLIN C group64/n7 source required' >&2
  exit 3
fi
MLX="$("${PY}" -c 'import mlx.core as m;from pathlib import Path;print(Path(m.__file__).resolve().parent)')"
if [[ ! -d "${MLX}/include/mlx" || ! -f "${MLX}/lib/libmlx.dylib" ]]; then
  echo 'BLOCKED: source-compatible MLX C++ headers/library required' >&2
  exit 3
fi
mkdir -p "${HERE}/build"
CPP="${HERE}/n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp"
TEST="${HERE}/test_n7_GPU_only_ownership_v11_portable.py"
if [[ ! -f "${CPP}" || ! -f "${TEST}" ]]; then
  echo 'BLOCKED: candidate C++ source or synthetic safety test missing' >&2
  exit 3
fi

"${CC}" -std=c11 -O2 -fPIC -dynamiclib -Wall -Wextra -Werror \
  -I"${REPO_ROOT}" "${REPO_ROOT}/qng64_dense.c" \
  -o "${HERE}/build/libbeglin_qng64_reference.dylib"
"${CXX}" -std=c++20 -O2 -fPIC -dynamiclib \
  -Wall -Wextra -Werror -Wno-unused-parameter -Wno-deprecated-copy \
  "-I${MLX}/include" "-I${HERE}" \
  "${CPP}" "-L${MLX}/lib" -lmlx "-Wl,-rpath,${MLX}/lib" \
  -o "${HERE}/build/libbeglin_n7_multi_projection_GPU_resident_v11_candidate.dylib"

"${PY}" -m py_compile "${TEST}"
# Child processes each construct new MLX GPU contexts / native registries
# and preserve append-only original receipts in separate QA directories.
QA_GROUP="$(mktemp -d "${HERE}/build/gpuowned-cold.XXXXXXXX")"
for RUN in 1 2 3; do
  QA_RUN="${QA_GROUP}/run_${RUN}"
  mkdir -p "${QA_RUN}"
  echo "BEGLIN_V11_GPU_COLD_PROCESS_RUN=${RUN}"
  BEGLIN_TEST_RECEIPTS_DIR="${QA_RUN}" PYTHONHASHSEED="${RUN}" \
    "${PY}" -u "${TEST}"
done

BEGLIN_V11_QA_GROUP="${QA_GROUP}" "${PY}" - <<'PY'
import hashlib,json,os,pathlib
group=pathlib.Path(os.environ["BEGLIN_V11_QA_GROUP"])
receipts=[]
for run in range(1,4):
    path=group/f"run_{run}"/"multi_n7_gpu_resident_Metal_synthetic_receipt_v11.json"
    if not path.is_file():raise RuntimeError("FAIL_CLOSED: missing isolated receipt")
    raw=path.read_bytes()
    d=json.loads(raw)
    if d.get("status")!="PASS_GENERAL_MULTIPLE_QNG64_N7_DEVICE_RESIDENT_GEMV":
        raise RuntimeError("FAIL_CLOSED: physical Metal v11 synthetic run not PASS")
    if (d.get("checkpoint_loaded") is not False
        or d.get("CXX_registry_duplicate_host_compact_buffers") is not False
        or d.get("computed_GPU_device_ownership_candidate") is not True
        or d.get("source_host_pointer_lifetime_pressure_test") is not True
        or d.get("registered_after_cleanup") != 0
        or d.get("matrix_count")!=6
        or d.get("GPU_device_resident_total_packed_and_scales_bytes")!=15052800
        or not (0<=d.get("max_cpu_C_ref_relative_L2",float("inf"))<1e-4)):
        raise RuntimeError("FAIL_CLOSED: candidate output / scope drift")
    receipts.append({
      "run":run,"source_receipt_sha256":hashlib.sha256(raw).hexdigest(),
      "max_C_relative_L2":d["max_cpu_C_ref_relative_L2"],
      "GPU_MLX_active_B":d["GPU_MLX_active_memory_with_all_matrices_bytes"],
    })
print(json.dumps({
  "status":"PASS_THREE_ISOLATED_COLD_PROCESSES_GPU_COMPUTED_OWNER_SYNTHETIC",
  "runs":receipts,
  "independent_CPU_compatible":True,
  "original_checkpoint_used":False,
  "production_promotion_allowed":False,
  "model_wide_GPU_exclusive_memory_verified":False
},sort_keys=True))
PY
