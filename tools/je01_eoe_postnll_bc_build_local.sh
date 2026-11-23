#!/bin/zsh
# JE01 post-NLL B/C source-pinned Apple MLX original27 native research compile.
# Scratch-only D50, no model/serving or holdout mutation.
set -euo pipefail
umask 077
PIN="$1"
ROOT="/Volumes/D50/TailnetCommander/work/je01_postnll_bc_native_compile_20261123_v1"
[[ "$PIN" =~ '^[0-9a-f]{40}$' ]]
test -d "$ROOT/checkout/.git"
test -x "$ROOT/venv/bin/cmake"
test ! -e "$ROOT/source"
test ! -e "$ROOT/build"
git -C "$ROOT/checkout" diff --quiet
git -C "$ROOT/checkout" diff --cached --quiet
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
git -C "$ROOT/checkout" fetch --no-tags --depth 1 origin research/je01-moe-bayesian-bit-allocation-20261117
test "$(git -C "$ROOT/checkout" rev-parse FETCH_HEAD)" = "$PIN"
git -C "$ROOT/checkout" checkout -B je01-postnll-bc-frozen-source "$PIN"
cd "$ROOT/checkout"
/opt/homebrew/bin/python3.11 tests/test_je01_eoe_postnll_bc_overlay_contract.py
/opt/homebrew/bin/python3.11 tests/test_je01_postnll_bc_exactbyte_native_bundle.py
echo "JE01_POSTNLL_BC_LOCAL_NATIVE_SOURCE_AND_PREREG_PASS"

/opt/homebrew/bin/python3.11 tools/je01_isolated_context384_stage.py --checkout "$ROOT/checkout" --scratch "$ROOT/source" --expected-commit "$PIN" > "$ROOT/base_stage.json"
echo "JE01_POSTNLL_BC_ORIGINAL384_ISOLATED_STAGE_PASS"
/opt/homebrew/bin/python3.11 tools/je01_scratch_cb384_overlay_stage.py --staged-source "$ROOT/source" > "$ROOT/cbatch_stage.json"
/opt/homebrew/bin/python3.11 - <<'PY'
import importlib.util,pathlib
p=pathlib.Path('tools/je01_xox_original256_fulltarget_teacherforced_nll.py').resolve()
spec=importlib.util.spec_from_file_location('je01_teacher_forced',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=pathlib.Path('/Volumes/D50/TailnetCommander/work/je01_postnll_bc_native_compile_20261123_v1')
f=root/'source/qwen_infer.c'
f.write_text(m.patch_cpu(f.read_text()))
PY
/opt/homebrew/bin/python3.11 tools/je01_eoe_original384_mixed_source_stage.py --scratch "$ROOT/source" > "$ROOT/q35_stage.json"
/opt/homebrew/bin/python3.11 tools/je01_eoe_original384_q45_scratch_stage.py --scratch "$ROOT/source" > "$ROOT/q45_stage.json"
/opt/homebrew/bin/python3.11 tools/je01_eoe_original384_affine_keep4_mix4_stage.py --scratch "$ROOT/source" > "$ROOT/affine_stage.json"
echo "JE01_POSTNLL_BC_NATIVE_CODEC_FULL_GOLD_SOURCE_STAGED_PASS"

/opt/homebrew/bin/python3.11 - <<'PY'
import importlib.util,pathlib,hashlib,json,subprocess
src=pathlib.Path('tools/je01_eoe_postnll_bc_native_overlay_patch.py').resolve()
spec=importlib.util.spec_from_file_location('je01_postnll_bc_patch',src)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=pathlib.Path('/Volumes/D50/TailnetCommander/work/je01_postnll_bc_native_compile_20261123_v1')
cpp=root/'source/mlx_moe.cpp'
c=root/'source/qwen_infer.c'
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cpp,c)}
new_cpp,new_c=m.patch(cpp.read_text(),c.read_text())
assert new_cpp.count('JE01_CAL240_POSTNLL_ARM')==1
assert new_cpp.count('altered_experts==7')==1
assert new_c.count(m.NEW_MARK)==1
cpp.write_text(new_cpp);c.write_text(new_c)
cm=root/'source/CMakeLists.txt'
old="OUTPUT_NAME qwen_infer_gpu_je01_context384"
assert cm.read_text().count(old)==1
cm.write_text(cm.read_text().replace(old,'OUTPUT_NAME je01_eoe_original384_cal240_bc_rolewise_fullgold',1))
pinned=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
record={'status':'JE01_POSTNLL_BC_ORIGINAL27_APPLE_MLX_NATIVE_SOURCE_STAGED',
        'source_patch_git_head':pinned,
        'patched_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cpp,c)},
        'original_sha256':before,
        'policy_arms':['B_PROTECT_EARLY_GATE','C_PROTECT_EARLY_UP'],
        'model_NLL':'NOT_RUN','production_modified':False}
(root/'source_patch_receipt.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print('JE01_POSTNLL_BC_ORIGINAL27_ROLEWISE_NATIVE_PATCH_PASS',flush=True)
PY

MLX_CMAKE_DIR="$(/opt/homebrew/bin/python3.11 -c 'import mlx,os;print(os.path.join(list(mlx.__path__)[0],"share","cmake"))')"
test -d "$MLX_CMAKE_DIR"
"$ROOT/venv/bin/cmake" -S "$ROOT/source" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_CMAKE_DIR="$MLX_CMAKE_DIR"
"$ROOT/venv/bin/cmake" --build "$ROOT/build" --target qwen_infer_gpu --parallel 2
mkdir -p "$ROOT/bin"
cp "$ROOT/build/je01_eoe_original384_cal240_bc_rolewise_fullgold" "$ROOT/bin/je01_eoe_original384_cal240_bc_rolewise_fullgold"
chmod 700 "$ROOT/bin/je01_eoe_original384_cal240_bc_rolewise_fullgold"
/opt/homebrew/bin/python3.11 - <<'PY'
import pathlib,hashlib,json
root=pathlib.Path('/Volumes/D50/TailnetCommander/work/je01_postnll_bc_native_compile_20261123_v1')
exe=root/'bin/je01_eoe_original384_cal240_bc_rolewise_fullgold'
patch=json.loads((root/'source_patch_receipt.json').read_text())
assert exe.is_file() and exe.stat().st_size>100000
receipt={'schema':'beglin-je01-original27-postnll-bc-local-native-build/1',
    'status':'JE01_POSTNLL_BC_7EXPERT_ORIGINAL27_NATIVE_COMPILE_PASS',
    'base_git_commit':patch['source_patch_git_head'],
    'source_patch':patch,
    'native_executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
    'native_executable_bytes':exe.stat().st_size,
    'model_gold_NLL':'NOT_RUN','production_modified':False}
(root/'JE01_POSTNLL_BC_LOCAL_NATIVE_BUILD_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print('JE01_POSTNLL_BC_LOCAL_NATIVE_COMPILE_PASS',json.dumps({
  'exe_sha256':receipt['native_executable_sha256'],
  'bytes':receipt['native_executable_bytes']},sort_keys=True),flush=True)
PY
