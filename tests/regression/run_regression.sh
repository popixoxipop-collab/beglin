#!/bin/bash
# Build and run the CPU-scalar regression tests in tests/regression/.
#
# Usage:  tests/regression/run_regression.sh [test_name ...]     (default: every tests/regression/test_*.c)
#   env:  REG_OUT=<dir>   build/output directory (default: <repo>/build/regression)
#
# Writes only under $REG_OUT. Exit code is non-zero if any test fails to build or reports FAIL.
#
# D-reg-2: tests #include qwen_infer.c (main renamed) and link only the plain loader objects with
#          -undefined dynamic_lookup, so SME2/KleidiAI/MLX are not needed.
#   WHY : exercises the real static functions on any Apple Silicon host in seconds.
#   COST: SME2 / GPU paths are only covered where a test stubs or compiles them separately.
#   EXIT: once the engine is a library, link it instead of including the .c file.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/tests/regression"
OUT="${REG_OUT:-$ROOT/build/regression}"
OBJ="$OUT/obj"
mkdir -p "$OBJ"

BASE="bpe_tokenizer gguf_cache gguf_load gguf_quants gguf_transcode gguf_write gguf_write_quants hf_config qng64_dense safetensors_load safetensors_quants"
BASE_OBJS=""
for f in $BASE; do
  src="$ROOT/$f.c"; obj="$OBJ/$f.o"
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    nice -n 10 clang -O1 -w -I"$ROOT" -c "$src" -o "$obj" || { echo "BUILD FAIL: $f.c"; exit 1; }
  fi
  BASE_OBJS="$BASE_OBJS $obj"
done

python3 "$HERE/mk_tiny_st.py" "$OUT/tiny.safetensors" || exit 1
export REG_ST="$OUT/tiny.safetensors"

if [ "$#" -gt 0 ]; then TESTS="$*"; else TESTS="$(cd "$HERE" && ls test_*.c | sed 's/\.c$//')"; fi
status=0
for t in $TESTS; do
  tobj="$OBJ/$t.o"; tbin="$OUT/$t"
  # a test includes qwen_infer.c and reg_common.h, so rebuild when either (or the test) changed
  if [ ! -f "$tobj" ] || [ "$HERE/$t.c" -nt "$tobj" ] || [ "$ROOT/qwen_infer.c" -nt "$tobj" ] || [ "$HERE/reg_common.h" -nt "$tobj" ]; then
    nice -n 10 clang -O1 -w -I"$ROOT" -c "$HERE/$t.c" -o "$tobj" || { echo "BUILD FAIL: $t"; status=1; continue; }
  fi
  nice -n 10 clang -O1 $BASE_OBJS "$tobj" -framework Accelerate -lm -lpthread \
    -Wl,-undefined,dynamic_lookup -o "$tbin" || { echo "LINK FAIL: $t"; status=1; continue; }
  echo "=== $t ==="
  if "$tbin" 2> "$OUT/$t.stderr"; then echo "PASS: $t"; else echo "FAILED: $t (exit $?)"; status=1; fi
done
exit $status
