# GPU/MLX build for qwen_infer -- the variant the README's own Build section
# does NOT cover. That section documents the CPU-only engine ("no MLX, no
# llama.cpp"); everything behind `#ifdef QWEN_GPU_MLX` (mlx_moe.cpp, all 15+
# run_moe_gpu_* gates) had no build recipe anywhere in this repo until this
# file. Verified building `qwen_infer_gpu` clean on an M4 (bob) against a
# pip-installed MLX wheel, 2026-09-02.
#
# Usage:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/qwen_infer_gpu          # env-var gated, same as the CPU binary
#
# MLX comes from the pip wheel, not a from-source MLX build: the wheel ships
# the full C++ side (`include/mlx/mlx.h`, `lib/libmlx.dylib`, and a CMake
# package config), so `pip install mlx` is the only dependency step. Override
# the search path if your interpreter differs:
#   cmake -B build -DMLX_CMAKE_DIR=/path/to/site-packages/mlx/share/cmake
#
# Two pitfalls, both of which fail SILENTLY -- keep them in mind before
# debugging a mysterious link error or a binary that ignores its env vars:
#
#   1. `ASM` must be in the project() language list. Without it CMake produces
#      no build rule for the KleidiAI `.S` files, configures without
#      complaint, and the failure only surfaces at link time as undefined
#      kai_commit_za / kai_get_sme_vector_length_u8 / kai_kernel_* symbols
#      that look like a missing source file rather than a missing language.
#
#   2. `QWEN_GPU_MLX` must be defined for qwen_infer.c. Without it every
#      run_moe_gpu_* gate compiles out, the link still succeeds (mlx_moe.cpp
#      just goes unused), and the binary silently falls through to CPU-only
#      dispatch no matter which QWEN_MOE_GPU_* env var you set -- no error,
#      no warning, just the wrong code path.

cmake_minimum_required(VERSION 3.20)
project(qwen_infer_gpu LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the MLX CMake package inside a Python that actually has mlx installed,
# unless the caller pins it explicitly. Two details that bite here: `mlx` is a
# namespace package, so `mlx.__file__` is None and only `mlx.__path__` resolves;
# and the first python3 on PATH is often the system one without mlx, so this
# tries each candidate and keeps the first that imports it.
if(NOT DEFINED MLX_CMAKE_DIR)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(_mlx_pythons ${Python3_EXECUTABLE} python3 /opt/homebrew/bin/python3 /usr/local/bin/python3)
  foreach(_py IN LISTS _mlx_pythons)
    if(_py AND NOT MLX_CMAKE_DIR)
      execute_process(
        COMMAND ${_py} -c
          "import mlx, os; print(os.path.join(list(mlx.__path__)[0], 'share', 'cmake'))"
        OUTPUT_VARIABLE _mlx_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _mlx_rc)
      if(_mlx_rc EQUAL 0 AND EXISTS "${_mlx_dir}")
        set(MLX_CMAKE_DIR "${_mlx_dir}")
        message(STATUS "MLX CMake package found via ${_py}: ${MLX_CMAKE_DIR}")
      endif()
    endif()
  endforeach()
endif()
if(NOT MLX_CMAKE_DIR)
  message(FATAL_ERROR
    "MLX not found. Install it (`pip install mlx`) or pass -DMLX_CMAKE_DIR=<site-packages>/mlx/share/cmake")
endif()
find_package(MLX CONFIG REQUIRED PATHS ${MLX_CMAKE_DIR})

# Main engine + loaders: plain compile, no arch flags. This is the
# caller-plain convention -- see RESULTS.md "caller-plain convention" for why
# compiling these with an SME/SVE -march flag can SIGILL on real hardware.
# The *_verify.c files are deliberately excluded: each carries its own main().
add_library(qwen_infer_obj OBJECT
  qwen_infer.c
  hf_config.c
  safetensors_load.c
  safetensors_quants.c
  gguf_load.c gguf_quants.c gguf_transcode.c gguf_cache.c
)
target_compile_options(qwen_infer_obj PRIVATE -O3 -w)
target_compile_definitions(qwen_infer_obj PRIVATE QWEN_GPU_MLX)   # pitfall 2

# SME2 kernel wrapper: dispatches through function pointers, no SME/SVE code
# of its own, but built with the arch flag so the dispatch targets resolve.
add_library(sme2_kai_obj OBJECT sme2_kai.c)
target_compile_options(sme2_kai_obj PRIVATE -O2 -march=armv9.2-a+sme2)

# Vendored KleidiAI kernels + their assembly. Standalone benchmark/test
# translation units in that directory carry their own main() and are excluded.
file(GLOB KLEIDI_SRCS kleidiai/*.c kleidiai/*.S)
list(FILTER KLEIDI_SRCS EXCLUDE REGEX
  "(kai_test|thread_scaling|prototype|vdsp_ref|verify_gemm|verify_real_repack)")
add_library(kleidi_obj OBJECT ${KLEIDI_SRCS})
target_compile_options(kleidi_obj PRIVATE
  -O2 -march=armv9.2-a+sme2 -I${CMAKE_SOURCE_DIR} -I${CMAKE_SOURCE_DIR}/kleidiai -w)

# MLX GPU MoE glue (C++).
add_library(mlx_moe_obj OBJECT mlx_moe.cpp)
target_compile_options(mlx_moe_obj PRIVATE -O3 -w)
target_link_libraries(mlx_moe_obj PRIVATE mlx)

add_executable(qwen_infer_gpu
  $<TARGET_OBJECTS:qwen_infer_obj>
  $<TARGET_OBJECTS:sme2_kai_obj>
  $<TARGET_OBJECTS:kleidi_obj>
  $<TARGET_OBJECTS:mlx_moe_obj>
)
target_link_libraries(qwen_infer_gpu PRIVATE mlx "-framework Accelerate" pthread)
