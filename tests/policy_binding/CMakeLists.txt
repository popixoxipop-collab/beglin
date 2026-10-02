cmake_minimum_required(VERSION 3.20)
project(p8_b_policy_binding_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED MLX_CMAKE_DIR)
  message(FATAL_ERROR "MLX_CMAKE_DIR must point to the installed MLX CMake package")
endif()
find_package(MLX CONFIG REQUIRED PATHS ${MLX_CMAKE_DIR})

add_executable(binding_runtime_probe
  binding_runtime_probe.cpp
  ../../mlx_moe.cpp
)
target_include_directories(binding_runtime_probe PRIVATE ../..)
target_compile_options(binding_runtime_probe PRIVATE -O2 -Wall -Wextra)
target_link_libraries(binding_runtime_probe PRIVATE mlx "-framework Accelerate" pthread)
