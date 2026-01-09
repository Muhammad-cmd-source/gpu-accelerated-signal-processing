cmake_minimum_required(VERSION 3.18)
project(hpd_pipeline CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Set CUDA architectures for MX 230 (Compute Capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA kernels library - IMPORTANT: Changed to OBJECT library
add_library(cuda_kernels OBJECT
    cpp/src/cuda_kernels.cu
)

target_include_directories(cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Signal processor library
add_library(processor STATIC
    cpp/src/processor.cpp
)

target_include_directories(processor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(processor PUBLIC
    cuda_kernels
)

# Benchmark executable - Link with CUDA runtime
add_executable(benchmark
    cpp/src/benchmark.cpp
    $<TARGET_OBJECTS:cuda_kernels>
)

target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(benchmark PRIVATE
    processor
    CUDA::cudart
)

set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(benchmark_large
    cpp/src/benchmark_large.cpp
    $<TARGET_OBJECTS:cuda_kernels>
)

target_include_directories(benchmark_large PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(benchmark_large PRIVATE
    processor
    CUDA::cudart
)

set_target_properties(benchmark_large PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)