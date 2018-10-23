set(TARGET sm_61)

# TODO: Which version do we want?
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fractional_gpu VERSION 1.0.1 LANGUAGES CXX CUDA DESCRIPTION "Split GPU into Fractional GPUs")

# Include the file contanining CMake options
include(config.cmake.in)

# Check for contraints on configurations

# For memory coloring, computational coloring is a must
if(FGPU_MEM_COLORING_ENABLED AND NOT FGPU_COMP_COLORING_ENABLE)
    message(FATAL_ERROR "FGPU_COMP_COLORING_ENABLE not defined")
endif()

# When userspace/test coloring is enabled, coloring must be enabled
if((FGPU_USER_MEM_COLORING_ENABLED AND NOT FGPU_MEM_COLORING_ENABLED) OR 
	(FGPU_TEST_MEM_COLORING_ENABLED AND NOT FGPU_MEM_COLORING_ENABLED))
    message(FATAL_ERROR "FGPU_MEM_COLORING_ENABLED not defined")
endif()

# Only one can be selected at a time
if(FGPU_USER_MEM_COLORING_ENABLED AND FGPU_TEST_MEM_COLORING_ENABLED)
    message(FATAL_ERROR "Both FGPU_USER_MEM_COLORING_ENABLED and FGPU_TEST_MEM_COLORING_ENABLED defined")
endif()


# Equivalent variables for kernel driver
if(FGPU_MEM_COLORING_ENABLED)
    set(UVM_MEM_COLORING 1)
endif()

if(FGPU_USER_MEM_COLORING_ENABLED)
    set(UVM_USER_MEM_COLORING 1)
endif()

if(FGPU_TEST_MEM_COLORING_ENABLED)
    set(UVM_TEST_MEM_COLORING 1)
endif()

# Set configuration variables for FGPU source code
configure_file(include/fgpu_internal_build_config.hpp.in            
    "${PROJECT_SOURCE_DIR}/include/fgpu_internal_build_config.hpp")

# Set configuration variables for Kernel driver source code
configure_file(driver/NVIDIA-Linux-x86_64-390.48/kernel/common/inc/uvm_config.h.in            
    "${PROJECT_SOURCE_DIR}/driver/NVIDIA-Linux-x86_64-390.48/kernel/common/inc/uvm_config.h")

# Configure the build type
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(include)
include_directories(programs/cuda_samples/common/inc)
include_directories(driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/)
include_directories(driver/NVIDIA-Linux-x86_64-390.48/kernel/common/inc/)

string(APPEND CMAKE_CUDA_FLAGS " -m 64 -arch=${TARGET}")

# Add programs to be built using persistent threads model
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_persistent_target target include_dir)

    add_executable(${target} ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} fractional_gpu cuda cudart nvidia-ml)
    target_include_directories(${target}  PUBLIC ${include_dir})

endfunction(add_persistent_target)

# Add programs to be built _without_ using persistent threads model
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_native_target target include_dir)

    add_executable(${target} ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_include_directories(${target}  PUBLIC ${include_dir})

endfunction(add_native_target)

# Library
add_library(fractional_gpu SHARED
    persistent/persistent.cu
    persistent/memory.cu
    persistent/allocator.cpp
)
set_property(TARGET fractional_gpu PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET fractional_gpu PROPERTY PUBLIC_HEADER
		include/fractional_gpu.h
		include/fractional_gpu_cuda.cuh)
target_include_directories(fractional_gpu PRIVATE include)
target_include_directories(fractional_gpu PRIVATE persistent)
target_compile_features(fractional_gpu PUBLIC cxx_std_11)
target_link_libraries(fractional_gpu cuda cudart nvidia-ml)

#server
add_persistent_target(fgpu_server programs programs/server.cu)

# Dummy
add_persistent_target(dummy_persistent programs/dummy_persistent
    programs/dummy_persistent/dummy_persistent.cu)

# Matrix Multiplication
add_native_target(matrixMul programs/cuda_samples/matrixMul
    programs/cuda_samples/matrixMul/matrixMul.cu)

add_persistent_target(matrixMul_persistent programs/cuda_samples/matrixMul_persistent
    programs/cuda_samples/matrixMul_persistent/matrixMul_persistent.cu)

# Sorting Algos
add_native_target(sortingNetworks programs/cuda_sample/sortingNetworks
    programs/cuda_samples/sortingNetworks/main.cpp
    programs/cuda_samples/sortingNetworks/bitonicSort.cu
    programs/cuda_samples/sortingNetworks/oddEvenMergeSort.cu
    programs/cuda_samples/sortingNetworks/sortingNetworks_validate.cpp)

add_persistent_target(sortingNetworks_persistent programs/cuda_sample/sortingNetworks_persistent
    programs/cuda_samples/sortingNetworks_persistent/main.cpp
    programs/cuda_samples/sortingNetworks_persistent/bitonicSort.cu
    programs/cuda_samples/sortingNetworks_persistent/oddEvenMergeSort.cu
    programs/cuda_samples/sortingNetworks_persistent/sortingNetworks_validate.cpp)

# Mandlebrot
add_native_target(mandlebrot programs/mandlebrot
    programs/mandlebrot/main.cu programs/mandlebrot/bmp.cpp)

add_persistent_target(mandlebrot_persistent programs/mandlebrot_persistent
    programs/mandlebrot_persistent/main.cu programs/mandlebrot_persistent/bmp.cpp)

# Vector Add
add_native_target(vector_add programs/vector_add
    programs/vector_add/main.cu)
add_persistent_target(vector_add_persistent programs/vector_add_persistent
    programs/vector_add_persistent/main.cu)

# Saxpy
add_native_target(saxpy programs/saxpy
    programs/saxpy/saxpy.cu)
add_persistent_target(saxpy_persistent programs/saxpy_persistent
    programs/saxpy_persistent/saxpy.cu)

# Membench
add_native_target(membench programs/membench
    programs/membench/membench.cu)
add_persistent_target(membench_persistent programs/membench_persistent
    programs/membench_persistent/membench.cu)

#add_persistent_target(test programs programs/test.cu)

# conjugateGradientMultiBlockCG
#add_native_target(conjugateGradientMultiBlockCG programs/cuda_samples/conjugateGradientMultiBlockCG
#    programs/cuda_samples/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu)

#add_persistent_target(conjugateGradientMultiBlockCG_persistent programs/cuda_samples/conjugateGradientMultiBlockCG_persistent
#    programs/cuda_samples/conjugateGradientMultiBlockCG_persistent/conjugateGradientMultiBlockCG_persistent.cu)


# Benchmarks
# Gausssian
add_persistent_target(rodinia_gaussian benchmarks/rodinia/cuda/gaussian
    benchmarks/rodinia/cuda/gaussian/gaussian.cu)

# CFD (Computational Fluid Dynamics)
add_persistent_target(rodinia_cfd benchmarks/rodinia/cuda/cfd
    benchmarks/rodinia/cuda/cfd/euler3d.cu)

# NN (Nearest Neighbours)
add_persistent_target(rodinia_nn benchmarks/rodinia/cuda/nn
    benchmarks/rodinia/cuda/nn/nn_cuda.cu)

# Fast Walsh Transform
add_persistent_target(cudaSDK_fwt benchmarks/cudaSDK/fastWalshTransform
    benchmarks/cudaSDK/fastWalshTransform/fastWalshTransform.cu
    benchmarks/cudaSDK/fastWalshTransform/fastWalshTransform_gold.cpp
    benchmarks/cudaSDK/fastWalshTransform/fastWalshTransform_kernel.cuh)

# Scalar Product
add_persistent_target(cudaSDK_sp benchmarks/cudaSDK/scalarProd
    benchmarks/cudaSDK/scalarProd/scalarProd.cu
    benchmarks/cudaSDK/scalarProd/scalarProd_cpu.cpp
    benchmarks/cudaSDK/scalarProd/scalarProd_kernel.cuh)

# Vector Add
add_persistent_target(cudaSDK_va benchmarks/cudaSDK/vectorAdd
    benchmarks/cudaSDK/vectorAdd/vectorAdd.cu)

# Matrix Multiplication with Shared mem
add_persistent_target(cudaSDK_mms benchmarks/cudaSDK/matrixMulShared
    benchmarks/cudaSDK/matrixMulShared/matrixMul.cu)

# Sorting Networks
add_persistent_target(cudaSDK_sn benchmarks/cudaSDK/sortingNetworks
    benchmarks/cudaSDK/sortingNetworks/main.cpp
    benchmarks/cudaSDK/sortingNetworks/sortingNetworks_validate.cpp
    benchmarks/cudaSDK/sortingNetworks/bitonicSort.cu)

# Reverse engineering code
if(FGPU_TEST_MEM_COLORING_ENABLED)
    # DRAM Bank and Cache Lines Hash Function Finder for GPU
    add_persistent_target(gpu_reverse_engineering reverse_engineering/
        reverse_engineering/reverse_engineering.cpp
        reverse_engineering/gpu.cu
        reverse_engineering/hash_function.cpp)
endif()
