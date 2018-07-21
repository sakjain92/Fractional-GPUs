set(TARGET sm_61)

# TODO: Which version do we want?
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fractional_gpu VERSION 1.0.1 LANGUAGES CXX CUDA DESCRIPTION "Split GPU into Fractional GPUs")

option(FGPU_MEM_COLORING_ENABLED "Enable memory coloring" ON)
option(FGPU_USER_MEM_COLORING_ENABLED "Enable userspace coloring" ON)

if(FGPU_MEM_COLORING_ENABLED)
    string(APPEND CMAKE_CUDA_FLAGS " -DFGPU_MEM_COLORING_ENABLED")
endif()

if(FGPU_USER_MEM_COLORING_ENABLED)
    string(APPEND CMAKE_CUDA_FLAGS " -DFGPU_USER_MEM_COLORING_ENABLED")
endif()

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(include)
include_directories(programs/cuda_samples/common/inc)
include_directories(driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/)
include_directories(driver/NVIDIA-Linux-x86_64-390.48/kernel/common/inc/)

string(APPEND CMAKE_CUDA_FLAGS " -arch=${TARGET}")

# Add programs to be built using persistent threads model
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_persistent_target target include_dir)

    add_executable(${target} persistent/persistent.cu persistent/memory.cu ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} cuda cudart nvidia-ml)
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
add_persistent_target(server programs programs/server.cu)

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

# conjugateGradientMultiBlockCG
#add_native_target(conjugateGradientMultiBlockCG programs/cuda_samples/conjugateGradientMultiBlockCG
#    programs/cuda_samples/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu)

#add_persistent_target(conjugateGradientMultiBlockCG_persistent programs/cuda_samples/conjugateGradientMultiBlockCG_persistent
#    programs/cuda_samples/conjugateGradientMultiBlockCG_persistent/conjugateGradientMultiBlockCG_persistent.cu)

