set(TARGET sm_61)

# TODO: Which version do we want?
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fractional_gpu LANGUAGES CXX CUDA DESCRIPTION "Split GPU into Fractional GPUs")

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(include)
include_directories(programs/cuda_samples/common/inc)

string(APPEND CMAKE_CUDA_FLAGS " -arch=${TARGET}")

# Add programs to be built using persistent threads model
function(add_persistent_target target include_dir)

    add_executable(${target} persistent/persistent.cu ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} cuda)
    target_include_directories(${target}  PUBLIC ${include_dir})

endfunction(add_persistent_target)

#server
add_persistent_target(server programs programs/server.cu)

# Dummy
add_persistent_target(dummy_persistent programs/dummy_persistent
    programs/dummy_persistent/dummy_persistent.cu)

# Matrix Multiplication
add_persistent_target(matrixMul_persistent programs/cuda_samples/matrixMul_persistent
    programs/cuda_samples/matrixMul_persistent/matrixMul_persistent.cu)

# Sorting Algos
add_persistent_target(sortingNetworks_persistent programs/cuda_sample/sortingNetworks_persistent
    programs/cuda_samples/sortingNetworks_persistent/main.cpp
    programs/cuda_samples/sortingNetworks_persistent/bitonicSort.cu
    programs/cuda_samples/sortingNetworks_persistent/oddEvenMergeSort.cu
    programs/cuda_samples/sortingNetworks_persistent/sortingNetworks_validate.cpp)
