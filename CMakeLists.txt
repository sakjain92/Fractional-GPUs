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
# First arg is target name
# Second arg is include directory
# Rest of the args are the files to be compiled into target
function(add_persistent_target target include_dir)

    add_executable(${target} persistent/persistent.cu ${ARGN})
    target_compile_features(${target} PUBLIC cxx_std_11)
    target_link_libraries(${target} cuda)
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

