set(TARGET sm_61)

# TODO: Which version do we want?
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fractional_gpu LANGUAGES CXX CUDA DESCRIPTION "Split GPU into Fractional GPUs")

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_BUILD_TYPE Release)

include_directories(include)

string(APPEND CMAKE_CUDA_FLAGS " -arch=${TARGET}")

#server
add_executable(server persistent/persistent.cu
		       programs/server.cu)

target_compile_features(server PUBLIC cxx_std_11)

target_link_libraries(server cuda)

# Matrix Multiplication
add_executable(matrixMul persistent/persistent.cu
		       programs/matrixMul.cu)

target_compile_features(matrixMul PUBLIC cxx_std_11)

target_link_libraries(matrixMul cuda)

# Dummy
add_executable(dummy persistent/persistent.cu
		       programs/dummy.cu)

target_compile_features(dummy PUBLIC cxx_std_11)

target_link_libraries(dummy cuda)
