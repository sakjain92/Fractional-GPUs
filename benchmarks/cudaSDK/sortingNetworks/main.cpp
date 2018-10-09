/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This sample implements bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks.
 * While generally subefficient on large sequences
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 *
 * Victor Podlozhnyuk, 07/09/2009
 */

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_timer.h>

#include "sortingNetworks_common.h"

#include <fractional_gpu.hpp>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int i, j, ret;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    printf("%s Starting...\n\n", argv[0]);

    uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
    uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;

    const uint             N = 1048576;
    const uint           DIR = 0;
    const uint     numValues = 65536;

    printf("Allocating and initializing host arrays...\n\n");
    h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_InputVal     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(N * sizeof(uint));
    srand(2001);

    for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand() % numValues;
        h_InputVal[i] = i;
    }

    printf("Allocating and initializing CUDA arrays...\n\n");
    ret = fgpu_memory_allocate((void **)&d_InputKey,  N * sizeof(uint));
    if (ret < 0)
        exit(EXIT_FAILURE);
    ret = fgpu_memory_allocate((void **)&d_InputVal,  N * sizeof(uint));
    if (ret < 0)
        exit(EXIT_FAILURE);
    ret = fgpu_memory_allocate((void **)&d_OutputKey,  N * sizeof(uint));
    if (ret < 0)
        exit(EXIT_FAILURE);
    ret = fgpu_memory_allocate((void **)&d_OutputVal,  N * sizeof(uint));
    if (ret < 0)
        exit(EXIT_FAILURE);

    int flag = 1;

    pstats_t stats;
    pstats_t kernel_stats;

    pstats_init(&stats);
    pstats_init(&kernel_stats);

    uint arrayLength = N;

    for (i = 0; i < 2; i++) {
        
        bool is_warmup = (i == 0);

        for (j = 0; j < num_iterations; j++) {

            double start;

            if (!test_execute_just_kernel() || j == 0) {

                start = dtime_usec(0);

                ret = fgpu_memory_copy_async(d_InputKey, h_InputKey, N * sizeof(uint), FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_memory_copy_async(d_InputVal, h_InputVal, N * sizeof(uint), FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_color_stream_synchronize();
                if (ret < 0)
                    exit(EXIT_FAILURE);
            }

            double kernel_start = dtime_usec(0);

            bitonicSort(d_OutputKey, d_OutputVal, d_InputKey, d_InputVal,
                        N / arrayLength, arrayLength, DIR);

            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                exit(EXIT_FAILURE);

            if (!is_warmup)
                pstats_add_observation(&kernel_stats, dtime_usec(kernel_start));

            if (!test_execute_just_kernel() || j == 0) {

                ret = fgpu_memory_copy_async(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint),  FGPU_COPY_GPU_TO_CPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_memory_copy_async(h_OutputValGPU, d_OutputVal, N * sizeof(uint),  FGPU_COPY_GPU_TO_CPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_color_stream_synchronize();
                if (ret < 0)
                    exit(EXIT_FAILURE);

                if (!is_warmup)
                    pstats_add_observation(&stats, dtime_usec(start));
            }   

        }
    }
        
    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
    int valuesFlag = validateValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);
    flag = flag && keysFlag && valuesFlag;

    if (!test_execute_just_kernel()) {
        printf("Overall Stats\n");
        pstats_print(&stats);
    }

    printf("Kernel Stats\n");
    pstats_print(&kernel_stats);

    printf("Shutting down...\n");
    fgpu_memory_free(d_OutputVal);
    fgpu_memory_free(d_OutputKey);
    fgpu_memory_free(d_InputVal);
    fgpu_memory_free(d_InputKey);
    free(h_OutputValGPU);
    free(h_OutputKeyGPU);
    free(h_InputVal);
    free(h_InputKey);

    if (!flag)
        printf("Verification Fail\n");

    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
