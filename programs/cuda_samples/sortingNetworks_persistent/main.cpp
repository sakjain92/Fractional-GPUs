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

#include <assert.h>

#include <fractional_gpu.hpp>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>
////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int ret, color;
    uint arrayLength;
    uint threadCount;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);


    printf("%s Starting...\n\n", argv[0]);

    printf("Starting up CUDA context...\n");

    uint *h_InputKey, *h_InputVal,    *h_OutputKeyGPU,    *h_OutputValGPU;
    uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;
    StopWatchInterface *hTimer = NULL;

    const uint             N = 1048576;
    const uint           DIR = 0;
    const uint     numValues = 65536;
    uint numIterations = 1;

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
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
    assert(ret == 0);
    ret = fgpu_memory_allocate((void **)&d_InputVal,  N * sizeof(uint));
    assert(ret == 0);
    ret = fgpu_memory_allocate((void **)&d_OutputKey, N * sizeof(uint));
    assert(ret == 0);
    ret = fgpu_memory_allocate((void **)&d_OutputVal, N * sizeof(uint));
    assert(ret == 0);
    
    ret = fgpu_memory_copy_async(d_InputKey, h_InputKey, N * sizeof(uint), FGPU_COPY_CPU_TO_GPU);
    assert(ret == 0);
    ret = fgpu_memory_copy_async(d_InputVal, h_InputVal, N * sizeof(uint), FGPU_COPY_CPU_TO_GPU);
    assert(ret == 0);

    int flag = 1;
    printf("Running GPU bitonic sort (%u identical iterations)...\n\n", numIterations);

    for (arrayLength = 64; arrayLength <= N; arrayLength *= 2)
    {
        printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
        ret = fgpu_color_stream_synchronize();
        assert(ret == 0);

        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        for (uint i = 0; i < numIterations; i++)
            threadCount = bitonicSort(
                              d_OutputKey,
                              d_OutputVal,
                              d_InputKey,
                              d_InputVal,
                              N / arrayLength,
                              arrayLength,
                              DIR
                          );

        ret = fgpu_color_stream_synchronize();
        assert(ret == 0);

        sdkStopTimer(&hTimer);
        printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer) / numIterations);

        if (arrayLength == N)
        {
            double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer) / numIterations;
            printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);
        }

        printf("\nValidating the results...\n");
        printf("...reading back GPU results\n");
        ret = fgpu_memory_copy_async(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), FGPU_COPY_GPU_TO_CPU);
        assert(ret == 0);
        ret = fgpu_memory_copy_async(h_OutputValGPU, d_OutputVal, N * sizeof(uint), FGPU_COPY_GPU_TO_CPU);
        assert(ret == 0);
        ret = fgpu_color_stream_synchronize();
        assert(ret == 0);

        int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
        int valuesFlag = validateValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);
        flag = flag && keysFlag && valuesFlag;

        printf("\n");
    }

    printf("Running in Loop\n");
    arrayLength = N;
    printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
    ret = fgpu_color_stream_synchronize();
    assert(ret == 0);


    double start, total;
    pstats_t stats;
    //Warmup
    for (uint i = 0; i < num_iterations; i++) {
        start = dtime_usec(0);
        threadCount = bitonicSort(
                d_OutputKey,
                d_OutputVal,
                d_InputKey,
                d_InputVal,
                N / arrayLength,
                arrayLength,
                DIR
                );
        ret = fgpu_color_stream_synchronize();
        if (ret < 0)
            return ret;
        total = dtime_usec(start);
        dprintf("Wamup:Array Length: %d, Time:%f us\n", N, total);
    }


    // Measurements
    pstats_init(&stats);
    start = dtime_usec(0);
    for (uint i = 0; i < num_iterations; i++) {
        double sub_start = dtime_usec(0);
        threadCount = bitonicSort(
                d_OutputKey,
                d_OutputVal,
                d_InputKey,
                d_InputVal,
                N / arrayLength,
                arrayLength,
                DIR
                );
        ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;
        pstats_add_observation(&stats, dtime_usec(sub_start));
    }

    total = dtime_usec(start);

    pstats_print(&stats);
    printf("Average time: %f ms\n\n", total / numIterations / 1000);

    double dTimeSecs = 1.0e-6 * total / numIterations;
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
            (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    fgpu_memory_free(d_OutputVal);
    fgpu_memory_free(d_OutputKey);
    fgpu_memory_free(d_InputVal);
    fgpu_memory_free(d_InputKey);
    free(h_OutputValGPU);
    free(h_OutputKeyGPU);
    free(h_InputVal);
    free(h_InputKey);

    test_deinitialize();

    if (!flag)
        printf("Verification Fail\n");
    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
