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

#include <common.h>
#include <fractional_gpu.h>
////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int ret, color;
    cudaError_t error;
    uint arrayLength;
    uint threadCount;

    printf("%s Starting...\n\n", argv[0]);

    printf("Starting up CUDA context...\n");

    if (argc != 2) {
        fprintf(stderr, "Insufficient number of arguments\n");
        exit(-1);
    }

    color = atoi(argv[1]);

    printf("Color selected:%d\n", color);

    ret = fgpu_init();
    if (ret < 0)
        return ret;

    ret = fgpu_set_color_prop(color, 128 * 1024 * 1024);
    if (ret < 0)
         return ret;

    uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;
    StopWatchInterface *hTimer = NULL;

    const uint             N = 1048576;
    const uint           DIR = 0;
    const uint     numValues = 65536;
    uint numIterations = 1;

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    srand(2001);

    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMallocManaged((void **)&d_InputKey,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMallocManaged((void **)&d_InputVal,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMallocManaged((void **)&d_OutputKey, N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMallocManaged((void **)&d_OutputVal, N * sizeof(uint));
    checkCudaErrors(error);

    for (uint i = 0; i < N; i++)
    {
        d_InputKey[i] = rand() % numValues;
        d_InputVal[i] = i;
    }

    error = cudaMemPrefetchAsync(d_InputKey,  N * sizeof(uint), 0);
    checkCudaErrors(error);
    error = cudaMemPrefetchAsync(d_InputVal,  N * sizeof(uint), 0);
    checkCudaErrors(error);
    error = cudaMemPrefetchAsync(d_OutputKey, N * sizeof(uint), 0);
    checkCudaErrors(error);
    error = cudaMemPrefetchAsync(d_OutputVal, N * sizeof(uint), 0);
    checkCudaErrors(error);
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);

    int flag = 1;
    printf("Running GPU bitonic sort (%u identical iterations)...\n\n", numIterations);

    for (arrayLength = 64; arrayLength <= N; arrayLength *= 2)
    {
        printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

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

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

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
        error = cudaMemPrefetchAsync(d_InputKey, N * sizeof(uint), CU_DEVICE_CPU);
        checkCudaErrors(error);
        error = cudaMemPrefetchAsync(d_OutputKey, N * sizeof(uint), CU_DEVICE_CPU);
        checkCudaErrors(error);
        error = cudaMemPrefetchAsync(d_OutputVal, N * sizeof(uint), CU_DEVICE_CPU);
        checkCudaErrors(error);
        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        int keysFlag = validateSortedKeys(d_OutputKey, d_InputKey, N / arrayLength, arrayLength, numValues, DIR);
        int valuesFlag = validateValues(d_OutputKey, d_OutputVal, d_InputKey, N / arrayLength, arrayLength);
        flag = flag && keysFlag && valuesFlag;

        printf("\n");
    }

    printf("Running in Loop\n");
    arrayLength = N;
    numIterations = 10000;
    printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);


    double start, total;
    pstats_t stats;
    //Warmup
    for (uint i = 0; i < numIterations; i++) {
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
        ret = gpuErrCheck(fgpu_color_stream_synchronize());
        if (ret < 0)
            return ret;
        total = dtime_usec(start);
        printf("Wamup:Array Length: %d, Time:%f us\n", N, total);
    }

    // Measurements
    pstats_init(&stats);
    start = dtime_usec(0);
    for (uint i = 0; i < numIterations; i++) {
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
        ret = gpuErrCheck(fgpu_color_stream_synchronize());
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

    // Termination - To overlap with others
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
     ret = gpuErrCheck(fgpu_color_stream_synchronize());
    if (ret < 0)
        return ret;

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    cudaFree(d_OutputVal);
    cudaFree(d_OutputKey);
    cudaFree(d_InputVal);
    cudaFree(d_InputKey);

    fgpu_deinit();

    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
