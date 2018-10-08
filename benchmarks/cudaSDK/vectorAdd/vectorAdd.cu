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
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 *
 * Modified the code to use double2 instead of float to increase memory
 * bandwidth used. Also the size of the array has been increases so that the
 * array doesn't get's cached in the L2.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__
FGPU_DEFINE_KERNEL(vectorAdd, const double2 *A, const double2 *B, double2 *C, int numElements)
{
    dim3 _blockIdx;
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        int i = blockDim.x * _blockIdx.x + threadIdx.x;

        if (i < numElements) {
            double2 a_data = A[i];
            double2 b_data = B[i];
            C[i].x = a_data.x + b_data.x;
            C[i].y = a_data.y + b_data.y;
        }
    }   
}

/**
 * Host main routine
 */
int main(int argc, char **argv)
{
    int i, j, ret, num_iterations;

    test_initialize(argc, argv, &num_iterations);

    // Print the vector length to be used, and compute its size
    int numElements = 1024 * 1024 * 2;
    size_t size = numElements * sizeof(double2);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    double2 *h_A = (double2 *)malloc(size);

    // Allocate the host input vector B
    double2 *h_B = (double2 *)malloc(size);

    // Allocate the host output vector C
    double2 *h_C = (double2 *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i].x = rand()/(double)RAND_MAX;
        h_A[i].y = rand()/(double)RAND_MAX;
        h_B[i].x = rand()/(double)RAND_MAX;
        h_B[i].y = rand()/(double)RAND_MAX;
    }

    // Allocate the device input vector A
    double2 *d_A = NULL;
    ret = fgpu_memory_allocate((void **)&d_A, size);
    if (ret < 0)
        exit(EXIT_FAILURE);


    // Allocate the device input vector B
    double2 *d_B = NULL;
    ret = fgpu_memory_allocate((void **)&d_B, size);
    if (ret < 0)
        exit(EXIT_FAILURE);

    // Allocate the device output vector C
    double2 *d_C = NULL;
    ret = fgpu_memory_allocate((void **)&d_C, size);
    if (ret < 0)
        exit(EXIT_FAILURE);

    pstats_t stats;
    pstats_t kernel_stats;

    pstats_init(&stats);
    pstats_init(&kernel_stats);

    for (i = 0; i < 2; i++) {

        bool is_warmup = (i == 0);

        for (j = 0; j < num_iterations; j++) {

            double start;

            if (!test_execute_just_kernel() || j == 0) {

                start = dtime_usec(0);

                // Copy the host input vectors A and B in host memory to the device input vectors in
                // device memory
                ret = fgpu_memory_copy_async(d_A, h_A, size, FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_memory_copy_async(d_B, h_B, size, FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_color_stream_synchronize();
                if (ret < 0)
                    exit(EXIT_FAILURE);
            }

            double kernel_start = dtime_usec(0);

            // Launch the Vector Add CUDA Kernel
            int threadsPerBlock = 256;
            int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
            
            ret = FGPU_LAUNCH_KERNEL(vectorAdd, blocksPerGrid, threadsPerBlock, 0, d_A, d_B, d_C, numElements);
            if (ret < 0)
                exit(EXIT_FAILURE);

            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                exit(EXIT_FAILURE);

            if (!is_warmup)
                pstats_add_observation(&kernel_stats, dtime_usec(kernel_start));
            
            if (!test_execute_just_kernel() || j == 0) {

                ret = fgpu_memory_copy_async(h_C, d_C, size, FGPU_COPY_GPU_TO_CPU);
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

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i].x + h_B[i].x - h_C[i].x) > 1e-5 || 
                fabs(h_A[i].y + h_B[i].y - h_C[i].y) > 1e-5 )
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    if (!test_execute_just_kernel()) {
        printf("Overall Stats\n");
        pstats_print(&stats);
    }

    printf("Kernel Stats\n");
    pstats_print(&kernel_stats);

    // Free device global memory
    fgpu_memory_free(d_A);
    fgpu_memory_free(d_B);
    fgpu_memory_free(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

