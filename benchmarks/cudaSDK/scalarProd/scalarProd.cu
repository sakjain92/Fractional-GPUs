/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample calculates scalar products of a
 * given set of input vector pairs
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int vectorN,
    int elementN
);



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cuh"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
const int VECTOR_N = 256;
//Number of elements per vector; arbitrary,
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const int ELEMENT_N = 4096;
//Total number of data elements
const int    DATA_N = VECTOR_N * ELEMENT_N;

const int   DATA_SZ = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N  * sizeof(float);



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B, *d_C;
    double delta, ref, sum_delta, sum_ref, L1norm;
    int i, j, ret;

    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    printf("%s Starting...\n\n", argv[0]);


    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_A     = (float *)malloc(DATA_SZ);
    h_B     = (float *)malloc(DATA_SZ);
    h_C_CPU = (float *)malloc(RESULT_SZ);
    h_C_GPU = (float *)malloc(RESULT_SZ);

    printf("...allocating GPU memory.\n");
    ret = fgpu_memory_allocate((void **)&d_A, DATA_SZ);
    if (ret < 0)
        exit(EXIT_FAILURE);

    ret = fgpu_memory_allocate((void **)&d_B, DATA_SZ);
    if (ret < 0)
        exit(EXIT_FAILURE);

    ret = fgpu_memory_allocate((void **)&d_C, RESULT_SZ);
    if (ret < 0)
        exit(EXIT_FAILURE);

    printf("...generating input data in CPU mem.\n");
    srand(123);

    //Generating input data on CPU
    for (i = 0; i < DATA_N; i++)
    {
        h_A[i] = RandFloat(0.0f, 1.0f);
        h_B[i] = RandFloat(0.0f, 1.0f);
    }

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

                ret = fgpu_memory_copy_async(d_A, h_A, DATA_SZ, FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_memory_copy_async(d_B, h_B, DATA_SZ, FGPU_COPY_CPU_TO_GPU);
                if (ret < 0)
                    exit(EXIT_FAILURE);

                ret = fgpu_color_stream_synchronize();
                if (ret < 0)
                    exit(EXIT_FAILURE);
            }

            double kernel_start = dtime_usec(0);

            ret = FGPU_LAUNCH_KERNEL(scalarProdGPU, 128, 256, 0, d_C, d_A, d_B, VECTOR_N, ELEMENT_N);
            if (ret < 0)
                exit(EXIT_FAILURE);

            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                exit(EXIT_FAILURE);

            if (!is_warmup)
                pstats_add_observation(&kernel_stats, dtime_usec(kernel_start));
            
            if (!test_execute_just_kernel() || j == 0) {

                ret = fgpu_memory_copy_async(h_C_GPU, d_C, RESULT_SZ, FGPU_COPY_GPU_TO_CPU);
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

    printf("Checking GPU results...\n");
    printf("..running CPU scalar product calculation\n");
    scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);

    printf("...comparing the results\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;

    for (i = 0; i < VECTOR_N; i++)
    {
        delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
        ref   = h_C_CPU[i];
        sum_delta += delta;
        sum_ref   += ref;
    }

    L1norm = sum_delta / sum_ref;

    if (!test_execute_just_kernel()) {
        printf("Overall Stats\n");
        pstats_print(&stats);
    }

    printf("Kernel Stats\n");
    pstats_print(&kernel_stats);
    
    printf("Shutting down...\n");
    fgpu_memory_free(d_C);
    fgpu_memory_free(d_B);
    fgpu_memory_free(d_A);
    free(h_C_GPU);
    free(h_C_CPU);
    free(h_B);
    free(h_A);

    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
