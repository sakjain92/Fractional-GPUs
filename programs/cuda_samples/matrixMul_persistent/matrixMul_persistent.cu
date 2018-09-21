#include <assert.h>
#include <errno.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> 
FGPU_DEFINE_KERNEL(matrixMulCUDA, float *C, float *A, float *B, int wA, int wB)
{
    fgpu_dev_ctx_t *ctx;
    dim3 _blockIdx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        // Block index
        int bx = _blockIdx.x;
        int by = _blockIdx.y;

        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd   = aBegin + wA - 1;

       // Step size used to iterate through the sub-matrices of A
        int aStep  = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep  = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
        {

            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            As[ty][tx] = FGPU_COLOR_LOAD(ctx, &A[a + wA * ty + tx]);
            Bs[ty][tx] = FGPU_COLOR_LOAD(ctx, &B[b + wB * ty + tx]);

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
#pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k)
            {
                Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        FGPU_COLOR_STORE(ctx, &C[c + wB * ty + tx], Csub);
    } FGPU_FOR_EACH_END;
}



void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}


/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int num_iterations)
{
    int block_size = 32;
    dim3 dimsA(10 * block_size, 10 * block_size, 1);
    dim3 dimsB(20 * block_size, 10 * block_size, 1);

    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }


    int ret;
    
    ret = fgpu_memory_allocate((void **) &d_A, mem_size_A);

    if (ret < 0)
    {
        printf("fgpu_memory_allocate h_A returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    ret = fgpu_memory_allocate((void **) &d_B, mem_size_B);

    if (ret < 0)
    {
        printf("fgpu_memory_allocate h_B returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    ret = fgpu_memory_allocate((void **) &d_C, mem_size_C);

    if (ret < 0)
    {
        printf("fgpu_memory_allocate h_C returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    ret = fgpu_memory_copy_async(d_A, h_A, mem_size_A, FGPU_COPY_CPU_TO_GPU);
    if (ret < 0) {
        printf("fgpu_memory_copy_async h_A returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    ret = fgpu_memory_copy_async(d_B, h_B, mem_size_B, FGPU_COPY_CPU_TO_GPU);
    if (ret < 0) {
         printf("fgpu_memory_copy_async h_A returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
    {
        printf("fgpu_color_stream_synchronize returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }


    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Execute the kernel
    int nIter = num_iterations;

    double start, total;
    pstats_t stats;

    // Init
    for (int j = 0; j < nIter; j++)
    {
        start = dtime_usec(0);

        if (block_size == 16)
        {
            ret = FGPU_LAUNCH_KERNEL(matrixMulCUDA<16>, grid, threads, 0, d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            ret = FGPU_LAUNCH_KERNEL(matrixMulCUDA<32>, grid, threads, 0, d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        if (ret < 0)
            return ret;
	
	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;

        total = dtime_usec(start);
        dprintf("Time:%f, BlockSize:%d, dimA.x:%d, dimA.y:%d, dimB.x:%d, dimB.y:%d\n", total, block_size, dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    }

    pstats_init(&stats);
    start = dtime_usec(0);
    for (int j = 0; j < nIter; j++)
    {
        double sub_start = dtime_usec(0);
        if (block_size == 16)
        {
            ret = FGPU_LAUNCH_KERNEL(matrixMulCUDA<16>, grid, threads, 0, d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            ret = FGPU_LAUNCH_KERNEL(matrixMulCUDA<32>, grid, threads, 0, d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        if (ret < 0)
            return ret;
        pstats_add_observation(&stats, dtime_usec(sub_start));
    }

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    total = dtime_usec(start);
    pstats_print(&stats);

    // Compute and print the performance
    double msecPerMatrixMul = total / nIter / 1000;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.6f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);


    // Copy result from device to host
    ret = fgpu_memory_copy_async(h_C, d_C, mem_size_C, FGPU_COPY_GPU_TO_CPU);
    if (ret < 0) {
         printf("fgpu_memory_copy_async h_A returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE);
    }

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
    {
        printf("fgpu_color_stream_synchronize returned error %s (code %d), line(%d)\n", strerror(errno), ret, __LINE__);
        exit(EXIT_FAILURE        );
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    fgpu_memory_free(d_A);
    fgpu_memory_free(d_B);
    fgpu_memory_free(d_C);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

int main(int argc, char *argv[])
{
    int ret;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);


    ret = matrixMultiply(num_iterations);
    if (ret < 0)
        return ret;

    test_deinitialize();

    return 0;
}
