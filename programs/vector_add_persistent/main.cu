#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

#define N (1024*1024*2)
#define THREADS_PER_BLOCK 1024

void serial_add(double2 *a, double2 *b, double2 *c, int n)
{
    for(int index=0;index<n;index++)
    {
        c[index].x = a[index].x + b[index].x;
        c[index].y = a[index].y + b[index].y;
    }
}

__global__
FGPU_DEFINE_KERNEL(vector_add, double2 *a, double2 *b, double2 *c, int n)
{
    fgpu_dev_ctx_t *ctx;
    dim3 _blockIdx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        int index = _blockIdx.x * blockDim.x + threadIdx.x;
        if (index < n) {
            double2 a_data = FGPU_COLOR_LOAD(ctx, &a[index]);
            double2 b_data = FGPU_COLOR_LOAD(ctx, &b[index]);
            double2 c_data = {a_data.x + b_data.x, a_data.y + b_data.y};
            FGPU_COLOR_STORE(ctx, &c[index], c_data);
        }   
    }   
}

void compare(double2 *cpu, double2 *gpu, int n)
{
    for (int i = 0; i < n; i++)
        if (cpu[i].x != gpu[i].x || cpu[i].y != gpu[i].y) {
            printf("ERROR: Comparision failed\n");
            return;
        }

    printf("Comparision success\n");
}

int runTest(int num_iterations)
{
    double2 *a, *b, *c_cpu, *c_gpu;
    double2 *d_a, *d_b, *d_c;
    int size = N * sizeof( double2 );
    int ret;

    a = (double2 *)malloc( size );
    b = (double2 *)malloc( size );
    c_cpu = (double2 *)malloc( size );
    c_gpu = (double2 *)malloc( size );

    for( int i = 0; i < N; i++ )
    {
        a[i].x = b[i].x = i;
        a[i].y = b[i].y = i + 1;

        c_cpu[i].x = c_gpu[i].x = 0;
        c_cpu[i].y = c_gpu[i].y = 0;
    }

    serial_add(a, b, c_cpu, N);

    ret = fgpu_memory_allocate((void **)&d_a, size);
    if (ret < 0)
        return ret;
    
    ret = fgpu_memory_allocate((void **)&d_b, size);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_allocate((void **)&d_c, size);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_copy_async(d_a, a, size, FGPU_COPY_CPU_TO_GPU);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_copy_async(d_b, b, size, FGPU_COPY_CPU_TO_GPU);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    dim3 threads(THREADS_PER_BLOCK, 1);
    dim3 grid((N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, 1);

    ret = FGPU_LAUNCH_KERNEL(vector_add, grid, threads, 0, d_a, d_b, d_c, N );
    if (ret < 0)
        return ret;

	ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    ret = fgpu_memory_copy_async(c_gpu, d_c, size, FGPU_COPY_GPU_TO_CPU);
    if (ret < 0)
        return ret;


    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    compare(c_cpu, c_gpu, N);


    // Execute the kernel
    pstats_t stats;

    // Warmup
    for (int i = 0; i < num_iterations; i++) {
        double sub_start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(vector_add, grid, threads, 0, d_a, d_b, d_c, N );
        if (ret < 0)
            return ret;

	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;

        dprintf("Time:%f\n", dtime_usec(sub_start));
    }

    // Measurements
    pstats_init(&stats);
    for (int i = 0; i < num_iterations; i++) {
        double sub_start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(vector_add, grid, threads, 0, d_a, d_b, d_c, N );
        if (ret < 0)
            return ret;

	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;
        pstats_add_observation(&stats, dtime_usec(sub_start));
    }
    
    pstats_print(&stats);

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    fgpu_memory_free(d_a);
    fgpu_memory_free(d_b);
    fgpu_memory_free(d_c);

    return 0;
}

int main(int argc, char **argv)
{
    int ret;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    ret = runTest(num_iterations);
    if (ret < 0)
        return ret;

    test_deinitialize();
}
