#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <common.h>
#include <fractional_gpu.h>
#include <fractional_gpu_cuda.cuh>

#define N (1024*1024*2)
#define THREADS_PER_BLOCK 1024

void serial_add(double2 *a, double2 *b, double2 *c, int n)
{
    for(int index=0;index<n;index++)
    {
        c[index].x = a[index].x*a[index].x + b[index].x*b[index].x;
        c[index].y = a[index].y*a[index].y + b[index].y*b[index].y;
    }
}

__global__
FGPU_DEFINE_KERNEL(vector_add, double2 *a, double2 *b, double2 *c, int n)
{
    fgpu_dev_ctx_t *ctx;
    uint3 _blockIdx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        int index = _blockIdx.x * blockDim.x + threadIdx.x;
        if (index < n) {
            double2 a_data = FGPU_COLOR_LOAD(ctx, &a[index]);
            double2 b_data = FGPU_COLOR_LOAD(ctx, &b[index]);
            double2 c_data = {a_data.x * a_data.x + b_data.x * b_data.x, a_data.y * a_data.y + b_data.y * b_data.y};
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

int runTest(void)
{
    double2 *h_a, *h_b, *c_cpu, *h_c_gpu;
    double2 *d_a, *d_b, *d_c_gpu;
    int size = N * sizeof( double2 );
    int ret;

    ret = fgpu_memory_allocate((void **)&h_a, size);
    if (ret < 0)
        return ret;
    
    ret = fgpu_memory_get_device_pointer((void **)&d_a, h_a);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_allocate((void **)&h_b, size);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_get_device_pointer((void **)&d_b, h_b);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_allocate((void **)&h_c_gpu, size);
    if (ret < 0)
        return ret;

    ret = fgpu_memory_get_device_pointer((void **)&d_c_gpu, h_c_gpu);
    if (ret < 0)
        return ret;

    c_cpu = (double2 *)malloc( size );
    if (c_cpu == NULL)
        return -1;

    for( int i = 0; i < N; i++ )
    {
        h_a[i].x = h_b[i].x = i;
        h_a[i].y = h_b[i].y = i + 1;

        c_cpu[i].x = h_c_gpu[i].x = 0;
        c_cpu[i].y = h_c_gpu[i].y = 0;
    }

    serial_add(h_a, h_b, c_cpu, N);

    dim3 threads(THREADS_PER_BLOCK, 1);
    dim3 grid((N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, 1);

    ret = fgpu_memory_prefetch_to_device_async(h_a, size);
    if (ret < 0)
        return ret;
    
    ret = fgpu_memory_prefetch_to_device_async(h_b, size);
    if (ret < 0)
        return ret;
    
    ret = fgpu_memory_prefetch_to_device_async(h_c_gpu, size);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    ret = FGPU_LAUNCH_KERNEL(grid, threads, 0, vector_add, d_a, d_b, d_c_gpu, N );
    if (ret < 0)
        return ret;

	ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    ret = fgpu_memory_prefetch_from_device_async(h_c_gpu, size);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    compare(c_cpu, h_c_gpu, N);

    ret = fgpu_memory_prefetch_to_device_async(h_c_gpu, size);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;


    // Execute the kernel
    int nIter = 132000;
    double start, total;
    pstats_t stats;

    // Warmup
    for (int i = 0; i < nIter; i++) {
        start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(grid, threads, 0, vector_add, d_a, d_b, d_c_gpu, N );
        if (ret < 0)
            return ret;

	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;

        total = dtime_usec(start);
        printf("Time:%f\n", total);
    }

    // Measurements
    pstats_init(&stats);
    for (int i = 0; i < nIter; i++) {
        double sub_start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(grid, threads, 0, vector_add, d_a, d_b, d_c_gpu, N );
        if (ret < 0)
            return ret;

	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;
        pstats_add_observation(&stats, dtime_usec(sub_start));
    }
    
    pstats_print(&stats);

    // Ending
    for (int i = 0; i < nIter; i++) {
        ret = FGPU_LAUNCH_KERNEL(grid, threads, 0, vector_add, d_a, d_b, d_c_gpu, N );
        if (ret < 0)
            return ret;

	    ret = fgpu_color_stream_synchronize();
    	if (ret < 0)
        	return ret;
    }

    fgpu_memory_free(h_a);
    fgpu_memory_free(h_b);
    fgpu_memory_free(h_c_gpu);
    free(c_cpu);

    return 0;
}

int main(int argc, char **argv)
{
    int ret, color;

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

    ret = runTest();
    if (ret < 0)
        return ret;

    fgpu_deinit();

}
