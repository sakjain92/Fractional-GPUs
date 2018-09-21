/* This program checks the memory bandwidth */
#include <stdio.h>
#include <assert.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

#define N                   (128 * 1024 * 1024)
#define PAGE_SIZE           (4 * 1024)

void transfer_one(void *dst, void *src, size_t size, enum fgpu_memory_copy_type kind)
{
    int ret = fgpu_memory_copy_async(dst, src, size, kind);
    if (ret == 0)
        exit(-1);
    fgpu_color_stream_synchronize();
     if (ret == 0)
        exit(-1);
}

double bandwidth(double time)
{
    return ((double)N) / time / 1000;
}

int main(int argc, char *argv[])
{
    char *x, *h_x, *d_x;
    double start;
    int ret;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    x = (char *)malloc(N*sizeof(char));
    assert(x);

    gpuErrAssert(cudaHostAlloc(&h_x, N*sizeof(char), cudaHostAllocDefault));

    ret = fgpu_memory_allocate((void **)&d_x, N);
    if (ret < 0)
        return ret;

    // Warmup
    printf("Doing Warmup\n");
    for (int i = 0; i < 3; i++) {

        /* Test one way transfer */
        transfer_one(d_x, x, N, FGPU_COPY_CPU_TO_GPU);
        transfer_one(h_x, d_x, N, FGPU_COPY_GPU_TO_CPU);
    }

    printf("Warmup done\n");

    for (int i = 0; i < 3; i++) {
    
        /* Test one way transfer */
        start = dtime_usec(0);
        transfer_one(d_x, x, N, FGPU_COPY_CPU_TO_GPU);
        printf("HostToDevice: Bandwidth:%f GB/s\n", bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(x, d_x, N, FGPU_COPY_GPU_TO_CPU);
        printf("DeviceToHost: Bandwidth:%f GB/s\n", bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(d_x, h_x, N, FGPU_COPY_CPU_TO_GPU);
        printf("HostToDevicePinned: Bandwidth:%f GB/s\n", bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(h_x, d_x, N, FGPU_COPY_GPU_TO_CPU);
        printf("DeviceToHostPinned: Bandwidth:%f GB/s\n", bandwidth(dtime_usec(start)));


        // Modify data;
        for (int i = 0; i < N; i += PAGE_SIZE)
            d_x[i] = 0;


        printf("\n\n");
    }

    free(x);
    cudaFreeHost(h_x);
    fgpu_memory_free(d_x);

    test_deinitialize();
}
