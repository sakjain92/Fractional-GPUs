/* This program checks the memory bandwidth */
#include <stdio.h>
#include <assert.h>

#include <common.h>
#include <fractional_gpu.h>
#include <fractional_gpu_cuda.cuh>

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

void transfer_managed_one(void *ptr, size_t size, enum fgpu_memory_copy_type kind)
{
    int ret;

    if (kind == FGPU_COPY_CPU_TO_GPU)
        ret = fgpu_memory_prefetch_to_device_async(ptr, size);
    else
        ret = fgpu_memory_prefetch_from_device_async(ptr, size);

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
    int color;

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
        transfer_managed_one(d_x, N, FGPU_COPY_CPU_TO_GPU);
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

        start = dtime_usec(0);
        transfer_managed_one(d_x, N, FGPU_COPY_CPU_TO_GPU);
        printf("MemprefetchToDevice: Bandwidth:%f GB/s\n", bandwidth(dtime_usec(start)));

        printf("\n\n");
    }

    free(x);
    cudaFreeHost(h_x);
    fgpu_memory_free(d_x);

}
