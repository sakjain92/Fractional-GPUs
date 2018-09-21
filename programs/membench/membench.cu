/* This program checks the memory bandwidth */
#include <stdio.h>
#include <assert.h>

#include <fractional_gpu.hpp>

#include <fractional_gpu_testing.hpp>

#define N                   (128 * 1024 * 1024)
#define PAGE_SIZE           (4 * 1024)
#define LARGE_PAGE_SIZE     (2 * 1024 * 1024)
#define HUGE_PAGE_SIZE      (128 * 1024 * 1024)

size_t page_sizes[] = {PAGE_SIZE, LARGE_PAGE_SIZE, HUGE_PAGE_SIZE, HUGE_PAGE_SIZE, LARGE_PAGE_SIZE, PAGE_SIZE};

void transfer_one(void *dst, void *src, size_t chunk, cudaMemcpyKind kind, cudaStream_t stream)
{
    for (int i = 0; i < N; i += chunk) {
        gpuErrAssert(cudaMemcpyAsync(dst, src, chunk, kind, stream));
        src = (void *)((uintptr_t)src + chunk);
        dst = (void *)((uintptr_t)dst + chunk);
    }
}

void transfer_two(void *dst1, void *src1, void *dst2, void *src2, size_t chunk,
        cudaMemcpyKind kind1, cudaMemcpyKind kind2,
        cudaStream_t stream1, cudaStream_t stream2)
{
    for (int i = 0; i < N; i += chunk) {
        gpuErrAssert(cudaMemcpyAsync(dst1, src1, chunk, kind1, stream1));
        src1 = (void *)((uintptr_t)src1 + chunk);
        dst1 = (void *)((uintptr_t)dst1 + chunk);
        gpuErrAssert(cudaMemcpyAsync(dst2, src2, chunk, kind2, stream2));
        src2 = (void *)((uintptr_t)src2 + chunk);
        dst2 = (void *)((uintptr_t)dst2 + chunk);
    }
}

void transfer_managed_one(void *ptr, size_t chunk, int dstDevice, cudaStream_t stream)
{
    for (int i = 0; i < N; i += chunk) {
        gpuErrAssert(cudaMemPrefetchAsync(ptr, chunk, dstDevice, stream));
        ptr = (void *)((uintptr_t)ptr + chunk);
    }
}
double bandwidth(double time)
{
    return ((double)N) / time / 1000;
}
int main(int argc, char *argv[])
{
    char *x, *y, *h_x, *h_y, *d_x, *d_y, *m_x, *m_y;
    cudaStream_t stream_x;
    cudaStream_t stream_y;
    double start;

    int num_iterations;

    test_initialize(argc, argv, &num_iterations);


    gpuErrAssert(cudaStreamCreate(&stream_x));
    gpuErrAssert(cudaStreamCreate(&stream_y));

    x = (char *)malloc(N*sizeof(char));
    y = (char *)malloc(N*sizeof(char));
    assert(x);
    assert(y);

    gpuErrAssert(cudaHostAlloc(&h_x, N*sizeof(char), cudaHostAllocDefault));
    gpuErrAssert(cudaHostAlloc(&h_y, N*sizeof(char), cudaHostAllocDefault));

    gpuErrAssert(cudaMalloc(&d_x, N*sizeof(char)));
    gpuErrAssert(cudaMalloc(&d_y, N*sizeof(char)));

    gpuErrAssert(cudaMallocManaged(&m_x, N*sizeof(char)));
    gpuErrAssert(cudaMallocManaged(&m_y, N*sizeof(char)));

    // Warmup
    printf("Doing Warmup\n");
    for (int i = 0; i < sizeof(page_sizes) / sizeof(page_sizes[0]); i++) {
        size_t page_size = page_sizes[i];

        /* Test one way transfer */
        transfer_one(d_x, x, page_size, cudaMemcpyHostToDevice, stream_x);
        gpuErrAssert(cudaStreamSynchronize(stream_x));

        transfer_one(y, d_y, page_size, cudaMemcpyDeviceToHost, stream_y);
        gpuErrAssert(cudaStreamSynchronize(stream_y));

        if (page_size != PAGE_SIZE) {
            transfer_managed_one(m_x, page_size, 0, stream_x);
            gpuErrAssert(cudaStreamSynchronize(stream_x));
        }   
    }

    printf("Warmup done\n");

    for (int i = 0; i < sizeof(page_sizes) / sizeof(page_sizes[0]); i++) {
        size_t page_size = page_sizes[i];
    
        /* Test one way transfer */
        start = dtime_usec(0);
        transfer_one(d_x, x, page_size, cudaMemcpyHostToDevice, stream_x);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        printf("HostToDevice: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(x, d_x, page_size, cudaMemcpyDeviceToHost, stream_x);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        printf("DeviceToHost: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(d_x, h_x, page_size, cudaMemcpyHostToDevice, stream_x);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        printf("HostToDevicePinned: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_one(h_x, d_x, page_size, cudaMemcpyDeviceToHost, stream_x);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        printf("DeviceToHostPinned: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));


        start = dtime_usec(0);
        transfer_two(d_x, x, y, d_y, page_size, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, stream_x, stream_y);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        gpuErrAssert(cudaStreamSynchronize(stream_y));
        printf("BothDirections: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));

        start = dtime_usec(0);
        transfer_two(h_x, x, y, h_y, page_size, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, stream_x, stream_y);
        gpuErrAssert(cudaStreamSynchronize(stream_x));
        gpuErrAssert(cudaStreamSynchronize(stream_y));
        printf("BothDirectionsPinned: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));

        // Too slow with page size 
        if (page_size != PAGE_SIZE) {
        
            // Modify data;
            for (int i = 0; i < N; i += PAGE_SIZE)
                m_x[i] = 0;

            start = dtime_usec(0);
            transfer_managed_one(m_x, page_size, 0, stream_x);
            gpuErrAssert(cudaStreamSynchronize(stream_x));
            printf("MemprefetchToDevice: PageSize:%zu, Bandwidth:%f GB/s\n", page_size, bandwidth(dtime_usec(start)));
        }

        printf("\n\n");
    }

    free(x);
    free(y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(m_x);
    cudaFree(m_y);

    test_deinitialize();
}
