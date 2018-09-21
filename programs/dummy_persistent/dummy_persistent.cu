#include <assert.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

FGPU_DEFINE_KERNEL(info, int A)
{
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();
    dim3 _blockIdx;

    uint sm;
    asm("mov.u32 %0, %smid;" : "=r"(sm));
    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            printf("BlockId:(%d, %d, %d), SM:%d, (Allowed:%d,%d), Arg:%d\n",
                    _blockIdx.x, _blockIdx.y, _blockIdx.z, sm,
		            ctx->start_sm, ctx->end_sm, A);
    } FGPU_FOR_EACH_END;
}

FGPU_DEFINE_VOID_KERNEL(dummy)
{
    FGPU_DEVICE_INIT();
    dim3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    } FGPU_FOR_EACH_END;
}

FGPU_DEFINE_KERNEL(simple, uint32_t *out)
{
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();
    dim3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 1], _blockIdx.x);
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 2], _blockIdx.y);
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 3], _blockIdx.z);
    } FGPU_FOR_EACH_END;
}

int main(int argc, char **argv)
{
    int ret;
    dim3 threads(32, 32, 1);
    dim3 grid(20, 10);
    double start;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    ret = FGPU_LAUNCH_KERNEL(info, grid, threads, 0, 100);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

/*
    for (int i = 0; i < 100; i++) {
        start = dtime_usec(0);
        ret = FGPU_LAUNCH__KERNEL_VOID(dummy, grid, threads, 0);
        assert(ret);
	    gpuErrAssert(fgpu_color_stream_synchronize(0));
        printf("Dummy Time:%f\n", dtime_usec(start));
    }
*/
    uint32_t *d_out;
    size_t sz = 2 * 1024 * 1024;
    ret = fgpu_memory_allocate((void **)&d_out, sz);
    if (ret < 0)
        return ret;

    for (int i = 0; i < num_iterations; i++) {
        start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(simple, grid, threads, 0, d_out);
        assert(ret == 0);
    	ret = fgpu_color_stream_synchronize();
    	assert(ret == 0);
        printf("Simple Time:%f\n", dtime_usec(start));
    }

    fgpu_memory_free(d_out);

    test_deinitialize();
    return 0;

}
