#include <assert.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

FGPU_DEFINE_KERNEL(info, int A)
{
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();
    uint3 _blockIdx;

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
    uint3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    } FGPU_FOR_EACH_END;
}

FGPU_DEFINE_KERNEL(simple, uint32_t *out)
{
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();
    uint3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 1], _blockIdx.x);
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 2], _blockIdx.y);
        FGPU_COLOR_STORE(ctx, &out[32 * blockIdx.x + _blockIdx.x + 3], _blockIdx.z);
    } FGPU_FOR_EACH_END;
}

int main()
{
    int ret;
    dim3 threads(32, 32, 1);
    dim3 grid(20, 10);
    double start;

    ret = fgpu_init();
    if (ret < 0)
        return ret;

    ret = fgpu_set_color_prop(0, 128 * 1024 * 1024);
    if (ret < 0)
        return ret;

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
    uint32_t *h_out, *d_out;
    size_t sz = 2 * 1024 * 1024;
    ret = fgpu_memory_allocate((void **)&h_out, sz);
    if (ret < 0)
        return ret;
    ret = fgpu_memory_prefetch_to_device_async(h_out, sz);
    if (ret < 0)
        return ret;
    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;
    ret = fgpu_memory_get_device_pointer((void **)&d_out, h_out);
    if (ret < 0)
        return ret;

    for (int i = 0; i < 10000; i++) {
        start = dtime_usec(0);
        ret = FGPU_LAUNCH_KERNEL(simple, grid, threads, 0, d_out);
        assert(ret == 0);
    	ret = fgpu_color_stream_synchronize();
    	assert(ret == 0);
        printf("Simple Time:%f\n", dtime_usec(start));
    }
   
    fgpu_deinit();
    return 0;

}
