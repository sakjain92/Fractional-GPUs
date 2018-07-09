#include <assert.h>

#include <common.h>
#include <fractional_gpu.h>


FGPU_DEFINE_KERNEL(info, int A)
{
    FGPU_DEVICE_INIT();
    uint3 _blockIdx;
    uint sm;
    asm("mov.u32 %0, %smid;" : "=r"(sm));
    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            printf("BlockId:(%d, %d, %d), SM:%d, (Allowed:%d,%d), Arg:%d\n",
                    _blockIdx.x, _blockIdx.y, _blockIdx.z, sm,
		    dev_fctx.start_sm, dev_fctx.end_sm, A);
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
    FGPU_DEVICE_INIT();
    uint3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        out[32 * blockIdx.x + _blockIdx.x + 1] = _blockIdx.x;
        out[32 * blockIdx.x + _blockIdx.x + 2] = _blockIdx.y;
        out[32 * blockIdx.x + _blockIdx.x + 3] = _blockIdx.z;
    } FGPU_FOR_EACH_END;
}

int main()
{
    int tag, ret;
    dim3 threads(32, 32, 1);
    dim3 grid(20, 10);
    double start;

    ret = fgpu_init();
    if (ret < 0)
        return ret;

    tag = FGPU_LAUNCH_KERNEL(0, grid, threads, 0, info, 100);
    if (tag < 0)
        return tag;

    ret = gpuErrCheck(fgpu_color_stream_synchronize(0));
    if (ret < 0)
        return ret;

/*
    for (int i = 0; i < 100; i++) {
        start = dtime_usec(0);
        tag = FGPU_LAUNCH_VOID_KERNEL(0, grid, threads, 0, dummy);
        assert(tag);
	gpuErrAssert(fgpu_color_stream_synchronize(0));
        printf("Dummy Time:%f\n", dtime_usec(start));
    }
*/
    uint32_t *d_out;
    gpuErrAssert(cudaMalloc(&d_out, 32 * 32 * sizeof(uint3)));
    for (int i = 0; i < 10000; i++) {
        start = dtime_usec(0);
        tag = FGPU_LAUNCH_KERNEL(0, grid, threads, 0, simple, d_out);
        assert(tag);
	gpuErrAssert(fgpu_color_stream_synchronize(0));
        printf("Simple Time:%f\n", dtime_usec(start));
    }
    
    fgpu_deinit();
    return 0;

}
