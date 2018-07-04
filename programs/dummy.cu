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
                    _blockIdx.x, _blockIdx.y, _blockIdx.z, sm, fctx.start_sm, fctx.end_sm, A);
    }
}

FGPU_DEFINE_VOID_KERNEL(dummy)
{
    FGPU_DEVICE_INIT();
    uint3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    }
}

FGPU_DEFINE_KERNEL(simple, uint3 *out)
{
    FGPU_DEVICE_INIT();
    uint3 _blockIdx;

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
        *out = _blockIdx;
    }
}

int main()
{
    int tag, ret;
    dim3 threads(32, 32, 1);
    dim3 grid(20, 10);
    double start;

    assert(fgpu_init() == 0);

    tag = FGPU_LAUNCH_KERNEL(0, grid, threads, 0, info, 100);
    assert(tag);
    ret = fgpu_wait_for_kernel(tag);
    assert(ret == 0);

/*
    for (int i = 0; i < 100; i++) {
        start = dtime_usec(0);
        tag = FGPU_LAUNCH_VOID_KERNEL(0, grid, threads, 0, dummy);
        assert(tag);
        ret = fgpu_wait_for_kernel(tag);
        assert(ret == 0);
        printf("Dummy Time:%f\n", dtime_usec(start));
    }
*/
    uint3 *d_out;
    gpuErrAssert(cudaMalloc(&d_out, sizeof(uint3)));
    for (int i = 0; i < 100; i++) {
        start = dtime_usec(0);
        tag = FGPU_LAUNCH_KERNEL(0, grid, threads, 0, simple, d_out);
        assert(tag);
        ret = fgpu_wait_for_kernel(tag);
        assert(ret == 0);
        printf("Dummy Time:%f\n", dtime_usec(start));
    }

    fgpu_deinit();
    return 0;

}
