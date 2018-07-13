/* This file contains cuda related external functions */
#ifndef __FRACTIONAL_GPU_CUDA_H__
#define __FRACTIONAL_GPU_CUDA_H__

#include <fractional_gpu.h>


/* Macro to define (modified) kernels (with no args) */
#define FGPU_DEFINE_VOID_KERNEL(func)                                       \
    __global__ void func(fgpu_dev_ctx_t dev_fctx)

/* Macro to define (modified) kernels */
#define FGPU_DEFINE_KERNEL(func, ...)                                       \
    __global__ void func(fgpu_dev_ctx_t dev_fctx, __VA_ARGS__)

/* 
 * Have to keep these functions as inlines because seperate compilation in CUDA
 * has high performance impact
 */
__device__ __forceinline__
int fgpu_device_init(const fgpu_dev_ctx_t *dev_ctx)
{
    uint sm;
    asm("mov.u32 %0, %smid;" : "=r"(sm));
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
       
        /* Prepare for the next function */
        dev_ctx->d_bindex->bindexes[dev_ctx->color].index[dev_ctx->index ^ 1] = 0;
        dev_ctx->d_dev_indicators->indicators[blockIdx.x].started = true;

        /* Pblocks launched on wrong SM have to wait for all other pblocks to be launched */
        if (sm < dev_ctx->start_sm || sm > dev_ctx->end_sm) {
            /* Poll in round robin fashion to avoid all reading same data at once */
            for (int i = blockIdx.x + 1; i < dev_ctx->num_pblock; i++)
                while(!dev_ctx->d_dev_indicators->indicators[i].started)
            for (int i = 0; i < blockIdx.x; i++)
                while(!dev_ctx->d_dev_indicators->indicators[i].started);
        }

        /* Note: This is the most time taking process */
        dev_ctx->d_host_indicators->indicators[blockIdx.x].started = true;
    }
  
    if (sm < dev_ctx->start_sm || sm > dev_ctx->end_sm) {
        __syncthreads();
        return -1;
    }
    
    return 0;
}

__device__ __forceinline__
int fgpu_device_get_blockIdx(fgpu_dev_ctx_t *dev_ctx, uint3 *_blockIdx)
{
    __shared__ int lblockIdx;
    __shared__ uint3 lblockIdx3D;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        uint blocks_left;
        uint num2Dblocks;
        uint x, y, z;

        lblockIdx = atomicAdd(&dev_ctx->d_bindex->bindexes[dev_ctx->color].index[dev_ctx->index], 1);

        num2Dblocks = dev_ctx->gridDim.x * dev_ctx->gridDim.y;
        z = lblockIdx / (num2Dblocks);
        blocks_left = lblockIdx - (z * num2Dblocks);
        y = blocks_left / dev_ctx->gridDim.x;
        x = blocks_left - y * dev_ctx->gridDim.x;
        lblockIdx3D.x = x;
        lblockIdx3D.y = y;
        lblockIdx3D.z = z;
    }
    __syncthreads();

    if (lblockIdx >= dev_ctx->num_blocks)
        return -1;

    *_blockIdx = lblockIdx3D;

    return 0;
}

#define FGPU_DEVICE_INIT()                                                  \
({                                                                          \
    if (fgpu_device_init(&dev_fctx) < 0)                                    \
        return;                                                             \
    dev_fctx.gridDim;                                                       \
 })

#define FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx)                               \
    for (; fgpu_device_get_blockIdx(&dev_fctx, &_blockIdx) == 0;)

#define FGPU_FOR_EACH_END


#endif /* __FRACTIONAL_GPU_CUDA_H__ */
