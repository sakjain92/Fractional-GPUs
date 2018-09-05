/* This file contains cuda related external functions */
#ifndef __FRACTIONAL_GPU_CUH__
#define __FRACTIONAL_GPU_CUH__

#include <fgpu_internal_common.hpp>
#include <fractional_gpu.hpp>

#ifdef FGPU_COMP_COLORING_ENABLE

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

        /* Note: This is the most time taking process */
	dev_ctx->d_host_indicators->indicators[blockIdx.x].started = true;

        dev_ctx->d_dev_indicators->indicators[blockIdx.x].started = dev_ctx->launch_index;

        /* Pblocks launched on wrong SM have to wait for all other pblocks to be launched */
        if (sm < dev_ctx->start_sm || sm > dev_ctx->end_sm) {
            /* Poll in round robin fashion to avoid all reading same data at once */
            for (int i = blockIdx.x + 1; i < dev_ctx->num_pblock; i++)
                while(dev_ctx->d_dev_indicators->indicators[i].started !=  dev_ctx->launch_index)

            for (int i = 0; i < blockIdx.x; i++)
                while(dev_ctx->d_dev_indicators->indicators[i].started !=  dev_ctx->launch_index);

        }

        /* Prepare for the next function */
        dev_ctx->d_bindex->bindexes[dev_ctx->color].index[dev_ctx->index ^ 1] = 0;
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
    &dev_fctx;                                                              \
 })

#define FGPU_GET_GRIDDIM(ctx)                                               \
    ctx->gridDim

#define FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx)                               \
    for (; fgpu_device_get_blockIdx(&dev_fctx, &_blockIdx) == 0;)

#define FGPU_FOR_EACH_END

#else /* FGPU_COMP_COLORING_ENABLE */

#define FGPU_DEFINE_VOID_KERNEL(func)                                       \
    __global__ void func(void)

#define FGPU_DEFINE_KERNEL(func, ...)                                       \
    __global__ void func(__VA_ARGS__)

#define FGPU_DEVICE_INIT()                                                  \
({                                                                          \
    (fgpu_dev_ctx_t *)NULL;                                                 \
 })

#define FGPU_GET_GRIDDIM(ctx)       (gridDim)

#define FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx)   _blockIdx = blockIdx;

#define FGPU_FOR_EACH_END

#endif /* FGPU_COMP_COLORING_ENABLE */

#if defined(FGPU_USER_MEM_COLORING_ENABLED)

// TODO: This should be per GPU based
#define FGPU_DEVICE_COLOR_SHIFT	            12
#define FGPU_DEVICE_PAGE_SIZE               (1 << FGPU_DEVICE_COLOR_SHIFT)
#define FGPU_DEVICE_PAGE_MASK               (~(FGPU_DEVICE_PAGE_SIZE - 1))
#define FGPU_DEVICE_COLOR_PATTERN           0x4e4c3		// Split cache vertically

#define FGPU_COLOR_LOAD(ctx, addr)              \
({                                              \
    void *ptr = fgpu_color_load(ctx, addr);     \
    *(typeof(addr))ptr;                         \
})

#define FGPU_COLOR_STORE(ctx, addr, value)      \
({                                              \
    void *ptr = fgpu_color_load(ctx, addr);     \
    *(typeof(addr))ptr = value;                 \
})

#define FGPU_COLOR_TRANSLATE_ADDR(ctx, addr)    \
({                                              \
    void *ptr = fgpu_color_load(ctx, addr);     \
    (typeof(addr))ptr;                          \
})

__device__ __forceinline__
void *fgpu_color_load(const fgpu_dev_ctx_t *ctx, const void *virt_offset)
{
    uint64_t true_virt_addr;
	uint64_t c_virt_offset = (uint64_t)virt_offset;
	uint64_t idx = ((c_virt_offset >> FGPU_DEVICE_COLOR_SHIFT) << 1);
	uint32_t pattern = (idx + ctx->start_idx) & FGPU_DEVICE_COLOR_PATTERN;
	uint8_t parity = __popc(pattern) & 0x1;
	idx += (parity != ctx->color);
	true_virt_addr = ctx->start_virt_addr + (idx << FGPU_DEVICE_COLOR_SHIFT) + (c_virt_offset & 0xFFF);
	return  (void *)true_virt_addr;

}

/* For host side */
inline void *fgpu_color_device_true_virt_addr(const uint64_t start_virt_addr, 
                                              uint64_t start_phy_addr,
                                              int color,
                                              const void *virt_addr)
{
    uint64_t true_virt_addr;
    uint64_t start_idx = start_phy_addr >> FGPU_DEVICE_COLOR_SHIFT;
	uint64_t c_virt_offset = (uint64_t)virt_addr - start_virt_addr;
	uint64_t idx = ((c_virt_offset >> FGPU_DEVICE_COLOR_SHIFT) << 1);
	uint32_t pattern = (idx + start_idx) & FGPU_DEVICE_COLOR_PATTERN;
	uint8_t parity = __builtin_popcount(pattern) & 0x1;
	idx += (parity != color);
	true_virt_addr = start_virt_addr + (idx << FGPU_DEVICE_COLOR_SHIFT) + (c_virt_offset & 0xFFF);
	return  (void *)true_virt_addr;
}

#else /* FGPU_USER_MEM_COLORING_ENABLED */

/* TODO: Here as ctx is not being used, we get warning from compiler */

#define FGPU_COLOR_LOAD(ctx, addr)              \
({                                              \
    *addr;                                      \
})

#define FGPU_COLOR_STORE(ctx, addr, value)      \
({                                              \
    *addr = value;                              \
})

#define FGPU_COLOR_TRANSLATE_ADDR(ctx, addr)    \
({                                              \
    addr;                                       \
})

#endif /* FGPU_USER_MEM_COLORING_ENABLED */


#endif /* __FRACTIONAL_GPU_CUH__ */
