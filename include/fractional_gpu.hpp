/* This file is header file for exposing API to external application */
#ifndef __FRACTIONAL_GPU_HPP__
#define __FRACTIONAL_GPU_HPP__

#include <inttypes.h>

#include <fgpu_internal_persistent.hpp>

#include <fgpu_internal_common.hpp>

/* Currently only the very first device is used */
#define FGPU_DEVICE_NUMBER  0

/* This structure is context that is handed over to kernel by host */
typedef struct fgpu_dev_ctx {
    volatile fgpu_indicators_t *d_host_indicators;  /* Used to indicate launch completion to host */
    volatile fgpu_bindex_t *d_dev_indicator;        /* Used to indicate launch completion to pblock */

    fgpu_bindex_t *d_bindex;        /* Used to gather block indexes */
    int color;                      /* Color to be used by the kernel */
    int index;                      /* Index within single program */
    dim3 gridDim;                   /* User provided grid dimensions */
    dim3 blockDim;                  /* User provided block dimensions */
    int num_blocks;                 /* Number of blocks to be spawned */
    int num_pblock;                 /* Number of persistent thread blocks spawned */
    int start_sm;
    int end_sm;
    int num_active_pblocks;	    /* Number of pblocks which will do computation */
    int _blockIdx;

#if defined(FGPU_USER_MEM_COLORING_ENABLED)
    uint64_t start_virt_addr;
    uint64_t start_idx;
#endif

} fgpu_dev_ctx_t;

enum fgpu_memory_copy_type {
    FGPU_COPY_CPU_TO_GPU,
    FGPU_COPY_GPU_TO_CPU,
    FGPU_COPY_GPU_TO_GPU,
    FGPU_COPY_CPU_TO_CPU,
    FGPU_COPY_DEFAULT,      /* Direction is automatically detected */
};

int fgpu_server_init(void);
void fgpu_server_deinit(void);
int fgpu_init(void);
void fgpu_deinit(void);
int fgpu_get_env_color(void);
size_t fgpu_get_env_color_mem_size(void);
bool fgpu_is_init_complete(void);
int fgpu_set_color_prop(int color, size_t mem_size);
bool fgpu_is_color_prop_set(void);
int fgpu_prepare_launch_kernel(fgpu_dev_ctx_t *ctx, const void *func, 
        size_t shared_mem, dim3 *_gridDim, cudaStream_t **stream);
int fgpu_complete_launch_kernel(fgpu_dev_ctx_t *ctx);
int fgpu_color_stream_synchronize(void);
int fpgpu_num_sm(int color, int *num_sm);
int fgpu_num_colors(void);

int fgpu_memory_get_device_info(int *num_colors, size_t *max_len);

int fgpu_memory_allocate(void **p, size_t len);
int fgpu_memory_free(void *p);

void *fgpu_memory_get_phy_address(void *addr);

int fgpu_memory_copy_async(void *dst, const void *src, size_t count,
                           enum fgpu_memory_copy_type type,
                           cudaStream_t stream = NULL);
int fgpu_memory_memset_async(void *address, int value, size_t count,
                            cudaStream_t stream = NULL);

#ifdef FGPU_COMP_COLORING_ENABLE

/* Macro to launch kernel - Returns a tag - Negative if error */
#define FGPU_LAUNCH_KERNEL(func, _gridDim, _blockDim, sharedMem, ...)       \
({                                                                          \
    fgpu_dev_ctx_t dev_fctx;                                                \
    int ret;                                                                \
    dim3 _lgridDim;                                                         \
    cudaStream_t *stream;                                                   \
    fgpu_set_ctx_dims(&dev_fctx, _gridDim, _blockDim);                      \
    dev_fctx._blockIdx =  -1;                                               \
    ret = fgpu_prepare_launch_kernel(&dev_fctx, (const void *)func,         \
            sharedMem, &_lgridDim, &stream);                                \
    if (ret >= 0) {                                                         \
        func<<<_lgridDim, _blockDim, sharedMem, *stream>>>(dev_fctx,        \
                __VA_ARGS__);                                               \
        ret = fgpu_complete_launch_kernel(&dev_fctx);                       \
    }                                                                       \
                                                                            \
    ret;                                                                    \
})

#define FGPU_LAUNCH_KERNEL_VOID(func, _gridDim, _blockDim, sharedMem)       \
({                                                                          \
    fgpu_dev_ctx_t dev_fctx;                                                \
    int ret;                                                                \
    dim3 _lgridDim;                                                         \
    cudaStream_t *stream;                                                   \
    fgpu_set_ctx_dims(&dev_fctx, _gridDim, _blockDim);                      \
    dev_fctx._blockIdx = -1;                                                \
    ret = fgpu_prepare_launch_kernel(&dev_fctx, (const void *)func,         \
            sharedMem, &_lgridDim, &stream);                                \
    if (ret >= 0) {                                                         \
        func<<<_lgridDim, _blockDim, sharedMem, *stream>>>(dev_fctx);       \
        ret = fgpu_complete_launch_kernel(&dev_fctx);                       \
    }                                                                       \
                                                                            \
    ret;                                                                    \
})

#else /* FGPU_COMP_COLORING_ENABLE */

/* Macro to launch kernel - Returns a tag - Negative if error */
#define FGPU_LAUNCH_KERNEL(func, _gridDim, _blockDim, sharedMem, ...)       \
({                                                                          \
    func<<<_gridDim, _blockDim, sharedMem>>>(__VA_ARGS__);                  \
    0;                                                                      \
 })

#define FGPU_LAUNCH_KERNEL_VOID(func, _gridDim, _blockDim, sharedMem)       \
({                                                                          \
    func<<<_gridDim, _blockDim, sharedMem>>>(void);                         \
    0;                                                                      \
})

#endif /* FGPU_COMP_COLORING_ENABLE */

#endif /* FRACTIONAL_GPU_HPP */
