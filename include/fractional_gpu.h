/* This file is header file for exposing API to external application */
#ifndef __FRACTIONAL_GPU_H__
#define __FRACTIONAL_GPU_H__

#include <inttypes.h>

#include <persistent.h>

/* This structure is context that is handed over to kernel by host */
typedef struct fgpu_dev_ctx {
    volatile fgpu_indicators_t *d_host_indicators;  /* Used to indicate launch completion to host */
    volatile fgpu_indicators_t *d_dev_indicators;  /* Used to indicate launch completion to pblock */

    fgpu_bindexes_t *d_bindex;      /* Used to gather block indexes */
    int color;                      /* Color to be used by the kernel */
    int index;                      /* Index within the color */
    uint3 gridDim;                  /* User provided grid dimensions */
    uint3 blockDim;                 /* User provided block dimensions */
    int num_blocks;                 /* Number of blocks to be spawned */
    int num_pblock;                 /* Number of persistent thread blocks spawned */
    int start_sm;
    int end_sm;
    int _blockIdx;
} fgpu_dev_ctx_t;

int fgpu_server_init(void);
void fgpu_server_deinit(void);
int fgpu_init(void);
void fgpu_deinit(void);
int fgpu_set_color_prop(int color, size_t mem_size);
int fgpu_prepare_launch_kernel(fgpu_dev_ctx_t *ctx, uint3 *_gridDim, cudaStream_t **stream);
int fgpu_complete_launch_kernel(fgpu_dev_ctx_t *ctx);
cudaError_t fgpu_color_stream_synchronize(void);
int fpgpu_num_sm(int color, int *num_sm);
int fgpu_num_colors(void);

/* Macro to launch kernel - Returns a tag - Negative if error */
#define FGPU_LAUNCH_KERNEL(_gridDim, _blockDim, sharedMem, func, ...)       \
({                                                                          \
    fgpu_dev_ctx_t dev_fctx;                                                \
    int ret;                                                                \
    uint3 _lgridDim;                                                        \
    cudaStream_t *stream;                                                   \
    dev_fctx.gridDim = _gridDim;                                            \
    dev_fctx.blockDim = _blockDim;                                          \
    dev_fctx._blockIdx =  -1;                                               \
    ret = fgpu_prepare_launch_kernel(&dev_fctx, &_lgridDim, &stream);       \
    if (ret >= 0) {                                                         \
        func<<<_lgridDim, _blockDim, sharedMem, *stream>>>(dev_fctx, __VA_ARGS__); \
        ret = fgpu_complete_launch_kernel(&dev_fctx);                       \
    }                                                                       \
                                                                            \
    ret;                                                                    \
})

#define FGPU_LAUNCH_VOID_KERNEL(_gridDim, _blockDim, sharedMem, func)       \
({                                                                          \
    fgpu_dev_ctx_t dev_fctx;                                                \
    int ret;                                                                \
    uint3 _lgridDim;                                                        \
    cudaStream_t *stream;                                                   \
    dev_fctx.gridDim = _gridDim;                                            \
    dev_fctx.blockDim = _blockDim;                                          \
    dev_fctx._blockIdx = -1;                                                \
    ret = fgpu_prepare_launch_kernel(&dev_fctx, &_lgridDim, &stream);       \
    if (ret >= 0) {                                                         \
        func<<<_lgridDim, _blockDim, sharedMem, *stream>>>(dev_fctx);       \
        ret = fgpu_complete_launch_kernel(&dev_fctx);                       \
    }                                                                       \
    fgpu_complete_launch_kernel(&dev_fctx);                                 \
                                                                            \
    ret;                                                                    \
})

#endif /* FRACTIONAL_GPU */
