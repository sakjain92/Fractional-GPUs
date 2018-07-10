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
int fgpu_prepare_launch_kernel(fgpu_dev_ctx_t *ctx, uint3 *_gridDim, cudaStream_t **stream);
int fgpu_complete_launch_kernel(fgpu_dev_ctx_t *ctx);
cudaError_t fgpu_color_stream_synchronize(int color);

#endif /* FRACTIONAL_GPU */
