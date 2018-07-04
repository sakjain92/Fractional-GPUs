/* This file contains API for persistent kernels */
#include <assert.h>
#include <map>

#include <common.h>
#include <fractional_gpu.h>
#include <persistent.h>

static volatile fgpu_indicators_t *h_indicators;
static volatile fgpu_indicators_t *d_indicators;

static fgpu_bindexes_t *d_bindexes;
static cudaStream_t streams[FGPU_MAX_NUM_COLORS];

/* Mapping between SMs and Colors */
static std::pair<uint32_t, uint32_t>color_to_sms[FGPU_MAX_NUM_COLORS];

static bool cur_indexes[FGPU_MAX_NUM_COLORS];
static int last_color = -1;
static int last_tags[FGPU_MAX_NUM_COLORS] = {-1};
static int cur_tag;

typedef struct kernel_info {
    int color;
    int num_pblocks_launched;
    bool is_done;
    int return_code;
} kernel_info_t;

/* Kernel tag -> Info kernels */
static std::map<int, kernel_info_t> tag_to_info;

/* Initializes color info - TODO: Make this unique per device */
static void init_color_info(void)
{
    assert(FGPU_NUM_COLORS == 2);
    assert(FGPU_NUM_SM == 15);

//    color_to_sms[0] = std::make_pair(0, 14);
//    color_to_sms[1] = std::make_pair(15, 15);

    color_to_sms[0] = std::make_pair(0, 14);
    color_to_sms[1] = std::make_pair(15, 15);


}

/* Initialize */
int fgpu_init(void)
{
    static bool is_initialized = false;
    int ret = 0;

    if (is_initialized)
        return 0;

    is_initialized = 1;

    /* Create seperate streams for each color */
    for (int i = 0; i < FGPU_MAX_NUM_COLORS; i++) {
        ret = gpuErrCheck(cudaStreamCreate(&streams[i]));
        if (ret < 0)
            goto cleanup;
    }

    /* Allocate host pinned indicators */
    ret = gpuErrCheck(cudaHostAlloc(&h_indicators, sizeof(fgpu_indicators_t),
                                    cudaHostAllocMapped));
    if (ret < 0)
        goto cleanup;

    ret = gpuErrCheck(cudaHostGetDevicePointer(&d_indicators, (void *)h_indicators, 0));
    if (ret < 0)
        goto cleanup;

    memset((void *)h_indicators, 0, sizeof(fgpu_indicators_t));

    /* Allocate bindexes on device memory */
    ret = gpuErrCheck(cudaMalloc(&d_bindexes, sizeof(fgpu_bindexes_t)));
    if (ret < 0)
        goto cleanup;

    assert(FGPU_MAX_NUM_COLORS > 0);

    ret = gpuErrCheck(cudaMemsetAsync(d_bindexes, 0, sizeof(fgpu_bindexes_t),
                                      streams[0]));
    if (ret < 0)
        goto cleanup;

    ret = gpuErrCheck(cudaStreamSynchronize(streams[0]));
    if (ret < 0)
        goto cleanup;

    init_color_info();

    return 0;

cleanup:
    fgpu_deinit();
    return ret;
}

/* Deinitializes */
void fgpu_deinit(void)
{
    if (d_bindexes != NULL)
        cudaFreeHost((void *)d_bindexes);
    if (h_indicators != NULL)
        cudaFreeHost((void *)h_indicators);

    for (int i = 0; i < FGPU_MAX_NUM_COLORS; i++)
        cudaStreamDestroy(streams[i]);
}

/* Wait for last launched kernel to be completely started */
static void wait_for_last_start(void)
{
    if (last_color >= 0) {
        int last_tag = last_tags[last_color];

        if (last_tag >= 0) {

            kernel_info_t last_info;

            last_info = tag_to_info[last_tag];
            
            /* 
             * Need to wait for the last launched kernel to indicate all blocks
             * have been launched.
             */
            assert(last_info.is_done == false);
            for (int i = 0; i < last_info.num_pblocks_launched; i++) {
                while (!h_indicators->indicators[i].started[last_info.color]);
                h_indicators->indicators[i].started[last_info.color] = false;
            }
            last_color = -1;
        }
    }
}

/* Wait for last launched kernel of a specific color to be completed */
static int wait_for_last_complete(int color)
{
    int ret = 0;;

    if (last_tags[color] >= 0) {
        kernel_info_t *last_info;
        ret = gpuErrCheck(cudaStreamSynchronize(streams[color]));
       
        last_info = &tag_to_info[last_tags[color]];
        last_info->is_done = true;
        last_info->return_code = ret;
        last_tags[color] = -1;

        if (last_color == color)
            last_color = -1;
    }

    return ret;

}
/* Prepare ctx before launch */
int fgpu_prepare_launch_kernel(fgpu_ctx_t *ctx, uint3 *_gridDim, cudaStream_t **stream)
{
    uint32_t num_blocks;
    uint32_t num_threads;
    uint32_t num_pblocks;
    int color = ctx->color;
    int ret;
    kernel_info_t info;

    int tag;
    if (color >= FGPU_NUM_COLORS || color < 0)
        return -1;

    num_blocks = ctx->gridDim.x * ctx->gridDim.y * ctx->gridDim.z;
    if (num_blocks == 0)
        return -1;

    num_threads = ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z;
    if (num_threads == 0 || num_threads > FGPU_NUM_THREADS_PER_SM)
        return -1;

    /* Num threads should be power of 2 */
    if (num_threads & (num_threads - 1) != 0)
        return -1;

    wait_for_last_start();

    ret = wait_for_last_complete(color);
    if (ret < 0)
        return ret;

    num_pblocks = (FGPU_NUM_SM * FGPU_NUM_THREADS_PER_SM) / num_threads;
    ctx->num_blocks = num_blocks;
    ctx->index = cur_indexes[color];
    cur_indexes[color] ^= 1;   /* Toggle the index */
    ctx->d_indicators = d_indicators;
    ctx->d_bindex = d_bindexes;
    ctx->start_sm = color_to_sms[color].first;
    ctx->end_sm = color_to_sms[color].second;

    tag = ++cur_tag;
    info.color = color;
    info.num_pblocks_launched = num_pblocks;
    info.is_done = false;
    tag_to_info[tag] = info;

    last_tags[color] = tag;
    last_color = color;

    _gridDim->x = num_pblocks;
    _gridDim->y = 1;
    _gridDim->z = 1;
    *stream = &streams[color];

    return tag;
}

int fgpu_wait_for_kernel(int tag)
{
    std::map<int, kernel_info_t>::iterator it = tag_to_info.find(tag);
    kernel_info_t info;
    int ret;

    if (it == tag_to_info.end())
        return -1;

    info = it->second;
    if (info.is_done) {
        return info.return_code;
    }

    assert(tag == last_tags[info.color]);

    ret = wait_for_last_complete(info.color);

    tag_to_info.erase(tag);
    return ret;
}

