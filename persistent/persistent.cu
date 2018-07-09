/* This file contains API for persistent kernels */
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <map>

#include <cuda.h>

#include <common.h>
#include <fractional_gpu.h>
#include <persistent.h>

/* Name of the shared files */
#define FGPU_SHMEM_NAME             "fgpu_shmem"
#define FGPU_SHMEM_HOST_NAME        "fgpu_host_shmem"

/* File used by MPS */
#define FGPU_MPS_CONTROL_NAME       "/tmp/nvidia-mps/control"

/* TODO: Add support for multiple devices */
/* TODO: Add proper logging mechanism */
/* TODO: Use CudaIPC to share device memory pointer to be safe */

/* Currently only the very first device is used */
#define FGPU_DEVICE_NUMBER  0

/* Look into cuMemHostRegister and cuMemHostGetFlags and cuInit*/
/*sysconf(_SC_THREAD_PROCESS_SHARED), pthread_mutexattr_setpshared
 pthread_condattr_setpshared */

/* Can't share streams between process - As limitation by CUDA library/driver */
cudaStream_t streams[FGPU_MAX_NUM_COLORS];

/* Each process maps host pinned memory individually in it's addr space */
static volatile fgpu_indicators_t *h_indicators;

/*
 * This structure contains all host side information for persistent thread ctx.
 * Everything in this structure is shared by processes using shared mem.
 */
typedef struct fgpu_host_ctx {
  
    int device;
    int num_colors;
    int num_sm;
    int max_num_threads_per_sm;

    std::pair<uint32_t, uint32_t> color_to_sms[FGPU_MAX_NUM_COLORS];

    volatile fgpu_indicators_t *d_host_indicators;
    volatile fgpu_indicators_t *d_dev_indicators;

    fgpu_bindexes_t *d_bindexes;
   
    /* Lock to allow only one process to launch at a time */
    pthread_mutex_t launch_lock;
    pthread_cond_t launch_cond;
    bool is_lauchpad_free;
    bool cur_indexes[FGPU_MAX_NUM_COLORS];
    int last_color;
    int last_num_pblocks_launched;

    /*
     * Lock to allow only one outstanding operation in each color stream
     * CUDA streams can't be shared between processes.
     */
    pthread_mutex_t streams_lock;
    pthread_cond_t streams_cond[FGPU_MAX_NUM_COLORS];
    bool is_stream_free[FGPU_MAX_NUM_COLORS];

} fgpu_host_ctx_t;

/* Host side context */
static fgpu_host_ctx_t *g_host_ctx;

/* Shared memories file descriptor */
static int shmem_fd = -1;
static int shmem_host_fd = -1;

/* Checks if MPS is enabled */
static bool is_mps_enabled(void)
{
    int ret;
    struct stat st;

    ret = stat(FGPU_MPS_CONTROL_NAME, &st);
    if (ret < 0)
        return false;

    return true;
}

/* Initializes mutex for use in shared memory */
static int init_shared_mutex(pthread_mutex_t *lock)
{
    int ret;
    pthread_mutexattr_t attr;
    
    ret = pthread_mutexattr_init(&attr);
    if (ret < 0) {
        fprintf(stderr, "Mutex attr failed to initialize\n");
        return ret;
    }

    ret = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (ret < 0) {
        fprintf(stderr, "Mutex attr couldn't be set to be shared\n");
        return ret;
    }

    ret = pthread_mutex_init(lock, &attr);
    if (ret < 0) {
        fprintf(stderr, "Mutex can't be initialized\n");
        return ret;
    }

    return 0;
}


/* Initializes conditional variable for use in shared memory */
static int init_shared_condvar(pthread_cond_t *cond)
{
    int ret;
    pthread_condattr_t attr;
    
    ret = pthread_condattr_init(&attr);
    if (ret < 0) {
        fprintf(stderr, "Condvar attr failed to initialize\n");
        return ret;
    }

    ret = pthread_condattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (ret < 0) {
        fprintf(stderr, "Condvar attr couldn't be set to be shared\n");
        return ret;
    }

    ret = pthread_cond_init(cond, &attr);
    if (ret < 0) {
        fprintf(stderr, "Condvar can't be initialized\n");
        return ret;
    }

    return 0;
}

/* Sets color info per device */
static int init_color_info(fgpu_host_ctx_t *host_ctx,
        const cudaDeviceProp *device_prop)
{
    int num_colors;
    int num_sm = device_prop->multiProcessorCount;
    int sm_per_color;

    /* 
     * Colors depends on the memory hieracy on device and limitations of
     * coloing. Presently only 2 colors are supported in case of userspace
     * coloring.
     */
    if (strcmp(device_prop->name, "GeForce GTX 1070") == 0) {
        
        num_colors = 2;
        host_ctx->num_colors = num_colors;

    } else {
        /* All CUDA devices are not currently supported */
        fprintf(stderr, "Unknown CUDA device\n");
        return -1;
    }

    assert(FGPU_MAX_NUM_COLORS >= num_colors);

    /*
     * Due to integer division, all colors might not be balanced perfectly
     * Currently we are treating all colors equally. This is not neccesary
     */
    sm_per_color = num_sm / num_colors;

    if (sm_per_color == 0) {
        fprintf(stderr, "Too few SMs/Too many colors\n");
        return -1;
    }

    printf("Device: \"%s\", Number of Colors:%d\n", device_prop->name, num_colors);
    for (int i = 0; i < num_colors; i++) {
        int start_sm;
        int end_sm;

        start_sm = i * sm_per_color;
        end_sm = (i + 1) * sm_per_color - 1;
        if (i == num_colors - 1)
            end_sm = num_sm - 1;

        host_ctx->color_to_sms[i] = std::make_pair(start_sm, end_sm);
        if (i == 0)
            host_ctx->color_to_sms[i] = std::make_pair(0, 14);
        else
            host_ctx->color_to_sms[i] = std::make_pair(15, 15);
        printf("Color:%d, SMs:(%d->%d)\n", i, start_sm, end_sm);
    }

    return 0;
}

/* Sets the device to first available device */
static int init_device_info(fgpu_host_ctx_t *host_ctx)
{
    int deviceCount = 0;
    cudaDeviceProp device_prop;
    size_t max_threads;

    int ret = gpuErrCheck(cudaGetDeviceCount(&deviceCount));
    if (ret < 0)
        return ret;

    if (deviceCount == 0) {
        fprintf(stderr, "Couldn't find any CUDA devices\n");    
        return -1;
    }

    assert(deviceCount > FGPU_DEVICE_NUMBER);

    ret = gpuErrCheck(cudaSetDevice(FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return ret;

    ret = gpuErrCheck(cudaGetDeviceProperties(&device_prop, FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return ret;

    max_threads = device_prop.maxThreadsPerMultiProcessor *
        device_prop.multiProcessorCount;

    if (max_threads > FGPU_MAX_NUM_PBLOCKS * FGPU_MIN_BLOCKDIMS) {
        fprintf(stderr, "Too many SMs/Threads in CUDA device\n");
        return -1;
    }

    if (device_prop.warpSize != FGPU_MIN_BLOCKDIMS) {
        fprintf(stderr, "Warp size of CUDA device is not correct\n");
        return -1;
    }

    host_ctx->device = FGPU_DEVICE_NUMBER;
    host_ctx->num_sm = device_prop.multiProcessorCount;
    host_ctx->max_num_threads_per_sm = device_prop.maxThreadsPerMultiProcessor;

    ret = init_color_info(host_ctx, &device_prop);
    if (ret < 0)
        return ret;
    
    return 0;
}

/* Initialize (by server)
 * Currently, assumption is made that client's are launch some time after
 * server is launched, allowing server to initialize properly before clients
 * start using shared memory.
 */
int fgpu_server_init(void)
{
    int ret = 0;
    size_t shmem_size;
    size_t page_size;

    ret = gpuDriverErrCheck(cuInit(0));
    if (ret < 0)
        goto err;
    
    if (!is_mps_enabled()) {
        fprintf(stderr, "MPS is not enabled\n");
        goto err;
    }

    /* Create the shared memory */
    ret = shmem_fd = shm_open(FGPU_SHMEM_NAME,
            O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "Couldn't open shmem\n");
        goto err;
    }

    page_size = sysconf(_SC_PAGE_SIZE);

    shmem_size = ROUND_UP(sizeof(fgpu_host_ctx_t), page_size);

    ret = ftruncate(shmem_fd, shmem_size);
    if (ret < 0) {
        fprintf(stderr, "Can't truncate shmem file\n");
        ret = -1;
        goto err;
    }

    g_host_ctx = (fgpu_host_ctx_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_fd, 0);
    if (g_host_ctx == NULL) {
        fprintf(stderr, "Can't map shmem\n");
        ret = -1;
        goto err;
    }

    /* Allocate bindexes on device memory */
    ret = gpuErrCheck(cudaMalloc(&g_host_ctx->d_bindexes,
                sizeof(fgpu_bindexes_t)));
    if (ret < 0)
        goto err;

    assert(FGPU_MAX_NUM_COLORS > 0);

    ret = gpuErrCheck(cudaMemset(g_host_ctx->d_bindexes, 0, sizeof(fgpu_bindexes_t)));
    if (ret < 0)
        goto err;


    ret = shmem_host_fd = shm_open(FGPU_SHMEM_HOST_NAME,
            O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "Couldn't open shmem\n");
        goto err;
    }

    shmem_size = ROUND_UP(sizeof(fgpu_indicators_t), page_size);

    ret = ftruncate(shmem_host_fd, shmem_size);
    if (ret < 0) {
        fprintf(stderr, "Can't truncate shmem (host) file\n");
        ret = -1;
        goto err;
    }

    h_indicators = (volatile fgpu_indicators_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_host_fd, 0);
    if (h_indicators == NULL) {
        fprintf(stderr, "Can't map shmem\n");
        ret = -1;
        goto err;
    }

    /* 
     * This function needs to be called after a CUDA function is called so that
     * the device context is created in that function. CUDA context is created
     * lazily.
     */
    ret = gpuDriverErrCheck(cuMemHostRegister((void *)h_indicators, shmem_size,
                CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
    if (ret < 0)
        goto err;
    
    ret = gpuErrCheck(cudaHostGetDevicePointer(&g_host_ctx->d_host_indicators,
                (void *)h_indicators, 0));
    if (ret < 0)
        goto err;

    memset((void *)h_indicators, 0, sizeof(fgpu_indicators_t));

    ret = gpuErrCheck(cudaMalloc(&g_host_ctx->d_dev_indicators,
                sizeof(fgpu_indicators_t)));
    if (ret < 0)
        goto err;

    ret = gpuErrCheck(cudaMemset((void *)g_host_ctx->d_dev_indicators, 0,
                sizeof(fgpu_indicators_t)));
    if (ret < 0)
        goto err;

    ret = init_device_info(g_host_ctx);
    if (ret < 0)
        goto err;

    ret = init_shared_mutex(&g_host_ctx->launch_lock);
    if (ret < 0)
        goto err;

    ret = init_shared_condvar(&g_host_ctx->launch_cond);
    if (ret < 0)
        goto err;

    g_host_ctx->is_lauchpad_free = true;
    g_host_ctx->last_num_pblocks_launched = 0;
    g_host_ctx->last_color = -1;

    ret = init_shared_mutex(&g_host_ctx->streams_lock);
    if (ret < 0)
        goto err;

    for (int i = 0; i < g_host_ctx->num_colors; i++) {
        ret = init_shared_condvar(&g_host_ctx->streams_cond[i]);
        if (ret < 0)
            goto err;
        g_host_ctx->is_stream_free[i] = true;
    }

    /* 
     * Server doesn't need to create streams because server is not launching
     * processes
     */
    return 0;

err:
    fgpu_server_deinit();
    return ret;
}

/* Deinitializes */
void fgpu_server_deinit(void)
{
    printf("Server Terminating. Waiting for device to be free\n");
    gpuErrCheck(cudaDeviceSynchronize());

    if (g_host_ctx) {
        if (g_host_ctx->d_bindexes != NULL)
            cudaFree((void *)g_host_ctx->d_bindexes);

        if (g_host_ctx->d_dev_indicators != NULL)
            cudaFree((void *)g_host_ctx->d_dev_indicators);
    }
    
    if (h_indicators != NULL) {
        cuMemHostUnregister((void *)h_indicators);
        cudaFreeHost((void *)h_indicators);
    }

    if (shmem_host_fd > 0)
        close(shmem_host_fd);
    
    if (shmem_fd > 0)
        close(shmem_fd);

    /* Remove links so that can be reused */
    shm_unlink(FGPU_SHMEM_HOST_NAME);
    shm_unlink(FGPU_SHMEM_NAME);
}

/* Initialization for non-server */
int fgpu_init(void)
{
    int ret;
    size_t page_size;
    size_t shmem_size;

    if (!is_mps_enabled()) {
        fprintf(stderr, "MPS is not enabled\n");
        goto err;
    }


    /* Create the shared memory */
    ret = shmem_fd = shm_open(FGPU_SHMEM_NAME, O_RDWR, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "Couldn't open shmem\n");
        goto err;
    }

    page_size = sysconf(_SC_PAGE_SIZE);

    shmem_size = ROUND_UP(sizeof(fgpu_host_ctx_t), page_size);
    g_host_ctx = (fgpu_host_ctx_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_fd, 0);
    if (g_host_ctx == NULL) {
        fprintf(stderr, "Can't map shmem\n");
        ret = -1;
        goto err;
    }

    ret = shmem_host_fd = shm_open(FGPU_SHMEM_HOST_NAME, O_RDWR, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "Couldn't open shmem\n");
        goto err;
    }

    shmem_size = ROUND_UP(sizeof(fgpu_indicators_t), page_size);

    h_indicators = (volatile fgpu_indicators_t *)mmap(NULL, shmem_size,
            PROT_READ | PROT_WRITE, MAP_SHARED, shmem_host_fd, 0);
    if (h_indicators == NULL) {
        fprintf(stderr, "Can't map shmem (host pinned)\n");
        ret = -1;
        goto err;
    }

    /* Create seperate streams for each color */
    for (int i = 0; i < g_host_ctx->num_colors; i++) {
        ret = gpuErrCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault));
        if (ret < 0)
            goto err;
    }

    return 0;

err:
    fgpu_server_deinit();
    return ret;
}

void fgpu_deinit(void)
{
    for (int i = 0; i < g_host_ctx->num_colors; i++) {
        cudaStreamDestroy(streams[i]);
    }

    if (shmem_host_fd > 0)
        close(shmem_host_fd);
    
    if (shmem_fd > 0)
        close(shmem_fd);
}


/* Wait for last launched kernel to be completely started */
static void wait_for_last_start(void)
{
    pthread_mutex_lock(&g_host_ctx->launch_lock);
    while (1) {
        if (g_host_ctx->is_lauchpad_free)
            break;
        pthread_cond_wait(&g_host_ctx->launch_cond, &g_host_ctx->launch_lock);
    }
    g_host_ctx->is_lauchpad_free = false;
    pthread_mutex_unlock(&g_host_ctx->launch_lock);

    /* Wait for all pblocks to be accounted for */
    for (int i = 0; i < g_host_ctx->last_num_pblocks_launched; i++) {
        while (!h_indicators->indicators[i].started);
        h_indicators->indicators[i].started = false;
    }
}

/* Called when cuda stream operation completes 
 * Unforntunately, currently Nvidia does't provide with stream callbacks with MPS
 */
static void stream_callback(int color)
{
    pthread_mutex_lock(&g_host_ctx->streams_lock);
    g_host_ctx->is_stream_free[color] = true;
    pthread_cond_signal(&g_host_ctx->streams_cond[color]);
    pthread_mutex_unlock(&g_host_ctx->streams_lock);
}

/* Called after kernel has been launched */
int fgpu_complete_launch_kernel(fgpu_dev_ctx_t *ctx)
{
    int ret;

    g_host_ctx->last_color = ctx->color;
    g_host_ctx->last_num_pblocks_launched = ctx->num_pblock;

    pthread_mutex_lock(&g_host_ctx->launch_lock);
    g_host_ctx->is_lauchpad_free = true;
    pthread_cond_signal(&g_host_ctx->launch_cond);
    pthread_mutex_unlock(&g_host_ctx->launch_lock);

    ret = gpuErrCheck(cudaStreamSynchronize(streams[ctx->color]));

    stream_callback(ctx->color);

    return ret;
}

/* Wait for last launched kernel of a specific color to be completed */
static void wait_for_last_complete(int color)
{
    pthread_mutex_lock(&g_host_ctx->streams_lock);
    while (1) {
        if (g_host_ctx->is_stream_free[color])
            break;
        pthread_cond_wait(&g_host_ctx->streams_cond[color], &g_host_ctx->streams_lock);
    }
    g_host_ctx->is_stream_free[color] = false;
    pthread_mutex_unlock(&g_host_ctx->streams_lock);
}


/* Prepare ctx before launch */
int fgpu_prepare_launch_kernel(fgpu_dev_ctx_t *ctx, uint3 *_gridDim, cudaStream_t **stream)
{
    uint32_t num_blocks;
    uint32_t num_threads;
    uint32_t num_pblocks;
    int color = ctx->color;

    if (color >= g_host_ctx->num_colors || color < 0)
        return -1;

    num_blocks = ctx->gridDim.x * ctx->gridDim.y * ctx->gridDim.z;
    if (num_blocks == 0)
        return -1;

    num_threads = ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z;
    if (num_threads == 0 || num_threads > g_host_ctx->max_num_threads_per_sm)
        return -1;

    /* Num threads should be power of 2 */
    /* TODO: Relax this constraint */
    if (num_threads & (num_threads - 1) != 0)
        return -1;

    if (num_threads < FGPU_MIN_BLOCKDIMS)
        return -1;
    
    wait_for_last_complete(color);

    wait_for_last_start();

    num_pblocks =
        (g_host_ctx->num_sm * g_host_ctx->max_num_threads_per_sm) / num_threads;

    ctx->num_blocks = num_blocks;
    ctx->index = g_host_ctx->cur_indexes[color];
    g_host_ctx->cur_indexes[color] ^= 1;   /* Toggle the index */
    ctx->d_host_indicators = g_host_ctx->d_host_indicators;
    ctx->d_dev_indicators = g_host_ctx->d_dev_indicators;
    ctx->d_bindex = g_host_ctx->d_bindexes;
    ctx->start_sm = g_host_ctx->color_to_sms[color].first;
    ctx->end_sm = g_host_ctx->color_to_sms[color].second;

    _gridDim->x = num_pblocks;
    _gridDim->y = 1;
    _gridDim->z = 1;
    *stream = &streams[color];

    return 0;
}

cudaError_t fgpu_color_stream_synchronize(int color)
{
    if (color >= g_host_ctx->num_colors || color < 0)
        return cudaErrorInvalidValue;

    return cudaStreamSynchronize(streams[color]);
}
