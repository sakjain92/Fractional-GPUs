/* This file contains API for persistent kernels */
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <map>
#include <string>

#include <cuda.h>

#include <fgpu_internal_common.hpp>
#include <fgpu_internal_memory.hpp>
#include <fgpu_internal_persistent.hpp>
#include <fractional_gpu.hpp>

/* TODO: Add support for multithreaded applications */

/* Name of the shared files */
#define FGPU_SHMEM_NAME             "fgpu_shmem"
#define FGPU_SHMEM_HOST_NAME        "fgpu_host_shmem"

/* File used by MPS */
#define FGPU_MPS_CONTROL_NAME       "/tmp/nvidia-mps/control"

/* TODO: Add support for multiple devices */
/* TODO: Add proper logging mechanism */
/* TODO: Use CudaIPC to share device memory pointer to be safe */

#define FGPU_INVALID_COLOR -1

/* Name of environment variables to check for color/size of colored mem */
#define FGPU_COLOR_ENV_NAME             "FGPU_COLOR_ENV"
#define FGPU_COLOR_MEM_SIZE_ENV_NAME    "FGPU_COLOR_MEM_SIZE_ENV"

/* Default values of color/size of colored mem */
#define FGPU_DEFAULT_COLOR              0
#define FGPU_DEFAULT_COLOR_MEM_SIZE     (1024 * 1024 * 1024) /* 1 GB */

/* List of supported GPUs */
static std::string supported_gpus[] = {"GeForce GTX 1070", "GeForce GTX 1080", "Tesla V100-SXM2-16GB"};

/* Look into cuMemHostRegister and cuMemHostGetFlags and cuInit*/
/*sysconf(_SC_THREAD_PROCESS_SHARED), pthread_mutexattr_setpshared
 pthread_condattr_setpshared */

/* Stream used for all operations. NULL stream is not used */
cudaStream_t color_stream;

/* Each process maps host pinned memory individually in it's addr space */
static volatile fgpu_indicators_t *h_indicators;
static volatile fgpu_indicators_t *d_host_indicators;
static volatile fgpu_bindex_t *d_dev_indicator;
static fgpu_bindex_t *d_bindex;
int cur_index;
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

    /* Lock to allow only one process to launch at a time */
    pthread_mutex_t launch_lock;
    pthread_cond_t launch_cond;
    bool is_lauchpad_free;
    int last_color;
    int last_num_pblocks_launched;
    int last_num_active_pblocks;

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

/* The set color for the process */
static int g_color = FGPU_INVALID_COLOR;

/* Checks if MPS is enabled */
static bool is_mps_enabled(void)
{
    int ret;
    struct stat st;
    cudaDeviceProp device_prop;

    /* Check if default file created by MPS exists ? */
    ret = stat(FGPU_MPS_CONTROL_NAME, &st);
    if (ret < 0)
        return false;

    /* Device must be set in exclusive mode */
    ret = gpuErrCheck(cudaGetDeviceProperties(&device_prop, FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return false;

    if (device_prop.computeMode != cudaComputeModeExclusiveProcess)
        return false;
    return true;

    /* XXX: All these conditions still now sufficient - Need to check to see
     *  if MPS process is running or not also.
     */
}

/* Initializes mutex for use in shared memory */
static int init_shared_mutex(pthread_mutex_t *lock)
{
    int ret;
    pthread_mutexattr_t attr;
    
    ret = pthread_mutexattr_init(&attr);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Mutex attr failed to initialize\n");
        return ret;
    }

    ret = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Mutex attr couldn't be set to be shared\n");
        return ret;
    }

    ret = pthread_mutex_init(lock, &attr);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Mutex can't be initialized\n");
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
        fprintf(stderr, "FGPU:Condvar attr failed to initialize\n");
        return ret;
    }

    ret = pthread_condattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Condvar attr couldn't be set to be shared\n");
        return ret;
    }

    ret = pthread_cond_init(cond, &attr);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Condvar can't be initialized\n");
        return ret;
    }

    return 0;
}

/* Sets color info per device */
static int init_color_info(fgpu_host_ctx_t *host_ctx, int device,
        const cudaDeviceProp *device_prop)
{
    int num_colors;
    int num_sm = device_prop->multiProcessorCount;
    int sm_per_color;
    int supported = false;
   
    /* Check device is supported */
    for (int i = 0; i < sizeof(supported_gpus)/sizeof(supported_gpus[0]); i++) {
        if (supported_gpus[i].compare(device_prop->name) == 0) {
            supported = true;
            break;
        }
    }

    if (!supported) {
        fprintf(stderr, "FGPU:Unknown CUDA device\n");
        return -ENXIO;
    }

    /* 
     * Since each color needs atleast one SM, so this is the upper bound.
     * From there, we find the lower bound
     */
    num_colors = num_sm;

    num_colors = FGPU_MAX_NUM_COLORS < num_colors ? FGPU_MAX_NUM_COLORS : num_colors;

    if (FGPU_PREFERRED_NUM_COLORS > 0) {
        num_colors = FGPU_PREFERRED_NUM_COLORS < num_colors ? FGPU_PREFERRED_NUM_COLORS : num_colors;
    }

#ifdef FGPU_MEM_COLORING_ENABLED
    /* If memory coloring is enabled, take the minimum of the colors */
    int mem_colors;
    int ret = fgpu_memory_get_device_info(&mem_colors, NULL);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Memory coloring enabled but can't get colors in kernel driver\n");
	    return ret;
    }

    /* 
     * When memory coloring enabled, the total number of colors available is 
     * equal to memory coloring.
     * TODO: Allow different memory and computational colors
     */
    if (num_colors < mem_colors) {
        fprintf(stderr, "FGPU:Memory coloring enabled but too less computational colors\n");
	    return -EINVAL;
    }

    num_colors = mem_colors;

#endif

    if (num_colors <= 0) {
        fprintf(stderr, "FGPU:Too less colors for coloring\n");
        return -EINVAL;
    }

    host_ctx->num_colors  = num_colors;

    /*
     * Due to integer division, all colors might not be balanced perfectly.
     * Currently we are treating all colors equally. This is not neccesary.
     */
    sm_per_color = num_sm / num_colors;

    if (sm_per_color == 0) {
        fprintf(stderr, "FGPU:Too few SMs/Too many colors\n");
        return -EINVAL;
    }

    printf("FGPU:Device: \"%s\", Number of Colors:%d\n", device_prop->name, num_colors);
    for (int i = 0; i < num_colors; i++) {
        int start_sm;
        int end_sm;

        start_sm = i * sm_per_color;
        end_sm = (i + 1) * sm_per_color - 1;
        if (i == num_colors - 1)
            end_sm = num_sm - 1;

        host_ctx->color_to_sms[i] = std::make_pair(start_sm, end_sm);
        printf("FGPU:Color:%d, SMs:(%d->%d)\n", i, start_sm, end_sm);
    }

    return 0;
}

/* Sets the device to first available device */
static int init_device_info(fgpu_host_ctx_t *host_ctx)
{
    int deviceCount = 0;
    cudaDeviceProp device_prop;

    int ret = gpuErrCheck(cudaGetDeviceCount(&deviceCount));
    if (ret < 0)
        return ret;

    if (deviceCount == 0) {
        fprintf(stderr, "FGPU:Couldn't find any CUDA devices\n");    
        return -ENXIO;
    }

    if (deviceCount <= FGPU_DEVICE_NUMBER) {
        fprintf(stderr, "FGPU:Couldn't find CUDA device\n");    
        return -ENXIO;
    }

    ret = gpuErrCheck(cudaSetDevice(FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return ret;

    ret = gpuErrCheck(cudaGetDeviceProperties(&device_prop, FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return ret;

    host_ctx->device = FGPU_DEVICE_NUMBER;
    host_ctx->num_sm = device_prop.multiProcessorCount;
    host_ctx->max_num_threads_per_sm = device_prop.maxThreadsPerMultiProcessor;

    ret = init_color_info(host_ctx, FGPU_DEVICE_NUMBER, &device_prop);
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

    /* Create the shared memory */
    ret = shmem_fd = shm_open(FGPU_SHMEM_NAME,
            O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Couldn't create shmem file."
                " Please delete file (%s) if exists\n", "/dev/shm/" FGPU_SHMEM_NAME);
        goto err;
    }

    page_size = sysconf(_SC_PAGE_SIZE);

    shmem_size = ROUND_UP(sizeof(fgpu_host_ctx_t), page_size);

    ret = ftruncate(shmem_fd, shmem_size);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Can't truncate shmem file\n");
        goto err;
    }

    g_host_ctx = (fgpu_host_ctx_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_fd, 0);
    if (g_host_ctx == NULL) {
        fprintf(stderr, "FGPU:Can't map shmem\n");
        ret = -errno;
        goto err;
    }

    ret = shmem_host_fd = shm_open(FGPU_SHMEM_HOST_NAME,
            O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Couldn't create shmem file."
                " Please delete file (%s) if exists\n", "/dev/shm/" FGPU_SHMEM_HOST_NAME);
        goto err;
    }

    shmem_size = ROUND_UP(sizeof(fgpu_indicators_t), page_size);

    ret = ftruncate(shmem_host_fd, shmem_size);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Can't truncate shmem (host) file\n");
        goto err;
    }

    h_indicators = (volatile fgpu_indicators_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_host_fd, 0);
    if (h_indicators == NULL) {
        fprintf(stderr, "FGPU:Can't map shmem\n");
        ret = -errno;
        goto err;
    }

    cudaFree(0);
    /* 
     * This function needs to be called after a CUDA function is called so that
     * the device context is created in that function. CUDA context is created
     * lazily.
     */
    ret = gpuDriverErrCheck(cuMemHostRegister((void *)h_indicators, shmem_size,
                CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
    if (ret < 0)
        goto err;
    
    ret = gpuErrCheck(cudaHostGetDevicePointer(&d_host_indicators,
                (void *)h_indicators, 0));
    if (ret < 0)
        goto err;

    memset((void *)h_indicators, 0, sizeof(fgpu_indicators_t));


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
    g_host_ctx->last_color = FGPU_INVALID_COLOR;

    ret = init_shared_mutex(&g_host_ctx->streams_lock);
    if (ret < 0)
        goto err;

    for (int i = 0; i < g_host_ctx->num_colors; i++) {
        ret = init_shared_condvar(&g_host_ctx->streams_cond[i]);
        if (ret < 0)
            goto err;
        g_host_ctx->is_stream_free[i] = true;
    }
     
    if (!is_mps_enabled()) {
        fprintf(stderr, "FGPU:MPS is not enabled\n");
        ret = -EIO;
        goto err;
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
    printf("FGPU:Server Terminating. Waiting for device to be free\n");
    gpuErrCheck(cudaDeviceSynchronize());

    if (h_indicators != NULL) {
        cuMemHostUnregister((void *)h_indicators);
        cudaFreeHost((void *)h_indicators);
    }

    if (shmem_host_fd > 0)
        close(shmem_host_fd);
    
    if (shmem_fd > 0)
        close(shmem_fd);

    /* TODO: Check errors here also */
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

    /* Create the shared memory */
    ret = shmem_fd = shm_open(FGPU_SHMEM_NAME, O_RDWR, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Couldn't open shmem\n");
        goto err;
    }

    page_size = sysconf(_SC_PAGE_SIZE);

    shmem_size = ROUND_UP(sizeof(fgpu_host_ctx_t), page_size);
    g_host_ctx = (fgpu_host_ctx_t *)mmap(NULL, shmem_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, shmem_fd, 0);
    if (g_host_ctx == NULL) {
        fprintf(stderr, "FGPU:Can't map shmem\n");
        ret = -errno;
        goto err;
    }

    ret = shmem_host_fd = shm_open(FGPU_SHMEM_HOST_NAME, O_RDWR, S_IRUSR | S_IWUSR);
    if (ret < 0) {
        fprintf(stderr, "FGPU:Couldn't open shmem\n");
        goto err;
    }

    shmem_size = ROUND_UP(sizeof(fgpu_indicators_t), page_size);

    h_indicators = (volatile fgpu_indicators_t *)mmap(NULL, shmem_size,
            PROT_READ | PROT_WRITE, MAP_SHARED, shmem_host_fd, 0);
    if (h_indicators == NULL) {
        fprintf(stderr, "FGPU:Can't map shmem (host pinned)\n");
        ret = -errno;
        goto err;
    }

        cudaFree(0);
    /* 
     * This function needs to be called after a CUDA function is called so that
     * the device context is created in that function. CUDA context is created
     * lazily.
     */
    shmem_size = ROUND_UP(sizeof(fgpu_indicators_t), page_size);
    ret = gpuDriverErrCheck(cuMemHostRegister((void *)h_indicators, shmem_size,
                CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
    if (ret < 0)
        goto err;

    ret = gpuErrCheck(cudaHostGetDevicePointer(&d_host_indicators,
                (void *)h_indicators, 0));
    if (ret < 0)
        goto err;

    /* Create a stream for all operations. Can't use NULL as used by all processes */
    ret = gpuErrCheck(cudaStreamCreateWithFlags(&color_stream, cudaStreamDefault));
    if (ret < 0)
        goto err;

    ret = gpuErrCheck(cudaSetDevice(FGPU_DEVICE_NUMBER));
    if (ret < 0)
        return ret;

    if (!is_mps_enabled()) {
        fprintf(stderr, "FGPU:MPS is not enabled\n");
        ret = -EIO;
        goto err;
    }

    return 0;

err:
    fgpu_deinit();
    return ret;
}

void fgpu_deinit(void)
{

    if (d_bindex)
        fgpu_memory_free((void *)d_bindex);

    if (d_dev_indicator)
        fgpu_memory_free((void *)d_dev_indicator);

#if defined(FGPU_MEM_COLORING_ENABLED)
    fgpu_memory_deinit();
#endif

    cudaStreamDestroy(color_stream);
    if (h_indicators != NULL)
        cuMemHostUnregister((void *)h_indicators);

    if (shmem_host_fd > 0)
        close(shmem_host_fd);
    
    if (shmem_fd > 0)
        close(shmem_fd);
}

static bool is_initialized(void)
{
    return g_host_ctx != NULL;
}

static bool is_color_set(void)
{
    return g_color != FGPU_INVALID_COLOR;
}

/* Returns either the color set via env variable or default value */
int fgpu_get_env_color(void)
{
    const char* tmp = getenv(FGPU_COLOR_ENV_NAME);
    if (!tmp)
        return FGPU_DEFAULT_COLOR;

    return atoi(tmp);
}

/* returns either the size of colored mem set via env variable or default value */
size_t fgpu_get_env_color_mem_size(void)
{
    const char* tmp = getenv(FGPU_COLOR_MEM_SIZE_ENV_NAME);
    if (!tmp)
        return FGPU_DEFAULT_COLOR_MEM_SIZE;

    return (size_t)atoll(tmp);
}

bool fgpu_is_init_complete(void)
{
    return is_initialized();
}

int fgpu_set_color_prop(int color, size_t mem_size)
{
    int ret;

    if (!is_initialized()) {
        fprintf(stderr, "FGPU:fgpu module not initialized\n");
        return -EINVAL;
    }

    if (is_color_set()) {
        fprintf(stderr, "FGPU:Color can be only set once\n");
        return -EINVAL;
    }

    if (color >= g_host_ctx->num_colors || color < 0) {
        fprintf(stderr, "FGPU:Invalid color\n");
        return -EINVAL;
    }

#ifdef FGPU_MEM_COLORING_ENABLED
    ret = fgpu_memory_set_colors_info(FGPU_DEVICE_NUMBER, color,
            mem_size, color_stream);
    if (ret < 0)
        return ret;
#endif

    g_color = color;

    ret = fgpu_memory_allocate((void **)&d_bindex, sizeof(struct fgpu_bindex));
    if (ret < 0)
        return ret;

    ret = fgpu_memory_allocate((void **)&d_dev_indicator, sizeof(struct fgpu_bindex));
    if (ret < 0)
        return ret;


    ret = fgpu_memory_memset_async((void *)d_bindex, 0, sizeof(struct fgpu_bindex));
    if (ret < 0)
        goto err;

    ret = fgpu_memory_memset_async((void *)d_dev_indicator, 0, sizeof(struct fgpu_bindex));
    if (ret < 0)
        goto err;


    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        goto err;

    return 0;

err:
    if (d_bindex) {
        fgpu_memory_free((void *)d_bindex);
        d_bindex = NULL;
    }
    
    if (d_dev_indicator) {
        fgpu_memory_free((void *)d_dev_indicator);
        d_dev_indicator = NULL;
    }

    g_color = FGPU_INVALID_COLOR;

    return ret;
}

bool fgpu_is_color_prop_set(void)
{
    if (!is_initialized())
        return false;

    return is_color_set();
}

/* Wait for last launched kernel to be completely started */
static void wait_for_last_start(void)
{
#if defined(FGPU_SERIALIZED_LAUNCH)
    pthread_mutex_lock(&g_host_ctx->launch_lock);
    while (1) {
        if (g_host_ctx->is_lauchpad_free)
            break;
        pthread_cond_wait(&g_host_ctx->launch_cond, &g_host_ctx->launch_lock);
    }
    g_host_ctx->is_lauchpad_free = false;
    pthread_mutex_unlock(&g_host_ctx->launch_lock);

#if defined(FGPU_COMPUTE_CHECK_ENABLED)
    int num_active_pblocks = 0;
#endif

    /* Wait for all pblocks to be accounted for */
    for (int i = 0; i < g_host_ctx->last_num_pblocks_launched; i++) {
        
        int __attribute__((unused))val;
        
        while (h_indicators->indicators[i].started == FGPU_NOT_PBLOCK_NOT_STARTED);
        
        val = h_indicators->indicators[i].started;
    
#if defined(FGPU_COMPUTE_CHECK_ENABLED)
        assert(val == FGPU_ACTIVE_PBLOCK_STARTED ||
                val == FGPU_INACTIVE_PBLOCK_STARTED);
        
        if (val == FGPU_ACTIVE_PBLOCK_STARTED)
            num_active_pblocks++;
#else
        assert(val == FGPU_GENERIC_PBLOCK_STARTED);
#endif

        h_indicators->indicators[i].started = FGPU_NOT_PBLOCK_NOT_STARTED;
    }

#if defined(FGPU_COMPUTE_CHECK_ENABLED)
    /* Some inactive blocks appear as active blocks */
    assert(num_active_pblocks >= g_host_ctx->last_num_active_pblocks);
#endif

#endif
}

/* 
 * Called when cuda stream operation completes. 
 * Unforntunately, currently Nvidia does't provide with stream callbacks with MPS
 */
static void stream_callback(int color)
{
    pthread_mutex_lock(&g_host_ctx->streams_lock);
    g_host_ctx->is_stream_free[color] = true;
    pthread_cond_signal(&g_host_ctx->streams_cond[color]);
    pthread_mutex_unlock(&g_host_ctx->streams_lock);
}

/* Overloaded function to get blockDim and gridDim */
void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, int _gridDim, int _blockDim)
{
    ctx->gridDim = dim3(_gridDim, 1, 1);
    ctx->blockDim = dim3(_blockDim, 1, 1);
}

void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, dim3 _gridDim, dim3 _blockDim)
{
    ctx->gridDim = _gridDim;
    ctx->blockDim = _blockDim;
}

void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, uint3 _gridDim, uint3 _blockDim)
{
    ctx->gridDim = _gridDim;
    ctx->blockDim = _blockDim;
}

/* Called after kernel has been launched */
int fgpu_complete_launch_kernel(fgpu_dev_ctx_t *ctx)
{
    int ret;

    if (!is_color_set()) {
        fprintf(stderr, "FGPU:Colors not set\n");
        return -EINVAL;
    }

#if defined(FGPU_SERIALIZED_LAUNCH)
    g_host_ctx->last_color = ctx->color;
    g_host_ctx->last_num_pblocks_launched = ctx->num_pblock;
    g_host_ctx->last_num_active_pblocks = ctx->num_active_pblocks;

    pthread_mutex_lock(&g_host_ctx->launch_lock);
    g_host_ctx->is_lauchpad_free = true;
    pthread_cond_signal(&g_host_ctx->launch_cond);
    pthread_mutex_unlock(&g_host_ctx->launch_lock);
#endif

    ret = gpuErrCheck(cudaStreamSynchronize(color_stream));

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
int fgpu_prepare_launch_kernel(fgpu_dev_ctx_t *ctx, const void *func,
        size_t shared_mem, dim3 *_gridDim, cudaStream_t **stream)
{
    uint32_t num_blocks;
    uint32_t num_threads;
    uint32_t num_pblocks;
    int num_pblocks_per_sm;
    int ret;

    if (!is_color_set()) {
        fprintf(stderr, "FGPU:Colors not set\n");
        return -1;
    }

    num_blocks = ctx->gridDim.x * ctx->gridDim.y * ctx->gridDim.z;
    if (num_blocks == 0) {
        fprintf(stderr, "FGPU:Invalid number of blocks\n");
        return -EINVAL;
    }

    num_threads = ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z;
    if (num_threads == 0 || num_threads > g_host_ctx->max_num_threads_per_sm) {
        fprintf(stderr, "Invalid number of threads in a block\n");
        return -EINVAL;
    }

    ret = 
        gpuErrCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&num_pblocks_per_sm,
                    func, num_threads, shared_mem, cudaOccupancyDisableCachingOverride));
    if (ret < 0)
        return ret;

    if (num_pblocks_per_sm == 0) {
        fprintf(stderr, "FGPU:Invalid grid/block/thread configuration\n");
        return -EINVAL;
    }

    num_pblocks = num_pblocks_per_sm * g_host_ctx->num_sm;

    if (num_pblocks > FGPU_MAX_NUM_PBLOCKS) {
        fprintf(stderr, "FGPU:FGPU_MAX_NUM_PBLOCKS is set too low\n");
        return -EINVAL;
    }

    wait_for_last_complete(g_color);

    wait_for_last_start();

    ctx->color = g_color;
    ctx->num_pblock = num_pblocks;
    ctx->num_blocks = num_blocks;
    ctx->index = cur_index;
    cur_index ^= 1; /* Toggle the index */
    ctx->d_host_indicators = d_host_indicators;
    ctx->d_dev_indicator = d_dev_indicator;
    ctx->d_bindex = d_bindex;
    ctx->start_sm = g_host_ctx->color_to_sms[g_color].first;
    ctx->end_sm = g_host_ctx->color_to_sms[g_color].second;
    ctx->num_active_pblocks = num_pblocks_per_sm * (ctx->end_sm - ctx->start_sm + 1);


#if defined(FGPU_USER_MEM_COLORING_ENABLED)
    ret = fgpu_get_memory_info(&ctx->start_virt_addr, &ctx->start_idx);
    if (ret < 0)
        return ret;
#endif

    _gridDim->x = num_pblocks;
    _gridDim->y = 1;
    _gridDim->z = 1;
    *stream = &color_stream;

    return 0;
}

int fgpu_color_stream_synchronize(void)
{
#ifdef FGPU_COMP_COLORING_ENABLE
    if (!is_color_set()) {
        fprintf(stderr, "FGPU:Colors not set\n");
        return -EINVAL;
    }

    return gpuErrCheck(cudaStreamSynchronize(color_stream));
#else
    return gpuErrCheck(cudaDeviceSynchronize());
#endif
}

int fgpu_num_sm(int color, int *num_sm)
{
    if (!is_initialized()) {
        fprintf(stderr, "FGPU:fgpu module not initialized\n");
        return -EINVAL;
    }

    if (color >= g_host_ctx->num_colors) {
        fprintf(stderr, "FGPU: Invalid Color\n");
        return -EINVAL;
    }

    *num_sm = g_host_ctx->color_to_sms[color].second - 
        g_host_ctx->color_to_sms[color].first + 1;
    return 0;
}

int fgpu_num_colors(void)
{
    if (!is_initialized()) {
        fprintf(stderr, "FGPU:fgpu module not initialized\n");
        return -EINVAL;
    }

    return g_host_ctx->num_colors;
}

int fgpu_memory_copy_async(void *dst, const void *src, size_t count,
                           enum fgpu_memory_copy_type type,
                           cudaStream_t stream)
{
    /* 
     * Instead of using default stream (which caused device wide synchronization)
     * Use process specific stream.
     */
    if (stream == NULL)
        stream = color_stream;
    return fgpu_memory_copy_async_internal(dst, src, count, type, stream);
}

int fgpu_memory_memset_async(void *address, int value, size_t count,
                            cudaStream_t stream)
{
    /* 
     * Instead of using default stream (which caused device wide synchronization)
     * Use process specific stream.
     */
    if (stream == NULL)
        stream = color_stream;

    return fgpu_memory_memset_async_internal(address, value, count, stream);
}
