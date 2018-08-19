/* This file handles reverse engineering dram bank addressing for GPU */
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

#include <reverse_engineering.h>

#include "common.h"
#include <fractional_gpu.h>
#include <fractional_gpu_cuda.cuh>

/* Device memory */
static uint64_t *d_sum;
static uint64_t *d_refresh_v;
static double *d_ticks;
static double *h_ticks;
static uint64_t *h_a;

/* 
 * Read enough data to implicitly flush L2 cache.
 * Uses p-chase to make sure compiler/hardware doesn't optimize away the code.
 */
__device__
uint64_t refresh_l2(volatile uint64_t *refresh_vaddr)
{
    uint64_t curindex = 0;
    uint64_t sum = 0;

    while (curindex != (uint64_t)-1) {
        curindex = refresh_vaddr[curindex];
        sum += curindex;
    }
    return sum;
}

/* 
 * Reads data from a_v and b_v arrays together and measure time. This can be
 * used to see if they both lie on same dram bank.
 * XXX: This function for some reason doesn't work correctly for all pairs
 * of a_v, b_v. It is giving false negatives.
 */
__global__
void read_dram_pair(volatile uint64_t *a_v, volatile uint64_t *b_v,
        volatile uint64_t *refresh_v, volatile double *ticks, volatile uint64_t *psum,
        double threshold)
{
    uint64_t curindex;
    uint64_t sum;
    uint64_t count;
    uint64_t mid;
    uint64_t previndex;
    const uint64_t sharednum = 1;
    __shared__ uint64_t s[sharednum];
    __shared__ uint64_t t[sharednum];
    uint64_t tsum;
    int i;
    double tick;

    for (i = 0; i < GPU_MAX_OUTER_LOOP + 1; i++) {

        sum = 0;
        curindex = 0;

        /* Evict all data from L2 cache */
        sum += refresh_l2(refresh_v);

        /* 
         * Measure time to read two different addresses together. If lie on
         * same bank, different rows, we expect to see a jump in time
         */
        while (curindex != (uint64_t)-1) {
            previndex = curindex;
            mid = clock64();
            sum += b_v[curindex];
            curindex = a_v[curindex];
            s[previndex] = curindex;
            t[previndex] = clock64() - mid;
        }
    
        /* Some extra code to make sure hardware/compiler doesn't optimize code */
        curindex = 0;
        tsum = 0;
        count = 0;
        while (curindex != (uint64_t)-1) {
            count++;
            tsum += t[curindex];
            curindex = s[curindex];
        }
       
        /* First run is warmup - Effects like TLB miss might impact*/
	    if (i == 0)
    	    continue;

        tick = ((double)tsum) / ((double)count);

        if (tick > threshold) {
            /* We don't expect threshold to be crossed on GPU (No timer interrupts) */
            printf("ERROR: Threshold:%f, Ticks:%f, i:%d, count: %ld\n", threshold, tick, i, count);
            i--;
            continue;
        }
        
        ticks[i - 1] = tick;
        psum[i - 1] = sum;
    }
}

/* 
 * Initialize the pointer chase for refresh vaddr to hinder an hardware
 * mechanism to predict access pattern.
 */
static void init_pointer_chase(uint64_t *array, size_t size, size_t offset)
{
    uint64_t num_elem = size / offset;
    uint64_t curindex;
    uint64_t i;

    assert(offset >= sizeof(uint64_t));
    assert((offset % sizeof(uint64_t)) == 0);

    for (i = 0, curindex = 0; i < num_elem; i++) {
        uint64_t nextindex;
        if (i == num_elem - 1)
            nextindex = (uint64_t)-1;
        else
            nextindex = curindex + (offset / sizeof(uint64_t));
        array[curindex] = nextindex;
        curindex = nextindex;
    }
}

/* Allocates contiguous memory and returns the starting physical address */
/* For this to work, Nvidia driver must be configured properly. */
void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p)
{
    size_t size = contiguous_size;
    void *gpu_mem;
    void *phy_start;
    int ret;
    
    ret = fgpu_memory_allocate(&gpu_mem, size);
    if (ret < 0)
        return NULL;
   
    phy_start = fgpu_memory_get_phy_address(gpu_mem);
    if (!phy_start)
        return NULL;

    *phy_start_p = phy_start;

    return gpu_mem;
}

/* Find the average time to read from arrays a and b together (from DRAM) */
double device_find_dram_read_time(void *_a, void *_b, double threshold)
{
    uint64_t *a = (uint64_t *)_a;
    uint64_t *b = (uint64_t *)_b;
    int i;
    double min_ticks, max_ticks, sum_ticks;

    gpuErrAssert(cudaMemcpy(a, h_a, GPU_L2_CACHE_LINE_SIZE, cudaMemcpyHostToDevice));
    read_dram_pair<<<1,1>>>(a, b, d_refresh_v, d_ticks, d_sum, threshold);
    gpuErrAssert(cudaDeviceSynchronize());
    gpuErrAssert(cudaMemcpy(h_ticks, d_ticks, GPU_MAX_OUTER_LOOP * sizeof(double), cudaMemcpyDeviceToHost));

    for (i = 0, min_ticks = LONG_MAX, sum_ticks = 0, max_ticks = 0; 
            i < GPU_MAX_OUTER_LOOP; i++) {
        double tick = h_ticks[i];

        assert(tick > 0);
		
        min_ticks = tick < min_ticks ? tick : min_ticks;
        max_ticks = tick > max_ticks ? tick : max_ticks;
        sum_ticks += tick;
    }

    dprintf("Min Ticks: %0.3f,\tAvg Ticks: %0.3f,\tMax Ticks: %0.3f\n",
            min_ticks, (sum_ticks * 1.0f) / GPU_MAX_OUTER_LOOP, max_ticks);
    /* Min ticks are more reliable source. Avg gets influences my outliers */
    return min_ticks; 
}

int device_init(size_t req_reserved_size, size_t *reserved_size)
{
    cudaDeviceProp deviceProp;
    size_t l2_size, global_memory, resv_memory;
    static uint64_t *h_refresh_v;
    size_t max_len;
    int num_colors;
    int ret;

    ret = fgpu_init();
    if (ret < 0) {
        fprintf(stderr, "fgpu_init() failed\n");
        return ret;
    }

    gpuErrAssert(cudaGetDeviceProperties(&deviceProp, 0));
    l2_size = deviceProp.l2CacheSize;

    if (l2_size < GPU_L2_CACHE_LINE_SIZE) {
        fprintf(stderr, "Invalid value for GPU_L2_CACHE_LINE_SIZE\n");
        return -1;
    }

    global_memory = deviceProp.totalGlobalMem;
    if (global_memory < (1ULL << GPU_MAX_BIT)) {
        fprintf(stderr, "Invalid value for GPU_MAX_BIT\n");
        return -1;
    }

    ret = fgpu_memory_get_device_info(&num_colors, &max_len);
    if (ret < 0)
        return ret;

    assert(num_colors == 1);

    /* Calculate the amount of memory that needs to be reserved */
    resv_memory = min(req_reserved_size, max_len);
    
    if (reserved_size)
        *reserved_size = resv_memory;

    ret = fgpu_set_color_prop(0, resv_memory);
    if (ret < 0) {
        fprintf(stderr, "fgpu_set_color_prop() failed\n");
        return ret;
    }

    gpuErrAssert(cudaMalloc(&d_refresh_v, l2_size));
    gpuErrAssert(cudaMalloc(&d_ticks, (GPU_MAX_OUTER_LOOP) * sizeof(double)));
    gpuErrAssert(cudaMalloc(&d_sum, (GPU_MAX_OUTER_LOOP) * sizeof(uint64_t)));
    h_ticks = (double *)malloc((GPU_MAX_OUTER_LOOP) * sizeof(double));
    assert(h_ticks);

    h_refresh_v = (uint64_t *)malloc(l2_size);
    if (!h_refresh_v)
        return -1;

    init_pointer_chase(h_refresh_v, l2_size, GPU_L2_CACHE_LINE_SIZE);
    gpuErrCheck(cudaMemcpy(d_refresh_v, h_refresh_v, l2_size, cudaMemcpyHostToDevice));

    free(h_refresh_v);

    h_a = (uint64_t *)malloc(GPU_L2_CACHE_LINE_SIZE);
    assert(h_a);

    init_pointer_chase(h_a, GPU_L2_CACHE_LINE_SIZE, GPU_L2_CACHE_LINE_SIZE);

    return (1 << GPU_MIN_BIT);
}

/* Any contiguous allocation can't be fully used. There is some internal overhead */
size_t device_allocation_overhead(void)
{
    return 2 * GPU_L2_CACHE_LINE_SIZE;
}
