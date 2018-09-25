/* This file handles reverse engineering dram bank addressing for GPU */

/* TODO: Instead of using any cuda function, use all FGPU functions */

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

#include <reverse_engineering.hpp>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#if !defined(FGPU_TEST_MEM_COLORING_ENABLED)
#error "FGPU_TEST_MEM_COLORING_ENABLED not defined. Needed for reverse engineering"
#endif

/* Device memory */
static uint64_t *d_sum;
static uint64_t *d_refresh_v;
static double *d_ticks;
static uint64_t **d_last_addr;
static double *h_ticks;
static uint64_t *h_a;
static uint64_t *d_count;
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

__global__
void gpu_pchase_setup(uint64_t *array, size_t size, size_t offset)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_elem = size / offset;
    if (index < num_elem - 1) {
        array[offset * index / sizeof(uint64_t)] = offset * (index + 1) / sizeof(uint64_t);
    }

    if (index == num_elem - 1) {
        array[offset * index / sizeof(uint64_t)] = (uintptr_t)-1;
    }
}

void gpu_init_pointer_chase(uint64_t *array, size_t size, size_t offset)
{
    uint64_t num_elem = size / offset;
    size_t num_threads = 32;
    size_t num_blocks = (num_elem + num_threads - 1) / num_threads;
    
    assert(offset >= sizeof(uint64_t));
    assert((offset % sizeof(uint64_t)) == 0);

    gpu_pchase_setup<<<num_blocks, num_threads>>>(array, size, offset);
    gpuErrAssert(cudaDeviceSynchronize());
}

/* Modifies one element of p-chase array */
__global__
void gpu_modify_pointer_chase(uint64_t *ct_min_addr, uint64_t index)
{
    *ct_min_addr = index;
}

int device_init(size_t req_reserved_size, size_t *reserved_size)
{
    cudaDeviceProp deviceProp;
    size_t l2_size, resv_memory;
    static uint64_t *h_refresh_v;
    size_t max_len;
    int num_colors;
    int ret;

    ret = fgpu_init();
    if (ret < 0) {
        fprintf(stderr, "fgpu_init() failed\n");
        return ret;
    }

    gpuErrAssert(cudaGetDeviceProperties(&deviceProp, FGPU_DEVICE_NUMBER));
    l2_size = deviceProp.l2CacheSize;

    if (l2_size < GPU_L2_CACHE_LINE_SIZE) {
        fprintf(stderr, "Invalid value for GPU_L2_CACHE_LINE_SIZE\n");
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
    gpuErrAssert(cudaMalloc(&d_last_addr, sizeof(void *)));
    gpuErrAssert(cudaMalloc(&d_count, sizeof(uint64_t)));
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

    return 0;
}

size_t device_allocation_overhead(void)
{
    return 2 * GPU_L2_CACHE_LINE_SIZE;
}

/* Finds the maximum physical bit of GPU - Based on total GPU physical memory */
int device_max_physical_bit(void)
{
    size_t free, total;
    gpuErrAssert(cudaMemGetInfo(&free, &total));

    /* Find highest bit set in total size */
    for (int i = sizeof(size_t) * 8 - 1; i >=0; i--) {
        if (total & (1ULL << i))
            return i;
    }

    return -1;
}

/* 
 * Finds the minimum physical bit of GPU that matters for hash function 
 * Based on the size of cache line. Anything bit less than log2(cache_line)
 * is not of consequence
 */
int device_min_physical_bit(void)
{
    return ilog2((unsigned long long)GPU_L2_CACHE_LINE_SIZE);
}
/**************** DRAM BANK HASH FUNCTION HELPER FUNCTIONS ********************/

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

/**************** CACHE LINE HASH FUNCTION HELPER FUNCTIONS *******************/
/* 
 * Keeps reading data from array till a cacheline eviction is noticed.
 * Base is the base address of the in-device array. (P-chase array)
 * Start count is the count from which to start searching.
 * End address is the address till where to keep reading.
 * Threshold is used to judge if an eviction took place.
 * Ret Addr is the address that caused eviction.
 * Ret Count is the count taken to find the current solution.
 * Psum is just to avoid compiler optimizations.
 */
__global__
void evict_cacheline(volatile uint64_t *base, uint64_t start_count, 
        volatile uint64_t *end_addr, uint64_t threshold, 
        uint64_t **ret_addr, uint64_t *ret_count, volatile uint64_t *psum)
{
    uint64_t curindex;
    volatile uint64_t *lastaddr;
    uint64_t count;
    uint64_t sum;
    uint64_t start_ticks, ticks;
    int confirm = 0;

#pragma unroll 1
    for (sum = 0, count = start_count, lastaddr = base; (uintptr_t)lastaddr < (uintptr_t)end_addr; count++) {

        lastaddr = base;
        curindex = __ldcs((uint64_t *)base);

#pragma unroll 1
        /* Read bunch of words */
        for (int i = 0; (i < count) && ((uintptr_t)lastaddr < (uintptr_t)end_addr); i++) {
            lastaddr = &base[curindex];
            curindex = __ldcs((uint64_t *)lastaddr);
            sum += curindex;
        }

        /* Read first word */
        start_ticks = clock64();
        curindex = __ldcs((uint64_t *)base);
        sum += curindex;
        ticks = clock64() - start_ticks;

        /* 
         * Has the word been evicted 
         * (Reading first time might seem like cache eviction due to cold miss)
         */
        if (ticks >= threshold && lastaddr != base) {
            confirm++;
            count--;
            
            /* Check multiple times if valid solution (to avoid noise effects)*/
            if (confirm == GPU_MAX_OUTER_LOOP) {
                *ret_addr = (uint64_t *)lastaddr;
                *ret_count = count;
                *psum = sum;
                return;
            }   
        } else {
            confirm = 0;
        }  

    }

    /* Couldn't find an address that causes eviction */
    *psum = sum;
    *ret_addr = NULL;
    *ret_count = 0;
}

/*
 * Returns the time to read word from cache
 */
__global__
void cacheline_read_time(volatile uint64_t *base, volatile uint64_t *end_addr,
        double *ticks, volatile uint64_t *psum)
{
    uint64_t curindex;
    volatile uint64_t *lastaddr;
    uint64_t count;
    uint64_t sum;
    uint64_t start_ticks, tick;

    for (sum = 0, count = 0; count < GPU_MAX_OUTER_LOOP + 1; count++) {
   
        /* Read first word */
        start_ticks = clock64();
        curindex = *base;
        sum += curindex;
        tick = clock64() - start_ticks;
        
        /* First read might be cold miss */
        if (count != 0)
            ticks[count - 1] = tick;

        lastaddr = base;
        /* Read bunch of words */
        for (int i = 0; i < count && ((uintptr_t)lastaddr < (uintptr_t)end_addr); i++) {
            lastaddr = &base[curindex];
            curindex = *lastaddr;
            sum += curindex;
        }
    }

    *psum = sum;
}

static uintptr_t ct_start_addr;
static uintptr_t ct_end_addr;
static uintptr_t ct_last_start_addr;        // Last start address for search
static uintptr_t ct_start_count;
static size_t ct_offset;

int device_cacheline_test_init(void *gpu_start_addr, size_t size)
{
    ct_start_addr = (uintptr_t)gpu_start_addr;
    ct_end_addr = ct_start_addr + size - 1;
    ct_offset = GPU_L2_CACHE_LINE_SIZE;

    gpu_init_pointer_chase((uint64_t *)gpu_start_addr, size, ct_offset);

    ct_last_start_addr = ct_start_addr;
    ct_start_count = 0;

    return 0;
}

/* Finds the avg/min/max for cacheline test */
int device_cacheline_test_find_threshold(size_t sample_size, double *avg)
{
    int i, j;
    double min_ticks, max_ticks, sum_ticks;
    double total_sum_ticks = 0;

    for (i = 0; i < sample_size; i++) {
        cacheline_read_time<<<1,1>>>((uint64_t *)ct_start_addr, (uint64_t *)ct_end_addr, d_ticks, d_sum);
        gpuErrAssert(cudaDeviceSynchronize());
        gpuErrAssert(cudaMemcpy(h_ticks, d_ticks, GPU_MAX_OUTER_LOOP * sizeof(double), cudaMemcpyDeviceToHost));

        for (j = 0, min_ticks = LONG_MAX, sum_ticks = 0, max_ticks = 0; 
                j < GPU_MAX_OUTER_LOOP; j++) {
            double tick = h_ticks[j];

            assert(tick > 0);
            
            min_ticks = tick < min_ticks ? tick : min_ticks;
            max_ticks = tick > max_ticks ? tick : max_ticks;
            sum_ticks += tick;
        }

        dprintf("Min Ticks: %0.3f,\tAvg Ticks: %0.3f,\tMax Ticks: %0.3f\n",
                min_ticks, (sum_ticks * 1.0f) / GPU_MAX_OUTER_LOOP, max_ticks);

        /* Min ticks are more reliable source. Avg gets influences my outliers */
        total_sum_ticks += min_ticks; 
    }

    *avg = total_sum_ticks / sample_size;
    return 0;
}

/*
 * Given an address '_a', finds another address that evicts a word at '_a'
 * from cache. Search for such an address is started after '_b' address and at
 * an offset of 'offset'
 */
void *device_find_cache_eviction_addr(void *_a, void *_b, size_t offset, double threshold)
{
    uintptr_t a = (uintptr_t)_a;
    uintptr_t b = (uintptr_t)_b;
    uint64_t index;
    uint64_t *last_addr;
    
    /* Currently only support one fixed address */
    if (a != ct_start_addr)
        return NULL;

    if (b < ct_start_addr || b > ct_end_addr)
        return NULL;

    /* Currently we only support one fixed offset */
    if (offset != ct_offset)
        return NULL;

    /* b needs to be subsequently keep increasing only */
    if (b < ct_last_start_addr)
        return NULL;

    if (b != ct_last_start_addr && b != ct_last_start_addr + ct_offset) {
        index = (b - ct_start_addr) / sizeof(uint64_t);
        if (index == 0)
            index += ct_offset / sizeof(uint64_t);

        gpu_modify_pointer_chase<<<1, 1>>>((uint64_t *)ct_start_addr, index);
        gpuErrAssert(cudaDeviceSynchronize());
        
        ct_last_start_addr = b;
        ct_start_count = 0;
    }

    evict_cacheline<<<1,1>>>((uint64_t *)ct_start_addr, ct_start_count,
            (uint64_t *)ct_end_addr, threshold, d_last_addr, d_count, d_sum);
    gpuErrAssert(cudaDeviceSynchronize());
    gpuErrAssert(cudaMemcpy(&last_addr, d_last_addr, sizeof(uint64_t *), cudaMemcpyDeviceToHost));
    gpuErrAssert(cudaMemcpy(&ct_start_count, d_count, sizeof(uint64_t *), cudaMemcpyDeviceToHost));


    /* Edit p-chase so as to skip last address next time */
    index = ((uintptr_t)last_addr + ct_offset - ct_start_addr) / sizeof(uint64_t);
    gpu_modify_pointer_chase<<<1, 1>>>((uint64_t *)((uintptr_t)last_addr - ct_offset), index);
    gpuErrAssert(cudaDeviceSynchronize());

    ct_last_start_addr = (uintptr_t)last_addr;

    return (void *)last_addr;
}
