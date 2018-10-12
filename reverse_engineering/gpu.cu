/* This file handles reverse engineering dram bank addressing for GPU */

/* TODO: Instead of using any cuda function, use all FGPU functions */

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

#include <vector>

#include <reverse_engineering.hpp>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#if !defined(FGPU_TEST_MEM_COLORING_ENABLED)
#error "FGPU_TEST_MEM_COLORING_ENABLED not defined. Needed for reverse engineering"
#endif

/* Device memory */
static uint64_t *d_sum;
static uint64_t *d_refresh_v;
static size_t max_custom_pchase_entires;
static uint64_t **d_custom_pchase;
static uint64_t **h_custom_pchase;
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

/* Creates a pchase array using base address as 'array' and addresses in 'addresses' */
__global__
void gpu_custom_pchase_setup(uint64_t *array, uint64_t **addresses, size_t num_entries)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uintptr_t offset = (uintptr_t)addresses[index] - (uintptr_t)array;

    if (index < num_entries - 1) {
        uint64_t val = (uintptr_t)addresses[index + 1] - (uintptr_t)array;

        assert((uintptr_t)addresses[index] >= (uintptr_t)array);
        assert((uintptr_t)addresses[index + 1] > (uintptr_t)addresses[index]);

        array[offset / sizeof(uint64_t)] = val / sizeof(uint64_t);
    }

    if (index == num_entries - 1) {
        array[offset / sizeof(uint64_t)] = (uintptr_t)-1;
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

/* 
 * Creates custom pchase. 
 * Callback function is used to create pchase. Takes an arguemnt and an address.
 * Returns the next address that should be in pchase.
 * Also returns the start and end address of pchase.
 */
void gpu_init_custom_pointer_chase(void *base_addr, void *(*cb)(void *addr, void *arg),
        void *arg, int num_entries, void **start_addr, void **end_addr)
{
    size_t found = 0;
    void *next_addr;
    void *prev_addr = base_addr;
    size_t num_threads = 32;
    size_t num_blocks;
    size_t max_entries = std::min((size_t)num_entries, max_custom_pchase_entires);

    for (found = 0; found < max_entries; found++) {

        next_addr = cb(prev_addr, arg);
        if (!next_addr)
            break;

        assert((uintptr_t)next_addr > (uintptr_t)prev_addr);

        h_custom_pchase[found] = (uint64_t *)next_addr;
        prev_addr = next_addr;
    }

    gpuErrAssert(cudaMemcpy(d_custom_pchase, h_custom_pchase, sizeof(uint64_t *) * found, cudaMemcpyHostToDevice));
    
    assert(found > 0);

    num_blocks = (found + num_threads - 1) / num_threads;
    gpu_custom_pchase_setup<<<num_blocks, num_threads>>>(h_custom_pchase[0], d_custom_pchase, found);
    gpuErrAssert(cudaDeviceSynchronize());

    if (start_addr)
        *start_addr = (void *)h_custom_pchase[0];

    if (end_addr)
        *end_addr = (void *)h_custom_pchase[found - 1];
}

/* 
 * Creates custom pchase for initialization 
 * Callback function is used to create pchase. Takes an arguemnt and an address.
 * Returns the next address that should be in pchase.
 * Also returns the pchase start and end address.
 */
static int device_custom_pchase_init(void *gpu_start_addr, 
        void *(*cb)(void *addr, void *arg), void *arg, int num_entries,
        void **start_addr, void **end_addr)
{
    gpu_init_custom_pointer_chase(gpu_start_addr, cb, arg, num_entries,
            start_addr, end_addr);

    return 0;
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
    size_t overheads = 1024 * 1024; // 1 MB

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

    /* Some reserved memroy might be used up by FGPU API. So take that as an
     * overhead.
     */
    if (reserved_size)
        *reserved_size = resv_memory - overheads;

    ret = fgpu_set_color_prop(0, resv_memory);
    if (ret < 0) {
        fprintf(stderr, "fgpu_set_color_prop() failed\n");
        return ret;
    }

    /* Enough entries in pchase to evict data out of L2 cache */
    max_custom_pchase_entires = (l2_size / GPU_L2_CACHE_LINE_SIZE) * 2;

    gpuErrAssert(cudaMalloc(&d_refresh_v, l2_size));

    gpuErrAssert(cudaMalloc(&d_custom_pchase, max_custom_pchase_entires * sizeof(uint64_t *)));
    h_custom_pchase = (uint64_t **)malloc(max_custom_pchase_entires * sizeof(uint64_t *));
    assert(h_custom_pchase);

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
 * Reads a certain number of elements from P-Chase array and checks if it causes
 * eviction of first element.
 * Base is the base address of the in-device array. (P-chase array)
 * Count is the number of elements to read in pchase
 * Threshold is used to judge if an eviction took place.
 * Reached end is set to true if we have reached the end of the p-chase array
 * Ret Addr is the address that caused eviction.
 * Psum is just to avoid compiler optimizations.
 * XXX: For some reason, if __noinline__ is not give, the compiler optimizes
 * away certain part of the function.
 * XXX: This function seems to be too noisy. For now, we are tring to suppress
 * the noise via repetition. The noise might be because we don't exactly
 * understand eviction policy. Eviction policy seems to be close to LRU but
 * compiler has certain amount of control over the eviction policy.
 */
__device__ __noinline__
bool is_cacheline_evicted(volatile uint64_t *base, uint64_t count, 
                            uint64_t threshold, bool *reached_end,
                            uint64_t **ret_addr, volatile uint64_t *psum)
{
    volatile uint64_t *lastaddr;
    uint64_t sum = 0;
    uint64_t start_ticks, ticks;
    int confirm = 0;
    uint64_t curindex;
    int tries = 0;
    int limit = GPU_MAX_OUTER_LOOP / 2;
    int max_tries = 2 * limit;

    *reached_end = false;

    /* 'tries' is used to avoid livelock due to high noise situation */
    while (confirm < limit && confirm > -1 * limit && tries < max_tries) {
        
        tries++;

        lastaddr = base;
        curindex = __ldcs((uint64_t *)base);

#pragma unroll 1
        /* Read bunch of words */
        for (int i = 0; i < count && curindex != (uint64_t)-1; i++) {
            lastaddr = &base[curindex];
            curindex = __ldcs((uint64_t *)lastaddr);
            sum += curindex;
        }

        if (curindex == (uint64_t)-1) {
            *reached_end = true;
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
            /* Check multiple times if valid solution (to avoid noise effects)*/
            if (confirm == limit) {
                *ret_addr = (uint64_t *)lastaddr;
                *psum = sum;
                return true;
            } else {
                continue;
            }

        } else {
            /* Due to noise, false negatives might crop in */
            confirm--;
            if (confirm == -1 * limit) {
                return false;
            } else {
                continue;
            }   
        }
    }

    return false;
}

/* 
 * Keeps reading data from array till a cacheline eviction is noticed.
 * Base is the base address of the in-device array. (P-chase array)
 * Start count is the count from which to start searching.
 * Threshold is used to judge if an eviction took place.
 * Ret Addr is the address that caused eviction.
 * Ret Count is the count taken to find the current solution.
 * Psum is just to avoid compiler optimizations.
 */
__global__
void evict_cacheline(volatile uint64_t *base, uint64_t start_count, 
        uint64_t threshold, uint64_t **ret_addr, uint64_t *ret_count, 
        volatile uint64_t *psum)
{
    uint64_t local_sum, sum = 0;
    bool reached_end = false;
    uint64_t count, offset, lower_bound, upper_bound;
    uint64_t *addr, *correct_addr = NULL;
    
    /* 
     * Using binary search to find the lastaddr that causes eviction
     * We don't know the upper bound. So start with doubling lower bound till
     * we find the upper bound and then narrow in.
     * Lower bound tracks the last tested value that didn't cause eviction.
     * Upper bound tracks the value of count we are testing currently. 
     * If successful eviction, the value we want is in (lower_bound, upper_bound]
     */

    offset = 1;
    lower_bound = start_count - 1;
    upper_bound = count = start_count;

#pragma unroll 1
    while (is_cacheline_evicted(base, count, threshold, &reached_end,
                &addr, &local_sum) == false && reached_end == false) {

        lower_bound = count;
        upper_bound = count = start_count + offset;
        offset *= 2;

        sum += local_sum;
    }

    if (reached_end)
        goto err;

    correct_addr = addr;

    /* Now do reverse binary search between the two bounds */
    while (lower_bound < upper_bound) {
  
        if (upper_bound == lower_bound + 1)
            goto success;

        count = (lower_bound + upper_bound) / 2;

        if (is_cacheline_evicted(base, count, threshold, &reached_end,
                    &addr, &local_sum) == false) {

            lower_bound = count;
        } else {

            correct_addr = addr;
            upper_bound = count;
            correct_addr = addr;
        }
    }

err:
    /* Couldn't find an address that causes eviction */
    *psum = sum;
    *ret_addr = NULL;
    *ret_count = 0;

success:
    /* Just checks */
    assert(is_cacheline_evicted(base, upper_bound, threshold, &reached_end,
                &addr, &local_sum) == true);
    assert(is_cacheline_evicted(base, lower_bound, threshold, &reached_end,
                &addr, &local_sum) == false);
    /* Success */
    *psum = sum;
    *ret_addr = correct_addr;
    *ret_count = upper_bound;
    return;
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
static size_t ct_num_words;                 // Number of words in a cacheline

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
 * Requires the pchase to be contiguous (not user defined)
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
        
        /* Round up b */
        b = ct_start_addr + (((b - ct_start_addr) + ct_offset - 1) / ct_offset) * ct_offset;

        index = (b - ct_start_addr) / sizeof(uint64_t);
        if (index == 0)
            index += ct_offset / sizeof(uint64_t);

        /* 
         * NOTE: We are modifying the second element (not the first).
         * This is avoid writing to first element (which might mess up the cache)
         * This is because even though the L2 Cache is LRU, it behaves differently
         * for reads and writes.
         */
        gpu_modify_pointer_chase<<<1, 1>>>((uint64_t *)ct_start_addr + ct_offset, index);
        gpuErrAssert(cudaDeviceSynchronize());
        
        ct_last_start_addr = b;
        ct_start_count = 0;
    }

    evict_cacheline<<<1,1>>>((uint64_t *)ct_start_addr, ct_start_count,
            threshold, d_last_addr, d_count, d_sum);
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

/*
 * Finds the number of words in a cacheline. Requires that device must be initilized with
 * cacheline specific pchase.
 */
int device_find_cacheline_words_count(void *gpu_start_addr, double threshold,
        void *(*cb)(void *addr, void *arg), void *arg, size_t *words)
{
    uint64_t count;
    int ret;
    void *start_addr;

    ret = device_custom_pchase_init(gpu_start_addr, cb, arg, INT_MAX, &start_addr, NULL);
    if (ret < 0)
        return ret;

    evict_cacheline<<<1,1>>>((uint64_t *)start_addr, 0, threshold, d_last_addr, 
            d_count, d_sum);
    gpuErrAssert(cudaDeviceSynchronize());
    gpuErrAssert(cudaMemcpy(&count, d_count, sizeof(uint64_t *), cudaMemcpyDeviceToHost));

    ct_num_words = count;
    *words = count;
    return 0;
}

/************************** MISCELLENOUS HELPER FUNCTIONS ***********************/

/* Only allow one block to run on SM0. Rest intereferning blocks should run
 * on other sms.
 */
__global__
void do_interference_exp(volatile uint64_t **bases, int num_interferening_blocks,
        int *sm0_blocks_count, int *interfereing_blocks_count, int loop_count,
        double *out_ticks, volatile uint64_t *psum)
{
    volatile uint64_t *base, *addr;
    uint64_t sum = 0;
    double start_ticks, ticks, max_ticks, min_ticks, sum_ticks;
    double avg_ticks;
    uint64_t curindex;
    uint sm;
    int i, count;
    
    asm("mov.u32 %0, %smid;" : "=r"(sm));

    base = bases[blockIdx.x];

    sum_ticks = 0;
    max_ticks = 0;
    min_ticks = INT_MAX;

    if (sm == 0) {
        /* Only allow one block in SM0 */
        int index = atomicAdd(sm0_blocks_count, 1);
        if (index != 0)
            return;

        base = bases[0];

#pragma unroll 1
        for (i = 0; i < loop_count; i++) {
            /* Read bunch of words */
            start_ticks = clock64();

            for (curindex = 0, count = 0; curindex != (uint64_t)-1; count++) {
                addr = &base[curindex];
                curindex = __ldcs((uint64_t *)addr);
                sum += curindex;
            }

            ticks = (clock64() - start_ticks) / count;;

            sum_ticks += ticks;
            max_ticks = ticks > max_ticks ? ticks : max_ticks;
            min_ticks = ticks < min_ticks ? ticks : min_ticks;
        }

        avg_ticks = (double)sum_ticks / loop_count;
        *out_ticks = avg_ticks;

    } else {
        /* Keep control over total number of interfering blocks */
        int index = atomicAdd(interfereing_blocks_count, 1);
        if (index >= num_interferening_blocks)
            return;

        base = bases[index + 1];

#pragma unroll 1
        /* Let the interference run for more time */
        for (int i = 0; i < 2 * loop_count; i++) {

            start_ticks = clock64();

            for (curindex = 0, count = 0; curindex != (uint64_t)-1; count++) {
                addr = &base[curindex];
                curindex = __ldcs((uint64_t *)addr);
                sum += curindex;
            }

            ticks = (clock64() - start_ticks) / count;;

            sum_ticks += ticks;
            max_ticks = ticks > max_ticks ? ticks : max_ticks;
            min_ticks = ticks < min_ticks ? ticks : min_ticks;
        }

        avg_ticks = (double)sum_ticks / 2 / loop_count;
    }

    if (blockIdx.x == 0)
        dprintf("Interference Exp: Block:%d, Loop Count:%d, Num Blocks:%d, "
                "Avg Ticks: %f, Min Ticks:%f, Max Ticks:%f\n",
                blockIdx.x, loop_count, gridDim.x,
                (double)avg_ticks, (double)min_ticks, (double)max_ticks);

    *psum = sum;
}


/*
 * Given two pchases, run experiment for measuring time.
 * Experiment: One thread per SM executes. SM0 runs the first pchase in a loop.
 * Other SMs run the secondary pchase in a loop.
 * Number of SMs is increases monotonically from 1 - Number of SMs.
 * Time taken by primary SM to access a single word is returned across multiple
 * interfering SMs.
 */
int device_run_interference_exp(void *gpu_start_addr, void *(*cb)(void *addr, void *arg),
        void *primary_arg, void *secondary_arg, int max_blocks, int loop_count, 
        std::vector<double> &time)
{
    void *start_addr;
    int ret;
    double ticks;
    int num_entries;
    uint64_t **d_base_address, **h_base_address;
    int i;
    int *d_inteferening_blocks, *d_sm0_blocks;


    /* 
     * If we know how many words are in cacheline, then use one more than that.
     * This ensures all data accesses go to DRAM (so Bank conflict issues can
     * be seen)
     */
    if (ct_num_words == 0)
        num_entries = INT_MAX;
    else
        num_entries = ct_num_words + 1;

    gpuErrAssert(cudaMalloc(&d_inteferening_blocks, sizeof(int)));
    gpuErrAssert(cudaMalloc(&d_sm0_blocks, sizeof(int)));
    gpuErrAssert(cudaMalloc(&d_base_address, max_blocks * sizeof(uint64_t *)));
    h_base_address = (uint64_t **)malloc(max_blocks * sizeof(uint64_t *));
    assert(h_base_address);

    for (start_addr = gpu_start_addr, i = 0; i < max_blocks; i++) {
        
        void *arg;
        void *pchase_start_addr, *pchase_end_addr;

        if (i == 0)
            arg = primary_arg;
        else
            arg = secondary_arg;

        ret = device_custom_pchase_init(start_addr, cb, arg, num_entries,
            &pchase_start_addr, &pchase_end_addr);
        if (ret < 0)
            return ret;

        h_base_address[i] = (uint64_t *)pchase_start_addr;

        start_addr = pchase_end_addr;
    }

    gpuErrAssert(cudaMemcpy(d_base_address, h_base_address, sizeof(uint64_t *) * max_blocks, cudaMemcpyHostToDevice));

    for (int i = 1; i <= max_blocks; i++) {

        gpuErrAssert(cudaMemset(d_inteferening_blocks, 0, sizeof(int)));
        gpuErrAssert(cudaMemset(d_sm0_blocks, 0, sizeof(int)));

        /* Launch some extra blocks, just in case. We deal with extra blocks inside kernel */
        do_interference_exp<<<2 * max_blocks, 1>>>((volatile uint64_t **)d_base_address, i - 1, 
                d_sm0_blocks, d_inteferening_blocks, loop_count, d_ticks, d_sum);
        gpuErrAssert(cudaDeviceSynchronize());
        gpuErrAssert(cudaMemcpy(&ticks, d_ticks, sizeof(double), cudaMemcpyDeviceToHost));
        time.push_back(ticks);
    }

    gpuErrAssert(cudaFree(d_base_address));
    gpuErrAssert(cudaFree(d_inteferening_blocks));
    gpuErrAssert(cudaFree(d_sm0_blocks));

    return 0;
}
