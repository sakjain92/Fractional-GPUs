#ifndef __REVERSE_ENGINEERING_HPP__
#define __REVERSE_ENGINEERING_HPP__

/* TODO: Add support for CPU also if needed */

#ifndef NDEBUG
#define dprintf(...)                        printf(__VA_ARGS__)
#else
#define dprintf(...)
#endif

/****************************** GPU DEFINES ***********************************/

/* All default GPU configurations are for GTX 1070. They might vary across GPUs */

/* The number of time a measurment is done to get better averages */
#define GPU_MAX_OUTER_LOOP                  10

/* 
 * The word size of L2 cache size.
 * Found via online nvidia forums.
 * Can be smaller than actual but will make code slower.
 */
#define GPU_L2_CACHE_WORD_SIZE              (32)

/* 
 * Found by experiment.
 * Can be smaller then actual value.
 * Can make it smaller but code will become slower 
 */
#define GPU_L2_CACHE_LINE_SIZE              (128)

/* 
 * By what percentage do the bank conflict cause delay as compared to avg read time 
 * Used as cutoff
 */
#define GPU_DRAM_OUTLIER_PERCENTAGE              10


/* 
 * By what percentage do the cacheline eviction cause delay as compared to avg read time 
 * Used as cutoff. Can be high as read from DRAM should be much slower than read from cache.
 */
#define GPU_CACHE_OUTLIER_PERCENTAGE              50

/****************************** COMMON DEFINES *******************************/
#define OUTLIER_DRAM_PERCENTAGE                  GPU_DRAM_OUTLIER_PERCENTAGE
#define OUTLIER_CACHE_PERCENTAGE                 GPU_CACHE_OUTLIER_PERCENTAGE

/* 
 * For CPU, any values above THRESHOLD_MULTIPLIER * avg is ignored (as might have
 * noise). This is because we are using wall clock time and interrupts might
 * cause huge jumps
 */
#define THRESHOLD_MULTIPLIER                5

/* Sample size to find out the threshold */
#define THRESHOLD_SAMPLE_SIZE               1000

/***************************** FUNCTION DECLARATIONS **************************/
size_t device_allocation_overhead(void);
int device_max_physical_bit(void);
int device_min_physical_bit(void);
void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p);
double device_find_dram_read_time(void *_a, void *_b, double threshold);
int device_init(size_t req_reserved_size, size_t *reserved_size);
int device_cacheline_test_init(void *gpu_start_addr, size_t size);
int device_cacheline_test_find_threshold(size_t sample_size, double *avg);
void *device_find_cache_eviction_addr(void *_a, void *_b, size_t offset, double threshold);

int device_find_cacheline_words_count(void *gpu_start_addr, double threshold,
        void *(*cb)(void *addr, void *arg), void *arg, size_t *words);

int device_run_interference_exp(void *gpu_start_addr, void *(*cb)(void *addr, void *arg),
        void *primary_arg, void *secondary_arg, int max_blocks, int loop_count, 
        std::vector<double> &time);

inline int ilog2(unsigned int x)
{
    return sizeof(unsigned int) * 8 - __builtin_clz(x) - 1;
}

inline int ilog2(unsigned long long x)
{
    return sizeof(unsigned long long ) * 8 - __builtin_clzll(x) - 1;
}

#endif /* __REVERSE_ENGINEERING_HPP__ */
