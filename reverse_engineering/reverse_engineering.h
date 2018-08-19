#ifndef __REVERSE_ENGINEERING_H__
#define __REVERSE_ENGINEERING_H__

/* TODO: Add support for CPU also if needed */

/* Uncomment to allow debug prints */
//#define ENABLE_DEBUG                           1

#if (ENABLE_DEBUG == 1)
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
#define GPU_OUTLIER_PERCENTAGE              10

/****************************** COMMON DEFINES *******************************/
#define OUTLIER_PERCENTAGE                  GPU_OUTLIER_PERCENTAGE
/* 
 * For CPU, any values above THRESHOLD_MULTIPLIER * avg is ignored (as might have
 * noise). This is because we are using wall clock time and interrupts might
 * cause huge jumps
 */
#define THRESHOLD_MULTIPLIER                5

/* Sample size to find out the threshold */
#define THRESHOLD_SAMPLE_SIZE               1000

/***************************** FUNCTION DECLARATIONS **************************/
void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p);
double device_find_dram_read_time(void *_a, void *_b, double threshold);
int device_init(size_t req_reserved_size, size_t *reserved_size);
size_t device_allocation_overhead(void);
int device_max_physical_bit(void);
int device_min_physical_bit(void);


inline int ilog2(unsigned int x)
{
    return sizeof(unsigned int) * 8 - __builtin_clz(x) - 1;
}

inline int ilog2(unsigned long long x)
{
    return sizeof(unsigned long long ) * 8 - __builtin_clzll(x) - 1;
}

#endif /* __REVERSE_ENGINEERING_H__ */
