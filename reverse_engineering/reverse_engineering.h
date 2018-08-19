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

/* The maximum and minimum physical address bit that forms part of hash function */
/* 
 * The max bit's upper limit is set by amount of gpu physical memory.
 * It can be lower. In such case, upper bits are not considered in hash function
 * that is reversed engineered.
 * The min bit can be found by trial and error.
 */
#define GPU_MAX_BIT                         (32)
#define GPU_MIN_BIT                         (10)

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
#define MAX_CONTIGUOUS_ALLOC                (1ULL << (GPU_MAX_BIT + 1))
#define MIN_BANK_SIZE                       (1ULL << GPU_MIN_BIT)

#define NUM_ENTRIES                         (MAX_CONTIGUOUS_ALLOC / MIN_BANK_SIZE)

#define MIN_BIT                             (GPU_MIN_BIT)
#define MAX_BIT                             (GPU_MAX_BIT)

#define OUTLIER_PERCENTAGE                  GPU_OUTLIER_PERCENTAGE
/* 
 * For CPU, any values above THRESHOLD_MULTIPLIER * avg is ignored (as might have
 * noise). This is because we are using wall clock time and interrupts might
 * cause huge jumps
 */
#define THRESHOLD_MULTIPLIER                5

/* Sample size to find out the threshold */
#define THRESHOLD_SAMPLE_SIZE               100

/***************************** FUNCTION DECLARATIONS **************************/
void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p);
double device_find_dram_read_time(void *_a, void *_b, double threshold);
int device_init(size_t req_reserved_size, size_t *reserved_size);
size_t device_allocation_overhead(void);

#endif /* __REVERSE_ENGINEERING_H__ */
