/* Contains helper functions for testing/benchmarking */
#ifndef __FRACTIONAL_GPU_TESTING_HPP__
#define __FRACTIONAL_GPU_TESTING_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#define DEFAULT_NUM_ITERATION   INT_MAX

#if defined(NDEBUG)
#define dprintf(...)
#else
#define dprintf(...)    printf(__VA_ARGS__)
#endif

#define USECPSEC 1000000ULL
inline double dtime_usec(unsigned long long start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return (double)(((tv.tv_sec*USECPSEC)+tv.tv_usec)-start);
}

/* For benchmarking applications */
typedef struct pstats {
    double sum;
    size_t count;
    std::vector<double> vals;
} pstats_t;

inline void pstats_init(pstats_t *stats)
{
    stats->count = stats->sum = 0;
    stats->vals.clear();
}

inline void pstats_add_observation(pstats_t *stats, double time)
{
    stats->count++;
    stats->sum += time;
    stats->vals.push_back(time);
}

inline void pstats_print(pstats_t *stats)
{
    double median;
    size_t count = stats->count;
    size_t count_95p = ((double)(count) * 95.0) / 100.0;
    size_t count_5p = ((double)(count) * 5.0) / 100.0;

    if (count == 0) {
        printf("STATS:No values\n");
        return;
    }

    std::sort(stats->vals.begin(), stats->vals.begin() + stats->vals.size());

    /* Count odd or even? */
    if (count & 0x1) {
        median = stats->vals[count / 2];
    } else {
        median = (stats->vals[count / 2 - 1] + stats->vals[count / 2]) / 2;
    }

    printf("STATS: Avg:%f, Median:%f, Min:%f, Max:%f, 5 percentile:%f, 95 percentile:%f, Count:%zu\n",
            stats->sum / stats->count, median,
            stats->vals[0], stats->vals[count - 1], stats->vals[count_5p], stats->vals[count_95p],
            stats->count);
}

/* 
 * Do we just execute kernel in a loop? Helpful during benchmarking and creating
 * background interference tasks.
 */
static bool execute_just_kernel = false;

/* To be called after initialization */
bool test_execute_just_kernel(void)
{
    return execute_just_kernel;
}

#if defined(USE_FGPU)

#include <fractional_gpu.hpp>

void print_usage(char **argv)
{
    fprintf(stderr, "Usage: %s -c <color> -m <memory size> -i <number of iterations> [OPTIONS]\n"
            "-k Execute only kernel (Default: Memcpy and kernels both executed)\n",
            argv[0]);
    fprintf(stderr, "Exiting\n");
    exit(-1);
}

/* Parse arguments to find color, memory size and number of iterations */
static inline void test_initialize(int argc, char **argv, int *out_num_iterations,
        bool do_init = true)
{
    int opt, ret;
    int color, num_iterations;
    size_t mem_size;

    /* Set default values */
    color = fgpu_get_env_color();
    mem_size = fgpu_get_env_color_mem_size();
    num_iterations = DEFAULT_NUM_ITERATION;

    while ((opt = getopt(argc, argv, "c:i:m:k")) != -1) {
        
        switch (opt) {
        
        case 'c':
            color = atoi(optarg);
            break;

        case 'm':
            mem_size = atoll(optarg);
            break;

        case 'i':
            num_iterations = atoi(optarg);
            break;

        case 'k':
            execute_just_kernel = true;
            break;

        default: /* '?' */
            fprintf(stderr, "Invalid arguments found\n");
            print_usage(argv);
            fprintf(stderr, "Exiting\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("Configuration:\n");

    printf("Computational Coloring:\t");
#if defined(FGPU_COMP_COLORING_ENABLE)
    printf("Enabled\n");
#else
    printf("Disabled\n");
#endif

    printf("Memory Coloring:\t");
#if defined(FGPU_MEM_COLORING_ENABLED)
    printf("Enabled\n");
#else
    printf("Disabled\n");
#endif


    printf("User memory coloring:\t");
#if defined(FGPU_USER_MEM_COLORING_ENABLED)
    printf("Enabled\n");
#else
    printf("Disabled\n");
#endif

    printf("Test memory coloring:\t");
#if defined(FGPU_TEST_MEM_COLORING_ENABLED)
    printf("Enabled\n");
#else
    printf("Disabled\n");
#endif

    if (do_init) {
        ret = fgpu_init();
        if (ret < 0) {
            fprintf(stderr, "Exiting as can't initialize fgpu\n");
            exit(EXIT_FAILURE);
        }

        ret = fgpu_set_color_prop(color, mem_size);
        if (ret < 0) {
            fprintf(stderr, "Exiting as unable to set color property\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("Color:\t%d\n", color);
    printf("Memory:\t%zd\n", mem_size);
    printf("Iterations:\t%d\n", num_iterations);
    printf("Color Initialized Delayed:%s\n", do_init ? "FALSE" : "TRUE");

    *out_num_iterations = num_iterations;
}

static inline void test_deinitialize()
{
    fgpu_deinit();
}

#else /* USE_FGPU */

void print_usage(char **argv)
{
    fprintf(stderr, "Usage: %s -i <number of iterations> [OPTIONS]\n"
            "-k Execute only kernel (Default: Memcpy and kernels both executed)\n",
            argv[0]);
    fprintf(stderr, "Exiting\n");
    exit(-1);
}

/* Parse arguments to find color, memory size and number of iterations */
static inline void test_initialize(int argc, char **argv, int *out_num_iterations,
        bool do_init = false)
{
    int opt;
    int num_iterations;
    
    num_iterations = DEFAULT_NUM_ITERATION;

    while ((opt = getopt(argc, argv, "i:k")) != -1) {
        
        switch (opt) {
        
        case 'i':
            num_iterations = atoi(optarg);
            break;

        case 'k':
            execute_just_kernel = true;

        default: /* '?' */
            fprintf(stderr, "Invalid arguments found\n");
            print_usage(argv);
            fprintf(stderr, "Exiting\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("Configuration:\n");
    printf("Overall Coloring: Disabled\n");
    printf("Iterations:\t%d\n", num_iterations);

    *out_num_iterations = num_iterations;
}

static inline void test_deinitialize()
{
}

#endif /* USE_FGPU */

#endif /* __FRACTIONAL_GPU_TESTING_HPP__ */
