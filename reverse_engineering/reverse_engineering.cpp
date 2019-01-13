#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <sys/sysinfo.h>
#include <sched.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <math.h>
#include <stdarg.h>

#include <algorithm>
#include <vector>

#include <reverse_engineering.hpp>

#include <hash_function.hpp>

/* TODO:
 * 1) On GTX 1070, DRAM Bank reverse engineering function get stuck sometime.
 * Possible bug.
 * 2) On V100, cache hierarchy is not clear. Number of words is shown as 3.
 * whereas we expect it to be 48. Where is the factor of 16?
 */
typedef struct cb_arg {
    uintptr_t phy_start;
    uintptr_t virt_start;
    double time;
    double min;
    double max;
    double nearest_nonoutlier;
    double threshold;
    double running_threshold;
} cb_arg_t;

typedef struct pchase_cb_arg {
    uintptr_t phy_start;
    uintptr_t virt_start;
    std::vector<hash_context_t *> ctx;
    std::vector<int> partition;
} pchase_cb_arg_t;

/* Global configuration parameters */

/* Parameters for DRAM Bank access time histogram */
static bool g_dram_histogram_enabled = false;
static bool g_dram_trendline_enabled = false;
static bool g_interference_enabled = false;
static int g_dram_sample_size = 1000;
static int g_dram_histogram_spacing = 10;
FILE *g_dram_histogram_fp;
FILE *g_dram_trendline_fp;
FILE *g_interference_fp;
char *g_dram_histogram_file;
char *g_dram_trendline_file;
char *g_interference_file;

/* Prints and hightlights string */
static void print_highlighted(const char *fmt, ...)
{
    va_list args;
    char buf[1000];
    size_t len;

    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    len = strlen(buf);
    for (size_t i = 0; i < len; i++)
        printf("*");
    printf("\n");

    printf("%s\n", buf);

    for (size_t i = 0; i < len; i++)
        printf("*");
    printf("\n");
}
/* Usage for current program */
void print_usage(char **argv)
{
    fprintf(stderr, "Usage: %s [OPTIONS]\n"
            "-H Filename for outputting DRAM access time histogram\n"
            "-I Filename for outputting DRAM Banks/Cachelines inteference results\n"
            "-T Filename for outputting DRAM access times\n"
            "-n Number of samples of DRAM. Default :%d\n"
            "-s Spacing for dram histogram. Default: %d\n", argv[0],
            g_dram_sample_size,
            g_dram_histogram_spacing);
}

void parse_args(int argc, char **argv) 
{
    int opt;

    while ((opt = getopt(argc, argv, "H:I:T:n:s:h")) != -1) {
        
        switch (opt) {
        
        case 'h':
            print_usage(argv);
            fprintf(stderr, "Exiting\n");
            exit(EXIT_SUCCESS);

        case 'n':
            g_dram_sample_size = atoi(optarg);
            break;

        case 's':
            g_dram_histogram_spacing = atoi(optarg);
            break;

        case 'H':
            g_dram_histogram_enabled = true;
            g_dram_histogram_file = optarg;
            g_dram_histogram_fp = fopen(optarg, "w");
            if (!g_dram_histogram_fp)
            {
                fprintf(stderr, "Error opening file %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;

        case 'I':
            g_interference_enabled = true;
            g_interference_file = optarg;
            g_interference_fp = fopen(optarg, "w");
            if (!g_interference_fp)
            {
                fprintf(stderr, "Error opening file %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;

        case 'T':
            g_dram_trendline_enabled = true;
            g_dram_trendline_file = optarg;
            g_dram_trendline_fp = fopen(optarg, "w");
            if (!g_dram_trendline_fp)
            {
                fprintf(stderr, "Error opening file %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;

        default: /* '?' */
            fprintf(stderr, "Invalid arguments found\n");
            print_usage(argv);
            fprintf(stderr, "Exiting\n");
            exit(EXIT_FAILURE);
        }
    }   
}

bool is_power_of_2(size_t x)
{
    return !(x & (x - 1));
}

bool check_dram_partition_pair(void *phy_addr1, void *phy_addr2, void *arg)
{
    cb_arg_t *data = (cb_arg_t *)arg;
    uintptr_t a, b;

    a = (uintptr_t)phy_addr1 - data->phy_start + data->virt_start;
    b = (uintptr_t)phy_addr2 - data->phy_start + data->virt_start;

    dprintf("Reading Time: PhyAddr1: %p,\t PhyAddr2:0x%p\n",
                phy_addr1, phy_addr2);
    dprintf("VirtAddr1: 0x%lx,\t VirtAddr2: 0x%lx\n", a, b);

    data->time = device_find_dram_read_time((void *)a, (void *)b, data->threshold);
    dprintf("Time:%f, Threshold:%f\n", data->time, data->running_threshold);
    
    data->min = data->time < data->min ? data->time : data->min;
    data->max = data->time > data->max ? data->time : data->max;

    if (data->time < data->running_threshold) {
        data->nearest_nonoutlier = data->time > data->nearest_nonoutlier ?
                        data->time : data->nearest_nonoutlier;
    }

    if (data->time >= data->running_threshold) {
        dprintf("Found valid pair: (%p, %p)\n", phy_addr1, phy_addr2);
        return true;
    }

    return false;
}
void *find_next_dram_partition_pair(void *phy_addr1, void *phy_start_addr, 
        void *phy_end_addr, size_t offset, void *arg)
{
    uintptr_t a, b;
    uintptr_t ustart_addr = (uintptr_t)phy_start_addr;
    uintptr_t uend_addr = (uintptr_t)phy_end_addr;

    cb_arg_t *data = (cb_arg_t *)arg;

    for (; ustart_addr <= uend_addr; ustart_addr += offset) {

        if (check_dram_partition_pair(phy_addr1, (void *)ustart_addr, arg))
            return (void *)ustart_addr;
    }

    return NULL;
}

static double get_histogram_bin(double time, double min_time, int spacing)
{
    int min_rounded = ((int)(min_time) / spacing) * spacing;
    int time_rounded = ((int)(time) / spacing) * spacing;

    return (time_rounded - min_rounded) / spacing;
}

static double get_histogram_bin_start_time(int bin, double min_time, int spacing)
{
    int min_rounded = ((int)(min_time) / spacing) * spacing;

    return (double)(min_rounded + spacing * bin);
}


/*
 * Finds the hash function for DRAM Banks
 * virt start and phy start are virtual/physical start address of a contiguous
 * memory range.
 */
static hash_context_t *run_dram_exp(void *virt_start, void *phy_start, 
        size_t allocated, int min_bit, int max_bit)
{
    void *phy_end;
    size_t min_row_size;
    void *row_start, *row_end;
    uintptr_t a, b, b_phy;
    double threshold = LONG_MAX;
    double time, sum, running_threshold, nearest_nonoutlier;
    double min, max;
    hash_context_t *hctx;
    cb_arg_t data;
    int ret;
    size_t offset = (1ULL << min_bit);
    size_t max_entries = allocated / offset;
    int count = std::min(max_entries, (size_t)THRESHOLD_SAMPLE_SIZE);
    std::vector<std::pair<void *, double>> times;
    std::vector<int> dram_histogram;

    if (g_dram_histogram_enabled || g_dram_trendline_enabled) {
        
        count = std::min(max_entries, (size_t)g_dram_sample_size);
        if (count != g_dram_sample_size) {
            fprintf(stderr, "WARNING: DRAM histogram sample size will be %d, instead of %d\n",
                    count, g_dram_sample_size);
        }
    }

    // Find running threshold
    min = LONG_MAX;
    max = 0;
    sum = 0;
    printf("Finding threshold\n", count);
    for (int i = 0; i < count; i++) {
        a = (uintptr_t)virt_start;
        b = a + offset * i;
        b_phy = (uintptr_t)phy_start +  offset * i;

        time = device_find_dram_read_time((void *)a, (void *)b, threshold);
        sum += time;
        min = time < min ? time : min;
        max = time > max ? time : max;
        times.push_back(std::pair<void *, double>((void *)b_phy, time));

        /* Print progress */
        printf("Done:%.1f%%\r", (float)(i * 100)/(float)(count));
    }
    printf("\n");
    
    double avg = sum / count;
    threshold = avg * THRESHOLD_MULTIPLIER;
    running_threshold = (avg * (100.0 + OUTLIER_DRAM_PERCENTAGE)) / 100.0;

    if (g_dram_histogram_enabled || g_dram_trendline_fp) {

        printf("Access Time: Threshold is: %f cycles, (Max: %f cycles, Min:%f cycles)\n",
            running_threshold, max, min);

        /* Create histogram bins - Add few extra bins at end just for verification by eyeballing */
        int max_bins = get_histogram_bin(max, min, g_dram_histogram_spacing) + 2;
        for (int i = 0; i < max_bins; i++) {
            dram_histogram.push_back(0);
        }


        if (g_dram_trendline_enabled) {
            print_highlighted("Outputting DRAM access time trendline's raw data to %s", g_dram_trendline_file);
            fprintf(g_dram_trendline_fp, "PrimaryAddress\t SecondaryAddress\t AccessTime(cycles)\t Threshold(cycles)\n");
        }

        for (int i = 0; i < times.size(); i++) {
            int bin;

            if (g_dram_trendline_enabled) {

                fprintf(g_dram_trendline_fp, "%p\t %p\t %f\t %f\n", phy_start,
                            times[i].first, times[i].second, running_threshold);
            }

            bin = get_histogram_bin(times[i].second, min, g_dram_histogram_spacing);
            dram_histogram[bin]++;
        }

        fflush(g_dram_trendline_fp);

        if (g_dram_histogram_enabled) {

            print_highlighted("Outputting DRAM access time histogram's raw data to %s", g_dram_histogram_file);
            fprintf(g_dram_histogram_fp, "StartAccessTime-EndAccessTime\t Count\n");
            for (int i = 0; i < max_bins; i++) {
                double bin_start_time = 
                    get_histogram_bin_start_time(i, min, g_dram_histogram_spacing);
                fprintf(g_dram_histogram_fp, "%d-%d\t %d\n",
                        (int)(bin_start_time), (int)(bin_start_time) + g_dram_histogram_spacing,
                        dram_histogram[i]);
            }
        }

	    fflush(g_dram_histogram_fp);
    } else {
        dprintf("Threshold is %f, Running threshold is: %f, (Max: %f, Min:%f)\n",
            threshold, running_threshold, max, min);
    }

    phy_end = (void *)((uintptr_t)phy_start + allocated - device_allocation_overhead());

    data.min = LONG_MAX;
    data.max = 0;
    data.nearest_nonoutlier = 0;
    data.threshold = threshold;
    data.running_threshold = running_threshold;
    data.phy_start = (uintptr_t)phy_start;
    data.virt_start = (uintptr_t)virt_start;

    /* 
     * The minimum bit supplied can be too small. It might make the whole
     * brute force search quite slow. So instead we try to update the min bit
     * by finding (approximately) the row size.
     * To do this, two steps are required:
     * 1) Find start addr of another row in same bank.
     * 2) Find the end address of the row.
     * The difference in these address tell the (minimum) row size.
     */
    printf("Finding DRAM row size (Might take a while)\n");
    
    row_start = NULL;
    /* Check already measured first */
    for (int i = 0; i < times.size(); i++) {
        if (times[i].second >= data.running_threshold) {
            row_start = times[i].first;
            break;
        }
    }

    if (!row_start) {
    
        row_start = times[times.size() - 1].first;
        row_start = find_next_dram_partition_pair(phy_start, row_start, phy_end,
                offset, &data);
        if (!row_start) {
            fprintf(stderr, "Couldn't find another address in same partition\n");
            return NULL;
        }
    }

    for (row_end = row_start; (uintptr_t)(row_end) <= (uintptr_t)phy_end; 
            row_end = (void *)((uintptr_t)row_end + offset)) {
        if (!check_dram_partition_pair(phy_start, row_end, &data))
            break;
    }

    if ((uintptr_t)row_end > (uintptr_t)phy_end) {
        fprintf(stderr, "Couldn't find end of DRAM row\n");
        return NULL;
    }

    min_row_size = (uintptr_t)row_end - (uintptr_t)row_start;

    /* Row size should be a power of 2 */
    if (min_row_size == 0 || !is_power_of_2(min_row_size)) {
        fprintf(stderr, "Min row size (%zu) seems to be incorrect\n", min_row_size);
        return NULL;
    }

    min_bit = ilog2((unsigned long long)min_row_size);
    offset = (1ULL << min_bit);
    
    printf("Physical address bits from which DRAM Banks are possibly derived: Max Bit:%d, Min Bit:%d\n", max_bit, min_bit);

    hctx = hash_init(min_bit, max_bit, phy_start, phy_end);
    assert(hctx);

    printf("Finding solutions\n");
    ret = hash_find_solutions2(hctx, &data, check_dram_partition_pair);

    dprintf("Min: %f, Max: %f, Percentage diff: %f\n",
                data.min, data.max, (data.max - data.min) / (data.min));
    dprintf("Nearest Nonoutlier: %f, Threshold: %f\n",
                data.nearest_nonoutlier, data.running_threshold);

    if (ret < 0) {
        fprintf(stderr, "No solutions found\n");
    } else {
        
        /* Sort before printing */
        hash_sort_solutions(hctx);

        print_highlighted("Hash Function for DRAM Banks:");
        hash_print_solutions(hctx);
    }

    return hctx;
}

void *find_next_cache_partition_pair(void *phy_addr1, void *phy_start_addr, 
        void *phy_end_addr, size_t offset, void *arg)
{
    void *phy_addr2;
    uintptr_t a, b;
    uintptr_t ret_addr;
    cb_arg_t *data = (cb_arg_t *)arg;

    a = (uintptr_t)phy_addr1 - data->phy_start + data->virt_start;
    b = (uintptr_t)phy_start_addr - data->phy_start + data->virt_start;

    dprintf("Trying to find pair for %p in the range [%p, %p)\n", phy_addr1, phy_start_addr, phy_end_addr);
    ret_addr = (uintptr_t)device_find_cache_eviction_addr((void * )a, (void *)b, offset, data->running_threshold);
    if (!ret_addr)
        return NULL;

    phy_addr2 = (void *)((uintptr_t)ret_addr - data->virt_start + data->phy_start);
    dprintf("Found valid pair: (%p, %p)\n", phy_addr1, phy_addr2);

    if ((uintptr_t)phy_addr2 > (uintptr_t)phy_end_addr)
        return NULL;

    return phy_addr2;
}

/* 
 * Returns the next word in same partitions start_virt_addr 
 * according to the 'include' and 'exclude' partitions 
 */
void *get_next_word(void *start_virt_addr, void *arg)
{
    pchase_cb_arg_t *data = (pchase_cb_arg_t *)arg;
    uintptr_t start_phy_addr = (uintptr_t)start_virt_addr - data->virt_start + 
        data->phy_start;
    uintptr_t next_phy_addr;
    bool found = false;

    next_phy_addr = (uintptr_t)hash_get_next_addr(data->ctx, data->partition, 
        (void *)start_phy_addr, (void *)-1);
    if (!next_phy_addr)
        return NULL;

    return (void *)(next_phy_addr - data->phy_start + data->virt_start);
}

/*
 * Finds the hash function for Cachelines
 * virt start and phy start are virtual/physical start address of a contiguous
 * memory range.
 */
static hash_context_t *run_cache_exp(void *virt_start, void *phy_start, size_t allocated, int min_bit, int max_bit)
{
    void *phy_end;
    hash_context_t *hctx;
    cb_arg_t data;
    int ret;
    size_t offset = (1ULL << min_bit);
    double avg, running_threshold;
    size_t num_words;
    pchase_cb_arg_t pchase_cb_arg;

    printf("Doing initialization\n");
    ret = device_cacheline_test_init(virt_start, allocated);
    if (ret < 0) {
        fprintf(stderr, "Couldn't initialize\n");
        return NULL;
    }

    // Find running threshold
    printf("Finding threshold\n");
    
    ret = device_cacheline_test_find_threshold(THRESHOLD_SAMPLE_SIZE, &avg);
    if (ret < 0) {
        fprintf(stderr, "Couldn't find the threshold\n");
        return NULL;
    }
    running_threshold = (avg * (100.0 + OUTLIER_CACHE_PERCENTAGE)) / 100.0;

    dprintf("Running threshold is: %f\n", running_threshold);

    phy_end = (void *)((uintptr_t)phy_start + allocated - device_allocation_overhead());

    data.running_threshold = running_threshold;
    data.phy_start = (uintptr_t)phy_start;
    data.virt_start = (uintptr_t)virt_start;

    hctx = hash_init(min_bit, max_bit, phy_start, phy_end);
    assert(hctx);

    printf("Finding solutions\n");
    ret = hash_find_solutions(hctx, &data, find_next_cache_partition_pair);
    if (ret < 0) {
        fprintf(stderr, "No solutions found\n");
    } else {
        /* Sort before printing */
        hash_sort_solutions(hctx);

        print_highlighted("Hash Function for Cacheline:");
        hash_print_solutions(hctx);
    }

    /* Find the number of words in a cacheline */
    pchase_cb_arg.virt_start = (uintptr_t)virt_start;
    pchase_cb_arg.phy_start = (uintptr_t)phy_start;
    pchase_cb_arg.ctx.push_back(hctx);
    pchase_cb_arg.partition.push_back(0);

    /* Re-init test */
    ret = device_find_cacheline_words_count(virt_start, data.running_threshold,
            get_next_word, &pchase_cb_arg, &num_words);
    if (ret < 0) {
        fprintf(stderr, "Couldn't find number of words in a cachline\n");
    }
    print_highlighted("Number of words in a cachline:%zd", num_words);
    
    return hctx;
}

static int run_interference_exp_helper(void *virt_start,
        pchase_cb_arg_t *primary_pchase_cb_arg,
        pchase_cb_arg_t *secondary_pchase_cb_arg,
        std::vector<double> &time)
{
    int ret;

    ret = device_run_interference_exp(virt_start, get_next_word,
            primary_pchase_cb_arg, secondary_pchase_cb_arg, 50, 1000, 
            time);
    if (ret < 0) {
        fprintf(stderr, "Couldn't run interference exp\n");
        return ret;
    }

    return 0;
}

static int run_interference_exp(void *virt_start, void *phy_start, 
        hash_context_t *cache_hctx, hash_context_t *dram_hctx, 
        hash_context_t *common_hctx)
{
    pchase_cb_arg_t primary_pchase_cb_arg;
    pchase_cb_arg_t secondary_pchase_cb_arg;
    int ret;

    primary_pchase_cb_arg.virt_start = (uintptr_t)virt_start;
    primary_pchase_cb_arg.phy_start = (uintptr_t)phy_start;
    secondary_pchase_cb_arg.virt_start = (uintptr_t)virt_start;
    secondary_pchase_cb_arg.phy_start = (uintptr_t)phy_start;
    
    std::vector<double> times[5];

    primary_pchase_cb_arg.ctx.clear();
    primary_pchase_cb_arg.partition.clear();
    primary_pchase_cb_arg.ctx.push_back(cache_hctx);
    primary_pchase_cb_arg.partition.push_back(0);
    primary_pchase_cb_arg.ctx.push_back(dram_hctx);
    primary_pchase_cb_arg.partition.push_back(0);
    primary_pchase_cb_arg.ctx.push_back(common_hctx);
    primary_pchase_cb_arg.partition.push_back(0);

    secondary_pchase_cb_arg.ctx.clear();
    secondary_pchase_cb_arg.partition.clear();
    secondary_pchase_cb_arg.ctx.push_back(cache_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(dram_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(common_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);

    printf("Running Interference Experiment: Same Cacheline, Same DRAM Bank, Same Partition\n");
    ret = run_interference_exp_helper(virt_start, &primary_pchase_cb_arg, 
            &secondary_pchase_cb_arg, times[0]);
    if (ret < 0)
        return ret;
    
    secondary_pchase_cb_arg.ctx.clear();
    secondary_pchase_cb_arg.partition.clear();
    secondary_pchase_cb_arg.ctx.push_back(cache_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(dram_hctx);
    secondary_pchase_cb_arg.partition.push_back(1);
    secondary_pchase_cb_arg.ctx.push_back(common_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);

    printf("Running Interference Experiment: Same Cacheline, Different DRAM Bank, Same Partition\n");
    ret = run_interference_exp_helper(virt_start, &primary_pchase_cb_arg, 
            &secondary_pchase_cb_arg, times[1]);
    if (ret < 0)
        return ret;

    secondary_pchase_cb_arg.ctx.clear();
    secondary_pchase_cb_arg.partition.clear();
    secondary_pchase_cb_arg.ctx.push_back(cache_hctx);
    secondary_pchase_cb_arg.partition.push_back(1);
    secondary_pchase_cb_arg.ctx.push_back(dram_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(common_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);

    printf("Running Interference Experiment: Different Cacheline, Same DRAM Bank, Same Partition\n");
    ret = run_interference_exp_helper(virt_start, &primary_pchase_cb_arg, 
            &secondary_pchase_cb_arg, times[2]);
    if (ret < 0)
        return ret;

    secondary_pchase_cb_arg.ctx.clear();
    secondary_pchase_cb_arg.partition.clear();
    secondary_pchase_cb_arg.ctx.push_back(cache_hctx);
    secondary_pchase_cb_arg.partition.push_back(1);
    secondary_pchase_cb_arg.ctx.push_back(dram_hctx);
    secondary_pchase_cb_arg.partition.push_back(1);
    secondary_pchase_cb_arg.ctx.push_back(common_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);

    printf("Running Interference Experiment: Different Cacheline, Different DRAM Bank, Same Partition\n");
    ret = run_interference_exp_helper(virt_start, &primary_pchase_cb_arg, 
            &secondary_pchase_cb_arg, times[3]);
    if (ret < 0)
        return ret;

    secondary_pchase_cb_arg.ctx.clear();
    secondary_pchase_cb_arg.partition.clear();
    secondary_pchase_cb_arg.ctx.push_back(cache_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(dram_hctx);
    secondary_pchase_cb_arg.partition.push_back(0);
    secondary_pchase_cb_arg.ctx.push_back(common_hctx);
    secondary_pchase_cb_arg.partition.push_back(1);

    /* 
     * Even if same dram and cache, but in different partition, hence in different
     * dram and cache partitions.
     */
    printf("Running Interference Experiment: Different Cacheline, Different DRAM Bank, Different Partition\n");
    ret = run_interference_exp_helper(virt_start, &primary_pchase_cb_arg, 
            &secondary_pchase_cb_arg, times[4]);
    if (ret < 0)
        return ret;

    /* Number of blocks/threads should be same in each */
    size_t s = times[0].size();
    for (int i = 1; i < 5; i++) {
        assert(times[i].size() == s);
    }

    assert(g_interference_fp);

    print_highlighted("Outputting Interference experiment results' raw data to %s", g_interference_file);
    fprintf(g_interference_fp, "NumThreads\t SameCachelineSameDRAMBank(SCSB)\t " 
            "SameCachelineDifferentDRAMBank(CSDB)\t DifferentCachelineSameDRAMBank(DCSB)\t " 
            "DifferentCachelineDifferentBank(DCDB)\t DifferentModule(DM)\n");
    
    for (int i = 0; i < s; i++) {
        fprintf(g_interference_fp, "%d\t ", i + 1);

        for (int j = 0; j < 5; j++) {
            fprintf(g_interference_fp, "%f\t ", times[j][i]);
        }
        
        fprintf(g_interference_fp, "\n");
    }

    fflush(g_interference_fp);
    return 0;
}

int main(int argc, char *argv[])
{
    void *virt_start;
    void *phy_start;
    size_t req_allocated, allocated;
    int ret;
    int max_bit, min_bit;
    hash_context_t *dram_hctx, *cache_hctx, *common_hctx;

    parse_args(argc, argv);

    max_bit = device_max_physical_bit();
    if (max_bit < 0) {
        fprintf(stderr, "Couldn't find the maximum bit\n");
        return -1;
    }

    min_bit = device_min_physical_bit();
    if (min_bit < 0) {
        fprintf(stderr, "Couldn't find the minimum bit\n");
        return -1;
    }

    req_allocated = (1ULL << (max_bit + 1)) - 1;

    ret = device_init(req_allocated, &allocated);
    if (ret < 0) {
        fprintf(stderr, "Init failed\n");
        return -1;
    }

    virt_start = device_allocate_contigous(allocated, &phy_start);
    if (virt_start == NULL) {
        fprintf(stderr, "Couldn't find the physical contiguous addresses\n");
        return -1;
    }

    printf("Finding DRAM Banks hash function\n");
    dram_hctx = run_dram_exp(virt_start, phy_start, allocated, min_bit, max_bit);
    if (dram_hctx == NULL) {
        fprintf(stderr, "Couldn't find DRAM Banks hash function\n");
        return -1;
    }

    printf("Finding Cacheline hash function\n");
    cache_hctx = run_cache_exp(virt_start, phy_start, allocated, min_bit, max_bit);
    if (cache_hctx == NULL) {
        fprintf(stderr, "Couldn't find Cacheline hash function\n");
        return -1;
    }

    print_highlighted("Finding common solutions between DRAM and Cache");
    common_hctx = hash_get_common_solutions(dram_hctx, cache_hctx);

    print_highlighted("Unique DRAM bank solutions");
    hash_print_solutions(dram_hctx);

    print_highlighted("Unique Cache solutions");
    hash_print_solutions(cache_hctx);

    if (g_interference_enabled) {
        print_highlighted("Running Interference Tests");
        ret = run_interference_exp(virt_start, phy_start, cache_hctx, dram_hctx, 
                common_hctx);
        if (ret < 0) {
            fprintf(stderr, "Couldn't run interference tests\n");
            return -1;
        }
    }

    hash_del(dram_hctx);
    hash_del(cache_hctx);
    hash_del(common_hctx);

    if (g_dram_histogram_fp)
        fclose(g_dram_histogram_fp);

    if (g_dram_trendline_fp)
        fclose(g_dram_trendline_fp);

    if (g_interference_fp)
        fclose(g_interference_fp);

    return 0;
}
