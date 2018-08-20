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
#include <algorithm>

#include <reverse_engineering.h>

#include <hash_function.h>

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

/*
 * Finds the hash function for DRAM Banks
 * virt start and phy start are virtual/physical start address of a contiguous
 * memory range.
 */
static int run_dram_exp(void *virt_start, void *phy_start, size_t allocated, int min_bit, int max_bit)
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
    std::vector<std::pair<void *, int>> times;

    // Find running threshold
    min = LONG_MAX;
    max = 0;
    sum = 0;
    printf("Finding threshold\n");
    for (int i = 0; i < count; i++) {
        a = (uintptr_t)virt_start;
        b = a + offset * i;
        b_phy = (uintptr_t)phy_start +  offset * i;

        time = device_find_dram_read_time((void *)a, (void *)b, threshold);
        sum += time;
        min = time < min ? time : min;
        max = time > max ? time : max;
        times.push_back(std::pair<void *, int>((void *)b_phy, time));

        /* Print progress */
        printf("Done:%.1f%%\r", (float)(i * 100)/(float)(count));
    }
    printf("\n");
    
    double avg = sum / count;
    threshold = avg * THRESHOLD_MULTIPLIER;
    running_threshold = (avg * (100.0 + OUTLIER_DRAM_PERCENTAGE)) / 100.0;

    dprintf("Threshold is %f, Running threshold is: %f, (Max: %f, Min:%f)\n",
            threshold, running_threshold, max, min);

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
    printf("Finding DRAM row size\n");
    
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
            return -1;
        }
    }

    for (row_end = row_start; (uintptr_t)(row_end) <= (uintptr_t)phy_end; 
            row_end = (void *)((uintptr_t)row_end + offset)) {
        if (!check_dram_partition_pair(phy_start, row_end, &data))
            break;
    }

    if ((uintptr_t)row_end > (uintptr_t)phy_end) {
        fprintf(stderr, "Couldn't find end of DRAM row\n");
        return -1;
    }

    min_row_size = (uintptr_t)row_end - (uintptr_t)row_start;

    /* Row size should be a power of 2 */
    if (min_row_size == 0 || !is_power_of_2(min_row_size)) {
        fprintf(stderr, "Min row size seems to be incorrect\n");
        return -1;
    }

    min_bit = ilog2((unsigned long long)min_row_size);
    offset = (1ULL << min_bit);
    
    printf("Max Bit:%d, Min Bit:%d\n", max_bit, min_bit);

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
        printf("Hash Function for DRAM Banks:\n");
        hash_print_solutions(hctx);
    }

    hash_del(hctx);

    return 0;
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

    ret_addr = (uintptr_t)device_find_cache_eviction_addr((void * )a, (void *)b, offset, data->running_threshold);
    if (!ret_addr)
        return NULL;

    phy_addr2 = (void *)((uintptr_t)ret_addr - data->virt_start + data->phy_start);
    dprintf("Found valid pair: (%p, %p)\n", phy_addr1, phy_addr2);

    return phy_addr2;
}

/*
 * Finds the hash function for Cachelines
 * virt start and phy start are virtual/physical start address of a contiguous
 * memory range.
 */
static int run_cache_exp(void *virt_start, void *phy_start, size_t allocated, int min_bit, int max_bit)
{
    void *phy_end;
    hash_context_t *hctx;
    cb_arg_t data;
    int ret;
    size_t offset = (1ULL << min_bit);
    double avg, running_threshold;

    printf("Doing initialization\n");
    ret = device_cacheline_test_init(virt_start, allocated);
    if (ret < 0) {
        fprintf(stderr, "Couldn't initialize\n");
        return -1;
    }

    // Find running threshold
    printf("Finding threshold\n");
    
    ret = device_cacheline_test_find_threshold(THRESHOLD_SAMPLE_SIZE, &avg);
    if (ret < 0) {
        fprintf(stderr, "Couldn't find the threshold\n");
        return -1;
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
        printf("Hash Function for Cacheline:\n");
        hash_print_solutions(hctx);
    }

    hash_del(hctx);

    return 0;
}

int main(int argc, char *argv[])
{
    void *virt_start;
    void *phy_start;
    size_t req_allocated, allocated;
    int ret;
    int max_bit, min_bit;

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
    ret = run_dram_exp(virt_start, phy_start, allocated, min_bit, max_bit);
    if (ret < 0) {
        fprintf(stderr, "Couldn't find DRAM Banks hash function\n");
        return -1;
    }

    printf("Finding Cacheline hash function\n");
    ret = run_cache_exp(virt_start, phy_start, allocated, min_bit, max_bit);
    if (ret < 0) {
        fprintf(stderr, "Couldn't find Cacheline hash function\n");
        return -1;
    }
    return 0;
}
