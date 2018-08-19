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

bool check_dram_partition_pair(void *addr1, void *addr2, void *arg)
{
    cb_arg_t *data = (cb_arg_t *)arg;
    uintptr_t a, b;

    a = (uintptr_t)addr1 - data->phy_start + data->virt_start;
    b = (uintptr_t)addr2 - data->phy_start + data->virt_start;

    dprintf("Reading Time: PhyAddr1: %p,\t PhyAddr2:0x%p\n",
                addr1, addr2);
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
        dprintf("Found valid pair\n");
        return true;
    }

    return false;
}
void *find_next_dram_partition_pair(void *addr1, void *start_addr, 
        void *end_addr, size_t offset, void *arg)
{
    uintptr_t a, b;
    uintptr_t ustart_addr = (uintptr_t)start_addr;
    uintptr_t uend_addr = (uintptr_t)end_addr;

    cb_arg_t *data = (cb_arg_t *)arg;

    for (; ustart_addr <= uend_addr; ustart_addr += offset) {

        if (check_dram_partition_pair(addr1, (void *)ustart_addr, arg))
            return (void *)ustart_addr;
    }

    return NULL;
}

/*
 * Finds the hash function for DRAM Banks
 * virt start and phy start are virtual/physical start address of a contiguous
 * memory range.
 */
static void run_dram_exp(void *virt_start, void *phy_start, size_t allocated)
{
    void *phy_end;
    uintptr_t a, b;
    double threshold = LONG_MAX;
    double time, sum, running_threshold, nearest_nonoutlier;
    double min, max;
    hash_context_t *hctx;
    cb_arg_t data;
    int ret;

    int count = std::min((size_t)NUM_ENTRIES, (size_t)THRESHOLD_SAMPLE_SIZE);

    printf("Finding DRAM Banks hash function\n");

    // Find running threshold
    min = LONG_MAX;
    max = 0;
    sum = 0;
    printf("Finding threshold\n");
    for (int i = 0; i < count; i++) {
        a = (uintptr_t)virt_start;
        b = a + MIN_BANK_SIZE * i;
        time = device_find_dram_read_time((void *)a, (void *)b, threshold);
        sum += time;
        min = time < min ? time : min;
        max = time > max ? time : max;

        /* Print progress */
        printf("Done:%.1f%%\r", (float)(i * 100)/(float)(count));
    }
    printf("\n");
    
    double avg = sum / count;
    threshold = avg * THRESHOLD_MULTIPLIER;
    running_threshold = (avg * (100.0 + OUTLIER_PERCENTAGE)) / 100.0;

    dprintf("Threshold is %f, Running threshold is: %f, (Max: %f, Min:%f)\n",
            threshold, running_threshold, max, min);

    phy_end = (void *)((uintptr_t)phy_start + allocated - device_allocation_overhead());

    hctx = hash_init(MIN_BIT, MAX_BIT, phy_start, phy_end);
    assert(hctx);

    data.min = LONG_MAX;
    data.max = 0;
    data.nearest_nonoutlier = 0;
    data.threshold = threshold;
    data.running_threshold = running_threshold;
    data.phy_start = (uintptr_t)phy_start;
    data.virt_start = (uintptr_t)virt_start;

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
}

int main(int argc, char *argv[])
{
    void *virt_start;
    void *phy_start;
    size_t allocated;
    int ret;

    ret = device_init(MAX_CONTIGUOUS_ALLOC, &allocated);
    if (ret < 0) {
        fprintf(stderr, "Init failed\n");
        return -1;
    }

    virt_start = device_allocate_contigous(allocated, &phy_start);
    if (virt_start == NULL) {
        fprintf(stderr, "Couldn't find the physical contiguous addresses\n");
        return -1;
    }

    run_dram_exp(virt_start, phy_start, allocated);

    return 0;
}
