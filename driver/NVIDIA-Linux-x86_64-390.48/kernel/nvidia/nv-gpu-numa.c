/*******************************************************************************
    Copyright (c) 2015-2017 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "nv-linux.h"
#include "nv-gpu-numa.h"
#include "nvmisc.h"

#ifndef NV_ITERATE_DIR_PRESENT
    struct dir_context 
    {
        const filldir_t actor;
        loff_t pos;
    };
    static int iterate_dir(struct file *filp, struct dir_context *ctx)
    {
        WARN_ON(1);
        return 0;
    }
#endif

#ifndef NV_STRNSTR_PRESENT
    static char *strnstr(const char *s1, const char *s2, size_t len)
    {
        return NULL;
    }
#endif

#ifndef NV_KERNEL_WRITE_PRESENT
    static size_t kernel_write(struct file *filp, const char *str, size_t len,
                                loff_t offset)
    {
        return 0;
    }
#endif

#ifndef NV_KSTRTOULL_PRESENT
    static int kstrtoull(const char *s, unsigned int base,
                         unsigned long long *res)
    {
        return 0;
    }
#endif

// -----------------------------------------------------------------------------
// TODO: fix bug 1735381, and don't do any of this stuff in kernel space:
#define NID_ARG_FMT            "hotpluggable_nodes="

#define NID_PATH               "/sys/devices/system/node/node"
#define MEMBLOCK_PREFIX        "memory"
#define BRING_ONLINE_CMD       "online_movable"
#define BRING_OFFLINE_CMD      "offline"
#define STATE_ONLINE           "online"
#define MEMBLK_STATE_PATH_FMT  "/sys/devices/system/memory/memory%d/state"
#define MEMBLK_SIZE_PATH       "/sys/devices/system/memory/block_size_bytes"
#define MEMORY_PROBE_PATH      "/sys/devices/system/memory/probe"
#define READ_BUFFER_SIZE           100
#define BUF_SIZE                   100
#define BUF_FOR_64BIT_INTEGER_SIZE 20
// end of TODO -----------------------------------------------------------------

extern nv_linux_state_t nv_ctl_device;

typedef enum 
{
    MEM_ONLINE,
    MEM_OFFLINE
} mem_state_t;

typedef struct struct_nv_dir_context
{
    struct dir_context ctx;
    int numa_node_id;
    NvU64 memblock_start_id;
    NvU64 memblock_end_id;
} nv_dir_context_t;

static inline char* mem_state_to_string(mem_state_t state)
{
    switch (state) 
    {
        case MEM_ONLINE:
            return "online";
        case MEM_OFFLINE:
            return "offline";
        default:
            return "invalid_state";
    }
}

// TODO: Bug 1735381: don't open files from within kernel code
static NV_STATUS bad_idea_write_string_to_file(const char *path_to_file,
                                               const char *write_buffer,
                                               size_t write_buffer_size)
{
    struct file *filp;
    int write_count;

    filp = filp_open(path_to_file, O_WRONLY, 0);
    if (IS_ERR(filp)) 
    {
        nv_printf(NV_DBG_ERRORS, "filp_open failed\n");
        return NV_ERR_NO_VALID_PATH;
    }

    write_count = kernel_write(filp, write_buffer, write_buffer_size, 0);

    filp_close(filp, NULL);

    if ((write_count > 0) && (write_count < write_buffer_size))
        return NV_ERR_INVALID_STATE;
    else if (write_count == -EEXIST)
        return NV_ERR_IN_USE; 
    else if (write_count < 0)
        return NV_ERR_GENERIC;

    // write_count == write_buffer_size:
    return NV_OK;
}

// This is a callback for iterate_dir. The callback records the range of memory
// block IDs assigned to this NUMA node. The return values are Linux kernel
// errno values, because the caller is Linux's iterate_dir() routine.
static int filldir_get_memblock_id(struct dir_context *ctx,
                                   const char *name,
                                   int name_len,
                                   loff_t offset,
                                   u64 ino,
                                   unsigned int d_type)
{
    nv_dir_context_t *ats_ctx = container_of(ctx, nv_dir_context_t, ctx);
    char name_copy[BUF_SIZE];
    NvU64 memblock_id = 0;

    // Check if this is a memory node
    if (!strnstr(name, "memory", name_len))
        return 0;

    if (name_len + 1 > BUF_SIZE)
        return -ERANGE;

    strncpy(name_copy, name, name_len);
    *(name_copy + name_len) = '\0';

    // Convert the memory block ID into an integer
    if (kstrtoull(name_copy + strlen(MEMBLOCK_PREFIX), 0, &memblock_id) != 0) 
    {
        nv_printf(NV_DBG_ERRORS, "memblock_id parsing failed. Path: %s\n", name_copy);
        return -ERANGE;
    }

    nv_printf(NV_DBG_INFO, "Found memblock entry %llu\n", memblock_id);

    // Record the smallest and largest assigned memblock IDs
    ats_ctx->memblock_start_id = min(ats_ctx->memblock_start_id, memblock_id);
    ats_ctx->memblock_end_id = max(ats_ctx->memblock_end_id, memblock_id);

    return 0;
}

/*
 * Brings memory block online using the sysfs memory-hotplug interface
 *   https://www.kernel.org/doc/Documentation/memory-hotplug.txt
 *
 * Note, since we don't currently offline memory on driver unload this routine
 * silently ignores requests when the existing memblock state matches the desired
 * state.
 */
static NV_STATUS change_memblock_state(int numa_node_id, int mem_block_id, mem_state_t new_state)
{
    NV_STATUS status;
    char numa_file_path[BUF_SIZE];
    const char *cmd;

    sprintf(numa_file_path, MEMBLK_STATE_PATH_FMT, mem_block_id);

    switch (new_state) 
    {
        case MEM_ONLINE:
            cmd = BRING_ONLINE_CMD;
            break;
        case MEM_OFFLINE:
            cmd = BRING_OFFLINE_CMD;
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    status = bad_idea_write_string_to_file(numa_file_path, cmd, strlen(cmd));
    if (status == NV_OK)
        nv_printf(NV_DBG_INFO, "Successfully changed state of %s to %s\n", numa_file_path,
                      mem_state_to_string(new_state));
    else
        nv_printf(NV_DBG_ERRORS, "Changing state of %s to %s failed. Error code: [%d]\n",
                      numa_file_path, mem_state_to_string(new_state), status);

    return status;
}

// Looks through NUMA nodes, finding the upper and lower bounds, and returns those.
// The assumption is that the nodes are physically contiguous, so that the intervening
// nodes do not need to be explicitly returned.
static NV_STATUS gather_memblock_ids_for_node
(
    NvU32  node_id,
    NvU64 *memblock_start_id,                                          
    NvU64 *memblock_end_id
)
{
    char numa_file_path[BUF_SIZE];
    struct file *filp;
    int err; 
    nv_dir_context_t ats_ctx = { .ctx.actor = (filldir_t)filldir_get_memblock_id };

    memset(numa_file_path, 0, sizeof(numa_file_path));
    sprintf(numa_file_path, "%s%d", NID_PATH, node_id);

    // TODO: Bug 1735381: don't open files from within kernel code.
    filp = filp_open(numa_file_path, O_RDONLY, 0);
    if (IS_ERR(filp)) 
    {
        nv_printf(NV_DBG_ERRORS, "filp_open failed\n");
        return NV_ERR_NO_VALID_PATH;
    }

    ats_ctx.memblock_start_id = NV_U64_MAX;
    ats_ctx.memblock_end_id = 0;
    ats_ctx.numa_node_id = node_id;

    err = iterate_dir(filp, &ats_ctx.ctx);

    filp_close(filp, NULL);

    if (err != 0) 
    {
        nv_printf(NV_DBG_ERRORS, "iterate_dir(path: %s) failed: %d\n", numa_file_path, err);
        return NV_ERR_NO_VALID_PATH;
    }

    // If the wrong directory was specified, iterate_dir can return success,
    // even though it never iterated any files in the directory. Make that case
    // also an error, by verifying that ats_ctx.memblock_start_id has been set.
    if (ats_ctx.memblock_start_id == NV_U64_MAX)
    {
        nv_printf(NV_DBG_ERRORS, "Failed to find any files in: %s\n", numa_file_path);
        return NV_ERR_NO_VALID_PATH;
    }

    *memblock_start_id = ats_ctx.memblock_start_id;
    *memblock_end_id = ats_ctx.memblock_end_id;

    return NV_OK;
}

static NV_STATUS change_numa_node_state
(
    NvU32       node_id, 
    NvU64       region_gpu_addr,
    NvU64       region_gpu_size, 
    NvU64       memblock_size, 
    mem_state_t new_state
)
{
    NV_STATUS status;
    NvU64 mem_begin, mem_end, memblock_id;
    NvU64 memblock_start_id = 0;
    NvU64 memblock_end_id = 0;
    NvU64 blocks_changed = 0;

    status = gather_memblock_ids_for_node(node_id, &memblock_start_id, &memblock_end_id);
    if (status != NV_OK)
        return status;
    if (memblock_start_id > memblock_end_id)
        return NV_ERR_OPERATING_SYSTEM;

    nv_printf(NV_DBG_INFO, "memblock ID range: %llu-%llu, memblock size: 0x%llx\n",
                memblock_start_id, memblock_end_id, memblock_size);

    mem_begin = region_gpu_addr;
    mem_end   = mem_begin + region_gpu_size - 1;

    nv_printf(NV_DBG_INFO, "GPU memory begin-end: 0x%llx-0x%llx\n", mem_begin, mem_end);

    if (new_state == MEM_ONLINE) 
    {
        // Online ALL memblocks backwards first to allow placement into zone movable
        // Issue discussed here: https://patchwork.kernel.org/patch/9625081/
        memblock_id = memblock_end_id;
        do {
            status = change_memblock_state(node_id, memblock_id, MEM_ONLINE);
            if (status == NV_OK)
                blocks_changed++;
        } while (memblock_id-- > memblock_start_id);
    }
    else if (new_state == MEM_OFFLINE) 
    {
        memblock_id = memblock_start_id;
        do {
            status = change_memblock_state(node_id, memblock_id, MEM_OFFLINE);
            // Ignore failures on the offline/driver unload path for now, it is possible to
            // fail and that case is not currently handled at all (e.g. should block driver unload)
            // Will be handled as part of Bug 1930447
            blocks_changed++;
        } while (memblock_id++ < memblock_end_id);
    }

    // Discard the status. Instead: if we got even one block changed, call it good enough
    // and return NV_OK.
    // TODO: figure out how to recover from "some, but not all requested blocks were
    // changed".
    if (blocks_changed * memblock_size < region_gpu_size) 
    {
        nv_printf(NV_DBG_ERRORS, "Changing the state of some of the memory to %s failed. Error code: [%d]\n",
                      mem_state_to_string(new_state), status);
    }

    if (blocks_changed == 0)
        return NV_ERR_INSUFFICIENT_RESOURCES;

    return NV_OK;
}

static NV_STATUS probe_node_memory
(
    NvU64 probe_base_addr, 
    NvU64 region_gpu_size, 
    NvU64 memblock_size
)
{
    NvU64 start_addr, ats_end_addr;
    char start_addr_str[BUF_SIZE];
    NV_STATUS status = NV_OK;

    ats_end_addr = probe_base_addr + region_gpu_size;
    
    if ((!NV_IS_ALIGNED(probe_base_addr, memblock_size)) || 
        (!NV_IS_ALIGNED(ats_end_addr, memblock_size)))
    {
        nv_printf(NV_DBG_ERRORS, "Probe ranges not aligned to memblock size!\n");
        return NV_ERR_INVALID_ADDRESS; 
    }

    for (start_addr = probe_base_addr;
         start_addr + memblock_size <= ats_end_addr;
         start_addr += memblock_size) 
    {
        sprintf(start_addr_str, "0x%llx", start_addr);

        nv_printf(NV_DBG_INFO, "Probing memory address %s\n", start_addr_str);

        status = bad_idea_write_string_to_file(MEMORY_PROBE_PATH,
                                               start_addr_str,
                                               strlen(start_addr_str));

        //
        // Checking if memory was already probed (e.g. in the previous invocation
        // of this function).
        //
        if (status == NV_ERR_IN_USE) 
        {
            status = NV_OK;
        }
        else if (status != NV_OK) 
        {
            nv_printf(NV_DBG_ERRORS, "Probing of memory address %s failed. Error code: [%d]\n",
                          start_addr_str, status);
            goto done;
        }
    }
done:
    return status;
}

NV_STATUS nv_numa_memblock_size
(
    NvU64 *memblock_size
)
{
    if (nv_ctl_device.numa_memblock_size == 0)
        return NV_ERR_INVALID_STATE;
    *memblock_size = nv_ctl_device.numa_memblock_size;
    return NV_OK;
}



/*! @brief
 *  We assume the physical memory has been allocated from RM before calling this function. 
 */
NV_STATUS nv_numa_online_memory
(
    NvU32  node_id,
    NvU64  region_gpu_addr,
    NvU64  region_gpu_size, 
    NvU64  ats_base_addr,
    NvU64  memblock_size,
    NvBool should_probe
)
{
    NV_STATUS status = NV_OK;

    // Otherwise we'll have a memory leak.
    if ((!NV_IS_ALIGNED(region_gpu_addr, memblock_size)) || 
        (!NV_IS_ALIGNED(region_gpu_size, memblock_size)))
    {
        nv_printf(NV_DBG_ERRORS, "Onlining range is not aligned to memblock size!\n");
        return NV_ERR_INVALID_ADDRESS; 
    }
    
    //
    // We can safely skip memory probing on kernels where memory comes enabled
    // (e.g. pseudop9).
    //
    if (should_probe)
    {
        status = probe_node_memory((ats_base_addr + region_gpu_addr), region_gpu_size, memblock_size);
        
        if (status != NV_OK) 
        {
            nv_printf(NV_DBG_ERRORS,"Probing memory failed. Error code: [%d]\n", status);
            return status;
        }
    }

    status = change_numa_node_state(node_id, region_gpu_addr, region_gpu_size, 
                                    memblock_size, MEM_ONLINE);
    if (status != NV_OK)
        goto error;

    nv_printf(NV_DBG_ERRORS, "XXX: Memory onlining completed!\n");
    
    return status;

error:
    nv_numa_offline_memory(node_id);
    return status;
}

void nv_numa_offline_memory(NvU32 node_id)
{
    // 
    // TODO: actually take the NUMA memory node offline. So far, we are only
    // free-ing memory, not really doing all that the function name implies.
    //
    // As it stands, attempting to take the NUMA memory node offline will
    // probably fail, because there's no guarantee that unplug will succeed,
    // even if the whole node belongs to ZONE_MOVABLE.
    //

    return; 
}

