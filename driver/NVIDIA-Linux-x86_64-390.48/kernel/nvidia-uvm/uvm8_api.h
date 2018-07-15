/*******************************************************************************
    Copyright (c) 2015 NVIDIA Corporation

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

#ifndef __UVM8_API_H__
#define __UVM8_API_H__

#include "uvmtypes.h"
#include "uvm_ioctl.h"
#include "uvm_linux.h"
#include "uvm8_lock.h"
#include "uvm8_thread_context.h"
#include "uvm8_kvmalloc.h"

// This weird number comes from UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS. That
// ioctl is called frequently so we don't want to allocate a copy every time.
// It's a little over 256 bytes in size.
#define UVM_MAX_IOCTL_PARAM_STACK_SIZE 288

// The UVM_ROUTE_CMD_* macros are only intended for use in the ioctl routines

// If the BUILD_BUG_ON fires, use UVM_ROUTE_CMD_ALLOC instead.
#define UVM_ROUTE_CMD_STACK(cmd, function_name)                             \
    case cmd:                                                               \
    {                                                                       \
        cmd##_PARAMS params;                                                \
        BUILD_BUG_ON(sizeof(params) > UVM_MAX_IOCTL_PARAM_STACK_SIZE);      \
        if (nv_copy_from_user(&params, (void __user*)arg, sizeof(params)))  \
            return -EFAULT;                                                 \
                                                                            \
        params.rmStatus = uvm_global_get_status();                          \
        if (params.rmStatus == NV_OK)                                       \
            params.rmStatus = function_name(&params, filp);                 \
        if (nv_copy_to_user((void __user*)arg, &params, sizeof(params)))    \
            return -EFAULT;                                                 \
                                                                            \
        uvm_thread_assert_all_unlocked();                                   \
                                                                            \
        return 0;                                                           \
    }

// If the BUILD_BUG_ON fires, use UVM_ROUTE_CMD_STACK instead
#define UVM_ROUTE_CMD_ALLOC(cmd, function_name)                              \
    case cmd:                                                                \
    {                                                                        \
        int ret = 0;                                                         \
        cmd##_PARAMS *params = uvm_kvmalloc(sizeof(*params));                \
        if (!params)                                                         \
            return -ENOMEM;                                                  \
        BUILD_BUG_ON(sizeof(*params) <= UVM_MAX_IOCTL_PARAM_STACK_SIZE);     \
        if (nv_copy_from_user(params, (void __user*)arg, sizeof(*params))) { \
            uvm_kvfree(params);                                              \
            return -EFAULT;                                                  \
        }                                                                    \
                                                                             \
        params->rmStatus = uvm_global_get_status();                          \
        if (params->rmStatus == NV_OK)                                       \
            params->rmStatus = function_name(params, filp);                  \
        if (nv_copy_to_user((void __user*)arg, params, sizeof(*params)))     \
            ret = -EFAULT;                                                   \
                                                                             \
        uvm_thread_assert_all_unlocked();                                    \
                                                                             \
        uvm_kvfree(params);                                                  \
        return ret;                                                          \
    }

// Validate input ranges from the user with specific alignment requirement
static bool uvm_api_range_invalid_aligned(NvU64 base, NvU64 length, NvU64 alignment)
{
    return !IS_ALIGNED(base, alignment)     ||
           !IS_ALIGNED(length, alignment)   ||
           base == 0                        ||
           length == 0                      ||
           base + length < base; // Overflow
}

// Most APIs require PAGE_SIZE alignment
static bool uvm_api_range_invalid(NvU64 base, NvU64 length)
{
    return uvm_api_range_invalid_aligned(base, length, PAGE_SIZE);
}

// Some APIs can only enforce 4K alignment as it's the smallest GPU page size
// even when the smallest host page is larger (e.g. 64K on ppc64le).
static bool uvm_api_range_invalid_4k(NvU64 base, NvU64 length)
{
    return uvm_api_range_invalid_aligned(base, length, 4096);
}

NV_STATUS uvm_api_is_8_supported(UVM_IS_8_SUPPORTED_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_pageable_mem_access(UVM_PAGEABLE_MEM_ACCESS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_pageable_mem_access_on_gpu(UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_register_gpu(UVM_REGISTER_GPU_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unregister_gpu(UVM_UNREGISTER_GPU_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_create_range_group(UVM_CREATE_RANGE_GROUP_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_destroy_range_group(UVM_DESTROY_RANGE_GROUP_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_enable_peer_access(UVM_ENABLE_PEER_ACCESS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_disable_peer_access(UVM_DISABLE_PEER_ACCESS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_set_range_group(UVM_SET_RANGE_GROUP_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_map_external_allocation(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_free(UVM_FREE_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_prevent_migration_range_groups(UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_allow_migration_range_groups(UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_set_preferred_location(UVM_SET_PREFERRED_LOCATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unset_preferred_location(UVM_UNSET_PREFERRED_LOCATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_set_accessed_by(UVM_SET_ACCESSED_BY_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unset_accessed_by(UVM_UNSET_ACCESSED_BY_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_register_gpu_va_space(UVM_REGISTER_GPU_VASPACE_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unregister_gpu_va_space(UVM_UNREGISTER_GPU_VASPACE_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_register_channel(UVM_REGISTER_CHANNEL_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unregister_channel(UVM_UNREGISTER_CHANNEL_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_enable_read_duplication(UVM_ENABLE_READ_DUPLICATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_disable_read_duplication(UVM_DISABLE_READ_DUPLICATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_migrate(UVM_MIGRATE_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_enable_system_wide_atomics(UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_disable_system_wide_atomics(UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_init_event_tracker(UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_set_notification_threshold(UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_event_queue_enable_events(UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_event_queue_disable_events(UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_enable_counters(UVM_TOOLS_ENABLE_COUNTERS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_disable_counters(UVM_TOOLS_DISABLE_COUNTERS_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_read_process_memory(UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_tools_write_process_memory(UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_map_dynamic_parallelism_region(UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_unmap_external_allocation(UVM_UNMAP_EXTERNAL_ALLOCATION_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_migrate_range_group(UVM_MIGRATE_RANGE_GROUP_PARAMS *params, struct file *filp);
NV_STATUS uvm_api_alloc_semaphore_pool(UVM_ALLOC_SEMAPHORE_POOL_PARAMS *params, struct file *filp);

#endif // __UVM8_API_H__
