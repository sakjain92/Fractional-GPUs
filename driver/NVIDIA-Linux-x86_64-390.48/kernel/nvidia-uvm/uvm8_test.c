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

#include "uvm8_api.h"
#include "uvm8_test.h"
#include "uvm8_test_ioctl.h"
#include "uvm8_global.h"
#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"
#include "uvm8_test_rng.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_perf_events.h"
#include "uvm8_tools.h"
#include "uvm8_mmu.h"
#include "uvm8_gpu_access_counters.h"

static NV_STATUS uvm8_test_get_gpu_ref_count(UVM_TEST_GET_GPU_REF_COUNT_PARAMS *params, struct file *filp)
{
    NvU64 retained_count = 0;
    uvm_gpu_t *gpu;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);

    if (gpu != NULL)
        retained_count = uvm_gpu_retained_count(gpu);

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    params->ref_count = retained_count;
    return NV_OK;
}

static NV_STATUS uvm8_test_peer_ref_count(UVM_TEST_PEER_REF_COUNT_PARAMS *params, struct file *filp)
{
    NvU64 registered_ref_count = 0;
    uvm_gpu_t *gpu_1 = NULL;
    uvm_gpu_t *gpu_2 = NULL;
    NV_STATUS status = NV_OK;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu_1 = uvm_gpu_get_by_uuid(&params->gpu_uuid_1);
    gpu_2 = uvm_gpu_get_by_uuid(&params->gpu_uuid_2);

    if (gpu_1 != NULL && gpu_2 != NULL) {
        uvm_gpu_peer_t *peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);
        registered_ref_count = peer_caps->ref_count;
    }
    else {
        status = NV_ERR_INVALID_DEVICE;
    }

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    params->ref_count = registered_ref_count;

    return status;
}

static NV_STATUS uvm8_test_make_channel_stops_immediate(
        UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE_PARAMS *params,
        struct file *filp)
{
    uvm_va_space_get(filp)->user_channel_stops_are_immediate = NV_TRUE;

    return NV_OK;
}

static NV_STATUS uvm8_test_nv_kthread_q(
        UVM_TEST_NV_KTHREAD_Q_PARAMS *params,
        struct file *filp)
{
    // The nv-kthread-q system returns 0 or -1, because it is not actually
    // part of UVM. UVM needs to run this test, because otherwise, the
    // nv-kthread-q code would not get adequate test coverage. That's because
    // UVM is the first user of nv-kthread-q.
    int result = nv_kthread_q_run_self_test();
    if (result == 0)
        return NV_OK;

    return NV_ERR_INVALID_STATE;
}

static NV_STATUS uvm8_test_get_kernel_virtual_address(
        UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS_PARAMS *params,
        struct file *filp)
{
    params->addr = (NvU64)uvm_va_space_get(filp);

    return NV_OK;
}

long uvm8_test_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    // Disable all test entry points if the module parameter wasn't provided.
    // These should not be enabled in a production environment.
    if (!uvm_enable_builtin_tests) {
        UVM_INFO_PRINT("ioctl %d not found. Did you mean to insmod with uvm_enable_builtin_tests=1?\n", cmd);
        return -EINVAL;
    }

    switch (cmd)
    {
        UVM_ROUTE_CMD_STACK(UVM_TEST_GET_GPU_REF_COUNT,             uvm8_test_get_gpu_ref_count);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RNG_SANITY,                    uvm8_test_rng_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_TREE_DIRECTED,           uvm8_test_range_tree_directed);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_TREE_RANDOM,             uvm8_test_range_tree_random);
        UVM_ROUTE_CMD_ALLOC(UVM_TEST_VA_RANGE_INFO,                 uvm8_test_va_range_info);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RM_MEM_SANITY,                 uvm8_test_rm_mem_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_GPU_SEMAPHORE_SANITY,          uvm8_test_gpu_semaphore_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PEER_REF_COUNT,                uvm8_test_peer_ref_count);
        UVM_ROUTE_CMD_STACK(UVM_TEST_VA_RANGE_SPLIT,                uvm8_test_va_range_split);
        UVM_ROUTE_CMD_STACK(UVM_TEST_VA_RANGE_INJECT_SPLIT_ERROR,   uvm8_test_va_range_inject_split_error);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PAGE_TREE,                     uvm8_test_page_tree);
        UVM_ROUTE_CMD_STACK(UVM_TEST_CHANGE_PTE_MAPPING,            uvm8_test_change_pte_mapping);
        UVM_ROUTE_CMD_STACK(UVM_TEST_TRACKER_SANITY,                uvm8_test_tracker_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PUSH_SANITY,                   uvm8_test_push_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_CHANNEL_SANITY,                uvm8_test_channel_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_CHANNEL_STRESS,                uvm8_test_channel_stress);
        UVM_ROUTE_CMD_STACK(UVM_TEST_CE_SANITY,                     uvm8_test_ce_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_VA_BLOCK_INFO,                 uvm8_test_va_block_info);
        UVM_ROUTE_CMD_STACK(UVM_TEST_LOCK_SANITY,                   uvm8_test_lock_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PERF_UTILS_SANITY,             uvm8_test_perf_utils_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_KVMALLOC,                      uvm8_test_kvmalloc);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_QUERY,                     uvm8_test_pmm_query);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_CHECK_LEAK,                uvm8_test_pmm_check_leak);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PERF_EVENTS_SANITY,            uvm8_test_perf_events_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PERF_MODULE_SANITY,            uvm8_test_perf_module_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_ALLOCATOR_SANITY,        uvm8_test_range_allocator_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_GET_RM_PTES,                   uvm8_test_get_rm_ptes);
        UVM_ROUTE_CMD_STACK(UVM_TEST_FAULT_BUFFER_FLUSH,            uvm8_test_fault_buffer_flush);
        UVM_ROUTE_CMD_STACK(UVM_TEST_INJECT_TOOLS_EVENT,            uvm8_test_inject_tools_event);
        UVM_ROUTE_CMD_STACK(UVM_TEST_INCREMENT_TOOLS_COUNTER,       uvm8_test_increment_tools_counter);
        UVM_ROUTE_CMD_STACK(UVM_TEST_MEM_SANITY,                    uvm8_test_mem_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_MMU_SANITY,                    uvm8_test_mmu_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE,  uvm8_test_make_channel_stops_immediate);
        UVM_ROUTE_CMD_STACK(UVM_TEST_VA_BLOCK_INJECT_ERROR,         uvm8_test_va_block_inject_error);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PEER_IDENTITY_MAPPINGS,        uvm8_test_peer_identity_mappings);
        UVM_ROUTE_CMD_ALLOC(UVM_TEST_VA_RESIDENCY_INFO,             uvm8_test_va_residency_info);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_ASYNC_ALLOC,               uvm8_test_pmm_async_alloc);
        UVM_ROUTE_CMD_STACK(UVM_TEST_SET_PREFETCH_FILTERING,        uvm8_test_set_prefetch_filtering);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_SANITY,                    uvm8_test_pmm_sanity);
        UVM_ROUTE_CMD_STACK(UVM_TEST_INVALIDATE_TLB,                uvm8_test_invalidate_tlb);
        UVM_ROUTE_CMD_STACK(UVM_TEST_VA_BLOCK,                      uvm8_test_va_block);
        UVM_ROUTE_CMD_STACK(UVM_TEST_EVICT_CHUNK,                   uvm8_test_evict_chunk);
        UVM_ROUTE_CMD_STACK(UVM_TEST_FLUSH_DEFERRED_WORK,           uvm8_test_flush_deferred_work);
        UVM_ROUTE_CMD_STACK(UVM_TEST_NV_KTHREAD_Q,                  uvm8_test_nv_kthread_q);
        UVM_ROUTE_CMD_STACK(UVM_TEST_SET_PAGE_PREFETCH_POLICY,      uvm8_test_set_page_prefetch_policy);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_GROUP_TREE,              uvm8_test_range_group_tree);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_GROUP_RANGE_INFO,        uvm8_test_range_group_range_info);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RANGE_GROUP_RANGE_COUNT,       uvm8_test_range_group_range_count);
        UVM_ROUTE_CMD_STACK(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE, uvm8_test_get_prefetch_faults_reenable_lapse);
        UVM_ROUTE_CMD_STACK(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE, uvm8_test_set_prefetch_faults_reenable_lapse);
        UVM_ROUTE_CMD_STACK(UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS,    uvm8_test_get_kernel_virtual_address);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMA_ALLOC_FREE,                uvm8_test_pma_alloc_free);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_ALLOC_FREE_ROOT,           uvm8_test_pmm_alloc_free_root);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_INJECT_PMA_EVICT_ERROR,    uvm8_test_pmm_inject_pma_evict_error);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RECONFIGURE_ACCESS_COUNTERS,   uvm8_test_reconfigure_access_counters);
        UVM_ROUTE_CMD_STACK(UVM_TEST_RESET_ACCESS_COUNTERS,         uvm8_test_reset_access_counters);
        UVM_ROUTE_CMD_STACK(UVM_TEST_SET_IGNORE_ACCESS_COUNTERS,    uvm8_test_set_ignore_access_counters);
        UVM_ROUTE_CMD_STACK(UVM_TEST_CHECK_CHANNEL_VA_SPACE,        uvm8_test_check_channel_va_space);
        UVM_ROUTE_CMD_STACK(UVM_TEST_ENABLE_NVLINK_PEER_ACCESS,     uvm8_test_enable_nvlink_peer_access);
        UVM_ROUTE_CMD_STACK(UVM_TEST_DISABLE_NVLINK_PEER_ACCESS,    uvm8_test_disable_nvlink_peer_access);
        UVM_ROUTE_CMD_STACK(UVM_TEST_GET_PAGE_THRASHING_POLICY,     uvm8_test_get_page_thrashing_policy);
        UVM_ROUTE_CMD_STACK(UVM_TEST_SET_PAGE_THRASHING_POLICY,     uvm8_test_set_page_thrashing_policy);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_SYSMEM,                    uvm8_test_pmm_sysmem);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_REVERSE_MAP,               uvm8_test_pmm_reverse_map);
        UVM_ROUTE_CMD_STACK(UVM_TEST_PMM_INDIRECT_PEERS,            uvm8_test_pmm_indirect_peers);
    }

    return -EINVAL;
}
