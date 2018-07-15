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

#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_lock.h"
#include "uvm8_global.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_perf_heuristics.h"
#include "uvm8_user_channel.h"
#include "uvm8_tools.h"
#include "uvm8_hal.h"
#include "uvm8_map_external.h"
#include "uvm_common.h"
#include "nv_uvm_interface.h"
#include "nv-kthread-q.h"

static NV_STATUS enable_peers(uvm_va_space_t *va_space, uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2);
static void disable_peers(uvm_va_space_t *va_space,
                          uvm_gpu_t *gpu0,
                          uvm_gpu_t *gpu1,
                          struct list_head *deferred_free_list);
static void remove_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space, struct list_head *deferred_free_list);

static void init_tools_data(uvm_va_space_t *va_space)
{
    int i;

    uvm_init_rwsem(&va_space->tools.lock, UVM_LOCK_ORDER_VA_SPACE_TOOLS);

    for (i = 0; i < ARRAY_SIZE(va_space->tools.counters); i++)
        INIT_LIST_HEAD(va_space->tools.counters + i);
    for (i = 0; i < ARRAY_SIZE(va_space->tools.queues); i++)
        INIT_LIST_HEAD(va_space->tools.queues + i);
}

static NV_STATUS init_npu_context(uvm_va_space_t *va_space, uvm_gpu_va_space_t *gpu_va_space)
{
#if defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT)
    if (uvm8_ats_mode) {
        struct npu_context *npu_context;

        // TODO: Bug 1896767: Add a callback here. See the comments on unsafe_mm
        //       in uvm8_va_space.h.
        UVM_ASSERT(current->mm == va_space->unsafe_mm);
        uvm_assert_mmap_sem_locked_write(&current->mm->mmap_sem);

        npu_context = pnv_npu2_init_context(gpu_va_space->gpu->pci_dev, (MSR_DR | MSR_PR | MSR_HV), NULL, NULL);
        if (IS_ERR(npu_context)) {
            NV_STATUS status = errno_to_nv_status(PTR_ERR(npu_context));
            UVM_ERR_PRINT("pnv_npu2_init_context failed: %s (%d), GPU %s\n",
                          nvstatusToString(status),
                          (int)PTR_ERR(npu_context),
                          gpu_va_space->gpu->name);
            return status;
        }

        gpu_va_space->npu_context = npu_context;
    }
#endif

    return NV_OK;
}

static void destroy_npu_context(uvm_gpu_va_space_t *gpu_va_space)
{
#if defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT)
    if (uvm8_ats_mode && gpu_va_space->npu_context) {
        // TODO: Bug 1896767: See the comments on unsafe_mm in uvm8_va_space.h
        pnv_npu2_destroy_context(gpu_va_space->npu_context, gpu_va_space->gpu->pci_dev);
        gpu_va_space->npu_context = NULL;
    }
#endif
}

static NV_STATUS register_gpu_nvlink_peers(uvm_va_space_t *va_space, uvm_gpu_t *gpu)
{
    uvm_gpu_t *other_gpu;

    uvm_assert_rwsem_locked(&va_space->lock);

    for_each_gpu_in_mask(other_gpu, &va_space->registered_gpus) {
        uvm_gpu_peer_t *peer_caps;

        if (other_gpu->id == gpu->id)
            continue;

        peer_caps = uvm_gpu_peer_caps(gpu, other_gpu);

        if (peer_caps->link_type >= UVM_GPU_LINK_NVLINK_1) {
            NV_STATUS status = enable_peers(va_space, gpu, other_gpu);
            if (status != NV_OK)
                return status;
        }
    }

    return NV_OK;
}

NV_STATUS uvm_va_space_create(struct inode *inode, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space;
    va_space = uvm_kvmalloc_zero(sizeof(*va_space));

    if (!va_space)
        return NV_ERR_NO_MEMORY;

    uvm_init_rwsem(&va_space->lock, UVM_LOCK_ORDER_VA_SPACE);
    uvm_mutex_init(&va_space->serialize_writers_lock, UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS);
    uvm_mutex_init(&va_space->read_acquire_write_release_lock, UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK);
    uvm_range_tree_init(&va_space->va_range_tree);

    // By default all struct files on the same inode share the same
    // address_space structure (the inode's) across all processes. This means
    // unmap_mapping_range would unmap virtual mappings across all processes on
    // that inode.
    //
    // Since the UVM driver uses the mapping offset as the VA of the file's
    // process, we need to isolate the mappings to each process.
    address_space_init_once(&va_space->mapping);
    va_space->mapping.host = inode;

    // Some paths in the kernel, for example force_page_cache_readahead which
    // can be invoked from user-space via madvise MADV_WILLNEED and fadvise
    // POSIX_FADV_WILLNEED, check the function pointers within
    // file->f_mapping->a_ops for validity. However, those paths assume that a_ops
    // itself is always valid. Handle that by using the inode's a_ops pointer,
    // which is what f_mapping->a_ops would point to anyway if we weren't re-
    // assigning f_mapping.
    va_space->mapping.a_ops = inode->i_mapping->a_ops;

#if defined(NV_ADDRESS_SPACE_HAS_BACKING_DEV_INFO)
    va_space->mapping.backing_dev_info = inode->i_mapping->backing_dev_info;
#endif

    // Init to 0 since we rely on atomic_inc_return behavior to return 1 as the first ID
    atomic64_set(&va_space->range_group_id_counter, 0);

    INIT_RADIX_TREE(&va_space->range_groups, NV_UVM_GFP_FLAGS);
    uvm_range_tree_init(&va_space->range_group_ranges);

    bitmap_zero(va_space->enabled_peers, UVM_MAX_UNIQUE_GPU_PAIRS);

    // CPU is not explicitly registered in the va space
    uvm_processor_mask_set(&va_space->can_access[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->accessible_from[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->can_copy_from[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->has_native_atomics[UVM_CPU_ID], UVM_CPU_ID);
    // CPU always participates in system-wide atomics
    uvm_processor_mask_set(&va_space->system_wide_atomics_enabled_processors, UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->faultable_processors, UVM_CPU_ID);

#if defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT)
    if (uvm8_ats_mode) {
        // TODO: Bug 1896767: Be as retrictive as possible when using
        //       unsafe_mm. See the comments on unsafe_mm in
        //       uvm8_va_space.h.
        va_space->unsafe_mm = current->mm;
    }
#endif

    filp->private_data = va_space;
    filp->f_mapping = &va_space->mapping;

    va_space->test_page_prefetch_enabled = true;

    init_tools_data(va_space);

    uvm_va_space_down_write(va_space);

    status = uvm_perf_init_va_space_events(va_space, &va_space->perf_events);
    if (status != NV_OK)
        goto fail;

    status = uvm_perf_heuristics_load(va_space);
    if (status != NV_OK)
        goto fail;

    status = uvm_gpu_init_va_space(va_space);
    if (status != NV_OK)
        goto fail;

    uvm_va_space_up_write(va_space);

    return NV_OK;

fail:
    uvm_perf_heuristics_unload(va_space);
    uvm_perf_destroy_va_space_events(&va_space->perf_events);
    uvm_va_space_up_write(va_space);

    uvm_kvfree(va_space);

    return status;
}

// This does *not* release the GPU, nor this GPU's peer pairings. Those are
// returned so the caller can do it after dropping the VA space lock.
static void unregister_gpu(uvm_va_space_t *va_space,
                           uvm_gpu_t *gpu,
                           struct list_head *deferred_free_list,
                           uvm_processor_mask_t *peers_to_release)
{
    uvm_gpu_t *peer_gpu;
    uvm_va_range_t *va_range;
    NvU32 peer_table_index;

    if (peers_to_release)
        uvm_processor_mask_zero(peers_to_release);

    // If a GPU VA Space was explicitly registered, but not explicitly
    // unregistered, unregister it and add all of its objects to the free list.
    remove_gpu_va_space(uvm_gpu_va_space_get(va_space, gpu), deferred_free_list);

    uvm_for_each_va_range(va_range, va_space)
        uvm_va_range_unregister_gpu(va_range, gpu, deferred_free_list);

    // If this GPU has any peer-to-peer pair that was explicitly enabled, but
    // not explicitly disabled, disable it.
    // Notably do this only after unregistering the GPU from VA ranges to make
    // sure there is no pending work using the peer mappings within the VA
    // blocks (in particular migrations using the peer identity mappings).
    for_each_gpu_in_mask(peer_gpu, &va_space->registered_gpus) {
        if (gpu == peer_gpu)
            continue;

        peer_table_index = uvm_gpu_peer_table_index(gpu->id, peer_gpu->id);
        if (test_bit(peer_table_index, va_space->enabled_peers)) {
            disable_peers(va_space, gpu, peer_gpu, deferred_free_list);
            if (peers_to_release)
                uvm_processor_mask_set(peers_to_release, peer_gpu->id);
        }
    }

    if (gpu->isr.replayable_faults.handling)
        uvm_processor_mask_clear(&va_space->faultable_processors, gpu->id);

    uvm_processor_mask_clear(&va_space->system_wide_atomics_enabled_processors, gpu->id);

    uvm_processor_mask_clear(&va_space->can_access[gpu->id], gpu->id);
    uvm_processor_mask_clear(&va_space->can_access[gpu->id], UVM_CPU_ID);
    uvm_processor_mask_clear(&va_space->can_access[UVM_CPU_ID], gpu->id);
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->can_access[gpu->id]));

    uvm_processor_mask_clear(&va_space->accessible_from[gpu->id], gpu->id);
    uvm_processor_mask_clear(&va_space->accessible_from[UVM_CPU_ID], gpu->id);
    uvm_processor_mask_clear(&va_space->accessible_from[gpu->id], UVM_CPU_ID);
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->accessible_from[gpu->id]));

    uvm_processor_mask_clear(&va_space->can_copy_from[gpu->id], gpu->id);
    uvm_processor_mask_clear(&va_space->can_copy_from[UVM_CPU_ID], gpu->id);
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->can_copy_from[gpu->id]));

    uvm_processor_mask_clear(&va_space->has_nvlink[gpu->id], UVM_CPU_ID);
    uvm_processor_mask_clear(&va_space->has_nvlink[UVM_CPU_ID], gpu->id);
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->has_nvlink[gpu->id]));

    UVM_ASSERT(uvm_processor_mask_empty(&va_space->indirect_peers[gpu->id]));

    uvm_processor_mask_clear(&va_space->has_native_atomics[gpu->id], UVM_CPU_ID);
    uvm_processor_mask_clear(&va_space->has_native_atomics[UVM_CPU_ID], gpu->id);
    uvm_processor_mask_clear(&va_space->has_native_atomics[gpu->id], gpu->id);
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->has_native_atomics[gpu->id]));

    uvm_processor_mask_clear(&va_space->registered_gpus, gpu->id);
}

static void gpu_va_space_stop_all_channels(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_user_channel_t *user_channel;

    list_for_each_entry(user_channel, &gpu_va_space->registered_channels, list_node)
        uvm_user_channel_stop(user_channel);

    // Prevent new channels from being registered since we'll be dropping the
    // VA space lock shortly with the expectation that no more channels will
    // arrive.
    atomic_set(&gpu_va_space->disallow_new_channels, 1);
}

// Detaches (unregisters) all user channels in a GPU VA space. The channels must
// have previously been stopped.
//
// The detached channels are added to the input list. The caller is expected to
// drop the VA space lock and call uvm_deferred_free_object_list to complete the
// destroy operation.
static void uvm_gpu_va_space_detach_all_channels(uvm_gpu_va_space_t *gpu_va_space, struct list_head *deferred_free_list)
{
    uvm_user_channel_t *user_channel, *next_channel;
    list_for_each_entry_safe(user_channel, next_channel, &gpu_va_space->registered_channels, list_node)
        uvm_user_channel_detach(user_channel, deferred_free_list);
}

void uvm_va_space_destroy(struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_range_t *va_range, *va_range_next;
    uvm_gpu_t *gpu, *peer_gpu;
    uvm_gpu_id_t gpu_id;

    uvm_processor_mask_t retained_gpus;
    LIST_HEAD(deferred_free_list);

    // Stop all channels before unmapping anything. This kills the channels and
    // prevents spurious MMU faults from being generated (bug 1722021), but
    // doesn't prevent the bottom half from servicing old faults for those
    // channels.
    //
    // This involves making RM calls, so we have to do that with the VA space
    // lock in read mode.
    uvm_va_space_down_read_rm(va_space);
    uvm_va_space_stop_all_user_channels(va_space);
    uvm_va_space_up_read_rm(va_space);

    // The bottom half GPU page fault handler(s) could still look up and use
    // this va_space via the GPU's instance_ptr_table. Lock them out while we
    // tear down. Once we're done, the bottom half will fail to find any
    // registered GPUs in the VA space, so those faults will be canceled.
    uvm_va_space_down_write(va_space);

    uvm_hmm_mirror_unregister(va_space);

    uvm_processor_mask_copy(&retained_gpus, &va_space->registered_gpus);
    bitmap_copy(va_space->enabled_peers_teardown, va_space->enabled_peers, UVM_MAX_UNIQUE_GPU_PAIRS);

    for_each_gpu_va_space(gpu_va_space, va_space)
        uvm_gpu_va_space_detach_all_channels(gpu_va_space, &deferred_free_list);

    // Destroy all VA ranges. We do this before unregistering the GPUs for
    // performance, since GPU unregister will walk all VA ranges in the VA space
    // multiple times.
    uvm_for_each_va_range_safe(va_range, va_range_next, va_space) {
        // All channel ranges should've been destroyed by the channel unregister
        // above
        UVM_ASSERT(va_range->type != UVM_VA_RANGE_TYPE_CHANNEL);
        uvm_va_range_destroy(va_range, &deferred_free_list);
    }

    uvm_range_group_radix_tree_destroy(va_space);

    // Unregister all GPUs in the VA space. Note that this does not release the
    // GPUs nor peers. We do that below.
    for_each_va_space_gpu(gpu, va_space)
        unregister_gpu(va_space, gpu, &deferred_free_list, NULL);

    uvm_perf_heuristics_unload(va_space);
    uvm_perf_destroy_va_space_events(&va_space->perf_events);

    uvm_va_space_up_write(va_space);

    UVM_ASSERT(uvm_processor_mask_empty(&va_space->registered_gpus));
    UVM_ASSERT(uvm_processor_mask_empty(&va_space->registered_gpu_va_spaces));

    // The instance pointer mappings for this VA space have been removed so no
    // new bottom halves can get to this VA space, but there could still be
    // bottom halves running from before we removed the mapping. Rather than
    // ref-count the VA space, just wait for them to finish.
    //
    // This is also required to synchronize any pending
    // block_deferred_accessed_by() work items.

    nv_kthread_q_flush(&g_uvm_global.global_q);

    for_each_gpu_id_in_mask(gpu_id, &retained_gpus) {
        gpu = uvm_gpu_get(gpu_id);

        if (gpu->isr.replayable_faults.handling ||
            gpu->isr.non_replayable_faults.handling ||
            gpu->isr.access_counters.handling)
            nv_kthread_q_flush(&gpu->isr.bottom_half_q);

        // The same applies to the kill channel kthreads. However, they need to
        // be flushed after their bottom-half counterparts since the latter may
        // schedule a channel kill.
        if (gpu->isr.non_replayable_faults.handling)
            nv_kthread_q_flush(&gpu->isr.kill_channel_q);
    }

    // ensure that there are no pending events that refer to this va_space
    uvm_tools_flush_events();

    // Perform cleanup we can't do while holding the VA space lock

    uvm_deferred_free_object_list(&deferred_free_list);

    uvm_mutex_lock(&g_uvm_global.global_lock);

    // Release the GPUs and their peer counts. Do not use for_each_gpu_in_mask
    // for the outer loop as it reads the GPU state, which might get destroyed.
    for_each_gpu_id_in_mask(gpu_id, &retained_gpus) {
        gpu = uvm_gpu_get(gpu_id);

        uvm_processor_mask_clear(&retained_gpus, gpu_id);

        for_each_gpu_in_mask(peer_gpu, &retained_gpus) {
            NvU32 peer_table_index = uvm_gpu_peer_table_index(gpu_id, peer_gpu->id);
            if (test_bit(peer_table_index, va_space->enabled_peers_teardown)) {
                uvm_gpu_peer_t *peer_caps = &g_uvm_global.peers[peer_table_index];

                if (peer_caps->link_type == UVM_GPU_LINK_PCIE)
                    uvm_gpu_release_pcie_peer_access(gpu, peer_gpu);

                __clear_bit(peer_table_index, va_space->enabled_peers_teardown);
            }
        }

        uvm_gpu_release_locked(gpu);
    }

    UVM_ASSERT(bitmap_empty(va_space->enabled_peers, UVM_MAX_UNIQUE_GPU_PAIRS));
    UVM_ASSERT(bitmap_empty(va_space->enabled_peers_teardown, UVM_MAX_UNIQUE_GPU_PAIRS));

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    filp->private_data = NULL;
    filp->f_mapping = NULL;

    uvm_kvfree(va_space);
}

void uvm_va_space_stop_all_user_channels(uvm_va_space_t *va_space)
{
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_user_channel_t *user_channel;

    // Skip if all channels have been already stopped.
    if (atomic_read(&va_space->user_channels_stopped))
        return;

    uvm_assert_rwsem_locked_read(&va_space->lock);

    for_each_gpu_va_space(gpu_va_space, va_space) {
        list_for_each_entry(user_channel, &gpu_va_space->registered_channels, list_node)
            uvm_user_channel_stop(user_channel);
    }

    // Since we're holding the VA space lock in read mode, multiple threads
    // could set this concurrently. user_channels_stopped never transitions back
    // to 0 after being set to 1 so that's not a problem.
    atomic_set(&va_space->user_channels_stopped, 1);
}

uvm_gpu_t *uvm_va_space_get_gpu_by_uuid(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid)
{
    uvm_gpu_t *gpu;

    for_each_va_space_gpu(gpu, va_space) {
        if (uvm_processor_uuid_eq(&gpu->uuid, gpu_uuid))
            return gpu;
    }

    return NULL;
}

uvm_gpu_t *uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(uvm_va_space_t *va_space,
                                                          const NvProcessorUuid *gpu_uuid)
{
    uvm_gpu_t *gpu;

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu || !uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, gpu->id))
        return NULL;

    return gpu;
}

bool uvm_va_space_can_read_duplicate(uvm_va_space_t *va_space, uvm_gpu_t *changing_gpu)
{
    uvm_processor_mask_t changing_gpu_mask;
    uvm_processor_mask_t non_faultable_gpus;
    uvm_processor_mask_t registered_gpu_va_spaces;

    uvm_processor_mask_zero(&changing_gpu_mask);

    if (changing_gpu)
        uvm_processor_mask_set(&changing_gpu_mask, changing_gpu->id);

    // flip the bit of the changing GPU to represent the state change in progress
    uvm_processor_mask_xor(&registered_gpu_va_spaces, &changing_gpu_mask, &va_space->registered_gpu_va_spaces);

    // Can't enable read-duplication if any non-fault-capable GPUs have GPU VA spaces registered
    return !uvm_processor_mask_andnot(&non_faultable_gpus, &registered_gpu_va_spaces, &va_space->faultable_processors);
}

// Note that the "VA space" in the function name refers to a UVM per-process VA space.
// (This is different from a per-GPU VA space.)
NV_STATUS uvm_va_space_register_gpu(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid)
{
    NV_STATUS status;
    uvm_gpu_t *gpu;
    uvm_gpu_t *other_gpu;

    status = uvm_gpu_retain_by_uuid(gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    uvm_va_space_down_write(va_space);

    // Make sure the gpu hasn't been already registered in this va space
    if (uvm_processor_mask_test(&va_space->registered_gpus, gpu->id)) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    // Mixing Volta and Pascal GPUs is not supported on P9 systems.
    for_each_gpu_in_mask(other_gpu, &va_space->registered_gpus) {
        if ((gpu->sysmem_link >= UVM_GPU_LINK_NVLINK_2 && other_gpu->sysmem_link <  UVM_GPU_LINK_NVLINK_2) ||
            (gpu->sysmem_link <  UVM_GPU_LINK_NVLINK_2 && other_gpu->sysmem_link >= UVM_GPU_LINK_NVLINK_2)) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }
    }

    uvm_processor_mask_set(&va_space->registered_gpus, gpu->id);
    if (gpu->isr.replayable_faults.handling) {
        uvm_processor_mask_set(&va_space->faultable_processors, gpu->id);
        // System-wide atomics are enabled by default
        uvm_processor_mask_set(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    }

    // All GPUs have native atomics on their own memory
    uvm_processor_mask_set(&va_space->has_native_atomics[gpu->id], gpu->id);

    if (gpu->sysmem_link >= UVM_GPU_LINK_NVLINK_1) {
        uvm_processor_mask_set(&va_space->has_nvlink[gpu->id], UVM_CPU_ID);
        uvm_processor_mask_set(&va_space->has_nvlink[UVM_CPU_ID], gpu->id);
    }

    if (gpu->sysmem_link >= UVM_GPU_LINK_NVLINK_2) {
        uvm_processor_mask_set(&va_space->has_native_atomics[gpu->id], UVM_CPU_ID);
        if (gpu->numa_info.enabled) {
            uvm_processor_mask_set(&va_space->can_access[UVM_CPU_ID], gpu->id);
            uvm_processor_mask_set(&va_space->accessible_from[gpu->id], UVM_CPU_ID);
            uvm_processor_mask_set(&va_space->has_native_atomics[UVM_CPU_ID], gpu->id);
        }
    }

    // All processors have direct access to their own memory
    uvm_processor_mask_set(&va_space->can_access[gpu->id], gpu->id);
    uvm_processor_mask_set(&va_space->accessible_from[gpu->id], gpu->id);

    // All GPUs have direct access to sysmem
    uvm_processor_mask_set(&va_space->can_access[gpu->id], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->accessible_from[UVM_CPU_ID], gpu->id);

    uvm_processor_mask_set(&va_space->can_copy_from[gpu->id], gpu->id);
    uvm_processor_mask_set(&va_space->can_copy_from[UVM_CPU_ID], gpu->id);

    status = register_gpu_nvlink_peers(va_space, gpu);
    if (status != NV_OK) {
        // Clear out all of the processor mask bits. No VA ranges have mapped or
        // allocated anything on this GPU yet if we fail here, so we don't need
        // a deferred_free_list.
        unregister_gpu(va_space, gpu, NULL, NULL);
    }

done:
    uvm_va_space_up_write(va_space);

    if (status != NV_OK)
        uvm_gpu_release(gpu);

    return status;
}

NV_STATUS uvm_va_space_unregister_gpu(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid)
{
    uvm_gpu_t *gpu;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_processor_id_t peer_gpu_id;
    uvm_processor_mask_t peers_to_release;
    LIST_HEAD(deferred_free_list);

    // Stopping channels requires holding the VA space lock in read mode, so do
    // it first. We start in write mode then drop to read in order to flush out
    // other threads which are in the read-mode portion of any of the register
    // or unregister operations.
    uvm_va_space_down_write(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        uvm_va_space_up_write(va_space);
        return NV_ERR_INVALID_DEVICE;
    }

    // We have to drop the VA space lock below mid-unregister. We have to
    // prevent any other threads from coming in during that window and allowing
    // new channels to enter the GPU. That means we must disallow:
    // - GPU VA space register
    // - GPU unregister (which would allow new GPU registers)
    if (uvm_processor_mask_test(&va_space->gpu_register_in_progress, gpu->id)) {
        uvm_va_space_up_write(va_space);
        return NV_ERR_INVALID_DEVICE;
    }

    uvm_processor_mask_set(&va_space->gpu_register_in_progress, gpu->id);

    uvm_va_space_downgrade_write_rm(va_space);

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    if (gpu_va_space)
        gpu_va_space_stop_all_channels(gpu_va_space);

    // We need to drop the lock to re-take it in write mode.
    uvm_gpu_retain(gpu);
    uvm_va_space_up_read_rm(va_space);

    // The mmap_sem lock is needed to establish CPU mappings to any pages
    // evicted from the GPU if accessed by CPU is set for them.
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);

    uvm_va_space_down_write(va_space);

    // We blocked out other GPU unregisters, so this GPU must still be
    // registered
    UVM_ASSERT(uvm_processor_mask_test(&va_space->registered_gpus, gpu->id));

    // This will call disable_peers for all GPU's peers, including NVLink
    unregister_gpu(va_space, gpu, &deferred_free_list, &peers_to_release);

    UVM_ASSERT(uvm_processor_mask_test(&va_space->gpu_register_in_progress, gpu->id));
    uvm_processor_mask_clear(&va_space->gpu_register_in_progress, gpu->id);

    uvm_va_space_up_write(va_space);
    uvm_up_read_mmap_sem(&current->mm->mmap_sem);

    // Drop the count we took above
    uvm_gpu_release(gpu);

    uvm_deferred_free_object_list(&deferred_free_list);

    // Release the VA space's GPU and peer counts
    uvm_mutex_lock(&g_uvm_global.global_lock);

    // Do not use for_each_gpu_in_mask as it reads the peer GPU state, which
    // might get destroyed when we release the peer entry.
    for_each_gpu_id_in_mask(peer_gpu_id, &peers_to_release) {
        uvm_gpu_peer_t *peer_caps = uvm_gpu_index_peer_caps(gpu->id, peer_gpu_id);

        if (peer_caps->link_type == UVM_GPU_LINK_PCIE)
            uvm_gpu_release_pcie_peer_access(gpu, uvm_gpu_get(peer_gpu_id));
    }
    uvm_gpu_release_locked(gpu);

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    return NV_OK;
}

// This does *not* release the global GPU peer entry
static void disable_peers(uvm_va_space_t *va_space,
                          uvm_gpu_t *gpu0,
                          uvm_gpu_t *gpu1,
                          struct list_head *deferred_free_list)
{
    NvU32 table_index = uvm_gpu_peer_table_index(gpu0->id, gpu1->id);
    uvm_va_range_t *va_range;

    if (!test_bit(table_index, va_space->enabled_peers))
        return;

    // Unmap all page tables in this VA space which have peer mappings between
    // these two GPUs.
    uvm_for_each_va_range(va_range, va_space)
        uvm_va_range_disable_peer(va_range, gpu0, gpu1, deferred_free_list);

    uvm_processor_mask_clear(&va_space->can_access[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->can_access[gpu1->id], gpu0->id);
    uvm_processor_mask_clear(&va_space->accessible_from[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->accessible_from[gpu1->id], gpu0->id);
    uvm_processor_mask_clear(&va_space->can_copy_from[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->can_copy_from[gpu1->id], gpu0->id);
    uvm_processor_mask_clear(&va_space->has_nvlink[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->has_nvlink[gpu1->id], gpu0->id);
    uvm_processor_mask_clear(&va_space->indirect_peers[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->indirect_peers[gpu1->id], gpu0->id);
    uvm_processor_mask_clear(&va_space->has_native_atomics[gpu0->id], gpu1->id);
    uvm_processor_mask_clear(&va_space->has_native_atomics[gpu1->id], gpu0->id);

    __clear_bit(table_index, va_space->enabled_peers);
}

static NV_STATUS enable_peers(uvm_va_space_t *va_space, uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_va_space_t *gpu_va_space_1, *gpu_va_space_2;
    NvU32 table_index = 0;
    uvm_gpu_peer_t *peer_caps;
    uvm_va_range_t *va_range;
    LIST_HEAD(deferred_free_list);

    uvm_assert_rwsem_locked_write(&va_space->lock);

    // We know the GPUs were retained already, so now verify that they've been
    // registered by this specific VA space.
    if (!uvm_processor_mask_test(&va_space->registered_gpus, gpu_1->id) ||
        !uvm_processor_mask_test(&va_space->registered_gpus, gpu_2->id)) {
        return NV_ERR_INVALID_DEVICE;
    }

    table_index = uvm_gpu_peer_table_index(gpu_1->id, gpu_2->id);
    peer_caps = &g_uvm_global.peers[table_index];

    UVM_ASSERT(!test_bit(table_index, va_space->enabled_peers));

    // If both GPUs have registered GPU VA spaces already, their big page sizes
    // must match.
    gpu_va_space_1 = uvm_gpu_va_space_get(va_space, gpu_1);
    gpu_va_space_2 = uvm_gpu_va_space_get(va_space, gpu_2);
    if (gpu_va_space_1 &&
        gpu_va_space_2 &&
        gpu_va_space_1->page_tables.big_page_size != gpu_va_space_2->page_tables.big_page_size) {
        return NV_ERR_NOT_COMPATIBLE;
    }

    uvm_processor_mask_set(&va_space->can_access[gpu_1->id], gpu_2->id);
    uvm_processor_mask_set(&va_space->can_access[gpu_2->id], gpu_1->id);
    uvm_processor_mask_set(&va_space->accessible_from[gpu_1->id], gpu_2->id);
    uvm_processor_mask_set(&va_space->accessible_from[gpu_2->id], gpu_1->id);

    if (gpu_1->peer_identity_mappings_supported) {
        UVM_ASSERT_MSG(gpu_2->peer_identity_mappings_supported, "GPU %s GPU %s\n", gpu_1->name, gpu_2->name);

        uvm_processor_mask_set(&va_space->can_copy_from[gpu_2->id], gpu_1->id);
        uvm_processor_mask_set(&va_space->can_copy_from[gpu_1->id], gpu_2->id);
    }

    // Pre-compute nvlink and native atomic masks for the new peers
    if (peer_caps->link_type >= UVM_GPU_LINK_NVLINK_1) {
        uvm_processor_mask_set(&va_space->has_nvlink[gpu_1->id], gpu_2->id);
        uvm_processor_mask_set(&va_space->has_nvlink[gpu_2->id], gpu_1->id);

        uvm_processor_mask_set(&va_space->has_native_atomics[gpu_1->id], gpu_2->id);
        uvm_processor_mask_set(&va_space->has_native_atomics[gpu_2->id], gpu_1->id);

        if (!peer_caps->is_direct_peer) {
            UVM_ASSERT(peer_caps->link_type >= UVM_GPU_LINK_NVLINK_2);
            UVM_ASSERT(gpu_1->numa_info.enabled);
            UVM_ASSERT(gpu_2->numa_info.enabled);

            uvm_processor_mask_set(&va_space->indirect_peers[gpu_1->id], gpu_2->id);
            uvm_processor_mask_set(&va_space->indirect_peers[gpu_2->id], gpu_1->id);
        }
    }

    __set_bit(table_index, va_space->enabled_peers);

    uvm_for_each_va_range(va_range, va_space) {
        status = uvm_va_range_enable_peer(va_range, gpu_1, gpu_2);
        if (status != NV_OK)
            break;
    }

    if (status != NV_OK) {
        disable_peers(va_space, gpu_1, gpu_2, &deferred_free_list);

        // uvm_va_range_disable_peer adds only external allocations to the list,
        // but uvm_va_range_enable_peer doesn't do anything for them.
        UVM_ASSERT(list_empty(&deferred_free_list));
    }

    return status;
}

static NV_STATUS retain_pcie_peers_from_uuids(const NvProcessorUuid *gpu_uuid_1,
                                              const NvProcessorUuid *gpu_uuid_2,
                                              uvm_gpu_t **gpu_1,
                                              uvm_gpu_t **gpu_2)
{
    // The UUIDs should have already been retained
    *gpu_1 = uvm_gpu_get_by_uuid(gpu_uuid_1);
    if (!*gpu_1)
        return NV_ERR_INVALID_DEVICE;

    *gpu_2 = uvm_gpu_get_by_uuid(gpu_uuid_2);
    if (!*gpu_2)
        return NV_ERR_INVALID_DEVICE;

    if ((*gpu_1)->id == (*gpu_2)->id)
        return NV_ERR_INVALID_DEVICE;

    return uvm_gpu_retain_pcie_peer_access(*gpu_1, *gpu_2);
}

static NV_STATUS retain_nvlink_peers_from_uuids(const NvProcessorUuid *gpu_uuid_1,
                                                const NvProcessorUuid *gpu_uuid_2,
                                                uvm_gpu_t **gpu_1,
                                                uvm_gpu_t **gpu_2)
{
    NV_STATUS status;
    uvm_gpu_peer_t *peer_caps;

    // The UUIDs should have already been retained
    status = uvm_gpu_retain_by_uuid_locked(gpu_uuid_1, gpu_1);
    if (status != NV_OK)
        return status;

    status = uvm_gpu_retain_by_uuid_locked(gpu_uuid_2, gpu_2);
    if (status != NV_OK) {
        uvm_gpu_release_locked(*gpu_1);
        return status;
    }

    if ((*gpu_1)->id == (*gpu_2)->id) {
        uvm_gpu_release_locked(*gpu_1);
        uvm_gpu_release_locked(*gpu_2);
        return NV_ERR_INVALID_DEVICE;
    }

    peer_caps = uvm_gpu_peer_caps(*gpu_1, *gpu_2);

    if (peer_caps->link_type >= UVM_GPU_LINK_NVLINK_1)
        return NV_OK;

    uvm_gpu_release_locked(*gpu_1);
    uvm_gpu_release_locked(*gpu_2);
    return NV_ERR_INVALID_DEVICE;
}

static bool uvm_va_space_pcie_peer_enabled(uvm_va_space_t *va_space, uvm_gpu_t *gpu1, uvm_gpu_t *gpu2)
{
    return !uvm_processor_mask_test(&va_space->has_nvlink[gpu1->id], gpu2->id) &&
           uvm_va_space_peer_enabled(va_space, gpu1, gpu2);
}

static bool uvm_va_space_nvlink_peer_enabled(uvm_va_space_t *va_space, uvm_gpu_t *gpu1, uvm_gpu_t *gpu2)
{
    return uvm_processor_mask_test(&va_space->has_nvlink[gpu1->id], gpu2->id);
}

static void free_gpu_va_space(nv_kref_t *nv_kref)
{
    uvm_gpu_va_space_t *gpu_va_space = container_of(nv_kref, uvm_gpu_va_space_t, kref);
    UVM_ASSERT(!gpu_va_space->va_space);
    uvm_kvfree(gpu_va_space);
}

void uvm_gpu_va_space_release(uvm_gpu_va_space_t *gpu_va_space)
{
    if (gpu_va_space)
        nv_kref_put(&gpu_va_space->kref, free_gpu_va_space);
}

static void destroy_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
{
    NV_STATUS status;

    if (!gpu_va_space)
        return;

    // The caller must first unregister this GPU VA space from its owning VA
    // space
    UVM_ASSERT(!gpu_va_space->va_space);

    if (gpu_va_space->did_set_page_directory) {
        status = uvm_rm_locked_call(nvUvmInterfaceUnsetPageDirectory(gpu_va_space->duped_gpu_va_space));
        UVM_ASSERT_MSG(status == NV_OK, "nvUvmInterfaceUnsetPageDirectory() failed: %s, GPU %s\n",
                       nvstatusToString(status), gpu_va_space->gpu->name);
    }

    if (gpu_va_space->page_tables.root)
        uvm_page_tree_deinit(&gpu_va_space->page_tables);

    if (gpu_va_space->duped_gpu_va_space)
        uvm_rm_locked_call_void(nvUvmInterfaceAddressSpaceDestroy(gpu_va_space->duped_gpu_va_space));

    uvm_gpu_va_space_release(gpu_va_space);
}

static NV_STATUS create_gpu_va_space(uvm_gpu_t *gpu,
                                     uvm_rm_user_object_t *user_rm_va_space,
                                     uvm_gpu_va_space_t **out_gpu_va_space)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_gpu_phys_address_t pdb_phys;
    NvU64 num_pdes;
    UvmGpuAddressSpaceInfo gpu_address_space_info;

    *out_gpu_va_space = NULL;

    gpu_va_space = uvm_kvmalloc_zero(sizeof(*gpu_va_space));
    if (!gpu_va_space)
        return NV_ERR_NO_MEMORY;

    gpu_va_space->gpu = gpu;
    INIT_LIST_HEAD(&gpu_va_space->registered_channels);
    INIT_LIST_HEAD(&gpu_va_space->channel_va_ranges);
    nv_kref_init(&gpu_va_space->kref);

    // TODO: Bug 1624521: This interface needs to use rm_control_fd to do
    //       validation.
    (void)user_rm_va_space->rm_control_fd;
    status = uvm_rm_locked_call(nvUvmInterfaceDupAddressSpace(g_uvm_global.rm_session_handle,
                                                              &gpu->uuid,
                                                              user_rm_va_space->user_client,
                                                              user_rm_va_space->user_object,
                                                              &gpu_va_space->duped_gpu_va_space,
                                                              &gpu_address_space_info));
    if (status != NV_OK) {
        UVM_DBG_PRINT("failed to dup address space with error: %s, for GPU:%s \n",
                nvstatusToString(status), gpu->name);
        goto error;
    }

    gpu_va_space->ats_enabled = gpu_address_space_info.atsEnabled;

    // If ATS support in the UVM driver isn't enabled, fail registration of GPU
    // VA spaces which have ATS enabled.
    //
    // TODO: Bug 1896767: Also check if the GPU VA space has ATS enabled in the
    //       GPU fault handler.
    if (!uvm8_ats_mode && gpu_va_space->ats_enabled) {
        UVM_INFO_PRINT("GPU VA space requires ATS but the UVM driver was not loaded with ATS support enabled (uvm8_ats_mode=1)\n");
        status = NV_ERR_INVALID_FLAGS;
        goto error;
    }

    // RM allows the creation of VA spaces on Pascal with 128k big pages. We
    // don't support that, so just fail those attempts.
    //
    // TODO: Bug 1789555: Remove this check once RM disallows this case.
    if (!gpu->arch_hal->mmu_mode_hal(gpu_address_space_info.bigPageSize)) {
        status = NV_ERR_INVALID_FLAGS;
        goto error;
    }

    // Set up this GPU's page tables
    UVM_ASSERT(gpu_va_space->page_tables.root == NULL);
    status = uvm_page_tree_init(gpu,
                                gpu_address_space_info.bigPageSize,
                                UVM_APERTURE_DEFAULT,
                                &gpu_va_space->page_tables);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Initializing the page tree failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    // Replace the existing PDB, if present, with the new one allocated by UVM.
    // This will fail if nvUvmInterfaceSetPageDirectory has already been called
    // on the RM VA space object, which prevents the user from registering twice
    // and corrupting our state.
    //
    // TODO: Bug 1733664: RM needs to preempt and disable channels during this
    //       operation.
    pdb_phys = uvm_page_tree_pdb(&gpu_va_space->page_tables)->addr;
    num_pdes = uvm_mmu_page_tree_entries(&gpu_va_space->page_tables, 0, UVM_PAGE_SIZE_AGNOSTIC);
    status = uvm_rm_locked_call(nvUvmInterfaceSetPageDirectory(gpu_va_space->duped_gpu_va_space,
                                                               pdb_phys.address,
                                                               num_pdes,
                                                               pdb_phys.aperture == UVM_APERTURE_VID));
    if (status != NV_OK) {
        UVM_DBG_PRINT("nvUvmInterfaceSetPageDirectory() failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);

        // Convert to the return code specified by uvm.h for already-registered
        // PDBs.
        if (status == NV_ERR_NOT_SUPPORTED)
            status = NV_ERR_INVALID_DEVICE;

        goto error;
    }

    gpu_va_space->did_set_page_directory = true;

    *out_gpu_va_space = gpu_va_space;
    return NV_OK;

error:
    destroy_gpu_va_space(gpu_va_space);
    return status;
}

static NV_STATUS add_gpu_va_space(uvm_va_space_t *va_space, uvm_gpu_va_space_t *gpu_va_space)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_gpu_t *other_gpu;
    uvm_gpu_va_space_t *other_gpu_va_space;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    // This GPU VA space must match its big page size with all enabled peers.
    for_each_gpu_in_mask(other_gpu, &va_space->registered_gpu_va_spaces) {
        UVM_ASSERT(other_gpu != gpu);

        if (!test_bit(uvm_gpu_peer_table_index(gpu->id, other_gpu->id), va_space->enabled_peers))
            continue;

        other_gpu_va_space = uvm_gpu_va_space_get(va_space, other_gpu);
        if (gpu_va_space->page_tables.big_page_size != other_gpu_va_space->page_tables.big_page_size)
            return NV_ERR_NOT_COMPATIBLE;
    }

    status = init_npu_context(va_space, gpu_va_space);
    if (status != NV_OK)
        return status;

    uvm_processor_mask_set(&va_space->registered_gpu_va_spaces, gpu->id);
    va_space->gpu_va_spaces[uvm_gpu_index(gpu->id)] = gpu_va_space;
    gpu_va_space->va_space = va_space;

    return NV_OK;
}

NV_STATUS uvm_va_space_register_gpu_va_space(uvm_va_space_t *va_space,
                                             uvm_rm_user_object_t *user_rm_va_space,
                                             const NvProcessorUuid *gpu_uuid)
{
    NV_STATUS status;
    uvm_gpu_t *gpu;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_range_t *va_range;
    LIST_HEAD(deferred_free_list);

    status = uvm_gpu_retain_by_uuid(gpu_uuid, &gpu);
    if (status != NV_OK) {
        // Convert to the return code specified by uvm.h for bad UUIDs
        if (status == NV_ERR_GPU_UUID_NOT_FOUND)
            return NV_ERR_INVALID_DEVICE;
        return status;
    }

    status = create_gpu_va_space(gpu, user_rm_va_space, &gpu_va_space);
    if (status != NV_OK) {
        uvm_gpu_release(gpu);
        return status;
    }

    uvm_down_write_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_write(va_space);

    if (!uvm_processor_mask_test(&va_space->registered_gpus, gpu->id)) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    // RM will return an error from create_gpu_va_space if the given RM VA space
    // object has already been registered by any VA space. Now we just need to
    // check if a different VA space has already been registered.
    if (uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, gpu->id)) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    // If a GPU unregister is in progress but temporarily dropped the VA space
    // lock, we can't register new GPU VA spaces.
    if (uvm_processor_mask_test(&va_space->gpu_register_in_progress, gpu->id)) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    status = add_gpu_va_space(va_space, gpu_va_space);
    if (status != NV_OK)
        goto error;

    // Tell the VA ranges that they can map this GPU, if they need to.
    //
    // Ideally we'd downgrade the VA space lock to read mode while adding new
    // mappings, but that would complicate error handling since we have to
    // remove the GPU VA space if any of these mappings fail.
    uvm_for_each_va_range(va_range, va_space) {
        status = uvm_va_range_add_gpu_va_space(va_range, gpu_va_space);
        if (status != NV_OK)
            goto error;
    }

    uvm_va_space_up_write(va_space);
    uvm_up_write_mmap_sem(&current->mm->mmap_sem);
    uvm_gpu_release(gpu);
    return NV_OK;

error:
    if (gpu_va_space->va_space) {
        remove_gpu_va_space(gpu_va_space, &deferred_free_list);

        // Nothing else could've been attached to this gpu_va_space (channels,
        // external allocations) since we're still holding the VA space lock.
        // Therefore the GPU VA space itself should be the only item in the
        // list, and we can just destroy it directly below.
        UVM_ASSERT(list_is_singular(&deferred_free_list));
    }

    uvm_va_space_up_write(va_space);
    uvm_up_write_mmap_sem(&current->mm->mmap_sem);

    destroy_gpu_va_space(gpu_va_space);
    uvm_gpu_release(gpu);
    return status;
}

// The caller must have stopped all channels under this gpu_va_space before
// calling this function.
static void remove_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space, struct list_head *deferred_free_list)
{
    uvm_va_space_t *va_space;
    uvm_va_range_t *va_range;

    if (!gpu_va_space || !gpu_va_space->va_space)
        return;

    UVM_ASSERT(gpu_va_space->gpu);
    va_space = gpu_va_space->va_space;
    uvm_assert_rwsem_locked_write(&va_space->lock);

    destroy_npu_context(gpu_va_space);

    uvm_gpu_va_space_detach_all_channels(gpu_va_space, deferred_free_list);

    // Removing all registered channels should've removed all VA ranges used by
    // those channels.
    UVM_ASSERT(list_empty(&gpu_va_space->channel_va_ranges));

    // Unmap all page tables in this VA space on this GPU.
    // TODO: Bug 1799173: This will need to add objects to deferred_free_list
    uvm_for_each_va_range(va_range, va_space)
        uvm_va_range_remove_gpu_va_space(va_range, gpu_va_space, deferred_free_list);

    uvm_deferred_free_object_add(deferred_free_list,
                                 &gpu_va_space->deferred_free,
                                 UVM_DEFERRED_FREE_OBJECT_GPU_VA_SPACE);

    uvm_processor_mask_clear(&va_space->registered_gpu_va_spaces, gpu_va_space->gpu->id);
    va_space->gpu_va_spaces[uvm_gpu_index(gpu_va_space->gpu->id)] = NULL;
    gpu_va_space->va_space = NULL;
}

NV_STATUS uvm_va_space_unregister_gpu_va_space(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    uvm_gpu_va_space_t *gpu_va_space;
    LIST_HEAD(deferred_free_list);

    // Stopping channels requires holding the VA space lock in read mode, so do
    // it first. This also takes the serialize_writers_lock, so we'll serialize
    // with other threads about to perform channel binds in
    // uvm_register_channel since.
    uvm_va_space_down_read_rm(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, gpu_uuid);
    if (!gpu) {
        uvm_va_space_up_read_rm(va_space);
        return NV_ERR_INVALID_DEVICE;
    }

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    UVM_ASSERT(gpu_va_space);

    gpu_va_space_stop_all_channels(gpu_va_space);

    // We need to drop the lock to re-take it in write mode
    uvm_gpu_va_space_retain(gpu_va_space);
    uvm_gpu_retain(gpu);
    uvm_va_space_up_read_rm(va_space);

    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_write(va_space);

    // We dropped the lock so we have to re-verify that this gpu_va_space is
    // still valid. If so, then the GPU is also still registered under the VA
    // space. If not, we raced with another unregister thread, so return an
    // an error for double-unregister.
    if (!gpu_va_space->va_space) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    UVM_ASSERT(gpu == uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, gpu_uuid));
    UVM_ASSERT(gpu_va_space == uvm_gpu_va_space_get(va_space, gpu));

    remove_gpu_va_space(gpu_va_space, &deferred_free_list);

done:
    uvm_va_space_up_write(va_space);
    uvm_up_read_mmap_sem(&current->mm->mmap_sem);
    uvm_deferred_free_object_list(&deferred_free_list);
    uvm_gpu_va_space_release(gpu_va_space);
    uvm_gpu_release(gpu);
    return status;
}

bool uvm_va_space_peer_enabled(uvm_va_space_t *va_space, uvm_gpu_t *gpu1, uvm_gpu_t *gpu2)
{
    size_t table_index;

    UVM_ASSERT(uvm_processor_mask_test(&va_space->registered_gpus, gpu1->id));
    UVM_ASSERT(uvm_processor_mask_test(&va_space->registered_gpus, gpu2->id));

    table_index = uvm_gpu_peer_table_index(gpu1->id, gpu2->id);
    return !!test_bit(table_index, va_space->enabled_peers);
}

uvm_processor_id_t uvm_processor_mask_find_closest_id(uvm_va_space_t *va_space,
                                                      const uvm_processor_mask_t *candidates,
                                                      uvm_processor_id_t src)
{
    uvm_processor_mask_t mask;
    uvm_processor_id_t id;

    // Highest priority: the local processor itself
    if (uvm_processor_mask_test(candidates, src))
        return src;

    // NVLINK peers. Prefer direct peers over indirect peers
    uvm_processor_mask_and(&mask, candidates, &va_space->has_nvlink[src]);
    if (uvm_processor_mask_andnot(&mask, &mask, &va_space->indirect_peers[src]))
        return uvm_processor_mask_find_first_id(&mask);

    if (uvm_processor_mask_and(&mask, candidates, &va_space->has_nvlink[src]))
        return uvm_processor_mask_find_first_id(&mask);

    // PCIe peers
    // TODO: Bug 1764943: Is copying from a PCI peer always better than copying from CPU?
    if (src != UVM_CPU_ID) {
        uvm_processor_mask_and(&mask, candidates, &va_space->can_access[src]);
        id = uvm_processor_mask_find_next_id(&mask, UVM_CPU_ID + 1);
        if (id != UVM_MAX_PROCESSORS)
            return id;
    }

    // No GPUs with direct access are in the mask. Prioritize the CPU next since
    // we always have direct access to it. Then if all else fails, just pick the
    // next available processor.
    //
    // We can do both of these with a single find_first_id, since the CPU is
    // always first. Note that if src is the CPU, we already checked whether the
    // CPU was present above.
    return uvm_processor_mask_find_first_id(candidates);
}

static void uvm_deferred_free_object_channel(uvm_deferred_free_object_t *object, uvm_processor_mask_t *flushed_gpus)
{
    uvm_user_channel_t *channel = container_of(object, uvm_user_channel_t, deferred_free);
    uvm_gpu_t *gpu = channel->gpu;

    // Flush out any faults with this instance pointer still in the buffer. This
    // prevents us from re-allocating the same instance pointer for a new
    // channel and mis-attributing old faults to it.
    if (gpu->replayable_faults_supported && !uvm_processor_mask_test(flushed_gpus, gpu->id)) {
        uvm_gpu_fault_buffer_flush(gpu);
        uvm_processor_mask_set(flushed_gpus, gpu->id);
    }

    uvm_user_channel_destroy_detached(channel);
}

void uvm_deferred_free_object_list(struct list_head *deferred_free_list)
{
    uvm_deferred_free_object_t *object, *next;
    uvm_processor_mask_t flushed_gpus;

    // Used if there are any channels in the list
    uvm_processor_mask_zero(&flushed_gpus);

    list_for_each_entry_safe(object, next, deferred_free_list, list_node) {
        list_del(&object->list_node);

        switch (object->type) {
            case UVM_DEFERRED_FREE_OBJECT_TYPE_CHANNEL:
                uvm_deferred_free_object_channel(object, &flushed_gpus);
                break;
            case UVM_DEFERRED_FREE_OBJECT_GPU_VA_SPACE:
                destroy_gpu_va_space(container_of(object, uvm_gpu_va_space_t, deferred_free));
                break;
            case UVM_DEFERRED_FREE_OBJECT_TYPE_EXTERNAL_ALLOCATION:
                uvm_ext_gpu_map_free(container_of(object, uvm_ext_gpu_map_t, deferred_free));
                break;
            default:
                UVM_ASSERT_MSG(0, "Invalid type %d\n", object->type);
        }
    }
}

uvm_user_channel_t *uvm_gpu_va_space_get_user_channel(uvm_gpu_va_space_t *gpu_va_space,
                                                      uvm_gpu_phys_address_t instance_ptr)
{
    uvm_user_channel_t *user_channel;
    uvm_va_space_t *va_space = gpu_va_space->va_space;

    UVM_ASSERT(va_space);
    uvm_assert_rwsem_locked(&va_space->lock);

    // TODO: Bug 1880191: This is called on every non-replayable fault service.
    // Evaluate the performance impact of this list traversal and potentially
    // replace it with something better.
    list_for_each_entry(user_channel, &gpu_va_space->registered_channels, list_node) {
        if (user_channel->instance_ptr.address == instance_ptr.address &&
            user_channel->instance_ptr.aperture == instance_ptr.aperture) {
            return user_channel;
        }
    }

    return NULL;
}

NV_STATUS uvm_api_enable_peer_access(UVM_ENABLE_PEER_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu_1 = NULL;
    uvm_gpu_t *gpu_2 = NULL;
    size_t table_index;

    uvm_mutex_lock(&g_uvm_global.global_lock);
    status = retain_pcie_peers_from_uuids(&params->gpuUuidA, &params->gpuUuidB, &gpu_1, &gpu_2);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    if (status != NV_OK)
        return status;

    uvm_va_space_down_write(va_space);

    table_index = uvm_gpu_peer_table_index(gpu_1->id, gpu_2->id);
    if (test_bit(table_index, va_space->enabled_peers))
        status = NV_ERR_INVALID_DEVICE;
    else
        status = enable_peers(va_space, gpu_1, gpu_2);

    uvm_va_space_up_write(va_space);

    if (status != NV_OK) {
        uvm_mutex_lock(&g_uvm_global.global_lock);
        uvm_gpu_release_pcie_peer_access(gpu_1, gpu_2);
        uvm_mutex_unlock(&g_uvm_global.global_lock);
    }

    return status;
}

NV_STATUS uvm_api_disable_peer_access(UVM_DISABLE_PEER_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu_1, *gpu_2;
    LIST_HEAD(deferred_free_list);

    uvm_va_space_down_write(va_space);

    gpu_1 = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpuUuidA);
    gpu_2 = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpuUuidB);

    if (!gpu_1 || !gpu_2) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    if (gpu_1->id == gpu_2->id) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    if (!uvm_va_space_pcie_peer_enabled(va_space, gpu_1, gpu_2)) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    disable_peers(va_space, gpu_1, gpu_2, &deferred_free_list);

    // disable_peers doesn't release the GPU peer ref count, which means the two
    // GPUs will remain retained even if another thread unregisters them from
    // this VA space after we drop the lock.
    uvm_va_space_up_write(va_space);

    uvm_deferred_free_object_list(&deferred_free_list);

    uvm_mutex_lock(&g_uvm_global.global_lock);
    uvm_gpu_release_pcie_peer_access(gpu_1, gpu_2);
    uvm_mutex_unlock(&g_uvm_global.global_lock);

    return NV_OK;

error:
    uvm_va_space_up_write(va_space);
    return status;
}

NV_STATUS uvm8_test_flush_deferred_work(UVM_TEST_FLUSH_DEFERRED_WORK_PARAMS *params, struct file *filp)
{
    UvmTestDeferredWorkType work_type = params->work_type;

    switch (work_type) {
        case UvmTestDeferredWorkTypeAcessedByMappings:
            nv_kthread_q_flush(&g_uvm_global.global_q);
            return NV_OK;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }
}

NV_STATUS uvm8_test_enable_nvlink_peer_access(UVM_TEST_ENABLE_NVLINK_PEER_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu_1 = NULL;
    uvm_gpu_t *gpu_2 = NULL;
    size_t table_index;

    // Retain GPUs so that they don't go away during peer registration
    uvm_mutex_lock(&g_uvm_global.global_lock);
    status = retain_nvlink_peers_from_uuids(&params->gpuUuidA, &params->gpuUuidB, &gpu_1, &gpu_2);
    uvm_mutex_unlock(&g_uvm_global.global_lock);

    if (status != NV_OK)
        return status;

    uvm_va_space_down_write(va_space);

    table_index = uvm_gpu_peer_table_index(gpu_1->id, gpu_2->id);

    // NVLink peers are automatically enabled in the VA space at VA space
    // registration time. In order to avoid tests having to keep track of the
    // different initial state for PCIe and NVLink peers, we just return NV_OK
    // if NVLink peer were already enabled.
    if (test_bit(table_index, va_space->enabled_peers))
        status = NV_OK;
    else
        status = enable_peers(va_space, gpu_1, gpu_2);

    uvm_va_space_up_write(va_space);

    uvm_gpu_release(gpu_1);
    uvm_gpu_release(gpu_2);

    return status;
}

NV_STATUS uvm8_test_disable_nvlink_peer_access(UVM_TEST_DISABLE_NVLINK_PEER_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu_1, *gpu_2;
    LIST_HEAD(deferred_free_list);

    uvm_va_space_down_write(va_space);

    gpu_1 = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpuUuidA);
    gpu_2 = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpuUuidB);

    if (!gpu_1 || !gpu_2) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    if (gpu_1->id == gpu_2->id) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    if (!uvm_va_space_nvlink_peer_enabled(va_space, gpu_1, gpu_2)) {
        status = NV_ERR_INVALID_DEVICE;
        goto error;
    }

    disable_peers(va_space, gpu_1, gpu_2, &deferred_free_list);

    uvm_va_space_up_write(va_space);

    uvm_deferred_free_object_list(&deferred_free_list);

    return NV_OK;

error:
    uvm_va_space_up_write(va_space);
    return status;
}
