/*******************************************************************************
    Copyright (c) 2017 NVIDIA Corporation

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

#include "nv_uvm_interface.h"
#include "uvm_common.h"
#include "uvm8_gpu_non_replayable_faults.h"
#include "uvm8_gpu.h"
#include "uvm8_hal.h"
#include "uvm8_lock.h"
#include "uvm8_tools.h"
#include "uvm8_user_channel.h"
#include "uvm8_va_block.h"
#include "uvm8_va_range.h"
#include "uvm8_kvmalloc.h"

// TODO: Bug 1881601: [uvm] Add fault handling overview for replayable and
// non-replayable faults

// There is no error handling in this function. The caller is in charge of
// calling uvm_gpu_fault_buffer_deinit_non_replayable_faults on failure.
NV_STATUS uvm_gpu_fault_buffer_init_non_replayable_faults(uvm_gpu_t *gpu)
{
    uvm_non_replayable_fault_buffer_info_t *non_replayable_faults = &gpu->fault_buffer_info.non_replayable;

    UVM_ASSERT(gpu->non_replayable_faults_supported);

    non_replayable_faults->shadow_buffer_copy = NULL;
    non_replayable_faults->fault_cache        = NULL;

    non_replayable_faults->max_faults = gpu->fault_buffer_info.rm_info.nonReplayable.bufferSize /
                                        gpu->fault_buffer_hal->entry_size(gpu);

    non_replayable_faults->shadow_buffer_copy =
        uvm_kvmalloc_zero(gpu->fault_buffer_info.rm_info.nonReplayable.bufferSize);
    if (!non_replayable_faults->shadow_buffer_copy)
        return NV_ERR_NO_MEMORY;

    non_replayable_faults->fault_cache = uvm_kvmalloc_zero(non_replayable_faults->max_faults *
                                                           sizeof(*non_replayable_faults->fault_cache));
    if (!non_replayable_faults->fault_cache)
        return NV_ERR_NO_MEMORY;

    uvm_tracker_init(&non_replayable_faults->clear_faulted_tracker);
    uvm_tracker_init(&non_replayable_faults->fault_service_tracker);

    return NV_OK;
}

void uvm_gpu_fault_buffer_deinit_non_replayable_faults(uvm_gpu_t *gpu)
{
    uvm_non_replayable_fault_buffer_info_t *non_replayable_faults = &gpu->fault_buffer_info.non_replayable;

    if (non_replayable_faults->fault_cache) {
        NV_STATUS status = uvm_tracker_wait_deinit(&non_replayable_faults->clear_faulted_tracker);
        if (status != NV_OK)
            UVM_ASSERT(status == uvm_global_get_status());

        UVM_ASSERT(uvm_tracker_is_empty(&non_replayable_faults->fault_service_tracker));
        uvm_tracker_deinit(&non_replayable_faults->fault_service_tracker);
    }

    uvm_kvfree(non_replayable_faults->shadow_buffer_copy);
    uvm_kvfree(non_replayable_faults->fault_cache);
    non_replayable_faults->shadow_buffer_copy = NULL;
    non_replayable_faults->fault_cache        = NULL;
}

bool uvm_gpu_non_replayable_faults_pending(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    NvBool has_pending_faults;

    UVM_ASSERT(gpu->isr.non_replayable_faults.handling);

    status = nvUvmInterfaceHasPendingNonReplayableFaults(&gpu->fault_buffer_info.rm_info,
                                                         &has_pending_faults);
    UVM_ASSERT(status == NV_OK);

    return has_pending_faults == NV_TRUE;
}

static NvU32 fetch_non_replayable_fault_buffer_entries(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    NvU32 i = 0;
    NvU32 cached_faults = 0;
    uvm_fault_buffer_entry_t *fault_cache;
    NvU32 entry_size = gpu->fault_buffer_hal->entry_size(gpu);
    uvm_non_replayable_fault_buffer_info_t *non_replayable_faults = &gpu->fault_buffer_info.non_replayable;
    char *current_hw_entry = (char *)non_replayable_faults->shadow_buffer_copy;

    fault_cache = non_replayable_faults->fault_cache;

    uvm_assert_mutex_locked(&gpu->isr.non_replayable_faults.service_lock);
    UVM_ASSERT(gpu->non_replayable_faults_supported);

    status = nvUvmInterfaceGetNonReplayableFaults(&gpu->fault_buffer_info.rm_info,
                                                  non_replayable_faults->shadow_buffer_copy,
                                                  &cached_faults);
    UVM_ASSERT(status == NV_OK);

    // Parse all faults
    for (i = 0; i < cached_faults; ++i) {
        uvm_fault_buffer_entry_t *fault_entry = &non_replayable_faults->fault_cache[i];

        gpu->fault_buffer_hal->parse_non_replayable_entry(gpu, current_hw_entry, fault_entry);

        // The GPU aligns the fault addresses to 4k, but all of our tracking is
        // done in PAGE_SIZE chunks which might be larger.
        fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address);

        // Make sure that all fields in the entry are properly initialized
        fault_entry->va_space = NULL;
        fault_entry->is_fatal = (fault_entry->fault_type >= UVM_FAULT_TYPE_FATAL);
        fault_entry->filtered = false;

        fault_entry->num_instances = 1;
        fault_entry->access_type_mask = uvm_fault_access_type_mask_bit(fault_entry->fault_access_type);
        INIT_LIST_HEAD(&fault_entry->merged_instances_list);

        // RM must handle all fatal fault types and they will not show up in the
        // shadow buffer
        UVM_ASSERT(!fault_entry->is_fatal);

        current_hw_entry += entry_size;
    }

    return cached_faults;
}

static NV_STATUS push_clear_faulted_on_gpu(uvm_gpu_t *gpu,
                                           uvm_user_channel_t *user_channel,
                                           uvm_fault_buffer_entry_t *fault_entry,
                                           NvU32 batch_id,
                                           uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_non_replayable_fault_buffer_info_t *non_replayable_faults = &gpu->fault_buffer_info.non_replayable;

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    tracker,
                                    &push,
                                    "Clearing set bit for address 0x%llx",
                                    fault_entry->fault_address);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Error acquiring tracker before clearing faulted: %s, GPU %s\n",
                      nvstatusToString(status), gpu->name);
        return status;
    }

    gpu->host_hal->clear_faulted_channel(&push, user_channel, fault_entry);

    uvm_tools_broadcast_replay(gpu, &push, batch_id, fault_entry->fault_source.client_type);

    uvm_push_end(&push);

    // Add this push to the GPU's clear_faulted_tracker so GPU removal can wait
    // on it.
    status = uvm_tracker_add_push_safe(&non_replayable_faults->clear_faulted_tracker, &push);

    return status;
}

static NV_STATUS service_non_replayable_fault_block_locked(uvm_gpu_t *gpu,
                                                           uvm_va_block_t *va_block,
                                                           uvm_va_block_retry_t *va_block_retry,
                                                           uvm_fault_buffer_entry_t *fault_entry,
                                                           uvm_fault_service_block_context_t *service_context)
{
    NV_STATUS status = NV_OK;
    uvm_page_index_t page_index;
    uvm_perf_thrashing_hint_t thrashing_hint;
    uvm_processor_id_t new_residency;
    bool read_duplicate;
    uvm_va_space_t *va_space;
    uvm_va_range_t *va_range = va_block->va_range;
    uvm_non_replayable_fault_buffer_info_t *non_replayable_faults = &gpu->fault_buffer_info.non_replayable;

    UVM_ASSERT(va_range);
    va_space = va_range->va_space;
    uvm_assert_rwsem_locked(&va_space->lock);
    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

    UVM_ASSERT(fault_entry->va_space == va_space);
    UVM_ASSERT(fault_entry->fault_address >= va_block->start);
    UVM_ASSERT(fault_entry->fault_address <= va_block->end);

    if (service_context->num_retries == 0) {
        // notify event to tools/performance heuristics. For now we use a
        // unique batch id per fault, since we clear the faulted channel for
        // each fault.
        uvm_perf_event_notify_gpu_fault(&va_space->perf_events,
                                        va_block,
                                        gpu->id,
                                        fault_entry,
                                        ++non_replayable_faults->batch_id,
                                        false);
    }

    // Check logical permissions
    status = uvm_va_range_check_logical_permissions(va_block->va_range,
                                                    gpu->id,
                                                    fault_entry->fault_access_type,
                                                    uvm_range_group_address_migratable(va_space,
                                                                                       fault_entry->fault_address));
    if (status != NV_OK) {
        fault_entry->is_fatal = true;
        fault_entry->fatal_reason = uvm_tools_status_to_fatal_fault_reason(status);
        return NV_OK;
    }

    // TODO: Bug 1880194: Revisit thrashing detection
    thrashing_hint.type = UVM_PERF_THRASHING_HINT_TYPE_NONE;

    service_context->read_duplicate_count = 0;
    service_context->thrashing_pin_count = 0;
    service_context->are_replayable_faults = false;

    page_index = uvm_va_block_cpu_page_index(va_block, fault_entry->fault_address);

    // Compute new residency and update the masks
    new_residency = uvm_va_block_select_residency_after_fault(va_block,
                                                              page_index,
                                                              gpu->id,
                                                              fault_entry->access_type_mask,
                                                              &thrashing_hint,
                                                              &read_duplicate);

    // Initialize the minimum necessary state in the fault service context
    uvm_processor_mask_zero(&service_context->resident_processors);

    // Set new residency and update the masks
    uvm_processor_mask_set(&service_context->resident_processors, new_residency);

    // The masks need to be fully zeroed as the fault region may grow due to prefetching
    uvm_page_mask_zero(&service_context->per_processor_masks[new_residency].new_residency);
    uvm_page_mask_set(&service_context->per_processor_masks[new_residency].new_residency, page_index);

    if (read_duplicate) {
        uvm_page_mask_zero(&service_context->read_duplicate_mask);
        uvm_page_mask_set(&service_context->read_duplicate_mask, page_index);
        service_context->read_duplicate_count = 1;
    }

    service_context->fault_access_type[page_index] = fault_entry->fault_access_type;

    service_context->fault_region = uvm_va_block_region_for_page(page_index);

    status = uvm_va_block_service_faults_locked(gpu->id, va_block, va_block_retry, service_context);

    ++service_context->num_retries;

    return status;
}

static NV_STATUS service_non_replayable_fault_block(uvm_gpu_t *gpu,
                                                    uvm_va_block_t *va_block,
                                                    uvm_fault_buffer_entry_t *fault_entry)
{
    NV_STATUS status, tracker_status;
    uvm_va_block_retry_t va_block_retry;
    uvm_fault_service_block_context_t *service_context = &gpu->fault_buffer_info.non_replayable.block_service_context;

    service_context->num_retries = 0;

    uvm_mutex_lock(&va_block->lock);

    status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                                       service_non_replayable_fault_block_locked(gpu,
                                                                                 va_block,
                                                                                 &va_block_retry,
                                                                                 fault_entry,
                                                                                 service_context));

    tracker_status = uvm_tracker_add_tracker_safe(&gpu->fault_buffer_info.non_replayable.fault_service_tracker,
                                                  &va_block->tracker);

    uvm_mutex_unlock(&va_block->lock);

    return status == NV_OK? tracker_status: status;
}

// See uvm_unregister_channel for comments on the the channel destruction
// sequence.
static void kill_channel_delayed(void *_user_channel)
{
    uvm_user_channel_t *user_channel = (uvm_user_channel_t *)_user_channel;
    uvm_va_space_t *va_space = user_channel->kill_channel.va_space;

    uvm_va_space_down_read_rm(va_space);
    // TODO: Bug 1873655: RM should log fatal MMU faults
    if (user_channel->gpu_va_space)
        uvm_user_channel_stop(user_channel);
    uvm_va_space_up_read_rm(va_space);

    uvm_user_channel_release(user_channel);
}

static void schedule_kill_channel(uvm_gpu_t *gpu,
                                  uvm_va_space_t *va_space,
                                  uvm_user_channel_t *user_channel)
{
    UVM_ASSERT(gpu);
    UVM_ASSERT(va_space);
    UVM_ASSERT(user_channel);

    if (user_channel->kill_channel.scheduled)
        return;

    user_channel->kill_channel.scheduled = true;
    user_channel->kill_channel.va_space = va_space;

    // Retain the channel here so it is not prematurely destroyed. It will be
    // released after channel cleanup in kill_channel_delayed.
    uvm_user_channel_retain(user_channel);

    // Schedule a work item to kill the channel
    nv_kthread_q_item_init(&user_channel->kill_channel.kill_channel_q_item,
                           kill_channel_delayed,
                           user_channel);

    nv_kthread_q_schedule_q_item(&gpu->isr.kill_channel_q,
                                 &user_channel->kill_channel.kill_channel_q_item);
}

static NV_STATUS service_non_replayable_fault(uvm_gpu_t *gpu,
                                              uvm_fault_buffer_entry_t *fault_entry)
{
    NV_STATUS status;
    uvm_user_channel_t *user_channel;
    uvm_va_block_t *va_block;
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_va_space_t *gpu_va_space;

    status = uvm_gpu_fault_entry_to_va_space(gpu, fault_entry, &va_space);
    if (status != NV_OK) {
        // The VA space lookup will fail if we're running concurrently with
        // removal of the channel from the VA space (channel unregister, GPU VA
        // space unregister, VA space destroy, etc). The other thread will stop
        // the channel and remove the channel from the table, so the faulting
        // condition will be gone. In the case of replayable faults we need to
        // flush the buffer, but here we can just ignore the entry and proceed
        // on.
        //
        // Note that we can't have any subcontext issues here, since non-
        // replayable faults only use the address space of their channel.
        UVM_ASSERT(status == NV_ERR_INVALID_CHANNEL);
        UVM_ASSERT(!va_space);
        return NV_OK;
    }
    else {
        UVM_ASSERT(va_space);
    }

    // TODO: Bug 1896767: Handle non-replayable ATS faults

    uvm_va_space_down_read(va_space);

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);

    if (!gpu_va_space) {
        // The va_space might have gone away. See the comment above.
        status = NV_OK;
        goto exit_no_channel;
    }

    fault_entry->va_space = va_space;

    user_channel = uvm_gpu_va_space_get_user_channel(gpu_va_space, fault_entry->instance_ptr);
    if (!user_channel) {
        // The channel might have gone away. See the comment above.
        status = NV_OK;
        goto exit_no_channel;
    }

    fault_entry->fault_source.channel_id = user_channel->hw_channel_id;

    status = uvm_va_block_find_create(fault_entry->va_space, fault_entry->fault_address, &va_block);
    if (status == NV_OK) {
        status = service_non_replayable_fault_block(gpu, va_block, fault_entry);

        // We are done, we clear the faulted bit on the channel, so it can be re-scheduled again
        if (status == NV_OK && !fault_entry->is_fatal) {
            status = push_clear_faulted_on_gpu(gpu,
                                               user_channel,
                                               fault_entry,
                                               gpu->fault_buffer_info.non_replayable.batch_id,
                                               &gpu->fault_buffer_info.non_replayable.fault_service_tracker);
            uvm_tracker_clear(&gpu->fault_buffer_info.non_replayable.fault_service_tracker);
        }

    }
    else if (status == NV_ERR_INVALID_ADDRESS) {
        fault_entry->is_fatal = true;
        fault_entry->fatal_reason = uvm_tools_status_to_fatal_fault_reason(status);

        // Do not return error due to logical errors in the application
        status = NV_OK;
    }

    if (fault_entry->is_fatal) {
        uvm_tools_record_gpu_fatal_fault(gpu->id, fault_entry->va_space, fault_entry, fault_entry->fatal_reason);

        // TODO: Bug 1873655: RM should log fatal MMU faults
        if (fault_entry->fatal_reason == UvmEventFatalReasonInvalidAddress) {
            pr_info("XID: CE fault on invalid address 0x%llx\n", fault_entry->fault_address);
        }
        else if (fault_entry->fatal_reason == UvmEventFatalReasonInvalidPermissions) {
            pr_info("XID: CE fault on address 0x%llx with invalid permissions %s\n",
                    fault_entry->fault_address,
                    uvm_fault_access_type_string(fault_entry->fault_access_type));
        }
    }

    // TODO: Bug 1873655: Define a protocol with RM in order to report CE MMU fatal faults
    if (status != NV_OK || fault_entry->is_fatal)
        schedule_kill_channel(gpu, va_space, user_channel);

exit_no_channel:
    uvm_va_space_up_read(va_space);

    return status;
}

void uvm_gpu_service_non_replayable_fault_buffer(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    NvU32 cached_faults;

    while ((cached_faults = fetch_non_replayable_fault_buffer_entries(gpu)) > 0) {
        NvU32 i;

        // Differently to replayable faults, we do not batch up and preprocess
        // non-replayable faults since getting multiple faults on the same
        // memory region is not very likely
        for (i = 0; i < cached_faults; ++i) {
            status = service_non_replayable_fault(gpu, &gpu->fault_buffer_info.non_replayable.fault_cache[i]);
            if (status != NV_OK)
                break;
        }
    }

    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing non-replayable faults on GPU: %s\n", gpu->name);
}
