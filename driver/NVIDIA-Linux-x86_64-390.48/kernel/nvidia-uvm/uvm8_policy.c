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

#include "uvm_ioctl.h"
#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"
#include "uvm8_api.h"
#include "uvm8_tracker.h"
#include "uvm8_gpu.h"

static bool preferred_location_is_va_range_split_needed(uvm_va_range_t *va_range, void *data)
{
    uvm_processor_id_t processor_id;
    UVM_ASSERT(data);
    processor_id = *(uvm_processor_id_t*)data;
    return (processor_id != va_range->preferred_location);
}

static NV_STATUS preferred_location_set(uvm_va_space_t *va_space,
                                        NvU64 base,
                                        NvU64 length,
                                        uvm_processor_id_t preferred_location,
                                        uvm_va_range_t **first_va_range_to_migrate)
{
    uvm_va_range_t *va_range, *va_range_last;
    NvU64 last_address = base + length - 1;
    NV_STATUS status = NV_OK;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (first_va_range_to_migrate)
        *first_va_range_to_migrate = NULL;

    status = uvm_va_space_split_span_as_needed(va_space,
                                               base,
                                               last_address + 1,
                                               preferred_location_is_va_range_split_needed,
                                               &preferred_location);
    if (status != NV_OK)
        return status;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        bool all_migratable = uvm_range_group_all_migratable(va_space,
                                                             max(base, va_range->node.start),
                                                             min(last_address, va_range->node.end));
        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(va_range->preferred_location == preferred_location);

        // If any part of the VA range group is non-migratable, check that the preferred location is
        // not a fault capable GPU
        if (!all_migratable &&
            preferred_location != UVM_MAX_PROCESSORS &&
            preferred_location != UVM_CPU_ID &&
            uvm_processor_mask_test(&va_space->faultable_processors, preferred_location))
            return NV_ERR_INVALID_DEVICE;

        status = uvm_va_range_set_preferred_location(va_range, preferred_location);
        if (status != NV_OK)
            return status;

        // Return the first VA range that needs to be migrated so the caller
        // function doesn't need to traverse the tree again
        if (first_va_range_to_migrate && !(*first_va_range_to_migrate) &&
            preferred_location != UVM_MAX_PROCESSORS && !all_migratable)
            *first_va_range_to_migrate = va_range;

    }

    // Check that we were able to iterate over the entire range without any gaps
    if (!va_range_last || va_range_last->node.end < last_address)
        return NV_ERR_INVALID_ADDRESS;

    return NV_OK;
}

NV_STATUS uvm_api_set_preferred_location(UVM_SET_PREFERRED_LOCATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *first_va_range_to_migrate = NULL;
    uvm_processor_id_t preferred_location_id;
    NV_STATUS status = NV_OK;
    bool has_va_space_write_lock;
    const NvU64 start = params->requestedBase;
    const NvU64 end = start + params->length - 1;

    UVM_ASSERT(va_space);

    // Check address and length alignment
    if (uvm_api_range_invalid(start, params->length))
        return NV_ERR_INVALID_ADDRESS;

    uvm_va_space_down_write(va_space);
    has_va_space_write_lock = true;

    // If the CPU is the preferred location, we don't have to find the associated uvm_gpu_t
    if (uvm_uuid_is_cpu(&params->preferredLocation)) {
        preferred_location_id = UVM_CPU_ID;
    }
    else {
        // Translate preferredLocation into a live GPU ID, and check that this
        // GPU can address the virtual address range
        uvm_gpu_t *gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->preferredLocation);

        if (!gpu)
            status = NV_ERR_INVALID_DEVICE;
        else if (!uvm_gpu_can_address(gpu, end))
            status = NV_ERR_OUT_OF_RANGE;

        if (status != NV_OK)
            goto done;

        preferred_location_id = gpu->id;
    }

    status = preferred_location_set(va_space, start, params->length, preferred_location_id,
                                    &first_va_range_to_migrate);
    if (status != NV_OK)
        goto done;

    // TODO: Bug 1765613: Unmap the preferred location's processor from any
    //       pages in this region which are not resident on the preferred
    //       location.

    // No VA range to migrate, early exit
    if (!first_va_range_to_migrate)
        goto done;

    uvm_va_space_downgrade_write(va_space);
    has_va_space_write_lock = false;

    // No need to check for holes in the VA ranges span here, this was checked by preferred_location_set
    for (va_range = first_va_range_to_migrate; va_range;
        va_range = uvm_va_space_iter_next(va_range, end)) {
        uvm_range_group_range_iter_t iter;
        NvU64 cur_start = max(start, va_range->node.start);
        NvU64 cur_end = min(end, va_range->node.end);

        uvm_range_group_for_each_migratability_in(&iter, va_space, cur_start, cur_end) {
            if (!iter.migratable) {
                status = uvm_range_group_va_range_migrate(va_range, iter.start, iter.end);
                if (status != NV_OK)
                    goto done;
            }
        }
    }

done:
    if (has_va_space_write_lock)
        uvm_va_space_up_write(va_space);
    else
        uvm_va_space_up_read(va_space);
    return status;
}

NV_STATUS uvm_api_unset_preferred_location(UVM_UNSET_PREFERRED_LOCATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NV_STATUS status = NV_OK;

    UVM_ASSERT(va_space);

    // Check address and length alignment
    if (uvm_api_range_invalid(params->requestedBase, params->length))
         return NV_ERR_INVALID_ADDRESS;

    uvm_va_space_down_write(va_space);
    status = preferred_location_set(va_space, params->requestedBase, params->length, UVM_MAX_PROCESSORS, NULL);
    uvm_va_space_up_write(va_space);
    return status;
}

static NV_STATUS va_block_set_accessed_by_locked(uvm_va_block_t *va_block,
                                                 uvm_va_block_context_t *va_block_context,
                                                 uvm_processor_id_t processor_id,
                                                 uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    NV_STATUS tracker_status;

    uvm_assert_mutex_locked(&va_block->lock);

    status = uvm_va_block_add_mappings(va_block,
                                       va_block_context,
                                       processor_id,
                                       uvm_va_block_region_from_block(va_block));

    tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &va_block->tracker);

    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_va_block_set_accessed_by(uvm_va_block_t *va_block, uvm_processor_id_t processor_id)
{
    NV_STATUS status;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    uvm_va_block_context_t *va_block_context;

    // Read duplication takes precedence over SetAccesedBy. Do not add mappings
    // if read duplication is enabled
    if (va_block->va_range->read_duplication == UVM_READ_DUPLICATION_ENABLED &&
        uvm_va_space_can_read_duplicate(va_block->va_range->va_space, NULL))
        return NV_OK;

    va_block_context = uvm_va_block_context_alloc();
    if (!va_block_context)
        return NV_ERR_NO_MEMORY;

    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, NULL,
            va_block_set_accessed_by_locked(va_block, va_block_context, processor_id, &local_tracker));

    uvm_va_block_context_free(va_block_context);

    // TODO: Bug 1767224: Combine all accessed_by operations into single tracker
    if (status == NV_OK)
        status = uvm_tracker_wait(&local_tracker);

    uvm_tracker_deinit(&local_tracker);
    return status;
}

typedef struct
{
    uvm_processor_id_t processor_id;
    bool set_bit;
} accessed_by_split_params_t;

static bool accessed_by_is_va_range_split_needed(uvm_va_range_t *va_range, void *data)
{
    accessed_by_split_params_t *params = (accessed_by_split_params_t*)data;
    UVM_ASSERT(params);

    return (uvm_processor_mask_test(&va_range->accessed_by, params->processor_id) != params->set_bit);
}

static NV_STATUS accessed_by_set(uvm_va_space_t *va_space,
                                 NvU64 base,
                                 NvU64 length,
                                 const NvProcessorUuid *processor_uuid,
                                 bool set_bit)
{
    uvm_processor_id_t processor_id = UVM_MAX_PROCESSORS;
    uvm_va_range_t *va_range, *va_range_last;
    const NvU64 last_address = base + length - 1;
    accessed_by_split_params_t split_params;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(va_space);

    // Check address and length alignment
    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ADDRESS;

    // We need mmap_sem if we might create CPU mappings
    if (uvm_uuid_is_cpu(processor_uuid)) {
        processor_id = UVM_CPU_ID;
        uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    }

    uvm_va_space_down_write(va_space);

    if (processor_id != UVM_CPU_ID) {
        // Translate processor_uuid into a live GPU ID, and check that this GPU
        // can address the virtual address range
        uvm_gpu_t *gpu = uvm_va_space_get_gpu_by_uuid(va_space, processor_uuid);
        if (!gpu)
            status = NV_ERR_INVALID_DEVICE;
        else if (!uvm_gpu_can_address(gpu, last_address))
            status = NV_ERR_OUT_OF_RANGE;

        if (status != NV_OK)
            goto done;

        processor_id = gpu->id;
    }

    split_params.processor_id = processor_id;
    split_params.set_bit = set_bit;
    status = uvm_va_space_split_span_as_needed(va_space,
                                               base,
                                               last_address + 1,
                                               accessed_by_is_va_range_split_needed,
                                               &split_params);
    if (status != NV_OK)
        goto done;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(uvm_processor_mask_test(&va_range->accessed_by, processor_id) == set_bit);

        if (set_bit) {
            status = uvm_va_range_set_accessed_by(va_range, processor_id);
            if (status != NV_OK)
                goto done;
        }
        else {
            uvm_va_range_unset_accessed_by(va_range, processor_id);
        }
    }

    // Check that we were able to iterate over the entire range without any gaps
    if (!va_range_last || va_range_last->node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

done:
    uvm_va_space_up_write(va_space);

    if (processor_id == UVM_CPU_ID)
        uvm_up_read_mmap_sem(&current->mm->mmap_sem);

    return status;
}

NV_STATUS uvm_api_set_accessed_by(UVM_SET_ACCESSED_BY_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return accessed_by_set(va_space, params->requestedBase, params->length, &params->accessedByUuid, true);
}

NV_STATUS uvm_api_unset_accessed_by(UVM_UNSET_ACCESSED_BY_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return accessed_by_set(va_space, params->requestedBase, params->length, &params->accessedByUuid, false);
}

NV_STATUS va_block_set_read_duplication_locked(uvm_va_block_t *va_block,
                                               uvm_va_block_retry_t *va_block_retry,
                                               uvm_va_block_context_t *va_block_context)
{
    NV_STATUS status;
    uvm_processor_id_t src_id;

    uvm_assert_mutex_locked(&va_block->lock);

    for_each_id_in_mask(src_id, &va_block->resident) {
        uvm_page_mask_t *resident_mask = uvm_va_block_resident_mask_get(va_block, src_id);

        // Calling uvm_va_block_make_resident_read_duplicate will break all
        // SetAccessedBy and remote mappings
        status = uvm_va_block_make_resident_read_duplicate(va_block,
                                                           va_block_retry,
                                                           va_block_context,
                                                           src_id,
                                                           uvm_va_block_region_from_block(va_block),
                                                           resident_mask,
                                                           NULL,
                                                           UVM_MAKE_RESIDENT_CAUSE_API_HINT);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

NV_STATUS uvm_va_block_set_read_duplication(uvm_va_block_t *va_block,
                                            uvm_va_block_context_t *va_block_context)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;

    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                                     va_block_set_read_duplication_locked(va_block,
                                                                          &va_block_retry,
                                                                          va_block_context));

    return status;
}

static NV_STATUS va_block_unset_read_duplication_locked(uvm_va_block_t *va_block,
                                                        uvm_va_block_retry_t *va_block_retry,
                                                        uvm_va_block_context_t *va_block_context,
                                                        uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    uvm_processor_id_t processor_id;
    uvm_va_range_t *va_range = va_block->va_range;
    uvm_va_block_region_t block_region = uvm_va_block_region_from_block(va_block);
    uvm_page_mask_t *break_read_duplication_pages = &va_block_context->caller_page_mask;

    uvm_assert_mutex_locked(&va_block->lock);

    // 1- Iterate over all processors with resident copies to avoid migrations
    // and invalidate the rest of copies

    // If preferred_location is set and has resident copies, give it preference
    if (va_range->preferred_location != UVM_MAX_PROCESSORS &&
        uvm_processor_mask_test(&va_block->resident, va_range->preferred_location)) {
        uvm_page_mask_t *resident_mask = uvm_va_block_resident_mask_get(va_block, va_range->preferred_location);
        bool is_mask_empty = !uvm_page_mask_and(break_read_duplication_pages,
                                                &va_block->read_duplicated_pages,
                                                resident_mask);

        if (!is_mask_empty) {
            // make_resident breaks read duplication
            status = uvm_va_block_make_resident(va_block,
                                                va_block_retry,
                                                va_block_context,
                                                va_range->preferred_location,
                                                block_region,
                                                break_read_duplication_pages,
                                                NULL,
                                                UVM_MAKE_RESIDENT_CAUSE_API_HINT);
            if (status != NV_OK)
                return status;
        }
    }

    // Then iterate over the rest of processors
    for_each_id_in_mask(processor_id, &va_block->resident) {
        uvm_page_mask_t *resident_mask;
        bool is_mask_empty;

        if (processor_id == va_range->preferred_location)
            continue;

        resident_mask = uvm_va_block_resident_mask_get(va_block, processor_id);
        is_mask_empty = !uvm_page_mask_and(break_read_duplication_pages,
                                           &va_block->read_duplicated_pages,
                                           resident_mask);
        if (is_mask_empty)
            continue;

        // make_resident breaks read duplication
        status = uvm_va_block_make_resident(va_block,
                                            va_block_retry,
                                            va_block_context,
                                            processor_id,
                                            block_region,
                                            break_read_duplication_pages,
                                            NULL,
                                            UVM_MAKE_RESIDENT_CAUSE_API_HINT);
        if (status != NV_OK)
            return status;
    }

    // 2- Re-establish SetAccessedBy mappings
    for_each_id_in_mask(processor_id, &va_block->va_range->accessed_by) {
        status = va_block_set_accessed_by_locked(va_block,
                                                 va_block_context,
                                                 processor_id,
                                                 out_tracker);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

NV_STATUS uvm_va_block_unset_read_duplication(uvm_va_block_t *va_block,
                                              uvm_va_block_context_t *va_block_context)
{
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    // Restore all SetAccessedBy mappings
    status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                                     va_block_unset_read_duplication_locked(va_block,
                                                                            &va_block_retry,
                                                                            va_block_context,
                                                                            &local_tracker));
    if (status == NV_OK)
        status = uvm_tracker_wait(&local_tracker);

    uvm_tracker_deinit(&local_tracker);

    return status;
}

static bool read_duplication_is_va_range_split_needed(uvm_va_range_t *va_range, void *data)
{
    uvm_read_duplication_policy_t new_policy;
    UVM_ASSERT(data);
    new_policy = *(uvm_read_duplication_policy_t *)data;
    return va_range->read_duplication != new_policy;
}

static NV_STATUS read_duplication_set(uvm_va_space_t *va_space,
                                      NvU64 base,
                                      NvU64 length,
                                      bool enable)
{
    uvm_va_range_t *va_range, *va_range_last;
    NvU64 last_address = base + length - 1;
    NV_STATUS status = NV_OK;

    // Note that we never set the policy back to UNSET
    uvm_read_duplication_policy_t new_policy = enable ? UVM_READ_DUPLICATION_ENABLED : UVM_READ_DUPLICATION_DISABLED;

    UVM_ASSERT(va_space);

    // Check address and length alignment
    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ADDRESS;

    // We need mmap_sem as we may create CPU mappings
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_write(va_space);

    status = uvm_va_space_split_span_as_needed(va_space,
                                               base,
                                               last_address + 1,
                                               read_duplication_is_va_range_split_needed,
                                               &new_policy);
    if (status != NV_OK)
        goto done;

    va_range_last = NULL;
    uvm_for_each_managed_va_range_in_contig(va_range, va_space, base, last_address) {
        va_range_last = va_range;

        // If we didn't split the ends, check that they match
        if (va_range->node.start < base || va_range->node.end > last_address)
            UVM_ASSERT(va_range->read_duplication == new_policy);

        // If the va_space cannot currently read duplicate, only change the user state.
        // All memory should already have read duplication unset.
        if (uvm_va_space_can_read_duplicate(va_space, NULL)) {

            // Handle SetAccessedBy mappings
            if (new_policy == UVM_READ_DUPLICATION_ENABLED) {
                status = uvm_va_range_set_read_duplication(va_range);
                if (status != NV_OK)
                    goto done;
            }
            else {
                uvm_va_range_unset_read_duplication(va_range);
            }
        }

        va_range->read_duplication = new_policy;
    }

    // Check that we were able to iterate over the entire range without any gaps
    if (!va_range_last || va_range_last->node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

done:
    uvm_va_space_up_write(va_space);
    uvm_up_read_mmap_sem(&current->mm->mmap_sem);
    return status;
}

NV_STATUS uvm_api_enable_read_duplication(UVM_ENABLE_READ_DUPLICATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return read_duplication_set(va_space, params->requestedBase, params->length, true);
}

NV_STATUS uvm_api_disable_read_duplication(UVM_DISABLE_READ_DUPLICATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return read_duplication_set(va_space, params->requestedBase, params->length, false);
}

static NV_STATUS system_wide_atomics_set(uvm_va_space_t *va_space, const NvProcessorUuid *gpu_uuid, bool enable)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    bool already_enabled;

    uvm_va_space_down_write(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    if (!uvm_processor_mask_test(&va_space->faultable_processors, gpu->id)) {
        status = NV_ERR_NOT_SUPPORTED;
        goto done;
    }

    already_enabled = uvm_processor_mask_test(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    if (enable && !already_enabled) {
        uvm_va_range_t *va_range;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        uvm_va_block_context_t *va_block_context = uvm_va_space_block_context(va_space);
        NV_STATUS tracker_status;

        // Revoke atomic mappings from the calling GPU
        uvm_for_each_va_range(va_range, va_space) {
            uvm_va_block_t *va_block;

            if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
                continue;

            for_each_va_block_in_va_range(va_range, va_block) {
                uvm_page_mask_t *non_resident_pages = &va_block_context->caller_page_mask;

                uvm_mutex_lock(&va_block->lock);

                if (!uvm_processor_mask_test(&va_block->mapped, gpu->id)) {
                    uvm_mutex_unlock(&va_block->lock);
                    continue;
                }

                uvm_page_mask_complement(non_resident_pages, &va_block->gpus[uvm_gpu_index(gpu->id)]->resident);

                status = uvm_va_block_revoke_prot(va_block,
                                                  va_block_context,
                                                  gpu->id,
                                                  uvm_va_block_region_from_block(va_block),
                                                  non_resident_pages,
                                                  UVM_PROT_READ_WRITE_ATOMIC,
                                                  &va_block->tracker);

                tracker_status = uvm_tracker_add_tracker_safe(&local_tracker, &va_block->tracker);

                uvm_mutex_unlock(&va_block->lock);

                if (status == NV_OK)
                    status = tracker_status;

                if (status != NV_OK) {
                    uvm_tracker_deinit(&local_tracker);
                    goto done;
                }
            }
        }
        status = uvm_tracker_wait_deinit(&local_tracker);

        uvm_processor_mask_set(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    }
    else if (!enable && already_enabled) {
        // TODO: Bug 1767229: Promote write mappings to atomic
        uvm_processor_mask_clear(&va_space->system_wide_atomics_enabled_processors, gpu->id);
    }

done:
    uvm_va_space_up_write(va_space);
    return status;
}

NV_STATUS uvm_api_enable_system_wide_atomics(UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return system_wide_atomics_set(va_space, &params->gpu_uuid, true);
}

NV_STATUS uvm_api_disable_system_wide_atomics(UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return system_wide_atomics_set(va_space, &params->gpu_uuid, false);
}
