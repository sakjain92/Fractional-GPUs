/*******************************************************************************
    Copyright (c) 2016 NVIDIA Corporation

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

#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_linux_ioctl.h"
#include "uvm8_global.h"
#include "uvm8_gpu.h"
#include "uvm8_lock.h"
#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"
#include "uvm8_tracker.h"
#include "uvm8_api.h"
#include "uvm8_channel.h"
#include "uvm8_push.h"
#include "uvm8_hal.h"
#include "uvm8_tools.h"

static NV_STATUS block_migrate_map_mapped_pages(uvm_va_block_t *va_block,
                                                uvm_va_block_retry_t *va_block_retry,
                                                uvm_va_block_context_t *va_block_context,
                                                uvm_va_block_region_t region,
                                                uvm_processor_id_t dest_id)
{
    uvm_prot_t prot;
    uvm_page_index_t page_index;
    NV_STATUS status = NV_OK;
    const uvm_page_mask_t *pages_mapped_on_destination = uvm_va_block_map_mask_get(va_block, dest_id);

    for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot)
        va_block_context->mask_by_prot[prot - 1].count = 0;

    // Only map those pages that are not already mapped on destination
    for_each_va_block_unset_page_in_region_mask(page_index, pages_mapped_on_destination, region) {
        prot = uvm_va_block_page_compute_highest_permission(va_block, dest_id, page_index);
        UVM_ASSERT(prot != UVM_PROT_NONE);

        if (va_block_context->mask_by_prot[prot - 1].count++ == 0)
            uvm_page_mask_zero(&va_block_context->mask_by_prot[prot - 1].page_mask);

        uvm_page_mask_set(&va_block_context->mask_by_prot[prot - 1].page_mask, page_index);
    }

    for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot) {
        if (va_block_context->mask_by_prot[prot - 1].count == 0)
            continue;

        // We pass UvmEventMapRemoteCauseInvalid since the destination processor
        // of a migration will never be mapped remotely
        status = uvm_va_block_map(va_block,
                                  va_block_context,
                                  dest_id,
                                  region,
                                  &va_block_context->mask_by_prot[prot - 1].page_mask,
                                  prot,
                                  UvmEventMapRemoteCauseInvalid,
                                  &va_block->tracker);
        if (status != NV_OK)
            break;

        // Whoever added the other mapping(s) should have already added
        // SetAccessedBy processors
    }

    return status;
}

static NV_STATUS block_migrate_map_unmapped_pages(uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_va_block_context_t *va_block_context,
                                                  uvm_va_block_region_t region,
                                                  uvm_processor_id_t dest_id)

{
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;

    // Save the mask of unmapped pages because it will change after the
    // first map operation
    uvm_page_mask_complement(&va_block_context->caller_page_mask, &va_block->maybe_mapped_pages);

    // Only map those pages that are not mapped anywhere else (likely due
    // to a first touch or a migration). We pass
    // UvmEventMapRemoteCauseInvalid since the destination processor of a
    // migration will never be mapped remotely.
    status = uvm_va_block_map(va_block,
                              va_block_context,
                              dest_id,
                              region,
                              &va_block_context->caller_page_mask,
                              UVM_PROT_READ_WRITE_ATOMIC,
                              UvmEventMapRemoteCauseInvalid,
                              &local_tracker);
    if (status != NV_OK)
        goto out;

    // Add mappings for AccessedBy processors
    //
    // No mappings within this call will operate on dest_id, so we don't
    // need to acquire the map operation above.
    status = uvm_va_block_add_mappings_after_migration(va_block,
                                                       va_block_context,
                                                       dest_id,
                                                       dest_id,
                                                       region,
                                                       &va_block_context->caller_page_mask,
                                                       UVM_PROT_READ_WRITE_ATOMIC,
                                                       NULL);

out:
    tracker_status = uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
    uvm_tracker_deinit(&local_tracker);
    return status == NV_OK ? tracker_status : status;
}

// Pages that are not mapped anywhere can be safely mapped with RWA permission.
// The rest of pages need to individually compute the maximum permission that
// does not require a revocation.
static NV_STATUS block_migrate_add_mappings(uvm_va_block_t *va_block,
                                            uvm_va_block_retry_t *va_block_retry,
                                            uvm_va_block_context_t *va_block_context,
                                            uvm_va_block_region_t region,
                                            uvm_processor_id_t dest_id)

{
    NV_STATUS status;

    status = block_migrate_map_unmapped_pages(va_block,
                                              va_block_retry,
                                              va_block_context,
                                              region,
                                              dest_id);
    if (status != NV_OK)
        return status;

    return block_migrate_map_mapped_pages(va_block,
                                          va_block_retry,
                                          va_block_context,
                                          region,
                                          dest_id);
}

NV_STATUS uvm_va_block_migrate_locked(uvm_va_block_t *va_block,
                                      uvm_va_block_retry_t *va_block_retry,
                                      uvm_va_block_context_t *va_block_context,
                                      uvm_va_block_region_t region,
                                      uvm_processor_id_t dest_id,
                                      bool do_mappings,
                                      uvm_tracker_t *out_tracker)
{
    NV_STATUS status, tracker_status = NV_OK;
    uvm_va_range_t *va_range = va_block->va_range;
    bool read_duplicate = va_range->read_duplication == UVM_READ_DUPLICATION_ENABLED &&
                          uvm_va_space_can_read_duplicate(va_range->va_space, NULL);

    uvm_assert_mutex_locked(&va_block->lock);

    uvm_page_mask_zero(&va_block_context->make_resident.pages_changed_residency);

    if (read_duplicate) {
        status = uvm_va_block_make_resident_read_duplicate(va_block,
                                                           va_block_retry,
                                                           va_block_context,
                                                           dest_id,
                                                           region,
                                                           NULL,
                                                           NULL,
                                                           UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);
    }
    else {
        status = uvm_va_block_make_resident(va_block,
                                            va_block_retry,
                                            va_block_context,
                                            dest_id,
                                            region,
                                            NULL,
                                            NULL,
                                            UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);
    }

    if (status == NV_OK && do_mappings) {
        // block_migrate_add_mappings will acquire the work from the above
        // make_resident call and update the VA block tracker.
        status = block_migrate_add_mappings(va_block, va_block_retry, va_block_context, region, dest_id);
    }

    if (out_tracker)
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &va_block->tracker);

    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS uvm_va_range_migrate(uvm_va_range_t *va_range,
                                      uvm_va_block_context_t *va_block_context,
                                      NvU64 base,
                                      NvU64 end,
                                      uvm_processor_id_t dest_id,
                                      bool do_mappings,
                                      uvm_tracker_t *out_tracker)
{
    uvm_va_block_t *va_block;
    size_t i;
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;
    uvm_va_block_region_t region;

    UVM_ASSERT(base >= va_range->node.start);
    UVM_ASSERT(end  <= va_range->node.end);
    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);
    uvm_assert_rwsem_locked(&va_range->va_space->lock);

    UVM_ASSERT(uvm_range_group_all_migratable(va_range->va_space, base, end));

    // Iterate over blocks, populating them if necessary
    for (i = uvm_va_range_block_index(va_range, base); i <= uvm_va_range_block_index(va_range, end); i++) {
        status = uvm_va_range_block_create(va_range, i, &va_block);
        if (status != NV_OK)
            return status;

        region = uvm_va_block_region_from_start_end(va_block, max(base, va_block->start), min(end, va_block->end));

        status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                uvm_va_block_migrate_locked(va_block,
                                            &va_block_retry,
                                            va_block_context,
                                            region,
                                            dest_id,
                                            do_mappings,
                                            out_tracker));
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS uvm_migrate_ranges(uvm_va_space_t *va_space,
                                    uvm_va_block_context_t *va_block_context,
                                    uvm_va_range_t *first_va_range,
                                    NvU64 base,
                                    NvU64 length,
                                    uvm_processor_id_t dest_id,
                                    bool do_mappings,
                                    uvm_tracker_t *out_tracker)
{
    uvm_va_range_t *va_range, *va_range_last;
    NvU64 end = base + length - 1;
    NV_STATUS status = NV_OK;
    bool skipped_migrate = false;

    UVM_ASSERT(first_va_range == uvm_va_space_iter_first(va_space, base, base));

    va_range_last = NULL;
    uvm_for_each_va_range_in_contig_from(va_range, va_space, first_va_range, end) {
        uvm_range_group_range_iter_t iter;
        va_range_last = va_range;

        // Only managed ranges can be migrated
        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
            status = NV_ERR_INVALID_ADDRESS;
            break;
        }

        // For UVM-Lite GPUs, the CUDA driver may suballocate a single va_range
        // into many range groups.  For this reason, we iterate over each va_range first
        // then through the range groups within.
        uvm_range_group_for_each_migratability_in(&iter,
                                                  va_space,
                                                  max(base, va_range->node.start),
                                                  min(end, va_range->node.end)) {
            // Skip non-migratable VA ranges
            if (!iter.migratable) {
                // Only return NV_WARN_MORE_PROCESSING_REQUIRED if the pages aren't
                // already resident at dest_id.
                if (va_range->preferred_location != dest_id)
                    skipped_migrate = true;
            }
            else if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, dest_id) &&
                     dest_id != va_range->preferred_location) {
                // Don't migrate to a non-faultable GPU that is in UVM-Lite mode,
                // unless it's the preferred location
                status = NV_ERR_INVALID_DEVICE;
                break;
            }
            else {
                status = uvm_va_range_migrate(va_range,
                                              va_block_context,
                                              iter.start,
                                              iter.end,
                                              dest_id,
                                              do_mappings,
                                              out_tracker);
                if (status != NV_OK)
                    break;
            }
        }
    }


    if (status != NV_OK)
        return status;

    // Check that we were able to iterate over the entire range without any gaps
    if (!va_range_last || va_range_last->node.end < end)
        return NV_ERR_INVALID_ADDRESS;

    if (skipped_migrate)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static bool is_single_block(uvm_va_range_t *first_va_range, NvU64 base, NvU64 length)
{
    NvU64 end = base + length - 1;

    if (end > first_va_range->node.end)
        return false;

    return uvm_va_range_block_index(first_va_range, base) == uvm_va_range_block_index(first_va_range, end);
}

static NV_STATUS uvm_migrate(uvm_va_space_t *va_space,
                             NvU64 base,
                             NvU64 length,
                             uvm_processor_id_t dest_id,
                             NvU32 migrate_flags,
                             uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_t *first_va_range = uvm_va_space_iter_first(va_space, base, base);
    uvm_va_block_context_t *va_block_context;
    bool do_mappings;
    bool do_two_passes;

    uvm_assert_mmap_sem_locked(&current->mm->mmap_sem);
    uvm_assert_rwsem_locked(&va_space->lock);

    if (!first_va_range || first_va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
        return NV_ERR_INVALID_ADDRESS;

    va_block_context = uvm_va_block_context_alloc();
    if (!va_block_context)
        return NV_ERR_NO_MEMORY;

    // We perform two passes (unless the migration only covers a single VA
    // block or UVM_MIGRATE_FLAG_SKIP_CPU_MAP is passed). This helps in the
    // following scenarios:
    //
    // - Migrations that add CPU mappings, since it is a synchronous operations
    // and delays the migration of the next VA blocks.
    // - Concurrent migrations. This is due to our current channel selection
    // logic that doesn't prevent false dependencies between independent
    // operations. For example, removal of mappings for outgoing transfers are
    // delayed by the mappings added by incoming transfers.
    // TODO: Bug 1764953: Re-evaluate the two-pass logic when channel selection
    // is overhauled.
    //
    // The two passes are as follows:
    //
    // 1- Transfer all VA blocks (do not add mappings)
    // 2- Go block by block reexecuting the transfer (in case someone moved it
    // since the first pass), and adding the mappings.
    do_mappings = (dest_id != UVM_CPU_ID) || !(migrate_flags & UVM_MIGRATE_FLAG_SKIP_CPU_MAP);
    do_two_passes = do_mappings && !is_single_block(first_va_range, base, length);

    if (do_two_passes) {
        status = uvm_migrate_ranges(va_space,
                                    va_block_context,
                                    first_va_range,
                                    base,
                                    length,
                                    dest_id,
                                    false,
                                    out_tracker);
    }

    if (status == NV_OK) {
        status = uvm_migrate_ranges(va_space,
                                    va_block_context,
                                    first_va_range,
                                    base,
                                    length,
                                    dest_id,
                                    do_mappings,
                                    out_tracker);
    }

    uvm_va_block_context_free(va_block_context);

    if (status != NV_OK)
        return status;

    return NV_OK;
}

static NV_STATUS uvm_push_async_user_sem_release(uvm_gpu_t *release_from_gpu,
                                                 uvm_va_range_semaphore_pool_t *sema_va_range,
                                                 NvU64 sema_user_addr,
                                                 NvU32 payload,
                                                 uvm_tracker_t *release_after_tracker)
{
    uvm_push_t push;
    NV_STATUS status;
    uvm_gpu_address_t sema_phys_addr;

    status = uvm_mem_map_gpu_phys(sema_va_range->mem, release_from_gpu);
    if (status != NV_OK)
        return status;

    sema_phys_addr = uvm_mem_gpu_address_physical(sema_va_range->mem, release_from_gpu,
            sema_user_addr - (NvU64)(uintptr_t)sema_va_range->mem->user.addr, 4);

    status = uvm_push_begin_acquire(release_from_gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    release_after_tracker,
                                    &push,
                                    "Pushing semaphore release (*0x%llx = %u)",
                                    sema_user_addr,
                                    (unsigned)payload);
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_push_begin_acquire() returned %d (%s)\n", status, nvstatusToString(status));
        return status;
    }

    release_from_gpu->host_hal->membar_sys(&push);
    release_from_gpu->ce_hal->memset_4(&push, sema_phys_addr, payload, 4);

    uvm_push_end(&push);

    uvm_mutex_lock(&sema_va_range->tracker_lock);
    status = uvm_tracker_add_push_safe(&sema_va_range->tracker, &push);
    uvm_tracker_remove_completed(&sema_va_range->tracker);
    uvm_mutex_unlock(&sema_va_range->tracker_lock);
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_tracker_add_push() returned %d (%s)\n", status, nvstatusToString(status));
        return status;
    }

    return NV_OK;
}

static void uvm_release_user_sem_from_cpu(uvm_mem_t *sema_mem, NvU64 user_addr, NvU32 payload)
{
    NvU64 sema_offset = user_addr - (NvU64)(uintptr_t)sema_mem->user.addr;
    NvU64 sema_page = uvm_div_pow2_64(sema_offset, sema_mem->chunk_size);
    NvU64 sema_page_offset = sema_offset & (sema_mem->chunk_size - 1);
    void *cpu_page_virt;
    void *cpu_addr;


    // Prevent processor speculation prior to accessing user-mapped memory to
    // avoid leaking information from side-channel attacks. Under speculation, a
    // valid VA range which does not contain this semaphore could be used by the
    // caller. It's unclear but likely that the user might be able to control
    // the data at that address. Auditing all potential ways that could happen
    // is difficult and error-prone, so to be on the safe side we'll just always
    // block speculation.
    //
    // TODO: Bug 2034846: Use common speculation_barrier implementation
    speculation_barrier();


    cpu_page_virt = kmap(sema_mem->sysmem.pages[sema_page]);
    cpu_addr = (char *)cpu_page_virt + sema_page_offset;
    UVM_WRITE_ONCE(*(NvU32 *)cpu_addr, payload);
    kunmap(sema_mem->sysmem.pages[sema_page]);
}

static NV_STATUS uvm_migrate_release_user_sem(UVM_MIGRATE_PARAMS *params, uvm_va_space_t *va_space,
                                              uvm_va_range_t *sema_va_range, uvm_gpu_t *dest_gpu,
                                              uvm_tracker_t *tracker_ptr, bool *wait_for_tracker_out)
{
    NV_STATUS status;
    uvm_mem_t *sema_mem = sema_va_range->semaphore_pool.mem;
    uvm_gpu_t *release_from = NULL;

    *wait_for_tracker_out = true;
    if (sema_va_range->semaphore_pool.owner)
        release_from = sema_va_range->semaphore_pool.owner;
    else
        release_from = dest_gpu;

    if (sema_va_range->semaphore_pool.owner == NULL && uvm_tracker_is_completed(tracker_ptr)) {
        // No GPU has the semaphore pool cached. Attempt eager release from CPU
        // if the tracker is already completed.
        *wait_for_tracker_out = false;
        uvm_release_user_sem_from_cpu(sema_mem, params->semaphoreAddress, params->semaphorePayload);
    }
    else {
        // Semaphore has to be released from a GPU because it is cached or we were unable
        // to release it from the CPU.
        if (!release_from) {
            // We did not do a CPU release, but the destination is CPU. This means the
            // tracker is not complete, and could be because accessed_by mappings are being
            // set up asynchronously, or because of the test-only flag
            // UVM_MIGRATE_FLAG_SKIP_CPU_MAP. However, this means there should be a registered
            // GPU since all CPU work is synchronous.
            release_from = uvm_processor_mask_find_first_gpu(&va_space->registered_gpus);
            UVM_ASSERT(release_from);
        }
        status = uvm_push_async_user_sem_release(release_from, &sema_va_range->semaphore_pool,
                                                 params->semaphoreAddress, params->semaphorePayload,
                                                 tracker_ptr);
        if (status != NV_OK) {
            UVM_ERR_PRINT("uvm_push_async_user_sem_release() returned %d (%s)\n",
                    status, nvstatusToString(status));
            return status;
        }
        else {
            *wait_for_tracker_out = false;
        }
    }

    return NV_OK;
}

NV_STATUS uvm_api_migrate(UVM_MIGRATE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    uvm_tracker_t *tracker_ptr = NULL;
    // NULL = CPU
    uvm_gpu_t *dest_gpu = NULL;
    uvm_va_range_t *sema_va_range = NULL;
    NV_STATUS status;
    NV_STATUS tracker_status = NV_OK;
    bool wait_for_tracker = true;

    if (uvm_api_range_invalid(params->base, params->length))
        return NV_ERR_INVALID_ADDRESS;

    if (params->flags & ~UVM_MIGRATE_FLAGS_ALL)
        return NV_ERR_INVALID_ARGUMENT;

    if ((params->flags & UVM_MIGRATE_FLAGS_TEST_ALL) && !uvm_enable_builtin_tests) {
        UVM_INFO_PRINT("Test flag set for UVM_MIGRATE. Did you mean to insmod with uvm_enable_builtin_tests=1?\n");
        return NV_ERR_INVALID_ARGUMENT;
    }

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_read(va_space);

    if (!(params->flags & UVM_MIGRATE_FLAG_ASYNC)) {
        if (params->semaphoreAddress != 0) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }
    }
    else {
        if (params->semaphoreAddress == 0) {
            if (params->semaphorePayload != 0) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }
        }
        else {
            sema_va_range = uvm_va_range_find(va_space, params->semaphoreAddress);
            if (!IS_ALIGNED(params->semaphoreAddress, sizeof(params->semaphorePayload)) ||
                    !sema_va_range || sema_va_range->type != UVM_VA_RANGE_TYPE_SEMAPHORE_POOL) {
                status = NV_ERR_INVALID_ADDRESS;
                goto done;
            }
        }
    }

    if (uvm_uuid_is_cpu(&params->destinationUuid)) {
        dest_gpu = NULL;
    }
    else {
        if (params->flags & UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)
            dest_gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->destinationUuid);
        else
            dest_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destinationUuid);

        if (!dest_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }

        if (!uvm_gpu_can_address(dest_gpu, params->base + params->length - 1)) {
            status = NV_ERR_OUT_OF_RANGE;
            goto done;
        }
    }

    // If we're synchronous or if we need to release a semaphore, use a tracker.
    if (!(params->flags & UVM_MIGRATE_FLAG_ASYNC) || params->semaphoreAddress)
        tracker_ptr = &tracker;

    status = uvm_migrate(va_space, params->base, params->length, (dest_gpu ? dest_gpu->id : UVM_CPU_ID),
                         params->flags, tracker_ptr);

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_sem_out_of_order(&current->mm->mmap_sem);

    if (tracker_ptr) {
        if (params->semaphoreAddress && status == NV_OK) {
            // Need to do a semaphore release.
            status = uvm_migrate_release_user_sem(params, va_space, sema_va_range,
                                                  dest_gpu, tracker_ptr, &wait_for_tracker);
        }

        if (wait_for_tracker) {
            // There was an error or we are sync. Even if there was an error, we
            // need to wait for work already dispatched to complete. Waiting on
            // a tracker requires the VA space lock to prevent GPUs being unregistered
            // during the wait.
            tracker_status = uvm_tracker_wait_deinit(tracker_ptr);
        }
        else {
            uvm_tracker_deinit(tracker_ptr);
        }
    }

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
    if (wait_for_tracker)
        uvm_tools_flush_events();

    // Only clobber status if we didn't hit an earlier error
    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_api_migrate_range_group(UVM_MIGRATE_RANGE_GROUP_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_range_group_t *range_group;
    uvm_range_group_range_t *rgr;
    uvm_processor_id_t dest_id;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU32 migrate_flags = 0;
    uvm_gpu_t *gpu = NULL;

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_read(va_space);

    if (uvm_uuid_is_cpu(&params->destinationUuid)) {
        dest_id = UVM_CPU_ID;
    }
    else {
        gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destinationUuid);
        if (!gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }

        dest_id = gpu->id;
    }

    range_group = radix_tree_lookup(&va_space->range_groups, params->rangeGroupId);
    if (!range_group) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto done;
    }

    // Migrate all VA ranges in the range group. uvm_migrate is used because it performs all
    // VA range validity checks.
    list_for_each_entry(rgr, &range_group->ranges, range_group_list_node) {
        NvU64 start = rgr->node.start;
        NvU64 end = rgr->node.end;

        if (gpu && !uvm_gpu_can_address(gpu, end))
            status = NV_ERR_OUT_OF_RANGE;
        else
            status = uvm_migrate(va_space, start, end - start + 1, dest_id, migrate_flags, &local_tracker);

        if (status != NV_OK)
            goto done;
    }

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_sem_out_of_order(&current->mm->mmap_sem);

    tracker_status = uvm_tracker_wait_deinit(&local_tracker);
    uvm_va_space_up_read(va_space);

    // This API is synchronous, so wait for migrations to finish
    uvm_tools_flush_events();

    return status == NV_OK? tracker_status : status;
}
