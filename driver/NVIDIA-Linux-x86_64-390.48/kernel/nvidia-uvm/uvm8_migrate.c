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

static void uvm_block_iter_deinitialization(uvm_block_iter_t *iter)
{
    struct page *page;
    NvU32 i;

    if (iter->cpu_block) {
        
        if (iter->cpu_block->cpu.pages) {

            // Release all pinned pages if any
            for (i = 0; i < PAGES_PER_UVM_VA_BLOCK; i++) {
                page = iter->cpu_block->cpu.pages[i];
                if (page) {
                    put_page(page);
                    iter->cpu_block->cpu.pages[i]= NULL;
                }
            }

            kfree(iter->cpu_block->cpu.pages);
            iter->cpu_block->cpu.pages = NULL;
        }

        kfree(iter->cpu_block);
        iter->cpu_block = NULL;
    }
}

// Iterate over all managed contiguous va_blocks till "length" is covered
// Length is in terms of color mem size
static NV_STATUS uvm_block_iter_initialization(uvm_va_space_t *va_space,
                                                NvU64 start,
                                                uvm_processor_id_t id,
                                                uvm_block_iter_t *iter)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_t *first_va_range;
    size_t block_index, range_end_block_index;
    struct page **page_array;

    uvm_assert_rwsem_locked(&va_space->lock);

    iter->start = start;
    iter->va_range = NULL;
    iter->cpu_block = NULL;

    first_va_range = uvm_va_space_iter_first(va_space, start, start);

    // If block not exists in uvm but lies on CPU, maybe it is backed by linux
    // pages
    // TODO: It can be that the start block is not within any range but
    // the subsequent blocks might be. We need to handle this behaviour
    if (!first_va_range & (id == UVM_CPU_ID)) {

        iter->cpu_block = kmalloc(sizeof(uvm_va_block_t), GFP_KERNEL);
        if (!iter->cpu_block) {
            status = NV_ERR_NO_MEMORY;
            goto err;
        }

        iter->cpu_block->is_linux_backed = true;

        page_array = kzalloc(sizeof(struct page *) * PAGES_PER_UVM_VA_BLOCK, GFP_KERNEL);
        if (!page_array) {
            status = NV_ERR_NO_MEMORY;
            goto err;
        }

        iter->cpu_block->cpu.pages = page_array;

        iter->next_block_index = start / UVM_VA_BLOCK_SIZE;
        iter->range_end_block_index = (size_t)-1;
        return NV_OK;
    }

    if (!first_va_range || first_va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto err;
    }

    block_index = uvm_va_range_block_index(first_va_range, max(start, first_va_range->node.start));
    range_end_block_index = uvm_va_range_block_index(first_va_range, first_va_range->node.end);

    iter->va_range = first_va_range;
    iter->next_block_index = block_index;
    iter->range_end_block_index = range_end_block_index;    

    return NV_OK;

err:
    uvm_block_iter_deinitialization(iter);
    return status;
}

static NV_STATUS uvm_block_iter_next_block(uvm_block_iter_t *iter,
                                uvm_va_block_t **out_block)
{
    NV_STATUS status = NV_OK;

    if (iter->cpu_block) {
        iter->cpu_block->start = iter->next_block_index * UVM_VA_BLOCK_SIZE;
        iter->cpu_block->end = iter->cpu_block->start + UVM_VA_BLOCK_SIZE - 1;

        *out_block = iter->cpu_block;

        goto out;
    }

    // Reached end of current range?
    if (iter->next_block_index > iter->range_end_block_index) {
        uvm_va_range_t *va_range;
	    NvU64 end = (NvU64)-1;

    	va_range = uvm_va_space_iter_next(iter->va_range, end);
        if (!va_range || va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
            return NV_ERR_INVALID_ADDRESS;
        }

        iter->next_block_index = uvm_va_range_block_index(va_range, va_range->node.start);
	    iter->range_end_block_index = uvm_va_range_block_index(va_range, va_range->node.end);
        iter->va_range = va_range;
    }

    status = uvm_va_range_block_create(iter->va_range, iter->next_block_index, out_block);
    if (status != NV_OK) {
        return status;
    }

out:
    iter->next_block_index++;
    return status;
}

void uvm_va_colored_block_region_init(NvU64 start, NvU64 length, NvU32 color,
        uvm_va_block_colored_region_t *region)
{
    region->start = start & ~(PAGE_SIZE - 1);
    region->page_offset = start & (PAGE_SIZE - 1);
    
    region->color = color;

    region->length = length;
    
    uvm_page_mask_zero(&region->page_mask);

    region->last_block_start = 0;
}

// Update a block color range for a va block
// Since this function depends on physical address, block should be locked
// before calling this function.
NV_STATUS uvm_update_va_colored_block_region(uvm_va_block_t *va_block,
                                               uvm_processor_id_t id,
                                               uvm_va_block_colored_region_t *region)
{
    NvU64 left = region->length;
    NvU64 start, end;
    uvm_page_index_t first, outer, last;
    uvm_gpu_phys_address_t phy_addr;
    uvm_gpu_t *gpu;
    NvU64 page_start, page_end, page_size, page_offset;
    NvU32 page_color;

    // No update needed if current block same as last block
    if (region->last_block_start && region->last_block_start == va_block->start)
        return NV_OK;

    uvm_page_mask_zero(&region->page_mask);

    page_offset = region->page_offset;

    // No coloring on CPU side
    if (id == UVM_CPU_ID) {
        int ret, i;
        struct page *page;

        start = max(va_block->start, region->start) + page_offset;
        end = min(va_block->end, start + region->length - 1);

        first = uvm_va_block_cpu_page_index(va_block, start);
        outer = uvm_va_block_cpu_page_index(va_block, end) + 1;

        uvm_page_mask_fill(&region->page_mask, first, outer);

        // Only linux backed pages need to be locked
        if (va_block->is_linux_backed) {
            // Release all previously pinned pages
            for (i = 0; i < PAGES_PER_UVM_VA_BLOCK; i++) {
                page = va_block->cpu.pages[i];
                if (page) {
                    put_page(page);
                    va_block->cpu.pages[i]= NULL;
                }
            }

            // Try pinning pages
            ret = NV_GET_USER_PAGES(va_block->start + first * PAGE_SIZE,
                    outer - first, true, false, &va_block->cpu.pages[first], NULL);
            if (ret < 0) {
                return NV_ERR_INVALID_ADDRESS;
            }
        }
        goto done;
    }

    // Only blocks on CPU can be linux backed
    UVM_ASSERT(!va_block->is_linux_backed);

    start = max(va_block->start, region->start);
    first = uvm_va_block_cpu_page_index(va_block, start);
    outer = uvm_va_block_cpu_page_index(va_block, va_block->end) + 1;
    last = first;

    gpu = uvm_gpu_get(id);

    // If physically contiguous, get the start phy address and then increment
    // Else find physical address for all the pages seperately
    if (uvm_block_is_phys_contig(va_block, id)) {
        
        uvm_page_index_t i;

        phy_addr = uvm_va_block_gpu_phys_page_address(va_block, first, gpu);

        for (i = first; i < outer && left != 0; i++, phy_addr.address += PAGE_SIZE) {
    
            page_color = gpu->arch_hal->phys_addr_to_transfer_color(gpu, phy_addr.address);
            if (page_color != region->color) {
                continue;
            }

	        last = i;

            uvm_page_mask_set(&region->page_mask, i);
            page_start = max(start, va_block->start + PAGE_SIZE * i) + page_offset;
            page_end = va_block->start + PAGE_SIZE * (i + 1) - 1;
            page_size = min(left, page_end - page_start + 1);
            left -= page_size;
            page_offset = 0;
        }
    } else {
        
        uvm_page_index_t i;

        for (i = first; i < outer && left != 0; i++) {
   
            phy_addr = uvm_va_block_gpu_phys_page_address(va_block, i, gpu);

            page_color = gpu->arch_hal->phys_addr_to_transfer_color(gpu, phy_addr.address);
            if (page_color != region->color) {
                continue;
            }
	
	        last = i;

            uvm_page_mask_set(&region->page_mask, i);
            page_start = max(start, va_block->start + PAGE_SIZE * i) + page_offset;
            page_end = va_block->start + PAGE_SIZE * (i + 1) - 1;
            page_size = min(left, page_end - page_start + 1);
            left -= page_size;
            page_offset = 0;
        }

    }

    outer = last + 1;

done:
    region->region.first = first;
    region->region.outer = outer;

    region->last_block_start = va_block->start;

    return NV_OK;
}

static NV_STATUS uvm_va_block_memcpy_colored_locked(uvm_va_block_t *src_va_block,
                                                    uvm_va_block_t *dest_va_block,
                                                    uvm_processor_id_t src_id,
                                                    uvm_processor_id_t dest_id,
                                                    uvm_va_block_colored_region_t *src_region,
                                                    uvm_va_block_colored_region_t *dest_region,
                                                    NvU64 *copied,
                                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS tracker_status;

    status = uvm_update_va_colored_block_region(src_va_block, src_id, src_region);
    if (status != NV_OK)
        goto out;

    status =uvm_update_va_colored_block_region(dest_va_block, dest_id, dest_region);
    if (status != NV_OK)
        goto out;

    status = block_copy_colored_pages_between(src_va_block,
                                                dest_va_block,
                                                src_id,
                                                dest_id,
                                                src_region,
                                                dest_region,
                                                copied,
                                                &local_tracker);

out:
    if (out_tracker) {
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    } else {
        // Add everything from the local tracker to the block's tracker.
        tracker_status = uvm_tracker_add_tracker_safe(&dest_va_block->tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    }

    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS uvm_va_block_memset_colored_locked(uvm_va_block_t *va_block,
                                                    uvm_processor_id_t id,
                                                    uvm_va_block_colored_region_t *region,
                                                    NvU8 value,
                                                    NvU64 *covered,
                                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS tracker_status;

    status = uvm_update_va_colored_block_region(va_block, id, region);
    if (status != NV_OK)
        goto out;

    status = block_memset_colored_pages(va_block,
                                        id,
                                        region,
                                        value,
                                        covered,
                                        &local_tracker);

out:
    if (out_tracker) {
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    } else {
        // Add everything from the local tracker to the block's tracker.
        tracker_status = uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    }

    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS uvm_memcpy_colored_blocks(uvm_va_space_t *va_space,
                                           NvU64 srcBase,
                                           NvU64 destBase,
                                           NvU64 length,
                                           NvU32 color,
                                           uvm_processor_id_t src_id,
                                           uvm_processor_id_t dest_id,
                                           uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_block_iter_t src_block_iter, dest_block_iter;
    uvm_va_block_t *src_va_block, *dest_va_block;
    NvU64 left = length;
    NvU64 copied;
    uvm_va_block_colored_region_t src_region, dest_region;

    status = uvm_block_iter_initialization(va_space, srcBase, src_id, &src_block_iter);
    if (status != NV_OK)
        return status;

    status = uvm_block_iter_initialization(va_space, destBase, dest_id, &dest_block_iter);
    if (status != NV_OK) {
        uvm_block_iter_deinitialization(&src_block_iter);
        return status;
    }

    uvm_va_colored_block_region_init(srcBase, length, color, &src_region);
    uvm_va_colored_block_region_init(destBase, length, color, &dest_region);


    while (left != 0) {

        // If current block has been done with, fetch the next block
        if (uvm_page_mask_empty(&src_region.page_mask)) {
            status = uvm_block_iter_next_block(&src_block_iter, &src_va_block);
            if (status != NV_OK)
                goto out;
        }

        if (uvm_page_mask_empty(&dest_region.page_mask)) {
            status = uvm_block_iter_next_block(&dest_block_iter, &dest_va_block);
            if (status != NV_OK)
                goto out;
        }

        status = UVM_VA_GENERIC_MULTI_BLOCK_LOCK_RETRY(src_va_block, dest_va_block,
                NULL, NULL,
                uvm_va_block_memcpy_colored_locked(src_va_block,
                    dest_va_block,
                    src_id,
                    dest_id,
                    &src_region,
                    &dest_region,
                    &copied,
                    out_tracker));

        if (status != NV_OK)
            goto out;

        left -= copied;
    }

out:
    uvm_block_iter_deinitialization(&src_block_iter);
    uvm_block_iter_deinitialization(&dest_block_iter);

    return status;
}

static NV_STATUS uvm_memset_colored_blocks(uvm_va_space_t *va_space,
                                            NvU64 base,
                                            NvU64 length,
                                            NvU8 value,
                                            NvU32 color,
                                            uvm_processor_id_t id,
                                            uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    uvm_block_iter_t block_iter;
    uvm_va_block_t *va_block;
    NvU64 left = length;
    NvU64 covered;
    uvm_va_block_colored_region_t region;

    status = uvm_block_iter_initialization(va_space, base, id, &block_iter);
    if (status != NV_OK)
        goto out;

    uvm_va_colored_block_region_init(base, length, color, &region);

    while (left != 0) {

        // If current block has been done with, fetch the next block
        if (uvm_page_mask_empty(&region.page_mask)) {
            status = uvm_block_iter_next_block(&block_iter, &va_block);
            if (status != NV_OK)
                goto out;
        }

        status = UVM_VA_BLOCK_LOCK_RETRY(va_block, NULL,
                uvm_va_block_memset_colored_locked(va_block,
                                                    id,
                                                    &region,
                                                    value,
                                                    &covered,
                                                    out_tracker));

        if (status != NV_OK)
            goto out;

        left -= covered;
    }

out:
    uvm_block_iter_deinitialization(&block_iter);

    return status;
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

static NV_STATUS uvm_memcpy_colored(uvm_va_space_t *va_space,
                                    NvU64 srcBase,
                                    NvU64 destBase,
                                    NvU64 length,
                                    NvU32 color,
                                    uvm_processor_id_t src_id,
                                    uvm_processor_id_t dest_id,
                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mmap_sem_locked(&current->mm->mmap_sem);
    uvm_assert_rwsem_locked(&va_space->lock);

    // TODO: Populate pages and map them
    status = uvm_memcpy_colored_blocks(va_space,
                                        srcBase,
                                        destBase,
                                        length,
                                        color,
                                        src_id,
                                        dest_id,
                                        out_tracker);

    if (status != NV_OK)
        return status;

    return NV_OK;
}

static NV_STATUS uvm_memset_colored(uvm_va_space_t *va_space,
                                    NvU64 base,
                                    NvU64 length,
                                    NvU8 value,
                                    NvU32 color,
                                    uvm_processor_id_t id,
                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mmap_sem_locked(&current->mm->mmap_sem);
    uvm_assert_rwsem_locked(&va_space->lock);

    // TODO: Populate pages and map them
    status = uvm_memset_colored_blocks(va_space,
                                       base,
                                       length,
                                       value,
                                       color,
                                       id,
                                       out_tracker);

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

NV_STATUS uvm_api_memcpy_colored(UVM_MEMCPY_COLORED_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    // NULL = CPU
    uvm_gpu_t *src_gpu = NULL;
    uvm_gpu_t *dest_gpu = NULL;
    NvU32 color = 0;
    NV_STATUS status;
    NV_STATUS tracker_status = NV_OK;

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_read(va_space);

    if (uvm_uuid_is_cpu(&params->srcUuid)) {
        src_gpu = NULL;
    }
    else {
        src_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->srcUuid);

        if (!src_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }
    }

    if (uvm_uuid_is_cpu(&params->destUuid)) {
        dest_gpu = NULL;
    }
    else {
        dest_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destUuid);

        if (!dest_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }
    }

    // Either atmost one src/dest lie on CPU or both lie on same GPU
    // Invalid configuration: Both lie on CPU or different GPUs
    if ((!src_gpu && !dest_gpu) || (src_gpu && dest_gpu && src_gpu->id != dest_gpu->id)) {
    	status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    // Atleast one is a GPU. Get it's color. If both on same GPU, then also only a single color exists.
    if (src_gpu) {
        status = uvm_pmm_get_current_process_color(&src_gpu->pmm, &color);
        if (status != NV_OK)
            goto done;
    } else {
        status = uvm_pmm_get_current_process_color(&dest_gpu->pmm, &color);
        if (status != NV_OK)
            goto done;
    }

    if (params->length == 0) {
        status = NV_OK;
        goto done;
    }

    // This is synchronous call, so using a tracker.
    status = uvm_memcpy_colored(va_space, params->srcBase, params->destBase, 
                                params->length, color, (src_gpu ? src_gpu->id : UVM_CPU_ID),
                                (dest_gpu ? dest_gpu->id : UVM_CPU_ID),
                                &tracker);

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_sem_out_of_order(&current->mm->mmap_sem);

    // There was an error or we are sync. Even if there was an error, we
    // need to wait for work already dispatched to complete. Waiting on
    // a tracker requires the VA space lock to prevent GPUs being unregistered
    // during the wait.
    tracker_status = uvm_tracker_wait_deinit(&tracker);

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
    uvm_tools_flush_events();

    // Only clobber status if we didn't hit an earlier error
    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_api_memset_colored(UVM_MEMSET_COLORED_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    uvm_gpu_t *gpu = NULL;
    NvU32 color = 0;
    NV_STATUS status;
    NV_STATUS tracker_status = NV_OK;

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_sem(&current->mm->mmap_sem);
    uvm_va_space_down_read(va_space);

    // Only GPU are supported (CPU can use memset() in userspace)
    if (uvm_uuid_is_cpu(&params->uuid)) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }
        
    gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    status = uvm_pmm_get_current_process_color(&gpu->pmm, &color);
    if (status != NV_OK)
        goto done;

    if (params->length == 0) {
        status = NV_OK;
        goto done;
    }

    // This is synchronous call, so using a tracker.
    status = uvm_memset_colored(va_space, params->base, params->length, params->value,
                                color, gpu->id, &tracker);

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_sem_out_of_order(&current->mm->mmap_sem);

    // There was an error or we are sync. Even if there was an error, we
    // need to wait for work already dispatched to complete. Waiting on
    // a tracker requires the VA space lock to prevent GPUs being unregistered
    // during the wait.
    tracker_status = uvm_tracker_wait_deinit(&tracker);

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
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

#if 0  // TODO: Find out why this cause non-realtime delays
       // (I suspect it has something to do with this function using background kernel thread)
    // This API is synchronous, so wait for migrations to finish
    uvm_tools_flush_events();
#endif

    return status == NV_OK? tracker_status : status;
}
