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
#include "uvm8_forward_decl.h"
#include "uvm8_lock.h"
#include "uvm8_mmu.h"
#include "uvm8_api.h"
#include "uvm8_global.h"
#include "uvm8_gpu.h"
#include "uvm8_push.h"
#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_tracker.h"
#include "uvm8_hal.h"
#include "uvm8_hal_types.h"
#include "uvm8_map_external.h"
#include "uvm8_pte_batch.h"
#include "uvm8_tlb_batch.h"
#include "nv_uvm_interface.h"

#include "uvm8_pushbuffer.h"

// Assume almost all of the push space can be used for PTEs leaving 1K of margin.
#define MAX_COPY_SIZE_PER_PUSH ((size_t)(UVM_MAX_PUSH_SIZE - 1024))


static NV_STATUS get_rm_ptes(uvm_va_range_t *va_range,
                             uvm_gpu_t *gpu,
                             NvU64 map_offset,
                             NvU64 map_size,
                             UvmGpuExternalMappingInfo *mapping_info)
{
    uvm_gpu_va_space_t *gpu_va_space = uvm_gpu_va_space_get(va_range->va_space, gpu);
    uvm_ext_gpu_map_t *ext_gpu_map;
    NV_STATUS status;

    // Both queries operate on the same UvmGpuExternalMappingInfo type
    if (va_range->type == UVM_VA_RANGE_TYPE_CHANNEL) {
        status = uvm_rm_locked_call(nvUvmInterfaceGetChannelResourcePtes(gpu_va_space->duped_gpu_va_space,
                                                                         va_range->channel.rm_descriptor,
                                                                         map_offset,
                                                                         map_size,
                                                                         mapping_info));
    }
    else {
        ext_gpu_map = uvm_va_range_ext_gpu_map(va_range, gpu);
        UVM_ASSERT(ext_gpu_map);
        status = uvm_rm_locked_call(nvUvmInterfaceGetExternalAllocPtes(gpu_va_space->duped_gpu_va_space,
                                                                       ext_gpu_map->dup_handle,
                                                                       map_offset,
                                                                       map_size,
                                                                       mapping_info));
    }

    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to get %s mappings for VA range [0x%llx, 0x%llx], offset 0x%llx, size 0x%llx: %s\n",
                      va_range->type == UVM_VA_RANGE_TYPE_CHANNEL ? "channel" : "external",
                      va_range->node.start, va_range->node.end, map_offset, map_size, nvstatusToString(status));
    }

    return status;
}

typedef struct
{
    // The VA range the buffer is for
    uvm_va_range_t *va_range;

    // The GPU that's mapping the VA range
    uvm_gpu_t *gpu;

    // Mapping info used for querying PTEs from RM
    UvmGpuExternalMappingInfo mapping_info;

    // Size of the buffer
    size_t buffer_size;

    // Page size in bytes
    NvU32 page_size;

    // Size of a single PTE in bytes
    NvU32 pte_size;

    // Max PTE offset covered by the VA range.
    //
    // Notably the mapping might not start at offset 0 and max PTE offset can be
    // larger than number of PTEs covering the VA range.
    size_t max_pte_offset;

    // Number of PTEs currently in the buffer
    size_t num_ptes;

    // PTE offset at which the currently buffered PTEs start.
    size_t pte_offset;
} uvm_pte_buffer_t;

// Max PTE buffer size is the size of the buffer used for querying PTEs from RM.
// It has to be big enough to amortize the cost of calling into RM, but small
// enough to fit in CPU caches as it's written and read multiple times on the
// CPU before it ends up in the pushbuffer.
// 96K seems to be a sweet spot at least on a Xeon W5580 system. This could use
// some benchmarking on more systems though.
#define MAX_PTE_BUFFER_SIZE ((size_t)96 * 1024)

static NV_STATUS uvm_pte_buffer_init(uvm_va_range_t *va_range,
                                     uvm_gpu_t *gpu,
                                     uvm_map_rm_params_t *map_rm_params,
                                     NvU32 page_size,
                                     uvm_pte_buffer_t *pte_buffer)
{
    uvm_gpu_va_space_t *gpu_va_space = uvm_gpu_va_space_get(va_range->va_space, gpu);
    uvm_page_tree_t *tree = &gpu_va_space->page_tables;
    size_t num_all_ptes;

    memset(pte_buffer, 0, sizeof(*pte_buffer));

    pte_buffer->va_range = va_range;
    pte_buffer->gpu = gpu;
    pte_buffer->mapping_info.cachingType = map_rm_params->caching_type;
    pte_buffer->mapping_info.mappingType = map_rm_params->mapping_type;
    pte_buffer->page_size = page_size;
    pte_buffer->pte_size = uvm_mmu_pte_size(tree, page_size);
    num_all_ptes = uvm_div_pow2_64(uvm_va_range_size(va_range), page_size);
    pte_buffer->max_pte_offset = uvm_div_pow2_64(map_rm_params->map_offset, page_size) + num_all_ptes;
    pte_buffer->buffer_size = min(MAX_PTE_BUFFER_SIZE, num_all_ptes * pte_buffer->pte_size);

    pte_buffer->mapping_info.pteBuffer = uvm_kvmalloc(pte_buffer->buffer_size);
    if (!pte_buffer->mapping_info.pteBuffer)
        return NV_ERR_NO_MEMORY;

    return NV_OK;
}

static void uvm_pte_buffer_deinit(uvm_pte_buffer_t *pte_buffer)
{
    uvm_kvfree(pte_buffer->mapping_info.pteBuffer);
}

// Get the PTEs for mapping the [map_offset, map_offset + map_size) VA range.
static NV_STATUS uvm_pte_buffer_get(uvm_pte_buffer_t *pte_buffer,
                                    NvU64 map_offset,
                                    NvU64 map_size,
                                    NvU64 **ptes_out)
{
    NV_STATUS status;
    size_t pte_offset;
    size_t num_ptes;
    size_t ptes_left;

    UVM_ASSERT(IS_ALIGNED(map_offset, pte_buffer->page_size));
    UVM_ASSERT(IS_ALIGNED(map_size, pte_buffer->page_size));

    pte_offset = uvm_div_pow2_64(map_offset, pte_buffer->page_size);
    num_ptes = uvm_div_pow2_64(map_size, pte_buffer->page_size);

    UVM_ASSERT(num_ptes <= pte_buffer->buffer_size / pte_buffer->pte_size);

    // If the requested range is already fully cached, just calculate its
    // offset within the the buffer and return.
    if (pte_buffer->pte_offset <= pte_offset && pte_buffer->pte_offset + pte_buffer->num_ptes >= pte_offset + num_ptes) {
        pte_offset -= pte_buffer->pte_offset;
        *ptes_out = (NvU64 *)((char *)pte_buffer->mapping_info.pteBuffer + pte_offset * pte_buffer->pte_size);
        return NV_OK;
    }

    // Otherwise get max possible PTEs from RM starting at the requested offset.
    pte_buffer->pte_offset = pte_offset;
    ptes_left = pte_buffer->max_pte_offset - pte_offset;
    pte_buffer->num_ptes = min(pte_buffer->buffer_size / pte_buffer->pte_size, ptes_left);

    UVM_ASSERT_MSG(pte_buffer->num_ptes >= num_ptes, "buffer num ptes %zu < num ptes %zu\n",
            pte_buffer->num_ptes, num_ptes);

    // TODO: Bug 1735291: RM can determine the buffer size from the map_size
    //       parameter.
    pte_buffer->mapping_info.pteBufferSize = pte_buffer->num_ptes * pte_buffer->pte_size;

    status = get_rm_ptes(pte_buffer->va_range,
                         pte_buffer->gpu,
                         map_offset,
                         pte_buffer->num_ptes * pte_buffer->page_size,
                         &pte_buffer->mapping_info);
    if (status != NV_OK)
        return status;

    *ptes_out = pte_buffer->mapping_info.pteBuffer;

    return NV_OK;
}

// Copies the input ptes buffer to the given physical address, with an optional
// TLB invalidate. The copy acquires the input tracker then updates it.
static NV_STATUS copy_ptes(uvm_page_tree_t *tree,
                           NvU64 page_size,
                           uvm_gpu_phys_address_t pte_addr,
                           NvU64 *ptes,
                           NvU32 num_ptes,
                           bool last_mapping,
                           NvU64 range_start,
                           NvU64 range_end,
                           uvm_tracker_t *tracker)
{
    uvm_push_t push;
    NV_STATUS status;
    NvU32 pte_size = uvm_mmu_pte_size(tree, page_size);

    UVM_ASSERT(pte_size * num_ptes <= MAX_COPY_SIZE_PER_PUSH);

    // CPU_TO_GPU because the data being transferred is within the pushbuffer
    status = uvm_push_begin_acquire(tree->gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                    tracker,
                                    &push,
                                    "Writing %u bytes of PTEs to {%s, 0x%llx}",
                                    pte_size * num_ptes,
                                    uvm_aperture_string(pte_addr.aperture),
                                    pte_addr.address);
    if (status != NV_OK)
        return status;

    uvm_pte_batch_single_write_ptes(&push, pte_addr, ptes, pte_size, num_ptes);

    if (last_mapping) {
        // Do a TLB invalidate if this is the last mapping in the VA range
        // Membar: This is a permissions upgrade, so no post-invalidate membar
        //         is needed.
        uvm_tlb_batch_single_invalidate(tree, &push, range_start, range_end - range_start + 1,
                page_size, UVM_MEMBAR_NONE);
    }
    else {
        // For pushes prior to the last one, the PTE batch write has
        // already pushed a membar that's enough to order the PTE writes
        // with the TLB invalidate in the last push and that's all
        // that's needed.
        // If a failure happens before the push for the last mapping, it is
        // still ok as what will follow is more CE writes to unmap the PTEs and
        // those will get ordered by the membar from the PTE batch.
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
    }

    uvm_push_end(&push);

    // The push acquired the tracker so it's ok to just overwrite it with
    // the entry tracking the push.
    uvm_tracker_overwrite_with_push(tracker, &push);

    return NV_OK;
}

// Map all of pt_range, which is contained with the va_range and begins at
// virtual address start. The PTE values are queried from RM and the pushed
// writes are added to the input tracker.
//
// If the mapped range ends on va_range->node.end, a TLB invalidate for upgrade
// is also issued.
NV_STATUS map_rm_pt_range(uvm_va_range_t *va_range,
                                 uvm_page_tree_t *tree,
                                 uvm_page_table_range_t *pt_range,
                                 uvm_pte_buffer_t *pte_buffer,
                                 NvU64 start,
                                 NvU64 map_offset,
                                 uvm_tracker_t *tracker)
{
    uvm_gpu_phys_address_t pte_addr;
    NvU64 page_size = pt_range->page_size;
    NvU32 pte_size = uvm_mmu_pte_size(tree, page_size);
    NvU64 addr, end;
    size_t max_ptes, ptes_left, num_ptes;
    NvU64 map_size;
    bool last_mapping;
    NV_STATUS status = NV_OK;

    end = start + uvm_page_table_range_size(pt_range) - 1;

    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_EXTERNAL || va_range->type == UVM_VA_RANGE_TYPE_CHANNEL);
    UVM_ASSERT(start >= va_range->node.start);
    UVM_ASSERT(end <= va_range->node.end);
    UVM_ASSERT(page_size & tree->hal->page_sizes());
    UVM_ASSERT(IS_ALIGNED(start, page_size));
    UVM_ASSERT(IS_ALIGNED(map_offset, page_size));

    pte_addr = uvm_page_table_range_entry_address(tree, pt_range, 0);
    max_ptes = min((size_t)(uvm_mmu_pde_coverage(tree, page_size) / page_size), MAX_COPY_SIZE_PER_PUSH / pte_size);
    max_ptes = min(max_ptes, pte_buffer->buffer_size / pte_size);

    addr = start;
    ptes_left = (size_t)uvm_div_pow2_64(uvm_page_table_range_size(pt_range), page_size);
    while (addr < end) {
        NvU64 *pte_bits;

        num_ptes = min(max_ptes, ptes_left);
        map_size = num_ptes * page_size;
        UVM_ASSERT(addr + map_size <= end + 1);

        status = uvm_pte_buffer_get(pte_buffer, map_offset, map_size, &pte_bits);
        if (status != NV_OK)
            return status;

        last_mapping = (addr + map_size - 1 == va_range->node.end);

        // These copies are technically independent, except for the last one
        // which issues the TLB invalidate and thus must wait for all others.
        // However, since each copy will saturate the bus anyway we force them
        // to serialize to avoid bus contention.
        status = copy_ptes(tree,
                           page_size,
                           pte_addr,
                           pte_bits,
                           num_ptes,
                           last_mapping,
                           va_range->node.start,
                           va_range->node.end,
                           tracker);
        if (status != NV_OK)
            return status;

        ptes_left -= num_ptes;
        pte_addr.address += num_ptes * pte_size;
        addr += map_size;
        map_offset += map_size;
    }

    return NV_OK;
}

// Determine the appropriate membar for downgrades on a VA range with type
// UVM_VA_RANGE_TYPE_EXTERNAL or UVM_VA_RANGE_TYPE_CHANNEL.
static uvm_membar_t va_range_downgrade_membar(uvm_va_range_t *va_range, uvm_gpu_t *mapped_gpu)
{
    uvm_ext_gpu_map_t *ext_gpu_map;

    if (va_range->type == UVM_VA_RANGE_TYPE_CHANNEL) {
        if (va_range->channel.aperture == UVM_APERTURE_VID)
            return UVM_MEMBAR_GPU;
        return UVM_MEMBAR_SYS;
    }

    ext_gpu_map = uvm_va_range_ext_gpu_map(va_range, mapped_gpu);
    if (ext_gpu_map->is_sysmem || mapped_gpu != ext_gpu_map->owning_gpu)
        return UVM_MEMBAR_SYS;
    return UVM_MEMBAR_GPU;
}

NV_STATUS uvm_va_range_map_rm_allocation(uvm_va_range_t *va_range,
                                         uvm_gpu_t *mapping_gpu,
                                         UvmGpuMemoryInfo *mem_info,
                                         uvm_map_rm_params_t *map_rm_params)
{
    uvm_gpu_va_space_t *gpu_va_space = uvm_gpu_va_space_get(va_range->va_space, mapping_gpu);
    uvm_page_tree_t *page_tree;
    uvm_pte_buffer_t pte_buffer;
    uvm_page_table_range_vec_t *pt_range_vec;
    uvm_page_table_range_t *pt_range;
    NvU64 addr, size;
    NvU64 map_offset = map_rm_params->map_offset;
    size_t i;
    NV_STATUS status;

    // map_rm_params contains the tracker we need to update with the work we
    // we push. However, that's only an output tracker for the caller to use to
    // wait for the whole operation at the end. Since we need to serialize our
    // pushes but not with the caller's tracker, we track our work separately
    // then add our work at the end.
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    UVM_ASSERT(gpu_va_space);
    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_EXTERNAL || va_range->type == UVM_VA_RANGE_TYPE_CHANNEL);
    UVM_ASSERT(uvm_gpu_can_address(mapping_gpu, va_range->node.end));
    UVM_ASSERT(IS_ALIGNED(mem_info->size, mem_info->pageSize));
    UVM_ASSERT(map_rm_params->tracker);

    page_tree = &gpu_va_space->page_tables;

    // Verify that the GPU VA space supports this page size
    if ((mem_info->pageSize & page_tree->hal->page_sizes()) == 0)
        return NV_ERR_INVALID_ADDRESS;

    // Consolidate input checks for API-level callers
    if (!uvm_va_range_is_aligned(va_range, mem_info->pageSize))
        return NV_ERR_INVALID_ADDRESS;

    if (!IS_ALIGNED(map_offset, mem_info->pageSize) ||
        map_offset + uvm_va_range_size(va_range) > mem_info->size)
        return NV_ERR_INVALID_OFFSET;

    status = uvm_pte_buffer_init(va_range, mapping_gpu, map_rm_params, mem_info->pageSize, &pte_buffer);
    if (status != NV_OK)
        return status;

    if (va_range->type == UVM_VA_RANGE_TYPE_EXTERNAL)
        pt_range_vec = &uvm_va_range_ext_gpu_map(va_range, mapping_gpu)->pt_range_vec;
    else
        pt_range_vec = &va_range->channel.pt_range_vec;

    // Allocate all page tables for this VA range.
    //
    // TODO: Bug 1766649: Benchmark to see if we get any performance improvement
    //       from parallelizing page range allocation with writing PTEs for
    //       earlier ranges.
    status = uvm_page_table_range_vec_init(page_tree,
                                           va_range->node.start,
                                           uvm_va_range_size(va_range),
                                           mem_info->pageSize,
                                           pt_range_vec);
    if (status != NV_OK)
        goto out;

    addr = va_range->node.start;
    for (i = 0; i < pt_range_vec->range_count; i++) {
        pt_range = &pt_range_vec->ranges[i];

        status = map_rm_pt_range(va_range, page_tree, pt_range, &pte_buffer, addr, map_offset, &local_tracker);
        if (status != NV_OK)
            goto out;

        size = uvm_page_table_range_size(pt_range);
        addr += size;
        map_offset += size;
    }

    status = uvm_tracker_add_tracker(map_rm_params->tracker, &local_tracker);
    if (status != NV_OK)
        goto out;

out:
    if (status != NV_OK) {
        // We could have any number of mappings in flight to these page tables,
        // so wait for everything before we clear and free them.
        if (uvm_tracker_wait(&local_tracker) != NV_OK) {
            // System-fatal error. Just leak.
            return status;
        }

        if (pt_range_vec->ranges) {
            uvm_page_table_range_vec_clear_ptes(pt_range_vec, va_range_downgrade_membar(va_range, mapping_gpu));
            uvm_page_table_range_vec_deinit(pt_range_vec);
        }
    }

    uvm_pte_buffer_deinit(&pte_buffer);
    uvm_tracker_deinit(&local_tracker);
    return status;
}

static bool uvm_api_mapping_type_invalid(UvmGpuMappingType map_type)
{
    switch (map_type) {
        case UvmGpuMappingTypeDefault:
        case UvmGpuMappingTypeReadWriteAtomic:
        case UvmGpuMappingTypeReadWrite:
        case UvmGpuMappingTypeReadOnly:
            return false;
    }
    return true;
}

static bool uvm_api_caching_type_invalid(UvmGpuCachingType cache_type)
{
    switch (cache_type) {
        case UvmGpuCachingTypeDefault:
        case UvmGpuCachingTypeForceUncached:
        case UvmGpuCachingTypeForceCached:
            return false;
    }
    return true;
}

static NV_STATUS set_ext_gpu_map_location(uvm_ext_gpu_map_t *ext_gpu_map,
                                          uvm_va_space_t *va_space,
                                          uvm_gpu_t *mapping_gpu,
                                          UvmGpuMemoryInfo *mem_info)
{
    uvm_gpu_t *owning_gpu;

    // This is a local or peer allocation, so the owning GPU must have been
    // registered.
    owning_gpu = uvm_va_space_get_gpu_by_uuid(va_space, &mem_info->uuid);
    if (!owning_gpu)
        return NV_ERR_INVALID_DEVICE;

    // Even if the allocation is in sysmem then it still matters which GPU owns
    // it, because our dup is not enough to keep the owning GPU around and that
    // exposes a bug in RM where the memory can outlast the GPU and then cause
    // crashes when it's eventually freed.
    // TODO: Bug 1811006: Bug tracking the RM issue, its fix might change the
    // semantics of sysmem allocations.
    if (mem_info->sysmem) {
        ext_gpu_map->owning_gpu = owning_gpu;
        ext_gpu_map->is_sysmem = true;
        return NV_OK;
    }

    if (owning_gpu != mapping_gpu) {
        // TODO: Bug 1757136: In SLI, the returned UUID may be different but a
        //       local mapping must be used. We need to query SLI groups to know
        //       that.
        if (!uvm_va_space_peer_enabled(va_space, mapping_gpu, owning_gpu))
            return NV_ERR_INVALID_DEVICE;
    }

    ext_gpu_map->owning_gpu = owning_gpu;
    ext_gpu_map->is_sysmem = false;
    return NV_OK;
}

static NV_STATUS uvm_map_external_allocation_on_gpu(uvm_va_range_t *va_range,
                                                    uvm_gpu_t *mapping_gpu,
                                                    uvm_rm_user_object_t *user_rm_mem,
                                                    uvm_map_rm_params_t *map_rm_params)
{
    uvm_va_space_t *va_space = va_range->va_space;
    uvm_ext_gpu_map_t *ext_gpu_map;
    UvmGpuMemoryInfo mem_info;
    NV_STATUS status;

    uvm_assert_rwsem_locked_read(&va_space->lock);

    // Check if the GPU already has a mapping here
    if (uvm_va_range_ext_gpu_map(va_range, mapping_gpu) != NULL)
        return NV_ERR_UVM_ADDRESS_IN_USE;

    // Check if the GPU can access the VA
    if (!uvm_gpu_can_address(mapping_gpu, va_range->node.end))
        return NV_ERR_OUT_OF_RANGE;

    ext_gpu_map = uvm_kvmalloc_zero(sizeof(*ext_gpu_map));
    if (!ext_gpu_map)
        return NV_ERR_NO_MEMORY;

    // Insert the ext_gpu_map into the VA range immediately since some of the
    // below calls require it to be there.
    va_range->external.gpu_maps[uvm_gpu_index(mapping_gpu->id)] = ext_gpu_map;
    uvm_processor_mask_set(&va_range->external.mapped_gpus, mapping_gpu->id);
    ext_gpu_map->gpu = mapping_gpu;

    // Error paths after this point may call uvm_va_range_ext_gpu_map, so do a
    // sanity check now to make sure it doesn't trigger any asserts.
    UVM_ASSERT(uvm_va_range_ext_gpu_map(va_range, mapping_gpu) == ext_gpu_map);

    // Dup the memory. This verifies the input handles, takes a ref count on the
    // physical allocation so it can't go away under us, and returns us the
    // allocation info.
    status = uvm_rm_locked_call(nvUvmInterfaceDupMemory(mapping_gpu->rm_address_space,
                                                        user_rm_mem->user_client,
                                                        user_rm_mem->user_object,
                                                        &ext_gpu_map->dup_handle,
                                                        &mem_info));
    if (status != NV_OK) {
        UVM_DBG_PRINT("Failed to dup memory handle {0x%x, 0x%x}: %s, GPU: %s\n",
                      user_rm_mem->user_client, user_rm_mem->user_object,
                      nvstatusToString(status), mapping_gpu->name);
        goto error;
    }

    status = set_ext_gpu_map_location(ext_gpu_map, va_space, mapping_gpu, &mem_info);
    if (status != NV_OK)
        goto error;

    status = uvm_va_range_map_rm_allocation(va_range, mapping_gpu, &mem_info, map_rm_params);
    if (status != NV_OK)
        goto error;

    return NV_OK;

error:
    uvm_ext_gpu_map_destroy(va_range, mapping_gpu, NULL);
    return status;
}

static NV_STATUS find_create_va_range(uvm_va_space_t *va_space, NvU64 base, NvU64 length, uvm_va_range_t **out_va_range)
{
    uvm_va_range_t *va_range;
    NV_STATUS status;

    *out_va_range = NULL;

    // Check if the VA range already exists. It's ok to create a new mapping in
    // an existing VA range, as long as the new mapping matches the bounds of
    // the VA range exactly.
    va_range = uvm_va_range_find(va_space, base);
    if (!va_range) {
        // No range exists at base, but we still might collide in [base,
        // base+length). Creating the VA range will fail in that case.
        status = uvm_va_range_create_external(va_space, base, length, &va_range);
        if (status != NV_OK) {
            UVM_DBG_PRINT_RL("Failed to create external VA range [0x%llx, 0x%llx)\n", base, base + length);
            return status;
        }
    }
    else if (va_range->type != UVM_VA_RANGE_TYPE_EXTERNAL   ||
             va_range->node.start != base                   ||
             va_range->node.end != base + length - 1) {
        return NV_ERR_UVM_ADDRESS_IN_USE;
    }

    *out_va_range = va_range;
    return NV_OK;
}

// Actual implementation of UvmMapExternalAllocation
static NV_STATUS uvm_map_external_allocation(uvm_va_space_t *va_space, UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *params)
{
    uvm_va_range_t *va_range = NULL;
    uvm_gpu_t *mapping_gpu;
    uvm_processor_mask_t mapped_gpus;
    NV_STATUS status = NV_OK;
    size_t i;
    uvm_map_rm_params_t map_rm_params;
    uvm_rm_user_object_t user_rm_mem =
    {
        .rm_control_fd = params->rmCtrlFd,
        .user_client   = params->hClient,
        .user_object   = params->hMemory
    };
    uvm_tracker_t tracker = UVM_TRACKER_INIT();

    // Before we know the page size used by the allocation, we can only enforce
    // 4K alignment as that's the minimum page size used for GPU allocations.
    // Later uvm_map_external_allocation_on_gpu() will enforce alignment to the
    // page size used by the allocation.
    if (uvm_api_range_invalid_4k(params->base, params->length))
        return NV_ERR_INVALID_ADDRESS;

    if (params->gpuAttributesCount == 0 || params->gpuAttributesCount > UVM_MAX_GPUS)
        return NV_ERR_INVALID_ARGUMENT;

    // We must take the VA space lock in write mode first, even if the VA range
    // already exists. See the below comment on the lock downgrade.
    uvm_va_space_down_write(va_space);

    // Insert the VA range (if not already present) with the VA space lock in
    // write mode.
    status = find_create_va_range(va_space, params->base, params->length, &va_range);
    if (status != NV_OK) {
        uvm_va_space_up_write(va_space);
        return status;
    }

    // The subsequent mappings will need to call into RM, which means we must
    // downgrade the VA space lock to read mode but keep the
    // serialize_writers_lock held. Although we're in read mode no other threads
    // could modify this VA range: other threads which call map and will attach
    // to the same VA range will first take the VA space lock in write mode
    // above, and threads which call unmap operate entirely with the lock in
    // write mode.
    uvm_va_space_downgrade_write_rm(va_space);

    uvm_processor_mask_zero(&mapped_gpus);
    for (i = 0; i < params->gpuAttributesCount; i++) {
        if (uvm_api_mapping_type_invalid(params->perGpuAttributes[i].gpuMappingType) ||
            uvm_api_caching_type_invalid(params->perGpuAttributes[i].gpuCachingType)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto error;
        }

        mapping_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->perGpuAttributes[i].gpuUuid);
        if (!mapping_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto error;
        }

        // Use a tracker to get as much parallelization as possible among GPUs,
        // so one GPU can have its PTE writes in flight while we're working on
        // the next one.
        map_rm_params.map_offset = params->offset;
        map_rm_params.mapping_type = params->perGpuAttributes[i].gpuMappingType;
        map_rm_params.caching_type = params->perGpuAttributes[i].gpuCachingType;
        map_rm_params.tracker = &tracker;
        status = uvm_map_external_allocation_on_gpu(va_range, mapping_gpu, &user_rm_mem, &map_rm_params);
        if (status != NV_OK)
            goto error;

        uvm_processor_mask_set(&mapped_gpus, mapping_gpu->id);
    }

    // Wait for outstanding page table operations to finish across all GPUs. We
    // just need to hold the VA space lock to prevent the GPUs on which we're
    // waiting from getting unregistered underneath us.
    status = uvm_tracker_wait_deinit(&tracker);

    uvm_va_space_up_read_rm(va_space);
    return status;

error:
    // We still have to wait for page table writes to finish, since the teardown
    // could free them.
    (void)uvm_tracker_wait_deinit(&tracker);

    // Tear down only those mappings we created during this call
    for_each_gpu_in_mask(mapping_gpu, &mapped_gpus)
        uvm_ext_gpu_map_destroy(va_range, mapping_gpu, NULL);

    if (uvm_processor_mask_empty(&va_range->external.mapped_gpus)) {
        // If there are no remaining mappings we need to destroy the VA range.
        // We need the VA space lock in write mode to do that, which means we
        // have to take a ref count on the VA range so the memory doesn't get
        // freed from under us.
        uvm_va_range_retain(va_range);
        uvm_va_space_up_read_rm(va_space);

        uvm_va_space_down_write(va_space);

        // There are three possible states for the VA range:
        // 1) !va_space. Some other thread destroyed the VA range after we
        //    dropped the lock, so we don't have to do so.
        //
        // 2) va_space, empty mask (common case). No other thread is using
        //    mappings in this VA range, so we need to destroy it.
        //
        // 3) va_space, mask not empty. Some other thread re-used the VA range
        //    after we dropped the lock so we can't destroy it.
        if (va_range->va_space && uvm_processor_mask_empty(&va_range->external.mapped_gpus))
            uvm_va_range_destroy(va_range, NULL);

        uvm_va_space_up_write(va_space);

        // In all cases we need to drop the extra count we took above.
        uvm_va_range_release(va_range);
    }
    else {
        uvm_va_space_up_read_rm(va_space);
    }

    return status;
}

NV_STATUS uvm_api_map_external_allocation(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_map_external_allocation(va_space, params);
}

// Version of free which returns but doesn't release the owning GPU
static uvm_gpu_t *uvm_ext_gpu_map_free_internal(uvm_ext_gpu_map_t *ext_gpu_map)
{
    NV_STATUS status;
    uvm_gpu_t *owning_gpu;

    if (!ext_gpu_map)
        return NULL;

    UVM_ASSERT(!ext_gpu_map->pt_range_vec.ranges);

    if (ext_gpu_map->dup_handle) {
        status = uvm_rm_locked_call(nvUvmInterfaceFreeDupedHandle(ext_gpu_map->gpu->rm_address_space,
                                                                  ext_gpu_map->dup_handle));
        UVM_ASSERT(status == NV_OK);
    }

    owning_gpu = ext_gpu_map->owning_gpu;
    uvm_kvfree(ext_gpu_map);

    return owning_gpu;
}

void uvm_ext_gpu_map_free(uvm_ext_gpu_map_t *ext_gpu_map)
{
    uvm_gpu_t *owning_gpu = uvm_ext_gpu_map_free_internal(ext_gpu_map);
    if (owning_gpu)
        uvm_gpu_release(owning_gpu);
}

void uvm_ext_gpu_map_destroy(uvm_va_range_t *va_range, uvm_gpu_t *mapped_gpu, struct list_head *deferred_free_list)
{
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_ext_gpu_map_t *ext_gpu_map = uvm_va_range_ext_gpu_map(va_range, mapped_gpu);

    if (!ext_gpu_map)
        return;

    gpu_va_space = uvm_gpu_va_space_get(va_range->va_space, mapped_gpu);
    UVM_ASSERT(gpu_va_space);

    // Unmap the PTEs
    if (ext_gpu_map->pt_range_vec.ranges) {
        uvm_page_table_range_vec_clear_ptes(&ext_gpu_map->pt_range_vec,
                                            va_range_downgrade_membar(va_range, mapped_gpu));
        uvm_page_table_range_vec_deinit(&ext_gpu_map->pt_range_vec);
    }

    if (deferred_free_list) {
        // If this is a GPU allocation, we have to prevent that GPU from going
        // away until we've freed the handle.
        if (ext_gpu_map->owning_gpu)
            uvm_gpu_retain(ext_gpu_map->owning_gpu);

        uvm_deferred_free_object_add(deferred_free_list,
                                     &ext_gpu_map->deferred_free,
                                     UVM_DEFERRED_FREE_OBJECT_TYPE_EXTERNAL_ALLOCATION);
    }
    else {
        uvm_ext_gpu_map_free_internal(ext_gpu_map);
    }

    va_range->external.gpu_maps[uvm_gpu_index(mapped_gpu->id)] = NULL;
    uvm_processor_mask_clear(&va_range->external.mapped_gpus, mapped_gpu->id);
}

static NV_STATUS uvm_unmap_external_allocation(uvm_va_space_t *va_space, NvU64 base, const NvProcessorUuid *gpu_uuid)
{
    uvm_va_range_t *va_range;
    uvm_gpu_t *gpu = NULL;
    NV_STATUS status = NV_OK;
    LIST_HEAD(deferred_free_list);

    // TODO: Bug 1799173: Consider a va_range lock for external ranges so we can
    //       do the unmap in read mode.
    uvm_va_space_down_write(va_space);

    va_range = uvm_va_range_find(va_space, base);
    if (!va_range || va_range->type != UVM_VA_RANGE_TYPE_EXTERNAL || va_range->node.start != base) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu || !uvm_va_range_ext_gpu_map(va_range, gpu)) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    // Retain the GPU which maps the allocation because it's the parent of
    // dup_handle. The owning GPU (if any) is retained internally by the
    // deferred free layer.
    uvm_gpu_retain(gpu);

    uvm_ext_gpu_map_destroy(va_range, gpu, &deferred_free_list);

out:
    uvm_va_space_up_write(va_space);

    if (status == NV_OK) {
        uvm_deferred_free_object_list(&deferred_free_list);
        uvm_gpu_release(gpu);
    }

    return status;
}

NV_STATUS uvm_api_unmap_external_allocation(UVM_UNMAP_EXTERNAL_ALLOCATION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_unmap_external_allocation(va_space, params->base, &params->gpuUuid);
}

// This destroys VA ranges created by UvmMapExternalAllocation,
// UvmMapDynamicParallelismRegion, and UvmAllocSemaphorePool *only*. VA ranges
// created by UvmMemMap and UvmAlloc go through mmap/munmap.
static NV_STATUS uvm_free(uvm_va_space_t *va_space, NvU64 base, NvU64 length)
{
    uvm_va_range_t *va_range;
    NV_STATUS status = NV_OK;
    uvm_processor_mask_t retained_mask;
    LIST_HEAD(deferred_free_list);

    if (uvm_api_range_invalid_4k(base, length))
        return NV_ERR_INVALID_ADDRESS;

    uvm_va_space_down_write(va_space);

    // Non-managed ranges are defined to not require splitting, so a partial
    // free attempt is an error.
    //
    // TODO: Bug 1763676: The length parameter may be needed for MPS. If not, it
    //       should be removed from the ioctl.
    va_range = uvm_va_range_find(va_space, base);
    if (!va_range                                    ||
        (va_range->type != UVM_VA_RANGE_TYPE_EXTERNAL &&
         va_range->type != UVM_VA_RANGE_TYPE_SKED_REFLECTED &&
         va_range->type != UVM_VA_RANGE_TYPE_SEMAPHORE_POOL) ||
        va_range->node.start != base                 ||
        va_range->node.end != base + length - 1) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    if (va_range->type == UVM_VA_RANGE_TYPE_SEMAPHORE_POOL &&
        uvm_processor_mask_test(&va_range->semaphore_pool.mem->mapped_on, UVM_CPU_ID)) {
        // Semaphore pools must be first unmapped from the CPU with munmap to
        // invalidate the vma.
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    if (va_range->type == UVM_VA_RANGE_TYPE_EXTERNAL) {
        // External ranges have deferred free work, so retain their GPUs
        uvm_processor_mask_copy(&retained_mask, &va_range->external.mapped_gpus);
        uvm_gpu_retain_mask(&retained_mask);
    }

    uvm_va_range_destroy(va_range, &deferred_free_list);

out:
    uvm_va_space_up_write(va_space);

    if (!list_empty(&deferred_free_list)) {
        UVM_ASSERT(status == NV_OK);
        uvm_deferred_free_object_list(&deferred_free_list);
        uvm_gpu_release_mask(&retained_mask);
    }

    return status;
}

NV_STATUS uvm_api_free(UVM_FREE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_free(va_space, params->base, params->length);
}
