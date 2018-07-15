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

#include "uvm8_gpu.h"
#include "uvm8_pmm_sysmem.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_va_block.h"

static struct kmem_cache *g_reverse_page_map_cache __read_mostly;

NV_STATUS uvm_pmm_sysmem_init(void)
{
    g_reverse_page_map_cache = NV_KMEM_CACHE_CREATE("uvm_pmm_sysmem_page_reverse_map_t",
                                                    uvm_reverse_map_t);
    if (!g_reverse_page_map_cache)
        return NV_ERR_NO_MEMORY;

    return NV_OK;
}

void uvm_pmm_sysmem_exit(void)
{
    kmem_cache_destroy_safe(&g_reverse_page_map_cache);
}

NV_STATUS uvm_pmm_sysmem_mappings_init(uvm_gpu_t *gpu, uvm_pmm_sysmem_mappings_t *sysmem_mappings)
{
    memset(sysmem_mappings, 0, sizeof(*sysmem_mappings));

    sysmem_mappings->gpu = gpu;

    uvm_spin_lock_init(&sysmem_mappings->reverse_map_lock, UVM_LOCK_ORDER_LEAF);
    uvm_init_radix_tree_preloadable(&sysmem_mappings->reverse_map_tree);

    return NV_OK;
}

void uvm_pmm_sysmem_mappings_deinit(uvm_pmm_sysmem_mappings_t *sysmem_mappings)
{
    if (sysmem_mappings->gpu) {
        UVM_ASSERT_MSG(radix_tree_empty(&sysmem_mappings->reverse_map_tree),
                       "radix_tree not empty for GPU %s\n",
                       sysmem_mappings->gpu->name);
    }

    sysmem_mappings->gpu = NULL;
}

// TODO: Bug 1995015: use a more efficient data structure for
// physically-contiguous allocations.
NV_STATUS uvm_pmm_sysmem_mappings_add_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                  NvU64 dma_addr,
                                                  NvU64 virt_addr,
                                                  NvU64 region_size,
                                                  uvm_va_block_t *va_block,
                                                  uvm_processor_id_t owner)
{
    int ret;
    uvm_reverse_map_t *new_reverse_map;
    NvU64 key;
    const NvU64 base_key = dma_addr / PAGE_SIZE;
    const NvU32 num_pages = region_size / PAGE_SIZE;
    uvm_page_index_t page_index;

    UVM_ASSERT(va_block);
    UVM_ASSERT(IS_ALIGNED(dma_addr, region_size));
    UVM_ASSERT(IS_ALIGNED(virt_addr, region_size));
    UVM_ASSERT(region_size <= UVM_VA_BLOCK_SIZE);
    UVM_ASSERT(is_power_of_2(region_size));
    UVM_ASSERT(uvm_va_block_contains_address(va_block, virt_addr));
    UVM_ASSERT(uvm_va_block_contains_address(va_block, virt_addr + region_size - 1));
    uvm_assert_mutex_locked(&va_block->lock);

    if (!sysmem_mappings->gpu->access_counters_supported)
        return NV_OK;

    new_reverse_map = kmem_cache_zalloc(g_reverse_page_map_cache, NV_UVM_GFP_FLAGS);
    if (!new_reverse_map)
        return NV_ERR_NO_MEMORY;

    page_index = uvm_va_block_cpu_page_index(va_block, virt_addr);

    new_reverse_map->va_block = va_block;
    new_reverse_map->region   = uvm_va_block_region(page_index, page_index + num_pages);
    new_reverse_map->owner    = owner;

    for (key = base_key; key < base_key + num_pages; ++key) {
        // Pre-load the tree to allocate memory outside of the table lock. This
        // returns with preemption disabled.
        ret = radix_tree_preload(NV_UVM_GFP_FLAGS);
        if (ret != 0) {
            NvU64 remove_key;

            uvm_spin_lock(&sysmem_mappings->reverse_map_lock);
            for (remove_key = base_key; remove_key < key; ++remove_key)
                (void *)radix_tree_delete(&sysmem_mappings->reverse_map_tree, remove_key);
            uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);

            kmem_cache_free(g_reverse_page_map_cache, new_reverse_map);

            return NV_ERR_NO_MEMORY;
        }

        uvm_spin_lock(&sysmem_mappings->reverse_map_lock);
        ret = radix_tree_insert(&sysmem_mappings->reverse_map_tree, key, new_reverse_map);
        uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);
        UVM_ASSERT(ret == 0);

        // This re-enables preemption
        radix_tree_preload_end();
    }


    return NV_OK;
}

static void pmm_sysmem_mappings_remove_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                   NvU64 dma_addr,
                                                   bool check_mapping)
{
    uvm_reverse_map_t *reverse_map;
    NvU64 key;
    const NvU64 base_key = dma_addr / PAGE_SIZE;

    if (!sysmem_mappings->gpu->access_counters_supported)
        return;

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);

    reverse_map = radix_tree_delete(&sysmem_mappings->reverse_map_tree, base_key);
    if (check_mapping)
        UVM_ASSERT(reverse_map);

    if (!reverse_map) {
        uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);
        return;
    }

    uvm_assert_mutex_locked(&reverse_map->va_block->lock);

    for (key = base_key + 1; key < base_key + uvm_va_block_region_num_pages(reverse_map->region); ++key) {
        uvm_reverse_map_t *curr_reverse_map = radix_tree_delete(&sysmem_mappings->reverse_map_tree, key);
        UVM_ASSERT(curr_reverse_map == reverse_map);
    }

    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);

    kmem_cache_free(g_reverse_page_map_cache, reverse_map);
}

void uvm_pmm_sysmem_mappings_remove_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr)
{
    pmm_sysmem_mappings_remove_gpu_mapping(sysmem_mappings, dma_addr, true);
}

void uvm_pmm_sysmem_mappings_remove_gpu_mapping_on_eviction(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr)
{
    pmm_sysmem_mappings_remove_gpu_mapping(sysmem_mappings, dma_addr, false);
}

void uvm_pmm_sysmem_mappings_reparent_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                  NvU64 dma_addr,
                                                  uvm_va_block_t *va_block)
{
    NvU64 virt_addr;
    uvm_reverse_map_t *reverse_map;
    const NvU64 base_key = dma_addr / PAGE_SIZE;
    uvm_page_index_t new_start_page;

    UVM_ASSERT(PAGE_ALIGNED(dma_addr));

    if (!sysmem_mappings->gpu->access_counters_supported)
        return;

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);

    reverse_map = radix_tree_lookup(&sysmem_mappings->reverse_map_tree, base_key);
    UVM_ASSERT(reverse_map);

    // Compute virt address by hand since the old VA block may be messed up
    // during split
    virt_addr = reverse_map->va_block->start + reverse_map->region.first * PAGE_SIZE;
    new_start_page = uvm_va_block_cpu_page_index(va_block, virt_addr);

    reverse_map->region   = uvm_va_block_region(new_start_page,
                                                new_start_page + uvm_va_block_region_num_pages(reverse_map->region));
    reverse_map->va_block = va_block;

    UVM_ASSERT(uvm_va_block_contains_address(va_block, uvm_reverse_map_start(reverse_map)));
    UVM_ASSERT(uvm_va_block_contains_address(va_block, uvm_reverse_map_end(reverse_map)));

    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);
}

NV_STATUS uvm_pmm_sysmem_mappings_split_gpu_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                     NvU64 dma_addr,
                                                     NvU64 new_region_size)
{
    uvm_reverse_map_t *orig_reverse_map;
    const NvU64 base_key = dma_addr / PAGE_SIZE;
    const size_t num_pages = new_region_size / PAGE_SIZE;
    size_t old_num_pages;
    size_t subregion, num_subregions;
    uvm_reverse_map_t **new_reverse_maps;

    UVM_ASSERT(IS_ALIGNED(dma_addr, new_region_size));
    UVM_ASSERT(new_region_size <= UVM_VA_BLOCK_SIZE);
    UVM_ASSERT(is_power_of_2(new_region_size));

    if (!sysmem_mappings->gpu->access_counters_supported)
        return NV_OK;

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);
    orig_reverse_map = radix_tree_lookup(&sysmem_mappings->reverse_map_tree, base_key);
    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);

    // We can access orig_reverse_map outside the tree lock because we hold the
    // VA block lock so we cannot have concurrent modifications in the tree for
    // the mappings of the chunks that belong to that VA block.
    UVM_ASSERT(orig_reverse_map);
    UVM_ASSERT(orig_reverse_map->va_block);
    uvm_assert_mutex_locked(&orig_reverse_map->va_block->lock);
    old_num_pages = uvm_va_block_region_num_pages(orig_reverse_map->region);
    UVM_ASSERT(num_pages < old_num_pages);

    num_subregions = old_num_pages / num_pages;

    new_reverse_maps = uvm_kvmalloc_zero(sizeof(*new_reverse_maps) * (num_subregions - 1));
    if (!new_reverse_maps)
        return NV_ERR_NO_MEMORY;

    // Allocate the descriptors for the new subregions
    for (subregion = 1; subregion < num_subregions; ++subregion) {
        uvm_reverse_map_t *new_reverse_map = kmem_cache_zalloc(g_reverse_page_map_cache, NV_UVM_GFP_FLAGS);
        uvm_page_index_t page_index = orig_reverse_map->region.first + num_pages * subregion;

        if (new_reverse_map == NULL) {
            // On error, free the previously-created descriptors
            while (--subregion != 0)
                kmem_cache_free(g_reverse_page_map_cache, new_reverse_maps[subregion - 1]);

            uvm_kvfree(new_reverse_maps);
            return NV_ERR_NO_MEMORY;
        }

        new_reverse_map->va_block = orig_reverse_map->va_block;
        new_reverse_map->region   = uvm_va_block_region(page_index, page_index + num_pages);
        new_reverse_map->owner    = orig_reverse_map->owner;

        new_reverse_maps[subregion - 1] = new_reverse_map;
    }

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);

    for (subregion = 1; subregion < num_subregions; ++subregion) {
        NvU64 key;

        for (key = base_key + num_pages * subregion; key < base_key + num_pages * (subregion + 1); ++key) {
            void **slot = radix_tree_lookup_slot(&sysmem_mappings->reverse_map_tree, key);
            UVM_ASSERT(slot);
            UVM_ASSERT(radix_tree_deref_slot(slot) == orig_reverse_map);

            NV_RADIX_TREE_REPLACE_SLOT(&sysmem_mappings->reverse_map_tree, slot, new_reverse_maps[subregion - 1]);
        }
    }

    orig_reverse_map->region = uvm_va_block_region(orig_reverse_map->region.first,
                                                   orig_reverse_map->region.first + num_pages);

    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);

    uvm_kvfree(new_reverse_maps);
    return NV_OK;
}

void uvm_pmm_sysmem_mappings_merge_gpu_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                NvU64 dma_addr,
                                                NvU64 new_region_size)
{
    uvm_reverse_map_t *first_reverse_map;
    uvm_page_index_t running_page_index;
    NvU64 key;
    const NvU64 base_key = dma_addr / PAGE_SIZE;
    const size_t num_pages = new_region_size / PAGE_SIZE;
    size_t num_mapping_pages;

    UVM_ASSERT(IS_ALIGNED(dma_addr, new_region_size));
    UVM_ASSERT(new_region_size <= UVM_VA_BLOCK_SIZE);
    UVM_ASSERT(is_power_of_2(new_region_size));

    if (!sysmem_mappings->gpu->access_counters_supported)
        return;

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);

    // Find the first mapping in the region
    first_reverse_map = radix_tree_lookup(&sysmem_mappings->reverse_map_tree, base_key);
    UVM_ASSERT(first_reverse_map);
    num_mapping_pages = uvm_va_block_region_num_pages(first_reverse_map->region);
    UVM_ASSERT(num_pages >= num_mapping_pages);
    UVM_ASSERT(IS_ALIGNED(base_key, num_mapping_pages));

    // The region in the tree matches the size of the merged region, just return
    if (num_pages == num_mapping_pages)
        goto unlock_no_update;

    // Otherwise update the rest of slots to point at the same reverse map
    // descriptor
    key = base_key + uvm_va_block_region_num_pages(first_reverse_map->region);
    running_page_index = first_reverse_map->region.outer;
    while (key < base_key + num_pages) {
        uvm_reverse_map_t *reverse_map = NULL;
        void **slot = radix_tree_lookup_slot(&sysmem_mappings->reverse_map_tree, key);
        size_t slot_index;
        UVM_ASSERT(slot);

        reverse_map = radix_tree_deref_slot(slot);
        UVM_ASSERT(reverse_map);
        UVM_ASSERT(reverse_map != first_reverse_map);
        UVM_ASSERT(reverse_map->va_block == first_reverse_map->va_block);
        UVM_ASSERT(reverse_map->owner == first_reverse_map->owner);
        UVM_ASSERT(reverse_map->region.first == running_page_index);

        NV_RADIX_TREE_REPLACE_SLOT(&sysmem_mappings->reverse_map_tree, slot, first_reverse_map);

        num_mapping_pages = uvm_va_block_region_num_pages(reverse_map->region);
        UVM_ASSERT(IS_ALIGNED(key, num_mapping_pages));
        UVM_ASSERT(key + num_mapping_pages <= base_key + num_pages);

        for (slot_index = 1; slot_index < num_mapping_pages; ++slot_index) {
            slot = radix_tree_lookup_slot(&sysmem_mappings->reverse_map_tree, key + slot_index);
            UVM_ASSERT(slot);
            UVM_ASSERT(reverse_map == radix_tree_deref_slot(slot));

            NV_RADIX_TREE_REPLACE_SLOT(&sysmem_mappings->reverse_map_tree, slot, first_reverse_map);
        }

        key += num_mapping_pages;
        running_page_index = reverse_map->region.outer;

        kmem_cache_free(g_reverse_page_map_cache, reverse_map);
    }

    // Grow the first mapping to cover the whole region
    first_reverse_map->region.outer = first_reverse_map->region.first + num_pages;

unlock_no_update:
    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);
}

size_t uvm_pmm_sysmem_mappings_dma_to_virt(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                           NvU64 dma_addr,
                                           NvU64 region_size,
                                           uvm_reverse_map_t *out_mappings,
                                           size_t max_out_mappings)
{
    NvU64 key;
    size_t num_mappings = 0;
    const NvU64 base_key = dma_addr / PAGE_SIZE;
    NvU32 num_pages = region_size / PAGE_SIZE;

    UVM_ASSERT(region_size >= PAGE_SIZE);
    UVM_ASSERT(PAGE_ALIGNED(region_size));
    UVM_ASSERT(sysmem_mappings->gpu->access_counters_supported);
    UVM_ASSERT(max_out_mappings > 0);

    uvm_spin_lock(&sysmem_mappings->reverse_map_lock);

    key = base_key;
    do {
        uvm_reverse_map_t *reverse_map = radix_tree_lookup(&sysmem_mappings->reverse_map_tree, key);

        if (reverse_map) {
            size_t num_chunk_pages = uvm_va_block_region_num_pages(reverse_map->region);
            NvU32 page_offset = key & (num_chunk_pages - 1);
            NvU32 num_mapping_pages = min(num_pages, (NvU32)num_chunk_pages - page_offset);

            // Sysmem mappings are removed during VA block destruction.
            // Therefore, we can safely retain the VA blocks as long as they
            // are in the reverse map and we hold the reverse map lock.
            uvm_va_block_retain(reverse_map->va_block);
            out_mappings[num_mappings]               = *reverse_map;
            out_mappings[num_mappings].region.first += page_offset;
            out_mappings[num_mappings].region.outer  = out_mappings[num_mappings].region.first + num_mapping_pages;

            if (++num_mappings == max_out_mappings)
                break;

            num_pages -= num_mapping_pages;
            key       += num_mapping_pages;
        }
        else {
            --num_pages;
            ++key;
        }
    }
    while (num_pages > 0);

    uvm_spin_unlock(&sysmem_mappings->reverse_map_lock);

    return num_mappings;
}
