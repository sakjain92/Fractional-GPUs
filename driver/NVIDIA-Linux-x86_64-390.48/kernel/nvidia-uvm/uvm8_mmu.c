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

#include "uvmtypes.h"
#include "uvm8_forward_decl.h"
#include "uvm8_gpu.h"
#include "uvm8_mmu.h"
#include "uvm8_hal.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_pte_batch.h"
#include "uvm8_tlb_batch.h"
#include "uvm8_push.h"
#include "uvm8_mem.h"
#include "uvm8_va_space.h"
#include <stdarg.h>

// The page tree has 5 levels on pascal, and the root is never freed by a normal 'put' operation
// which leaves a maximum of 4 levels
#define MAX_OPERATION_DEPTH 4

// Wrappers for push begin handling channel_manager not being there when running
// the page tree unit test
#define page_tree_begin_acquire(tree, tracker, push, format, ...) ({                                                            \
    NV_STATUS status = NV_OK;                                                                                                   \
    uvm_channel_manager_t *manager = (tree)->gpu->channel_manager;                                                              \
    if (manager != NULL)                                                                                                        \
        status = uvm_push_begin_acquire(manager, UVM_CHANNEL_TYPE_GPU_INTERNAL, (tracker), (push), (format), ##__VA_ARGS__);    \
     else                                                                                                                       \
        status = uvm_push_begin_fake((tree)->gpu, (push));                                                                      \
    status;                                                                                                                     \
})

// Default location of page table allocations
static uvm_aperture_t page_table_aperture = UVM_APERTURE_VID;

static char *uvm_page_table_location;
module_param(uvm_page_table_location, charp, S_IRUGO);
MODULE_PARM_DESC(uvm_page_table_location,
                "Set the location for UVM-allocated page tables. Choices are: vid, sys.");

NV_STATUS uvm_mmu_init(void)
{
    UVM_ASSERT(page_table_aperture == UVM_APERTURE_SYS || page_table_aperture == UVM_APERTURE_VID);

    if (!uvm_page_table_location)
        return NV_OK;

    // TODO: Bug 1766651: Add modes for testing, e.g. alternating vidmem and
    //       sysmem etc.
    if (strcmp(uvm_page_table_location, "vid") == 0) {
        page_table_aperture = UVM_APERTURE_VID;
    }
    else if (strcmp(uvm_page_table_location, "sys") == 0) {
        page_table_aperture = UVM_APERTURE_SYS;
    }
    else {
        pr_info("Invalid uvm_page_table_location %s. Using %s instead.\n",
                uvm_page_table_location,
                page_table_aperture == UVM_APERTURE_SYS ? "sys" : "vid");
    }

    return NV_OK;
}

static NV_STATUS phys_mem_allocate_sysmem(uvm_page_tree_t *tree, NvLength size, uvm_mmu_page_table_alloc_t *out)
{
    NV_STATUS status = NV_OK;
    NvU64 dma_addr;
    out->handle.page = alloc_pages(NV_UVM_GFP_FLAGS | __GFP_ZERO, get_order(size));
    if (out->handle.page == NULL)
        return NV_ERR_NO_MEMORY;

    // Check for fake GPUs from the unit test
    if (tree->gpu->pci_dev)
        status = uvm_gpu_map_cpu_pages(tree->gpu, out->handle.page, UVM_PAGE_ALIGN_UP(size), &dma_addr);
    else
        dma_addr = page_to_phys(out->handle.page);

    if (status != NV_OK) {
        __free_pages(out->handle.page, get_order(size));
        return status;
    }

    out->addr = uvm_gpu_phys_address(UVM_APERTURE_SYS, dma_addr);
    out->size = size;

    return NV_OK;
}

static NV_STATUS phys_mem_allocate_vidmem(uvm_page_tree_t *tree, NvLength size, uvm_pmm_alloc_flags_t pmm_flags, uvm_mmu_page_table_alloc_t *out)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = tree->gpu;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

    status = uvm_pmm_gpu_alloc_kernel(&gpu->pmm, 1, size, pmm_flags, &out->handle.chunk, &local_tracker);
    if (status != NV_OK)
        return status;

    if (!uvm_tracker_is_empty(&local_tracker)) {
        uvm_mutex_lock(&tree->lock);
        status = uvm_tracker_add_tracker_safe(&tree->tracker, &local_tracker);
        uvm_mutex_unlock(&tree->lock);
    }

    uvm_tracker_deinit(&local_tracker);

    if (status != NV_OK) {
        uvm_pmm_gpu_free(&tree->gpu->pmm, out->handle.chunk, NULL);
        return status;
    }

    out->addr = uvm_gpu_phys_address(UVM_APERTURE_VID, out->handle.chunk->address);
    out->size = size;

    return status;
}

static NV_STATUS phys_mem_allocate(uvm_page_tree_t *tree, NvLength size, uvm_aperture_t location, uvm_pmm_alloc_flags_t pmm_flags, uvm_mmu_page_table_alloc_t *out)
{
    memset(out, 0, sizeof(*out));

    if (location == UVM_APERTURE_SYS)
        return phys_mem_allocate_sysmem(tree, size, out);
    else
        return phys_mem_allocate_vidmem(tree, size, pmm_flags, out);
}

static void phys_mem_deallocate_vidmem(uvm_page_tree_t *tree, uvm_mmu_page_table_alloc_t *ptr)
{
    uvm_assert_mutex_locked(&tree->lock);
    UVM_ASSERT(ptr->addr.aperture == UVM_APERTURE_VID);

    uvm_pmm_gpu_free(&tree->gpu->pmm, ptr->handle.chunk, &tree->tracker);
}

static void phys_mem_deallocate_sysmem(uvm_page_tree_t *tree, uvm_mmu_page_table_alloc_t *ptr)
{
    NV_STATUS status;

    uvm_assert_mutex_locked(&tree->lock);

    // Synchronize any pending operations before freeing the memory that might
    // be used by them.
    status = uvm_tracker_wait(&tree->tracker);
    if (status != NV_OK)
        UVM_ASSERT(status == uvm_global_get_status());

    UVM_ASSERT(ptr->addr.aperture == UVM_APERTURE_SYS);
    if (tree->gpu->pci_dev)
        uvm_gpu_unmap_cpu_pages(tree->gpu, ptr->addr.address, UVM_PAGE_ALIGN_UP(ptr->size));
    __free_pages(ptr->handle.page, get_order(ptr->size));
}

static void phys_mem_deallocate(uvm_page_tree_t *tree, uvm_mmu_page_table_alloc_t *ptr)
{
    if (ptr->addr.aperture == UVM_APERTURE_SYS)
        phys_mem_deallocate_sysmem(tree, ptr);
    else
        phys_mem_deallocate_vidmem(tree, ptr);

    memset(ptr, 0, sizeof(*ptr));
}

static void page_table_range_init(uvm_page_table_range_t *range,
                                 NvU32 page_size,
                                 uvm_page_directory_t *dir,
                                 NvU32 start_index,
                                 NvU32 end_index)
{
    range->table = dir;
    range->start_index = start_index;
    range->entry_count = 1 + end_index - start_index;
    range->page_size = page_size;
    dir->ref_count += range->entry_count;
}

static void phys_mem_init(uvm_page_tree_t *tree, NvU32 page_size, uvm_page_directory_t *dir, uvm_push_t *push)
{
    NvU64 clear_bits[2];
    uvm_mmu_mode_hal_t *hal = tree->hal;

    if (dir->depth == tree->hal->page_table_depth(page_size)) {
        *clear_bits = 0; // Invalid PTE
    }
    else {
        // passing in NULL for the phys_allocs will mark the child entries as invalid
        uvm_mmu_page_table_alloc_t *phys_allocs[2] = {NULL, NULL};
        hal->make_pde(clear_bits, phys_allocs, dir->depth);

        // Make sure that using only clear_bits[0] will work
        UVM_ASSERT(hal->entry_size(dir->depth) == sizeof(clear_bits[0]) || clear_bits[0] == clear_bits[1]);
    }

    // initialize the memory to a reasonable value
    tree->gpu->ce_hal->memset_8(push,
                                uvm_gpu_address_from_phys(dir->phys_alloc.addr),
                                *clear_bits,
                                dir->phys_alloc.size);
}

static uvm_aperture_t page_tree_pick_location_for_dir(uvm_page_tree_t *tree, NvU32 page_size, NvU32 depth,
        uvm_pmm_alloc_flags_t pmm_flags)
{
    // If the caller required a specific location, use that
    if (tree->location != UVM_APERTURE_DEFAULT)
        return tree->location;

    // Otherwise, let the module parameter decide
    return page_table_aperture;
}

static uvm_page_directory_t *allocate_directory_with_location(uvm_page_tree_t *tree, NvU32 page_size, NvU32 depth,
        uvm_aperture_t location, uvm_pmm_alloc_flags_t pmm_flags)
{
    NV_STATUS status;
    uvm_mmu_mode_hal_t *hal = tree->hal;
    NvU32 entry_count;
    NvLength phys_alloc_size = hal->allocation_size(depth, page_size);
    uvm_page_directory_t *dir;

    UVM_ASSERT(location == UVM_APERTURE_VID || location == UVM_APERTURE_SYS);

    // The page tree doesn't cache PTEs so space is not allocated for entries that are always PTEs.
    // 2M PTEs may later become PDEs so pass UVM_PAGE_SIZE_AGNOSTIC, not page_size.
    if (depth == hal->page_table_depth(UVM_PAGE_SIZE_AGNOSTIC))
        entry_count = 0;
    else
        entry_count = hal->entries_per_index(depth) << hal->index_bits(depth, page_size);

    dir = uvm_kvmalloc_zero(sizeof(uvm_page_directory_t) + sizeof(dir->entries[0]) * entry_count);
    if (dir == NULL)
        return NULL;

    status = phys_mem_allocate(tree, phys_alloc_size, location, pmm_flags, &dir->phys_alloc);
    if (status == NV_ERR_NO_MEMORY && location == UVM_APERTURE_VID && (pmm_flags & UVM_PMM_ALLOC_FLAGS_EVICT) != 0) {
        // Fall back to sysmem if allocating page tables in vidmem with eviction fails
        status = phys_mem_allocate(tree, phys_alloc_size, UVM_APERTURE_SYS, pmm_flags, &dir->phys_alloc);
    }

    if (status != NV_OK) {
        uvm_kvfree(dir);
        return NULL;
    }
    dir->depth = depth;

    return dir;
}

static uvm_page_directory_t *allocate_directory(uvm_page_tree_t *tree, NvU32 page_size, NvU32 depth,
        uvm_pmm_alloc_flags_t pmm_flags)
{
    uvm_aperture_t location = page_tree_pick_location_for_dir(tree, page_size, depth, pmm_flags);
    return allocate_directory_with_location(tree, page_size, depth, location, pmm_flags);
}

static inline NvU32 entry_index_from_vaddr(NvU64 vaddr, NvU32 addr_bit_shift, NvU32 bits)
{
    NvU64 mask = ((NvU64)1 << bits) - 1;
    return (NvU32)((vaddr >> addr_bit_shift) & mask);
}

static inline NvU32 index_to_entry(uvm_mmu_mode_hal_t *hal, NvU32 entry_index, NvU32 depth, NvU32 page_size)
{
    return hal->entries_per_index(depth) * entry_index + hal->entry_offset(depth, page_size);
}

static uvm_page_directory_t *host_pde_write(uvm_page_directory_t *dir, uvm_page_directory_t *parent, NvU32 index_in_parent)
{
    dir->host_parent = parent;
    dir->index_in_parent = index_in_parent;
    parent->ref_count++;
    return dir;
}

static void pde_write(uvm_page_tree_t *tree, uvm_page_directory_t *dir, NvU32 entry_index, bool force_clear, uvm_push_t *push)
{
    NvU32 i;
    NvU64 entry_bits[2];
    uvm_mmu_page_table_alloc_t *phys_allocs[2];
    uvm_mmu_mode_hal_t *hal = tree->hal;
    NvU64 dev_entry = dir->phys_alloc.addr.address + hal->entry_size(dir->depth) * entry_index;
    NvU32 entries_per_index = hal->entries_per_index(dir->depth);
    NvU32 entry_size = hal->entry_size(dir->depth);
    NvU32 memset_count = entry_size / sizeof(entry_bits[0]);
    NvU32 membar_flag;

    UVM_ASSERT(sizeof(entry_bits) >= entry_size);

    // extract physical allocs from non-null entries
    for (i = 0; i < entries_per_index; i++) {
        uvm_page_directory_t *entry = dir->entries[entries_per_index * entry_index + i];
        if (entry == NULL || force_clear)
            phys_allocs[i] = NULL;
        else
            phys_allocs[i] = &entry->phys_alloc;
    }

    // make_pde always writes the whole PDE, even if it is a dual entry.
    hal->make_pde(entry_bits, phys_allocs, dir->depth);

    membar_flag = 0;
    if (uvm_push_get_and_reset_flag(push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE))
        membar_flag = UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE;
    else if (uvm_push_get_and_reset_flag(push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_GPU))
        membar_flag = UVM_PUSH_FLAG_CE_NEXT_MEMBAR_GPU;

    // each entry could be either 1 or 2 64-bit words.
    for (i = 0; i < memset_count; i++) {
        // Always respect the caller's pipelining setting for the first push. The second can pipeline.
        if (i != 0)
            uvm_push_set_flag(push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);

        // No membar is needed until the last copy. Otherwise, use caller's membar flag
        if (i != memset_count - 1)
            uvm_push_set_flag(push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
        else if (membar_flag)
            uvm_push_set_flag(push, membar_flag);

        tree->gpu->ce_hal->memset_8(push,
                uvm_gpu_address_physical(dir->phys_alloc.addr.aperture, dev_entry + sizeof(entry_bits[0]) * i),
                entry_bits[i],
                sizeof(entry_bits[0]));
    }
}

static void host_pde_clear(uvm_page_tree_t *tree, uvm_page_directory_t *dir, NvU32 entry_index, NvU32 page_size)
{
    UVM_ASSERT(dir->ref_count > 0);

    dir->entries[index_to_entry(tree->hal, entry_index, dir->depth, page_size)] = NULL;
    dir->ref_count--;
}

static void pde_clear(uvm_page_tree_t *tree, uvm_page_directory_t *dir, NvU32 entry_index, NvU32 page_size, void *push)
{
    host_pde_clear(tree, dir, entry_index, page_size);
    pde_write(tree, dir, entry_index, false, push);
}

static uvm_chunk_sizes_mask_t allocation_sizes_for_big_page_size(uvm_gpu_t *gpu, NvU32 big_page_size)
{
    uvm_chunk_sizes_mask_t alloc_sizes = 0;
    uvm_mmu_mode_hal_t *hal = gpu->arch_hal->mmu_mode_hal(big_page_size);

    if (hal != NULL) {
        unsigned long page_size_log2;
        unsigned long page_sizes = hal->page_sizes();
        BUILD_BUG_ON(sizeof(hal->page_sizes()) > sizeof(page_sizes));

        for_each_set_bit(page_size_log2, &page_sizes, BITS_PER_LONG) {
            NvU32 i;
            uvm_chunk_size_t page_size = (uvm_chunk_size_t)1 << page_size_log2;
            for (i = 0; i <= hal->page_table_depth(page_size); i++)
                alloc_sizes |= hal->allocation_size(i, page_size);
        }
    }

    return alloc_sizes;
}

static NvU32 page_sizes_for_big_page_size(uvm_gpu_t *gpu, NvU32 big_page_size)
{
    uvm_mmu_mode_hal_t *hal = gpu->arch_hal->mmu_mode_hal(big_page_size);

    if (hal != NULL)
        return hal->page_sizes();
    else
        return 0;
}

static void page_tree_end(uvm_page_tree_t *tree, uvm_push_t *push)
{
    if (tree->gpu->channel_manager != NULL)
        uvm_push_end(push);
    else
        uvm_push_end_fake(push);
}

static void page_tree_tracker_overwrite_with_push(uvm_page_tree_t *tree, uvm_push_t *push)
{
    uvm_assert_mutex_locked(&tree->lock);

    // No GPU work to track for fake GPU testing
    if (tree->gpu->channel_manager == NULL)
        return;

    uvm_tracker_overwrite_with_push(&tree->tracker, push);
}

static NV_STATUS page_tree_end_and_wait(uvm_page_tree_t *tree, uvm_push_t *push)
{
    if (tree->gpu->channel_manager != NULL)
        return uvm_push_end_and_wait(push);
    else
        uvm_push_end_fake(push);

    return NV_OK;
}

// initialize new page tables and insert them into the tree
static NV_STATUS write_gpu_state(uvm_page_tree_t *tree,
                                 NvU32 page_size,
                                 NvS32 invalidate_depth,
                                 NvU32 used_count,
                                 uvm_page_directory_t **dirs_used)
{
    NvS32 i;
    uvm_push_t push;
    NV_STATUS status;

    // The logic of what membar is needed when is pretty subtle, please refer to
    // the UVM Functional Spec (section 5.1) for all the details.
    uvm_membar_t membar_after_writes = UVM_MEMBAR_GPU;

    uvm_assert_mutex_locked(&tree->lock);

    if (used_count == 0)
        return NV_OK;

    status = page_tree_begin_acquire(tree, &tree->tracker, &push, "write_gpu_state: %u dirs", used_count);
    if (status != NV_OK)
        return status;

    // only do GPU work once all the allocations have succeeded
    // first, zero-out the new allocations
    for (i = 0; i < used_count; i++) {
        // Appropriate membar will be done after all the writes. Pipelining can
        // be enabled as they are all initializing newly allocated memory that
        // cannot have any writes pending.
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);

        phys_mem_init(tree, page_size, dirs_used[i], &push);

        if (dirs_used[i]->phys_alloc.addr.aperture == UVM_APERTURE_SYS)
            membar_after_writes = UVM_MEMBAR_SYS;
    }

    // Only a single membar is needed between the memsets of the page tables
    // and the writes of the PDEs pointing to those page tables.
    // The membar can be local if all of the page tables and PDEs are in GPU memory,
    // but must be a sysmembar if any of them are in sysmem.
    tree->gpu->host_hal->wait_for_idle(&push);
    uvm_hal_membar(tree->gpu, &push, membar_after_writes);

    // Reset back to a local membar by default
    membar_after_writes = UVM_MEMBAR_GPU;

    // write entries bottom up, so that they are valid once they're inserted into the tree
    for (i = used_count - 1; i >= 0; i--) {
        uvm_page_directory_t *dir = dirs_used[i];

        // Appropriate membar will be done after all the writes. Pipelining can
        // be enabled as they are all independent and we just did a WFI above.
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
        pde_write(tree, dir->host_parent, dir->index_in_parent, false, &push);

        // If any of the written PDEs is in sysmem, a sysmembar is needed before
        // the TLB invalidate.
        // Notably sysmembar is needed even though the writer (CE) and reader (MMU) are
        // on the same GPU, because CE physical writes take the L2 bypass path.
        if (dir->host_parent->phys_alloc.addr.aperture == UVM_APERTURE_SYS)
            membar_after_writes = UVM_MEMBAR_SYS;
    }

    tree->gpu->host_hal->wait_for_idle(&push);
    uvm_hal_membar(tree->gpu, &push, membar_after_writes);

    UVM_ASSERT(invalidate_depth >= 0);

    // Upgrades don't have to flush out accesses, so no membar is needed on the TLB invalidate.
    tree->gpu->host_hal->tlb_invalidate_all(&push, uvm_page_tree_pdb(tree)->addr, invalidate_depth, UVM_MEMBAR_NONE);

    // We just did the appropriate membar after the WFI, so no need for another
    // one in push_end().
    // At least currently as if the L2 bypass path changes to only require a GPU
    // membar between PDE write and TLB invalidate, we'll need to push a
    // sysmembar so the end-of-push semaphore is ordered behind the PDE writes.
    uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
    page_tree_end(tree, &push);
    page_tree_tracker_overwrite_with_push(tree, &push);

    return NV_OK;
}

static void free_unused_directories(uvm_page_tree_t *tree,
                                    NvU32 used_count,
                                    uvm_page_directory_t **dirs_used,
                                    uvm_page_directory_t **dir_cache)
{
    NvU32 i;

    // free unused entries
    for (i = 0; i < MAX_OPERATION_DEPTH; i++) {
        uvm_page_directory_t *dir = dir_cache[i];
        if (dir != NULL) {
            NvU32 j;

            for (j = 0; j < used_count; j++) {
                if (dir == dirs_used[j])
                    break;
            }

            if (j == used_count) {
                phys_mem_deallocate(tree, &dir->phys_alloc);
                uvm_kvfree(dir);
            }
        }
    }

}

NV_STATUS uvm_page_tree_init(uvm_gpu_t *gpu, NvU32 big_page_size, uvm_aperture_t location, uvm_page_tree_t *tree)
{
    uvm_push_t push;
    NV_STATUS status;
    BUILD_BUG_ON(sizeof(uvm_page_directory_t) != offsetof(uvm_page_directory_t, entries));

    UVM_ASSERT_MSG(location == UVM_APERTURE_VID ||
                   location == UVM_APERTURE_SYS ||
                   location == UVM_APERTURE_DEFAULT,
                   "Invalid location %s (%d)\n", uvm_aperture_string(location), (int)location);

    memset(tree, 0, sizeof(*tree));
    uvm_mutex_init(&tree->lock, UVM_LOCK_ORDER_PAGE_TREE);
    tree->hal = gpu->arch_hal->mmu_mode_hal(big_page_size);
    UVM_ASSERT(tree->hal != NULL);
    tree->gpu = gpu;
    tree->big_page_size = big_page_size;
    tree->location = location;

    uvm_tracker_init(&tree->tracker);

    tree->root = allocate_directory(tree, UVM_PAGE_SIZE_AGNOSTIC, 0, UVM_PMM_ALLOC_FLAGS_EVICT);

    if (tree->root == NULL)
        return NV_ERR_NO_MEMORY;

    status = page_tree_begin_acquire(tree, &tree->tracker, &push, "init page tree");
    if (status != NV_OK)
        return status;

    phys_mem_init(tree, UVM_PAGE_SIZE_AGNOSTIC, tree->root, &push);
    return page_tree_end_and_wait(tree, &push);
}

void uvm_page_tree_deinit(uvm_page_tree_t *tree)
{
    UVM_ASSERT(tree->root->ref_count == 0);

    uvm_mutex_lock(&tree->lock);
    (void)uvm_tracker_wait(&tree->tracker);
    phys_mem_deallocate(tree, &tree->root->phys_alloc);
    uvm_mutex_unlock(&tree->lock);

    uvm_tracker_deinit(&tree->tracker);
    uvm_kvfree(tree->root);
}

void uvm_page_tree_put_ptes_async(uvm_page_tree_t *tree, uvm_page_table_range_t *range)
{
    NvU32 free_count = 0;
    NvU32 i;
    uvm_page_directory_t *free_queue[MAX_OPERATION_DEPTH];
    uvm_page_directory_t *dir = range->table;
    uvm_push_t push;
    NV_STATUS status;
    NvU32 invalidate_depth;

    // The logic of what membar is needed when is pretty subtle, please refer to
    // the UVM Functional Spec (section 5.1) for all the details.
    uvm_membar_t membar_after_pde_clears = UVM_MEMBAR_GPU;
    uvm_membar_t membar_after_invalidate = UVM_MEMBAR_GPU;

    uvm_mutex_lock(&tree->lock);

    // release the range
    UVM_ASSERT(dir->ref_count >= range->entry_count);
    dir->ref_count -= range->entry_count;

    // traverse until we hit an in-use page, or the root
    while (dir->host_parent != NULL && dir->ref_count == 0) {
        uvm_page_directory_t *parent = dir->host_parent;

        if (free_count == 0) {
            // begin a push which will be submitted before the memory gets freed
            status = page_tree_begin_acquire(tree, &tree->tracker, &push, "put ptes: start: %u, count: %u",
                                     range->start_index, range->entry_count);
            // Failure to get a push can only happen if we've hit a fatal UVM
            // channel error. We can't perform the unmap, so just leave things
            // in place for debug.
            if (status != NV_OK) {
                UVM_ASSERT(status == uvm_global_get_status());
                dir->ref_count += range->entry_count;
                uvm_mutex_unlock(&tree->lock);
                return;
            }
        }

        // All writes can be pipelined as put_ptes() cannot be called with any
        // operations pending on the affected PTEs and PDEs.
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);

        // Don't issue any membars as part of the clear, a single membar will be
        // done below before the invalidate.
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
        pde_clear(tree, dir->host_parent, dir->index_in_parent, range->page_size, &push);

        invalidate_depth = dir->host_parent->depth;

        // If any of the pointed to PDEs were in sysmem then a SYS membar is
        // required after the TLB invalidate.
        if (dir->phys_alloc.addr.aperture == UVM_APERTURE_SYS)
            membar_after_invalidate = UVM_MEMBAR_SYS;

        // If any of the cleared PDEs were in sysmem then a SYS membar is
        // required after the clears and before the TLB invalidate.
        if (dir->host_parent->phys_alloc.addr.aperture == UVM_APERTURE_SYS)
            membar_after_pde_clears = UVM_MEMBAR_SYS;

        // add this dir to the queue of directories that should be freed once the
        // tracker value of the associated PDE writes is known
        UVM_ASSERT(free_count < MAX_OPERATION_DEPTH);
        free_queue[free_count++] = dir;

        dir = parent;
    }

    if (free_count == 0) {
        uvm_mutex_unlock(&tree->lock);
        return;
    }

    tree->gpu->host_hal->wait_for_idle(&push);
    uvm_hal_membar(tree->gpu, &push, membar_after_pde_clears);
    tree->gpu->host_hal->tlb_invalidate_all(&push,
                                            uvm_page_tree_pdb(tree)->addr,
                                            invalidate_depth,
                                            membar_after_invalidate);

    // We just did the appropriate membar above, no need for another one in push_end().
    // At least currently as if the L2 bypass path changes to only require a GPU
    // membar between PDE write and TLB invalidate, we'll need to push a
    // sysmembar so the end-of-push semaphore is ordered behind the PDE writes.
    uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
    page_tree_end(tree, &push);
    page_tree_tracker_overwrite_with_push(tree, &push);

    // now that we've traversed all the way up the tree, free everything
    for (i = 0; i < free_count; i++) {
        phys_mem_deallocate(tree, &free_queue[i]->phys_alloc);
        uvm_kvfree(free_queue[i]);
    }

    uvm_mutex_unlock(&tree->lock);
}

void uvm_page_tree_put_ptes(uvm_page_tree_t *tree, uvm_page_table_range_t *range)
{
    uvm_page_tree_put_ptes_async(tree, range);
    (void)uvm_page_tree_wait(tree);
}

NV_STATUS uvm_page_tree_wait(uvm_page_tree_t *tree)
{
    NV_STATUS status;

    uvm_mutex_lock(&tree->lock);

    status = uvm_tracker_wait(&tree->tracker);

    uvm_mutex_unlock(&tree->lock);

    return status;
}

NV_STATUS try_get_ptes(uvm_page_tree_t *tree,
                       NvU32 page_size,
                       NvU64 start,
                       NvLength size,
                       uvm_page_table_range_t *range,
                       NvU32 *cur_depth,
                       uvm_page_directory_t **dir_cache)
{
    uvm_mmu_mode_hal_t *hal = tree->hal;
    // bit index just beyond the most significant bit used to index the current entry
    NvU32 addr_bit_shift = hal->num_va_bits();
    // track depth upon which the invalidate occured
    NvS32 invalidate_depth = -1;
    uvm_page_directory_t *dir = tree->root;
    // directories used in attempt
    NvU32 used_count = 0;
    NvU32 i;
    uvm_page_directory_t *dirs_used[MAX_OPERATION_DEPTH];

    uvm_assert_mutex_locked(&tree->lock);

    UVM_ASSERT(is_power_of_2(page_size));

    // ensure that the caller has specified a valid page size
    UVM_ASSERT((page_size & hal->page_sizes()) != 0);

    // This algorithm will work with unaligned ranges, but the caller's intent is unclear
    UVM_ASSERT_MSG(start % page_size == 0 && size % page_size == 0, "start 0x%llx size 0x%zx page_size 0x%x",
            start, (size_t)size, page_size);

    // The GPU should be capable of addressing the passed range
    UVM_ASSERT(uvm_gpu_can_address(tree->gpu, start + size - 1));

    while (true) {
        // index of the entry, for the first byte of the range, within its containing directory
        NvU32 start_index;
        // index of the entry, for the last byte of the range, within its containing directory
        NvU32 end_index;
        // pointer to PDE/PTE
        uvm_page_directory_t **entry;
        NvU32 index_bits = hal->index_bits(dir->depth, page_size);

        addr_bit_shift -= index_bits;
        start_index = entry_index_from_vaddr(start, addr_bit_shift, index_bits);
        end_index = entry_index_from_vaddr(start + size - 1, addr_bit_shift, index_bits);

        UVM_ASSERT(start_index <= end_index && end_index < (1 << index_bits));

        entry = dir->entries + index_to_entry(hal, start_index, dir->depth, page_size);

        if (dir->depth == hal->page_table_depth(page_size)) {
            page_table_range_init(range, page_size, dir, start_index, end_index);
            break;
        }
        else {
            UVM_ASSERT(start_index == end_index);

            if (*entry == NULL) {
                if (dir_cache[dir->depth] == NULL) {
                    *cur_depth = dir->depth;
                    // Undo the changes to the tree so that the dir cache remains private to the thread
                    for (i = 0; i < used_count; i++)
                        host_pde_clear(tree, dirs_used[i]->host_parent, dirs_used[i]->index_in_parent, page_size);

                    return NV_ERR_MORE_PROCESSING_REQUIRED;
                }

                *entry = host_pde_write(dir_cache[dir->depth], dir, start_index);
                dirs_used[used_count++] = *entry;

                if (invalidate_depth == -1)
                    invalidate_depth = dir->depth;
            }
        }
        dir = *entry;
    }

    free_unused_directories(tree, used_count, dirs_used, dir_cache);
    return write_gpu_state(tree, page_size, invalidate_depth, used_count, dirs_used);
}

NV_STATUS uvm_page_tree_get_ptes_async(uvm_page_tree_t *tree, NvU32 page_size, NvU64 start, NvLength size,
        uvm_pmm_alloc_flags_t pmm_flags, uvm_page_table_range_t *range)
{
    NV_STATUS status;
    NvU32 cur_depth = 0;
    uvm_page_directory_t *dir_cache[MAX_OPERATION_DEPTH];
    memset(dir_cache, 0, sizeof(dir_cache));

    uvm_mutex_lock(&tree->lock);
    while ((status = try_get_ptes(tree,
                                  page_size,
                                  start,
                                  size,
                                  range,
                                  &cur_depth,
                                  dir_cache)) == NV_ERR_MORE_PROCESSING_REQUIRED) {
        uvm_mutex_unlock(&tree->lock);

        // try_get_ptes never needs depth 0, so store a directory at its parent's depth
        // TODO: Bug 1766655: Allocate everything below cur_depth instead of
        //       retrying for every level.
        dir_cache[cur_depth] = allocate_directory(tree, page_size, cur_depth + 1, pmm_flags);
        if (dir_cache[cur_depth] == NULL) {
            uvm_mutex_lock(&tree->lock);
            free_unused_directories(tree, 0, NULL, dir_cache);
            uvm_mutex_unlock(&tree->lock);
            return NV_ERR_NO_MEMORY;
        }

        uvm_mutex_lock(&tree->lock);
    }
    uvm_mutex_unlock(&tree->lock);
    return status;
}

NV_STATUS uvm_page_tree_get_ptes(uvm_page_tree_t *tree, NvU32 page_size, NvU64 start, NvLength size,
        uvm_pmm_alloc_flags_t pmm_flags, uvm_page_table_range_t *range)
{
    NV_STATUS status = uvm_page_tree_get_ptes_async(tree, page_size, start, size, pmm_flags, range);
    if (status != NV_OK)
        return status;

    return uvm_page_tree_wait(tree);
}

void uvm_page_table_range_get_upper(uvm_page_tree_t *tree,
                                    uvm_page_table_range_t *existing,
                                    uvm_page_table_range_t *upper,
                                    NvU32 num_upper_pages)
{
    NvU32 upper_start_index = existing->start_index + (existing->entry_count - num_upper_pages);
    NvU32 upper_end_index = upper_start_index + num_upper_pages - 1;

    UVM_ASSERT(num_upper_pages);
    UVM_ASSERT(num_upper_pages <= existing->entry_count);

    uvm_mutex_lock(&tree->lock);
    page_table_range_init(upper, existing->page_size, existing->table, upper_start_index, upper_end_index);
    uvm_mutex_unlock(&tree->lock);
}

void uvm_page_table_range_shrink(uvm_page_tree_t *tree, uvm_page_table_range_t *range, NvU32 new_page_count)
{
    UVM_ASSERT(range->entry_count >= new_page_count);

    if (new_page_count > 0) {
        // Take a ref count on the smaller portion of the PTEs, then drop the
        // entire old range.
        uvm_mutex_lock(&tree->lock);

        UVM_ASSERT(range->table->ref_count >= range->entry_count);
        range->table->ref_count -= (range->entry_count - new_page_count);

        uvm_mutex_unlock(&tree->lock);

        range->entry_count = new_page_count;
    }
    else {
        uvm_page_tree_put_ptes(tree, range);
    }
}

NV_STATUS uvm_page_tree_get_entry(uvm_page_tree_t *tree, NvU32 page_size, NvU64 start, uvm_pmm_alloc_flags_t pmm_flags, uvm_page_table_range_t *single)
{
    NV_STATUS status = uvm_page_tree_get_ptes(tree, page_size, start, page_size, pmm_flags, single);
    UVM_ASSERT(single->entry_count == 1);
    return status;
}

void uvm_page_tree_write_pde(uvm_page_tree_t *tree, uvm_page_table_range_t *single, uvm_push_t *push)
{
    UVM_ASSERT(single->entry_count == 1);
    pde_write(tree, single->table, single->start_index, false, push);
}

void uvm_page_tree_clear_pde(uvm_page_tree_t *tree, uvm_page_table_range_t *single, uvm_push_t *push)
{
    UVM_ASSERT(single->entry_count == 1);
    pde_write(tree, single->table, single->start_index, true, push);
}

static NV_STATUS poison_ptes(uvm_page_tree_t *tree,
                             uvm_page_directory_t *pte_dir,
                             uvm_page_directory_t *parent,
                             NvU32 page_size)
{
    NV_STATUS status;
    uvm_push_t push;

    uvm_assert_mutex_locked(&tree->lock);

    UVM_ASSERT(pte_dir->depth == tree->hal->page_table_depth(page_size));

    status = page_tree_begin_acquire(tree, &tree->tracker, &push, "Poisoning child table of page size %u", page_size);
    if (status != NV_OK)
        return status;

    tree->gpu->ce_hal->memset_8(&push,
                                uvm_gpu_address_from_phys(pte_dir->phys_alloc.addr),
                                tree->hal->poisoned_pte(page_size),
                                pte_dir->phys_alloc.size);

    // If both the new PTEs and the parent PDE are in vidmem, then a GPU-
    // local membar is enough to keep the memset of the PTEs ordered with
    // any later write of the PDE. Otherwise we need a sysmembar. See the
    // comments in write_gpu_state.
    if (pte_dir->phys_alloc.addr.aperture == UVM_APERTURE_VID &&
        parent->phys_alloc.addr.aperture == UVM_APERTURE_VID)
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_GPU);

    page_tree_end(tree, &push);

    // The push acquired the tracker so it's ok to just overwrite it with
    // the entry tracking the push.
    page_tree_tracker_overwrite_with_push(tree, &push);

    return NV_OK;
}

NV_STATUS uvm_page_tree_alloc_table(uvm_page_tree_t *tree,
                                    NvU32 page_size,
                                    uvm_pmm_alloc_flags_t pmm_flags,
                                    uvm_page_table_range_t *single,
                                    uvm_page_table_range_t *children)
{
    bool should_free = false;
    uvm_page_directory_t **entry;
    uvm_page_directory_t *dir;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(single->entry_count == 1);

    entry = single->table->entries + index_to_entry(tree->hal,
                                                    single->start_index,
                                                    single->table->depth,
                                                    page_size);

    dir = allocate_directory(tree, page_size, single->table->depth + 1, pmm_flags);
    if (dir == NULL)
        return NV_ERR_NO_MEMORY;

    uvm_mutex_lock(&tree->lock);

    // The caller is responsible for initializing this table, so enforce that on
    // debug builds.
    if (UVM_IS_DEBUG()) {
        status = poison_ptes(tree, dir, single->table, page_size);
        if (status != NV_OK)
            goto out;
    }

    status = uvm_tracker_wait(&tree->tracker);
    if (status != NV_OK)
        goto out;

    // the range always refers to the entire page table
    children->start_index = 0;
    children->entry_count = 1 << tree->hal->index_bits(dir->depth, page_size);
    children->page_size = page_size;

    // is this entry currently unassigned?
    if (*entry == NULL) {
        children->table = dir;
        *entry = dir;
        host_pde_write(dir, single->table, single->start_index);
    }
    else {
        should_free = true;
        children->table = *entry;
    }
    children->table->ref_count += children->entry_count;

out:
    if (should_free || status != NV_OK) {
        phys_mem_deallocate(tree, &dir->phys_alloc);
        uvm_kvfree(dir);
    }
    uvm_mutex_unlock(&tree->lock);

    return status;
}

uvm_mmu_page_table_alloc_t *uvm_page_tree_pdb(uvm_page_tree_t *tree)
{
    return &tree->root->phys_alloc;
}

uvm_chunk_sizes_mask_t uvm_mmu_kernel_chunk_sizes(uvm_gpu_t *gpu)
{
    return allocation_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_64K) |
           allocation_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_128K);
}

uvm_chunk_sizes_mask_t uvm_mmu_all_user_chunk_sizes(uvm_gpu_t *gpu)
{
    // TODO: Only use color page size when need to allocate colored page
    uvm_chunk_sizes_mask_t sizes;
    
    sizes = page_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_64K)  |
                                page_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_128K) |
                                PAGE_SIZE;

    // Although we may have to map PTEs smaller than PAGE_SIZE, user (managed)
    // memory is never allocated with granularity smaller than PAGE_SIZE. Force
    // PAGE_SIZE to be supported and the smallest allowed size so we don't have
    // to handle allocating multiple chunks per page.
    return sizes & PAGE_MASK;
}

uvm_chunk_sizes_mask_t uvm_mmu_user_chunk_sizes(uvm_gpu_t *gpu)
{
    // TODO: Only use color page size when need to allocate colored page
    uvm_chunk_sizes_mask_t sizes;
    
    sizes = page_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_64K)  |
                                page_sizes_for_big_page_size(gpu, UVM_PAGE_SIZE_128K) |
                                PAGE_SIZE;

    // If coloring is supported, force the maximum page size to be chunk size
    if (uvm_gpu_supports_coloring(gpu)) {

        sizes &= (gpu->colored_allocation_chunk_size << 1) - 1;
    }

    // Although we may have to map PTEs smaller than PAGE_SIZE, user (managed)
    // memory is never allocated with granularity smaller than PAGE_SIZE. Force
    // PAGE_SIZE to be supported and the smallest allowed size so we don't have
    // to handle allocating multiple chunks per page.
    return sizes & PAGE_MASK;
}

static size_t range_vec_calc_range_count(uvm_page_table_range_vec_t *range_vec)
{
    NvU64 pde_coverage = uvm_mmu_pde_coverage(range_vec->tree, range_vec->page_size);
    NvU64 aligned_start = UVM_ALIGN_DOWN(range_vec->start, pde_coverage);
    NvU64 aligned_end = UVM_ALIGN_UP(range_vec->start + range_vec->size, pde_coverage);
    size_t count = uvm_div_pow2_64(aligned_end - aligned_start, pde_coverage);

    UVM_ASSERT(count != 0);

    return count;
}

static NvU64 range_vec_calc_range_start(uvm_page_table_range_vec_t *range_vec, size_t i)
{
    NvU64 pde_coverage = uvm_mmu_pde_coverage(range_vec->tree, range_vec->page_size);
    NvU64 aligned_start = UVM_ALIGN_DOWN(range_vec->start, pde_coverage);
    NvU64 range_start = aligned_start + i * pde_coverage;
    return max(range_vec->start, range_start);
}

static NvU64 range_vec_calc_range_end(uvm_page_table_range_vec_t *range_vec, size_t i)
{
    NvU64 pde_coverage = uvm_mmu_pde_coverage(range_vec->tree, range_vec->page_size);
    NvU64 range_start = range_vec_calc_range_start(range_vec, i);
    NvU64 max_range_end = UVM_ALIGN_UP(range_start + 1, pde_coverage);
    return min(range_vec->start + range_vec->size, max_range_end);
}

static NvU64 range_vec_calc_range_size(uvm_page_table_range_vec_t *range_vec, size_t i)
{
    return range_vec_calc_range_end(range_vec, i) - range_vec_calc_range_start(range_vec, i);
}

NV_STATUS uvm_page_table_range_vec_init(uvm_page_tree_t *tree, NvU64 start, NvU64 size, NvU32 page_size,
        uvm_page_table_range_vec_t *range_vec)
{
    NV_STATUS status;
    size_t i;

    UVM_ASSERT(size != 0);
    UVM_ASSERT_MSG(IS_ALIGNED(start, page_size), "start 0x%llx page_size 0x%x\n", start, page_size);
    UVM_ASSERT_MSG(IS_ALIGNED(size, page_size), "size 0x%llx page_size 0x%x\n", size, page_size);

    range_vec->tree = tree;
    range_vec->page_size = page_size;
    range_vec->start = start;
    range_vec->size = size;
    range_vec->range_count = range_vec_calc_range_count(range_vec);

    range_vec->ranges = uvm_kvmalloc_zero(sizeof(*range_vec->ranges) * range_vec->range_count);
    if (!range_vec->ranges) {
        status = NV_ERR_NO_MEMORY;
        goto error;
    }

    for (i = 0; i < range_vec->range_count; ++i) {
        uvm_page_table_range_t *range = &range_vec->ranges[i];

        NvU64 range_start = range_vec_calc_range_start(range_vec, i);
        NvU64 range_size = range_vec_calc_range_size(range_vec, i);

        status = uvm_page_tree_get_ptes_async(tree, page_size, range_start, range_size, UVM_PMM_ALLOC_FLAGS_EVICT, range);
        if (status != NV_OK) {
            UVM_ERR_PRINT("Failed to get PTEs for subrange %zd [0x%llx, 0x%llx) size 0x%llx, part of [0x%llx, 0x%llx)\n",
                    i, range_start, range_start + range_size, range_size,
                    start, size);
            goto error;
        }
    }
    return uvm_page_tree_wait(tree);

error:
    uvm_page_table_range_vec_deinit(range_vec);
    return status;
}

NV_STATUS uvm_page_table_range_vec_create(uvm_page_tree_t *tree, NvU64 start, NvU64 size, NvU32 page_size,
        uvm_page_table_range_vec_t **range_vec_out)
{
    NV_STATUS status;
    uvm_page_table_range_vec_t *range_vec;

    range_vec = uvm_kvmalloc(sizeof(*range_vec));
    if (!range_vec)
        return NV_ERR_NO_MEMORY;

    status = uvm_page_table_range_vec_init(tree, start, size, page_size, range_vec);
    if (status != NV_OK)
        goto error;

    *range_vec_out = range_vec;

    return NV_OK;

error:
    uvm_kvfree(range_vec);
    return status;
}

NV_STATUS uvm_page_table_range_vec_clear_ptes(uvm_page_table_range_vec_t *range_vec, uvm_membar_t tlb_membar)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;
    size_t i;
    uvm_page_tree_t *tree = range_vec->tree;
    uvm_gpu_t *gpu = tree->gpu;
    NvU32 page_size = range_vec->page_size;
    NvU32 entry_size = uvm_mmu_pte_size(tree, page_size);
    NvU64 invalid_pte = 0;
    uvm_push_t push;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();

    uvm_pte_batch_t pte_batch;

    UVM_ASSERT(range_vec);
    UVM_ASSERT(tree);
    UVM_ASSERT(gpu);

    i = 0;
    while (i < range_vec->range_count) {
        // Acquiring the previous push is not necessary for correctness as all
        // the memsets can be done independently, but scheduling a lot of
        // independent work for a big range could end up hogging the GPU
        // for a long time while not providing much improvement.
        status = page_tree_begin_acquire(tree, &tracker, &push, "Clearing PTEs for [0x%llx, 0x%llx)",
                    range_vec->start, range_vec->start + range_vec->size);

        if (status != NV_OK)
            goto done;

        uvm_pte_batch_begin(&push, &pte_batch);

        for (; i < range_vec->range_count; ++i) {
            uvm_page_table_range_t *range = &range_vec->ranges[i];
            uvm_gpu_phys_address_t first_entry_pa = uvm_page_table_range_entry_address(tree, range, 0);
            uvm_pte_batch_clear_ptes(&pte_batch, first_entry_pa, invalid_pte, entry_size, range->entry_count);

            if (!uvm_push_has_space(&push, 512)) {
                // Stop pushing the clears once we get close to a full push
                break;
            }
        }

        uvm_pte_batch_end(&pte_batch);

        if (i == range_vec->range_count)
            uvm_tlb_batch_single_invalidate(tree, &push, range_vec->start, range_vec->size, page_size, tlb_membar);

        page_tree_end(tree, &push);

        // Skip the tracking if in unit test mode
        if (!tree->gpu->channel_manager)
            continue;

        // The push acquired the tracker so it's ok to just overwrite it with
        // the entry tracking the push.
        uvm_tracker_overwrite_with_push(&tracker, &push);
    }

done:
    tracker_status = uvm_tracker_wait_deinit(&tracker);
    if (status == NV_OK)
        status = tracker_status;

    return status;
}

void uvm_page_table_range_vec_deinit(uvm_page_table_range_vec_t *range_vec)
{
    size_t i;
    if (!range_vec)
        return;

    if (range_vec->ranges) {
        for (i = 0; i < range_vec->range_count; ++i) {
            uvm_page_table_range_t *range = &range_vec->ranges[i];
            if (!range->entry_count)
                break;
            uvm_page_tree_put_ptes_async(range_vec->tree, range);
        }
        (void)uvm_page_tree_wait(range_vec->tree);

        uvm_kvfree(range_vec->ranges);
    }

    memset(range_vec, 0, sizeof(*range_vec));
}

void uvm_page_table_range_vec_destroy(uvm_page_table_range_vec_t *range_vec)
{
    if (!range_vec)
        return;

    uvm_page_table_range_vec_deinit(range_vec);

    uvm_kvfree(range_vec);
}

NV_STATUS uvm_page_table_range_vec_write_ptes(uvm_page_table_range_vec_t *range_vec, uvm_membar_t tlb_membar,
        uvm_page_table_range_pte_maker_t pte_maker, void *caller_data)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;
    NvU32 entry;
    size_t i;
    uvm_page_tree_t *tree = range_vec->tree;
    uvm_gpu_t *gpu = tree->gpu;
    NvU32 entry_size = uvm_mmu_pte_size(tree, range_vec->page_size);

    uvm_push_t push;
    uvm_pte_batch_t pte_batch;
    NvU64 offset = 0;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    bool last_push = false;

    // Use as much push space as possible leaving 1K of margin
    static const NvU32 max_total_entry_size_per_push = UVM_MAX_PUSH_SIZE - 1024;

    NvU32 max_entries_per_push = max_total_entry_size_per_push / entry_size;

    for (i = 0; i < range_vec->range_count; ++i) {
        uvm_page_table_range_t *range = &range_vec->ranges[i];
        NvU64 range_start = range_vec_calc_range_start(range_vec, i);
        NvU64 range_size = range_vec_calc_range_size(range_vec, i);
        uvm_gpu_phys_address_t entry_addr = uvm_page_table_range_entry_address(tree, range, 0);
        entry = 0;

        while (entry < range->entry_count) {
            NvU32 entry_limit_this_push = min(range->entry_count, entry + max_entries_per_push);

            // Acquiring the previous push is not necessary for correctness as all
            // the PTE writes can be done independently, but scheduling a lot of
            // independent work for a big range could end up hogging the GPU
            // for a long time while not providing much improvement.
            status = page_tree_begin_acquire(tree, &tracker, &push,
                    "Writing PTEs for range at [0x%llx, 0x%llx), subrange of range vec at [0x%llx, 0x%llx)",
                    range_start, range_start + range_size,
                    range_vec->start, range_vec->start + range_vec->size);
            if (status != NV_OK) {
                UVM_ERR_PRINT("Failed to begin push for writing PTEs: %s GPU %s\n", nvstatusToString(status), gpu->name);
                goto done;
            }

            uvm_pte_batch_begin(&push, &pte_batch);

            for (; entry < entry_limit_this_push; ++entry) {
                NvU64 pte_bits = pte_maker(range_vec, offset, caller_data);
                uvm_pte_batch_write_pte(&pte_batch, entry_addr, pte_bits, entry_size);
                offset += range_vec->page_size;
                entry_addr.address += entry_size;
            }

            last_push = (i == range_vec->range_count - 1) && entry == range->entry_count;

            uvm_pte_batch_end(&pte_batch);

            if (last_push) {
                // Invalidate TLBs as part of the last push
                uvm_tlb_batch_single_invalidate(tree, &push,
                        range_vec->start, range_vec->size, range_vec->page_size, tlb_membar);
            }
            else {
                // For pushes prior to the last one, uvm_pte_batch_end() has
                // already pushed a membar that's enough to order the PTE writes
                // with the TLB invalidate in the last push and that's all
                // that's needed.
                uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
            }

            page_tree_end(tree, &push);

            // Skip the tracking if in unit test mode
            if (!tree->gpu->channel_manager)
                continue;

            // The push acquired the tracker so it's ok to just overwrite it with
            // the entry tracking the push.
            uvm_tracker_overwrite_with_push(&tracker, &push);
        }
    }

done:
    tracker_status = uvm_tracker_wait_deinit(&tracker);
    if (status == NV_OK)
        status = tracker_status;
    return status;
}

static NvU64 identity_mapping_pte_maker(uvm_page_table_range_vec_t *range_vec, NvU64 offset, void *data)
{
    uvm_aperture_t aperture = *(uvm_aperture_t*)data;
    bool is_vol = (aperture != UVM_APERTURE_VID);
    return range_vec->tree->hal->make_pte(aperture, offset, UVM_PROT_READ_WRITE_ATOMIC, is_vol, range_vec->page_size);
}

static NV_STATUS create_identity_mapping(uvm_gpu_t *gpu, NvU64 base, NvU64 size, uvm_aperture_t aperture, NvU32 page_size,
        uvm_page_table_range_vec_t **range_vec)
{
    NV_STATUS status;

    status = uvm_page_table_range_vec_create(&gpu->address_space_tree, base, size, page_size, range_vec);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init range vec for aperture %d identity mapping at [0x%llx, 0x%llx): %s, GPU %s\n",
                aperture, base, base + size, nvstatusToString(status), gpu->name);
        return status;
    }

    status = uvm_page_table_range_vec_write_ptes(*range_vec, UVM_MEMBAR_NONE, identity_mapping_pte_maker, &aperture);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to write PTEs for aperture %d identity mapping at [0x%llx, 0x%llx): %s, GPU %s\n",
                aperture, base, base + size, nvstatusToString(status), gpu->name);
        return status;
    }

    return NV_OK;
}

static NV_STATUS create_big_page_identity_mapping(uvm_gpu_t *gpu)
{
    NvU64 base = gpu->big_page.identity_mapping.base;
    NvU32 big_page_size = gpu->big_page.internal_size;
    NvU64 size = UVM_ALIGN_UP(gpu->vidmem_max_allocatable_address + 1, big_page_size);

    UVM_ASSERT(gpu->big_page.swizzling);
    UVM_ASSERT(base);
    UVM_ASSERT(size);
    UVM_ASSERT(big_page_size);

    return create_identity_mapping(gpu, base, size, UVM_APERTURE_VID, big_page_size, &gpu->big_page.identity_mapping.range_vec);
}

NV_STATUS uvm_mmu_create_big_page_identity_mappings(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    if (gpu->big_page.swizzling)
        status = create_big_page_identity_mapping(gpu);

    return status;
}

NV_STATUS uvm_mmu_create_peer_identity_mappings(uvm_gpu_t *gpu, uvm_gpu_t *peer)
{
    NvU32 page_size = uvm_mmu_biggest_page_size(&gpu->address_space_tree);
    NvU64 size = UVM_ALIGN_UP(peer->vidmem_max_allocatable_address + 1, page_size);
    uvm_aperture_t aperture = uvm_gpu_peer_aperture(gpu, peer);
    NvU32 peer_id = UVM_APERTURE_PEER_ID(aperture);
    NvU64 base = gpu->peer_mappings[peer_id].base;

    if (!gpu->peer_identity_mappings_supported)
        return NV_OK;

    UVM_ASSERT(page_size);
    UVM_ASSERT(size);
    UVM_ASSERT(size <= UVM_PEER_IDENTITY_VA_SIZE);
    UVM_ASSERT(base);

    return create_identity_mapping(gpu,
                                   base,
                                   size,
                                   aperture,
                                   page_size,
                                   &gpu->peer_mappings[peer_id].range_vec);
}

void uvm_mmu_destroy_big_page_identity_mappings(uvm_gpu_t *gpu)
{
    if (gpu->big_page.identity_mapping.range_vec) {
        // The self identity mappings point to local GPU memory and hence can
        // use a GPU membar for the invalidates.
        (void)uvm_page_table_range_vec_clear_ptes(gpu->big_page.identity_mapping.range_vec, UVM_MEMBAR_GPU);
    }

    uvm_page_table_range_vec_destroy(gpu->big_page.identity_mapping.range_vec);
    gpu->big_page.identity_mapping.range_vec = NULL;
}

void uvm_mmu_destroy_peer_identity_mappings(uvm_gpu_t *gpu, uvm_gpu_t *peer)
{
    if (gpu->peer_identity_mappings_supported) {
        uvm_gpu_identity_mapping_t *mapping = gpu->peer_mappings + UVM_APERTURE_PEER_ID(uvm_gpu_peer_aperture(gpu, peer));

        if (mapping->range_vec)
            (void)uvm_page_table_range_vec_clear_ptes(mapping->range_vec, UVM_MEMBAR_SYS);

        uvm_page_table_range_vec_destroy(mapping->range_vec);
        mapping->range_vec = NULL;
    }
}

uvm_gpu_address_t uvm_mmu_gpu_address_for_big_page_physical(uvm_gpu_address_t physical, uvm_gpu_t *gpu)
{
    UVM_ASSERT(!physical.is_virtual);
    UVM_ASSERT(physical.aperture == UVM_APERTURE_VID);

    if (gpu->big_page.swizzling) {
        UVM_ASSERT(gpu->big_page.identity_mapping.range_vec != NULL);
        return uvm_gpu_address_virtual(gpu->big_page.identity_mapping.base + physical.address);
    }
    return physical;
}

void uvm_mmu_init_gpu_peer_addresses(uvm_gpu_t *gpu)
{
    NvU32 i;
    BUILD_BUG_ON(UVM_APERTURE_PEER_0 != 0);
    if (gpu->peer_identity_mappings_supported) {
        for (i = UVM_APERTURE_PEER_0; i < UVM_APERTURE_PEER_MAX; i++) {
            gpu->peer_mappings[i].base = gpu->rm_va_base +
                                         gpu->rm_va_size +
                                         UVM_PEER_IDENTITY_VA_SIZE *
                                         UVM_APERTURE_PEER_ID(i);
        }
    }

    UVM_ASSERT(gpu->uvm_mem_va_base >= gpu->peer_mappings[UVM_APERTURE_PEER_MAX - 1].base + UVM_PEER_IDENTITY_VA_SIZE);
}

NV_STATUS uvm8_test_invalidate_tlb(UVM_TEST_INVALIDATE_TLB_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = NULL;
    uvm_push_t push;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_va_space_t *gpu_va_space;

    // Check parameter values
    if (params->membar < UvmInvalidateTlbMemBarNone ||
        params->membar > UvmInvalidateTlbMemBarLocal) {
        return NV_ERR_INVALID_PARAMETER;
    }

    if (params->target_va_mode < UvmTargetVaModeAll ||
        params->target_va_mode > UvmTargetVaModeTargeted) {
        return NV_ERR_INVALID_PARAMETER;
    }

    if (params->target_pdb_mode < UvmTargetPdbModeAll ||
        params->target_pdb_mode > UvmTargetPdbModeOne) {
        return NV_ERR_INVALID_PARAMETER;
    }

    if (params->page_table_level < UvmInvalidatePageTableLevelAll ||
        params->page_table_level > UvmInvalidatePageTableLevelPde3) {
        return NV_ERR_INVALID_PARAMETER;
    }

    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto unlock_exit;
    }

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    UVM_ASSERT(gpu_va_space);

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &push,
                            "Pushing test invalidate, GPU %s", gpu->name);
    if (status == NV_OK)
        gpu->host_hal->tlb_invalidate_test(&push, uvm_page_tree_pdb(&gpu_va_space->page_tables)->addr, params);

unlock_exit:
    // Wait for the invalidation to be performed
    if (status == NV_OK)
        status = uvm_push_end_and_wait(&push);

    uvm_va_space_up_read(va_space);

    return status;
}
