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

#ifndef __UVM8_PMM_SYSMEM_H__
#define __UVM8_PMM_SYSMEM_H__

#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm8_forward_decl.h"
#include "uvm8_lock.h"

// Module to keep handle per-GPU mappings to sysmem physical memory. Notably,
// this implements a reverse map of the DMA address to {va_block, virt_addr}.
// This is required by the GPU access counters feature since they may provide a
// physical address in the notification packet (GPA notifications). We use the
// table to obtain the VAs of the memory regions being accessed remotely. The
// reverse map is implemented by a radix tree, which is indexed using the
// DMA address. For now, only PAGE_SIZE translations are supported (i.e. no
// big/huge pages).
//
// TODO: Bug 1995015: add support for physically-contiguous mappings.
struct uvm_pmm_sysmem_mappings_struct
{
    uvm_gpu_t                                      *gpu;

    struct radix_tree_root             reverse_map_tree;

    uvm_spinlock_t                     reverse_map_lock;
};

// See comments in uvm_linux.h
#ifdef NV_RADIX_TREE_REPLACE_SLOT_PRESENT
#define uvm_pmm_sysmem_mappings_indirect_supported() true
#else
#define uvm_pmm_sysmem_mappings_indirect_supported() false
#endif

// Global initialization/exit functions, that need to be called during driver
// initialization/tear-down. These are needed to allocate/free global internal
// data structures.
NV_STATUS uvm_pmm_sysmem_init(void);
void uvm_pmm_sysmem_exit(void);

// Initialize per-GPU sysmem mapping tracking
NV_STATUS uvm_pmm_sysmem_mappings_init(uvm_gpu_t *gpu, uvm_pmm_sysmem_mappings_t *sysmem_mappings);

// Destroy per-GPU sysmem mapping tracking. The caller must ensure that all the
// mappings have been removed before calling this function.
void uvm_pmm_sysmem_mappings_deinit(uvm_pmm_sysmem_mappings_t *sysmem_mappings);

// If the GPU used to initialize sysmem_mappings supports access counters, the
// dma_addr -> {va_block, virt_addr} mapping is inserted in the reverse map.
NV_STATUS uvm_pmm_sysmem_mappings_add_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                  NvU64 dma_addr,
                                                  NvU64 virt_addr,
                                                  NvU64 region_size,
                                                  uvm_va_block_t *va_block,
                                                  uvm_processor_id_t owner);

static NV_STATUS uvm_pmm_sysmem_mappings_add_gpu_chunk_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                               NvU64 dma_addr,
                                                               NvU64 virt_addr,
                                                               NvU64 region_size,
                                                               uvm_va_block_t *va_block,
                                                               uvm_gpu_id_t owner)
{
    if (!uvm_pmm_sysmem_mappings_indirect_supported())
        return NV_OK;

    return uvm_pmm_sysmem_mappings_add_gpu_mapping(sysmem_mappings,
                                                   dma_addr,
                                                   virt_addr,
                                                   region_size,
                                                   va_block,
                                                   owner);
}


// If the GPU used to initialize sysmem_mappings supports access counters, the
// entries for the physical region starting at dma_addr are removed from the
// reverse map.
void uvm_pmm_sysmem_mappings_remove_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr);

static void uvm_pmm_sysmem_mappings_remove_gpu_chunk_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr)
{
    if (uvm_pmm_sysmem_mappings_indirect_supported())
        uvm_pmm_sysmem_mappings_remove_gpu_mapping(sysmem_mappings, dma_addr);
}

// Like uvm_pmm_sysmem_mappings_remove_gpu_mapping but it doesn't assert if the
// mapping doesn't exist. See uvm_va_block_evict_chunks for more information.
void uvm_pmm_sysmem_mappings_remove_gpu_mapping_on_eviction(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr);

// If the GPU used to initialize sysmem_mappings supports access counters, the
// mapping for the region starting at dma_addr is updated with va_block.
// This is required on VA block split.
void uvm_pmm_sysmem_mappings_reparent_gpu_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                  NvU64 dma_addr,
                                                  uvm_va_block_t *va_block);

static void uvm_pmm_sysmem_mappings_reparent_gpu_chunk_mapping(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                               NvU64 dma_addr,
                                                               uvm_va_block_t *va_block)
{
    if (uvm_pmm_sysmem_mappings_indirect_supported())
        uvm_pmm_sysmem_mappings_reparent_gpu_mapping(sysmem_mappings, dma_addr, va_block);
}

// If the GPU used to initialize sysmem_mappings supports access counters, the
// mapping for the region starting at dma_addr is split into regions of
// new_region_size. new_region_size must be a power of two and smaller than the
// previously-registered size.
NV_STATUS uvm_pmm_sysmem_mappings_split_gpu_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                     NvU64 dma_addr,
                                                     NvU64 new_region_size);

static NV_STATUS uvm_pmm_sysmem_mappings_split_gpu_chunk_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                                  NvU64 dma_addr,
                                                                  NvU64 new_region_size)
{
    if (!uvm_pmm_sysmem_mappings_indirect_supported())
        return NV_OK;

    return uvm_pmm_sysmem_mappings_split_gpu_mappings(sysmem_mappings, dma_addr, new_region_size);
}

// If the GPU used to initialize sysmem_mappings supports access counters, all
// the mappings within the region [dma_addr, dma_addr + new_region_size) are
// merged into a single mapping. new_region_size must be a power of two. The
// whole region must be previously populated with mappings and all of them must
// have the same VA block and processor owner.
void uvm_pmm_sysmem_mappings_merge_gpu_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                NvU64 dma_addr,
                                                NvU64 new_region_size);

static void uvm_pmm_sysmem_mappings_merge_gpu_chunk_mappings(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                                             NvU64 dma_addr,
                                                             NvU64 new_region_size)
{
    if (uvm_pmm_sysmem_mappings_indirect_supported())
        uvm_pmm_sysmem_mappings_merge_gpu_mappings(sysmem_mappings, dma_addr, new_region_size);
}

// Obtain the {va_block, virt_addr} information for the mappings in the given
// [dma_addr:dma_addr + region_size) range. dma_addr and region_size must be
// page-aligned.
//
// Valid translations are written to out_mappings sequentially (there are no
// gaps). max_out_mappings are written, at most. The caller is required to
// provide enough entries in out_mappings.
//
// The VA Block in each returned translation entry is retained, and it's up to
// the caller to release them
size_t uvm_pmm_sysmem_mappings_dma_to_virt(uvm_pmm_sysmem_mappings_t *sysmem_mappings,
                                           NvU64 dma_addr,
                                           NvU64 region_size,
                                           uvm_reverse_map_t *out_mappings,
                                           size_t max_out_mappings);

#endif
