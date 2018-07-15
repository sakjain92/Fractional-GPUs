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

#ifndef __UVM8_MEM_H__
#define __UVM8_MEM_H__

#include "uvm8_forward_decl.h"
#include "uvm8_processors.h"
#include "uvm8_hal_types.h"
#include "uvm8_pmm_gpu.h"
#include "uvm8_range_allocator.h"

//
// This module provides an abstraction for UVM-managed allocations, both sysmem
// and vidmem, which can be mapped on GPUs in internal or user VA spaces or on
// the CPU, or accessed physically.
//
// As opposed to the uvm_rm_mem_* abstraction, this module has no dependencies
// on the UVM-RM interface and implements all the functionality on top of other
// UVM abstractions. Specifically, vidmem is allocated from PMM and sysmem is
// allocated directly from the kernel (in the future PMM will support sysmem as
// well and then this module can switch over). And GPU mappings are created
// through the page table range vector (uvm_page_table_range_vec_t) and CPU
// mappings (only sysmem) use vmap directly.
//
// The module currently allows the following:
//  - sysmem allocation and mapping on all GPUs and the CPU
//  - vidmem allocation and mapping on the GPU backing the allocation
//
// Additionally, helpers for accessing the allocations physically are provided,
// which allows skipping virtual mappings if not necessary (e.g. allocating a
// single CPU page and accessing it from the GPU).
//
// For internal mappings, GPU VA ranges used for mapping the allocations are
// allocated from a global range allocator (uvm_range_allocator_t) and are
// currently offset by a GPU specific offset (gpu->uvm_mem_va_base). This would
// change if the first limitation below is lifted and UVM can control the VA
// starting at 0. For user mappings, a fixed VA is provided externally.
//
// The lifetime of vidmem allocations cannot exceed the lifetime of the GPU
// (uvm_gpu_t) it's allocated on, but sysmem allocations can have module
// lifetime.
//
// Future additions:
//  - Per processor caching attributes (longer term, the envisioned use-case is
//    for GPU semaphore caching, which requires the first limitation below to be
//    lifted)
//
// Limitations:
//  - On Pascal+ limited to VAs over 40bit due to how the internal VA is shared
//    with RM. This implies it cannot be used for e.g. pushbuffer nor sempahores
//    currently. At some point in the future UVM should be able
//    to take full control of the VA (or at least the bottom 40bits of it)
//    and this limitation would be lifted. See comments around
//    gpu->rm_va_base for more details.
//  - Mapping vidmem is not possible on the CPU. Lifting this limitation would
//    need to be investigated if it's desired. The problem is that the BAR1
//    space (that's used for such mappings) is limited and controlled by RM and
//    may not be easy to interop with vidmem allocations from PMM.
//


// The size of the VA used for mapping uvm_mem_t allocations
// 128 GBs should be plenty for internal allocations and fits easily on all
// supported architectures.
#define UVM_MEM_VA_SIZE (128ull * 1024 * 1024 * 1024)

typedef struct
{
    // The GPU to allocate memory from, or NULL for sysmem.
    uvm_gpu_t *backing_gpu;

    // Size of the allocation, in bytes.
    // The only restriction is for it to be non-0.
    NvU64 size;

    // Desired page size to use, in bytes.
    //
    // If this is a CPU allocation, the physical allocation chunk has to be
    // aligned to PAGE_SIZE and the allocation will be mapped with the largest
    // PTEs possible on the GPUs. If set to UVM_PAGE_SIZE_DEFAULT, PAGE_SIZE
    // size will be used.
    //
    // For a GPU allocation, if set to UVM_PAGE_SIZE_DEFAULT, GPU mappings will
    // use the largest page size supported by the backing GPU which is not
    // larger than size. Otherwise, the desired page size will be used.
    //
    // CPU mappings will always use PAGE_SIZE, so the physical allocation chunk
    // has to be aligned to PAGE_SIZE.
    NvU32 page_size;

    // The user VA space to map the allocation in, or NULL to map internally.
    uvm_va_space_t *user_va_space;

    // The address to map at for user mappings. Unused for internal mappings.
    void *user_addr;
} uvm_mem_alloc_params_t;

typedef struct
{
    uvm_prot_t protection;
    bool is_volatile;
} uvm_mem_gpu_mapping_attrs_t;

struct uvm_mem_struct
{
    // The GPU the physical memory is allocated on. Or NULL for sysmem.
    //
    // For GPU allocations, the lifetime of the allocation cannot extend the
    // lifetime of the GPU. For CPU allocations there is no lifetime limitation.
    uvm_gpu_t *backing_gpu;

    // Size of the physical chunks.
    NvU32 chunk_size;

    union
    {
        struct
        {
            uvm_gpu_chunk_t **chunks;
        } vidmem;

        struct
        {
            struct page **pages;

            // Per GPU IOMMU mappings of the pages
            NvU64 *dma_addrs[UVM_MAX_GPUS];
        } sysmem;
    };

    // Count of chunks (vidmem) or CPU pages (sysmem) above
    size_t chunks_count;

    // Size of the allocation
    NvU64 size;

    // Size of the physical allocation backing
    NvU64 physical_allocation_size;

    // The backing may be mapped in either UVM's internal VA space or a user
    // VA space, but not both.
    bool is_user_allocation;

    // Mask of processors the memory is mapped on
    uvm_processor_mask_t mapped_phys_on;
    uvm_processor_mask_t mapped_on;

    // Page table ranges for all GPUs
    uvm_page_table_range_vec_t *range_vecs[UVM_MAX_GPUS];

    union {
        // Information specific to allocations mapped in UVM internal VA space.
        struct {
            // Range allocation for the GPU VA, allocated at uvm_mem_alloc() and
            // persisting until uvm_mem_free().
            uvm_range_allocation_t range_alloc;

            // CPU address of the allocation if mapped on the CPU
            void *cpu_addr;
        } kernel;

        // Information specific to allocations mapped in a user VA space.
        struct {
            uvm_va_space_t *va_space;

            // The VA to map the allocation at on all processors
            void *addr;
        } user;
    };
};

NV_STATUS uvm_mem_global_init(void);
void uvm_mem_global_exit(void);

// Fill out attrs_out from attrs.  attrs_out must not be null.  attrs_out may be prepopulated
// with default values, which are not overwritten if the corresponding field in attrs has a
// default value.  The gpu corresponding to attrs->gpuUuid is optionally returned in gpu_out
// if it is not NULL.
// Returns an error if attrs is invalid.
NV_STATUS uvm_mem_translate_gpu_attributes(UvmGpuMappingAttributes *attrs,
        uvm_va_space_t *va_space, uvm_gpu_t **gpu_out, uvm_mem_gpu_mapping_attrs_t *attrs_out);

uvm_chunk_sizes_mask_t uvm_mem_kernel_chunk_sizes(uvm_gpu_t *gpu);

// Allocate memory described by the mem_desc
//
// See comments for uvm_mem_alloc_params_t for details of the parameters.
// The memory is immediately accessible physically:
//  - sysmem from any GPU
//  - vidmem from the GPU backing the allocation
//
// The memory may be mapped in either UVM's internal VA space or a user VA space.
// va_space represents the user VA space to create mappings in. A null value means
// to create internal mappings. This function does not create the mappings.
//
// The allocation can be mapped with uvm_mem_map_*() functions below.
NV_STATUS uvm_mem_alloc(uvm_mem_alloc_params_t *params, uvm_mem_t **mem_out);

// Map the allocation on the specified processors in the UVM VA space.
NV_STATUS uvm_mem_map_kernel(uvm_mem_t *mem, uvm_processor_mask_t *mask);

// Helper for allocating sysmem
static NV_STATUS uvm_mem_alloc_sysmem(NvU64 size, uvm_mem_t **mem_out)
{
    uvm_mem_alloc_params_t params = { 0 };
    params.size = size;
    params.backing_gpu = NULL;
    params.page_size = UVM_PAGE_SIZE_DEFAULT;

    return uvm_mem_alloc(&params, mem_out);
}

// Helper for allocating vidmem with the default page size
static NV_STATUS uvm_mem_alloc_vidmem(NvU64 size, uvm_gpu_t *gpu, uvm_mem_t **mem_out)
{
    uvm_mem_alloc_params_t params = { 0 };
    params.size = size;
    params.backing_gpu = gpu;
    params.page_size = UVM_PAGE_SIZE_DEFAULT;

    return uvm_mem_alloc(&params, mem_out);
}

// Free the memory.
// Clear all mappings and free the memory
void uvm_mem_free(uvm_mem_t *mem);

// Map/Unmap on the CPU
//
// For user-mapped memory, the mapping will be under 'vma'. 'vma' is unused for
// kernel-mapped (internal) memory.
NV_STATUS uvm_mem_map_cpu(uvm_mem_t *mem, struct vm_area_struct *vma);
void uvm_mem_unmap_cpu(uvm_mem_t *mem);

// Map/Unmap sysmem for physical access on a GPU
//
//
NV_STATUS uvm_mem_map_gpu_phys(uvm_mem_t *mem, uvm_gpu_t *gpu);
void uvm_mem_unmap_gpu_phys(uvm_mem_t *mem, uvm_gpu_t *gpu);

// Map/Unmap on a GPU
//
// Mapping is supported for:
//  - sysmem on any GPU
//  - vidmem on the GPU backing the allocation
//
// Notably the allocation has to be unmapped from any GPUs before they are
// released by the UVM driver.
NV_STATUS uvm_mem_map_gpu_kernel(uvm_mem_t *mem, uvm_gpu_t *gpu);
NV_STATUS uvm_mem_map_gpu_user(uvm_mem_t *mem, uvm_gpu_t *gpu, uvm_mem_gpu_mapping_attrs_t *attrs);
void uvm_mem_unmap_gpu(uvm_mem_t *mem, uvm_gpu_t *gpu);

// Get the CPU address
//
// The allocation has to be mapped on the CPU prior to calling this function.
void *uvm_mem_get_cpu_addr_kernel(uvm_mem_t *mem);

// Get the GPU VA
//
// The allocation has to be mapped on the given GPU prior to calling this function.
NvU64 uvm_mem_get_gpu_va_kernel(uvm_mem_t *mem, uvm_gpu_t *gpu);

// Helper for getting a virtual uvm_gpu_address_t
uvm_gpu_address_t uvm_mem_gpu_address_virtual_kernel(uvm_mem_t *mem, uvm_gpu_t *gpu);

// Helpers for getting both types of GPU physical addresses.
//
// Offset and size are used to return the address of the correct physical chunk
// and check that the allocation is physically contiguous for the given range.
uvm_gpu_phys_address_t uvm_mem_gpu_physical(uvm_mem_t *mem, uvm_gpu_t *gpu, NvU64 offset, NvU64 size);
uvm_gpu_address_t uvm_mem_gpu_address_physical(uvm_mem_t *mem, uvm_gpu_t *gpu, NvU64 offset, NvU64 size);

// Helper to get an address suitable for accessing_gpu (which may be the backing
// GPU) to access with CE. Note that mappings for indirect peers are not
// created automatically.
uvm_gpu_address_t uvm_mem_gpu_address_copy(uvm_mem_t *mem, uvm_gpu_t *accessing_gpu, NvU64 offset, NvU64 size);

static bool uvm_mem_is_sysmem(uvm_mem_t *mem)
{
    return mem->backing_gpu == NULL;
}

static bool uvm_mem_is_vidmem(uvm_mem_t *mem)
{
    return !uvm_mem_is_sysmem(mem);
}

static bool uvm_mem_is_local_vidmem(uvm_mem_t *mem, uvm_gpu_t *gpu)
{
    return mem->backing_gpu == gpu;
}

// Helper for allocating sysmem and mapping it on the CPU
static NV_STATUS uvm_mem_alloc_sysmem_and_map_cpu_kernel(NvU64 size, uvm_mem_t **mem_out)
{
    NV_STATUS status;
    uvm_mem_t *mem;
    uvm_mem_alloc_params_t params = { 0 };

    params.backing_gpu = NULL;
    params.size = size;
    status = uvm_mem_alloc(&params, &mem);
    if (status != NV_OK)
        return status;
    status = uvm_mem_map_cpu(mem, NULL);
    if (status != NV_OK) {
        uvm_mem_free(mem);
        return status;
    }

    *mem_out = mem;
    return NV_OK;
}

#endif // __UVM8_MEM_H__
