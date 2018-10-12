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

#include "uvm8_hal.h"
#include "uvm8_gpu.h"
#include "uvm8_mem.h"
#include "uvm8_volta_fault_buffer.h"

void uvm_hal_volta_arch_init_properties(uvm_gpu_t *gpu)
{
    gpu->big_page.swizzling = false;

    gpu->tlb_batch.va_invalidate_supported = true;

    gpu->tlb_batch.va_range_invalidate_supported = true;

    // TODO: Bug 1767241: Run benchmarks to figure out a good number
    gpu->tlb_batch.max_ranges = 8;

    gpu->fault_buffer_info.replayable.utlb_count = gpu->gpc_count * uvm_volta_get_utlbs_per_gpc(gpu);
    {
        uvm_fault_buffer_entry_t *dummy;
        UVM_ASSERT(gpu->fault_buffer_info.replayable.utlb_count <= (1 << (sizeof(dummy->fault_source.utlb_id) * 8)));
    }

    // A single top level PDE on Volta covers 128 TB and that's the minimum
    // size that can be used.
    gpu->rm_va_base = 0;
    gpu->rm_va_size = 128ull * 1024 * 1024 * 1024 * 1024;

    gpu->uvm_mem_va_base = 384ull * 1024 * 1024 * 1024 * 1024;
    gpu->uvm_mem_va_size = UVM_MEM_VA_SIZE;

    gpu->peer_identity_mappings_supported = true;

    // Not all units on Volta support 49-bit addressing, including those which
    // access channel buffers.
    gpu->max_channel_va = 1ULL << 40;

    // Volta can map sysmem with any page size
    gpu->can_map_sysmem_with_large_pages = true;

    // Prefetch instructions will generate faults
    gpu->prefetch_fault_supported = true;

    // Pascal and Volta require post-invalidate membars to flush out HSHUB. See
    // bug 1975028. All GV100-class chips supported by UVM have HSHUB.
    UVM_ASSERT(gpu->architecture == NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100);
    gpu->num_hshub_tlb_invalidate_membars = 2;

    // Volta can place GPFIFO in vidmem
    gpu->gpfifo_in_vidmem_supported = true;

    gpu->replayable_faults_supported = true;

    gpu->non_replayable_faults_supported = true;

    gpu->access_counters_supported = true;

    gpu->fault_cancel_va_supported = true;

#if defined(UVM_MEM_COLORING)

   // During testing, only one color is needed. We just need contiguous phy memory.
   // For userspace coloring, during allocation we just need contiguous memory.
   // But during transfer memory, we need to be color aware.
   // For kernel coloring, everythng is transparent to userspace application and hence
   // it needs to be color aware.
#if defined(UVM_TEST_MEM_COLORING)

    gpu->num_allocation_mem_colors = 1;
    gpu->num_transfer_mem_colors = 1;
    gpu->colored_allocation_chunk_size = UVM_PAGE_SIZE_2M;
    gpu->colored_transfer_chunk_size = UVM_PAGE_SIZE_2M;

#elif defined(UVM_USER_MEM_COLORING)

    gpu->num_allocation_mem_colors = 1;
    gpu->num_transfer_mem_colors = 2;
    gpu->colored_allocation_chunk_size = UVM_PAGE_SIZE_2M;
    gpu->colored_transfer_chunk_size = UVM_PAGE_SIZE_4K;

#else /* Kernel coloring */
    /* 
     * Even though V100 seem to have more than 2 colors, 
     * with one color having more than 64K page size,
     * for now we stay with 2 and 4K pages
     */
    gpu->num_allocation_mem_colors = 2;
    gpu->num_transfer_mem_colors = 2;
    gpu->colored_allocation_chunk_size = UVM_PAGE_SIZE_4K;
    gpu->colored_transfer_chunk_size = UVM_PAGE_SIZE_4K;

#endif

#else
    gpu->num_allocation_mem_colors = 0;
    gpu->num_transfer_mem_colors = 0;
#endif
}
