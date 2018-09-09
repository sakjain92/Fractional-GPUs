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

void uvm_hal_kepler_arch_init_properties(uvm_gpu_t *gpu)
{
    gpu->big_page.swizzling = true;

    // 128 GB should be enough for all current RM allocations and leaves enough
    // space for UVM internal mappings.
    // A single top level PDE covers 64 or 128 MB on Kepler so 128 GB is fine to use.
    gpu->rm_va_base = 0;
    gpu->rm_va_size = 128ull * 1024 * 1024 * 1024;

    gpu->big_page.identity_mapping.base = gpu->rm_va_base + gpu->rm_va_size;

    gpu->tlb_batch.va_invalidate_supported = false;

    gpu->uvm_mem_va_base = 768ull * 1024 * 1024 * 1024;
    gpu->uvm_mem_va_size = UVM_MEM_VA_SIZE;

    // We don't have a compelling use case in UVM-Lite for direct peer
    // migrations between GPUs. Additionally, supporting this would be
    // problematic with big page swizzling because we'd have to create two
    // identity mappings, one for big and one for 4k. The 4k mappings would eat
    // up lots of memory for page tables so just don't bother.
    gpu->peer_identity_mappings_supported = false;

    gpu->max_channel_va = 1ULL << 40;

    // Kepler can only map sysmem with 4K pages
    gpu->can_map_sysmem_with_large_pages = false;

    // Kepler cannot place GPFIFO in vidmem
    gpu->gpfifo_in_vidmem_supported = false;

    gpu->replayable_faults_supported = false;

    gpu->non_replayable_faults_supported = false;

    gpu->access_counters_supported = false;

    gpu->fault_cancel_va_supported = false;

    gpu->num_allocation_mem_colors = 0;
    gpu->num_transfer_mem_colors = 0;
}
