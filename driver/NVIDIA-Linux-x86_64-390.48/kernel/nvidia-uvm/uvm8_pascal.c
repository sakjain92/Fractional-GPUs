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
#include "uvm8_pascal_fault_buffer.h"

static unsigned uvm_force_prefetch_fault_support = 0;
module_param(uvm_force_prefetch_fault_support, uint, S_IRUGO);

void uvm_hal_pascal_arch_init_properties(uvm_gpu_t *gpu)
{
    gpu->big_page.swizzling = false;

    gpu->tlb_batch.va_invalidate_supported = true;

    gpu->tlb_batch.va_range_invalidate_supported = false;

    // TODO: Bug 1767241: Run benchmarks to figure out a good number
    gpu->tlb_batch.max_pages = 32;

    gpu->fault_buffer_info.replayable.utlb_count = gpu->gpc_count * uvm_pascal_get_utlbs_per_gpc(gpu);
    {
        uvm_fault_buffer_entry_t *dummy;
        UVM_ASSERT(gpu->fault_buffer_info.replayable.utlb_count <= (1 << (sizeof(dummy->fault_source.utlb_id) * 8)));
    }

    // A single top level PDE on Pascal covers 128 TB and that's the minimum
    // size that can be used.
    gpu->rm_va_base = 0;
    gpu->rm_va_size = 128ull * 1024 * 1024 * 1024 * 1024;

    gpu->uvm_mem_va_base = 384ull * 1024 * 1024 * 1024 * 1024;
    gpu->uvm_mem_va_size = UVM_MEM_VA_SIZE;

    gpu->peer_identity_mappings_supported = true;

    // Not all units on Pascal support 49-bit addressing, including those which
    // access channel buffers.
    gpu->max_channel_va = 1ULL << 40;

    // Pascal can map sysmem with any page size
    gpu->can_map_sysmem_with_large_pages = true;

    // Prefetch faults are disabled by default in Pascal
    gpu->prefetch_fault_supported = uvm_force_prefetch_fault_support != 0;

    // Pascal and Volta require post-invalidate membars to flush out HSHUB. See
    // bug 1975028. GP10x chips do not have HSHUB, so they don't need any.
    UVM_ASSERT(gpu->architecture == NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100);
    if (gpu->implementation == NV2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP100 ||
        gpu->implementation == NV2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP000)
        gpu->num_hshub_tlb_invalidate_membars = 2;
    else
        gpu->num_hshub_tlb_invalidate_membars = 0;

    // Pascal cannot place GPFIFO in vidmem
    gpu->gpfifo_in_vidmem_supported = false;

    gpu->replayable_faults_supported = true;

    gpu->non_replayable_faults_supported = false;

    gpu->access_counters_supported = false;

    gpu->fault_cancel_va_supported = false;
}
