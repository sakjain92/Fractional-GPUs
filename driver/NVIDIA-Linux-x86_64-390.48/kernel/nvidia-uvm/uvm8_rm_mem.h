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

#ifndef __UVM8_RM_MEM_H__
#define __UVM8_RM_MEM_H__

#include "uvm8_forward_decl.h"
#include "uvm8_processors.h"
#include "uvm8_test_ioctl.h"

typedef enum
{
    UVM_RM_MEM_TYPE_GPU,
    UVM_RM_MEM_TYPE_SYS,
} uvm_rm_mem_type_t;

// Abstraction for memory allocations done through the UVM-RM interface
struct uvm_rm_mem_struct
{
    // Type of the memory
    uvm_rm_mem_type_t type;

    // Mask of processors the memory is mapped on
    uvm_processor_mask_t mapped_on;

    // VAs on all processors, non-0 for processors that it's mapped on
    NvU64 vas[UVM_MAX_PROCESSORS];

    // The GPU the allocation originated from
    uvm_gpu_t *gpu_owner;

    // Size of the allocation
    NvLength size;
};

// Allocate memory of type and size in the GPU's address space.
// The GPU cannot be NULL and the memory is going to mapped on the GPU for the
// lifetime of the allocation. For sysmem allocations other GPUs can have a
// mapping created and removed dynamically with the uvm_rm_mem_(un)map_gpu()
// functions.
//
// Locking:
//  - Internally acquires:
//    - RM API lock
//    - RM GPUs lock
NV_STATUS uvm_rm_mem_alloc(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out);

// Free the memory.
// Clear all mappings and free the memory
//
// Locking same as uvm_rm_mem_alloc()
void uvm_rm_mem_free(uvm_rm_mem_t *rm_mem);

// Map/Unmap on the CPU
// Locking same as uvm_rm_mem_alloc()
NV_STATUS uvm_rm_mem_map_cpu(uvm_rm_mem_t *rm_mem);
void uvm_rm_mem_unmap_cpu(uvm_rm_mem_t *rm_mem);

// Shortcut for uvm_rm_mem_alloc() + uvm_rm_mem_map_cpu().
// The function fails and nothing is allocated if any of the intermediate steps fail.
//
// Locking same as uvm_rm_mem_alloc()
NV_STATUS uvm_rm_mem_alloc_and_map_cpu(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out);

// Shortcut for uvm_rm_mem_alloc_and_map_cpu() + uvm_rm_mem_map_all_gpus()
// The function fails and nothing is allocated if any of the intermediate steps fail.
//
// Locking same as uvm_rm_mem_alloc()
NV_STATUS uvm_rm_mem_alloc_and_map_all(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out);

// Map/Unmap on a GPU
// Supported only for sysmem (UVM_RM_MEM_TYPE_SYS). The GPU has to be different
// from the one the memory was originally allocated for.
//
// Locking same as uvm_rm_mem_alloc()
NV_STATUS uvm_rm_mem_map_gpu(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu);
void uvm_rm_mem_unmap_gpu(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu);

// Map on all GPUs retained by the UVM driver that do not yet have this allocation mapped
//
// Locking same as uvm_rm_mem_alloc()
NV_STATUS uvm_rm_mem_map_all_gpus(uvm_rm_mem_t *rm_mem);

// Get the CPU VA
void *uvm_rm_mem_get_cpu_va(uvm_rm_mem_t *rm_mem);

// Get the GPU VA
NvU64 uvm_rm_mem_get_gpu_va(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu);

#endif // __UVM8_RM_MEM_H__
