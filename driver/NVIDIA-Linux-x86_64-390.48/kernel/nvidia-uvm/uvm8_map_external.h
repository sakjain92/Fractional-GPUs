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

#ifndef __UVM8_MAP_EXTERNAL_H__
#define __UVM8_MAP_EXTERNAL_H__

#include "uvm8_forward_decl.h"
#include "uvm8_va_range.h"
#include "uvm8_tracker.h"
#include "nv_uvm_types.h"
#include "uvmtypes.h"

typedef struct
{
    NvU64 map_offset;
    UvmGpuMappingType mapping_type;
    UvmGpuCachingType caching_type;
    uvm_tracker_t *tracker;
} uvm_map_rm_params_t;


// User-facing APIs (uvm_api_map_external_allocation, uvm_api_free) are declared
// uvm8_api.h.

// Queries RM for the PTEs appropriate to the VA range and mem_info, allocates
// page tables for the VA range, and writes the PTEs.
//
// va_range must have type UVM_VA_RANGE_TYPE_EXTERNAL or
// UVM_VA_RANGE_TYPE_CHANNEL. The allocation descriptor given to RM is looked up
// from the VA range.
//
// This does not wait for the PTE writes to complete. The work is added to the
// tracker in map_rm_params.
NV_STATUS uvm_va_range_map_rm_allocation(uvm_va_range_t *va_range,
                                         uvm_gpu_t *mapping_gpu,
                                         UvmGpuMemoryInfo *mem_info,
                                         uvm_map_rm_params_t *map_rm_params);

// Removes and frees the external mapping for mapping_gpu from va_range. If
// deferred_free_list is NULL, the RM handle is freed immediately by this
// function. Otherwise the GPU which owns the allocation (if any) is retained
// and the handle is added to the list for later processing by
// uvm_deferred_free_object_list.
//
// The caller is responsible for making sure that mapping_gpu is retained across
// those calls.
void uvm_ext_gpu_map_destroy(uvm_va_range_t *va_range, uvm_gpu_t *mapped_gpu, struct list_head *deferred_free_list);

// Deferred free function which frees the RM handle and the object itself.
void uvm_ext_gpu_map_free(uvm_ext_gpu_map_t *ext_gpu_map);

#endif // __UVM8_MAP_EXTERNAL_H__
