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

#include "uvm8_rm_mem.h"
#include "uvm8_test.h"
#include "uvm8_test_ioctl.h"
#include "uvm8_va_space.h"
#include "uvm8_kvmalloc.h"

static NV_STATUS alloc_and_map_cpu(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, size_t size, uvm_rm_mem_t **mem_out)
{
    NV_STATUS status;
    uvm_rm_mem_t *rm_mem;
    void *cpu_va;

    status = uvm_rm_mem_alloc(gpu, type, size, &rm_mem);
    TEST_CHECK_RET(status == NV_OK);

    *mem_out = rm_mem;

    // Map
    status = uvm_rm_mem_map_cpu(rm_mem);
    TEST_CHECK_RET(status == NV_OK);
    TEST_CHECK_RET(uvm_rm_mem_get_cpu_va(rm_mem) != NULL);

    // Mapping if already mapped is OK
    status = uvm_rm_mem_map_cpu(rm_mem);
    TEST_CHECK_RET(status == NV_OK);

    // Unmap
    uvm_rm_mem_unmap_cpu(rm_mem);
    // Unmapping already unmapped also OK
    uvm_rm_mem_unmap_cpu(rm_mem);

    // Map again
    status = uvm_rm_mem_map_cpu(rm_mem);
    TEST_CHECK_RET(status == NV_OK);

    cpu_va = uvm_rm_mem_get_cpu_va(rm_mem);
    TEST_CHECK_RET(cpu_va != NULL);

    // Check that the CPU VA is writable
    memset(cpu_va, 0, size);

    return NV_OK;
}

static NV_STATUS map_other_gpus(uvm_rm_mem_t *rm_mem, uvm_processor_mask_t *other_gpus)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = rm_mem->gpu_owner;
    uvm_gpu_t *other_gpu;

    for_each_gpu_in_mask(other_gpu, other_gpus) {
        if (other_gpu == gpu)
            continue;
        status = uvm_rm_mem_map_gpu(rm_mem, other_gpu);
        TEST_CHECK_RET(status == NV_OK);
        TEST_CHECK_RET(uvm_rm_mem_get_gpu_va(rm_mem, other_gpu) != 0);
        status = uvm_rm_mem_map_gpu(rm_mem, other_gpu);
        TEST_CHECK_RET(status == NV_OK);

        uvm_rm_mem_unmap_gpu(rm_mem, other_gpu);
        uvm_rm_mem_unmap_gpu(rm_mem, other_gpu);

        status = uvm_rm_mem_map_gpu(rm_mem, other_gpu);
        TEST_CHECK_RET(status == NV_OK);
        TEST_CHECK_RET(uvm_rm_mem_get_gpu_va(rm_mem, other_gpu) != 0);
    }

    return NV_OK;
}

static NV_STATUS test_all_gpus_in_va(uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    NvU32 gpu_count = 0;
    uvm_rm_mem_t **all_mem = NULL;
    NvU32 allocation_count;
    NvU32 current_alloc = 0;

    // Create allocations of these types
    static const uvm_rm_mem_type_t mem_types[] = { UVM_RM_MEM_TYPE_SYS, UVM_RM_MEM_TYPE_GPU };
    // Create allocations of these sizes
    static const size_t sizes[] = {1, 4, 16, 128, 1024, 4096, 1024 * 1024, 4 * 1024 * 1024};
    int i, j;

    uvm_assert_rwsem_locked(&va_space->lock);

    for_each_gpu_in_mask(gpu, &va_space->registered_gpus)
        ++gpu_count;

    if (gpu_count == 0)
        return NV_ERR_INVALID_STATE;

    allocation_count = gpu_count * ARRAY_SIZE(sizes) * ARRAY_SIZE(mem_types);

    all_mem = uvm_kvmalloc_zero(sizeof(*all_mem) * allocation_count);

    if (all_mem == NULL)
        return NV_ERR_NO_MEMORY;

    for_each_gpu_in_mask(gpu, &va_space->registered_gpus) {
        for (i = 0; i < ARRAY_SIZE(sizes); ++i) {
            for (j = 0; j < ARRAY_SIZE(mem_types); ++j) {
                uvm_rm_mem_t *rm_mem;
                uvm_rm_mem_type_t mem_type = mem_types[j];
                // Create an allocation in the GPU's address space and map it on the CPU
                status = alloc_and_map_cpu(gpu, mem_type, sizes[i], &rm_mem);
                if (status != NV_OK) {
                    UVM_TEST_PRINT("Failed to alloc and map on the CPU\n");
                    goto cleanup;
                }
                all_mem[current_alloc++] = rm_mem;
                if (mem_type == UVM_RM_MEM_TYPE_SYS) {
                    // For sysmem allocations also map them on all other GPUs
                    status = map_other_gpus(rm_mem, &va_space->registered_gpus);
                    if (status != NV_OK) {
                        UVM_TEST_PRINT("Failed to map on other GPUs\n");
                        goto cleanup;
                    }
                }
            }
        }
    }

    if (current_alloc != allocation_count) {
        UVM_TEST_PRINT("Unexpected allocation count %u != %u\n", current_alloc, allocation_count);
        status = NV_ERR_INVALID_STATE;
    }

cleanup:
    for (i = 0; i < current_alloc; ++i)
        uvm_rm_mem_free(all_mem[i]);

    uvm_kvfree(all_mem);

    return status;
}

NV_STATUS uvm8_test_rm_mem_sanity(UVM_TEST_RM_MEM_SANITY_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    uvm_va_space_down_read_rm(va_space);

    status = test_all_gpus_in_va(va_space);

    uvm_va_space_up_read_rm(va_space);

    return status;
}
