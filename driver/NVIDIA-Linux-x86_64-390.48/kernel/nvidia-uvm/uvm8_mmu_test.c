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

#include "uvm8_gpu.h"
#include "uvm8_hal.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_mem.h"
#include "uvm8_push.h"
#include "uvm8_va_space.h"
#include "uvm8_test.h"
#include "uvm8_test_ioctl.h"

static NV_STATUS copy_wait(uvm_gpu_t *gpu, uvm_gpu_address_t dst, uvm_gpu_address_t src, size_t size)
{
    uvm_push_t push;
    NV_STATUS status;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_ANY,
                            &push,
                            "Copying {%s, 0x%llx} to {%s, 0x%llx} size %zu",
                            uvm_gpu_address_aperture_string(src), src.address,
                            uvm_gpu_address_aperture_string(dst), dst.address,
                            size);
    if (status != NV_OK)
        return status;

    gpu->ce_hal->memcopy(&push, dst, src, size);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS swizzle_phys_wait(uvm_gpu_t *gpu, NvU64 phys_addr, uvm_gpu_swizzle_op_t op)
{
    NV_STATUS status, tracker_status;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    status = uvm_gpu_swizzle_phys(gpu, phys_addr, op, &tracker);
    tracker_status = uvm_tracker_wait_deinit(&tracker);
    return status == NV_OK ? tracker_status : status;
}

// Test that big page swizzling happens as expected for GPUs with the
// gpu->big_page_swizzling property set, and that the big page identity mapping
// works correctly to deswizzle.
static NV_STATUS test_big_page_swizzling(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    uvm_mem_t *gpu_mem = NULL;
    uvm_mem_t *sys_mem_gold = NULL, *sys_mem_verif = NULL;
    int i;
    NvU64 *sys_verif, *sys_gold;
    NvU32 big_page_size = gpu->big_page.internal_size;
    uvm_gpu_address_t gpu_virtual;
    uvm_gpu_address_t gpu_physical;
    int use_identity_mapping;

    uvm_mem_alloc_params_t mem_params_gpu = { 0 };

    mem_params_gpu.backing_gpu = gpu;
    mem_params_gpu.size = big_page_size;
    mem_params_gpu.page_size = big_page_size;

    TEST_NV_CHECK_GOTO(uvm_mem_alloc(&mem_params_gpu, &gpu_mem), done);
    TEST_NV_CHECK_GOTO(uvm_mem_map_gpu_kernel(gpu_mem, gpu), done);
    gpu_virtual = uvm_mem_gpu_address_virtual_kernel(gpu_mem, gpu);

    TEST_NV_CHECK_GOTO(uvm_mem_alloc_sysmem_and_map_cpu_kernel(big_page_size, &sys_mem_gold), done);
    TEST_NV_CHECK_GOTO(uvm_mem_map_gpu_kernel(sys_mem_gold, gpu), done);
    sys_gold = uvm_mem_get_cpu_addr_kernel(sys_mem_gold);

    TEST_NV_CHECK_GOTO(uvm_mem_alloc_sysmem_and_map_cpu_kernel(big_page_size, &sys_mem_verif), done);
    TEST_NV_CHECK_GOTO(uvm_mem_map_gpu_kernel(sys_mem_verif, gpu), done);
    sys_verif = uvm_mem_get_cpu_addr_kernel(sys_mem_verif);

    for (i = 0; i < big_page_size / sizeof(NvU64); ++i)
        sys_gold[i] = i;

    // Test basic identity copies
    for (use_identity_mapping = 0; use_identity_mapping <= 1; use_identity_mapping++) {
        bool mismatch = false;

        memset(sys_verif, 0, big_page_size);

        gpu_physical = uvm_mem_gpu_address_physical(gpu_mem, gpu, 0, big_page_size);
        if (use_identity_mapping)
            gpu_physical = uvm_mmu_gpu_address_for_big_page_physical(gpu_physical, gpu);

        // Copy known system memory into the GPU using virtual big page mapping
        // and copy it back using the physical address. If identity mappings are
        // being used, the physical address has been converted to a virtual one
        // above.
        TEST_NV_CHECK_GOTO(copy_wait(gpu,
                                     gpu_virtual,
                                     uvm_mem_gpu_address_virtual_kernel(sys_mem_gold, gpu),
                                     big_page_size), done);

        TEST_NV_CHECK_GOTO(copy_wait(gpu,
                                     uvm_mem_gpu_address_virtual_kernel(sys_mem_verif, gpu),
                                     gpu_physical,
                                     big_page_size), done);

        for (i = 0; i < big_page_size / sizeof(NvU64); ++i) {
            if (sys_verif[i] != sys_gold[i]) {
                mismatch = true;
                break;
            }
        }
        if (use_identity_mapping && mismatch) {
            UVM_TEST_PRINT("Invalid value at %d = %llu instead of %llu, GPU %s\n",
                    i, sys_verif[i], sys_gold[i], gpu->name);
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
        else if (!use_identity_mapping && !mismatch) {
            UVM_TEST_PRINT("Everything matching even though it shouldn't, GPU %s\n", gpu->name);
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
    }

    // Test uvm_gpu_swizzle_phys

    // Copy pattern as physical (unswizzled)
    gpu_physical = uvm_mem_gpu_address_physical(gpu_mem, gpu, 0, big_page_size);
    TEST_NV_CHECK_GOTO(copy_wait(gpu,
                                 gpu_physical,
                                 uvm_mem_gpu_address_virtual_kernel(sys_mem_gold, gpu),
                                 big_page_size), done);

    // Swizzle then copy back with the identity mapping. Since both swizzle,
    // they should match.
    TEST_NV_CHECK_GOTO(swizzle_phys_wait(gpu, gpu_physical.address, UVM_GPU_SWIZZLE_OP_SWIZZLE), done);
    memset(sys_verif, 0, big_page_size);
    TEST_NV_CHECK_GOTO(copy_wait(gpu,
                                 uvm_mem_gpu_address_virtual_kernel(sys_mem_verif, gpu),
                                 uvm_mmu_gpu_address_for_big_page_physical(gpu_physical, gpu),
                                 big_page_size), done);
    TEST_CHECK_GOTO(memcmp(sys_verif, sys_gold, big_page_size) == 0, done);

    // Deswizzle then copy back with the physical address. They should match.
    TEST_NV_CHECK_GOTO(swizzle_phys_wait(gpu, gpu_physical.address, UVM_GPU_SWIZZLE_OP_DESWIZZLE), done);
    memset(sys_verif, 0, big_page_size);
    TEST_NV_CHECK_GOTO(copy_wait(gpu,
                                 uvm_mem_gpu_address_virtual_kernel(sys_mem_verif, gpu),
                                 gpu_physical,
                                 big_page_size), done);
    TEST_CHECK_GOTO(memcmp(sys_verif, sys_gold, big_page_size) == 0, done);

done:
    uvm_mem_free(gpu_mem);
    uvm_mem_free(sys_mem_gold);
    uvm_mem_free(sys_mem_verif);

    return status;
}

static NV_STATUS test_all_gpus(uvm_va_space_t *va_space)
{
    uvm_gpu_t *gpu;
    for_each_va_space_gpu(gpu, va_space) {
        if (gpu->big_page.swizzling)
            MEM_NV_CHECK_RET(test_big_page_swizzling(gpu), NV_OK);
    }

    return NV_OK;
}

NV_STATUS uvm8_test_mmu_sanity(UVM_TEST_MMU_SANITY_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    uvm_va_space_down_read(va_space);

    status = test_all_gpus(va_space);

    uvm_va_space_up_read(va_space);

    return status;
}
