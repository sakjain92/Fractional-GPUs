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
#include "uvm8_kvmalloc.h"
#include "uvm8_mem.h"
#include "uvm8_push.h"
#include "uvm8_test.h"
#include "uvm8_test_ioctl.h"
#include "uvm8_va_space.h"

static NvU32 first_page_size(NvU32 page_sizes)
{
    return page_sizes & ~(page_sizes - 1);
}

#define for_each_page_size(page_size, page_sizes)                                   \
    for (page_size = first_page_size(page_sizes);                                   \
         page_size;                                                                 \
         page_size = first_page_size((page_sizes) & ~(page_size | (page_size - 1))))

static NV_STATUS check_accessible_from_gpu(uvm_gpu_t *gpu, uvm_mem_t *mem)
{
    NV_STATUS status;
    uvm_mem_t *sys_mem = NULL;
    uvm_push_t push;
    NvU64 *sys_verif;
    size_t i;
    NvU64 verif_size = mem->size;
    NvU64 offset;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();

    verif_size = UVM_ALIGN_UP(verif_size, sizeof(*sys_verif));

    UVM_ASSERT(mem->physical_allocation_size >= verif_size);
    UVM_ASSERT(verif_size >= sizeof(*sys_verif));

    status = uvm_mem_alloc_sysmem_and_map_cpu_kernel(verif_size, &sys_mem);
    TEST_CHECK_GOTO(status == NV_OK, done);
    status = uvm_mem_map_gpu_kernel(sys_mem, gpu);
    TEST_CHECK_GOTO(status == NV_OK, done);

    sys_verif = (NvU64*)uvm_mem_get_cpu_addr_kernel(sys_mem);

    for (i = 0; i < verif_size / sizeof(*sys_verif); ++i)
        sys_verif[i] = mem->size + i;

    // Copy from sys_mem to mem using physical access for mem and virtual for sys_mem in mem->page_size chunks
    for (offset = 0; offset < verif_size; offset += mem->chunk_size) {
        size_t size_this_time = min((NvU64)mem->chunk_size, verif_size - offset);
        uvm_gpu_address_t mem_gpu_phys = uvm_mem_gpu_address_physical(mem, gpu, offset, size_this_time);
        uvm_gpu_address_t sys_mem_gpu_virt = uvm_mem_gpu_address_virtual_kernel(sys_mem, gpu);
        sys_mem_gpu_virt.address += offset;

        status = uvm_push_begin(gpu->channel_manager, UVM_CHANNEL_TYPE_CPU_TO_GPU, &push,
                "Memcopy %zd bytes from virtual sys_mem 0x%llx to physical %smem 0x%llx page_size %u",
                size_this_time,
                sys_mem_gpu_virt.address,
                mem_gpu_phys.aperture == UVM_APERTURE_SYS ? "sys" : "vid", mem_gpu_phys.address, mem->chunk_size);
        TEST_CHECK_GOTO(status == NV_OK, done);

        if (uvm_mem_is_vidmem(mem) && mem->chunk_size == gpu->big_page.internal_size)
            mem_gpu_phys = uvm_mmu_gpu_address_for_big_page_physical(mem_gpu_phys, gpu);

        gpu->ce_hal->memcopy(&push, mem_gpu_phys, sys_mem_gpu_virt, size_this_time);

        uvm_push_end(&push);
        status = uvm_tracker_add_push(&tracker, &push);
        TEST_CHECK_GOTO(status == NV_OK, done);
    }

    status = uvm_tracker_wait(&tracker);
    TEST_CHECK_GOTO(status == NV_OK, done);

    memset(sys_verif, 0, verif_size);

    // Copy back to sys_mem from mem using physical access for sys_mem and virtual for mem in sys_mem->page_size chunks
    for (offset = 0; offset < verif_size; offset += sys_mem->chunk_size) {
        size_t size_this_time = min((NvU64)sys_mem->chunk_size, verif_size - offset);
        uvm_gpu_address_t sys_mem_gpu_phys = uvm_mem_gpu_address_physical(sys_mem, gpu, offset, size_this_time);
        uvm_gpu_address_t mem_gpu_virt = uvm_mem_gpu_address_virtual_kernel(mem, gpu);
        mem_gpu_virt.address += offset;

        status = uvm_push_begin(gpu->channel_manager, UVM_CHANNEL_TYPE_GPU_TO_CPU, &push,
                "Memcopy %zd bytes from virtual mem 0x%llx to physical sysmem 0x%llx",
                size_this_time, mem_gpu_virt.address, sys_mem_gpu_phys.address);
        TEST_CHECK_GOTO(status == NV_OK, done);

        gpu->ce_hal->memcopy(&push, sys_mem_gpu_phys, mem_gpu_virt, size_this_time);

        uvm_push_end(&push);
        status = uvm_tracker_add_push(&tracker, &push);
        TEST_CHECK_GOTO(status == NV_OK, done);
    }

    status = uvm_tracker_wait(&tracker);
    TEST_CHECK_GOTO(status == NV_OK, done);

    for (i = 0; i < verif_size / sizeof(*sys_verif); ++i) {
        if (sys_verif[i] != mem->size + i) {
            UVM_TEST_PRINT("Verif failed for %zd = 0x%llx instead of 0x%llx, verif_size=0x%llx mem(size=0x%llx, page_size=%u, processor=%u)\n",
                    i, sys_verif[i], (NvU64)(verif_size + i),
                    verif_size, mem->size, mem->chunk_size, mem->backing_gpu ? mem->backing_gpu->id : UVM_CPU_ID);
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
    }

done:
    (void)uvm_tracker_wait(&tracker);
    uvm_tracker_deinit(&tracker);
    if (sys_mem)
        uvm_mem_free(sys_mem);

    return status;
}

static NV_STATUS test_map_gpu(uvm_mem_t *mem, uvm_gpu_t *gpu)
{
    NV_STATUS status;
    NvU64 gpu_va;

    status = uvm_mem_map_gpu_kernel(mem, gpu);
    TEST_CHECK_RET(status == NV_OK);
    gpu_va = uvm_mem_get_gpu_va_kernel(mem, gpu);

    TEST_CHECK_RET(gpu_va >= gpu->uvm_mem_va_base);
    TEST_CHECK_RET(gpu_va + mem->physical_allocation_size <= gpu->uvm_mem_va_base + gpu->uvm_mem_va_size);

    // Mapping if already mapped is OK
    status = uvm_mem_map_gpu_kernel(mem, gpu);
    TEST_CHECK_RET(status == NV_OK);

    // Unmap
    uvm_mem_unmap_gpu(mem, gpu);
    // Unmapping already unmapped also OK
    uvm_mem_unmap_gpu(mem, gpu);

    // Map again
    status = uvm_mem_map_gpu_kernel(mem, gpu);
    TEST_CHECK_RET(status == NV_OK);

    // Should get the same VA
    TEST_CHECK_RET(gpu_va == uvm_mem_get_gpu_va_kernel(mem, gpu));

    return check_accessible_from_gpu(gpu, mem);
}

static NV_STATUS test_map_cpu(uvm_mem_t *mem)
{
    NV_STATUS status;
    char *cpu_addr;

    if (uvm_mem_is_vidmem(mem))
        UVM_ASSERT(mem->backing_gpu->numa_info.enabled);

    // Map
    status = uvm_mem_map_cpu(mem, NULL);
    TEST_CHECK_RET(status == NV_OK);
    TEST_CHECK_RET(uvm_mem_get_cpu_addr_kernel(mem) != NULL);

    // Mapping if already mapped is OK
    status = uvm_mem_map_cpu(mem, NULL);
    TEST_CHECK_RET(status == NV_OK);

    // Unmap
    uvm_mem_unmap_cpu(mem);
    // Unmapping already unmapped also OK
    uvm_mem_unmap_cpu(mem);

    // Map again
    status = uvm_mem_map_cpu(mem, NULL);
    TEST_CHECK_RET(status == NV_OK);

    cpu_addr = uvm_mem_get_cpu_addr_kernel(mem);
    TEST_CHECK_RET(cpu_addr != NULL);

    memset(cpu_addr, 3, mem->size);

    return NV_OK;
}

static NV_STATUS test_alloc_sysmem(uvm_va_space_t *va_space, NvU32 page_size, size_t size, uvm_mem_t **mem_out)
{
    NV_STATUS status;
    uvm_mem_t *mem;
    uvm_gpu_t *gpu;
    uvm_mem_alloc_params_t params = { 0 };

    params.size = size;
    params.page_size = page_size;

    status = uvm_mem_alloc(&params, &mem);
    TEST_CHECK_GOTO(status == NV_OK, error);

    TEST_CHECK_GOTO(test_map_cpu(mem) == NV_OK, error);

    for_each_va_space_gpu(gpu, va_space)
        TEST_CHECK_GOTO(test_map_gpu(mem, gpu) == NV_OK, error);

    *mem_out = mem;

    return NV_OK;

error:
    uvm_mem_free(mem);
    return status;
}

static NV_STATUS test_alloc_vidmem(uvm_gpu_t *gpu, NvU32 page_size, size_t size, uvm_mem_t **mem_out)
{
    NV_STATUS status;
    uvm_mem_t *mem;
    uvm_mem_alloc_params_t params = { 0 };

    params.backing_gpu = gpu;
    params.page_size = page_size;
    params.size = size;

    status = uvm_mem_alloc(&params, &mem);
    TEST_CHECK_GOTO(status == NV_OK, error);

    if (page_size == UVM_PAGE_SIZE_DEFAULT) {
        if (gpu->numa_info.enabled)
            TEST_CHECK_GOTO(mem->chunk_size >= PAGE_SIZE && mem->chunk_size <= max(size, (size_t)PAGE_SIZE), error);
        else
            TEST_CHECK_GOTO(mem->chunk_size == UVM_PAGE_SIZE_4K || mem->chunk_size <= size, error);
    }

    TEST_CHECK_GOTO(test_map_gpu(mem, gpu) == NV_OK, error);

    if (gpu->numa_info.enabled && (page_size == UVM_PAGE_SIZE_DEFAULT || page_size >= PAGE_SIZE))
        TEST_CHECK_GOTO(test_map_cpu(mem) == NV_OK, error);

    *mem_out = mem;

    return NV_OK;

error:
    uvm_mem_free(mem);
    return status;
}

static NV_STATUS test_all(uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    NvU32 gpu_count;
    uvm_mem_t **all_mem = NULL;
    NvU32 allocation_count;
    NvU32 current_alloc = 0;

    // Create allocations of these sizes
    static const size_t sizes[] = {1, 4, 16, 1024, 4096, 1024 * 1024, 7 * 1024 * 1024 + 17 };

    // Pascal+ can map sysmem with 4K, 64K and 2M PTEs, other GPUs can only use
    // 4K. Test all of the sizes supported by Pascal+ and 128K to match big page
    // size on pre-Pascal GPUs with 128K big page size.
    static const NvU32 cpu_chunk_sizes = PAGE_SIZE | UVM_PAGE_SIZE_64K | UVM_PAGE_SIZE_128K | UVM_PAGE_SIZE_2M;

    // All supported page sizes will be tested, CPU has the most with 4 and +1
    // for the default.
    static const int max_supported_page_sizes = 4 + 1;
    int i;

    gpu_count = uvm_processor_mask_get_gpu_count(&va_space->registered_gpus);

    // +1 for the CPU
    allocation_count = (gpu_count + 1) * max_supported_page_sizes * ARRAY_SIZE(sizes);

    all_mem = uvm_kvmalloc_zero(sizeof(*all_mem) * allocation_count);

    if (all_mem == NULL)
        return NV_ERR_NO_MEMORY;

    for (i = 0; i < ARRAY_SIZE(sizes); ++i) {
        NvU32 page_size = 0;
        uvm_mem_t *mem;

        status = test_alloc_sysmem(va_space, UVM_PAGE_SIZE_DEFAULT, sizes[i], &mem);
        if (status != NV_OK) {
            UVM_TEST_PRINT("Failed to alloc sysmem size %zd, page_size default\n", sizes[i], page_size);
            goto cleanup;
        }
        all_mem[current_alloc++] = mem;

        for_each_page_size(page_size, cpu_chunk_sizes) {
            status = test_alloc_sysmem(va_space, page_size, sizes[i], &mem);
            if (status != NV_OK) {
                UVM_TEST_PRINT("Failed to alloc sysmem size %zd, page_size %u\n", sizes[i], page_size);
                goto cleanup;
            }
            all_mem[current_alloc++] = mem;
        }

        for_each_va_space_gpu(gpu, va_space) {
            NvU32 page_sizes = gpu->address_space_tree.hal->page_sizes();

            UVM_ASSERT(max_supported_page_sizes >= hweight_long(page_sizes));

            status = test_alloc_vidmem(gpu, UVM_PAGE_SIZE_DEFAULT, sizes[i], &mem);
            if (status != NV_OK) {
                UVM_TEST_PRINT("Test alloc vidmem failed, page_size default size %zd GPU %s\n", sizes[i], gpu->name);
                goto cleanup;
            }
            all_mem[current_alloc++] = mem;

            for_each_page_size(page_size, page_sizes) {
                status = test_alloc_vidmem(gpu, page_size, sizes[i], &mem);
                if (status != NV_OK) {
                    UVM_TEST_PRINT("Test alloc vidmem failed, page_size %u size %zd GPU %s\n", page_size, sizes[i], gpu->name);
                    goto cleanup;
                }
                all_mem[current_alloc++] = mem;
            }
        }
    }

cleanup:
    for (i = 0; i < current_alloc; ++i)
        uvm_mem_free(all_mem[i]);

    uvm_kvfree(all_mem);

    return status;
}

static NV_STATUS test_basic_vidmem(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    NvU32 page_size;
    NvU32 page_sizes = gpu->address_space_tree.hal->page_sizes();
    NvU32 biggest_page_size = uvm_mmu_biggest_page_size(&gpu->address_space_tree);
    NvU32 smallest_page_size = page_sizes & ~(page_sizes - 1);
    uvm_mem_t *mem = NULL;

    for_each_page_size(page_size, page_sizes) {
        TEST_CHECK_GOTO(uvm_mem_alloc_vidmem(page_size - 1, gpu, &mem) == NV_OK, done);
        if (gpu->numa_info.enabled)
            TEST_CHECK_GOTO(mem->chunk_size >= PAGE_SIZE && mem->chunk_size <= max(page_size, (NvU32)PAGE_SIZE), done);
        else
            TEST_CHECK_GOTO(mem->chunk_size < page_size || page_size == smallest_page_size, done);
        uvm_mem_free(mem);
        mem = NULL;

        TEST_CHECK_GOTO(uvm_mem_alloc_vidmem(page_size, gpu, &mem) == NV_OK, done);
        if (gpu->numa_info.enabled)
            TEST_CHECK_GOTO(mem->chunk_size == max(page_size, (NvU32)PAGE_SIZE), done);
        else
            TEST_CHECK_GOTO(mem->chunk_size == page_size, done);
        uvm_mem_free(mem);
        mem = NULL;
    }

    TEST_CHECK_GOTO(uvm_mem_alloc_vidmem(5 * biggest_page_size - 1, gpu, &mem) == NV_OK, done);
    TEST_CHECK_GOTO(mem->chunk_size == biggest_page_size, done);

done:
    uvm_mem_free(mem);
    return status;
}

static NV_STATUS test_basic_sysmem(void)
{
    NV_STATUS status = NV_OK;
    uvm_mem_t *mem = NULL;
    int i;
    static const size_t sizes[] = { 1, PAGE_SIZE - 1, PAGE_SIZE, 7 * PAGE_SIZE };

    for (i = 0; i < ARRAY_SIZE(sizes); ++i) {
        size_t size = sizes[i];
        TEST_CHECK_GOTO(uvm_mem_alloc_sysmem(size, &mem) == NV_OK, done);
        TEST_CHECK_GOTO(mem->chunk_size == PAGE_SIZE, done);
        uvm_mem_free(mem);
        mem = NULL;
    }

done:
    uvm_mem_free(mem);
    return status;
}

static NV_STATUS test_basic(uvm_va_space_t *va_space)
{
    uvm_gpu_t *gpu;

    TEST_CHECK_RET(test_basic_sysmem() == NV_OK);

    for_each_va_space_gpu(gpu, va_space)
        TEST_CHECK_RET(test_basic_vidmem(gpu) == NV_OK);

    return NV_OK;
}

static NV_STATUS tests(uvm_va_space_t *va_space)
{
    TEST_CHECK_RET(test_basic(va_space) == NV_OK);
    TEST_CHECK_RET(test_all(va_space) == NV_OK);

    return NV_OK;
}

NV_STATUS uvm8_test_mem_sanity(UVM_TEST_MEM_SANITY_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    uvm_va_space_down_read(va_space);

    status = tests(va_space);

    uvm_va_space_up_read(va_space);

    return status;
}
