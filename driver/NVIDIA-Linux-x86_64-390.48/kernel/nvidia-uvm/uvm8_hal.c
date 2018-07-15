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

#include "uvm8_hal.h"
#include "uvm8_kvmalloc.h"

#include "cla06f.h"
#include "cla06fsubch.h"
#include "cla16f.h"
#include "cla0b5.h"
#include "clb069.h"
#include "clb06f.h"
#include "clb0b5.h"
#include "clc06f.h"
#include "clc0b5.h"
#include "clc1b5.h"
#include "ctrl2080mc.h"
#include "clc3b5.h"
#include "clc36f.h"
#include "clc369.h"
#include "clc365.h"





#define CE_OP_COUNT (sizeof(uvm_ce_hal_t) / sizeof(void *))
#define HOST_OP_COUNT (sizeof(uvm_host_hal_t) / sizeof(void *))
#define ARCH_OP_COUNT (sizeof(uvm_arch_hal_t) / sizeof(void *))
#define FAULT_BUFFER_OP_COUNT (sizeof(uvm_fault_buffer_hal_t) / sizeof(void *))
#define ACCESS_COUNTER_BUFFER_OP_COUNT (sizeof(uvm_access_counter_buffer_hal_t) / sizeof(void *))

// Table for copy engine functions.
// Each entry is associated with a copy engine class through the 'class' field.
// By setting the 'parent_class' field, a class will inherit the parent class's
// functions for any fields left NULL when uvm_hal_init_table() runs upon module load.
// The parent class must appear earlier in the array than the child.
static uvm_hal_class_ops_t ce_table[] =
{
    {
        .id = KEPLER_DMA_COPY_A,
        .u.ce_ops = {
            .init = uvm_hal_kepler_ce_init,
            .semaphore_release = uvm_hal_kepler_ce_semaphore_release,
            .semaphore_timestamp = uvm_hal_kepler_ce_semaphore_timestamp,
            .semaphore_reduction_inc = uvm_hal_kepler_ce_semaphore_reduction_inc,
            .offset_out = uvm_hal_kepler_ce_offset_out,
            .offset_in_out = uvm_hal_kepler_ce_offset_in_out,
            .memcopy = uvm_hal_kepler_ce_memcopy,
            .memcopy_v_to_v = uvm_hal_kepler_ce_memcopy_v_to_v,
            .memset_1 = uvm_hal_kepler_ce_memset_1,
            .memset_4 = uvm_hal_kepler_ce_memset_4,
            .memset_8 = uvm_hal_kepler_ce_memset_8,
            .memset_v_4 = uvm_hal_kepler_ce_memset_v_4,
        }
    },
    {
        .id = MAXWELL_DMA_COPY_A,
        .parent_id = KEPLER_DMA_COPY_A,
        .u.ce_ops = {}
    },
    {
        .id = PASCAL_DMA_COPY_A,
        .parent_id = MAXWELL_DMA_COPY_A,
        .u.ce_ops = {
            .offset_out = uvm_hal_pascal_ce_offset_out,
            .offset_in_out = uvm_hal_pascal_ce_offset_in_out,
        }
    },
    {
        .id = PASCAL_DMA_COPY_B,
        .parent_id = PASCAL_DMA_COPY_A,
        .u.ce_ops = {}
    },
    {
        .id = VOLTA_DMA_COPY_A,
        .parent_id = PASCAL_DMA_COPY_A,
        .u.ce_ops = {},
    },







};

// Table for GPFIFO functions.  Same idea as the copy engine table.
static uvm_hal_class_ops_t host_table[] =
{
    {
        .id = KEPLER_CHANNEL_GPFIFO_A,
        .u.host_ops = {
            .init = uvm_hal_kepler_host_init_noop,
            .wait_for_idle = uvm_hal_kepler_host_wait_for_idle_a06f,
            .membar_sys = uvm_hal_kepler_host_membar_sys,
            // No MEMBAR GPU until Pascal, just do a MEMBAR SYS.
            .membar_gpu = uvm_hal_kepler_host_membar_sys,
            .noop = uvm_hal_kepler_host_noop,
            .interrupt = uvm_hal_kepler_host_interrupt,
            .semaphore_acquire = uvm_hal_kepler_host_semaphore_acquire,
            .semaphore_release = uvm_hal_kepler_host_semaphore_release,
            .set_gpfifo_entry = uvm_hal_kepler_host_set_gpfifo_entry,
            .write_gpu_put = uvm_hal_kepler_host_write_gpu_put,
            .tlb_invalidate_all = uvm_hal_kepler_host_tlb_invalidate_all,
            .tlb_invalidate_va = uvm_hal_kepler_host_tlb_invalidate_va,
            .tlb_invalidate_test = uvm_hal_kepler_host_tlb_invalidate_test,
            .replay_faults = uvm_hal_kepler_replay_faults_unsupported,
            .cancel_faults_targeted = uvm_hal_kepler_cancel_faults_targeted_unsupported,
            .cancel_faults_va = uvm_hal_kepler_cancel_faults_va_unsupported,
            .clear_faulted_channel = uvm_hal_kepler_host_clear_faulted_channel_unsupported,
            .access_counter_clear_all = uvm_hal_kepler_access_counter_clear_all_unsupported,
            .access_counter_clear_type = uvm_hal_kepler_access_counter_clear_type_unsupported,
            .access_counter_clear_targeted = uvm_hal_kepler_access_counter_clear_targeted_unsupported,
        }
    },
    {
        .id = KEPLER_CHANNEL_GPFIFO_B,
        .parent_id = KEPLER_CHANNEL_GPFIFO_A,
        .u.host_ops = {
            .wait_for_idle = uvm_hal_kepler_host_wait_for_idle_a16f,
        }
    },
    {
        .id = MAXWELL_CHANNEL_GPFIFO_A,
        .parent_id = KEPLER_CHANNEL_GPFIFO_A,
        .u.host_ops = {
            .tlb_invalidate_all = uvm_hal_maxwell_host_tlb_invalidate_all,
        }
    },
    {
        .id = PASCAL_CHANNEL_GPFIFO_A,
        .parent_id = MAXWELL_CHANNEL_GPFIFO_A,
        .u.host_ops = {
            .init = uvm_hal_pascal_host_init,
            .membar_sys = uvm_hal_pascal_host_membar_sys,
            .membar_gpu = uvm_hal_pascal_host_membar_gpu,
            .tlb_invalidate_all = uvm_hal_pascal_host_tlb_invalidate_all,
            .tlb_invalidate_va = uvm_hal_pascal_host_tlb_invalidate_va,
            .tlb_invalidate_test = uvm_hal_pascal_host_tlb_invalidate_test,
            .replay_faults = uvm_hal_pascal_replay_faults,
            .cancel_faults_targeted = uvm_hal_pascal_cancel_faults_targeted,
        }
    },
    {
        .id = VOLTA_CHANNEL_GPFIFO_A,
        .parent_id = PASCAL_CHANNEL_GPFIFO_A,
        .u.host_ops = {
            .write_gpu_put = uvm_hal_volta_host_write_gpu_put,
            .tlb_invalidate_va = uvm_hal_volta_host_tlb_invalidate_va,
            .cancel_faults_va = uvm_hal_volta_cancel_faults_va,
            .clear_faulted_channel = uvm_hal_volta_host_clear_faulted_channel,
            .access_counter_clear_all = uvm_hal_volta_access_counter_clear_all,
            .access_counter_clear_type = uvm_hal_volta_access_counter_clear_type,
            .access_counter_clear_targeted = uvm_hal_volta_access_counter_clear_targeted,
        }
    },







};

static uvm_hal_class_ops_t arch_table[] =
{
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK100,
        .u.arch_ops = {
            .init_properties = uvm_hal_kepler_arch_init_properties,
            .mmu_mode_hal = uvm_hal_mmu_mode_kepler,
            .enable_prefetch_faults = uvm_hal_kepler_mmu_enable_prefetch_faults_unsupported,
            .disable_prefetch_faults = uvm_hal_kepler_mmu_disable_prefetch_faults_unsupported,
        }
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK110,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK100,
        .u.arch_ops = {}
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK200,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK100,
        .u.arch_ops = {}
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM000,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK100,
        .u.arch_ops = {
            .init_properties = uvm_hal_maxwell_arch_init_properties,
        }
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM000,
        .u.arch_ops = {}
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM000,
        .u.arch_ops = {
            .init_properties = uvm_hal_pascal_arch_init_properties,
            .mmu_mode_hal = uvm_hal_mmu_mode_pascal,
            .enable_prefetch_faults = uvm_hal_pascal_mmu_enable_prefetch_faults,
            .disable_prefetch_faults = uvm_hal_pascal_mmu_disable_prefetch_faults,
        }
    },
    {
        .id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100,
        .parent_id = NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100,
        .u.arch_ops = {
            .init_properties = uvm_hal_volta_arch_init_properties,
            .mmu_mode_hal = uvm_hal_mmu_mode_volta,
        },
    },










};

static uvm_hal_class_ops_t fault_buffer_table[] =
{
    {
        .id = MAXWELL_FAULT_BUFFER_A,
        .u.fault_buffer_ops = {
            .enable_replayable_faults  = uvm_hal_pascal_enable_replayable_faults,
            .disable_replayable_faults = uvm_hal_pascal_disable_replayable_faults,
            .read_put = uvm_hal_pascal_fault_buffer_read_put,
            .read_get = uvm_hal_pascal_fault_buffer_read_get,
            .write_get = uvm_hal_pascal_fault_buffer_write_get,
            .parse_entry = uvm_hal_pascal_fault_buffer_parse_entry,
            .entry_is_valid = uvm_hal_pascal_fault_buffer_entry_is_valid,
            .entry_clear_valid = uvm_hal_pascal_fault_buffer_entry_clear_valid,
            .entry_size = uvm_hal_pascal_fault_buffer_entry_size,
            .parse_non_replayable_entry = uvm_hal_pascal_fault_buffer_parse_non_replayable_entry_unsupported,
        }
    },
    {
        .id = MMU_FAULT_BUFFER,
        .parent_id = MAXWELL_FAULT_BUFFER_A,
        .u.fault_buffer_ops = {
            .read_put = uvm_hal_volta_fault_buffer_read_put,
            .read_get = uvm_hal_volta_fault_buffer_read_get,
            .write_get = uvm_hal_volta_fault_buffer_write_get,
            .parse_entry = uvm_hal_volta_fault_buffer_parse_entry,
            .parse_non_replayable_entry = uvm_hal_volta_fault_buffer_parse_non_replayable_entry,
        },
    }
};

static uvm_hal_class_ops_t access_counter_buffer_table[] =
{
    {
        .id = ACCESS_COUNTER_NOTIFY_BUFFER,
        .u.access_counter_buffer_ops = {
            .enable_access_counter_notifications  = uvm_hal_volta_enable_access_counter_notifications,
            .disable_access_counter_notifications = uvm_hal_volta_disable_access_counter_notifications,
            .parse_entry = uvm_hal_volta_access_counter_buffer_parse_entry,
            .entry_is_valid = uvm_hal_volta_access_counter_buffer_entry_is_valid,
            .entry_clear_valid = uvm_hal_volta_access_counter_buffer_entry_clear_valid,
            .entry_size = uvm_hal_volta_access_counter_buffer_entry_size,
        },
    }
};

static inline uvm_hal_class_ops_t *ops_find_by_id(uvm_hal_class_ops_t *table, NvU32 row_count, NvU32 id)
{
    NvLength i;

    // go through array and match on class.
    for (i = 0; i < row_count; i++) {
        if (table[i].id == id)
            return table + i;
    }

    return NULL;
}

// use memcmp to check for function pointer assignment in a well defined, general way.
static inline bool op_is_null(uvm_hal_class_ops_t *row, NvLength op_idx, NvLength op_offset)
{
    void *temp = NULL;
    return memcmp(&temp, (char *)row + op_offset + sizeof(void *) * op_idx, sizeof(void *)) == 0;
}

// use memcpy to copy function pointers in a well defined, general way.
static inline void op_copy(uvm_hal_class_ops_t *dst, uvm_hal_class_ops_t *src, NvLength op_idx, NvLength op_offset)
{
    void *m_dst = (char *)dst + op_offset + sizeof(void *) * op_idx;
    void *m_src = (char *)src + op_offset + sizeof(void *) * op_idx;
    memcpy(m_dst, m_src, sizeof(void *));
}

static inline NV_STATUS ops_init_from_parent(uvm_hal_class_ops_t *table,
                                             NvU32 row_count,
                                             NvLength op_count,
                                             NvLength op_offset)
{
    NvLength i;

    for (i = 0; i < row_count; i++) {
        NvLength j;
        uvm_hal_class_ops_t *parent = NULL;

        if (table[i].parent_id != 0) {
            parent = ops_find_by_id(table, i, table[i].parent_id);
            if (parent == NULL)
                return NV_ERR_INVALID_CLASS;

            // Go through all the ops and assign from parent's corresponding op if NULL
            for (j = 0; j < op_count; j++) {
                if (op_is_null(table + i, j, op_offset))
                    op_copy(table + i, parent, j, op_offset);
            }
        }

        // At this point, it is an error to have missing HAL operations
        for (j = 0; j < op_count; j++) {
            if (op_is_null(table + i, j, op_offset))
                return NV_ERR_INVALID_STATE;
        }
    }

    return NV_OK;
}

NV_STATUS uvm_hal_init_table(void)
{
    NV_STATUS status;

    status = ops_init_from_parent(ce_table, ARRAY_SIZE(ce_table), CE_OP_COUNT, offsetof(uvm_hal_class_ops_t, u.ce_ops));
    if (status != NV_OK) {
        UVM_ERR_PRINT("ops_init_from_parent(ce_table) failed: %s\n", nvstatusToString(status));
        return status;
    }

    status = ops_init_from_parent(host_table, ARRAY_SIZE(host_table), HOST_OP_COUNT, offsetof(uvm_hal_class_ops_t, u.host_ops));
    if (status != NV_OK) {
        UVM_ERR_PRINT("ops_init_from_parent(host_table) failed: %s\n", nvstatusToString(status));
        return status;
    }

    status = ops_init_from_parent(arch_table, ARRAY_SIZE(arch_table), ARCH_OP_COUNT, offsetof(uvm_hal_class_ops_t, u.arch_ops));
    if (status != NV_OK) {
        UVM_ERR_PRINT("ops_init_from_parent(arch_table) failed: %s\n", nvstatusToString(status));
        return status;
    }

    status = ops_init_from_parent(fault_buffer_table, ARRAY_SIZE(fault_buffer_table), FAULT_BUFFER_OP_COUNT,
                                  offsetof(uvm_hal_class_ops_t, u.fault_buffer_ops));
    if (status != NV_OK) {
        UVM_ERR_PRINT("ops_init_from_parent(fault_buffer_table) failed: %s\n", nvstatusToString(status));
        return status;
    }

    status = ops_init_from_parent(access_counter_buffer_table,
                                  ARRAY_SIZE(access_counter_buffer_table),
                                  ACCESS_COUNTER_BUFFER_OP_COUNT,
                                  offsetof(uvm_hal_class_ops_t, u.access_counter_buffer_ops));
    if (status != NV_OK) {
        UVM_ERR_PRINT("ops_init_from_parent(access_counter_buffer_table) failed: %s\n", nvstatusToString(status));
        return status;
    }

    return NV_OK;
}

NV_STATUS uvm_hal_init_gpu(uvm_gpu_t *gpu)
{
    uvm_hal_class_ops_t *class_ops = ops_find_by_id(ce_table, ARRAY_SIZE(ce_table), gpu->ce_class);
    if (class_ops == NULL) {
        UVM_ERR_PRINT("Unsupported ce class: 0x%X, GPU %s\n", gpu->ce_class, gpu->name);
        return NV_ERR_INVALID_CLASS;
    }

    gpu->ce_hal = &class_ops->u.ce_ops;

    class_ops = ops_find_by_id(host_table, ARRAY_SIZE(host_table), gpu->host_class);
    if (class_ops == NULL) {
        UVM_ERR_PRINT("Unsupported host class: 0x%X, GPU %s\n", gpu->host_class, gpu->name);
        return NV_ERR_INVALID_CLASS;
    }

    gpu->host_hal = &class_ops->u.host_ops;

    class_ops = ops_find_by_id(arch_table, ARRAY_SIZE(arch_table), gpu->architecture);
    if (class_ops == NULL) {
        UVM_ERR_PRINT("Unsupported GPU architecture: 0x%X, GPU %s\n", gpu->architecture, gpu->name);
        return NV_ERR_INVALID_CLASS;
    }

    gpu->arch_hal = &class_ops->u.arch_ops;

    // Initialize the fault buffer hal only for GPUs supporting faults (with non-0 fault buffer class).
    if (gpu->fault_buffer_class != 0) {
        class_ops = ops_find_by_id(fault_buffer_table, ARRAY_SIZE(fault_buffer_table), gpu->fault_buffer_class);
        if (class_ops == NULL) {
            UVM_ERR_PRINT("Unsupported fault buffer class: 0x%X, GPU %s\n", gpu->fault_buffer_class, gpu->name);
            return NV_ERR_INVALID_CLASS;
        }
        gpu->fault_buffer_hal = &class_ops->u.fault_buffer_ops;
    }
    else {
        gpu->fault_buffer_hal = NULL;
    }

    // Initialize the access counter buffer hal only for GPUs supporting access counters (with non-0 access counter
    // buffer class).
    if (gpu->access_counter_buffer_class != 0) {
        class_ops = ops_find_by_id(access_counter_buffer_table,
                                   ARRAY_SIZE(access_counter_buffer_table),
                                   gpu->access_counter_buffer_class);
        if (class_ops == NULL) {
            UVM_ERR_PRINT("Unsupported access counter buffer class: 0x%X, GPU %s\n",
                          gpu->access_counter_buffer_class,
                          gpu->name);
            return NV_ERR_INVALID_CLASS;
        }
        gpu->access_counter_buffer_hal = &class_ops->u.access_counter_buffer_ops;
    }
    else {
        gpu->access_counter_buffer_hal = NULL;
    }

    return NV_OK;
}

void uvm_hal_tlb_invalidate_membar(uvm_push_t *push, uvm_membar_t membar)
{
    uvm_gpu_t *gpu;
    NvU32 i;

    if (membar == UVM_MEMBAR_NONE)
        return;

    gpu = uvm_push_get_gpu(push);

    for (i = 0; i < gpu->num_hshub_tlb_invalidate_membars; i++)
        gpu->host_hal->membar_gpu(push);

    uvm_hal_membar(gpu, push, membar);
}

const char *uvm_aperture_string(uvm_aperture_t aperture)
{
    BUILD_BUG_ON(UVM_APERTURE_MAX != 12);

    switch (aperture) {
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_0);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_1);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_2);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_3);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_4);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_5);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_6);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_7);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_PEER_MAX);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_SYS);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_VID);
        UVM_ENUM_STRING_CASE(UVM_APERTURE_DEFAULT);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_prot_string(uvm_prot_t prot)
{
    BUILD_BUG_ON(UVM_PROT_MAX != 4);

    switch (prot) {
        UVM_ENUM_STRING_CASE(UVM_PROT_NONE);
        UVM_ENUM_STRING_CASE(UVM_PROT_READ_ONLY);
        UVM_ENUM_STRING_CASE(UVM_PROT_READ_WRITE);
        UVM_ENUM_STRING_CASE(UVM_PROT_READ_WRITE_ATOMIC);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_membar_string(uvm_membar_t membar)
{
    switch (membar) {
        UVM_ENUM_STRING_CASE(UVM_MEMBAR_SYS);
        UVM_ENUM_STRING_CASE(UVM_MEMBAR_GPU);
        UVM_ENUM_STRING_CASE(UVM_MEMBAR_NONE);
    }

    return "UNKNOWN";
}

const char *uvm_fault_access_type_string(uvm_fault_access_type_t fault_access_type)
{
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT != 5);

    switch (fault_access_type) {
        UVM_ENUM_STRING_CASE(UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG);
        UVM_ENUM_STRING_CASE(UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK);
        UVM_ENUM_STRING_CASE(UVM_FAULT_ACCESS_TYPE_WRITE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_ACCESS_TYPE_READ);
        UVM_ENUM_STRING_CASE(UVM_FAULT_ACCESS_TYPE_PREFETCH);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_fault_type_string(uvm_fault_type_t fault_type)
{
    BUILD_BUG_ON(UVM_FAULT_TYPE_COUNT != 16);

    switch (fault_type) {
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_INVALID_PDE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_INVALID_PTE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_ATOMIC);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_WRITE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_READ);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_PDE_SIZE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_VA_LIMIT_VIOLATION);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_UNBOUND_INST_BLOCK);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_PRIV_VIOLATION);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_PITCH_MASK_VIOLATION);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_WORK_CREATION);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_UNSUPPORTED_APERTURE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_COMPRESSION_FAILURE);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_UNSUPPORTED_KIND);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_REGION_VIOLATION);
        UVM_ENUM_STRING_CASE(UVM_FAULT_TYPE_POISONED);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_fault_client_type_string(uvm_fault_client_type_t fault_client_type)
{
    BUILD_BUG_ON(UVM_FAULT_CLIENT_TYPE_COUNT != 2);

    switch (fault_client_type) {
        UVM_ENUM_STRING_CASE(UVM_FAULT_CLIENT_TYPE_GPC);
        UVM_ENUM_STRING_CASE(UVM_FAULT_CLIENT_TYPE_HUB);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_mmu_engine_type_string(uvm_mmu_engine_type_t mmu_engine_type)
{
    BUILD_BUG_ON(UVM_MMU_ENGINE_TYPE_COUNT != 13);

    switch (mmu_engine_type) {
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_GRAPHICS);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_DISPLAY);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_IFB);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_BAR);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_HOST);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_SEC);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_PERF);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_NVDEC);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_CE);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_PWR_PMU);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_PTP);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_NVENC);
        UVM_ENUM_STRING_CASE(UVM_MMU_ENGINE_TYPE_PHYSICAL);
        UVM_ENUM_STRING_DEFAULT();
    }
}

void uvm_hal_print_fault_entry(uvm_fault_buffer_entry_t *entry)
{
    UVM_DBG_PRINT("fault_address:                    0x%llx\n", entry->fault_address);
    UVM_DBG_PRINT("    fault_instance_ptr:           {0x%llx:%s}\n", entry->instance_ptr.address,
                                                                     uvm_aperture_string(entry->instance_ptr.aperture));
    UVM_DBG_PRINT("    fault_type:                   %s\n", uvm_fault_type_string(entry->fault_type));
    UVM_DBG_PRINT("    fault_access_type:            %s\n", uvm_fault_access_type_string(entry->fault_access_type));
    UVM_DBG_PRINT("    is_replayable:                %s\n", entry->is_replayable? "true": "false");
    UVM_DBG_PRINT("    is_virtual:                   %s\n", entry->is_virtual? "true": "false");
    UVM_DBG_PRINT("    in_protected_mode:            %s\n", entry->in_protected_mode? "true": "false");
    UVM_DBG_PRINT("    fault_source.client_type:     %s\n", uvm_fault_client_type_string(entry->fault_source.client_type));
    UVM_DBG_PRINT("    fault_source.client_id:       %d\n", entry->fault_source.client_id);
    UVM_DBG_PRINT("    fault_source.gpc_id:          %d\n", entry->fault_source.gpc_id);
    UVM_DBG_PRINT("    fault_source.mmu_engine_id:   %d\n", entry->fault_source.mmu_engine_id);
    UVM_DBG_PRINT("    fault_source.mmu_engine_type: %s\n",
                  uvm_mmu_engine_type_string(entry->fault_source.mmu_engine_type));
    UVM_DBG_PRINT("    timestamp:                    %llu\n", entry->timestamp);
}

const char *uvm_access_counter_type_string(uvm_access_counter_type_t access_counter_type)
{
    BUILD_BUG_ON(UVM_ACCESS_COUNTER_TYPE_MAX != 2);

    switch (access_counter_type) {
        UVM_ENUM_STRING_CASE(UVM_ACCESS_COUNTER_TYPE_MIMC);
        UVM_ENUM_STRING_CASE(UVM_ACCESS_COUNTER_TYPE_MOMC);
        UVM_ENUM_STRING_DEFAULT();
    }
}

void uvm_hal_print_access_counter_buffer_entry(uvm_access_counter_buffer_entry_t *entry)
{
    if (!entry->address.is_virtual) {
        UVM_DBG_PRINT("physical address: {0x%llx:%s}\n", entry->address.address,
                                                         uvm_aperture_string(entry->address.aperture));
    }
    else {
        UVM_DBG_PRINT("virtual address: 0x%llx\n", entry->address.address);
        UVM_DBG_PRINT("    instance_ptr    {0x%llx:%s}\n", entry->virtual_info.instance_ptr.address,
                                                    uvm_aperture_string(entry->virtual_info.instance_ptr.aperture));
        UVM_DBG_PRINT("    mmu_engine_type %s\n", uvm_mmu_engine_type_string(entry->virtual_info.mmu_engine_type));
        UVM_DBG_PRINT("    mmu_engine_id   %u\n", entry->virtual_info.mmu_engine_id);
        UVM_DBG_PRINT("    ve_id           %u\n", entry->virtual_info.ve_id);
    }

    UVM_DBG_PRINT("    is_virtual      %u\n", entry->address.is_virtual);
    UVM_DBG_PRINT("    counter_type    %s\n", uvm_access_counter_type_string(entry->counter_type));
    UVM_DBG_PRINT("    counter_value   %u\n", entry->counter_value);
    UVM_DBG_PRINT("    subgranularity  0x%08x\n", entry->sub_granularity);
    UVM_DBG_PRINT("    bank            %u\n", entry->bank);
    UVM_DBG_PRINT("    tag             %x\n", entry->tag);
}
