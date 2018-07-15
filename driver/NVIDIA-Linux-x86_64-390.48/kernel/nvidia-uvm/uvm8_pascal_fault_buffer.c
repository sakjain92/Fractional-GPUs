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

#include "uvm_linux.h"
#include "uvm8_gpu.h"
#include "uvm8_hal.h"
#include "uvm8_push.h"
#include "hwref/pascal/gp100/dev_fault.h"
#include "hwref/pascal/gp100/dev_master.h"
#include "clb069.h"
#include "uvm8_pascal_fault_buffer.h"

typedef struct {
    NvU8 bufferEntry[NVB069_FAULT_BUF_SIZE];
} fault_buffer_entry_b069_t;


void uvm_hal_pascal_enable_replayable_faults(uvm_gpu_t *gpu)
{
    volatile NvU32 *reg;
    NvU32 mask;

    reg = gpu->fault_buffer_info.rm_info.replayable.pPmcIntrEnSet;
    mask = gpu->fault_buffer_info.rm_info.replayable.replayableFaultMask;

    UVM_WRITE_ONCE(*reg, mask);
}

void uvm_hal_pascal_disable_replayable_faults(uvm_gpu_t *gpu)
{
    volatile NvU32 *reg;
    NvU32 mask;

    reg = gpu->fault_buffer_info.rm_info.replayable.pPmcIntrEnClear;
    mask = gpu->fault_buffer_info.rm_info.replayable.replayableFaultMask;

    UVM_WRITE_ONCE(*reg, mask);
}

NvU32 uvm_hal_pascal_fault_buffer_read_put(uvm_gpu_t *gpu)
{
    NvU32 put = UVM_READ_ONCE(*gpu->fault_buffer_info.rm_info.replayable.pFaultBufferPut);
    UVM_ASSERT(put < gpu->fault_buffer_info.replayable.max_faults);

    return put;
}

NvU32 uvm_hal_pascal_fault_buffer_read_get(uvm_gpu_t *gpu)
{
    NvU32 get = UVM_READ_ONCE(*gpu->fault_buffer_info.rm_info.replayable.pFaultBufferGet);
    UVM_ASSERT(get < gpu->fault_buffer_info.replayable.max_faults);

    return get;
}

void uvm_hal_pascal_fault_buffer_write_get(uvm_gpu_t *gpu, NvU32 index)
{
    UVM_ASSERT(index < gpu->fault_buffer_info.replayable.max_faults);

    UVM_WRITE_ONCE(*gpu->fault_buffer_info.rm_info.replayable.pFaultBufferGet, index);
}

static uvm_fault_access_type_t get_fault_access_type(const NvU32 *fault_entry)
{
    NvU32 hw_access_type_value = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, ACCESS_TYPE);

    switch (hw_access_type_value)
    {
        case NV_PFAULT_ACCESS_TYPE_READ:
            return UVM_FAULT_ACCESS_TYPE_READ;
        case NV_PFAULT_ACCESS_TYPE_WRITE:
            return UVM_FAULT_ACCESS_TYPE_WRITE;
        case NV_PFAULT_ACCESS_TYPE_ATOMIC:
            return UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG;
        case NV_PFAULT_ACCESS_TYPE_PREFETCH:
            return UVM_FAULT_ACCESS_TYPE_PREFETCH;
    }

    UVM_ASSERT_MSG(false, "Invalid fault access type value: %d\n", hw_access_type_value);

    return UVM_FAULT_ACCESS_TYPE_COUNT;
}

static uvm_fault_type_t get_fault_type(const NvU32 *fault_entry)
{
    NvU32 hw_fault_type_value = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, FAULT_TYPE);

    switch (hw_fault_type_value)
    {
        case NV_PFAULT_FAULT_TYPE_PDE:
            return UVM_FAULT_TYPE_INVALID_PDE;
        case NV_PFAULT_FAULT_TYPE_PTE:
            return UVM_FAULT_TYPE_INVALID_PTE;
        case NV_PFAULT_FAULT_TYPE_RO_VIOLATION:
            return UVM_FAULT_TYPE_WRITE;
        case NV_PFAULT_FAULT_TYPE_ATOMIC_VIOLATION:
            return UVM_FAULT_TYPE_ATOMIC;

        case NV_PFAULT_FAULT_TYPE_PDE_SIZE:
            return UVM_FAULT_TYPE_PDE_SIZE;
        case NV_PFAULT_FAULT_TYPE_VA_LIMIT_VIOLATION:
            return UVM_FAULT_TYPE_VA_LIMIT_VIOLATION;
        case NV_PFAULT_FAULT_TYPE_UNBOUND_INST_BLOCK:
            return UVM_FAULT_TYPE_UNBOUND_INST_BLOCK;
        case NV_PFAULT_FAULT_TYPE_PRIV_VIOLATION:
            return UVM_FAULT_TYPE_PRIV_VIOLATION;
        case NV_PFAULT_FAULT_TYPE_PITCH_MASK_VIOLATION:
            return UVM_FAULT_TYPE_PITCH_MASK_VIOLATION;
        case NV_PFAULT_FAULT_TYPE_WORK_CREATION:
            return UVM_FAULT_TYPE_WORK_CREATION;
        case NV_PFAULT_FAULT_TYPE_UNSUPPORTED_APERTURE:
            return UVM_FAULT_TYPE_UNSUPPORTED_APERTURE;
        case NV_PFAULT_FAULT_TYPE_COMPRESSION_FAILURE:
            return UVM_FAULT_TYPE_COMPRESSION_FAILURE;
        case NV_PFAULT_FAULT_TYPE_UNSUPPORTED_KIND:
            return UVM_FAULT_TYPE_UNSUPPORTED_KIND;
        case NV_PFAULT_FAULT_TYPE_REGION_VIOLATION:
            return UVM_FAULT_TYPE_REGION_VIOLATION;
        case NV_PFAULT_FAULT_TYPE_POISONED:
            return UVM_FAULT_TYPE_POISONED;
    }

    UVM_ASSERT_MSG(false, "Invalid fault type value: %d\n", hw_fault_type_value);

    return UVM_FAULT_TYPE_COUNT;
}

static uvm_fault_client_type_t get_fault_client_type(const NvU32 *fault_entry)
{
    NvU32 hw_client_type_value = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, MMU_CLIENT_TYPE);

    switch (hw_client_type_value)
    {
        case NV_PFAULT_MMU_CLIENT_TYPE_GPC:
            return UVM_FAULT_CLIENT_TYPE_GPC;
        case NV_PFAULT_MMU_CLIENT_TYPE_HUB:
            return UVM_FAULT_CLIENT_TYPE_HUB;
    }

    UVM_ASSERT_MSG(false, "Invalid mmu client type value: %d\n", hw_client_type_value);

    return UVM_FAULT_CLIENT_TYPE_COUNT;
}

static NvU16 get_utlb_id_gpc(NvU16 client_id)
{
    switch (client_id)
    {
        case NV_PFAULT_CLIENT_GPC_RAST:
        case NV_PFAULT_CLIENT_GPC_GCC:
        case NV_PFAULT_CLIENT_GPC_GPCCS:
            return UVM_PASCAL_GPC_UTLB_ID_RGG;
        case NV_PFAULT_CLIENT_GPC_PE_0:
        case NV_PFAULT_CLIENT_GPC_TPCCS_0:
        case NV_PFAULT_CLIENT_GPC_L1_0:
        case NV_PFAULT_CLIENT_GPC_T1_0:
        case NV_PFAULT_CLIENT_GPC_L1_1:
        case NV_PFAULT_CLIENT_GPC_T1_1:
            return UVM_PASCAL_GPC_UTLB_ID_LTP0;
        case NV_PFAULT_CLIENT_GPC_PE_1:
        case NV_PFAULT_CLIENT_GPC_TPCCS_1:
        case NV_PFAULT_CLIENT_GPC_L1_2:
        case NV_PFAULT_CLIENT_GPC_T1_2:
        case NV_PFAULT_CLIENT_GPC_L1_3:
        case NV_PFAULT_CLIENT_GPC_T1_3:
            return UVM_PASCAL_GPC_UTLB_ID_LTP1;
        case NV_PFAULT_CLIENT_GPC_PE_2:
        case NV_PFAULT_CLIENT_GPC_TPCCS_2:
        case NV_PFAULT_CLIENT_GPC_L1_4:
        case NV_PFAULT_CLIENT_GPC_T1_4:
        case NV_PFAULT_CLIENT_GPC_L1_5:
        case NV_PFAULT_CLIENT_GPC_T1_5:
            return UVM_PASCAL_GPC_UTLB_ID_LTP2;
        case NV_PFAULT_CLIENT_GPC_PE_3:
        case NV_PFAULT_CLIENT_GPC_TPCCS_3:
        case NV_PFAULT_CLIENT_GPC_L1_6:
        case NV_PFAULT_CLIENT_GPC_T1_6:
        case NV_PFAULT_CLIENT_GPC_L1_7:
        case NV_PFAULT_CLIENT_GPC_T1_7:
            return UVM_PASCAL_GPC_UTLB_ID_LTP3;
        case NV_PFAULT_CLIENT_GPC_PE_4:
        case NV_PFAULT_CLIENT_GPC_TPCCS_4:
        case NV_PFAULT_CLIENT_GPC_L1_8:
        case NV_PFAULT_CLIENT_GPC_T1_8:
        case NV_PFAULT_CLIENT_GPC_L1_9:
        case NV_PFAULT_CLIENT_GPC_T1_9:
            return UVM_PASCAL_GPC_UTLB_ID_LTP4;
        default:
            UVM_ASSERT_MSG(false, "Invalid client value: 0x%x\n", client_id);
    }

    return 0;
}

static uvm_aperture_t get_fault_inst_aperture(NvU32 *fault_entry)
{
    NvU32 hw_aperture_value = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, INST_APERTURE);

    switch (hw_aperture_value)
    {
        case NVB069_FAULT_BUF_ENTRY_INST_APERTURE_VID_MEM:
            return UVM_APERTURE_VID;
        case NVB069_FAULT_BUF_ENTRY_INST_APERTURE_SYS_MEM_COHERENT:
        case NVB069_FAULT_BUF_ENTRY_INST_APERTURE_SYS_MEM_NONCOHERENT:
             return UVM_APERTURE_SYS;
    }

    UVM_ASSERT_MSG(false, "Invalid inst aperture value: %d\n", hw_aperture_value);

    return UVM_APERTURE_MAX;
}

static NvU32 *get_fault_buffer_entry(uvm_gpu_t *gpu, NvU32 index)
{
    fault_buffer_entry_b069_t *buffer_start;
    NvU32 *fault_entry;

    UVM_ASSERT(index < gpu->fault_buffer_info.replayable.max_faults);

    buffer_start = (fault_buffer_entry_b069_t *)gpu->fault_buffer_info.rm_info.replayable.bufferAddress;
    fault_entry = (NvU32 *)&buffer_start[index];

    return fault_entry;
}

void uvm_hal_pascal_fault_buffer_parse_entry(uvm_gpu_t *gpu, NvU32 index, uvm_fault_buffer_entry_t *buffer_entry)
{
    NV_STATUS status;
    NvU32 *fault_entry;
    NvU64 addr_hi, addr_lo;
    NvU64 timestamp_hi, timestamp_lo;
    NvU16 gpc_utlb_id;
    NvU32 utlb_id;

    status = NV_OK;

    fault_entry = get_fault_buffer_entry(gpu, index);

    // Valid bit must be set before this function is called
    UVM_ASSERT(gpu->fault_buffer_hal->entry_is_valid(gpu, index));

    addr_hi = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, INST_HI);
    addr_lo = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, INST_LO);
    buffer_entry->instance_ptr.address = addr_lo + (addr_hi << HWSIZE_MW(B069, FAULT_BUF_ENTRY, INST_LO));
    // HW value contains the 4K page number. Shift to build the full address
    buffer_entry->instance_ptr.address <<= 12;

    buffer_entry->instance_ptr.aperture = get_fault_inst_aperture(fault_entry);

    addr_hi = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, ADDR_HI);
    addr_lo = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, ADDR_LO);
    buffer_entry->fault_address = addr_lo + (addr_hi << HWSIZE_MW(B069, FAULT_BUF_ENTRY, ADDR_LO));
    buffer_entry->fault_address = uvm_address_get_canonical_form(buffer_entry->fault_address);

    timestamp_hi = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, TIMESTAMP_HI);
    timestamp_lo = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, TIMESTAMP_LO);
    buffer_entry->timestamp = timestamp_lo + (timestamp_hi << HWSIZE_MW(B069, FAULT_BUF_ENTRY, TIMESTAMP_LO));

    buffer_entry->fault_type = get_fault_type(fault_entry);

    buffer_entry->fault_access_type = get_fault_access_type(fault_entry);

    buffer_entry->fault_source.client_type = get_fault_client_type(fault_entry);
    if (buffer_entry->fault_source.client_type == UVM_FAULT_CLIENT_TYPE_HUB)
        UVM_ASSERT_MSG(false, "Invalid client type: HUB\n");

    buffer_entry->fault_source.client_id = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, CLIENT);
    BUILD_BUG_ON(sizeof(buffer_entry->fault_source.client_id) * 8 < DRF_SIZE_MW(NVB069_FAULT_BUF_ENTRY_CLIENT));

    buffer_entry->fault_source.gpc_id = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, GPC_ID);
    BUILD_BUG_ON(sizeof(buffer_entry->fault_source.gpc_id) * 8 < DRF_SIZE_MW(NVB069_FAULT_BUF_ENTRY_GPC_ID));

    gpc_utlb_id = get_utlb_id_gpc(buffer_entry->fault_source.client_id);
    UVM_ASSERT(gpc_utlb_id < uvm_pascal_get_utlbs_per_gpc(gpu));

    // Compute global uTLB id
    utlb_id = buffer_entry->fault_source.gpc_id * uvm_pascal_get_utlbs_per_gpc(gpu) + gpc_utlb_id;
    UVM_ASSERT(utlb_id < gpu->fault_buffer_info.replayable.utlb_count);

    buffer_entry->fault_source.utlb_id = utlb_id;

    buffer_entry->is_replayable = true;
    buffer_entry->is_virtual = true;
    buffer_entry->in_protected_mode = false;
    buffer_entry->fault_source.mmu_engine_type = UVM_MMU_ENGINE_TYPE_GRAPHICS;
    buffer_entry->fault_source.mmu_engine_id = NV_PFAULT_MMU_ENG_ID_GRAPHICS;
    buffer_entry->fault_source.ve_id = 0;

    // Automatically clear valid bit for the entry in the fault buffer
    uvm_hal_pascal_fault_buffer_entry_clear_valid(gpu, index);
}

bool uvm_hal_pascal_fault_buffer_entry_is_valid(uvm_gpu_t *gpu, NvU32 index)
{
    NvU32 *fault_entry;
    bool is_valid;

    fault_entry = get_fault_buffer_entry(gpu, index);

    is_valid = READ_HWVALUE_MW(fault_entry, B069, FAULT_BUF_ENTRY, VALID);

    return is_valid;
}

void uvm_hal_pascal_fault_buffer_entry_clear_valid(uvm_gpu_t *gpu, NvU32 index)
{
    NvU32 *fault_entry;

    fault_entry = get_fault_buffer_entry(gpu, index);

    WRITE_HWCONST_MW(fault_entry, B069, FAULT_BUF_ENTRY, VALID, FALSE);
}

NvU32 uvm_hal_pascal_fault_buffer_entry_size(uvm_gpu_t *gpu)
{
    return NVB069_FAULT_BUF_SIZE;
}

void uvm_hal_pascal_fault_buffer_parse_non_replayable_entry_unsupported(uvm_gpu_t *gpu,
                                                                        void *fault_packet,
                                                                        uvm_fault_buffer_entry_t *buffer_entry)
{
    UVM_ASSERT_MSG(false, "fault_buffer_parse_non_replayable_entry called on Pascal GPU\n");
}
