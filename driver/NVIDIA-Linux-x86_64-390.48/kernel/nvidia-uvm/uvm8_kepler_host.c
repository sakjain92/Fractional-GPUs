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

#include "uvm_linux.h"
#include "uvm8_hal_types.h"
#include "uvm8_hal.h"
#include "uvm8_push.h"
#include "cla06f.h"
#include "cla16f.h"

void uvm_hal_kepler_host_wait_for_idle_a06f(uvm_push_t *push)
{
    NV_PUSH_1U(A06F, SET_REFERENCE, 0);
}

void uvm_hal_kepler_host_wait_for_idle_a16f(uvm_push_t *push)
{
    NV_PUSH_1U(A16F, WFI, 0);
}

void uvm_hal_kepler_host_membar_sys(uvm_push_t *push)
{
    NV_PUSH_1U(A06F, MEM_OP_B,
       HWCONST(A06F, MEM_OP_B, OPERATION, SYSMEMBAR_FLUSH));
}

void uvm_hal_kepler_host_tlb_invalidate_all(uvm_push_t *push, uvm_gpu_phys_address_t pdb, NvU32 depth, uvm_membar_t membar)
{
    NvU32 target;

    UVM_ASSERT_MSG(pdb.aperture == UVM_APERTURE_VID || pdb.aperture == UVM_APERTURE_SYS, "aperture: %u", pdb.aperture);

    // Only Pascal+ supports invalidating down from a specific depth.
    (void)depth;

    (void)membar;

    if (pdb.aperture == UVM_APERTURE_VID)
        target = HWCONST(A06F, MEM_OP_A, TLB_INVALIDATE_TARGET, VID_MEM);
    else
        target = HWCONST(A06F, MEM_OP_A, TLB_INVALIDATE_TARGET, SYS_MEM_COHERENT);

    UVM_ASSERT_MSG(IS_ALIGNED(pdb.address, 1 << 12), "pdb 0x%llx\n", pdb.address);
    pdb.address >>= 12;

    NV_PUSH_2U(A06F, MEM_OP_A, target |
                               HWVALUE(A06F, MEM_OP_A, TLB_INVALIDATE_ADDR, pdb.address),
                     MEM_OP_B, HWCONST(A06F, MEM_OP_B, OPERATION, MMU_TLB_INVALIDATE) |
                               HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_PDB, ONE) |
                               HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_GPC, ENABLE));
}

void uvm_hal_kepler_host_tlb_invalidate_va(uvm_push_t *push, uvm_gpu_phys_address_t pdb, NvU32 depth, NvU64 base, NvU64 size, NvU32 page_size, uvm_membar_t membar)
{
    // No per VA invalidate on Kepler, redirect to invalidate all.
    uvm_push_get_gpu(push)->host_hal->tlb_invalidate_all(push, pdb, depth, membar);
}

void uvm_hal_kepler_host_tlb_invalidate_test(uvm_push_t *push, uvm_gpu_phys_address_t pdb,
                                             UVM_TEST_INVALIDATE_TLB_PARAMS *params)
{
    NvU32 target_pdb = 0;
    NvU32 pdb_mode_value;
    NvU32 invalidate_gpc_value;

    // Only Pascal+ supports invalidating down from a specific depth. We invalidate all
    if (params->target_pdb_mode == UvmTargetPdbModeOne) {
        UVM_ASSERT_MSG(IS_ALIGNED(pdb.address, 1 << 12), "pdb 0x%llx\n", pdb.address);
        UVM_ASSERT_MSG(pdb.aperture == UVM_APERTURE_VID || pdb.aperture == UVM_APERTURE_SYS,
                       "aperture: %u", pdb.aperture);
        pdb.address >>= 12;

        if (pdb.aperture == UVM_APERTURE_VID)
            target_pdb = HWCONST(A06F, MEM_OP_A, TLB_INVALIDATE_TARGET, VID_MEM);
        else
            target_pdb = HWCONST(A06F, MEM_OP_A, TLB_INVALIDATE_TARGET, SYS_MEM_COHERENT);

        target_pdb |= HWVALUE(A06F, MEM_OP_A, TLB_INVALIDATE_ADDR, pdb.address);
        pdb_mode_value = HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_PDB, ONE);
    }
    else {
        pdb_mode_value = HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_PDB, ALL);
    }

    if (params->disable_gpc_invalidate)
        invalidate_gpc_value = HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_GPC, DISABLE);
    else
        invalidate_gpc_value = HWCONST(A06F, MEM_OP_B, MMU_TLB_INVALIDATE_GPC, ENABLE);

    NV_PUSH_2U(A06F, MEM_OP_A, target_pdb,
                     MEM_OP_B, HWCONST(A06F, MEM_OP_B, OPERATION, MMU_TLB_INVALIDATE) |
                               pdb_mode_value |
                               invalidate_gpc_value);
}

void uvm_hal_kepler_host_noop(uvm_push_t *push, NvU32 size)
{
    UVM_ASSERT_MSG(size % 4 == 0, "size %u\n", size);

    if (size == 0)
        return;

    // size is in bytes so divide by the method size (4 bytes)
    size /= 4;

    while (size > 0) {
        // noop_this_time includes the NOP method itself and hence can be
        // up to COUNT_MAX + 1.
        NvU32 noop_this_time = min(UVM_METHOD_COUNT_MAX + 1, size);

        // -1 for the NOP method itself.
        NV_PUSH_NU_NONINC(A06F, NOP, noop_this_time - 1);

        size -= noop_this_time;
    }
}

void uvm_hal_kepler_host_interrupt(uvm_push_t *push)
{
    NV_PUSH_1U(A06F, NON_STALL_INTERRUPT, 0);
}

void uvm_hal_kepler_host_semaphore_release(uvm_push_t *push, uvm_gpu_semaphore_t *semaphore, NvU32 payload)
{
    uvm_gpu_t *gpu = uvm_push_get_gpu(push);
    NvU64 semaphore_va = uvm_gpu_semaphore_get_gpu_va(semaphore, gpu);
    NV_PUSH_4U(A06F, SEMAPHOREA, NvOffset_HI32(semaphore_va),
                     SEMAPHOREB, NvOffset_LO32(semaphore_va),
                     SEMAPHOREC, payload,
                     SEMAPHORED, HWCONST(A06F, SEMAPHORED, OPERATION, RELEASE) |
                                 HWCONST(A06F, SEMAPHORED, RELEASE_SIZE, 4BYTE)|
                                 HWCONST(A06F, SEMAPHORED, RELEASE_WFI, DIS));
}

void uvm_hal_kepler_host_semaphore_acquire(uvm_push_t *push, uvm_gpu_semaphore_t *semaphore, NvU32 payload)
{
    uvm_gpu_t *gpu = uvm_push_get_gpu(push);
    NvU64 semaphore_va = uvm_gpu_semaphore_get_gpu_va(semaphore, gpu);
    NV_PUSH_4U(A06F, SEMAPHOREA, NvOffset_HI32(semaphore_va),
                     SEMAPHOREB, NvOffset_LO32(semaphore_va),
                     SEMAPHOREC, payload,
                     SEMAPHORED, HWCONST(A06F, SEMAPHORED, ACQUIRE_SWITCH, ENABLED) |
                                 HWCONST(A06F, SEMAPHORED, OPERATION, ACQ_GEQ));

}

void uvm_hal_kepler_host_set_gpfifo_entry(NvU64 *fifo_entry, NvU64 pushbuffer_va, NvU32 pushbuffer_length)
{
    NvU64 fifo_entry_value;

    UVM_ASSERT_MSG(pushbuffer_va % 4 == 0, "pushbuffer va unaligned: %llu\n", pushbuffer_va);
    UVM_ASSERT_MSG(pushbuffer_length % 4 == 0, "pushbuffer length unaligned: %u\n", pushbuffer_length);

    fifo_entry_value =          HWVALUE(A06F, GP_ENTRY0, GET, NvU64_LO32(pushbuffer_va) >> 2);
    fifo_entry_value |= (NvU64)(HWVALUE(A06F, GP_ENTRY1, GET_HI, NvU64_HI32(pushbuffer_va)) |
                                HWVALUE(A06F, GP_ENTRY1, LENGTH, pushbuffer_length >> 2) |
                                HWCONST(A06F, GP_ENTRY1, PRIV,   KERNEL)) << 32;

    *fifo_entry = fifo_entry_value;
}

void uvm_hal_kepler_host_write_gpu_put(uvm_channel_t *channel, NvU32 gpu_put)
{
    UVM_WRITE_ONCE(*channel->channel_info.gpPut, gpu_put);
}

void uvm_hal_kepler_host_init_noop(uvm_push_t *push)
{
}

void uvm_hal_kepler_replay_faults_unsupported(uvm_push_t *push, uvm_fault_replay_type_t type)
{
    UVM_ASSERT_MSG(false, "host replay_faults called on Kepler GPU\n");
}

void uvm_hal_kepler_cancel_faults_targeted_unsupported(uvm_push_t *push, uvm_gpu_phys_address_t instance_ptr,
                                                       NvU32 gpc_id, NvU32 client_id)
{
    UVM_ASSERT_MSG(false, "host cancel_faults_targeted called on Kepler GPU\n");
}

void uvm_hal_kepler_cancel_faults_va_unsupported(uvm_push_t *push,
                                                 uvm_gpu_phys_address_t pdb,
                                                 const uvm_fault_buffer_entry_t *fault_entry,
                                                 uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    UVM_ASSERT_MSG(false, "host cancel_faults_va called on Kepler GPU\n");
}

void uvm_hal_kepler_host_clear_faulted_channel_unsupported(uvm_push_t *push,
                                                           uvm_user_channel_t *user_channel,
                                                           uvm_fault_buffer_entry_t *buffer_entry)
{
    UVM_ASSERT_MSG(false, "host clear_faulted_channel called on Kepler GPU\n");
}

void uvm_hal_kepler_access_counter_clear_all_unsupported(uvm_push_t *push)
{
    UVM_ASSERT_MSG(false, "host access_counter_clear_all called on Kepler GPU\n");
}

void uvm_hal_kepler_access_counter_clear_type_unsupported(uvm_push_t *push, uvm_access_counter_type_t type)
{
    UVM_ASSERT_MSG(false, "host access_counter_clear_type called on Kepler GPU\n");
}

void uvm_hal_kepler_access_counter_clear_targeted_unsupported(uvm_push_t *push,
                                                              uvm_access_counter_buffer_entry_t *buffer_entry)
{
    UVM_ASSERT_MSG(false, "host access_counter_clear_targeted called on Kepler GPU\n");
}
