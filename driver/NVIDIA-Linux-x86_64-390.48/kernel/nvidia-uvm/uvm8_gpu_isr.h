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

#ifndef __UVM8_GPU_ISR_H__
#define __UVM8_GPU_ISR_H__

#include "nv-kthread-q.h"
#include "uvm_common.h"
#include "uvm8_lock.h"
#include "uvm8_forward_decl.h"

// ISR handling state for a specific interrupt type
typedef struct
{
    // Protects against changes to the GPU data structures used by the handling routines of this
    // interrupt type.
    uvm_mutex_t service_lock;

    // Bottom-half to be executed for this interrupt. There is one bottom-half per interrupt type.
    nv_kthread_q_item_t bottom_half_q_item;

    // This is set to true during add_gpu(), if the GPU supports the interrupt. It is set
    // back to false during remove_gpu(). interrupts_lock must be held in order to write
    // this variable.
    bool handling;

    // Variable set in uvm_gpu_disable_isr() during remove_gpu() to indicate if this type of
    // interrupt was being handled by the driver
    bool was_handling;

    // Number of the bottom-half invocations for this interrupt on a GPU over its
    // lifetime
    NvU64 bottom_half_count;

    // Number of times the function that disables this type of interrupt has been called without
    // a corresponding call to the function that enables it. If this is > 0, interrupts are
    // disabled. This field is protected by interrupts_lock. This field is only valid for interrupts
    // directly owned by UVM:
    // - replayable_faults
    // - access_counters
    NvU64 disable_intr_ref_count;
} uvm_intr_handler_t;

// State for all ISR handling in UVM
typedef struct
{
    // There is exactly one nv_kthread_q per GPU. It is used for the ISR bottom
    // halves. So N CPUs will be servicing M GPUs, in general. There is one bottom-half
    // per interrupt type.
    nv_kthread_q_t bottom_half_q;

    // Protects the state of interrupts (enabled/disabled) and whether the GPU is
    // currently handling them. Taken in both interrupt and process context.
    uvm_spinlock_irqsave_t interrupts_lock;

    uvm_intr_handler_t replayable_faults;
    uvm_intr_handler_t non_replayable_faults;
    uvm_intr_handler_t access_counters;

    // Kernel thread used to kill channels on fatal non-replayable faults.
    // This is needed because we cannot call into RM from the bottom-half to
    // avoid deadlocks.
    nv_kthread_q_t kill_channel_q;

    // Number of top-half ISRs called for this GPU over its lifetime
    NvU64 interrupt_count;
} uvm_isr_info_t;

// Entry point for interrupt handling. This is called from RM's top half
NV_STATUS uvm8_isr_top_half(const NvProcessorUuid *gpu_uuid);

// Initialize ISR handling state
NV_STATUS uvm_gpu_init_isr(uvm_gpu_t *gpu);

// Prevent new bottom halves from being scheduled. This is called during GPU
// removal
void uvm_gpu_disable_isr(uvm_gpu_t *gpu);

// Destroy ISR handling state and return interrupt ownership to RM. This is
// called during GPU removal
void uvm_gpu_deinit_isr(uvm_gpu_t *gpu);

// Take gpu->isr.replayable_faults.lock from a non-top/bottom half thread.
// This will also disable replayable page fault interrupts (if supported by the
// GPU) because the top half attempts to take this lock, and we would cause an
// interrupt storm if we didn't disable them first.
//
// The GPU must have been previously retained.
void uvm_gpu_replayable_faults_isr_lock(uvm_gpu_t *gpu);

// Unlock gpu->isr.replayable_faults.lock, possibly re-enabling replayable page
// fault interrupts. Unlike uvm_gpu_replayable_faults_isr_lock(), which should only
// called from non-top/bottom half threads, this can be called by any thread.
void uvm_gpu_replayable_faults_isr_unlock(uvm_gpu_t *gpu);

// Lock/unlock routines for non-replayable faults. These do not need to prevent
// interrupt storms since the GPU fault buffers for non-replayable faults are
// managed by RM. Unlike uvm_gpu_replayable_faults_isr_lock, the GPU does not
// need to have been previously retained.
void uvm_gpu_non_replayable_faults_isr_lock(uvm_gpu_t *gpu);
void uvm_gpu_non_replayable_faults_isr_unlock(uvm_gpu_t *gpu);

// See uvm_gpu_replayable_faults_isr_lock/unlock
void uvm_gpu_access_counters_isr_lock(uvm_gpu_t *gpu);
void uvm_gpu_access_counters_isr_unlock(uvm_gpu_t *gpu);

#endif // __UVM8_GPU_ISR_H__
