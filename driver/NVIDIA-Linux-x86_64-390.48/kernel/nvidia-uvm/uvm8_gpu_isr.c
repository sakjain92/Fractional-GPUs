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

#include "uvm8_global.h"
#include "uvm8_gpu_isr.h"
#include "uvm8_hal.h"
#include "uvm8_gpu.h"
#include "uvm8_gpu_access_counters.h"
#include "uvm8_gpu_non_replayable_faults.h"

// For use by the nv_kthread_q that is servicing the replayable fault bottom
// half, only.
static void replayable_faults_isr_bottom_half(void *args);

// For use by the nv_kthread_q that is servicing the replayable fault bottom
// half, only.
static void non_replayable_faults_isr_bottom_half(void *args);

// For use by the nv_kthread_q that is servicing the replayable fault bottom
// half, only.
static void access_counters_isr_bottom_half(void *args);

// Increments the reference count tracking whether replayable page fault
// interrupts should be enabled. The caller is guaranteed that replayable page
// faults are disabled upon return. Interrupts might already be disabled prior
// to making this call. Each call is ref-counted, so this must be paired with a
// call to uvm_gpu_replayable_faults_enable().
//
// gpu->isr.interrupts_lock must be held to call this function.
static void uvm_gpu_replayable_faults_disable(uvm_gpu_t *gpu);

// Decrements the reference count tracking whether replayable page fault
// interrupts should be enabled. Only once the count reaches 0 are the HW
// interrupts actually enabled, so this call does not guarantee that the
// interrupts have been re-enabled upon return.
//
// uvm_gpu_replayable_faults_disable() must have been called prior to calling
// this function.
//
// gpu->isr.interrupts_lock must be held to call this function.
static void uvm_gpu_replayable_faults_enable(uvm_gpu_t *gpu);

// Like uvm_gpu_replayable_faults_disable/uvm_gpu_replayable_faults_enable, but
// operates on access_counters interrupts.
static void uvm_gpu_access_counters_disable(uvm_gpu_t *gpu);
static void uvm_gpu_access_counters_enable(uvm_gpu_t *gpu);

static unsigned schedule_replayable_faults_handler(uvm_gpu_t *gpu)
{
    // handling gets set to false for all handlers during removal, so quit if
    // the GPU is in the process of being removed. 
    if (gpu->isr.replayable_faults.handling) {
        // TODO: Bug 1766600: add support to lockdep, for leaving this lock
        //       acquired (the bottom half eventually releases it).
        if (mutex_trylock(&gpu->isr.replayable_faults.service_lock.m) != 0) {
            if (uvm_gpu_replayable_faults_pending(gpu)) {
                nv_kref_get(&gpu->gpu_kref);

                uvm_gpu_replayable_faults_disable(gpu);

                // Schedule a bottom half, but do *not* release the GPU ISR
                // lock. The bottom half releases the GPU ISR lock as part of
                // its cleanup.
                nv_kthread_q_schedule_q_item(&gpu->isr.bottom_half_q,
                                             &gpu->isr.replayable_faults.bottom_half_q_item);
                return 1;
            }
            else {
                mutex_unlock(&gpu->isr.replayable_faults.service_lock.m);
            }
        }
    }

    return 0;
}

static unsigned schedule_non_replayable_faults_handler(uvm_gpu_t *gpu)
{
    // handling gets set to false for all handlers during removal, so quit if
    // the GPU is in the process of being removed. 
    if (gpu->isr.non_replayable_faults.handling) {
        // Non-replayable_faults are stored in a synchronized circular queue
        // shared by RM/UVM. Therefore, we can query the number of pending
        // faults. This type of faults are not replayed and since RM advances
        // GET to PUT when copying the fault packets to the queue, no further
        // interrupts will be triggered by the gpu and faults may stay
        // unserviced. Therefore, if there is a fault in the queue, we schedule
        // a bottom half unconditionally.
        if (uvm_gpu_non_replayable_faults_pending(gpu)) {
            bool scheduled;
            nv_kref_get(&gpu->gpu_kref);

            scheduled = nv_kthread_q_schedule_q_item(&gpu->isr.bottom_half_q,
                                                     &gpu->isr.non_replayable_faults.bottom_half_q_item) != 0;

            // If the q_item did not get scheduled because it was already
            // queued, that instance will handle the pending faults. Just
            // drop the GPU kref.
            if (!scheduled)
                uvm_gpu_kref_put(gpu);

            return 1;
        }
    }

    return 0;
}

static unsigned schedule_access_counters_handler(uvm_gpu_t *gpu)
{
    // See schedule_replayable_faults_handler for more details
    if (gpu->isr.access_counters.handling) {
        if (mutex_trylock(&gpu->isr.access_counters.service_lock.m) != 0) {
            if (uvm_gpu_access_counters_pending(gpu)) {
                nv_kref_get(&gpu->gpu_kref);

                uvm_gpu_access_counters_disable(gpu);

                nv_kthread_q_schedule_q_item(&gpu->isr.bottom_half_q,
                                             &gpu->isr.access_counters.bottom_half_q_item);

                return 1;
            }
            else {
                mutex_unlock(&gpu->isr.access_counters.service_lock.m);
            }
        }
    }

    return 0;
}

// This is called from RM's top-half ISR (see: the nvidia_isr() function), and UVM is given a
// chance to handle the interrupt, before most of the RM processing. UVM communicates what it
// did, back to RM, via the return code:
//
//     NV_OK:
//         UVM handled an interrupt.
//
//     NV_WARN_MORE_PROCESSING_REQUIRED:
//         UVM did not schedule a bottom half, because it was unable to get the locks it
//         needed, but there is still UVM work to be done. RM will return "not handled" to the
//         Linux kernel, *unless* RM handled other faults in its top half. In that case, the
//         fact that UVM did not handle its interrupt is lost. However, life and interrupt
//         processing continues anyway: the GPU will soon raise another interrupt, because
//         that's what it does when there are replayable page faults remaining (GET != PUT in
//         the fault buffer).
//
//     NV_ERR_NO_INTR_PENDING:
//         UVM did not find any work to do. Currently this is handled in RM in exactly the same
//         way as NV_WARN_MORE_PROCESSING_REQUIRED is handled. However, the extra precision is
//         available for the future. RM's interrupt handling tends to evolve as new chips and
//         new interrupts get created.

NV_STATUS uvm8_isr_top_half(const NvProcessorUuid *gpu_uuid)
{
    uvm_gpu_t *gpu;
    unsigned num_handlers_scheduled = 0;

    if (!in_interrupt()) {
        // Early-out if not in interrupt context. This happens with
        // CONFIG_DEBUG_SHIRQ enabled where the interrupt handler is called as
        // part of its removal to make sure it's prepared for being called even
        // when it's being freed.
        // This breaks the assumption that the UVM driver is called in atomic
        // context only in the interrupt context, which
        // uvm_thread_context_retain() relies on.
        return NV_OK;
    }

    if (!gpu_uuid) {
        // This can happen early in the main GPU driver initialization, because
        // that involves testing interrupts before the GPU is fully set up.
        return NV_ERR_NO_INTR_PENDING;
    }

    uvm_spin_lock_irqsave(&g_uvm_global.gpu_table_lock);

    gpu = uvm_gpu_get_by_uuid_locked(gpu_uuid);

    if (gpu == NULL) {
        uvm_spin_unlock_irqrestore(&g_uvm_global.gpu_table_lock);
        return NV_ERR_NO_INTR_PENDING;
    }

    // We take a reference during the top half, and an additional reference for
    // each scheduled bottom. References are dropped at the end of the bottom
    // halves.
    nv_kref_get(&gpu->gpu_kref);
    uvm_spin_unlock_irqrestore(&g_uvm_global.gpu_table_lock);

    // We don't need an atomic to increment this count since only this top half
    // writes it, and only one top half can run per GPU at a time.
    ++gpu->isr.interrupt_count;

    // Now that we got a GPU object, lock it so that it can't be removed without us noticing.
    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    num_handlers_scheduled += schedule_replayable_faults_handler(gpu);
    num_handlers_scheduled += schedule_non_replayable_faults_handler(gpu);
    num_handlers_scheduled += schedule_access_counters_handler(gpu);

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);

    uvm_gpu_kref_put(gpu);

    if (num_handlers_scheduled == 0)
        return NV_WARN_MORE_PROCESSING_REQUIRED;
    else
        return NV_OK;
}

NV_STATUS uvm_gpu_init_isr(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    char kthread_name[TASK_COMM_LEN + 1];

    if (gpu->replayable_faults_supported) {
        status = uvm_gpu_fault_buffer_init(gpu);
        if (status != NV_OK) {
            UVM_ERR_PRINT("Failed to initialize GPU fault buffer: %s, GPU: %s\n", nvstatusToString(status), gpu->name);
            return status;
        }

        nv_kthread_q_item_init(&gpu->isr.replayable_faults.bottom_half_q_item,
                               replayable_faults_isr_bottom_half,
                               gpu);

        gpu->isr.replayable_faults.handling = true;

        if (gpu->non_replayable_faults_supported) {
            nv_kthread_q_item_init(&gpu->isr.non_replayable_faults.bottom_half_q_item,
                                   non_replayable_faults_isr_bottom_half,
                                   gpu);

            gpu->isr.non_replayable_faults.handling = true;
        }

        if (gpu->access_counters_supported) {
            status = uvm_gpu_init_access_counters(gpu);
            // TODO: Bug 2000358: Remove this WAR when we are done with Turing emulation
            if (status == NV_OK) {
                nv_kthread_q_item_init(&gpu->isr.access_counters.bottom_half_q_item,
                                       access_counters_isr_bottom_half,
                                       gpu);

                gpu->isr.access_counters.handling = true;
            }
            else if (status == NV_ERR_FEATURE_NOT_ENABLED) {
                UVM_ERR_PRINT("GPU access counters not enabled: %s, GPU: %s\n",
                              nvstatusToString(status),
                              gpu->name);
            }
            else {
                UVM_ERR_PRINT("Failed to initialize GPU access counters: %s, GPU: %s\n",
                              nvstatusToString(status),
                              gpu->name);
                return status;
            }
        }

        snprintf(kthread_name, sizeof(kthread_name), "UVM GPU%u BH", gpu->id);

        status = errno_to_nv_status(nv_kthread_q_init(&gpu->isr.bottom_half_q, kthread_name));
        if (status != NV_OK) {
            UVM_ERR_PRINT("Failed in nv_kthread_q_init for bottom_half_q: %s, GPU %s\n",
                          nvstatusToString(status),
                          gpu->name);
            return status;
        }

        if (gpu->isr.non_replayable_faults.handling) {
            snprintf(kthread_name, sizeof(kthread_name), "UVM GPU%u KC", gpu->id);

            status = errno_to_nv_status(nv_kthread_q_init(&gpu->isr.kill_channel_q, kthread_name));
            if (status != NV_OK) {
                UVM_ERR_PRINT("Failed in nv_kthread_q_init for kill_channel_q: %s, GPU %s\n",
                              nvstatusToString(status),
                              gpu->name);
                return status;
            }
        }
    }

    return NV_OK;
}

void uvm_gpu_disable_isr(uvm_gpu_t *gpu)
{
    UVM_ASSERT(g_uvm_global.gpus[uvm_gpu_index(gpu->id)] != gpu);

    // Now that the GPU is safely out of the global table, lock the GPU and mark
    // it as no longer handling interrupts so the top half knows not to schedule
    // any more bottom halves.
    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    if (gpu->isr.replayable_faults.handling)
        uvm_gpu_replayable_faults_disable(gpu);

    if (gpu->isr.access_counters.handling)
        uvm_gpu_access_counters_disable(gpu);

    gpu->isr.replayable_faults.was_handling = gpu->isr.replayable_faults.handling;
    gpu->isr.non_replayable_faults.was_handling = gpu->isr.non_replayable_faults.handling;
    gpu->isr.access_counters.was_handling = gpu->isr.access_counters.handling;

    gpu->isr.replayable_faults.handling = false;
    gpu->isr.non_replayable_faults.handling = false;
    gpu->isr.access_counters.handling = false;

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);

    // Flush all bottom half ISR work items and stop the nv_kthread_q that is
    // servicing this GPU's bottom halves. Note that this requires that the
    // bottom half never take the global lock, since we're holding it here.
    if (gpu->isr.access_counters.was_handling ||
        gpu->isr.non_replayable_faults.was_handling ||
        gpu->isr.replayable_faults.was_handling) {
        // Note that it's safe to call nv_kthread_q_stop() even if
        // nv_kthread_q_init() failed in uvm_gpu_init_isr().
        nv_kthread_q_stop(&gpu->isr.bottom_half_q);
    }

    if (gpu->isr.non_replayable_faults.was_handling)
        nv_kthread_q_stop(&gpu->isr.kill_channel_q);
}

void uvm_gpu_deinit_isr(uvm_gpu_t *gpu)
{
    // Return ownership to RM:
    if (gpu->isr.replayable_faults.was_handling || gpu->isr.non_replayable_faults.was_handling) {
        // No user threads could have anything left on replayable_faults.disable_intr_ref_count
        // since they must retain the GPU across uvm_gpu_replayable_faults_isr_lock/
        // uvm_gpu_replayable_faults_isr_unlock. This means the uvm_gpu_replayable_faults_disable
        // above could only have raced with bottom halves.
        //
        // If we cleared replayable_faults.handling before the bottom half
        // got to its uvm_gpu_replayable_faults_isr_unlock, when it eventually reached
        // uvm_gpu_replayable_faults_isr_unlock it would have skipped the disable, leaving us with
        // extra ref counts here.
        //
        // In any case we're guaranteed that replayable faults interrupts are disabled
        // and can't get re-enabled, so we can safely ignore the ref count value
        // and just clean things up.
        if (gpu->isr.replayable_faults.was_handling) {
            UVM_ASSERT_MSG(gpu->isr.replayable_faults.disable_intr_ref_count > 0,
                           "%s replayable_faults.disable_intr_ref_count: %llu\n",
                           gpu->name, gpu->isr.replayable_faults.disable_intr_ref_count);
        }
        uvm_gpu_fault_buffer_deinit(gpu);
    }

    if (gpu->isr.access_counters.was_handling) {
        // See above comment on replayable_faults.disable_intr_ref_count
        UVM_ASSERT_MSG(gpu->isr.access_counters.disable_intr_ref_count > 0,
                       "%s access_counters.disable_intr_ref_count: %llu\n",
                       gpu->name, gpu->isr.access_counters.disable_intr_ref_count);
        uvm_gpu_deinit_access_counters(gpu);
    }
}

static void replayable_faults_isr_bottom_half(void *args)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)args;

    UVM_ASSERT(gpu->replayable_faults_supported);

    // Multiple bottom halves for replayable faults can be running
    // concurrently, but only one can be running this function for a given GPU
    // since we enter with the replayable_faults.service_lock held.
    ++gpu->isr.replayable_faults.bottom_half_count;

    uvm_gpu_service_replayable_faults(gpu);

    uvm_gpu_replayable_faults_isr_unlock(gpu);
    uvm_gpu_kref_put(gpu);
}

static void non_replayable_faults_isr_bottom_half(void *args)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)args;

    UVM_ASSERT(gpu->non_replayable_faults_supported);

    uvm_gpu_non_replayable_faults_isr_lock(gpu);

    // Multiple bottom halves for non-replayable faults can be running
    // concurrently, but only one can enter this section for a given GPU
    // since we acquired the non_replayable_faults.service_lock
    ++gpu->isr.non_replayable_faults.bottom_half_count;

    uvm_gpu_service_non_replayable_fault_buffer(gpu);

    uvm_gpu_non_replayable_faults_isr_unlock(gpu);
    uvm_gpu_kref_put(gpu);
}

static void access_counters_isr_bottom_half(void *args)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)args;

    UVM_ASSERT(gpu->access_counters_supported);

    // Multiple bottom halves for counter notifications can be running
    // concurrently, but only one can be running this function for a given GPU
    // since we enter with the access_counters_isr_lock held.
    ++gpu->isr.access_counters.bottom_half_count;

    uvm_gpu_service_access_counters(gpu);

    uvm_gpu_access_counters_isr_unlock(gpu);
    uvm_gpu_kref_put(gpu);
}

void uvm_gpu_replayable_faults_isr_lock(uvm_gpu_t *gpu)
{
    UVM_ASSERT(uvm_gpu_retained_count(gpu) > 0);

    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    // Bump the disable ref count. This guarantees that the bottom half or
    // another thread trying to take the replayable_faults.service_lock won't inadvertently re-enable
    // interrupts during this locking sequence.
    if (gpu->isr.replayable_faults.handling)
        uvm_gpu_replayable_faults_disable(gpu);

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);

    // Now that we know replayable fault interrupts can't get enabled, take the
    // lock. This has to be a raw call without the uvm_lock.h wrappers: although
    // this function is called from non-interrupt context, the corresponding
    // uvm_gpu_replayable_faults_isr_unlock() function is also used by the bottom half, which
    // pairs its unlock with the raw call in the top half.
    mutex_lock(&gpu->isr.replayable_faults.service_lock.m);
}

void uvm_gpu_replayable_faults_isr_unlock(uvm_gpu_t *gpu)
{
    UVM_ASSERT(nv_kref_read(&gpu->gpu_kref) > 0);

    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    // The following sequence is delicate:
    //
    //     1) Enable replayable page fault interrupts
    //     2) Unlock GPU isr.replayable_faults.service_lock (mutex)
    //     3) Unlock isr.interrupts_lock (spin lock)
    //
    // ...because the moment that page fault interrupts are reenabled, the top half will start
    // receiving them. As the gpu->isr.replayable_faults.service_lock is still held, the top half
    // will start returning NV_WARN_MORE_PROCESSING_REQUIRED, due to failing the attempted
    // mutex_trylock(&gpu->replayable_faults.service_lock). This can lead to an interrupt storm, which
    // will cripple the system, and often cause Linux to permanently disable the GPU's interrupt
    // line.
    //
    // In order to avoid such an interrupt storm, the gpu->isr.interrupts_lock
    // (which is acquired via spinlock_irqsave, thus disabling local CPU interrupts) is held until
    // after releasing the ISR mutex. That way, once local interrupts are enabled, the mutex
    // is available for the top half to acquire. This avoids a storm on the local CPU, but
    // still allows a small window of high interrupts to occur, if another CPU handles the
    // interrupt. However, in that case, the local CPU is not being slowed down
    // (interrupted), and you'll notice that the very next instruction after enabling page
    // fault interrupts is to unlock the ISR mutex. Such a small window may allow a few
    // interrupts, but not enough to be any sort of problem.

    if (gpu->isr.replayable_faults.handling) {
        // Turn page fault interrupts back on, unless remove_gpu() has already removed this GPU
        // from the GPU table. remove_gpu() indicates that situation by setting
        // gpu->replayable_faults.handling to false.
        //
        // This path can only be taken from the bottom half. User threads
        // calling this function must have previously retained the GPU, so they
        // can't race with remove_gpu.
        //
        // TODO: Bug 1766600: Assert that we're in a bottom half thread, once
        //       that's tracked by the lock assertion code.
        //
        // Note that if we're in the bottom half and the GPU was removed before
        // we checked replayable_faults.handling, we won't drop our interrupt
        // disable ref count from the corresponding top-half call to
        // uvm_gpu_replayable_faults_disable. That's ok because remove_gpu
        // ignores the refcount after waiting for the bottom half to finish.
        uvm_gpu_replayable_faults_enable(gpu);
    }

    // Raw unlock call, to correspond to the raw lock call in the top half:
    mutex_unlock(&gpu->isr.replayable_faults.service_lock.m);

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);
}

void uvm_gpu_non_replayable_faults_isr_lock(uvm_gpu_t *gpu)
{
    // TODO: Bug 1766600: This function is called from the bottom half, and
    // the following assert cannot be enabled. For the moment, we check gpu_kref
    // instead.
    // UVM_ASSERT(uvm_gpu_retained_count(gpu) > 0);
    UVM_ASSERT(nv_kref_read(&gpu->gpu_kref) > 0);

    // We can use UVM locking routines for next faults because
    // gpu->isr.non_replayable_faults.service_lock is acquired/released from the
    // same execution thread.
    uvm_mutex_lock(&gpu->isr.non_replayable_faults.service_lock);
}

void uvm_gpu_non_replayable_faults_isr_unlock(uvm_gpu_t *gpu)
{
    UVM_ASSERT(nv_kref_read(&gpu->gpu_kref) > 0);

    uvm_mutex_unlock(&gpu->isr.non_replayable_faults.service_lock);
}

void uvm_gpu_access_counters_isr_lock(uvm_gpu_t *gpu)
{
    UVM_ASSERT(uvm_gpu_retained_count(gpu) > 0);

    // See comments in uvm_gpu_replayable_faults_isr_unlock

    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    if (gpu->isr.access_counters.handling)
        uvm_gpu_access_counters_disable(gpu);

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);

    mutex_lock(&gpu->isr.access_counters.service_lock.m);
}

void uvm_gpu_access_counters_isr_unlock(uvm_gpu_t *gpu)
{
    UVM_ASSERT(nv_kref_read(&gpu->gpu_kref) > 0);

    uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

    // See comments in uvm_gpu_replayable_faults_isr_unlock

    if (gpu->isr.access_counters.handling)
        uvm_gpu_access_counters_enable(gpu);

    mutex_unlock(&gpu->isr.access_counters.service_lock.m);

    uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);
}

static void uvm_gpu_replayable_faults_disable(uvm_gpu_t *gpu)
{
    uvm_assert_spinlock_locked(&gpu->isr.interrupts_lock);
    UVM_ASSERT(gpu->isr.replayable_faults.handling);

    if (gpu->isr.replayable_faults.disable_intr_ref_count == 0)
        gpu->fault_buffer_hal->disable_replayable_faults(gpu);

    ++gpu->isr.replayable_faults.disable_intr_ref_count;
}

static void uvm_gpu_replayable_faults_enable(uvm_gpu_t *gpu)
{
    uvm_assert_spinlock_locked(&gpu->isr.interrupts_lock);
    UVM_ASSERT(gpu->isr.replayable_faults.handling);
    UVM_ASSERT(gpu->isr.replayable_faults.disable_intr_ref_count > 0);

    --gpu->isr.replayable_faults.disable_intr_ref_count;
    if (gpu->isr.replayable_faults.disable_intr_ref_count == 0)
        gpu->fault_buffer_hal->enable_replayable_faults(gpu);
}

static void uvm_gpu_access_counters_disable(uvm_gpu_t *gpu)
{
    uvm_assert_spinlock_locked(&gpu->isr.interrupts_lock);
    UVM_ASSERT(gpu->isr.access_counters.handling);

    if (gpu->isr.access_counters.disable_intr_ref_count == 0)
        gpu->access_counter_buffer_hal->disable_access_counter_notifications(gpu);

    ++gpu->isr.access_counters.disable_intr_ref_count;
}

static void uvm_gpu_access_counters_enable(uvm_gpu_t *gpu)
{
    uvm_assert_spinlock_locked(&gpu->isr.interrupts_lock);
    UVM_ASSERT(gpu->isr.access_counters.handling);
    UVM_ASSERT(gpu->isr.access_counters.disable_intr_ref_count > 0);

    --gpu->isr.access_counters.disable_intr_ref_count;
    if (gpu->isr.access_counters.disable_intr_ref_count == 0)
        gpu->access_counter_buffer_hal->enable_access_counter_notifications(gpu);
}
