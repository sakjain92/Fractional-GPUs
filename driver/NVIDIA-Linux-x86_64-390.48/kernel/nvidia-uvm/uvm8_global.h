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

#ifndef __UVM8_GLOBAL_H__
#define __UVM8_GLOBAL_H__

#include "nv_uvm_types.h"
#include "uvm_linux.h"
#include "uvm_common.h"
#include "uvm8_processors.h"
#include "uvm8_gpu.h"
#include "uvm8_lock.h"
#include "nv-kthread-q.h"

// Global state of the uvm driver
typedef struct
{
    // Mask of retained GPUs.
    // Note that GPUs are added to this mask as the last step of add_gpu() and
    // removed from it as the first step of remove_gpu() implying that a GPU
    // that's being initialized or deinitialized will not be in it.
    uvm_processor_mask_t retained_gpus;

    // Array of the GPUs retained by uvm
    // Note that GPUs will have ids offset by 1 to accomodate the UVM_CPU_ID (0),
    // so e.g. gpus[0] will have GPU id = 1.
    // A GPU entry is unused iff it does not exist (is a NULL pointer) in this table.
    uvm_gpu_t *gpus[UVM_MAX_GPUS];

    // A global RM session (RM client)
    // Created on module load and destroyed on module unload
    uvmGpuSessionHandle rm_session_handle;

    // peer-to-peer table
    // peer info is added and removed from this table when usermode
    // driver calls UvmEnablePeerAccess and UvmDisablePeerAccess
    // respectively.
    uvm_gpu_peer_t peers[UVM_MAX_UNIQUE_GPU_PAIRS];

    // Stores an NV_STATUS, once it becomes != NV_OK, the driver should refuse to
    // do most anything other than try and clean up as much as possible.
    // An example of a fatal error is an unrecoverable ECC error on one of the
    // GPUs.
    atomic_t fatal_error;

    // A flag to disable the assert on fatal error
    // To be used by tests and only consulted if tests are enabled.
    bool disable_fatal_error_assert;

    // Lock protecting the global state
    uvm_mutex_t global_lock;

    // This lock synchronizes addition and removal of GPUs from UVM's global table.
    // It must be held whenever g_uvm_global.gpus[] is written. In order to read
    // from this table, you must hold either the gpu_table_lock, or the global_lock.
    //
    // This is a leaf lock.
    uvm_spinlock_irqsave_t gpu_table_lock;

    // Number of simulated/emulated devices that have registered with UVM
    unsigned num_simulated_devices;

    // A single queue for deferred work that is non-GPU-specific.
    nv_kthread_q_t global_q;
} uvm_global_t;

extern uvm_global_t g_uvm_global;

// Initialize global uvm state
NV_STATUS uvm_global_init(void);

// Deinitialize global state (called from module exit)
void uvm_global_exit(void);

// Get a gpu by its id.
// Returns a pointer to the GPU object, or NULL if not found.
//
// LOCKING: requires that you hold the gpu_table_lock, the global_lock, or have
// retained the gpu.
static uvm_gpu_t *uvm_gpu_get(uvm_gpu_id_t gpu_id)
{
    UVM_ASSERT_MSG(gpu_id > 0 && gpu_id < UVM_MAX_PROCESSORS, "gpu id %u\n", gpu_id);

    return g_uvm_global.gpus[uvm_gpu_index(gpu_id)];
}

static const char *uvm_processor_name(uvm_processor_id_t id)
{
    if (id == UVM_CPU_ID)
        return "0: CPU";
    else
        return uvm_gpu_get(id)->name;
}

static uvm_gpu_t *uvm_processor_mask_find_first_gpu(const uvm_processor_mask_t *mask)
{
    uvm_gpu_t *gpu;
    uvm_gpu_id_t gpu_id = uvm_processor_mask_find_next_id(mask, UVM_CPU_ID + 1);
    if (gpu_id == UVM_MAX_PROCESSORS)
        return NULL;

    gpu = uvm_gpu_get(gpu_id);

    // If there is valid GPU id in the mask, assert that the corresponding
    // uvm_gpu_t is present. Otherwise it would stop a for_each_gpu_in_mask()
    // loop pre-maturely. Today, this could only happen in remove_gpu() because
    // the GPU being removed is deleted from the global table very early.
    UVM_ASSERT_MSG(gpu, "gpu_id %u\n", gpu_id);

    return gpu;
}

static uvm_gpu_t *uvm_processor_mask_find_next_gpu(const uvm_processor_mask_t *mask, uvm_gpu_t *gpu)
{
    uvm_gpu_id_t gpu_id;
    if (gpu == NULL)
        return NULL;

    gpu_id = uvm_processor_mask_find_next_id(mask, gpu->id + 1);
    if (gpu_id == UVM_MAX_PROCESSORS)
        return NULL;

    gpu = uvm_gpu_get(gpu_id);

    // See comment in uvm_processor_mask_find_first_gpu().
    UVM_ASSERT_MSG(gpu, "gpu_id %u\n", gpu_id);

    return gpu;
}

// Helper to iterate over all GPUs (uvm_gpu_t) set in a mask. DO NOT USE if the current GPU
// GPUs can be destroyed within the loop or have been already removed from the GPU table.
#define for_each_gpu_in_mask(gpu, mask)                                 \
    for (gpu = uvm_processor_mask_find_first_gpu(mask);                 \
         gpu != NULL;                                                   \
         gpu = uvm_processor_mask_find_next_gpu(mask, gpu))

// Same as for_each_gpu_in_mask() but also asserts the passed-in rwsem is locked
#define for_each_gpu_in_mask_with_rwsem(gpu, mask, rwsem)               \
    for (({uvm_assert_rwsem_locked(rwsem);                              \
           gpu = uvm_processor_mask_find_first_gpu(mask);});            \
           gpu != NULL;                                                 \
           gpu = uvm_processor_mask_find_next_gpu(mask, gpu))

// Same as for_each_gpu_in_mask() but also asserts the passed-in mutex is locked
#define for_each_gpu_in_mask_with_mutex(gpu, mask, mutex)               \
    for (({uvm_assert_mutex_locked(mutex);                              \
           gpu = uvm_processor_mask_find_first_gpu(mask);});            \
           gpu != NULL;                                                 \
           gpu = uvm_processor_mask_find_next_gpu(mask, gpu))

// Helper to iterate over all GPUs retained by the UVM driver (across all va spaces)
#define for_each_global_gpu(gpu) \
    for_each_gpu_in_mask_with_mutex(gpu, &g_uvm_global.retained_gpus, &g_uvm_global.global_lock)

// Helper to iterate over all GPUs registered in a UVM VA space
#define for_each_va_space_gpu(gpu, va_space) \
    for_each_gpu_in_mask_with_rwsem(gpu, &va_space->registered_gpus, &va_space->lock)

static bool global_is_fatal_error_assert_disabled(void)
{
    // Only allow the assert to be disabled if tests are enabled
    if (!uvm_enable_builtin_tests)
        return false;

    return g_uvm_global.disable_fatal_error_assert;
}

// Set a global fatal error
// Once that happens the the driver should refuse to do anything other than try
// and clean up as much as possible.
// An example of a fatal error is an unrecoverable ECC error on one of the
// GPUs.
// Use a macro so that the assert below provides precise file and line info and
// a backtrace.
#define uvm_global_set_fatal_error(error)                                       \
    do {                                                                        \
        if (!global_is_fatal_error_assert_disabled())                           \
            UVM_ASSERT_MSG(0, "Fatal error: %s\n", nvstatusToString(error));    \
        uvm_global_set_fatal_error_impl(error);                                 \
    } while (0)
void uvm_global_set_fatal_error_impl(NV_STATUS error);

// Get the global status
static NV_STATUS uvm_global_get_status(void)
{
    return atomic_read(&g_uvm_global.fatal_error);
}

// Reset global fatal error
// This is to be used by tests triggering the global error on purpose only.
// Returns the value of the global error field that existed just before this
// reset call was made.
NV_STATUS uvm_global_reset_fatal_error(void);

#endif // __UVM8_GLOBAL_H__
