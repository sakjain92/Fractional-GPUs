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

#ifndef __UVM8_THREAD_CONTEXT_H__
#define __UVM8_THREAD_CONTEXT_H__

//
// UVM thread contexts
//
// UVM thread contexts provide thread local storage for all logical threads
// executing in the UVM driver. Both user and interrupt contexts are supported.
//
// Currently thread contexts are used purely for tracking lock correctness and
// store information about locks held by each thread.
//

#include "uvm8_forward_decl.h"
#include "uvm8_lock.h"
#include "uvm_common.h"
#include "uvm_linux.h"

struct uvm_thread_context_struct
{
    NvU32 ref_count;

    // The corresponding task
    // Only set for user contexts
    struct task_struct *task;

    // pid of the task
    // Only set for user contexts
    pid_t pid;

    // Opt-out of lock tracking if >0
    NvU32 skip_lock_tracking;

    // Bitmap of acquired lock orders
    DECLARE_BITMAP(acquired_lock_orders, UVM_LOCK_ORDER_COUNT);

    // Bitmap of exclusively acquired lock orders
    DECLARE_BITMAP(exclusive_acquired_lock_orders, UVM_LOCK_ORDER_COUNT);

    // Array of pointers to acquired locks. Indexed by lock order.
    void *acquired_locks[UVM_LOCK_ORDER_COUNT];
};

NV_STATUS uvm_thread_context_init(void);
void uvm_thread_context_exit(void);

// Retain the current thread context
// For user context, the first call will allocate the uvm_thread_context_t for it.
// Notably this requires the first call in user context to be done in sleepable
// context.
// For interrupt contexts, the uvm_thread_context_t is pre-allocated
// and the call will always succeed.
uvm_thread_context_t *uvm_thread_context_retain(void);

// Release the current thread context
// For user contexts, the last call matching the first retain() call will free the
// context.
void uvm_thread_context_release(void);

// Get the current thread context
// Returns NULL if context hasn't been retained
uvm_thread_context_t *uvm_thread_context(void);

// Disable lock tracking in the current thread context
// Lock tracking is enabled by default, but can be disabled by using this function.
// The disable lock tracking calls are refcounted so to enable tracking back all
// of the disable calls have to be paired with an enable call.
// This also retains the thread context and has to be paired with a later
// uvm_thread_context_enable_lock_tracking() call.
//
// This is needed in some tests that need to violate lock ordering, e.g. one of
// the push tests acquires the push sema multiple times.
void uvm_thread_context_disable_lock_tracking(void);

// Enable back lock tracking in the current thread context
// This also releases the thread context and has to be paired with a previous
// uvm_thread_context_disable_lock_tracking() call.
// The lock tracking is enabled back only when all previous disable calls have
// been paired with an enable call.
void uvm_thread_context_enable_lock_tracking(void);

#endif // __UVM8_THREAD_CONTEXT_H__
