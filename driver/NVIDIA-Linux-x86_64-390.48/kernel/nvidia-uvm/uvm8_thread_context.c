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

#include "uvm8_forward_decl.h"
#include "uvm8_thread_context.h"

#include "uvm_linux.h"
#include "uvm_common.h"

// Use a raw spinlock as thread contexts are used for lock tracking
static spinlock_t g_lock;

// Radix tree for user contexts, mapping get_current()->pid to a uvm_thread_context_t.
static struct radix_tree_root g_user_context_tree;

// Cache for allocating uvm_thread_context_t
static struct kmem_cache *g_uvm_thread_context_cache __read_mostly;

// Per cpu uvm_thread_context_t used for interrupt context
static DEFINE_PER_CPU(uvm_thread_context_t, interrupt_thread_context);

NV_STATUS uvm_thread_context_init(void)
{
    spin_lock_init(&g_lock);
    uvm_init_radix_tree_preloadable(&g_user_context_tree);

    g_uvm_thread_context_cache = NV_KMEM_CACHE_CREATE("uvm_thread_context_t", uvm_thread_context_t);
    if (!g_uvm_thread_context_cache)
        return NV_ERR_NO_MEMORY;

    return NV_OK;
}

void uvm_thread_context_exit(void)
{
    uvm_thread_context_t *thread_context;

    while (radix_tree_gang_lookup(&g_user_context_tree, (void**)&thread_context, 0, 1)) {
        radix_tree_delete(&g_user_context_tree, thread_context->pid);
        UVM_ERR_PRINT("Left-over thread_context 0x%llx pid %u\n", (NvU64)thread_context, thread_context->pid);
        UVM_ASSERT(__uvm_check_all_unlocked(thread_context));
        kmem_cache_free(g_uvm_thread_context_cache, thread_context);
    }

    kmem_cache_destroy_safe(&g_uvm_thread_context_cache);
}

static uvm_thread_context_t *uvm_thread_context_user(void)
{
    unsigned long flags;
    unsigned long key;
    uvm_thread_context_t *thread_context;

    key = (unsigned long)get_current()->pid;

    spin_lock_irqsave(&g_lock, flags);
    thread_context = (uvm_thread_context_t *)radix_tree_lookup(&g_user_context_tree, key);
    spin_unlock_irqrestore(&g_lock, flags);

    return thread_context;
}

static uvm_thread_context_t *uvm_thread_context_user_retain(void)
{
    unsigned long flags;
    uvm_thread_context_t *thread_context = uvm_thread_context_user();

    if (thread_context == NULL) {
        int ret;
        thread_context = kmem_cache_zalloc(g_uvm_thread_context_cache, NV_UVM_GFP_FLAGS);
        if (thread_context == NULL)
            return NULL;

        thread_context->task = get_current();
        thread_context->pid = get_current()->pid;

        // Preload allocates nodes into a per-cpu cache and disables preemption
        // on success so that they are guaranteed to be available for the next
        // insert operation. Preemption is re-enabled with preload_end() later.
        ret = radix_tree_preload(NV_UVM_GFP_FLAGS);
        if (ret != 0) {
            kmem_cache_free(g_uvm_thread_context_cache, thread_context);
            return NULL;
        }

        spin_lock_irqsave(&g_lock, flags);
        // After preloading this should always succeed
        ret = radix_tree_insert(&g_user_context_tree, thread_context->pid, thread_context);
        spin_unlock_irqrestore(&g_lock, flags);

        radix_tree_preload_end();

        UVM_ASSERT_MSG(ret == 0, "Insert failed after a successful preload: %d\n", ret);
    }
    else {
        UVM_ASSERT(thread_context->task == get_current());
    }

    ++thread_context->ref_count;

    return thread_context;
}

static void uvm_thread_context_user_release(void)
{
    unsigned long flags;
    uvm_thread_context_t *thread_context = uvm_thread_context_user();

    if (!thread_context)
        return;

    UVM_ASSERT(thread_context->task == get_current());
    UVM_ASSERT(thread_context->pid == get_current()->pid);
    UVM_ASSERT(thread_context->ref_count > 0);

    if (--thread_context->ref_count == 0) {
        uvm_thread_context_t *removed;

        spin_lock_irqsave(&g_lock, flags);
        removed = radix_tree_delete(&g_user_context_tree, thread_context->pid);
        spin_unlock_irqrestore(&g_lock, flags);

        UVM_ASSERT(removed == thread_context);
        kmem_cache_free(g_uvm_thread_context_cache, removed);
    }
}

static uvm_thread_context_t *uvm_thread_context_interrupt(void)
{
    uvm_thread_context_t *thread_context;

    // As we are in interrupt anyway it would be best to just use this_cpu_ptr()
    // but it was added in 2.6.33 and the interface is non-trivial to implement
    // prior to that.
    thread_context = &get_cpu_var(interrupt_thread_context);
    put_cpu_var(interrupt_thread_context);

    return thread_context;
}

static uvm_thread_context_t *uvm_thread_context_interrupt_retain(void)
{
    uvm_thread_context_t *thread_context = uvm_thread_context_interrupt();

    ++thread_context->ref_count;

    return thread_context;
}

static void uvm_thread_context_interrupt_release(void)
{
    uvm_thread_context_t *thread_context = uvm_thread_context_interrupt();

    UVM_ASSERT(thread_context->ref_count > 0);

    --thread_context->ref_count;
}

uvm_thread_context_t *uvm_thread_context(void)
{
    if (in_interrupt())
        return uvm_thread_context_interrupt();
    else
        return uvm_thread_context_user();
}

uvm_thread_context_t *uvm_thread_context_retain(void)
{
    if (in_interrupt())
        return uvm_thread_context_interrupt_retain();
    else
        return uvm_thread_context_user_retain();
}

void uvm_thread_context_release(void)
{
    if (in_interrupt())
        uvm_thread_context_interrupt_release();
    else
        uvm_thread_context_user_release();
}

void uvm_thread_context_disable_lock_tracking(void)
{
    uvm_thread_context_t *thread_context = uvm_thread_context_retain();
    if (!thread_context)
        return;

    ++thread_context->skip_lock_tracking;
}


void uvm_thread_context_enable_lock_tracking(void)
{
    uvm_thread_context_t *thread_context = uvm_thread_context();
    if (!thread_context)
        return;

    UVM_ASSERT(thread_context->skip_lock_tracking > 0);

    --thread_context->skip_lock_tracking;

    uvm_thread_context_release();
}
