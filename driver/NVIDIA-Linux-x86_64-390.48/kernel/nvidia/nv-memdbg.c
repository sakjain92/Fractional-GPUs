/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include "nv-memdbg.h"
#include "nv-linux.h"

/* track who's allocating memory and print out a list of leaked allocations at
 * teardown.
 */

typedef struct {
    struct rb_node rb_node;
    void *addr;
    NvU64 size;
    NvU32 line;
    const char *file;
} nv_memdbg_node_t;

struct
{
    struct rb_root rb_root;
    NvU64 untracked_bytes;
    NvU64 num_untracked_allocs;
    nv_spinlock_t lock;
} g_nv_memdbg;

void nv_memdbg_init(void)
{
    NV_SPIN_LOCK_INIT(&g_nv_memdbg.lock);
    g_nv_memdbg.rb_root = RB_ROOT;
}

static nv_memdbg_node_t *nv_memdbg_node_entry(struct rb_node *rb_node)
{
    return rb_entry(rb_node, nv_memdbg_node_t, rb_node);
}

static void nv_memdbg_insert_node(nv_memdbg_node_t *new)
{
    nv_memdbg_node_t *node;
    struct rb_node **rb_node = &g_nv_memdbg.rb_root.rb_node;
    struct rb_node *rb_parent = NULL;

    while (*rb_node)
    {
        node = nv_memdbg_node_entry(*rb_node);

        WARN_ON(new->addr == node->addr);

        rb_parent = *rb_node;

        if (new->addr < node->addr)
            rb_node = &(*rb_node)->rb_left;
        else
            rb_node = &(*rb_node)->rb_right;
    }

    rb_link_node(&new->rb_node, rb_parent, rb_node);
    rb_insert_color(&new->rb_node, &g_nv_memdbg.rb_root);
}

static nv_memdbg_node_t *nv_memdbg_remove_node(void *addr)
{
    nv_memdbg_node_t *node = NULL;
    struct rb_node *rb_node = g_nv_memdbg.rb_root.rb_node;

    while (rb_node)
    {
        node = nv_memdbg_node_entry(rb_node);
        if (addr == node->addr)
            break;
        else if (addr < node->addr)
            rb_node = rb_node->rb_left;
        else
            rb_node = rb_node->rb_right;
    }

    WARN_ON(!node || node->addr != addr);

    rb_erase(&node->rb_node, &g_nv_memdbg.rb_root);
    return node;
}

void nv_memdbg_add(void *addr, NvU64 size, const char *file, int line)
{
    nv_memdbg_node_t *node;
    unsigned long flags;

    WARN_ON(addr == NULL);

    /* If node allocation fails, we can still update the untracked counters */
    node = kmalloc(sizeof(*node),
                   NV_MAY_SLEEP() ? NV_GFP_KERNEL : NV_GFP_ATOMIC);
    if (node)
    {
        node->addr = addr;
        node->size = size;
        node->file = file;
        node->line = line;
    }

    NV_SPIN_LOCK_IRQSAVE(&g_nv_memdbg.lock, flags);

    if (node)
    {
        nv_memdbg_insert_node(node);
    }
    else
    {
        ++g_nv_memdbg.num_untracked_allocs;
        g_nv_memdbg.untracked_bytes += size;
    }

    NV_SPIN_UNLOCK_IRQRESTORE(&g_nv_memdbg.lock, flags);
}

void nv_memdbg_remove(void *addr, NvU64 size, const char *file, int line)
{
    nv_memdbg_node_t *node;
    unsigned long flags;

    NV_SPIN_LOCK_IRQSAVE(&g_nv_memdbg.lock, flags);

    node = nv_memdbg_remove_node(addr);
    if (!node)
    {
        WARN_ON(g_nv_memdbg.num_untracked_allocs == 0);
        WARN_ON(g_nv_memdbg.untracked_bytes < size);
        --g_nv_memdbg.num_untracked_allocs;
        g_nv_memdbg.untracked_bytes -= size;
    }

    NV_SPIN_UNLOCK_IRQRESTORE(&g_nv_memdbg.lock, flags);

    if (node)
    {
        if (node->size != size)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: size mismatch on free: %llu != %llu\n",
                size, node->size);
            nv_printf(NV_DBG_ERRORS,
                "NVRM:     allocation: 0x%p @ %s:%d\n",
                node->addr, node->file, node->line);
            os_dbg_breakpoint();
        }

        kfree(node);
    }
}

void nv_memdbg_exit(void)
{
    nv_memdbg_node_t *node;
    NvU64 leaked_bytes = 0, num_leaked_allocs = 0;

    if (!RB_EMPTY_ROOT(&g_nv_memdbg.rb_root))
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: list of leaked memory allocations:\n");
    }

    while (!RB_EMPTY_ROOT(&g_nv_memdbg.rb_root))
    {
        node = nv_memdbg_node_entry(rb_first(&g_nv_memdbg.rb_root));

        leaked_bytes += node->size;
        ++num_leaked_allocs;

        nv_printf(NV_DBG_ERRORS,
            "NVRM:    %llu bytes, 0x%p @ %s:%d\n",
            node->size, node->addr, node->file, node->line);

        rb_erase(&node->rb_node, &g_nv_memdbg.rb_root);
        kfree(node);
    }

    /* If we failed to allocate a node at some point, we may have leaked memory
     * even if the tree is empty */
    if (num_leaked_allocs > 0 || g_nv_memdbg.num_untracked_allocs > 0)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: total leaked memory: %llu bytes in %llu allocations\n",
            leaked_bytes + g_nv_memdbg.untracked_bytes,
            num_leaked_allocs + g_nv_memdbg.num_untracked_allocs);

        if (g_nv_memdbg.num_untracked_allocs > 0)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM:                      %llu bytes in %llu allocations untracked\n",
                g_nv_memdbg.untracked_bytes, g_nv_memdbg.num_untracked_allocs);
        }
    }
}
