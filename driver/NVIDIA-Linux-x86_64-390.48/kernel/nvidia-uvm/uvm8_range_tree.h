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

#ifndef __UVM8_RANGE_TREE_H__
#define __UVM8_RANGE_TREE_H__

#include "uvm_linux.h"
#include "nvstatus.h"

// Tree-based data structure for looking up and iterating over objects with
// provided [start, end] ranges. The ranges are not allowed to overlap.
//
// All locking is up to the caller.

typedef struct uvm_range_tree_struct
{
    // Tree of uvm_range_tree_node_t's sorted by start.
    struct rb_root rb_root;

    // List of uvm_range_tree_node_t's sorted by start. This is an optimization
    // to avoid calling rb_next and rb_prev frequently, particularly while
    // iterating.
    struct list_head head;
} uvm_range_tree_t;

typedef struct uvm_range_tree_node_struct
{
    NvU64 start;
    // end is inclusive
    NvU64 end;

    struct rb_node rb_node;
    struct list_head list;
} uvm_range_tree_node_t;


void uvm_range_tree_init(uvm_range_tree_t *tree);

// Set node->start and node->end before calling this function. Overlapping
// ranges are not allowed. If the new node overlaps with an existing range node,
// NV_ERR_UVM_ADDRESS_IN_USE is returned.
NV_STATUS uvm_range_tree_add(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);

void uvm_range_tree_remove(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);

// Shrink an existing node to [new_start, new_end].
// The new range needs to be a subrange of the range being updated, that is
// new_start needs to be greater or equal to node->start and new_end needs to be
// lesser or equal to node->end.
void uvm_range_tree_shrink_node(uvm_range_tree_t *tree, uvm_range_tree_node_t *node, NvU64 new_start, NvU64 new_end);

// Splits an existing node into two pieces, with the new node always after the
// existing node. The caller must set new->start before calling this function.
// existing should not be modified by the caller. On return, existing will
// contain its updated smaller bounds.
//
// Before: [----------- existing ------------]
// After:  [---- existing ----][---- new ----]
//                             ^new->start
void uvm_range_tree_split(uvm_range_tree_t *tree,
                          uvm_range_tree_node_t *existing,
                          uvm_range_tree_node_t *new);

// Attempts to merge the given node with the prev/next node in address order.
// If the prev/next node is not adjacent to the given node, NULL is returned.
// Otherwise the provided node is kept in the tree and extended to cover the
// adjacent node. The adjacent node is removed and returned.
uvm_range_tree_node_t *uvm_range_tree_merge_prev(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);
uvm_range_tree_node_t *uvm_range_tree_merge_next(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);

// Returns the node containing addr, if any
uvm_range_tree_node_t *uvm_range_tree_find(uvm_range_tree_t *tree, NvU64 addr);

// Returns the prev/next node in address order, or NULL if none exists
uvm_range_tree_node_t *uvm_range_tree_prev(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);
uvm_range_tree_node_t *uvm_range_tree_next(uvm_range_tree_t *tree, uvm_range_tree_node_t *node);

// Returns the first node in the range [start, end], if any
uvm_range_tree_node_t *uvm_range_tree_iter_first(uvm_range_tree_t *tree, NvU64 start, NvU64 end);

// Returns the node following the provided node in address order, if that node's
// start <= the provided end.
uvm_range_tree_node_t *uvm_range_tree_iter_next(uvm_range_tree_t *tree, uvm_range_tree_node_t *node, NvU64 end);

#define uvm_range_tree_for_each(node, tree) list_for_each_entry((node), &(tree)->head, list)

#define uvm_range_tree_for_each_in(node, tree, start, end)              \
    for ((node) = uvm_range_tree_iter_first((tree), (start), (end));    \
         (node);                                                        \
         (node) = uvm_range_tree_iter_next((tree), (node), (end)))

#endif // __UVM8_RANGE_TREE_H__
