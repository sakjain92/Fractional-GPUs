/* 
 * This is a custom allocator to manage memory of a given buffer
 * without read/writing on the buffer.
 * Currently it is a naive implementation using 'next-fit' logic (single threaded)
 */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#include <map>

#include <fgpu_internal_allocator.hpp>

/* Represents a chunk of memory */
typedef struct node {
    Q_NEW_LINK(node) all_link;
    Q_NEW_LINK(node) free_link;
    void *address;
    size_t size;
    bool is_free;
} node_t;

/* Head for the queue of nodes */
Q_NEW_HEAD(node_list, node);

/* Represents the context saved for an allocator */
typedef struct allocator {
    struct node_list free_list;                         /* List of free nodes */
    struct node_list all_list;                          /* List of all nodes */
    void *start_address;
    size_t size;
    size_t alignment;                                   /* Alignment requirement for all addresses */
    std::map<void *, node_t *> nodes_map;               /* Nodes are both mainted in map and list */
} allocator_t;

/* Does sanity check on a node */
static void do_node_sanity_check(allocator_t *ctx, node_t *node)
{
    assert((uintptr_t)node->address % ctx->alignment == 0);
    assert(node->size % ctx->alignment == 0);
}

/* Creates a new free node, starting at 'address' and of size 'size' */
static node_t *new_free_node(allocator_t *ctx, void *address, size_t size)
{
    node_t *node;

    node = new node_t;

    Q_INIT_ELEM(node, free_link);
    Q_INIT_ELEM(node, all_link);

    node->address = address;
    node->size = size;
    node->is_free = false;

    do_node_sanity_check(ctx, node);

    return node;
}

/* Mark a node as allocated and remove from free list */
static void mark_node_allocated(allocator_t *ctx, node_t *node)
{
    do_node_sanity_check(ctx, node);
    assert(node->is_free);

    Q_REMOVE(&ctx->free_list, node, free_link);
    node->is_free = false;
}

/* Merges free nodes */
static void merge_nodes(allocator_t *ctx, node_t *node, node_t *prev, node_t *next)
{
    assert(node);
    assert(node->is_free);
    do_node_sanity_check(ctx, node);

    if (prev) {
        assert(prev->is_free);
        assert((uintptr_t)prev->address + prev->size == (uintptr_t)node->address);
        do_node_sanity_check(ctx, prev);
    }

    if (next) {
        assert(next->is_free);
        assert((uintptr_t)node->address + node->size == (uintptr_t)next->address);
        do_node_sanity_check(ctx, next);
    }

    if (prev && next) {
        /* Consolidate all into prev node */
        prev->size += node->size + next->size;

        Q_REMOVE(&ctx->all_list, node, all_link);
        Q_REMOVE(&ctx->all_list, next, all_link);

        Q_REMOVE(&ctx->free_list, node, free_link);
        Q_REMOVE(&ctx->free_list, next, free_link);

        ctx->nodes_map.erase(node->address);
        ctx->nodes_map.erase(next->address);

        delete node;
        delete next;
    
    } else if (prev) {
        /* Consolidate all into prev node */
        prev->size += node->size;

        Q_REMOVE(&ctx->all_list, node, all_link);

        Q_REMOVE(&ctx->free_list, node, free_link);

        ctx->nodes_map.erase(node->address);

        delete node;

    } else if (next) {
        /* Consolidate all into current node */
        node->size += next->size;

        Q_REMOVE(&ctx->all_list, next, all_link);

        Q_REMOVE(&ctx->free_list, next, free_link);

        ctx->nodes_map.erase(next->address);

        delete next;
    }
}

/* 
 * Mark already existing node as free node 
 * This function takes care of merging nodes
 */
static void mark_node_free(allocator_t *ctx, node_t *node)
{
    node_t *prev, *next;

    do_node_sanity_check(ctx, node);
    assert(!node->is_free);
    Q_CHECK_REMOVED(node, free_link);

    node->is_free = true;

    prev = Q_GET_PREV(node, all_link);
    next = Q_GET_NEXT(node, all_link);


    /* Try to merge */
    if (prev && next && (prev->is_free || next->is_free)) {

        if (prev->is_free && next->is_free) {
            Q_INSERT_AFTER(&ctx->free_list, prev, node, free_link);
            merge_nodes(ctx, node, prev, next);
        } else if (prev->is_free) {
            Q_INSERT_AFTER(&ctx->free_list, prev, node, free_link);
            merge_nodes(ctx, node, prev, NULL);
        } else {
            assert(next->is_free);
            Q_INSERT_BEFORE(&ctx->free_list, next, node, free_link);
            merge_nodes(ctx, node, NULL, next);
        }
    
    } else if (prev && prev->is_free) {

        Q_INSERT_AFTER(&ctx->free_list, prev, node, free_link);
        merge_nodes(ctx, node, prev, NULL);

    } else if (next && next->is_free) {

        Q_INSERT_BEFORE(&ctx->free_list, next, node, free_link);
        merge_nodes(ctx, node, NULL, next);

    } else if (!prev && !next) {
        
        /* Node is the first node */
        Q_INSERT_FRONT(&ctx->all_list, node, all_link);
        Q_INSERT_FRONT(&ctx->free_list, node, free_link);
        ctx->nodes_map[node->address] = node;

    } else {
        /* 
         * Insert node in appropriate place in the free list.
         * Both prev and next are not free so don't know the appropriate place
         * in queue.
         * TODO: This is O(N). Try to reduce this.
         */
        node_t *prev_free_node = NULL;

        Q_FOREACH_PREV_FROM(prev_free_node, node, &ctx->all_list, all_link) {
            if (prev_free_node->is_free)
                break;
        }

        if (prev_free_node == NULL) {
            Q_INSERT_FRONT(&ctx->free_list, node, free_link);
        } else {
            Q_INSERT_AFTER(&ctx->free_list, prev_free_node, node, free_link);
        }
    }
}

/*
 * Splits the node into two node such that node's size is 'size'
 */
static void split_node(allocator_t *ctx, node_t *node, size_t size)
{
    node_t *next;
    uintptr_t next_address;
    size_t next_size;

    do_node_sanity_check(ctx, node);
    assert(node->size > size);
    assert(node->is_free);

    next_size = node->size - size;
    node->size = size;
    next_address = (uintptr_t)node->address + node->size;

    next = new_free_node(ctx, (void *)next_address, next_size);

    next->is_free = true;
    Q_INSERT_AFTER(&ctx->all_list, node, next, all_link);
    Q_INSERT_AFTER(&ctx->free_list, node, next, free_link);
    ctx->nodes_map[(void *)next_address] = next;

    do_node_sanity_check(ctx, node);
    do_node_sanity_check(ctx, next);
}

/* 
 * Finds a node of atleast size 'size'.
 * Splits nodes if needed.
 * Currently naive implementation searches over all the free nodes till first
 * sufficiently large chunk is found.
 * Returns NULL if no free node of appropriate size is found.
 */
static node_t *get_free_node(allocator_t *ctx, size_t size)
{
    node_t *node = NULL;
    size_t allocation_size;

    /* Round up allocation size */
    allocation_size = (size + ctx->alignment - 1) & ~(ctx->alignment - 1);

    Q_FOREACH(node, &ctx->free_list, free_link) {
        if (node->size >= allocation_size) {
            break;
        }
    }

    if (!node)
        return NULL;

    assert(node->is_free);

    if (node->size > allocation_size) {
        split_node(ctx, node, allocation_size);
    }

    mark_node_allocated(ctx, node);

    return node;
}

/* 
 * Initializes an allocator context. 
 * Buf points to the start address.
 * Size denotes total size of buffer.
 * Alignment denotes the alignment of all subsequent allocations.
 * Returns 0 on success, otherwise failure
 */
allocator_t *allocator_init(void *buf, size_t size, size_t alignment)
{
    uintptr_t start_address, round_address;
    size_t mask = alignment - 1;
    allocator_t *ctx;
    size_t node_size;
    node_t *node;

    /* Alignment should be power of 2 */
    if (alignment & mask) {
        return NULL;
    }

    start_address = (uintptr_t)buf;
    round_address = (start_address + alignment - 1) & ~mask;

    if (size < round_address - start_address) {
        return NULL;
    }

    /* Round down size and also take into account the rouding up effect above */
    node_size = size - (round_address - start_address);
    node_size = node_size & ~mask;

    ctx = new allocator_t;

    Q_INIT_HEAD(&ctx->free_list);
    Q_INIT_HEAD(&ctx->all_list);

    ctx->size = size;
    ctx->start_address = buf;
    ctx->alignment = alignment;

    /* Insert the whole buffer as a free node */
    node = new_free_node(ctx, (void *)round_address, node_size);
    mark_node_free(ctx, node);

    return ctx;
}

/* Allocated a buffer of atleast size 'size' and returns it */
void *allocator_alloc(allocator_t *ctx, size_t size)
{
    node_t *node;

    node = get_free_node(ctx, size);

    if (!node)
        return NULL;

    return node->address;
}

/* Frees up a buffer */
void allocator_free(allocator_t *ctx, void *address)
{
    std::map<void *, node_t *>::iterator it;
    node_t *node;

    it = ctx->nodes_map.find(address);
    assert(it != ctx->nodes_map.end());

    node = it->second;

    mark_node_free(ctx, node);
}

/* Frees up the allocator */
void allocator_deinit(allocator_t *ctx)
{
    std::map<void *, node_t *>::iterator it;

    for (it = ctx->nodes_map.begin(); it != ctx->nodes_map.end(); ++it)
        delete it->second;

    ctx->nodes_map.clear();
    
    delete ctx;
}
