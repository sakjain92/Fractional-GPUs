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

//
// High level description of PMM is in the header file, here some implementation
// details are discussed.
//
// There is one PMM object per GPU and the PMM state among GPUs is completely
// separate with the exception of a few shared kmem caches.
//
// PMM allocates all of the memory it manages from PMA which is the common GPU
// Physical Memory Allocator shared by UVM and RM (it's included as part of RM,
// but logically separate from it).
//
// The state of each GPU memory chunk is tracked in uvm_gpu_chunk_t objects.
// Each chunk has a type, size and state. Type and size are persistent
// throughout chunk's lifetime while its state changes as it's allocated, split,
// merged and freed.
//
// PMM maintains a pre-allocated flat array of root chunks covering all possible
// physical allocations that can be returned from PMA. For simplicity, PMM
// always allocates 2M (UVM_CHUNK_SIZE_MAX) chunks from PMA and each naturally
// aligned 2M chunk represents a single root chunk. The root chunks array is
// indexed by the physical address of each chunk divided by UVM_CHUNK_SIZE_MAX
// allowing for a simple and fast lookup of root chunks.
//
// Each root chunk has a tracker for any pending operations on the root chunk
// (including all of its subchunks in case it's split) to support asynchronous
// alloc and free. Each tracker is protected by a separate bitlock (see
// root_chunk_lock()) as synchronizing any pending operations might take a long
// time and it would be undesirable for that to block other operations of PMM.
// Notably some synchronization is required as part of allocation to handle GPU
// lifetime issues across VA spaces (see comments in uvm_pmm_gpu_alloc()). Bit
// locks (instead of a mutex in each root chunk) are used to save space.
//
// All free chunks (UVM_PMM_GPU_CHUNK_STATE_FREE) are kept on free lists, with
// one list per each combination of memory type and chunk size (see usage of
// uvm_pmm_gpu_t::free_list for reference). This allows for a very quick
// allocation and freeing of chunks in case the right size is already available
// on alloc or no merges are required on free. See claim_free_chunk() for
// allocation and chunk_free_locked() for freeing.
//
// When a chunk is allocated it transitions into the temporarily pinned state
// (UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED) until it's unpinned when it becomes
// allocated (UVM_PMM_GPU_CHUNK_STATE_ALLOCATED). This transition is only
// meaningful for user memory chunks where temporarily pinned chunks cannot be
// evicted. Kernel memory type chunks do not support eviction at all and they
// are transitioned into the allocated state as part of the allocation itself
// (see uvm_pmm_gpu_alloc_kernel). When the chunk is freed it transitions back
// to the free state and is placed on an appropriate free list.
//
// To support smaller allocations, PMM internally splits and merges root chunks
// as needed. Splitting and merging is protected by an exclusive lock
// (uvm_pmm_gpu_t::lock) to prevent PMM from over-allocating root chunks in case
// multiple threads race for a small allocation and there are no free chunks
// immediately available.
//
// Splitting is performed lazily, i.e. chunks are only split when a chunk of the
// requested type and size is not available. Splits are only done to the next
// smaller size and hence may need to be performed multiple times recursively to
// get to the desired chunk size. See alloc_chunk_with_splits(). All split
// chunks under the root chunk form a tree with all internal nodes being in
// split state and leaf nodes being in any of the free, allocated or pinned
// states.
//
// Merging is performed eagerly, i.e. whenever all chunks under a parent (split)
// chunk become free, they are merged into one bigger chunk. See
// free_chunk_with_merges().
//
// Splitting and merging already allocated chunks is also exposed to the users of
// allocated chunks. See uvm_pmm_gpu_split_chunk() and uvm_pmm_gpu_merge_chunk().
//
// As splits and merges are protected by a single PMM mutex, they are only
// performed when really necessary. See alloc_chunk() that falls back to split
// only as the last step and free_chunk() that similarly first tries performing
// a quick free.
//
// When a memory allocation from PMA fails and eviction is requested, PMM will
// check whether it can evict any user memory chunks to satisfy the request.
// All allocated user memory root chunks are tracked in an LRU list
// (root_chunks.va_block_used). A root chunk is moved to the tail of that list
// whenever any of its subchunks is allocated (unpinned) by a VA block (see
// uvm_pmm_gpu_unpin_temp()). When a root chunk is selected for eviction, it has
// the eviction flag set (see pick_root_chunk_to_evict()). This flag affects
// many of the PMM operations on all of the subchunks of the root chunk being
// evicted. See usage of (root_)chunk_is_in_eviction(), in particular in
// chunk_free_locked() and claim_free_chunk().
//
// To evict a root chunk, all of its free subchunks are pinned, then all
// resident pages backed by it are moved to the CPU one VA block at a time.
// After all of them are moved, the root chunk is merged and returned to the
// caller. See evict_root_chunk() for details.
//
// Eviction is also possible to be triggered by PMA. This makes it possible for
// other PMA clients (most importantly RM which CUDA uses for non-UVM
// allocations) to successfully allocate memory from the user memory pool
// allocated by UVM. UVM registers two eviction callbacks with PMA that PMA
// calls as needed to perform the eviction:
//  - uvm_pmm_gpu_pma_evict_range - for evicting a physical range
//  - uvm_pmm_gpu_pma_evict_pages - for evicting a number of pages
//
// Both of them perform the eviction using the same building blocks as internal
// eviction, but see their implementation and references to pma.h for more
// details.
//
// PMM locking
// - PMM mutex
//   Exclusive lock protecting both internal and external splits and merges, and
//   eviction.
//
// - PMM list lock
//   Protects state transitions of chunks and their movement among lists.
//
// - PMM root chunk bit locks
//   Each bit lock protects the corresponding root chunk's allocation, freeing
//   from/to PMA, root chunk trackers, and root chunk indirect_peer mappings.
//
// - PMA allocation/eviction lock
//   A read-write semaphore used by the eviction path to flush any pending
//   allocations. See usage of pma_lock in alloc_root_chunk() and
//   uvm_pmm_gpu_pma_evict_range().
//
// == Trade-offs ===
//
// In general, PMM is optimized towards Pascal+ and 2M VA blocks (that's also
// the UVM_CHUNK_SIZE_MAX) as Pascal+ makes much heavier use of PMM:
//  - Oversubscription is Pascal+ only
//  - On pre-Pascal (UVM-Lite) CUDA currently pre-populates all managed memory
//    and hence performance matters mostly only during CUDA memory allocation.
//  - On Pascal+ CUDA doesn't pre-populate and memory is allocated on first
//    touch.
//
// The root chunk size matching the VA block chunk size allows PMM to avoid
// having to split and merge for the hopefully (HMM might make this hard) common
// allocation size of 2M on Pascal+.
//
// Careful benchmarks and tweaking of PMM are yet to be performed, but there is
// some evidence for PMA to potentially cause issues for oversubscription (see
// bug 1775408).
//

#include "uvm_common.h"
#include "nv_uvm_interface.h"
#include "uvm8_gpu.h"
#include "uvm8_pmm_gpu.h"
#include "uvm8_mem.h"
#include "uvm8_mmu.h"
#include "uvm8_global.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_va_space.h"
#include "uvm8_va_block.h"
#include "uvm8_test.h"
#include "uvm_linux.h"

static int uvm_global_oversubscription = 1;
module_param(uvm_global_oversubscription, int, S_IRUGO);
MODULE_PARM_DESC(uvm_global_oversubscription, "Enable (1) or disable (0) global oversubscription support.");

// Helper type for refcounting cache
typedef struct
{
    // Cache for given split size
    struct kmem_cache *cache;

    // Number of GPUs using given split size
    NvU32 refcount;

    // Name of cache
    char name[32];
} kmem_cache_ref_t;

struct uvm_pmm_gpu_chunk_suballoc_struct
{
    // Number of allocated chunks (including pinned ones)
    NvU32 allocated;

    // Number of pinned leaf chunks under this chunk
    //
    // Tracked only for suballocs of root chunks to know whether a root chunk
    // can be evicted. This is not in the uvm_gpu_root_chunk_t itself to stop
    // the root chunks array from growing too much.
    // TODO: Bug 1765193: Consider moving this to a union with the parent
    // pointer in uvm_gpu_chunk_t as root chunks never have a parent or just put
    // in the root chunk directly.
    // TODO: Bug 1765193: This could be NvU16 if we enforce the smallest chunk
    // size to be at least 2^21 / 2^16 = 32 bytes.
    NvU32 pinned_leaf_chunks;

    // Array of all child subchunks
    // TODO: Bug 1765461: Can the array be inlined? It could save the parent
    //       pointer.
    uvm_gpu_chunk_t *subchunks[0];
};

typedef enum
{
    CHUNK_WALK_PRE_ORDER,
    CHUNK_WALK_POST_ORDER
} chunk_walk_order_t;

typedef NV_STATUS (*chunk_walk_func_t)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data);

// Cache for allocation of uvm_pmm_gpu_chunk_suballoc_t. At index n it stores
// a suballoc structure for size 2**n.
//
// For convenience of init/deinit code level 0 is for allocation of chunks
static kmem_cache_ref_t chunk_split_cache[UVM_PMM_CHUNK_SPLIT_CACHE_SIZES];
#define CHUNK_CACHE chunk_split_cache[0].cache

const char *uvm_pmm_gpu_memory_type_string(uvm_pmm_gpu_memory_type_t type)
{
    switch (type) {
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_MEMORY_TYPE_USER);
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_MEMORY_TYPE_KERNEL);
        UVM_ENUM_STRING_DEFAULT();
    }
}

const char *uvm_pmm_gpu_chunk_state_string(uvm_pmm_gpu_chunk_state_t state)
{
    switch (state) {
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_CHUNK_STATE_FREE);
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
        UVM_ENUM_STRING_CASE(UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
        UVM_ENUM_STRING_DEFAULT();
    }
}

// The PMA APIs that can be called from PMA eviction callbacks (pmaPinPages and
// pmaFreePages*) need to be called differently depending whether it's as part
// of PMA eviction or not. The PMM context is used to plumb that information
// through the stack in a couple of places.
typedef enum
{
    PMM_CONTEXT_DEFAULT,
    PMM_CONTEXT_PMA_EVICTION,
} uvm_pmm_context_t;

// Freeing the root chunk not only needs to differentiate between two different
// contexts for calling pmaFreePages(), but also in some cases the free back to
// PMA needs to be skipped altogether.
typedef enum
{
    FREE_ROOT_CHUNK_MODE_DEFAULT,
    FREE_ROOT_CHUNK_MODE_PMA_EVICTION,
    FREE_ROOT_CHUNK_MODE_SKIP_PMA_FREE
} free_root_chunk_mode_t;

static free_root_chunk_mode_t free_root_chunk_mode_from_pmm_context(uvm_pmm_context_t pmm_context)
{
    switch (pmm_context) {
        case PMM_CONTEXT_DEFAULT:
            return FREE_ROOT_CHUNK_MODE_DEFAULT;
        case PMM_CONTEXT_PMA_EVICTION:
            return FREE_ROOT_CHUNK_MODE_PMA_EVICTION;
        default:
            UVM_ASSERT_MSG(false, "Invalid PMM context: 0x%x\n", pmm_context);
            return FREE_ROOT_CHUNK_MODE_DEFAULT;
    }
}

static NV_STATUS alloc_chunk(uvm_pmm_gpu_t *pmm,
                             uvm_pmm_gpu_memory_type_t type,
                             uvm_chunk_size_t chunk_size,
                             uvm_pmm_alloc_flags_t flags,
                             uvm_gpu_chunk_t **chunk);
static NV_STATUS alloc_root_chunk(uvm_pmm_gpu_t *pmm,
                            uvm_pmm_gpu_memory_type_t type,
                            uvm_gpu_chunk_t **chunk);
static void free_root_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk, free_root_chunk_mode_t free_mode);
static NV_STATUS split_gpu_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static void free_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static void free_chunk_with_merges(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static void free_next_available_root_chunk(uvm_pmm_gpu_t *pmm);
static NV_STATUS init_chunk_split_cache(uvm_pmm_gpu_t *pmm);
static NV_STATUS init_chunk_split_cache_level(uvm_pmm_gpu_t *pmm, size_t level);
static void deinit_chunk_split_cache(uvm_pmm_gpu_t *pmm);
static struct list_head *find_free_list(uvm_pmm_gpu_t *pmm, uvm_pmm_gpu_memory_type_t type, uvm_chunk_size_t chunk_size);
static bool check_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static struct list_head *find_free_list_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static void chunk_free_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
static uvm_gpu_chunk_t *claim_free_chunk(uvm_pmm_gpu_t *pmm, uvm_pmm_gpu_memory_type_t type, uvm_chunk_size_t chunk_size);
static NV_STATUS uvm_pmm_gpu_pma_evict_pages_wrapper(void *void_pmm,
                                                     NvU32 page_size,
                                                     NvU64 *pages,
                                                     NvU32 num_pages_to_evict,
                                                     NvU64 phys_start,
                                                     NvU64 phys_end);
static NV_STATUS uvm_pmm_gpu_pma_evict_range_wrapper(void *void_pmm, NvU64 phys_begin, NvU64 phys_end);

static size_t root_chunk_index(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    size_t index = root_chunk->chunk.address / UVM_CHUNK_SIZE_MAX;
    UVM_ASSERT(index < pmm->root_chunks.count);
    return index;
}

static void root_chunk_lock(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    uvm_bit_lock(&pmm->root_chunks.bitlocks, root_chunk_index(pmm, root_chunk));
}

static void uvm_assert_root_chunk_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    uvm_assert_bit_locked(&pmm->root_chunks.bitlocks, root_chunk_index(pmm, root_chunk));
}

static void root_chunk_unlock(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    uvm_bit_unlock(&pmm->root_chunks.bitlocks, root_chunk_index(pmm, root_chunk));
}

// TODO: Bug 1795559: Remove once PMA eviction is considered safe enough not to
// have an opt-out.
static bool gpu_supports_pma_eviction(uvm_gpu_t *gpu)
{
    return uvm_global_oversubscription && uvm_gpu_supports_eviction(gpu);
}

NV_STATUS uvm_pmm_gpu_init(uvm_gpu_t *gpu, uvm_pmm_gpu_t *pmm)
{
    typedef uvm_chunk_sizes_mask_t (*chunk_size_init_func_t)(uvm_gpu_t *gpu);
    static const chunk_size_init_func_t chunk_size_init_func[][UVM_PMM_GPU_MEMORY_TYPE_COUNT] = {
        {uvm_mmu_user_chunk_sizes, uvm_mmu_kernel_chunk_sizes},
        {NULL, uvm_mem_kernel_chunk_sizes},
    };
    NV_STATUS status = NV_OK;
    size_t i, j;

    // UVM_CHUNK_SIZE_INVALID is UVM_CHUNK_SIZE_MAX shifted left by 1. This protects
    // UVM_CHUNK_SIZE_INVALID from being negative
    BUILD_BUG_ON(UVM_CHUNK_SIZE_MAX >= UVM_CHUNK_SIZE_INVALID);

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    memset(pmm, 0, sizeof(*pmm));

    for (i = 0; i < ARRAY_SIZE(pmm->free_list); i++) {
        for (j = 0; j < ARRAY_SIZE(pmm->free_list[i]); j++)
            INIT_LIST_HEAD(&pmm->free_list[i][j]);
    }
    INIT_LIST_HEAD(&pmm->root_chunks.va_block_used);
    INIT_LIST_HEAD(&pmm->root_chunks.va_block_unused);

    uvm_mutex_init(&pmm->lock, UVM_LOCK_ORDER_PMM);
    uvm_init_rwsem(&pmm->pma_lock, UVM_LOCK_ORDER_PMM_PMA);
    uvm_spin_lock_init(&pmm->list_lock, UVM_LOCK_ORDER_LEAF);

    pmm->gpu = gpu;

    for (i = 0; i < UVM_PMM_GPU_MEMORY_TYPE_COUNT; i++) {
        pmm->chunk_sizes[i] = 0;
        // Add the common root chunk size to all memory types
        pmm->chunk_sizes[i] |= UVM_CHUNK_SIZE_MAX;
        for (j = 0; j < ARRAY_SIZE(chunk_size_init_func); j++) {
            if (chunk_size_init_func[j][i])
                pmm->chunk_sizes[i] |= chunk_size_init_func[j][i](gpu);
        }
        UVM_ASSERT(pmm->chunk_sizes[i] < UVM_CHUNK_SIZE_INVALID);
        UVM_ASSERT_MSG(hweight_long(pmm->chunk_sizes[i]) <= UVM_MAX_CHUNK_SIZES,
                "chunk sizes %lu, max chunk sizes %u\n", hweight_long(pmm->chunk_sizes[i]), UVM_MAX_CHUNK_SIZES);
    }

    status = init_chunk_split_cache(pmm);
    if (status != NV_OK)
        goto cleanup;

    // Assert that max physical address of the GPU is not unreasonably big for
    // creating the flat array of root chunks. Currently the worst case is a
    // Maxwell GPU that has 0.5 GB of its physical memory mapped at the 64GB
    // physical address. 256GB should provide reasonable amount of
    // future-proofing and results in 128K chunks which is still manageable.
    UVM_ASSERT_MSG(gpu->vidmem_max_allocatable_address < 256ull * 1024 * 1024 * 1024,
                   "Max physical address over 256GB: %llu\n",
                   gpu->vidmem_max_allocatable_address);

    // Align up the size to have a root chunk for the last part of the FB. PMM
    // won't be able to allocate it, if it doesn't fit a whole root chunk, but
    // it's convenient to have it for uvm8_test_pma_alloc_free().
    pmm->root_chunks.count = UVM_ALIGN_UP(gpu->vidmem_max_allocatable_address, UVM_CHUNK_SIZE_MAX) / UVM_CHUNK_SIZE_MAX;
    pmm->root_chunks.array = uvm_kvmalloc_zero(sizeof(*pmm->root_chunks.array) * pmm->root_chunks.count);
    if (!pmm->root_chunks.array) {
        status = NV_ERR_NO_MEMORY;
        goto cleanup;
    }

    // Initialize all root chunks to be PMA owned and set their addresses
    for (i = 0; i < pmm->root_chunks.count; ++i) {
        uvm_gpu_chunk_t *chunk = &pmm->root_chunks.array[i].chunk;

        INIT_LIST_HEAD(&chunk->list);
        chunk->state = UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED;
        uvm_gpu_chunk_set_size(chunk, UVM_CHUNK_SIZE_MAX);
        chunk->address = i * UVM_CHUNK_SIZE_MAX;
        chunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;
    }

    status = uvm_bit_locks_init(&pmm->root_chunks.bitlocks, pmm->root_chunks.count, UVM_LOCK_ORDER_PMM_ROOT_CHUNK);
    if (status != NV_OK)
        goto cleanup;

    status = uvm_rm_locked_call(nvUvmInterfaceGetPmaObject(&gpu->uuid, &pmm->pma));
    if (status != NV_OK)
        goto cleanup;

    if (gpu_supports_pma_eviction(gpu)) {
        status = nvUvmInterfacePmaRegisterEvictionCallbacks(pmm->pma,
                                                            uvm_pmm_gpu_pma_evict_pages_wrapper,
                                                            uvm_pmm_gpu_pma_evict_range_wrapper,
                                                            pmm);
        if (status != NV_OK)
            goto cleanup;
    }

    return NV_OK;
cleanup:
    uvm_pmm_gpu_deinit(pmm);
    return status;
}

void uvm_pmm_gpu_deinit(uvm_pmm_gpu_t *pmm)
{
    size_t i, j;

    if (!pmm || !pmm->gpu)
        return;

    if (gpu_supports_pma_eviction(pmm->gpu))
        nvUvmInterfacePmaUnregisterEvictionCallbacks(pmm->pma);

    // TODO: Bug 1766184: Handle ECC/RC
    if (!bitmap_empty(pmm->chunk_split_cache_initialized, UVM_PMM_CHUNK_SPLIT_CACHE_SIZES)) {
        for (i = 0; i < ARRAY_SIZE(pmm->free_list); i++) {
            for (j = 0; j < ARRAY_SIZE(pmm->free_list[i]); j++)
                UVM_ASSERT_MSG(list_empty(&pmm->free_list[i][j]), "i: %s, j: %zu\n",
                               uvm_pmm_gpu_memory_type_string(i), j);
        }
    }

    uvm_bit_locks_deinit(&pmm->root_chunks.bitlocks);

    for (i = 0; i < ARRAY_SIZE(pmm->root_chunks.indirect_peer); i++) {
        UVM_ASSERT(pmm->root_chunks.indirect_peer[i].dma_addrs == NULL);
        UVM_ASSERT(atomic64_read(&pmm->root_chunks.indirect_peer[i].map_count) == 0);
    }

    if (pmm->root_chunks.array) {
        // Make sure that all chunks have been returned to PMA
        for (i = 0; i < pmm->root_chunks.count; ++i) {
            uvm_gpu_chunk_t *chunk = &pmm->root_chunks.array[i].chunk;
            UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED,
                           "index %zu state %s GPU %s\n",
                           i,
                           uvm_pmm_gpu_chunk_state_string(chunk->state),
                           pmm->gpu->name);
        }
    }
    uvm_kvfree(pmm->root_chunks.array);

    deinit_chunk_split_cache(pmm);

    pmm->gpu = NULL;
}

static uvm_gpu_root_chunk_t *root_chunk_from_address(uvm_pmm_gpu_t *pmm, NvU64 addr)
{
    size_t index = addr / UVM_CHUNK_SIZE_MAX;
    uvm_gpu_root_chunk_t *root_chunk = &pmm->root_chunks.array[index];

    UVM_ASSERT_MSG(addr <= pmm->gpu->vidmem_max_allocatable_address,
                   "Address 0x%llx vidmem max phys 0x%llx GPU %s\n",
                   addr,
                   pmm->gpu->vidmem_max_allocatable_address,
                   pmm->gpu->name);
    UVM_ASSERT(root_chunk->chunk.address == UVM_ALIGN_DOWN(addr, UVM_CHUNK_SIZE_MAX));

    return root_chunk;
}

static uvm_gpu_root_chunk_t *root_chunk_from_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    return root_chunk_from_address(pmm, chunk->address);
}

static bool chunk_is_root_chunk(uvm_gpu_chunk_t *chunk)
{
    return uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX;
}

static bool chunk_is_root_chunk_pinned(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    chunk = &root_chunk->chunk;

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED)
        return true;
    else if (chunk->state != UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT)
        return false;

    UVM_ASSERT(chunk->suballoc);

    return chunk->suballoc->pinned_leaf_chunks > 0;
}

// Pin a chunk and update its root chunk's pinned leaf chunks count if the chunk is not a root chunk
static void chunk_pin(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);
    UVM_ASSERT(chunk->state != UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
    chunk->state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED;

    if (chunk_is_root_chunk(chunk))
        return;

    // For subchunks, update the pinned leaf chunks count tracked in the suballoc of the root chunk.
    chunk = &root_chunk->chunk;

    // The passed-in subchunk is not the root chunk so the root chunk has to be split
    UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT, "chunk state %s\n",
            uvm_pmm_gpu_chunk_state_string(chunk->state));

    chunk->suballoc->pinned_leaf_chunks++;
}

// Unpin a chunk and update its root chunk's pinned leaf chunks count if the chunk is not a root chunk
static void chunk_unpin(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_pmm_gpu_chunk_state_t new_state)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
    UVM_ASSERT(chunk->va_block == NULL);
    UVM_ASSERT(chunk_is_root_chunk_pinned(pmm, chunk));
    UVM_ASSERT(new_state != UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    chunk->state = new_state;

    if (chunk_is_root_chunk(chunk))
        return;

    // For subchunks, update the pinned leaf chunks count tracked in the suballoc of the root chunk.
    chunk = &root_chunk->chunk;

    // The passed-in subchunk is not the root chunk so the root chunk has to be split
    UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT, "chunk state %s\n",
            uvm_pmm_gpu_chunk_state_string(chunk->state));

    UVM_ASSERT(chunk->suballoc->pinned_leaf_chunks != 0);
    chunk->suballoc->pinned_leaf_chunks--;
}

static void uvm_gpu_chunk_set_in_eviction(uvm_gpu_chunk_t *chunk, bool in_eviction)
{
    UVM_ASSERT(chunk->type == UVM_PMM_GPU_MEMORY_TYPE_USER);
    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX);
    chunk->in_eviction = in_eviction;
}

// A helper that queries the eviction flag of root chunk of the given chunk.
// Eviction is only tracked for root chunks.
static bool chunk_is_in_eviction(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    return root_chunk_from_chunk(pmm, chunk)->chunk.in_eviction;
}

struct page *uvm_gpu_chunk_to_page(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    NvU64 sys_addr = chunk->address + pmm->gpu->numa_info.system_memory_window_start;
    unsigned long pfn = sys_addr >> PAGE_SHIFT;

    UVM_ASSERT(sys_addr + uvm_gpu_chunk_get_size(chunk) <= pmm->gpu->numa_info.system_memory_window_end + 1);
    UVM_ASSERT(pmm->gpu->numa_info.enabled);

    return pfn_to_page(pfn);
}

void uvm_pmm_gpu_sync(uvm_pmm_gpu_t *pmm)
{
    size_t i;

    if (!pmm->gpu)
        return;

    // Just go over all root chunks and sync the ones that are not PMA OWNED.
    // This is slow, but uvm_pmm_gpu_sync() is a rarely used operation not
    // critical for performance.
    for (i = 0; i < pmm->root_chunks.count; ++i) {
        uvm_gpu_root_chunk_t *root_chunk = &pmm->root_chunks.array[i];

        root_chunk_lock(pmm, root_chunk);
        if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED) {
            NV_STATUS status = uvm_tracker_wait(&root_chunk->tracker);
            if (status != NV_OK)
                UVM_ASSERT(status == uvm_global_get_status());
        }
        root_chunk_unlock(pmm, root_chunk);
    }
}

NV_STATUS uvm_pmm_gpu_alloc(uvm_pmm_gpu_t *pmm,
                            size_t num_chunks,
                            uvm_chunk_size_t chunk_size,
                            uvm_pmm_gpu_memory_type_t mem_type,
                            uvm_pmm_alloc_flags_t flags,
                            uvm_gpu_chunk_t **chunks,
                            uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    size_t i;

    UVM_ASSERT((unsigned)mem_type < UVM_PMM_GPU_MEMORY_TYPE_COUNT);
    UVM_ASSERT_MSG(is_power_of_2(chunk_size), "chunk size %u\n", chunk_size);
    UVM_ASSERT_MSG(chunk_size & pmm->chunk_sizes[mem_type], "chunk size %u\n", chunk_size);
    UVM_ASSERT(num_chunks == 0 || chunks);
    UVM_ASSERT((flags & UVM_PMM_ALLOC_FLAGS_MASK) == flags);

    if (flags & UVM_PMM_ALLOC_FLAGS_EVICT) {
        // If eviction is requested then VA block locks need to be lockable
        uvm_assert_lockable_order(UVM_LOCK_ORDER_VA_BLOCK);
    }

    for (i = 0; i < num_chunks; i++) {
        uvm_gpu_root_chunk_t *root_chunk;

        status = alloc_chunk(pmm, mem_type, chunk_size, flags, &chunks[i]);
        if (status != NV_OK)
            goto error;

        root_chunk = root_chunk_from_chunk(pmm, chunks[i]);

        root_chunk_lock(pmm, root_chunk);
        uvm_tracker_remove_completed(&root_chunk->tracker);
        status = uvm_tracker_add_tracker_safe(&local_tracker, &root_chunk->tracker);
        root_chunk_unlock(pmm, root_chunk);

        if (status != NV_OK) {
            i++;
            goto error;
        }
    }

    // Before we return to the caller, we need to ensure that the tracker only
    // contains tracker entries belonging to the PMM's GPU. Otherwise we
    // could leak trackers for other GPUs into VA spaces which never
    // registered those GPUs, causing lifetime problems when those GPUs go
    // away.
    status = uvm_tracker_wait_for_other_gpus(&local_tracker, pmm->gpu);
    if (status != NV_OK)
        goto error;

    if (out_tracker) {
        status = uvm_tracker_add_tracker_safe(out_tracker, &local_tracker);
        uvm_tracker_clear(&local_tracker);
        if (status != NV_OK)
            goto error;
    }

    return uvm_tracker_wait_deinit(&local_tracker);

error:
    uvm_tracker_deinit(&local_tracker);
    while (i-- > 0)
        free_chunk(pmm, chunks[i]);

    // Reset the array to make error handling easier for callers.
    memset(chunks, 0, sizeof(chunks[0]) * num_chunks);

    return status;
}

NV_STATUS uvm_pmm_gpu_alloc_kernel(uvm_pmm_gpu_t *pmm,
                                   size_t num_chunks,
                                   uvm_chunk_size_t chunk_size,
                                   uvm_pmm_alloc_flags_t flags,
                                   uvm_gpu_chunk_t **chunks,
                                   uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    size_t i;

    status = uvm_pmm_gpu_alloc(pmm, num_chunks, chunk_size, UVM_PMM_GPU_MEMORY_TYPE_KERNEL, flags, chunks, out_tracker);
    if (status != NV_OK)
        return status;

    for (i = 0; i < num_chunks; ++i) {
        UVM_ASSERT(chunks[i]->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

        uvm_spin_lock(&pmm->list_lock);
        chunk_unpin(pmm, chunks[i], UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
        uvm_spin_unlock(&pmm->list_lock);
    }

    return NV_OK;
}

static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (chunk->type == UVM_PMM_GPU_MEMORY_TYPE_USER) {
        if (chunk_is_root_chunk_pinned(pmm, chunk)) {
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
            list_del_init(&root_chunk->chunk.list);
        }
        else if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_FREE) {
            UVM_ASSERT(root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
                       root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
            list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);
        }
    }

    // TODO: Bug 1757148: Improve fragmentation of split chunks
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE)
        list_move_tail(&chunk->list, find_free_list_chunk(pmm, chunk));
    else if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED)
        list_del_init(&chunk->list);
}

void uvm_pmm_gpu_unpin_temp(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_va_block_t *va_block)
{
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
    UVM_ASSERT(chunk->type == UVM_PMM_GPU_MEMORY_TYPE_USER);

    INIT_LIST_HEAD(&chunk->list);

    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(!chunk->va_block);
    UVM_ASSERT(va_block);
    UVM_ASSERT(chunk->va_block_page_index < uvm_va_block_num_cpu_pages(va_block));

    chunk_unpin(pmm, chunk, UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
    chunk->va_block = va_block;
    chunk_update_lists_locked(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);
}

void uvm_pmm_gpu_free(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_gpu_root_chunk_t *root_chunk;

    if (!chunk)
        return;

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    root_chunk = root_chunk_from_chunk(pmm, chunk);

    if (tracker) {
        uvm_tracker_remove_completed(tracker);

        root_chunk_lock(pmm, root_chunk);

        // Remove any completed entries from the root tracker to prevent it from
        // growing too much over time.
        uvm_tracker_remove_completed(&root_chunk->tracker);

        status = uvm_tracker_add_tracker_safe(&root_chunk->tracker, tracker);
        if (status != NV_OK)
            UVM_ASSERT(status == uvm_global_get_status());

        root_chunk_unlock(pmm, root_chunk);
    }

    free_chunk(pmm, chunk);
}

static NvU32 num_subchunks(uvm_gpu_chunk_t *parent)
{
    uvm_chunk_size_t parent_size, child_size;
    UVM_ASSERT(parent->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
    parent_size = uvm_gpu_chunk_get_size(parent);
    child_size = uvm_gpu_chunk_get_size(parent->suballoc->subchunks[0]);
    return (NvU32)uvm_div_pow2_64(parent_size, child_size);
}

static uvm_gpu_chunk_t *next_sibling(uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_chunk_t *parent = chunk->parent;
    size_t index;

    UVM_ASSERT(parent);
    UVM_ASSERT(parent->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);

    index = (size_t)uvm_div_pow2_64(chunk->address - parent->address, uvm_gpu_chunk_get_size(chunk));
    UVM_ASSERT(index < num_subchunks(parent));

    ++index;
    if (index == num_subchunks(parent))
        return NULL;

    return parent->suballoc->subchunks[index];
}

// Check that the chunk is in a mergeable state: all children must be pinned or
// or all children must be allocated with the same reverse mapping.
//
// Always returns true so it can be called from an assert macro.
static bool assert_chunk_mergeable(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_chunk_t *first_child = chunk->suballoc->subchunks[0];
    uvm_va_block_t *child_va_block = first_child->va_block;
    size_t i;

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
    UVM_ASSERT(first_child->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
               first_child->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);

    for (i = 1; i < num_subchunks(chunk); i++) {
        uvm_gpu_chunk_t *child = chunk->suballoc->subchunks[i];

        UVM_ASSERT(child->state == first_child->state);
        if (first_child->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
            uvm_gpu_chunk_t *prev_child = chunk->suballoc->subchunks[i-1];

            UVM_ASSERT(child->va_block == child_va_block);
            UVM_ASSERT(child->va_block_page_index ==
                       prev_child->va_block_page_index + uvm_gpu_chunk_get_size(prev_child) / PAGE_SIZE);
        }
    }

    if (first_child->state == UVM_PMM_GPU_CHUNK_STATE_FREE) {
        UVM_ASSERT(chunk->suballoc->allocated == 0);
    }
    else {
        UVM_ASSERT_MSG(chunk->suballoc->allocated == num_subchunks(chunk), "%u != %u\n",
                chunk->suballoc->allocated, num_subchunks(chunk));
    }

    return true;
}

// Merges a previously-split chunk. Assumes that all of its children have
// uniform state. This only merges leaves, so none of the children can be in the
// split state themselves.
//
// The children need to be removed from any lists before the merge.
//
// The merged chunk inherits the former state of its children.
static void merge_gpu_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_pmm_gpu_chunk_suballoc_t *suballoc;
    uvm_gpu_chunk_t *subchunk;
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);
    uvm_pmm_gpu_chunk_state_t child_state;
    size_t i, num_sub = num_subchunks(chunk);

    uvm_assert_mutex_locked(&pmm->lock);
    UVM_ASSERT(assert_chunk_mergeable(pmm, chunk));

    // Transition the chunk state under the list lock first and then clean up
    // the subchunk state.
    uvm_spin_lock(&pmm->list_lock);

    child_state = chunk->suballoc->subchunks[0]->state;

    if (child_state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
        subchunk = chunk->suballoc->subchunks[0];
        UVM_ASSERT(subchunk->va_block);
        chunk->va_block = subchunk->va_block;
        chunk->va_block_page_index = subchunk->va_block_page_index;
    }
    else if (child_state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED) {
        UVM_ASSERT(root_chunk->chunk.suballoc->pinned_leaf_chunks >= num_sub);
        root_chunk->chunk.suballoc->pinned_leaf_chunks += 1 - num_sub;
    }

    chunk->state = child_state;
    suballoc = chunk->suballoc;
    chunk->suballoc = NULL;

    uvm_spin_unlock(&pmm->list_lock);

    for (i = 0; i < num_sub; i++) {
        subchunk = suballoc->subchunks[i];

        // The subchunks should have been removed from their lists prior to the
        // merge.
        UVM_ASSERT(list_empty(&subchunk->list));

        if (child_state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED)
            UVM_ASSERT(subchunk->va_block != NULL);

        kmem_cache_free(CHUNK_CACHE, subchunk);
    }

    kmem_cache_free(chunk_split_cache[ilog2(num_sub)].cache, suballoc);
}

// Checks that chunk is below ancestor in the tree. Always returns true so it
// can be called from an assert macro.
static bool assert_chunk_under(uvm_gpu_chunk_t *chunk, uvm_gpu_chunk_t *ancestor)
{
    UVM_ASSERT(ancestor->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
    UVM_ASSERT(ancestor->suballoc);
    UVM_ASSERT(ancestor->address <= chunk->address);
    UVM_ASSERT(chunk->address < ancestor->address + uvm_gpu_chunk_get_size(ancestor));
    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) <= uvm_gpu_chunk_get_size(ancestor));
    return true;
}

// Traverses the chunk tree from start in the given traversal order.
//
// If the callback returns a status value of NV_WARN_NOTHING_TO_DO when doing
// pre-order traversal, the traversal skips walking below that chunk. In all
// other cases, returning any non-NV_OK value stops the walk immediately and
// returns that status to the caller.
//
// Be careful modifying the tree from the callback. Changing the tree below the
// input chunk is fine and modifying the input chunk itself is fine, but the
// callback must not modify the tree above the input chunk. If that is needed,
// return a non-NV_OK status from the walk and re-start the walk.
static NV_STATUS chunk_walk(uvm_pmm_gpu_t *pmm,
                            uvm_gpu_chunk_t *start,
                            chunk_walk_func_t func,
                            void *data,
                            chunk_walk_order_t order)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_chunk_t *curr, *sibling;

    curr = start;

    do {
        if (curr != start)
            UVM_ASSERT(assert_chunk_under(curr, start));

        if (order == CHUNK_WALK_PRE_ORDER) {
            status = func(pmm, curr, data);
            if (status != NV_OK && status != NV_WARN_NOTHING_TO_DO)
                return status;
        }

        // Skip downward traversal on pre-order if requested
        if (status != NV_WARN_NOTHING_TO_DO && curr->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT) {
            // If the chunk is split, walk down
            curr = curr->suballoc->subchunks[0];
        }
        else {
            // This is a leaf chunk. If not start itself, check siblings.
            while (curr != start) {
                if (order == CHUNK_WALK_POST_ORDER) {
                    status = func(pmm, curr, data);
                    if (status != NV_OK)
                        return status;
                }

                sibling = next_sibling(curr);
                if (sibling) {
                    curr = sibling;
                    break;
                }

                // curr is the last chunk in its parent. Walk up and try again.
                curr = curr->parent;
                UVM_ASSERT(curr);
                UVM_ASSERT(curr->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
            }
        }
    } while (curr != start);

    // Invoke the final callback for start
    if (order == CHUNK_WALK_POST_ORDER)
        return func(pmm, curr, data);

    return NV_OK;
}

static NV_STATUS chunk_walk_pre_order(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *start, chunk_walk_func_t func, void *data)
{
    return chunk_walk(pmm, start, func, data, CHUNK_WALK_PRE_ORDER);
}

static NV_STATUS chunk_walk_post_order(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *start, chunk_walk_func_t func, void *data)
{
    return chunk_walk(pmm, start, func, data, CHUNK_WALK_POST_ORDER);
}

typedef struct
{
    // Target size for the leaf subchunks
    uvm_chunk_size_t min_size;

    // Number of subchunks split to this point. If the subchunks array is non-
    // NULL, this is the number of elements currently in the array.
    size_t num_subchunks_curr;

    // Number of subchunks needed for the whole split
    size_t num_subchunks_total;

    // Storage for the final split chunks. May be NULL.
    uvm_gpu_chunk_t **subchunks;

    // For testing
    bool inject_error;
} split_walk_t;

static NV_STATUS split_walk_func(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    uvm_chunk_size_t chunk_size, child_size;
    uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[chunk->type];
    size_t i, num_children;
    split_walk_t *args = data;
    NV_STATUS status;

    chunk_size = uvm_gpu_chunk_get_size(chunk);
    UVM_ASSERT(chunk_size > args->min_size);

    child_size = uvm_chunk_find_prev_size(chunk_sizes, chunk_size);
    UVM_ASSERT(child_size != UVM_CHUNK_SIZE_INVALID);
    num_children = chunk_size / child_size;

    if (unlikely(args->inject_error)) {
        // Inject errors on the last split
        if (child_size == args->min_size &&
            args->num_subchunks_curr + num_children == args->num_subchunks_total)
            chunk->inject_split_error = true;
    }

    status = split_gpu_chunk(pmm, chunk);
    if (status != NV_OK)
        return status;

    // If we've hit our target, add all child subchunks to the array
    if (child_size == args->min_size) {
        for (i = 0; i < num_children; i++) {
            UVM_ASSERT(args->num_subchunks_curr < args->num_subchunks_total);
            if (args->subchunks)
                args->subchunks[args->num_subchunks_curr] = chunk->suballoc->subchunks[i];
            ++args->num_subchunks_curr;
        }

        // No need to walk below this chunk
        return NV_WARN_NOTHING_TO_DO;
    }

    return NV_OK;
}

static NV_STATUS merge_walk_func(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    // The merge walk uses post-order traversal, so all subchunks are guaranteed
    // to have already been merged.
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT)
        merge_gpu_chunk(pmm, chunk);
    return NV_OK;
}

static void uvm_pmm_gpu_merge_chunk_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    NV_STATUS status;

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);

    uvm_assert_mutex_locked(&pmm->lock);

    status = chunk_walk_post_order(pmm, chunk, merge_walk_func, NULL);

    // merge_walk_func can't fail
    UVM_ASSERT(status == NV_OK);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
}

NV_STATUS uvm_pmm_gpu_split_chunk(uvm_pmm_gpu_t *pmm,
                                  uvm_gpu_chunk_t *chunk,
                                  uvm_chunk_size_t subchunk_size,
                                  uvm_gpu_chunk_t **subchunks)
{
    NV_STATUS status;
    split_walk_t walk_args =
    {
        .min_size               = subchunk_size,
        .num_subchunks_curr     = 0,
        .num_subchunks_total    = uvm_gpu_chunk_get_size(chunk) / subchunk_size,
        .subchunks              = subchunks,
        .inject_error           = chunk->inject_split_error,
    };

    UVM_ASSERT(is_power_of_2(subchunk_size));
    UVM_ASSERT(subchunk_size & pmm->chunk_sizes[chunk->type]);
    UVM_ASSERT(subchunk_size < uvm_gpu_chunk_get_size(chunk));

    uvm_mutex_lock(&pmm->lock);

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    // If we're supposed to inject an error, clear out the root chunk's flag so
    // we can inject after nearly all chunks have been split. Otherwise
    // split_gpu_chunk will fail on the first try, without creating the tree.
    if (unlikely(walk_args.inject_error))
        chunk->inject_split_error = false;

    status = chunk_walk_pre_order(pmm, chunk, split_walk_func, &walk_args);
    if (status != NV_OK) {
        // Put the chunk back in its original state
        uvm_pmm_gpu_merge_chunk_locked(pmm, chunk);
    }
    else {
        UVM_ASSERT(walk_args.num_subchunks_curr == walk_args.num_subchunks_total);
    }

    uvm_mutex_unlock(&pmm->lock);
    return status;
}

typedef struct
{
    size_t num_written;
    size_t num_to_write;
    size_t num_to_skip;
    uvm_gpu_chunk_t **subchunks;
} get_subchunks_walk_t;

static NV_STATUS get_subchunks_walk_func(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    get_subchunks_walk_t *args = data;

    // We're only collecting leaf chunks
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT)
        return NV_OK;

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    if (args->num_to_skip) {
        --args->num_to_skip;
        return NV_OK;
    }

    UVM_ASSERT(args->num_written < args->num_to_write);
    args->subchunks[args->num_written++] = chunk;

    // Bail immediately once we hit our limit. Note that this is not an error:
    // we just need to exit the walk.
    if (args->num_written == args->num_to_write)
        return NV_ERR_OUT_OF_RANGE;

    return NV_OK;
}

size_t uvm_pmm_gpu_get_subchunks(uvm_pmm_gpu_t *pmm,
                                 uvm_gpu_chunk_t *parent,
                                 size_t start_index,
                                 size_t num_subchunks,
                                 uvm_gpu_chunk_t **subchunks)
{
    NV_STATUS status;

    get_subchunks_walk_t walk_args =
    {
        .num_written    = 0,
        .num_to_write   = num_subchunks,
        .num_to_skip    = start_index,
        .subchunks      = subchunks,
    };

    if (num_subchunks == 0)
        return 0;

    UVM_ASSERT(parent->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               parent->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
               parent->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);

    uvm_mutex_lock(&pmm->lock);

    // Either pre- or post-order would work. Pick post-order just because we
    // only care about leaf chunks and we may exit early, so we'd get slightly
    // fewer callbacks.
    status = chunk_walk_post_order(pmm, parent, get_subchunks_walk_func, &walk_args);
    if (status != NV_OK) {
        UVM_ASSERT(status == NV_ERR_OUT_OF_RANGE);
        UVM_ASSERT(walk_args.num_written == walk_args.num_to_write);
    }

    uvm_mutex_unlock(&pmm->lock);
    return walk_args.num_written;
}

static uvm_gpu_chunk_t *list_first_chunk(struct list_head *list)
{
    return list_first_entry_or_null(list, uvm_gpu_chunk_t, list);
}

void uvm_pmm_gpu_merge_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_mutex_lock(&pmm->lock);
    uvm_pmm_gpu_merge_chunk_locked(pmm, chunk);
    uvm_mutex_unlock(&pmm->lock);
}

static void root_chunk_unmap_indirect_peer(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk, uvm_gpu_t *other_gpu)
{
    NvU64 *dma_addrs = pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].dma_addrs;
    size_t index = root_chunk_index(pmm, root_chunk);
    long long new_count;
    NV_STATUS status;

    uvm_assert_root_chunk_locked(pmm, root_chunk);
    UVM_ASSERT(dma_addrs);
    UVM_ASSERT(root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);
    UVM_ASSERT(uvm_processor_mask_test(&root_chunk->indirect_peers_mapped, other_gpu->id));

    // The tracker could have work which require the indirect peer mappings to
    // remain until finished, such as PTE unmaps of this chunk from indirect
    // peers, so we need to wait. We also need to wait on the entire tracker,
    // not just other_gpu's entries, because there might be implicit chained
    // dependencies in the tracker.
    //
    // We know there can't be any other work which requires these mappings:
    // - If we're freeing the root chunk back to PMA or switching types of the
    //   root chunk, nothing else can reference the chunk.
    //
    // - If the chunk is still allocated then global peer access must be in the
    //   process of being disabled, say because one of the GPUs is being
    //   unregistered. We know that all VA spaces must have already called
    //   disable_peers and have waited on those PTE unmaps. The chunk could be
    //   freed concurrently with this indirect peer unmap, but that will be
    //   serialized by the root chunk lock.
    status = uvm_tracker_wait(&root_chunk->tracker);
    if (status != NV_OK)
        UVM_ASSERT(uvm_global_get_status() != NV_OK);

    uvm_gpu_unmap_cpu_pages(other_gpu, dma_addrs[index], UVM_CHUNK_SIZE_MAX);
    uvm_processor_mask_clear(&root_chunk->indirect_peers_mapped, other_gpu->id);
    new_count = atomic64_dec_return(&pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].map_count);
    UVM_ASSERT(new_count >= 0);
}

static void root_chunk_unmap_indirect_peers(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    uvm_gpu_t *other_gpu;
    for_each_gpu_in_mask(other_gpu, &root_chunk->indirect_peers_mapped)
        root_chunk_unmap_indirect_peer(pmm, root_chunk, other_gpu);
}

NV_STATUS uvm_pmm_gpu_indirect_peer_init(uvm_pmm_gpu_t *pmm, uvm_gpu_t *accessing_gpu)
{
    NvU64 *dma_addrs;
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(uvm_gpus_are_indirect_peers(pmm->gpu, accessing_gpu));
    UVM_ASSERT(!pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].dma_addrs);
    UVM_ASSERT(atomic64_read(&pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].map_count) == 0);

    // Each root chunk tracks whether it has a mapping to a given indirect peer,
    // so we don't need to initialize this array.
    dma_addrs = uvm_kvmalloc(pmm->root_chunks.count * sizeof(dma_addrs[0]));
    if (!dma_addrs)
        status = NV_ERR_NO_MEMORY;
    else
        pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].dma_addrs = dma_addrs;

    return status;
}

static bool check_indirect_peer_empty(uvm_pmm_gpu_t *pmm, uvm_gpu_t *other_gpu)
{
    atomic64_t *map_count = &pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].map_count;
    size_t i;

    for (i = 0; i < pmm->root_chunks.count; i++) {
        uvm_gpu_root_chunk_t *root_chunk = &pmm->root_chunks.array[i];

        // This doesn't take the root chunk lock because checking the mask is an
        // atomic operation.
        if (uvm_processor_mask_test(&root_chunk->indirect_peers_mapped, other_gpu->id)) {
            UVM_ASSERT(atomic64_read(map_count) > 0);
            return false;
        }
    }

    UVM_ASSERT(atomic64_read(map_count) == 0);
    return true;
}

void uvm_pmm_gpu_indirect_peer_destroy(uvm_pmm_gpu_t *pmm, uvm_gpu_t *other_gpu)
{
    NvU64 *dma_addrs = pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].dma_addrs;
    atomic64_t *map_count = &pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].map_count;
    size_t i;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(uvm_gpus_are_indirect_peers(pmm->gpu, other_gpu));

    if (!dma_addrs) {
        UVM_ASSERT(check_indirect_peer_empty(pmm, other_gpu));
        return;
    }

    // Just go over all root chunks and unmap them. This is slow, but it is not
    // a frequent operation.
    for (i = 0; i < pmm->root_chunks.count && atomic64_read(map_count); i++) {
        uvm_gpu_root_chunk_t *root_chunk = &pmm->root_chunks.array[i];

        // Take the root chunk lock to prevent chunks from transitioning in or
        // out of the PMA_OWNED state, and to serialize updates to the tracker
        // and indirect_peers_mapped mask. Note that indirect peers besides
        // other_gpu could be trying to create mappings concurrently.
        root_chunk_lock(pmm, root_chunk);

        if (root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED)
            UVM_ASSERT(uvm_processor_mask_empty(&root_chunk->indirect_peers_mapped));
        else if (uvm_processor_mask_test(&root_chunk->indirect_peers_mapped, other_gpu->id))
            root_chunk_unmap_indirect_peer(pmm, root_chunk, other_gpu);

        root_chunk_unlock(pmm, root_chunk);
    }

    UVM_ASSERT(check_indirect_peer_empty(pmm, other_gpu));

    uvm_kvfree(dma_addrs);
    pmm->root_chunks.indirect_peer[uvm_gpu_index(other_gpu->id)].dma_addrs = NULL;
}

NV_STATUS uvm_pmm_gpu_indirect_peer_map(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);
    size_t index = root_chunk_index(pmm, root_chunk);
    NvU64 *dma_addrs = pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].dma_addrs;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);

    UVM_ASSERT(uvm_gpus_are_indirect_peers(pmm->gpu, accessing_gpu));
    UVM_ASSERT(dma_addrs);

    // Serialize:
    //  - Concurrent mappings to this root chunk (same or different GPUs)
    //  - Concurrent unmappings of this root chunk (must be a different GPU)
    root_chunk_lock(pmm, root_chunk);

    if (!uvm_processor_mask_test(&root_chunk->indirect_peers_mapped, accessing_gpu->id)) {
        status = uvm_gpu_map_cpu_pages(accessing_gpu,
                                       uvm_gpu_chunk_to_page(pmm, &root_chunk->chunk),
                                       UVM_CHUNK_SIZE_MAX,
                                       &dma_addrs[index]);
        if (status == NV_OK) {
            uvm_processor_mask_set(&root_chunk->indirect_peers_mapped, accessing_gpu->id);
            atomic64_inc(&pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].map_count);
        }
    }

    root_chunk_unlock(pmm, root_chunk);
    return status;
}

NvU64 uvm_pmm_gpu_indirect_peer_addr(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu)
{
    NvU64 *dma_addrs = pmm->root_chunks.indirect_peer[uvm_gpu_index(accessing_gpu->id)].dma_addrs;
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);
    size_t index = root_chunk_index(pmm, root_chunk);
    NvU64 chunk_offset = chunk->address - root_chunk->chunk.address;

    UVM_ASSERT(uvm_gpus_are_indirect_peers(pmm->gpu, accessing_gpu));
    UVM_ASSERT(dma_addrs);
    UVM_ASSERT(uvm_processor_mask_test(&root_chunk->indirect_peers_mapped, accessing_gpu->id));
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);

    return dma_addrs[index] + chunk_offset;
}

static NV_STATUS evict_root_chunk_from_va_block(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk, uvm_va_block_t *va_block)
{
    NV_STATUS status;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();

    UVM_ASSERT(va_block);

    // To evict the chunks from the VA block we need to lock it, but we already
    // have the PMM lock held. Unlock it first and re-lock it after.
    uvm_mutex_unlock(&pmm->lock);

    uvm_mutex_lock(&va_block->lock);

    status = uvm_va_block_evict_chunks(va_block, pmm->gpu, &root_chunk->chunk, &tracker);

    uvm_mutex_unlock(&va_block->lock);

    // The block has been retained by find_and_retain_va_block_to_evict(),
    // release it here as it's not needed any more. Notably do that even if
    // uvm_va_block_evict_chunks() fails.
    uvm_va_block_release(va_block);

    if (status == NV_OK) {
        root_chunk_lock(pmm, root_chunk);
        status = uvm_tracker_add_tracker_safe(&root_chunk->tracker, &tracker);
        root_chunk_unlock(pmm, root_chunk);
    }

    uvm_tracker_deinit(&tracker);

    uvm_mutex_lock(&pmm->lock);

    return status;
}

void uvm_pmm_gpu_mark_chunk_evicted(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(chunk_is_in_eviction(pmm, chunk));
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
    UVM_ASSERT(chunk->va_block != NULL);

    chunk->va_block = NULL;
    chunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;
    chunk_pin(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);
}

static NV_STATUS pin_free_chunks_func(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    uvm_assert_mutex_locked(&pmm->lock);

    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(chunk_is_in_eviction(pmm, chunk));

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE) {
        list_del_init(&chunk->list);
        chunk_pin(pmm, chunk);
        if (chunk->parent)
            chunk->parent->suballoc->allocated++;
    }

    uvm_spin_unlock(&pmm->list_lock);

    return NV_OK;
}

static NV_STATUS free_first_pinned_chunk_func(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    uvm_assert_mutex_locked(&pmm->lock);

    UVM_ASSERT(!chunk_is_in_eviction(pmm, chunk));

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED) {
        free_chunk_with_merges(pmm, chunk);
        return NV_ERR_MORE_DATA_AVAILABLE;
    }

    return NV_OK;
}

typedef struct
{
    uvm_va_block_t *va_block_to_evict_from;
} evict_data_t;

static NV_STATUS find_and_retain_va_block_to_evict(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    NV_STATUS status = NV_OK;
    evict_data_t *evict_data = (evict_data_t *)data;

    UVM_ASSERT(evict_data->va_block_to_evict_from == NULL);

    uvm_spin_lock(&pmm->list_lock);

    // All free chunks should have been pinned already by pin_free_chunks_func().
    UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
                   chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED ||
                   chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT,
                   "state %s\n", uvm_pmm_gpu_chunk_state_string(chunk->state));

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
        UVM_ASSERT(chunk->va_block);
        evict_data->va_block_to_evict_from = chunk->va_block;
        uvm_va_block_retain(chunk->va_block);
        status = NV_ERR_MORE_DATA_AVAILABLE;
    }

    uvm_spin_unlock(&pmm->list_lock);

    return status;
}

static NV_STATUS evict_root_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk)
{
    NV_STATUS status;
    NV_STATUS free_status;
    uvm_gpu_chunk_t *chunk = &root_chunk->chunk;

    uvm_assert_mutex_locked(&pmm->lock);

    // First pin all the free subchunks
    status = chunk_walk_pre_order(pmm, chunk, pin_free_chunks_func, NULL);
    UVM_ASSERT(status == NV_OK);
    while (1) {
        evict_data_t evict = {0};
        status = chunk_walk_pre_order(pmm, chunk, find_and_retain_va_block_to_evict, &evict);

        // find_and_retain_va_block_to_evict() returns NV_ERR_MORE_DATA_AVAILABLE
        // immediately after finding the first VA block to evict from and NV_OK
        // if no more blocks are left.
        if (status != NV_ERR_MORE_DATA_AVAILABLE) {
            UVM_ASSERT(status == NV_OK);
            break;
        }

        // Evict the chunks from the VA block. Notably this will unlock and
        // re-lock the PMM mutex. This is ok as we don't rely on any PMM state
        // that can change across the calls. In particular, the walk to pick the
        // next VA block to evict above is always started from the root chunk.
        status = evict_root_chunk_from_va_block(pmm, root_chunk, evict.va_block_to_evict_from);
        if (status != NV_OK)
            goto error;
    }

    // All of the leaf chunks should be pinned now, merge them all back into a
    // pinned root chunk.
    uvm_pmm_gpu_merge_chunk_locked(pmm, chunk);

    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);
    uvm_gpu_chunk_set_in_eviction(chunk, false);

    uvm_spin_unlock(&pmm->list_lock);

    return NV_OK;

error:
    // On error we need to free all the chunks that we were able to evict so
    // far. They should all be pinned.

    // Clear the eviction state so any new chunks freed by other threads are
    // actually freed instead of pinned. We need the list lock to make the
    // eviction check and conditional pin in chunk_free_locked atomic with our
    // free-if-pinned loop below.
    uvm_spin_lock(&pmm->list_lock);

    uvm_gpu_chunk_set_in_eviction(chunk, false);

    // In case we didn't manage to evict any chunks and hence the root is still
    // unpinned, we need to put it back on an eviction list.
    // chunk_update_lists_locked() will do that.
    chunk_update_lists_locked(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);

    do {
        free_status = chunk_walk_pre_order(pmm, chunk, free_first_pinned_chunk_func, NULL);
    } while (free_status == NV_ERR_MORE_DATA_AVAILABLE);
    UVM_ASSERT(free_status == NV_OK);

    free_next_available_root_chunk(pmm);

    return status;
}

static bool chunk_is_evictable(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (root_chunk->chunk.state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED)
        return false;

    if (chunk_is_root_chunk_pinned(pmm, chunk))
        return false;

    if (chunk_is_in_eviction(pmm, chunk))
        return false;

    // An evictable chunk's root should be on one of the eviction lists.
    UVM_ASSERT(!list_empty(&root_chunk->chunk.list));

    return true;
}

static void chunk_start_eviction(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);
    chunk = &root_chunk->chunk;

    uvm_assert_spinlock_locked(&pmm->list_lock);

    UVM_ASSERT(chunk_is_evictable(pmm, chunk));
    UVM_ASSERT(!list_empty(&chunk->list));

    list_del_init(&chunk->list);
    uvm_gpu_chunk_set_in_eviction(chunk, true);
}

static void root_chunk_update_eviction_list(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX);
    UVM_ASSERT(chunk->type == UVM_PMM_GPU_MEMORY_TYPE_USER);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    if (!chunk_is_root_chunk_pinned(pmm, chunk) && !chunk_is_in_eviction(pmm, chunk)) {
        // An unpinned chunk not selected for eviction should be on one of the
        // eviction lists.
        UVM_ASSERT(!list_empty(&chunk->list));

        list_move_tail(&chunk->list, list);
    }

    uvm_spin_unlock(&pmm->list_lock);
}

void uvm_pmm_gpu_mark_root_chunk_used(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_used);
}

void uvm_pmm_gpu_mark_root_chunk_unused(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    root_chunk_update_eviction_list(pmm, chunk, &pmm->root_chunks.va_block_unused);
}

static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    uvm_spin_lock(&pmm->list_lock);

    chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // TODO: Bug 1765193: Move the chunks to the tail of the used list whenever
    // they get mapped.
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    if (chunk)
        chunk_start_eviction(pmm, chunk);

    uvm_spin_unlock(&pmm->list_lock);

    if (chunk)
        return root_chunk_from_chunk(pmm, chunk);
    return NULL;
}

static NV_STATUS pick_and_evict_root_chunk(uvm_pmm_gpu_t *pmm,
                                           uvm_pmm_gpu_memory_type_t type,
                                           uvm_pmm_context_t pmm_context,
                                           uvm_gpu_chunk_t **out_chunk)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk;
    uvm_gpu_root_chunk_t *root_chunk;

    UVM_ASSERT(uvm_gpu_supports_eviction(pmm->gpu));

    uvm_assert_mutex_locked(&pmm->lock);

    root_chunk = pick_root_chunk_to_evict(pmm);
    if (!root_chunk)
        return NV_ERR_NO_MEMORY;

    status = evict_root_chunk(pmm, root_chunk);
    if (status != NV_OK)
        return status;

    chunk = &root_chunk->chunk;

    if (type == UVM_PMM_GPU_MEMORY_TYPE_KERNEL) {
        NvU32 flags = 0;
        if (pmm_context == PMM_CONTEXT_PMA_EVICTION)
            flags |= UVM_PMA_CALLED_FROM_PMA_EVICTION;

        // Transitioning user memory type to kernel memory type requires pinning
        // it so that PMA doesn't pick it for eviction.
        status = nvUvmInterfacePmaPinPages(pmm->pma,
                                           &chunk->address,
                                           1,
                                           UVM_CHUNK_SIZE_MAX,
                                           flags);
        if (status == NV_ERR_IN_USE) {
            // Pinning can fail if some of the pages have been chosen for
            // eviction already. In that case free the root chunk back to PMA
            // and let the caller retry.
            free_root_chunk(pmm, root_chunk, free_root_chunk_mode_from_pmm_context(pmm_context));

            return status;
        }

        UVM_ASSERT_MSG(status == NV_OK,
                       "pmaPinPages(root_chunk=0x%llx) failed unexpectedly: %s\n",
                       chunk->address,
                       nvstatusToString(status));

        // Unmap any indirect peer physical mappings for this chunk, since
        // kernel chunks generally don't need them.
        root_chunk_lock(pmm, root_chunk);
        root_chunk_unmap_indirect_peers(pmm, root_chunk);
        root_chunk_unlock(pmm, root_chunk);

        uvm_spin_lock(&pmm->list_lock);
        chunk->type = UVM_PMM_GPU_MEMORY_TYPE_KERNEL;
        uvm_spin_unlock(&pmm->list_lock);
    }

    *out_chunk = chunk;
    return NV_OK;
}

static NV_STATUS pick_and_evict_root_chunk_retry(uvm_pmm_gpu_t *pmm,
                                                 uvm_pmm_gpu_memory_type_t type,
                                                 uvm_pmm_context_t pmm_context,
                                                 uvm_gpu_chunk_t **out_chunk)
{
    NV_STATUS status;

    // Eviction can fail if the chunk gets selected for PMA eviction at
    // the same time. Keep retrying.
    do {
        status = pick_and_evict_root_chunk(pmm, type, pmm_context, out_chunk);
    } while (status == NV_ERR_IN_USE);

    return status;
}

static uvm_gpu_chunk_t *claim_free_chunk(uvm_pmm_gpu_t *pmm, uvm_pmm_gpu_memory_type_t type, uvm_chunk_size_t chunk_size)
{
    uvm_gpu_chunk_t *chunk;
    struct list_head *free_list = find_free_list(pmm, type, chunk_size);

    uvm_spin_lock(&pmm->list_lock);

    chunk = list_first_chunk(free_list);

    // Remove chunks that have been picked for eviction from the free lists. The
    // eviction path does it with pin_free_chunks_func(), but there is a window
    // between when a root chunk is chosen for eviction and all of its subchunks
    // are removed from free lists.
    while (chunk && chunk_is_in_eviction(pmm, chunk)) {
        list_del_init(&chunk->list);
        chunk = list_first_chunk(free_list);
    }

    if (!chunk)
        goto out;

    UVM_ASSERT_MSG(uvm_gpu_chunk_get_size(chunk) == chunk_size, "chunk size %u expected %u\n",
            uvm_gpu_chunk_get_size(chunk), chunk_size);
    UVM_ASSERT(chunk->type == type);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE);
    UVM_ASSERT(!chunk_is_in_eviction(pmm, chunk));

    if (chunk->parent) {
        UVM_ASSERT(chunk->parent->suballoc);
        UVM_ASSERT(chunk->parent->type == type);
        UVM_ASSERT(chunk->parent->suballoc->allocated < num_subchunks(chunk->parent));
        chunk->parent->suballoc->allocated++;
    }

    chunk_pin(pmm, chunk);
    chunk_update_lists_locked(pmm, chunk);

out:
    uvm_spin_unlock(&pmm->list_lock);

    return chunk;
}

static NV_STATUS alloc_or_evict_root_chunk(uvm_pmm_gpu_t *pmm,
                                           uvm_pmm_gpu_memory_type_t type,
                                           uvm_pmm_alloc_flags_t flags,
                                           uvm_gpu_chunk_t **chunk_out)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk;

    status = alloc_root_chunk(pmm, type, &chunk);
    if (status != NV_OK) {
        if ((flags & UVM_PMM_ALLOC_FLAGS_EVICT) && uvm_gpu_supports_eviction(pmm->gpu))
            status = pick_and_evict_root_chunk_retry(pmm, type, PMM_CONTEXT_DEFAULT, chunk_out);

        return status;
    }

    *chunk_out = chunk;
    return status;
}

// Same as alloc_or_evit_root_chunk(), but without the PMM lock held.
static NV_STATUS alloc_or_evict_root_chunk_unlocked(uvm_pmm_gpu_t *pmm,
                                                    uvm_pmm_gpu_memory_type_t type,
                                                    uvm_pmm_alloc_flags_t flags,
                                                    uvm_gpu_chunk_t **chunk_out)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk;

    status = alloc_root_chunk(pmm, type, &chunk);
    if (status != NV_OK) {
        if ((flags & UVM_PMM_ALLOC_FLAGS_EVICT) && uvm_gpu_supports_eviction(pmm->gpu)) {
            uvm_mutex_lock(&pmm->lock);
            status = pick_and_evict_root_chunk_retry(pmm, type, PMM_CONTEXT_DEFAULT, chunk_out);
            uvm_mutex_unlock(&pmm->lock);
        }

        return status;
    }

    *chunk_out = chunk;
    return status;
}

static NV_STATUS alloc_chunk_with_splits(uvm_pmm_gpu_t *pmm,
                                         uvm_pmm_gpu_memory_type_t type,
                                         uvm_chunk_size_t chunk_size,
                                         uvm_pmm_alloc_flags_t flags,
                                         uvm_gpu_chunk_t **out_chunk)
{
    NV_STATUS status;
    uvm_chunk_size_t cur_size;
    uvm_gpu_chunk_t *chunk;
    uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[type];

    uvm_assert_mutex_locked(&pmm->lock);
    UVM_ASSERT(chunk_size != UVM_CHUNK_SIZE_MAX);

    // Check for a free chunk again in case a different thread freed something
    // up while this thread was waiting for the PMM lock.
    chunk = claim_free_chunk(pmm, type, chunk_size);
    if (chunk) {
        // A free chunk was claimed, return immediately.
        UVM_ASSERT(check_chunk(pmm, chunk));

        *out_chunk = chunk;
        return NV_OK;
    }

    cur_size = chunk_size;

    // Look for a bigger free chunk that can be split
    for_each_chunk_size_from(cur_size, chunk_sizes) {
        chunk = claim_free_chunk(pmm, type, cur_size);
        if (chunk)
            break;
    }

    if (unlikely(!chunk)) {
        status = alloc_or_evict_root_chunk(pmm, type, flags, &chunk);
        if (status != NV_OK)
            return status;
        cur_size = UVM_CHUNK_SIZE_MAX;
        UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == cur_size);
    }

    UVM_ASSERT(chunk);

    for_each_chunk_size_rev_from(cur_size, chunk_sizes) {
        NvU32 i;
        uvm_gpu_chunk_t *parent;

        UVM_ASSERT(uvm_gpu_chunk_get_size(chunk)  == cur_size);
        UVM_ASSERT(chunk->type  == type);
        UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

        if (chunk->parent) {
            UVM_ASSERT(chunk->parent->suballoc);
            UVM_ASSERT(uvm_gpu_chunk_get_size(chunk->parent) == uvm_chunk_find_next_size(chunk_sizes, cur_size));
            UVM_ASSERT(chunk->parent->type == type);
            UVM_ASSERT_MSG(chunk->parent->suballoc->allocated <= num_subchunks(chunk->parent), "allocated %u num %u\n",
                    chunk->parent->suballoc->allocated, num_subchunks(chunk->parent));
        }

        if (cur_size == chunk_size) {
            *out_chunk = chunk;
            return NV_OK;
        }

        status = split_gpu_chunk(pmm, chunk);
        if (status != NV_OK) {
            free_chunk_with_merges(pmm, chunk);
            return status;
        }

        parent = chunk;

        // Use the first subchunk for further splitting, if needed.
        chunk = parent->suballoc->subchunks[0];

        // And add the rest to the free list
        uvm_spin_lock(&pmm->list_lock);

        for (i = 1; i < num_subchunks(parent); ++i)
            chunk_free_locked(pmm, parent->suballoc->subchunks[i]);

        uvm_spin_unlock(&pmm->list_lock);
    }
    UVM_PANIC();
}

// Allocates a single chunk of a given size. If needed splits a chunk of bigger size
// or, if that is not possible, allocates from PMA or evicts.
NV_STATUS alloc_chunk(uvm_pmm_gpu_t *pmm,
                      uvm_pmm_gpu_memory_type_t type,
                      uvm_chunk_size_t chunk_size,
                      uvm_pmm_alloc_flags_t flags,
                      uvm_gpu_chunk_t **out_chunk)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk;

    chunk = claim_free_chunk(pmm, type, chunk_size);
    if (chunk) {
        // A free chunk could be claimed, we are done.
        *out_chunk = chunk;
        return NV_OK;
    }

    if (chunk_size == UVM_CHUNK_SIZE_MAX) {
        // For chunks of root chunk size we won't be doing any splitting so we
        // can just directly try allocating without holding the PMM lock. If
        // eviction is necessary, the lock will be acquired internally.
        status = alloc_or_evict_root_chunk_unlocked(pmm, type, flags, &chunk);
        if (status != NV_OK)
            return status;

        *out_chunk = chunk;
        return NV_OK;
    }

    // We didn't find a free chunk and we will require splits so acquire the PMM lock.
    uvm_mutex_lock(&pmm->lock);

    status = alloc_chunk_with_splits(pmm, type, chunk_size, flags, &chunk);

    uvm_mutex_unlock(&pmm->lock);

    if (status != NV_OK) {
        free_next_available_root_chunk(pmm);
        return status;
    }

    *out_chunk = chunk;

    return NV_OK;
}

NV_STATUS alloc_root_chunk(uvm_pmm_gpu_t *pmm,
                     uvm_pmm_gpu_memory_type_t type,
                     uvm_gpu_chunk_t **out_chunk)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk;
    uvm_gpu_root_chunk_t *root_chunk;
    UvmGpuPointer pa = ~(UvmGpuPointer)0;
    UvmPmaAllocationOptions options = {0};

    options.flags = UVM_PMA_ALLOCATE_DONT_EVICT;
    switch (type) {
        case UVM_PMM_GPU_MEMORY_TYPE_USER:
            if (!gpu_supports_pma_eviction(pmm->gpu))
                options.flags |= UVM_PMA_ALLOCATE_PINNED;
            break;
        case UVM_PMM_GPU_MEMORY_TYPE_KERNEL:
            options.flags |= UVM_PMA_ALLOCATE_PINNED;
            break;
        default:
            UVM_ASSERT(0);
    }

    // Acquire the PMA lock for read so that uvm_pmm_gpu_pma_evict_range() can
    // flush out any pending allocs.
    uvm_down_read(&pmm->pma_lock);

    status = nvUvmInterfacePmaAllocPages(pmm->pma, 1, UVM_CHUNK_SIZE_MAX, &options, &pa);
    if (status != NV_OK) {
        uvm_up_read(&pmm->pma_lock);
        return status;
    }

    UVM_ASSERT_MSG(IS_ALIGNED(pa, UVM_CHUNK_SIZE_MAX), "Address 0x%llx\n", pa);
    UVM_ASSERT_MSG(pa + UVM_CHUNK_SIZE_MAX - 1 <= pmm->gpu->vidmem_max_allocatable_address,
                   "Address 0x%llx, max physical 0x%llx\n",
                   pa,
                   pmm->gpu->vidmem_max_allocatable_address);

    root_chunk = root_chunk_from_address(pmm, pa);
    chunk = &root_chunk->chunk;

    root_chunk_lock(pmm, root_chunk);

    uvm_tracker_init(&root_chunk->tracker);

    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED,
                   "Address 0x%llx state %s GPU %s\n",
                   pa,
                   uvm_pmm_gpu_chunk_state_string(chunk->state),
                   pmm->gpu->name);

    UVM_ASSERT(chunk->address == pa);
    UVM_ASSERT(chunk->parent == NULL);
    UVM_ASSERT(chunk->suballoc == NULL);
    UVM_ASSERT(chunk->va_block == NULL);
    UVM_ASSERT(chunk->va_block_page_index == PAGES_PER_UVM_VA_BLOCK);
    UVM_ASSERT(list_empty(&chunk->list));
    UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == UVM_CHUNK_SIZE_MAX);

    chunk->type = type;
    chunk->state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED;

    uvm_spin_unlock(&pmm->list_lock);

    root_chunk_unlock(pmm, root_chunk);

    uvm_up_read(&pmm->pma_lock);

    *out_chunk = chunk;
    return NV_OK;
}

void free_root_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root_chunk, free_root_chunk_mode_t free_mode)
{
    NV_STATUS status;
    uvm_gpu_chunk_t *chunk = &root_chunk->chunk;
    NvU32 flags = 0;

    // Acquire the PMA lock for read so that uvm_pmm_gpu_pma_evict_range() can
    // flush out any pending frees.
    uvm_down_read(&pmm->pma_lock);

    root_chunk_lock(pmm, root_chunk);

    root_chunk_unmap_indirect_peers(pmm, root_chunk);

    status = uvm_tracker_wait_deinit(&root_chunk->tracker);
    if (status != NV_OK) {
        // TODO: Bug 1766184: Handle RC/ECC. For now just go ahead and free the chunk anyway.
        UVM_ASSERT(uvm_global_get_status() != NV_OK);
    }

    uvm_spin_lock(&pmm->list_lock);

    UVM_ASSERT_MSG(chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED,
                   "Address 0x%llx state %s GPU %s\n",
                   chunk->address,
                   uvm_pmm_gpu_chunk_state_string(chunk->state),
                   pmm->gpu->name);
    UVM_ASSERT(list_empty(&chunk->list));

    chunk_unpin(pmm, chunk, UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);

    uvm_spin_unlock(&pmm->list_lock);

    root_chunk_unlock(pmm, root_chunk);

    if (free_mode == FREE_ROOT_CHUNK_MODE_SKIP_PMA_FREE) {
        uvm_up_read(&pmm->pma_lock);
        return;
    }

    if (free_mode == FREE_ROOT_CHUNK_MODE_PMA_EVICTION)
        flags |= UVM_PMA_CALLED_FROM_PMA_EVICTION;

    nvUvmInterfacePmaFreePages(pmm->pma, &chunk->address, 1, UVM_CHUNK_SIZE_MAX, flags);

    uvm_up_read(&pmm->pma_lock);
}

// Splits the input chunk into subchunks of the next size down. The chunk state
// can be UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED or UVM_PMM_GPU_CHUNK_STATE_ALLOCATED.
//
// UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED: This is a split for allocation.
//
// UVM_PMM_GPU_CHUNK_STATE_ALLOCATED: This is an in-place split. The new chunks
// are also marked allocated and they inherit the reverse map from the original.
//
// The PMM lock must be held when calling this function.
NV_STATUS split_gpu_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_chunk_size_t chunk_size = uvm_gpu_chunk_get_size(chunk);
    uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[chunk->type];
    uvm_chunk_size_t subchunk_size;
    size_t cache_idx, num_sub;
    int i;
    NV_STATUS status;
    uvm_pmm_gpu_chunk_suballoc_t *suballoc;
    uvm_gpu_chunk_t *subchunk;
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_mutex_locked(&pmm->lock);
    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    subchunk_size = uvm_chunk_find_prev_size(chunk_sizes, chunk_size);
    UVM_ASSERT(subchunk_size != UVM_CHUNK_SIZE_INVALID);

    num_sub = chunk_size / subchunk_size;
    cache_idx = ilog2(num_sub);
    UVM_ASSERT(chunk_split_cache[cache_idx].cache != NULL);

    suballoc = kmem_cache_zalloc(chunk_split_cache[cache_idx].cache, NV_UVM_GFP_FLAGS);
    if (suballoc == NULL)
        return NV_ERR_NO_MEMORY;

    for (i = 0; i < num_sub; i++) {
        // If requested, inject a failure on the last subchunk
        if (unlikely(chunk->inject_split_error) && i == num_sub - 1) {
            status = NV_ERR_NO_MEMORY;
            goto cleanup;
        }

        subchunk = kmem_cache_zalloc(CHUNK_CACHE, NV_UVM_GFP_FLAGS);
        if (!subchunk) {
            status = NV_ERR_NO_MEMORY;
            goto cleanup;
        }
        suballoc->subchunks[i] = subchunk;

        subchunk->address = chunk->address + i * subchunk_size;
        subchunk->type = chunk->type;
        uvm_gpu_chunk_set_size(subchunk, subchunk_size);
        subchunk->parent = chunk;
        subchunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;
        INIT_LIST_HEAD(&subchunk->list);

        // The child inherits the parent's state.
        subchunk->state = chunk->state;

        if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
            UVM_ASSERT(chunk->va_block);
            uvm_assert_mutex_locked(&chunk->va_block->lock);
            subchunk->va_block = chunk->va_block;
            subchunk->va_block_page_index = chunk->va_block_page_index + (i * subchunk_size) / PAGE_SIZE;
        }
    }

    // We're splitting an allocated or pinned chunk in-place.
    suballoc->allocated = num_sub;

    // Now that all of the subchunk state has been initialized, transition the
    // parent into the split state under the list lock.
    uvm_spin_lock(&pmm->list_lock);

    chunk->suballoc = suballoc;

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
        chunk->va_block = NULL;
        chunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;
    }
    else if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED) {
        // -1 for the parent chunk that is going to transition into the split state.
        root_chunk->chunk.suballoc->pinned_leaf_chunks += num_sub - 1;

        // When a pinned root chunk gets split, the count starts at 0 not
        // accounting for the root chunk itself so add the 1 back.
        if (chunk_is_root_chunk(chunk))
            root_chunk->chunk.suballoc->pinned_leaf_chunks += 1;
    }

    chunk->state = UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT;

    uvm_spin_unlock(&pmm->list_lock);

    return NV_OK;
cleanup:
    for (i = 0; i < num_sub; i++) {
        if (suballoc->subchunks[i] == NULL)
            break;
        kmem_cache_free(CHUNK_CACHE, suballoc->subchunks[i]);
    }
    kmem_cache_free(chunk_split_cache[cache_idx].cache, suballoc);
    return status;
}

// Sanity check the chunk. Always returns true so it can be called from an
// assert macro.
static bool check_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[chunk->type];
    uvm_gpu_chunk_t *parent = chunk->parent;
    uvm_chunk_size_t chunk_size = uvm_gpu_chunk_get_size(chunk);
    uvm_chunk_size_t parent_size;

    UVM_ASSERT(chunk_size & chunk_sizes);
    UVM_ASSERT(IS_ALIGNED(chunk->address, chunk_size));

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT)
        UVM_ASSERT(chunk_size > uvm_chunk_find_first_size(chunk_sizes));

    if (parent) {
        UVM_ASSERT(parent->type == chunk->type);

        parent_size = uvm_gpu_chunk_get_size(parent);
        UVM_ASSERT(uvm_chunk_find_next_size(chunk_sizes, chunk_size) == parent_size);
        UVM_ASSERT(parent_size <= uvm_chunk_find_last_size(chunk_sizes));

        UVM_ASSERT(parent->state == UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT);
        UVM_ASSERT(parent->suballoc);
        UVM_ASSERT(parent->suballoc->allocated > 0);
        UVM_ASSERT(parent->suballoc->allocated <= num_subchunks(parent));

        UVM_ASSERT(parent->address <= chunk->address);
        UVM_ASSERT(chunk->address < parent->address + parent_size);
    }
    else {
        UVM_ASSERT(chunk_size == uvm_chunk_find_last_size(chunk_sizes));
    }

    return true;
}

static bool chunk_is_last_allocated_child(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_assert_spinlock_locked(&pmm->list_lock);

    if (!chunk->parent)
        return false;

    return chunk->parent->suballoc->allocated == 1;
}

static void chunk_free_locked(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_chunk(pmm, chunk);

    uvm_assert_spinlock_locked(&pmm->list_lock);

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    if (root_chunk->chunk.in_eviction) {
        // A root chunk with pinned subchunks would never be picked for eviction
        // so this one has to be in the allocated state. Pin it and let the
        // evicting thread pick it up.
        UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED);
        UVM_ASSERT(chunk->va_block != NULL);
        UVM_ASSERT(chunk->va_block_page_index != PAGES_PER_UVM_VA_BLOCK);
        UVM_ASSERT(list_empty(&chunk->list));
        chunk->va_block = NULL;
        chunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;
        chunk_pin(pmm, chunk);
        return;
    }

    if (chunk->parent) {
        UVM_ASSERT(chunk->parent->suballoc->allocated > 0);
        --chunk->parent->suballoc->allocated;
        if (chunk->parent->suballoc->allocated == 0) {
            // Freeing the last subchunk should trigger a merge and the PMM
            // mutex is required to perform it.
            uvm_assert_mutex_locked(&pmm->lock);
        }
    }

    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED) {
        chunk_unpin(pmm, chunk, UVM_PMM_GPU_CHUNK_STATE_FREE);
    }
    else {
        chunk->state = UVM_PMM_GPU_CHUNK_STATE_FREE;
        chunk->va_block = NULL;
    }

    chunk->va_block_page_index = PAGES_PER_UVM_VA_BLOCK;

    chunk_update_lists_locked(pmm, chunk);
}

static bool try_chunk_free(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    bool freed = false;

    uvm_spin_lock(&pmm->list_lock);

    // Chunks that are the last allocated child need to trigger a merge and are
    // handled by free_or_prepare_for_merge().
    if (!chunk_is_last_allocated_child(pmm, chunk)) {
        chunk_free_locked(pmm, chunk);
        freed = true;
    }

    uvm_spin_unlock(&pmm->list_lock);

    return freed;
}

// Return NULL if the chunk could be freed immediately. Otherwise, if the chunk
// was the last allocated child, return the parent chunk to be merged with all
// of its children taken off the free list in TEMP_PINNED state.
static uvm_gpu_chunk_t *free_or_prepare_for_merge(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_chunk_t *parent = NULL;
    NvU32 i;

    uvm_assert_mutex_locked(&pmm->lock);

    if (!chunk->parent) {
        bool freed = try_chunk_free(pmm, chunk);

        // Freeing a root chunk should never fail
        UVM_ASSERT(freed);

        return NULL;
    }

    uvm_spin_lock(&pmm->list_lock);

    if (chunk_is_last_allocated_child(pmm, chunk))
        parent = chunk->parent;

    chunk_free_locked(pmm, chunk);

    if (parent == NULL) {
        UVM_ASSERT(chunk->parent->suballoc->allocated != 0);
        goto done;
    }

    UVM_ASSERT(chunk->parent->suballoc->allocated == 0);

    // Pin all the subchunks to prepare them for being merged.
    for (i = 0; i < num_subchunks(chunk->parent); ++i) {
        uvm_gpu_chunk_t *subchunk = chunk->parent->suballoc->subchunks[i];

        UVM_ASSERT(subchunk->state == UVM_PMM_GPU_CHUNK_STATE_FREE);

        list_del_init(&subchunk->list);
        subchunk->state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED;
    }
    root_chunk_from_chunk(pmm, chunk)->chunk.suballoc->pinned_leaf_chunks += num_subchunks(chunk->parent);

    chunk->parent->suballoc->allocated = num_subchunks(chunk->parent);
    parent = chunk->parent;

done:
    uvm_spin_unlock(&pmm->list_lock);

    return parent;
}

static void free_chunk_with_merges(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    uvm_assert_mutex_locked(&pmm->lock);

    while (1) {
        UVM_ASSERT(check_chunk(pmm, chunk));

        chunk = free_or_prepare_for_merge(pmm, chunk);
        if (!chunk)
            break;

        merge_gpu_chunk(pmm, chunk);
    }
}

// Mark the chunk as free and put it on the free list. If this is a suballocated
// chunk and the parent has no more allocated chunks, the parent is freed and so
// on up the tree.
static void free_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    bool try_free = true;
    bool is_root = chunk_is_root_chunk(chunk);

    UVM_ASSERT(chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED ||
               chunk->state == UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED);

    chunk->inject_split_error = false;

    if (try_chunk_free(pmm, chunk)) {
        try_free = is_root;
    }
    else {
        // Freeing a chunk can only fail if it requires merging. Take the PMM lock
        // and free it with merges supported.
        uvm_mutex_lock(&pmm->lock);
        free_chunk_with_merges(pmm, chunk);
        uvm_mutex_unlock(&pmm->lock);
    }

    // Once try_chunk_free succeeds or free_chunk_with_merges returns, it's no
    // longer safe to access chunk in general. All you know is that the
    // chunk you freed was put on the free list by the call. Since the spin lock
    // has been dropped, any other thread could have come in and allocated the
    // chunk in the meantime. Therefore, this next step just looks for a
    // root chunk to free, without assuming that one is actually there.

    if (try_free)
        free_next_available_root_chunk(pmm);
}

// Finds and frees the next root chunk (if any) that can be freed.
void free_next_available_root_chunk(uvm_pmm_gpu_t *pmm)
{
    uvm_pmm_gpu_memory_type_t memory_type;
    uvm_chunk_size_t chunk_size;
    uvm_gpu_chunk_t *result;

    uvm_spin_lock(&pmm->list_lock);
    for (memory_type = 0; memory_type < UVM_PMM_GPU_MEMORY_TYPE_COUNT; memory_type++) {
        chunk_size = UVM_CHUNK_SIZE_MAX;
        UVM_ASSERT(uvm_chunk_find_last_size(pmm->chunk_sizes[memory_type]) == UVM_CHUNK_SIZE_MAX);
        result = list_first_chunk(find_free_list(pmm, memory_type, chunk_size));
        if (result != NULL) {
            list_del_init(&result->list);
            UVM_ASSERT(result->state == UVM_PMM_GPU_CHUNK_STATE_FREE);
            UVM_ASSERT(uvm_gpu_chunk_get_size(result) == chunk_size);
            UVM_ASSERT(result->type == memory_type);

            // The chunk has been freed and removed from the free list so it
            // can't get allocated again, but it could be targeted for eviction
            // by physical address. Pin it temporarily to protect the chunk from
            // eviction between dropping the list lock and taking the root chunk
            // lock.
            chunk_pin(pmm, result);
            break;
        }
    }
    uvm_spin_unlock(&pmm->list_lock);

    if (result != NULL)
        free_root_chunk(pmm, root_chunk_from_chunk(pmm, result), FREE_ROOT_CHUNK_MODE_DEFAULT);
}

// Get free list for given chunk size
struct list_head *find_free_list(uvm_pmm_gpu_t *pmm, uvm_pmm_gpu_memory_type_t type, uvm_chunk_size_t chunk_size)
{
    uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[type];
    size_t idx = hweight_long(chunk_sizes & (chunk_size - 1));
    UVM_ASSERT(is_power_of_2(chunk_size));
    UVM_ASSERT_MSG(chunk_size & chunk_sizes, "chunk size 0x%x chunk sizes 0x%x\n", chunk_size, chunk_sizes);
    return &pmm->free_list[type][idx];
}

struct list_head *find_free_list_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk)
{
    return find_free_list(pmm, chunk->type, uvm_gpu_chunk_get_size(chunk));
}

static bool uvm_pmm_should_inject_pma_eviction_error(uvm_pmm_gpu_t *pmm)
{
    uvm_assert_mutex_locked(&pmm->lock);

    if (unlikely(pmm->inject_pma_evict_error_after_num_chunks > 0))
        return --pmm->inject_pma_evict_error_after_num_chunks == 0;

    return false;
}

// See the documentation of pmaEvictPagesCb_t in pma.h for details of the
// expected semantics.
static NV_STATUS uvm_pmm_gpu_pma_evict_pages(void *void_pmm,
                                             NvU32 page_size,
                                             NvU64 *pages,
                                             NvU32 num_pages_to_evict,
                                             NvU64 phys_start,
                                             NvU64 phys_end)
{
    NV_STATUS status;
    uvm_pmm_gpu_t *pmm = (uvm_pmm_gpu_t *)void_pmm;
    uvm_gpu_chunk_t *chunk;
    NvU64 num_pages_evicted_so_far = 0;
    NvU64 num_pages_left_to_evict = num_pages_to_evict;
    const NvU64 pages_per_chunk = UVM_CHUNK_SIZE_MAX / page_size;

    UVM_ASSERT(IS_ALIGNED(UVM_CHUNK_SIZE_MAX, page_size));
    UVM_ASSERT(UVM_CHUNK_SIZE_MAX >= page_size);

    while (num_pages_left_to_evict > 0) {
        uvm_gpu_root_chunk_t *root_chunk;
        uvm_page_index_t page_index;
        NvU64 pages_this_time = min(pages_per_chunk, num_pages_left_to_evict);

        uvm_mutex_lock(&pmm->lock);

        if (uvm_pmm_should_inject_pma_eviction_error(pmm)) {
            status = NV_ERR_NO_MEMORY;
        }
        else {
            status = pick_and_evict_root_chunk_retry(pmm,
                                                     UVM_PMM_GPU_MEMORY_TYPE_KERNEL,
                                                     PMM_CONTEXT_PMA_EVICTION,
                                                     &chunk);
        }
        uvm_mutex_unlock(&pmm->lock);

        // TODO: Bug 1795559: Consider waiting for any pinned user allocations
        // to be unpinned.
        if (status != NV_OK)
            goto error;

        root_chunk = root_chunk_from_chunk(pmm, chunk);

        if (chunk->address < phys_start || chunk->address + UVM_CHUNK_SIZE_MAX > phys_end) {
            // If the chunk we get is outside of the physical range requested,
            // just give up and return an error.
            //
            // TODO: Bug 1795559: PMA pre-populates the array of pages with a
            // list of candidates that were unpinned before triggering eviction.
            // If they were marked for eviction, we could fall back to evicting
            // those instead and be sure that it succeeds.
            free_root_chunk(pmm, root_chunk, FREE_ROOT_CHUNK_MODE_PMA_EVICTION);
            status = NV_ERR_NO_MEMORY;
            goto error;
        }

        // Free the root chunk as far as PMM's state is concerned, but skip the
        // free back to PMA as that would make it available for other PMA
        // allocations.
        free_root_chunk(pmm, root_chunk, FREE_ROOT_CHUNK_MODE_SKIP_PMA_FREE);

        for (page_index = 0; page_index < pages_this_time; page_index++)
            pages[num_pages_evicted_so_far++] = chunk->address + page_index * page_size;

        num_pages_left_to_evict -= pages_this_time;

        // If we didn't use a whole root chunk, free its tail back to PMA
        // directly.
        if (pages_this_time != pages_per_chunk) {
            NvU64 address = chunk->address + pages_this_time * page_size;
            NvU64 num_pages = pages_per_chunk - pages_this_time;
            // Free the whole tail as a contiguous allocation
            nvUvmInterfacePmaFreePages(pmm->pma,
                                       &address,
                                       num_pages,
                                       page_size,
                                       UVM_PMA_CALLED_FROM_PMA_EVICTION | UVM_PMA_ALLOCATE_CONTIGUOUS);
        }
    }

    return NV_OK;

error:
    // On error, free all of the evicted pages back to PMA directly.
    if (num_pages_evicted_so_far > 0)
        nvUvmInterfacePmaFreePages(pmm->pma, pages, num_pages_evicted_so_far, page_size, UVM_PMA_CALLED_FROM_PMA_EVICTION);

    return status;
}

NV_STATUS uvm_pmm_gpu_pma_evict_pages_wrapper(void *void_pmm,
                                              NvU32 page_size,
                                              NvU64 *pages,
                                              NvU32 num_pages_to_evict,
                                              NvU64 phys_start,
                                              NvU64 phys_end)
{
    NV_STATUS status;

    // RM invokes the eviction callbacks with its API lock held, but not its GPU
    // lock.
    uvm_record_lock_rm_api();
    status = uvm_pmm_gpu_pma_evict_pages(void_pmm, page_size, pages, num_pages_to_evict, phys_start, phys_end);
    uvm_record_unlock_rm_api();
    return status;
}

// See the documentation of pmaEvictRangeCb_t in pma.h for details of the
// expected semantics.
static NV_STATUS uvm_pmm_gpu_pma_evict_range(void *void_pmm, NvU64 phys_begin, NvU64 phys_end)
{
    NV_STATUS status;
    uvm_pmm_gpu_t *pmm = (uvm_pmm_gpu_t *)void_pmm;
    NvU64 address = UVM_ALIGN_DOWN(phys_begin, UVM_CHUNK_SIZE_MAX);

    UVM_ASSERT_MSG(phys_begin <= phys_end, "range [0x%llx, 0x%llx]\n", phys_begin, phys_end);
    UVM_ASSERT_MSG(phys_end <= pmm->gpu->vidmem_max_allocatable_address,
                   "range [0x%llx, 0x%llx]\n",
                   phys_begin,
                   phys_end);

    // Make sure that all pending allocations, that could have started before
    // the eviction callback was called, are done. This is required to guarantee
    // that any address that, PMA thinks, is owned by UVM has been indeed recorded
    // in PMM's state. Taking the pma_lock in write mode will make sure all
    // readers (pending allocations and frees) are done, but will also
    // unnecessarily stop new allocations from starting until it's released.
    // TODO: Bug 1795559: SRCU would likely be better for this type of
    // synchronization, but that's GPL. Figure out whether we can do anything
    // better easily.
    uvm_down_write(&pmm->pma_lock);
    uvm_up_write(&pmm->pma_lock);

    for (; address <= phys_end; address += UVM_CHUNK_SIZE_MAX) {
        uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_address(pmm, address);
        uvm_gpu_chunk_t *chunk = &root_chunk->chunk;
        bool eviction_started = false;
        uvm_spin_loop_t spin;
        bool should_inject_error;

        uvm_spin_loop_init(&spin);

        // Wait until we can start eviction or the chunk is returned to PMA
        do {
            uvm_spin_lock(&pmm->list_lock);

            if (chunk->state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED) {
                UVM_ASSERT(chunk->type == UVM_PMM_GPU_MEMORY_TYPE_USER);

                if (chunk_is_evictable(pmm, chunk)) {
                    chunk_start_eviction(pmm, chunk);
                    eviction_started = true;
                }
            }

            uvm_spin_unlock(&pmm->list_lock);

            // TODO: Bug 1795559: Replace this with a wait queue.
            if (UVM_SPIN_LOOP(&spin) == NV_ERR_TIMEOUT_RETRY) {
                UVM_ERR_PRINT("Stuck waiting for root chunk 0x%llx to be unpinned, giving up\n", chunk->address);
                return NV_ERR_NO_MEMORY;
            }
        } while (!eviction_started && chunk->state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);

        // The eviction callback gets called with a physical range that might be
        // only partially allocated by UVM. Skip the chunks that UVM doesn't own.
        if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED)
            continue;

        uvm_mutex_lock(&pmm->lock);

        status = evict_root_chunk(pmm, root_chunk);
        should_inject_error = uvm_pmm_should_inject_pma_eviction_error(pmm);

        uvm_mutex_unlock(&pmm->lock);

        if (status != NV_OK)
            return status;

        free_root_chunk(pmm, root_chunk, FREE_ROOT_CHUNK_MODE_PMA_EVICTION);

        if (should_inject_error)
            return NV_ERR_NO_MEMORY;
    }

    // Make sure that all pending frees for chunks that the eviction above could
    // have observed as PMA owned are done. This is required to guarantee that
    // any address that, PMM thinks, is owned by PMA, has been actually freed
    // back to PMA. Taking the pma_lock in write mode will make sure all
    // readers (pending frees) are done, but will also unnecessarily stop new
    // allocations and frees from starting until it's released.
    uvm_down_write(&pmm->pma_lock);
    uvm_up_write(&pmm->pma_lock);

    return NV_OK;
}

NV_STATUS uvm_pmm_gpu_pma_evict_range_wrapper(void *void_pmm, NvU64 phys_begin, NvU64 phys_end)
{
    NV_STATUS status;

    // RM invokes the eviction callbacks with its API lock held, but not its GPU
    // lock.
    uvm_record_lock_rm_api();
    status = uvm_pmm_gpu_pma_evict_range(void_pmm, phys_begin, phys_end);
    uvm_record_unlock_rm_api();
    return status;
}

// Initializes the split cache for given GPU.
//
// It walks through all memory splits - in other words all ratios of neighboring
// pairs of sizes - and allocates kmem cache for them, unless they are already
// allocated.
//
// It also bumps the refcount if this GPU did not use such split yet.
NV_STATUS init_chunk_split_cache(uvm_pmm_gpu_t *pmm)
{
    NV_STATUS status;
    uvm_pmm_gpu_memory_type_t type;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    for (type = 0; type < UVM_PMM_GPU_MEMORY_TYPE_COUNT; type++) {
        uvm_chunk_size_t prev_size, cur_size;
        uvm_chunk_sizes_mask_t chunk_sizes = pmm->chunk_sizes[type];
        // Iterate over each pair of neighboring sizes. Note that same level
        // may be visited multiple times and it is handled internally by
        // init_chunk_split_cache_level
        prev_size = uvm_chunk_find_first_size(chunk_sizes);
        cur_size = uvm_chunk_find_next_size(chunk_sizes, prev_size);
        for_each_chunk_size_from(cur_size, chunk_sizes) {
            size_t subchunk_count = cur_size / prev_size;
            size_t level = ilog2(subchunk_count);
            status = init_chunk_split_cache_level(pmm, level);
            if (status != NV_OK)
                goto cleanup;
            prev_size = cur_size;
        }
    }

    status = init_chunk_split_cache_level(pmm, 0);
    if (status != NV_OK)
        goto cleanup;

    return NV_OK;
cleanup:
    deinit_chunk_split_cache(pmm);
    return status;
}

NV_STATUS init_chunk_split_cache_level(uvm_pmm_gpu_t *pmm, size_t level)
{
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    if (!test_bit(level, pmm->chunk_split_cache_initialized)) {
        if (!chunk_split_cache[level].cache) {
            size_t size;
            size_t align;
            if (level == 0) {
                strncpy(chunk_split_cache[level].name, "uvm_gpu_chunk_t", sizeof(chunk_split_cache[level].name) - 1);
                size = sizeof(uvm_gpu_chunk_t);
                align = __alignof__(uvm_gpu_chunk_t);
            } else {
                snprintf(chunk_split_cache[level].name,
                         sizeof(chunk_split_cache[level].name),
                         "uvm_gpu_chunk_%u", (unsigned)level);
                size = sizeof(uvm_pmm_gpu_chunk_suballoc_t) + (sizeof(uvm_gpu_chunk_t *) << level);
                align = __alignof__(uvm_pmm_gpu_chunk_suballoc_t);
            }
            chunk_split_cache[level].cache = NV_KMEM_CACHE_CREATE_FULL(chunk_split_cache[level].name, size, align, 0, NULL);
            if (!chunk_split_cache[level].cache)
                return NV_ERR_NO_MEMORY;
            chunk_split_cache[level].refcount = 1;
        } else {
            chunk_split_cache[level].refcount++;
            UVM_ASSERT_MSG(chunk_split_cache[level].refcount != 0, "Overflow of refcount\n");
        }
        __set_bit(level, pmm->chunk_split_cache_initialized);
    }
    return NV_OK;
}

void deinit_chunk_split_cache(uvm_pmm_gpu_t *pmm)
{
    unsigned long subchunk_count_log2;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    for_each_set_bit(subchunk_count_log2, pmm->chunk_split_cache_initialized, UVM_PMM_CHUNK_SPLIT_CACHE_SIZES) {
        UVM_ASSERT(chunk_split_cache[subchunk_count_log2].refcount > 0);
        if (--chunk_split_cache[subchunk_count_log2].refcount == 0) {
            kmem_cache_destroy(chunk_split_cache[subchunk_count_log2].cache);
            chunk_split_cache[subchunk_count_log2].cache = NULL;
        }
        __clear_bit(subchunk_count_log2, pmm->chunk_split_cache_initialized);
    }
}

typedef struct
{
    // Start/end of the physical region to be traversed (IN)
    NvU64 phys_start;
    NvU64 phys_end;

    // Pointer to the array of mappins where to store results (OUT)
    uvm_reverse_map_t *mappings;

    // Number of entries written to mappings (OUT)
    NvU32 num_mappings;
} get_chunk_mappings_data_t;

// Chunk traversal function used for phys-to-virt translation. These are the
// possible return values.
//
// - NV_ERR_OUT_OF_RANGE: no allocated physical chunks were found
// - NV_ERR_MORE_DATA_AVAILABLE: allocated physical chunks were found
// - NV_OK: allocated physical chunks may have been found. Check num_mappings
static NV_STATUS get_chunk_mappings_in_range(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, void *data)
{
    get_chunk_mappings_data_t *get_chunk_mappings_data = (get_chunk_mappings_data_t *)data;
    NvU64 chunk_end = chunk->address + uvm_gpu_chunk_get_size(chunk) - 1;

    uvm_assert_mutex_locked(&pmm->lock);

    // Kernel chunks do not have assigned VA blocks so we can just skip them
    if (chunk->type == UVM_PMM_GPU_MEMORY_TYPE_KERNEL)
        return NV_WARN_NOTHING_TO_DO;

    // This chunk is located before the requested physical range. Skip its
    // children and keep going
    if (chunk_end < get_chunk_mappings_data->phys_start)
        return NV_WARN_NOTHING_TO_DO;

    // We are beyond the search phys range. Stop traversing.
    if (chunk->address > get_chunk_mappings_data->phys_end) {
        if (get_chunk_mappings_data->num_mappings > 0)
            return NV_ERR_MORE_DATA_AVAILABLE;
        else
            return NV_ERR_OUT_OF_RANGE;
    }

    uvm_spin_lock(&pmm->list_lock);

    // Return results for allocated leaf chunks, only
    if (chunk->state == UVM_PMM_GPU_CHUNK_STATE_ALLOCATED) {
        uvm_reverse_map_t *reverse_map;

        UVM_ASSERT(chunk->va_block);
        uvm_va_block_retain(chunk->va_block);

        reverse_map = &get_chunk_mappings_data->mappings[get_chunk_mappings_data->num_mappings];

        reverse_map->va_block = chunk->va_block;
        reverse_map->region   = uvm_va_block_region(chunk->va_block_page_index,
                                                    chunk->va_block_page_index + uvm_gpu_chunk_get_size(chunk) / PAGE_SIZE);
        reverse_map->owner    = pmm->gpu->id;

        // If we land in the middle of a chunk, adjust the offset
        if (get_chunk_mappings_data->phys_start > chunk->address) {
            NvU64 offset = get_chunk_mappings_data->phys_start - chunk->address;

            reverse_map->region.first += offset / PAGE_SIZE;
        }

        // If the physical range doesn't cover the whole chunk, adjust num_pages
        if (get_chunk_mappings_data->phys_end < chunk_end)
            reverse_map->region.outer -= (chunk_end - get_chunk_mappings_data->phys_end) / PAGE_SIZE;

        ++get_chunk_mappings_data->num_mappings;
    }

    uvm_spin_unlock(&pmm->list_lock);

    return NV_OK;
}

NvU32 uvm_pmm_gpu_phys_to_virt(uvm_pmm_gpu_t *pmm, NvU64 phys_addr, NvU64 region_size, uvm_reverse_map_t *out_mappings)
{
    NvU64 chunk_base_addr = UVM_ALIGN_DOWN(phys_addr, UVM_CHUNK_SIZE_MAX);
    NvU64 size_in_chunk = min(UVM_CHUNK_SIZE_MAX - (phys_addr - chunk_base_addr), region_size);
    NvU32 num_mappings = 0;

    UVM_ASSERT(PAGE_ALIGNED(phys_addr));
    UVM_ASSERT(PAGE_ALIGNED(region_size));

    uvm_mutex_lock(&pmm->lock);

    // Traverse the whole requested region
    do {
        NV_STATUS status = NV_OK;
        uvm_gpu_root_chunk_t *root_chunk = root_chunk_from_address(pmm, phys_addr);
        uvm_gpu_chunk_t *chunk = &root_chunk->chunk;
        get_chunk_mappings_data_t get_chunk_mappings_data;

        get_chunk_mappings_data.phys_start   = phys_addr;
        get_chunk_mappings_data.phys_end     = phys_addr + size_in_chunk - 1;
        get_chunk_mappings_data.mappings     = out_mappings + num_mappings;
        get_chunk_mappings_data.num_mappings = 0;

        // Walk the chunks for the current root chunk
        status = chunk_walk_pre_order(pmm,
                                      chunk,
                                      get_chunk_mappings_in_range,
                                      &get_chunk_mappings_data);
        if (status == NV_ERR_OUT_OF_RANGE)
            break;

        if (get_chunk_mappings_data.num_mappings > 0) {
            UVM_ASSERT(status == NV_OK || status == NV_ERR_MORE_DATA_AVAILABLE);
            num_mappings += get_chunk_mappings_data.num_mappings;
        }
        else {
            UVM_ASSERT(status == NV_OK);
        }

        region_size -= size_in_chunk;
        phys_addr += size_in_chunk;
        size_in_chunk = min((NvU64)UVM_CHUNK_SIZE_MAX, region_size);
    } while (region_size > 0);

    uvm_mutex_unlock(&pmm->lock);

    return num_mappings;
}

NV_STATUS uvm8_test_evict_chunk(UVM_TEST_EVICT_CHUNK_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_va_block_t *block = NULL;
    uvm_gpu_root_chunk_t *root_chunk = NULL;
    uvm_pmm_gpu_t *pmm;

    params->chunk_was_evicted = NV_FALSE;
    params->evicted_physical_address = 0;
    params->chunk_size_backing_virtual = 0;

    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu || !uvm_gpu_supports_eviction(gpu)) {
        uvm_va_space_up_read(va_space);
        return NV_ERR_INVALID_DEVICE;
    }
    pmm = &gpu->pmm;

    // Retain the GPU before unlocking the VA space so that it sticks around.
    uvm_gpu_retain(gpu);

    // For virtual mode, look up and retain the block first so that eviction can
    // be started without the VA space lock held.
    if (params->eviction_mode == UvmTestEvictModeVirtual) {
        status = uvm_va_block_find_create(va_space, params->address, &block);
        if (status != NV_OK) {
            uvm_va_space_up_read(va_space);
            goto out;
        }

        // Retain the block before unlocking the VA space lock so that we can
        // safely access it later.
        uvm_va_block_retain(block);
    }

    // Unlock the VA space to emulate real eviction better where a VA space lock
    // may not be held or may be held for a different VA space.
    uvm_va_space_up_read(va_space);

    if (params->eviction_mode == UvmTestEvictModeVirtual) {
        UVM_ASSERT(block);

        uvm_mutex_lock(&block->lock);

        // As the VA space lock is not held we need to make sure the block
        // is still alive.
        if (block->va_range != NULL) {
            // The block might have been split in the meantime and may no longer
            // cover the address as a result.
            if (params->address >= block->start && params->address <= block->end) {
                uvm_gpu_chunk_t *chunk = uvm_va_block_lookup_gpu_chunk(block, gpu, params->address);

                uvm_spin_lock(&pmm->list_lock);
                if (chunk && chunk_is_evictable(pmm, chunk)) {
                    chunk_start_eviction(pmm, chunk);
                    root_chunk = root_chunk_from_chunk(pmm, chunk);
                    params->chunk_size_backing_virtual = uvm_gpu_chunk_get_size(chunk);
                }
                uvm_spin_unlock(&pmm->list_lock);
            }
        }
        else {
            // Consider it an error to free the block before the eviction ioctl
            // is done.
            status = NV_ERR_INVALID_ADDRESS;
        }

        uvm_mutex_unlock(&block->lock);
        uvm_va_block_release(block);

        if (status != NV_OK)
            goto out;
    }
    else if (params->eviction_mode == UvmTestEvictModePhysical) {
        uvm_gpu_chunk_t *chunk;
        size_t index = params->address / UVM_CHUNK_SIZE_MAX;

        if (index >= pmm->root_chunks.count) {
            status = NV_ERR_INVALID_ADDRESS;
            goto out;
        }

        root_chunk = &pmm->root_chunks.array[index];
        chunk = &root_chunk->chunk;

        uvm_spin_lock(&pmm->list_lock);

        if (chunk_is_evictable(pmm, chunk))
            chunk_start_eviction(pmm, chunk);
        else
            chunk = NULL;

        uvm_spin_unlock(&pmm->list_lock);

        if (!chunk)
            root_chunk = NULL;
    }
    else if (params->eviction_mode == UvmTestEvictModeDefault) {
        root_chunk = pick_root_chunk_to_evict(pmm);
    }
    else {
        UVM_DBG_PRINT("Invalid eviction mode: 0x%x\n", params->eviction_mode);
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    if (!root_chunk) {
        // Not finding a chunk to evict is not considered an error, the caller
        // can inspect the targeted_chunk_size to see whether anything was evicted.
        goto out;
    }

    uvm_mutex_lock(&pmm->lock);
    status = evict_root_chunk(pmm, root_chunk);
    uvm_mutex_unlock(&pmm->lock);

    if (status != NV_OK)
        goto out;

    params->chunk_was_evicted = NV_TRUE;
    params->evicted_physical_address = root_chunk->chunk.address;
    free_chunk(pmm, &root_chunk->chunk);

out:
    uvm_gpu_release(gpu);
    return status;
}

static NV_STATUS test_check_pma_allocated_chunks(uvm_pmm_gpu_t *pmm,
                                                 UVM_TEST_PMA_ALLOC_FREE_PARAMS *params,
                                                 NvU64 *pages)
{
    NV_STATUS status = NV_OK;
    NvU32 i;

    for (i = 0; i < params->num_pages; ++i) {
        uvm_gpu_root_chunk_t *root_chunk;
        NvU64 address;
        if (params->contiguous)
            address = pages[0] + params->page_size * i;
        else
            address = pages[i];

        root_chunk = root_chunk_from_address(pmm, address);

        if (!IS_ALIGNED(address, params->page_size)) {
            UVM_TEST_PRINT("Returned unaligned address 0x%llx page size %u\n", address, params->page_size);
            status = NV_ERR_INVALID_STATE;
        }

        // The chunk should still be in the PMA owned state
        uvm_spin_lock(&pmm->list_lock);
        if (root_chunk->chunk.state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED) {
            UVM_TEST_PRINT("Root chunk 0x%llx invalid state: %s, allocated [0x%llx, 0x%llx)\n",
                           root_chunk->chunk.address,
                           uvm_pmm_gpu_chunk_state_string(root_chunk->chunk.state),
                           address, address + params->page_size);
            status = NV_ERR_INVALID_STATE;
        }
        uvm_spin_unlock(&pmm->list_lock);
    }
    return status;
}

NV_STATUS uvm8_test_pma_alloc_free(UVM_TEST_PMA_ALLOC_FREE_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    uvm_pmm_gpu_t *pmm;
    NvU64 page;
    NvU64 *pages = NULL;

    UvmPmaAllocationOptions options = {0};

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    pmm = &gpu->pmm;

    options.flags = UVM_PMA_ALLOCATE_PINNED;
    if (params->contiguous) {
        options.flags |= UVM_PMA_ALLOCATE_CONTIGUOUS;
        pages = &page;
    }
    else {
        pages = uvm_kvmalloc(sizeof(*pages) * params->num_pages);
        if (!pages) {
            status = NV_ERR_NO_MEMORY;
            goto out;
        }
    }
    if (params->phys_begin != 0 || params->phys_end != 0) {
        options.physBegin = params->phys_begin;
        options.physEnd = params->phys_end;
        options.flags |= UVM_PMA_ALLOCATE_SPECIFY_ADDRESS_RANGE;
    }

    status = nvUvmInterfacePmaAllocPages(pmm->pma, params->num_pages, params->page_size, &options, pages);
    if (status != NV_OK)
        goto out;

    status = test_check_pma_allocated_chunks(pmm, params, pages);
    if (status != NV_OK) {
        UVM_TEST_PRINT("Failed before the nap\n");
        goto free;
    }

    if (params->nap_us_before_free)
        usleep_range(params->nap_us_before_free, params->nap_us_before_free + 10);

    status = test_check_pma_allocated_chunks(pmm, params, pages);
    if (status != NV_OK)
        UVM_TEST_PRINT("Failed after the nap\n");

free:
    nvUvmInterfacePmaFreePages(gpu->pmm.pma, pages, params->num_pages, params->page_size, options.flags);

out:
    if (!params->contiguous)
        uvm_kvfree(pages);

    uvm_gpu_release(gpu);
    return status;
}

NV_STATUS uvm8_test_pmm_alloc_free_root(UVM_TEST_PMM_ALLOC_FREE_ROOT_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;
    uvm_pmm_gpu_t *pmm;
    uvm_gpu_chunk_t *chunk;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    pmm = &gpu->pmm;

    status = uvm_pmm_gpu_alloc_user(pmm, 1, UVM_CHUNK_SIZE_MAX, UVM_PMM_ALLOC_FLAGS_EVICT, &chunk, &tracker);

    if (status != NV_OK)
        goto out;

    if (params->nap_us_before_free)
        usleep_range(params->nap_us_before_free, params->nap_us_before_free + 10);

    uvm_pmm_gpu_free(pmm, chunk, NULL);
    uvm_tracker_deinit(&tracker);

out:
    uvm_gpu_release(gpu);
    return status;
}

NV_STATUS uvm8_test_pmm_inject_pma_evict_error(UVM_TEST_PMM_INJECT_PMA_EVICT_ERROR_PARAMS *params, struct file *filp)
{
    NV_STATUS status;
    uvm_gpu_t *gpu;
    uvm_pmm_gpu_t *pmm;

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    pmm = &gpu->pmm;

    uvm_mutex_lock(&pmm->lock);
    pmm->inject_pma_evict_error_after_num_chunks = params->error_after_num_chunks;
    uvm_mutex_unlock(&pmm->lock);

    uvm_gpu_release(gpu);
    return NV_OK;
}
