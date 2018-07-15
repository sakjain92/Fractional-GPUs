/*******************************************************************************
    Copyright (c) 2015 NVidia Corporation

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

#ifndef __UVM8_TEST_IOCTL_H__
#define __UVM8_TEST_IOCTL_H__

#ifndef __KERNEL__
#include "g_nvconfig.h"
#endif
#include "uvmtypes.h"
#include "uvm_ioctl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Offset the test ioctl to leave space for the api ones
#define UVM8_TEST_IOCTL_BASE(i)                         UVM_IOCTL_BASE(200 + i)

#define UVM_TEST_GET_GPU_REF_COUNT                      UVM8_TEST_IOCTL_BASE(0)
typedef struct
{
    // In params
    NvProcessorUuid gpu_uuid;
    // Out params
    NvU64           ref_count NV_ALIGN_BYTES(8);
    NV_STATUS       rmStatus;
} UVM_TEST_GET_GPU_REF_COUNT_PARAMS;

#define UVM_TEST_RNG_SANITY                             UVM8_TEST_IOCTL_BASE(1)
typedef struct
{
    NV_STATUS rmStatus;
} UVM_TEST_RNG_SANITY_PARAMS;

#define UVM_TEST_RANGE_TREE_DIRECTED                    UVM8_TEST_IOCTL_BASE(2)
typedef struct
{
    NV_STATUS rmStatus;
} UVM_TEST_RANGE_TREE_DIRECTED_PARAMS;

#define UVM_TEST_RANGE_TREE_RANDOM                      UVM8_TEST_IOCTL_BASE(3)
typedef struct
{
    NvU32     seed;                                 // In
    NvU64     main_iterations    NV_ALIGN_BYTES(8); // In
    NvU32     verbose;                              // In

    // Probability (0-100)
    //
    // When the test starts up, it adds and splits ranges with high_probability.
    // Eventually when adds and splits fail too often, they'll invert their
    // probability to 100 - high_probability. They'll switch back when the tree
    // becomes too empty.
    //
    // This can be < 50, but the test will not be very interesting.
    NvU32     high_probability;                     // In

    // Probability (0-100)
    //
    // Every main iteration a group of operations is selected with this
    // probability. The group consists of either "add/remove" or "split/merge."
    // This is the chance that the "add/remove" group is selected each
    // iteration.
    NvU32     add_remove_shrink_group_probability;

    // Probability (0-100)
    //
    // Probability of picking the shrink operation instead of add/remove if the
    // add/remove/shrink group of operations is selected.
    NvU32     shrink_probability;

    // The number of collision verification checks to make each main iteration
    NvU32     collision_checks;                     // In

    // The number of tree iterator verification checks to make each main
    // iteration.
    NvU32     iterator_checks;                      // In

    // Highest range value to use
    NvU64     max_end            NV_ALIGN_BYTES(8); // In

    // Maximum number of range nodes to put in the tree
    NvU64     max_ranges         NV_ALIGN_BYTES(8); // In

    // Maximum number of range nodes to add or remove at one time
    NvU64     max_batch_count    NV_ALIGN_BYTES(8); // In

    // add, split, and merge operations all operate on randomly-selected ranges
    // or nodes. It's possible, sometimes even likely, that the operation cannot
    // be performed on the selected range or node.
    //
    // For example, when a range node is added its range is selected at random
    // without regard to range nodes already in the tree. If a collision occurs
    // when the test attempts to add that node to the tree, a new, smaller
    // random range is selected and the attempt is made again.
    //
    // max_attempts is the maximum number of times to keep picking new ranges or
    // nodes before giving up on the operation.
    NvU32     max_attempts;                          // In

    struct
    {
        NvU64 total_adds         NV_ALIGN_BYTES(8);
        NvU64 failed_adds        NV_ALIGN_BYTES(8);
        NvU64 max_attempts_add   NV_ALIGN_BYTES(8);
        NvU64 total_removes      NV_ALIGN_BYTES(8);
        NvU64 total_splits       NV_ALIGN_BYTES(8);
        NvU64 failed_splits      NV_ALIGN_BYTES(8);
        NvU64 max_attempts_split NV_ALIGN_BYTES(8);
        NvU64 total_merges       NV_ALIGN_BYTES(8);
        NvU64 failed_merges      NV_ALIGN_BYTES(8);
        NvU64 max_attempts_merge NV_ALIGN_BYTES(8);
        NvU64 total_shrinks      NV_ALIGN_BYTES(8);
        NvU64 failed_shrinks     NV_ALIGN_BYTES(8);
    } stats;                                        // Out

    NV_STATUS rmStatus;                             // Out
} UVM_TEST_RANGE_TREE_RANDOM_PARAMS;

// Keep this in sync with uvm_va_range_type_t in uvm8_va_range.h
typedef enum
{
    UVM_TEST_VA_RANGE_TYPE_INVALID = 0,
    UVM_TEST_VA_RANGE_TYPE_MANAGED,
    UVM_TEST_VA_RANGE_TYPE_EXTERNAL,
    UVM_TEST_VA_RANGE_TYPE_CHANNEL,
    UVM_TEST_VA_RANGE_TYPE_SKED_REFLECTED,
    UVM_TEST_VA_RANGE_TYPE_SEMAPHORE_POOL,
    UVM_TEST_VA_RANGE_TYPE_MAX
} UVM_TEST_VA_RANGE_TYPE;

// Keep this in sync with uvm_read_duplication_t in uvm8_va_range.h
typedef enum
{
    UVM_TEST_READ_DUPLICATION_UNSET = 0,
    UVM_TEST_READ_DUPLICATION_ENABLED,
    UVM_TEST_READ_DUPLICATION_DISABLED,
    UVM_TEST_READ_DUPLICATION_MAX
} UVM_TEST_READ_DUPLICATION_POLICY;

typedef struct
{
    // Note: if this is a zombie or not owned by the calling process, the vma info
    // will not be filled out and is invalid.
    NvU64  vma_start NV_ALIGN_BYTES(8); // Out
    NvU64  vma_end   NV_ALIGN_BYTES(8); // Out, inclusive
    NvBool is_zombie;                   // Out
    // Note: if this is a zombie, this field is meaningless.
    NvBool owned_by_calling_process;    // Out
} UVM_TEST_VA_RANGE_INFO_MANAGED;

#define UVM_TEST_VA_RANGE_INFO                          UVM8_TEST_IOCTL_BASE(4)
typedef struct
{
    NvU64                           lookup_address                   NV_ALIGN_BYTES(8); // In

    NvU64                           va_range_start                   NV_ALIGN_BYTES(8); // Out
    NvU64                           va_range_end                     NV_ALIGN_BYTES(8); // Out, inclusive
    NvU32                           read_duplication;                                   // Out (UVM_TEST_READ_DUPLICATION_POLICY)
    NvProcessorUuid                 preferred_location;                                 // Out
    NvProcessorUuid                 accessed_by[UVM_MAX_PROCESSORS];                    // Out
    NvU32                           accessed_by_count;                                  // Out
    NvU32                           type;                                               // Out (UVM_TEST_VA_RANGE_TYPE)
    union
    {
        UVM_TEST_VA_RANGE_INFO_MANAGED managed                       NV_ALIGN_BYTES(8); // Out
        // More here eventually
    };

    // NV_ERR_INVALID_ADDRESS   lookup_address doesn't match a UVM range
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_VA_RANGE_INFO_PARAMS;

#define UVM_TEST_RM_MEM_SANITY                          UVM8_TEST_IOCTL_BASE(5)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_RM_MEM_SANITY_PARAMS;

#define UVM_TEST_GPU_SEMAPHORE_SANITY                   UVM8_TEST_IOCTL_BASE(6)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_GPU_SEMAPHORE_SANITY_PARAMS;

#define UVM_TEST_PEER_REF_COUNT                         UVM8_TEST_IOCTL_BASE(7)
typedef struct
{
    // In params
    NvProcessorUuid gpu_uuid_1;
    NvProcessorUuid gpu_uuid_2;

    // Out params
    NV_STATUS       rmStatus;
    NvU64           ref_count   NV_ALIGN_BYTES(8);
} UVM_TEST_PEER_REF_COUNT_PARAMS;

// Force an existing UVM range to split. split_address will be the new end of
// the existing range. A new range will be created covering
// [split_address+1, original end].
//
// Error returns:
// NV_ERR_INVALID_ADDRESS
//  - split_address+1 isn't page-aligned
//  - split_address doesn't match a splittable UVM range
//  - The range cannot be split at split_address because split_address is
//    already the end of the range.
#define UVM_TEST_VA_RANGE_SPLIT                         UVM8_TEST_IOCTL_BASE(8)
typedef struct
{
    NvU64     split_address NV_ALIGN_BYTES(8); // In
    NV_STATUS rmStatus;                        // Out
} UVM_TEST_VA_RANGE_SPLIT_PARAMS;

// Forces the next range split on the range covering lookup_address to fail with
// an out-of-memory error. Only the next split will fail. Subsequent ones will
// succeed. The split can come from any source, such as vma splitting or
// UVM_TEST_VA_RANGE_SPLIT.
//
// Error returns:
// NV_ERR_INVALID_ADDRESS
//  - lookup_address doesn't match a UVM range
#define UVM_TEST_VA_RANGE_INJECT_SPLIT_ERROR            UVM8_TEST_IOCTL_BASE(9)
typedef struct
{
    NvU64     lookup_address NV_ALIGN_BYTES(8); // In
    NV_STATUS rmStatus;                         // Out
} UVM_TEST_VA_RANGE_INJECT_SPLIT_ERROR_PARAMS;

#define UVM_TEST_PAGE_TREE                              UVM8_TEST_IOCTL_BASE(10)
typedef struct
{
    NV_STATUS rmStatus;                     // Out
} UVM_TEST_PAGE_TREE_PARAMS;

// Given a VA and a target processor, forcibly set that processor's mapping to
// the VA to the given permissions. This may require changing other processors'
// mappings. For example, setting an atomic mapping for a given GPU might make
// other GPUs' mappings read-only.
//
// If the mapping changes from invalid to anything else, this call always
// attempts to create direct mappings from the given processor to the current
// physical memory backing the target address. If a direct mapping cannot be
// created, or no physical memory currently backs the VA,
// NV_ERR_INVALID_OPERATION is returned.
//
// uuid is allowed to be NV_PROCESSOR_UUID_CPU_DEFAULT.
//
// Error returns:
// NV_ERR_INVALID_DEVICE
//  - uuid is an unknown value
//  - uuid is a GPU that hasn't been registered with this process
//
// NV_ERR_INVALID_ADDRESS
// - VA is unknown to the kernel
// - VA isn't aligned to the system page size
//
// NV_ERR_INVALID_STATE
// - A mapping for va can't be accessed because it belongs to another process
//
// NV_ERR_INVALID_ARGUMENT
// - mapping is not a valid enum value
//
// NV_ERR_INVALID_ACCESS_TYPE
// - The mapping permissions aren't logically allowed. For example,
//   UVM_TEST_PTE_MAPPING_READ_WRITE can't be set on a read-only mapping.
//
// NV_ERR_INVALID_OPERATION
// - mapping is not UVM_TEST_PTE_MAPPING_INVALID, and a direct mapping from the
//   given processor to the physical memory currently backing VA cannot be
//   created.
#define UVM_TEST_CHANGE_PTE_MAPPING                     UVM8_TEST_IOCTL_BASE(11)

typedef enum
{
    UVM_TEST_PTE_MAPPING_INVALID = 0,
    UVM_TEST_PTE_MAPPING_READ_ONLY,
    UVM_TEST_PTE_MAPPING_READ_WRITE,
    UVM_TEST_PTE_MAPPING_READ_WRITE_ATOMIC,
    UVM_TEST_PTE_MAPPING_MAX
} UVM_TEST_PTE_MAPPING;

typedef struct
{
    NvProcessorUuid      uuid      NV_ALIGN_BYTES(8); // In
    NvU64                va        NV_ALIGN_BYTES(8); // In
    NvU32                mapping;                     // In (UVM_TEST_PTE_MAPPING)
    NV_STATUS            rmStatus;                    // Out
} UVM_TEST_CHANGE_PTE_MAPPING_PARAMS;

#define UVM_TEST_TRACKER_SANITY                         UVM8_TEST_IOCTL_BASE(12)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_TRACKER_SANITY_PARAMS;

#define UVM_TEST_PUSH_SANITY                            UVM8_TEST_IOCTL_BASE(13)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_PUSH_SANITY_PARAMS;

#define UVM_TEST_CHANNEL_SANITY                         UVM8_TEST_IOCTL_BASE(14)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_CHANNEL_SANITY_PARAMS;

typedef enum
{
    UVM_TEST_CHANNEL_STRESS_MODE_NOOP_PUSH = 0,
    UVM_TEST_CHANNEL_STRESS_MODE_UPDATE_CHANNELS,
    UVM_TEST_CHANNEL_STRESS_MODE_STREAM,
} UVM_TEST_CHANNEL_STRESS_MODE;

#define UVM_TEST_CHANNEL_STRESS                         UVM8_TEST_IOCTL_BASE(15)
typedef struct
{
    NvU32     mode;                   // In

    // Number of iterations:
    //   mode == NOOP_PUSH: number of noop pushes
    //   mode == UPDATE_CHANNELS: number of updates
    //   mode == STREAM: number of iterations per stream
    NvU32     iterations;

    NvU32     num_streams;            // In, used only for mode == UVM_TEST_CHANNEL_STRESS_MODE_STREAM
    NvU32     seed;                   // In
    NvU32     verbose;                // In
    NV_STATUS rmStatus;               // Out
} UVM_TEST_CHANNEL_STRESS_PARAMS;

#define UVM_TEST_CE_SANITY                              UVM8_TEST_IOCTL_BASE(16)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_CE_SANITY_PARAMS;

#define UVM_TEST_VA_BLOCK_INFO                          UVM8_TEST_IOCTL_BASE(17)

// See UVM_VA_BLOCK_SIZE in uvm8_va_block.h for an explanation of this number
#define UVM_TEST_VA_BLOCK_SIZE (2*1024*1024)

typedef struct
{
    NvU64     lookup_address    NV_ALIGN_BYTES(8); // In


    NvU64     va_block_start    NV_ALIGN_BYTES(8); // Out
    NvU64     va_block_end      NV_ALIGN_BYTES(8); // Out, inclusive

    // NV_ERR_INVALID_ADDRESS   lookup_address doesn't match a UVM range
    //
    // NV_ERR_INVALID_STATE     lookup_address matched a UVM range on this file
    //                          but the range can't be accessed because it
    //                          belongs to another process
    // NV_ERR_OBJECT_NOT_FOUND  lookup_address matched a UVM range on this file
    //                          but the corresponding block has not yet been
    //                          populated
    NV_STATUS rmStatus;                            // Out
} UVM_TEST_VA_BLOCK_INFO_PARAMS;

#define UVM_TEST_LOCK_SANITY                            UVM8_TEST_IOCTL_BASE(18)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_LOCK_SANITY_PARAMS;

#define UVM_TEST_PERF_UTILS_SANITY                      UVM8_TEST_IOCTL_BASE(19)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_PERF_UTILS_SANITY_PARAMS;

#define UVM_TEST_KVMALLOC                               UVM8_TEST_IOCTL_BASE(20)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_KVMALLOC_PARAMS;

#define UVM_TEST_PMM_QUERY                              UVM8_TEST_IOCTL_BASE(21)
typedef enum
{
    // Get the value of valid user allocations as key
    UVM_TEST_CHUNK_SIZE_GET_USER_SIZE
} uvm_test_pmm_query_key_t;

typedef struct
{
    // In params
    NvProcessorUuid gpu_uuid;
    NvU64 key;
    // Out params
    NvU64 value;
    NV_STATUS rmStatus;
} UVM_TEST_PMM_QUERY_PARAMS;

#define UVM_TEST_PMM_CHECK_LEAK                         UVM8_TEST_IOCTL_BASE(22)

typedef struct
{
    NvProcessorUuid gpu_uuid; // In
    NvU64 chunk_size;         // In
    NvS64 alloc_limit;        // In. Number of chunks to allocate. -1 means unlimited
    NvU64 allocated;          // Out. Number of chunks actually allocated
    NV_STATUS rmStatus;       // Out
} UVM_TEST_PMM_CHECK_LEAK_PARAMS;

#define UVM_TEST_PERF_EVENTS_SANITY                     UVM8_TEST_IOCTL_BASE(23)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_PERF_EVENTS_SANITY_PARAMS;

#define UVM_TEST_PERF_MODULE_SANITY                     UVM8_TEST_IOCTL_BASE(24)
typedef struct
{
    // In params
    NvU64 range_address              NV_ALIGN_BYTES(8);
    NvU32 range_size;
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_PERF_MODULE_SANITY_PARAMS;

#define UVM_TEST_RANGE_ALLOCATOR_SANITY                 UVM8_TEST_IOCTL_BASE(25)
typedef struct
{
    // In params
    NvU32 verbose;
    NvU32 seed;
    NvU32 iters;

    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_RANGE_ALLOCATOR_SANITY_PARAMS;

#define UVM_TEST_GET_RM_PTES                            UVM8_TEST_IOCTL_BASE(26)
typedef enum
{
    UVM_TEST_GET_RM_PTES_SINGLE_GPU = 0,
    UVM_TEST_GET_RM_PTES_MULTI_GPU_SUPPORTED,
    UVM_TEST_GET_RM_PTES_MULTI_GPU_SLI_SUPPORTED,
    UVM_TEST_GET_RM_PTES_MULTI_GPU_NOT_SUPPORTED,
    UVM_TEST_GET_RM_PTES_MAX
} UVM_TEST_PTE_RM_PTES_TEST_MODE;

typedef struct
{
    // In
    NvS32 rmCtrlFd;             // For future use. (security check)
    NvHandle hClient;
    NvHandle hMemory;
    NvU32 test_mode;            // (UVM_TEST_PTE_RM_PTES_TEST_MODE)
    NvU64 size                  NV_ALIGN_BYTES(8);
    NvProcessorUuid gpu_uuid;

    // Out
    NV_STATUS rmStatus;
} UVM_TEST_GET_RM_PTES_PARAMS;

#define UVM_TEST_FAULT_BUFFER_FLUSH                     UVM8_TEST_IOCTL_BASE(27)
typedef struct
{
    NvU64 iterations;           // In
    NV_STATUS rmStatus;         // Out
} UVM_TEST_FAULT_BUFFER_FLUSH_PARAMS;

#define UVM_TEST_INJECT_TOOLS_EVENT                     UVM8_TEST_IOCTL_BASE(28)
typedef struct
{
    // In params
    UvmEventEntry entry; // contains only NvUxx types
    NvU32 count;

    // Out param
    NV_STATUS rmStatus;
} UVM_TEST_INJECT_TOOLS_EVENT_PARAMS;

#define UVM_TEST_INCREMENT_TOOLS_COUNTER                UVM8_TEST_IOCTL_BASE(29)
typedef struct
{
    // In params
    NvU64 amount                     NV_ALIGN_BYTES(8); // amount to increment
    NvU32 counter;                                      // name of counter
    NvProcessorUuid processor;
    NvU32 count;                                        // number of times to increment

    // Out param
    NV_STATUS rmStatus;
} UVM_TEST_INCREMENT_TOOLS_COUNTER_PARAMS;

#define UVM_TEST_MEM_SANITY                             UVM8_TEST_IOCTL_BASE(30)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_MEM_SANITY_PARAMS;

#define UVM_TEST_MMU_SANITY                             UVM8_TEST_IOCTL_BASE(31)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_MMU_SANITY_PARAMS;

#define UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE           UVM8_TEST_IOCTL_BASE(32)
typedef struct
{
    // Out params
    NV_STATUS rmStatus;
} UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE_PARAMS;

// Inject an error into the va block covering the lookup_address
//
// If page_table_allocation_retry_force_count is non-0 then the next count
// page table allocations under the va block will be forced to do
// allocation-retry.
//
// If user_pages_allocation_retry_force_count is non-0 then the next count user
// memory allocations under the va block will be forced to do allocation-retry.
//
// If eviction_failure is NV_TRUE, the next eviction attempt from the VA block
// will fail with NV_ERR_NO_MEMORY.
//
// Error returns:
// NV_ERR_INVALID_ADDRESS
//  - lookup_address doesn't match a UVM range
#define UVM_TEST_VA_BLOCK_INJECT_ERROR                  UVM8_TEST_IOCTL_BASE(33)
typedef struct
{
    NvU64     lookup_address NV_ALIGN_BYTES(8);         // In
    NvU32     page_table_allocation_retry_force_count;  // In
    NvU32     user_pages_allocation_retry_force_count;  // In
    NvBool    eviction_error;                           // In
    NV_STATUS rmStatus;                                 // Out
} UVM_TEST_VA_BLOCK_INJECT_ERROR_PARAMS;

#define UVM_TEST_PEER_IDENTITY_MAPPINGS                 UVM8_TEST_IOCTL_BASE(34)
typedef struct
{
    // In params
    NvProcessorUuid gpuA;
    NvProcessorUuid gpuB;
    // Out param
    NV_STATUS rmStatus;
} UVM_TEST_PEER_IDENTITY_MAPPINGS_PARAMS;

#define UVM_TEST_VA_RESIDENCY_INFO                      UVM8_TEST_IOCTL_BASE(35)
typedef struct
{
    NvU64                           lookup_address                   NV_ALIGN_BYTES(8); // In

    // Array of processors which have a resident copy of the page containing
    // lookup_address.
    NvProcessorUuid                 resident_on[UVM_MAX_PROCESSORS];                    // Out
    NvU32                           resident_on_count;                                  // Out

    // The size of the physical allocation backing lookup_address. Only the
    // system-page-sized portion of this allocation which contains
    // lookup_address is guaranteed to be resident on the corresponding
    // processor.
    NvU32                           resident_physical_size[UVM_MAX_PROCESSORS];         // Out

    // The physical address of the physical allocation backing lookup_address.
    NvU64                           resident_physical_address[UVM_MAX_PROCESSORS] NV_ALIGN_BYTES(8); // Out

    // Array of processors which have a virtual mapping covering lookup_address.
    NvProcessorUuid                 mapped_on[UVM_MAX_PROCESSORS];                      // Out
    NvU32                           mapping_type[UVM_MAX_PROCESSORS];                   // Out
    NvU32                           mapped_on_count;                                    // Out

    // The size of the virtual mapping covering lookup_address on each
    // mapped_on processor.
    NvU32                           page_size[UVM_MAX_PROCESSORS];                      // Out

    // Array of processors which have physical memory populated that would back
    // lookup_address if it was resident.
    NvProcessorUuid                 populated_on[UVM_MAX_PROCESSORS];                   // Out
    NvU32                           populated_on_count;                                 // Out

    NV_STATUS rmStatus;                                                                 // Out
} UVM_TEST_VA_RESIDENCY_INFO_PARAMS;

#define UVM_TEST_PMM_ASYNC_ALLOC                        UVM8_TEST_IOCTL_BASE(36)
typedef struct
{
    NvProcessorUuid gpu_uuid;                           // In
    NvU32 num_chunks;                                   // In
    NvU32 num_work_iterations;                          // In
    NV_STATUS rmStatus;                                 // Out
} UVM_TEST_PMM_ASYNC_ALLOC_PARAMS;

typedef enum
{
    UVM_TEST_PREFETCH_FILTERING_MODE_FILTER_ALL,  // Disable all prefetch faults
    UVM_TEST_PREFETCH_FILTERING_MODE_FILTER_NONE, // Enable all prefetch faults
} UvmTestPrefetchFilteringMode;

#define UVM_TEST_SET_PREFETCH_FILTERING                 UVM8_TEST_IOCTL_BASE(37)
typedef struct
{
    NvProcessorUuid gpu_uuid;                           // In
    NvU32           filtering_mode;                     // In (UvmTestPrefetchFilteringMode)
    NV_STATUS       rmStatus;                           // Out
} UVM_TEST_SET_PREFETCH_FILTERING_PARAMS;

typedef enum
{
    UvmTestPmmSanityModeFull  = 1,
    UvmTestPmmSanityModeBasic = 2,
} UvmTestPmmSanityMode;

#define UVM_TEST_PMM_SANITY                             UVM8_TEST_IOCTL_BASE(40)
typedef struct
{
    // Test mode of type UvmTestPmmSanityMode
    NvU32         mode; // In
    NV_STATUS rmStatus; // Out
} UVM_TEST_PMM_SANITY_PARAMS;

typedef enum
{
    UvmInvalidateTlbMemBarNone  = 1,
    UvmInvalidateTlbMemBarSys   = 2,
    UvmInvalidateTlbMemBarLocal = 3,
} UvmInvalidateTlbMembarType;

typedef enum
{
    UvmInvalidatePageTableLevelAll = 1,
    UvmInvalidatePageTableLevelPte = 2,
    UvmInvalidatePageTableLevelPde0 = 3,
    UvmInvalidatePageTableLevelPde1 = 4,
    UvmInvalidatePageTableLevelPde2 = 5,
    UvmInvalidatePageTableLevelPde3 = 6
} UvmInvalidatePageTableLevel;

typedef enum
{
    UvmTargetPdbModeAll = 1,
    UvmTargetPdbModeOne = 2
} UvmTargetPdbMode;

typedef enum
{
    UvmTargetVaModeAll      = 1,
    UvmTargetVaModeTargeted = 2,
} UvmTargetVaMode;

#define UVM_TEST_INVALIDATE_TLB                         UVM8_TEST_IOCTL_BASE(41)
typedef struct
{
    // In params

    NvProcessorUuid  gpu_uuid;
    NvU64            va NV_ALIGN_BYTES(8);
    NvU32            target_va_mode;           // UvmTargetVaMode
    NvU32            target_pdb_mode;          // UvmTargetPdbMode
    NvU32            page_table_level;         // UvmInvalidatePageTableLevel
    NvU32            membar;                   // UvmInvalidateTlbMembarType
    NvBool           disable_gpc_invalidate;

    // Out params
    NV_STATUS        rmStatus;
} UVM_TEST_INVALIDATE_TLB_PARAMS;

#define UVM_TEST_VA_BLOCK                               UVM8_TEST_IOCTL_BASE(42)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_VA_BLOCK_PARAMS;

typedef enum
{
    // Default policy based eviction
    //
    // Evicts a chunk that the default eviction path would pick.
    UvmTestEvictModeDefault = 1,

    // Virtual address based eviction
    //
    // Evicts the root chunk that the chunk backing the provided virtual address
    // belongs to.
    UvmTestEvictModeVirtual,

    // Physical address based eviction
    //
    // Evicts the root chunk covering the provided physical address.
    UvmTestEvictModePhysical,
} UvmTestEvictMode;

// Evict a chunk chosen according to one the test eviction modes specified
// above. Eviction may not always be possible, but as long as the arguments are
// valid NV_OK will be returned. To check whether eviction happened, the
// chunk_was_evicted flag needs to be inspected.
#define UVM_TEST_EVICT_CHUNK                            UVM8_TEST_IOCTL_BASE(43)
typedef struct
{
    // The GPU to evict from, has to be registered in the VA space.
    NvProcessorUuid                 gpu_uuid;                                           // In

    // UvmTestEvictMode
    NvU32                           eviction_mode;                                      // In

    // Virtual or physical address if evictionMode is UvmTestEvictModeVirtual or
    // UvmTestEvictModePhysical.
    NvU64                           address                          NV_ALIGN_BYTES(8); // In

    // Flag indicating whether the eviction was performed.
    NvBool                          chunk_was_evicted;                                  // Out

    // Physical address of the evicted root chunk. Notably 0 is a valid physical address.
    NvU64                           evicted_physical_address         NV_ALIGN_BYTES(8); // Out

    // For the virtual eviction mode, returns the size of the chunk that was
    // backing the virtual address before being evicted. 0 otherwise.
    NvU64                           chunk_size_backing_virtual       NV_ALIGN_BYTES(8); // Out

    NV_STATUS rmStatus;                                                                 // Out
} UVM_TEST_EVICT_CHUNK_PARAMS;

typedef enum
{
    // Flush deferred accessed by mappings
    UvmTestDeferredWorkTypeAcessedByMappings = 1,
} UvmTestDeferredWorkType;

#define UVM_TEST_FLUSH_DEFERRED_WORK                    UVM8_TEST_IOCTL_BASE(44)
typedef struct
{
    // UvmTestDeferredWorkType
    NvU32                           work_type;                                          // In

    NV_STATUS rmStatus;                                                                 // Out
} UVM_TEST_FLUSH_DEFERRED_WORK_PARAMS;

#define UVM_TEST_NV_KTHREAD_Q                           UVM8_TEST_IOCTL_BASE(45)
typedef struct
{
    NV_STATUS rmStatus; // Out
} UVM_TEST_NV_KTHREAD_Q_PARAMS;

typedef enum
{
    UVM_TEST_PAGE_PREFETCH_POLICY_ENABLE = 0,
    UVM_TEST_PAGE_PREFETCH_POLICY_DISABLE,
    UVM_TEST_PAGE_PREFETCH_POLICY_MAX
} UVM_TEST_PAGE_PREFETCH_POLICY;

#define UVM_TEST_SET_PAGE_PREFETCH_POLICY               UVM8_TEST_IOCTL_BASE(46)
typedef struct
{
    NvU32       policy; // In (UVM_TEST_PAGE_PREFETCH_POLICY)
    NV_STATUS rmStatus; // Out
} UVM_TEST_SET_PAGE_PREFETCH_POLICY_PARAMS;

#define UVM_TEST_RANGE_GROUP_TREE                       UVM8_TEST_IOCTL_BASE(47)
typedef struct
{
    NvU64 rangeGroupIds[4]                                           NV_ALIGN_BYTES(8); // In
    NV_STATUS rmStatus;                                                                 // Out
} UVM_TEST_RANGE_GROUP_TREE_PARAMS;

#define UVM_TEST_RANGE_GROUP_RANGE_INFO                 UVM8_TEST_IOCTL_BASE(48)
typedef struct
{
    NvU64                           lookup_address                   NV_ALIGN_BYTES(8); // In

    NvU64                           range_group_range_start          NV_ALIGN_BYTES(8); // Out
    NvU64                           range_group_range_end            NV_ALIGN_BYTES(8); // Out, inclusive
    NvU64                           range_group_id                   NV_ALIGN_BYTES(8); // Out
    NvU32                           range_group_present;                                // Out
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_RANGE_GROUP_RANGE_INFO_PARAMS;

#define UVM_TEST_RANGE_GROUP_RANGE_COUNT                UVM8_TEST_IOCTL_BASE(49)
typedef struct
{
    NvU64                           rangeGroupId                     NV_ALIGN_BYTES(8); // In
    NvU64                           count                            NV_ALIGN_BYTES(8); // Out
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_RANGE_GROUP_RANGE_COUNT_PARAMS;

#define UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE     UVM8_TEST_IOCTL_BASE(50)
typedef struct
{
    NvU32       reenable_lapse; // Out: Lapse in miliseconds
    NV_STATUS         rmStatus; // Out
} UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS;

#define UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE     UVM8_TEST_IOCTL_BASE(51)
typedef struct
{
    NvU32       reenable_lapse; // In: Lapse in miliseconds
    NV_STATUS         rmStatus; // Out
} UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS;

#define UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS             UVM8_TEST_IOCTL_BASE(52)
typedef struct
{
    NvU64                           addr                            NV_ALIGN_BYTES(8); // Out
    NV_STATUS                       rmStatus;                                          // Out
} UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS_PARAMS;

// Allocate and free memory directly from PMA with eviction enabled. This allows
// to simulate RM-like allocations, but without the RM API lock serializing
// everything.
#define UVM_TEST_PMA_ALLOC_FREE                         UVM8_TEST_IOCTL_BASE(53)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvU32                           page_size;
    NvBool                          contiguous;
    NvU64                           num_pages                        NV_ALIGN_BYTES(8); // In
    NvU64                           phys_begin                       NV_ALIGN_BYTES(8); // In
    NvU64                           phys_end                         NV_ALIGN_BYTES(8); // In
    NvU32                           nap_us_before_free;                                 // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMA_ALLOC_FREE_PARAMS;

// Allocate and free user memory directly from PMM with eviction enabled.
//
// Provides a direct way of exercising PMM allocs, eviction and frees of user
// memory type.
#define UVM_TEST_PMM_ALLOC_FREE_ROOT                    UVM8_TEST_IOCTL_BASE(54)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvU32                           nap_us_before_free;                                 // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMM_ALLOC_FREE_ROOT_PARAMS;

// Inject a PMA eviction error after the specified number of chunks are
// evicted.
#define UVM_TEST_PMM_INJECT_PMA_EVICT_ERROR             UVM8_TEST_IOCTL_BASE(55)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvU32                           error_after_num_chunks;                             // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMM_INJECT_PMA_EVICT_ERROR_PARAMS;

// Change configuration of access counters. This call will disable access
// counters and reenable them using the new configuration. All previous
// notifications will be lost
#define UVM_TEST_RECONFIGURE_ACCESS_COUNTERS            UVM8_TEST_IOCTL_BASE(56)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In

    // Type UVM_ACCESS_COUNTER_GRANULARITY from nv_uvm_types.h
    NvU32                           mimc_granularity;                                   // In
    NvU32                           momc_granularity;                                   // In

    // Type UVM_ACCESS_COUNTER_USE_LIMIT from nv_uvm_types.h
    NvU32                           mimc_use_limit;                                     // In
    NvU32                           momc_use_limit;                                     // In

    NvU32                           threshold;                                          // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_RECONFIGURE_ACCESS_COUNTERS_PARAMS;

typedef enum
{
    UVM_TEST_ACCESS_COUNTER_RESET_MODE_ALL = 0,
    UVM_TEST_ACCESS_COUNTER_RESET_MODE_TARGETED,
    UVM_TEST_ACCESS_COUNTER_RESET_MODE_MAX
} UVM_TEST_ACCESS_COUNTER_RESET_MODE;

typedef enum
{
    UVM_TEST_ACCESS_COUNTER_TYPE_MIMC = 0,
    UVM_TEST_ACCESS_COUNTER_TYPE_MOMC,
    UVM_TEST_ACCESS_COUNTER_TYPE_MAX
} UVM_TEST_ACCESS_COUNTER_TYPE;

// Clear the contents of the access counters. This call supports different
// modes for targeted/global resets.
#define UVM_TEST_RESET_ACCESS_COUNTERS                  UVM8_TEST_IOCTL_BASE(57)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In

    // Type UVM_TEST_ACCESS_COUNTER_RESET_MODE
    NvU32                           mode;                                               // In

    // Type UVM_TEST_ACCESS_COUNTER_TYPE
    NvU32                           counter_type;                                       // In

    NvU32                           bank;                                               // In
    NvU32                           tag;                                                // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_RESET_ACCESS_COUNTERS_PARAMS;

// Do not handle access counter notifications when they arrive. This call is
// used to force an overflow of the access counter notification buffer
#define UVM_TEST_SET_IGNORE_ACCESS_COUNTERS             UVM8_TEST_IOCTL_BASE(58)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvBool                          ignore;                                             // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_SET_IGNORE_ACCESS_COUNTERS_PARAMS;

// Verifies that the given channel is registered under the UVM VA space of
// vaSpaceFd. Returns NV_OK if so, NV_ERR_INVALID_CHANNEL if not.
#define UVM_TEST_CHECK_CHANNEL_VA_SPACE                 UVM8_TEST_IOCTL_BASE(59)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvS32                           rm_ctrl_fd;                                         // In
    NvHandle                        client;                                             // In
    NvHandle                        channel;                                            // In
    NvU32                           ve_id;                                              // In
    NvS32                           va_space_fd;                                        // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_CHECK_CHANNEL_VA_SPACE_PARAMS;

//
// UvmTestEnableNvlinkPeerAccess
//
#define UVM_TEST_ENABLE_NVLINK_PEER_ACCESS              UVM8_TEST_IOCTL_BASE(60)
typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_TEST_ENABLE_NVLINK_PEER_ACCESS_PARAMS;

//
// UvmTestDisableNvlinkPeerAccess
//
#define UVM_TEST_DISABLE_NVLINK_PEER_ACCESS             UVM8_TEST_IOCTL_BASE(61)
typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_TEST_DISABLE_NVLINK_PEER_ACCESS_PARAMS;

typedef enum
{
    UVM_TEST_PAGE_THRASHING_POLICY_ENABLE = 0,
    UVM_TEST_PAGE_THRASHING_POLICY_DISABLE,
    UVM_TEST_PAGE_THRASHING_POLICY_MAX
} UVM_TEST_PAGE_THRASHING_POLICY;

#define UVM_TEST_GET_PAGE_THRASHING_POLICY              UVM8_TEST_IOCTL_BASE(62)
typedef struct
{
    NvU32                           policy;                                             // Out (UVM_TEST_PAGE_THRASHING_POLICY)
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_GET_PAGE_THRASHING_POLICY_PARAMS;

#define UVM_TEST_SET_PAGE_THRASHING_POLICY              UVM8_TEST_IOCTL_BASE(63)
typedef struct
{
    NvU32                           policy;                                             // In (UVM_TEST_PAGE_THRASHING_POLICY)
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_SET_PAGE_THRASHING_POLICY_PARAMS;

#define UVM_TEST_PMM_SYSMEM                             UVM8_TEST_IOCTL_BASE(64)
typedef struct
{
    NvU64                           range_address1                   NV_ALIGN_BYTES(8); // In
    NvU64                           range_address2                   NV_ALIGN_BYTES(8); // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMM_SYSMEM_PARAMS;

#define UVM_TEST_PMM_REVERSE_MAP                        UVM8_TEST_IOCTL_BASE(65)
typedef struct
{
    NvProcessorUuid                 gpu_uuid;                                           // In
    NvU64                           range_address1                   NV_ALIGN_BYTES(8); // In
    NvU64                           range_address2                   NV_ALIGN_BYTES(8); // In
    NvU64                           range_size2                      NV_ALIGN_BYTES(8); // In
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMM_REVERSE_MAP_PARAMS;

#define UVM_TEST_PMM_INDIRECT_PEERS                     UVM8_TEST_IOCTL_BASE(66)
typedef struct
{
    NV_STATUS                       rmStatus;                                           // Out
} UVM_TEST_PMM_INDIRECT_PEERS_PARAMS;

#ifdef __cplusplus
}
#endif

#endif // __UVM8_TEST_IOCTL_H__
