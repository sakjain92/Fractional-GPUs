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

#ifndef __UVM8_GPU_H__
#define __UVM8_GPU_H__

#include "nvtypes.h"
#include "nvmisc.h"
#include "uvmtypes.h"
#include "nv_uvm_types.h"
#include "uvm_linux.h"
#include "nv-kref.h"
#include "uvm_common.h"
#include "ctrl2080mc.h"
#include "uvm8_forward_decl.h"
#include "uvm8_processors.h"
#include "uvm8_pmm_gpu.h"
#include "uvm8_pmm_sysmem.h"
#include "uvm8_mmu.h"
#include "uvm8_gpu_replayable_faults.h"
#include "uvm8_gpu_isr.h"
#include "uvm8_hal_types.h"
#include "uvm8_hmm.h"
#include "uvm8_va_block_types.h"
#include "nv-kthread-q.h"

// Buffer length to store uvm gpu id, RM device name and gpu uuid.
#define UVM_GPU_NICE_NAME_BUFFER_LENGTH (sizeof("ID 999: : ") + \
            UVM_GPU_NAME_LENGTH + UVM_GPU_UUID_TEXT_BUFFER_LENGTH)

#define UVM_GPU_MAGIC_VALUE 0xc001d00d12341993ULL

extern int uvm8_ats_mode;

typedef struct
{
    // Number of faults from this uTLB that have been fetched but have not been serviced yet
    NvU32 num_pending_faults;

    // Whether the uTLB contains fatal faults
    bool has_fatal_faults;

    // We have issued a replay of type START_ACK_ALL while containing fatal faults. This puts
    // the uTLB in lockdown mode and no new translations are accepted
    bool in_lockdown;

    // We have issued a cancel on this uTLB
    bool cancelled;

    uvm_fault_buffer_entry_t prev_fatal_fault;

    // Last fetched fault that was originated from this uTLB. Used for fault
    // filtering.
    uvm_fault_buffer_entry_t *last_fault;
} uvm_fault_utlb_info_t;

struct uvm_fault_service_block_context_struct
{
    //
    // Fields initialized by CPU/GPU fault handling routines
    //

    // Processors that will be the residency of pages that faulted
    uvm_processor_mask_t resident_processors;

    // VA block region that contains all the faults
    uvm_va_block_region_t fault_region;

    // Array of type uvm_fault_access_type_t that contains the type of the
    // access that caused the fault to be serviced for each page
    NvU8 fault_access_type[PAGES_PER_UVM_VA_BLOCK];

    // Number of times the fault service operation has been retried
    unsigned num_retries;

    // Pages that need to be pinned due to thrashing
    uvm_page_mask_t thrashing_pin_mask;

    // Number of pages that need to be pinned due to thrashing. This is the same
    // value as the result of bitmap_weight(thrashing_pin_mask)
    unsigned thrashing_pin_count;

    // Pages that can be read-duplicated
    uvm_page_mask_t read_duplicate_mask;

    // Number of pages that can be read-duplicated. This is the same value as
    // the result of bitmap_weight(read_duplicate_count_mask)
    unsigned read_duplicate_count;

    //
    // Fields used by the CPU fault handling routine
    //

    struct {
        // Node of the list of fault service contexts used by the CPU
        struct list_head service_context_list;

        // A mask of GPUs that need to be checked for ECC errors before the CPU
        // fault handler returns, but after the VA space lock has been unlocked to
        // avoid the RM/UVM VA space lock deadlocks.
        uvm_processor_mask_t fault_gpus_to_check_for_ecc;

        // This is set to throttle page fault thrashing.
        NvU64 wakeup_time_stamp;

        // This is set if the page migrated to/from the GPU and CPU.
        bool fault_did_migrate;
    } cpu;

    //
    // Fields managed by the common fault handling routine
    //

    struct
    {
        // Pages that need to be mapped with the corresponding protection
        uvm_page_mask_t mask;

        // Number of pages that need to be mapped with the corresponding
        // protections. This is the same value as the result of
        // bitmap_weight(mask)
        unsigned count;
    } mappings_by_prot[UVM_PROT_MAX - 1];

    // Mask with the pages that did not migrate to the processor (they were
    // already resident) in the last call to uvm_va_block_make_resident.
    // This is used to compute the pages that need to revoke mapping permissions
    // from other processors.
    uvm_page_mask_t did_not_migrate_mask;

    // Pages whose permissions need to be revoked from other processors
    uvm_page_mask_t revocation_mask;

    struct
    {
        // Per-processor mask with the pages that will be resident after servicing.
        // We need one mask per processor because we may coalesce faults that
        // trigger migrations to different processors.
        uvm_page_mask_t new_residency;
    } per_processor_masks[UVM_MAX_PROCESSORS];

    // State used by the VA block routines called by the fault handler
    uvm_va_block_context_t block_context;

    // Whether the information refers to replayable or non-replayable faults
    bool are_replayable_faults;
};

struct uvm_fault_service_batch_context_struct
{
    // Array of elements fetched from the GPU fault buffer. The number of
    // elements in this array is exactly max_batch_size
    uvm_fault_buffer_entry_t *fault_cache;

    // Array of pointers to elements in fault cache used for fault
    // preprocessing. The number of elements in this array is exactly
    // max_batch_size
    uvm_fault_buffer_entry_t **ordered_fault_cache;

    // Per uTLB fault information. Used for replay policies and fault
    // cancellation on Pascal
    uvm_fault_utlb_info_t *utlbs;

    // Largest uTLB id seen in a GPU fault
    NvU32 max_utlb_id;

    NvU32 num_cached_faults;

    NvU32 num_coalesced_faults;

    bool has_fatal_faults;

    bool has_throttled_faults;

    NvU32 num_invalid_prefetch_faults;

    NvU32 num_duplicate_faults;

    NvU32 num_replays;

    // Unique id (per-GPU) generated for tools events recording
    NvU32 batch_id;

    uvm_tracker_t tracker;

    // Boolean used to avoid sorting the fault batch by instance_ptr if we
    // determine at fetch time that all the faults in the batch report the same
    // instance_ptr
    bool is_single_instance_ptr;

    // Last fetched fault. Used for fault filtering.
    uvm_fault_buffer_entry_t *last_fault;
};

typedef struct
{
    // Fault buffer information and structures provided by RM
    UvmGpuFaultInfo rm_info;

    // Maximum number of faults to be processed in batch before fetching new
    // entries from the GPU buffer
    NvU32 max_batch_size;

    struct uvm_replayable_fault_buffer_info_struct
    {
        // Maximum number of faults entries that can be stored in the buffer
        NvU32 max_faults;

        // Cached value of the GPU GET register to minimize the round-trips
        // over PCIe
        NvU32 cached_get;

        // Cached value of the GPU PUT register to minimize the round-trips over
        // PCIe
        NvU32 cached_put;

        // Policy that determines when GPU replays are issued during normal
        // fault servicing
        uvm_perf_fault_replay_policy_t replay_policy;

        // Tracker used to aggregate replay operations, needed for fault cancel
        // and GPU removal
        uvm_tracker_t replay_tracker;

        // If there is a ratio larger than replay_update_put_ratio of duplicate
        // faults in a batch, PUT pointer is updated before flushing the buffer
        // that comes before the replay method.
        NvU32 replay_update_put_ratio;

        // Fault statistics. These fields are per-GPU and most of them are only
        // updated during fault servicing, and can be safely incremented.
        // Migrations may be triggered by different GPUs and need to be
        // incremented using atomics
        struct
        {
            NvU64 num_prefetch_faults;

            NvU64 num_read_faults;

            NvU64 num_write_faults;

            NvU64 num_atomic_faults;

            NvU64 num_duplicate_faults;

            atomic64_t num_pages_out;

            atomic64_t num_pages_in;

            NvU64 num_replays;

            NvU64 num_replays_ack_all;
        } stats;

        // Number of uTLBs in the chip
        NvU32 utlb_count;

        // Context structure used to service a GPU fault batch
        uvm_fault_service_batch_context_t batch_service_context;

        // Structure used to coalesce fault servicing in a VA block
        uvm_fault_service_block_context_t block_service_context;
    } replayable;

    struct uvm_non_replayable_fault_buffer_info_struct
    {
        // Maximum number of faults entries that can be stored in the buffer
        NvU32 max_faults;

        // Tracker used to aggregate clear faulted operations, needed for GPU
        // removal
        uvm_tracker_t clear_faulted_tracker;

        // Buffer used to store elements popped out from the queue shared with
        // RM for fault servicing.
        void *shadow_buffer_copy;

        // Array of elements fetched from the GPU fault buffer. The number of
        // elements in this array is exactly max_batch_size
        uvm_fault_buffer_entry_t *fault_cache;

        // Fault statistics. See replayable fault stats for more details.
        struct
        {
            NvU64 num_read_faults;

            NvU64 num_write_faults;

            NvU64 num_atomic_faults;

            NvU64 num_physical_faults;
        } stats;

        // Tracker which temporarily holds the work pushed to service faults
        uvm_tracker_t fault_service_tracker;

        // Structure used to coalesce fault servicing in a VA block
        uvm_fault_service_block_context_t block_service_context;

        // Unique id (per-GPU) generated for tools events recording
        NvU32 batch_id;
    } non_replayable;

    // Flag that tells if prefetch faults are enabled in HW
    bool prefetch_faults_enabled;

    // Timestamp when prefetch faults where disabled last time
    NvU64 disable_prefetch_faults_timestamp;
} uvm_fault_buffer_info_t;

typedef struct
{
    // True if the platform supports HW coherence (P9) and RM has exposed the
    // GPU's memory as a NUMA node to the kernel.
    bool enabled;

    // Range in the system physical address space where the memory of this GPU
    // is mapped
    NvU64 system_memory_window_start;
    NvU64 system_memory_window_end;

    NvU64 memblock_size;

    unsigned node_id;
} uvm_numa_info_t;

struct uvm_access_counter_service_batch_context_struct
{
    uvm_access_counter_buffer_entry_t *notification_cache;

    NvU32 num_cached_notifications;

    struct
    {
        uvm_access_counter_buffer_entry_t   **notifications;

        NvU32                             num_notifications;

        // Boolean used to avoid sorting the fault batch by instance_ptr if we
        // determine at fetch time that all the access counter notifications in the
        // batch report the same instance_ptr
        bool is_single_instance_ptr;
    } virt;

    struct
    {
        uvm_access_counter_buffer_entry_t    **notifications;
        uvm_reverse_map_t                      *translations;

        NvU32                              num_notifications;

        // Boolean used to avoid sorting the fault batch by aperture if we
        // determine at fetch time that all the access counter notifications in the
        // batch report the same aperture
        bool                              is_single_aperture;
    } phys;

    // Unique id (per-GPU) generated for tools events recording
    NvU32 batch_id;
};

typedef struct
{
    UvmGpuAccessCntrInfo rm_info;

    NvU32 max_notifications;

    NvU32 max_batch_size;

    // Cached value of the GPU GET register to minimize the round-trips
    // over PCIe
    NvU32 cached_get;

    // Cached value of the GPU PUT register to minimize the round-trips over
    // PCIe
    NvU32 cached_put;

    // Current access counter configuration. During normal operation this
    // information is computed once during GPU initialization. However, tests
    // may override it to try different configuration values.
    struct
    {
        // The actual HW reconfiguration is performed by RM. This is the
        // struct with the configuration parameters passed to do it.
        UvmGpuAccessCntrConfig rm;

        // These values are precomputed by the access counter notification
        // handling code. See comments for UVM_MAX_TRANSLATION_SIZE in
        // uvm8_gpu_access_counters.c for more details.
        struct
        {
            NvU64 translation_size;

            NvU64 translations_per_counter;
        } mimc;

        struct
        {
            NvU64 translation_size;

            NvU64 translations_per_counter;
        } momc;
    } current_config;

    // Some tests disable notification processing in order to force a
    // notification buffer overflow.
    bool ignore_notifications;

    // Context structure used to service a GPU access counter batch
    uvm_access_counter_service_batch_context_t batch_service_context;
} uvm_access_counter_buffer_info_t;

typedef struct
{
    // VA where the identity mapping should be mapped in the internal VA
    // space managed by uvm_gpu_t.address_space_tree (see below).
    NvU64 base;

    // Page tables with the mapping.
    uvm_page_table_range_vec_t *range_vec;
} uvm_gpu_identity_mapping_t;

typedef enum
{
    UVM_GPU_LINK_INVALID = 0,
    UVM_GPU_LINK_PCIE,
    UVM_GPU_LINK_NVLINK_1,
    UVM_GPU_LINK_NVLINK_2,
    UVM_GPU_LINK_MAX
} uvm_gpu_link_type_t;

struct uvm_gpu_struct
{
    // Reference count for how many places are holding onto a GPU (internal to UVM driver).
    // This includes any GPUs we know about, not just GPUs that are registered with a VA space.
    // Most GPUs end up being registered, but there are brief periods when they are not
    // registered, such as during interrupt handling, and in add_gpu() or remove_gpu().
    nv_kref_t gpu_kref;

    // Refcount of the gpu, i.e. how many times it has been retained. This is
    // roughly a count of how many times it has been registered with a VA space,
    // except that some paths retain the GPU temporarily without a VA space.
    //
    // While this is >0, the GPU can't be removed. This differs from gpu_kref,
    // which merely prevents the uvm_gpu_t object from being freed.
    //
    // In most cases this count is protected by the global lock: retaining a GPU
    // from a UUID and any release require the global lock to be taken. But it's
    // also useful for a caller to retain a GPU they've already retained, in
    // which case there's no need to take the global lock. This can happen when
    // an operation needs to drop the VA space lock but continue operating on a
    // GPU. This is an atomic variable to handle those cases.
    //
    // Security note: keep it as a 64-bit counter to prevent overflow cases (a
    // user can create a lot of va spaces and register the gpu with them).
    atomic64_t retained_count;

    // A unique uvm gpu id in range [1, UVM_MAX_PROCESSORS)
    uvm_gpu_id_t id;

    // The gpu's uuid
    NvProcessorUuid uuid;

    // Nice printable name including the uvm gpu id, ascii name from RM and uuid
    char name[UVM_GPU_NICE_NAME_BUFFER_LENGTH];

    // Reference to the Linux PCI device
    //
    // The reference to the PCI device remains valid as long as the GPU is
    // registered with RM's Linux layer (between nvUvmInterfaceRegisterGpu() and
    // nvUvmInterfaceUnregisterGpu()).
    struct pci_dev *pci_dev;

    // PCI device of NVLink Processing Unit (NPU) on PowerPC platforms.
    struct pci_dev *npu_dev;

    // The physical address range addressable by the GPU
    //
    // The GPU has its NV_PFB_XV_UPPER_ADDR register set by RM to
    // dma_addressable_start (in bifSetupDmaWindow_IMPL()) and hence when
    // referencing sysmem from the GPU, dma_addressable_start should be
    // subtracted from the physical address. uvm_gpu_map_cpu_pages() takes care
    // of that.
    NvU64 dma_addressable_start;
    NvU64 dma_addressable_limit;

    // Total size (in bytes) of physically mapped (with uvm_gpu_map_cpu_pages)
    // sysmem pages, used for leak detection.
    atomic64_t mapped_cpu_pages_size;

    // Should be UVM_GPU_MAGIC_VALUE. Used for memory checking.
    NvU64 magic;

    // Gpu architecture; NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_*
    NvU32 architecture;

    // Gpu implementation; NV2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_*
    NvU32 implementation;

    // Host (gpfifo) class; *_CHANNEL_GPFIFO_*, e.g. KEPLER_CHANNEL_GPFIFO_A
    NvU32 host_class;

    // Copy engine (dma) class; *_DMA_COPY_*, e.g. KEPLER_DMA_COPY_A
    NvU32 ce_class;

    // Fault buffer class; *_FAULT_BUFFER_*, e.g. MAXWELL_FAULT_BUFFER_A
    NvU32 fault_buffer_class;

    // Access counter buffer class; ACCESS_COUNTER_NOTIFY_BUFFER
    NvU32 access_counter_buffer_class;

    // Hardware Abstraction Layer
    uvm_host_hal_t *host_hal;
    uvm_ce_hal_t *ce_hal;
    uvm_arch_hal_t *arch_hal;
    uvm_fault_buffer_hal_t *fault_buffer_hal;
    uvm_access_counter_buffer_hal_t *access_counter_buffer_hal;

    // The amount of video memory the GPU has in total, in bytes.
    NvU64 vidmem_size;

    // Max (inclusive) physical address of this GPU's vidmem that the driver can
    // allocate through PMM (PMA).
    NvU64 vidmem_max_allocatable_address;

    struct
    {
        // Big page size used by the internal UVM VA space
        // Notably it may be different than the big page size used by a user's VA
        // space in general.
        NvU32 internal_size;

        // Whether big page mappings have the physical memory swizzled.
        // On some architectures (Kepler) physical memory mapped with a big page
        // size doesn't follow the common 1-1 mapping where each offset within the
        // big page maps to the same offset within the mapped physical memory.
        // Swizzling makes physically accessing memory mapped with big pages
        // infeasible and has to be worked around by creating virtual identity
        // mappings (see big_page_self_identity_mapping below).
        bool swizzling;

        // Big page self identity mapping
        // Used only on GPUs that have big page swizzling enabled. Notably
        // that's only Kepler which only supports one big page size at a time
        // and hence only a single mapping is needed.
        uvm_gpu_identity_mapping_t identity_mapping;

        struct
        {
            // These fields are only used on GPUs that have big page swizzling
            // enabled.

            // Staging memory used for converting between formats
            uvm_gpu_chunk_t *chunk;

            // Tracker for all staging operations performed using chunk
            uvm_tracker_t tracker;

            // Lock protecting tracker
            uvm_mutex_t lock;
        } staging;
    } big_page;

    // lazily-created peer identity mappings
    bool peer_identity_mappings_supported;
    uvm_gpu_identity_mapping_t peer_mappings[UVM_APERTURE_PEER_MAX];

    // lazily-populated array of peer GPUs, indexed by peer id
    uvm_gpu_t *peer_gpus[UVM_APERTURE_PEER_MAX];

    // Leaf spinlok used to synchronize access to the peer_gpus table so that
    // it can be safely accessed from the access counters bottom half
    uvm_spinlock_t peer_gpus_lock;

    // Whether the GPU can trigger faults on prefetch instructions
    bool prefetch_fault_supported;

    // Number of membars required to flush out HSHUB following a TLB invalidate
    NvU32 num_hshub_tlb_invalidate_membars;

    // Whether the channels can configure GPFIFO in vidmem
    bool gpfifo_in_vidmem_supported;

    bool replayable_faults_supported;

    bool non_replayable_faults_supported;

    bool access_counters_supported;

    bool fault_cancel_va_supported;

    // Parameters used by the TLB batching API
    struct
    {
        // Is the targeted (single page) VA invalidate supported at all?
        NvBool va_invalidate_supported;

        // Is the VA range invalidate supported?
        NvBool va_range_invalidate_supported;

        union
        {
            // Maximum (inclusive) number of single page invalidations before falling back to invalidate all
            NvU32 max_pages;

            // Maximum (inclusive) number of range invalidations before falling back to invalidate all
            NvU32 max_ranges;
        };
    } tlb_batch;

    // Largest VA (exclusive) which can be used for channel buffer mappings
    NvU64 max_channel_va;

    // Indicates whether the GPU can map sysmem with pages larger than 4k
    bool can_map_sysmem_with_large_pages;

    // VA base and size of the RM managed part of the internal UVM VA space.
    //
    // The internal UVM VA is shared with RM by RM controlling some of the top
    // level PDEs and leaving the rest for UVM to control.
    // On Pascal a single top level PDE covers 128 TB of VA and given that
    // semaphores and other allocations limited to 40bit are currently allocated
    // through RM, RM needs to control the [0, 128TB) VA range at least for now.
    // On Kepler and Maxwell limit RMs VA to [0, 128GB) that should easily fit
    // all RM allocations and leave enough space for UVM.
    NvU64 rm_va_base;
    NvU64 rm_va_size;

    // Base and size of the GPU VA used for uvm_mem_t allocations mapped in the
    // internal address_space_tree.
    NvU64 uvm_mem_va_base;
    NvU64 uvm_mem_va_size;

    // RM address space handle used in many of the UVM/RM APIs
    // Represents both an RM device and a GPU VA in RM.
    uvmGpuAddressSpaceHandle rm_address_space;

    // Page tree used for the internal UVM VA space shared with RM
    uvm_page_tree_t address_space_tree;

    // Set to true during add_gpu() as soon as the RM's address space is moved
    // to the address_space_tree.
    bool rm_address_space_moved_to_page_tree;

    // ECC handling
    // In order to trap ECC errors as soon as possible the driver has the hw
    // interrupt register mapped directly. If an ECC interrupt is ever noticed
    // to be pending, then the UVM driver needs to:
    //
    //   1) ask RM to service interrupts, and then
    //   2) inspect the ECC error notifier state.
    //
    // Notably, checking for channel errors is not enough, because ECC errors
    // can be pending, even after a channel has become idle.
    //
    // See more details in uvm_gpu_check_ecc_error().
    struct
    {
        // Does the GPU have ECC enabled?
        bool enabled;

        // Direct mapping of the 32-bit part of the hw interrupt tree that has
        // the ECC bits.
        volatile NvU32 *hw_interrupt_tree_location;

        // Mask to get the ECC interrupt bits from the 32-bits above.
        NvU32 mask;

        // Set to true by RM when a fatal ECC error is encountered (requires
        // asking RM to service pending interrupts to be current).
        NvBool *error_notifier;
    } ecc;

    UvmGpuCopyEngineCaps ce_caps[UVM_COPY_ENGINE_COUNT_MAX];

    uvm_gpu_semaphore_pool_t *semaphore_pool;

    uvm_channel_manager_t *channel_manager;

    struct
    {
        // Procfs entry for the GPU directory
        struct proc_dir_entry *dir;

        // Procfs entry for the uuid symlink to the GPU directory
        struct proc_dir_entry *dir_uuid_symlink;

        // Procfs entry for the info file
        struct proc_dir_entry *info_file;

        // Procfs entry for the stats file
        struct proc_dir_entry *fault_stats_file;
    } procfs;

    uvm_pmm_gpu_t pmm;

    uvm_pmm_sysmem_mappings_t pmm_sysmem_mappings;

    // Interrupt handling state and locks
    uvm_isr_info_t isr;

    // Fault buffer info. This is only valid if supports_replayable_faults is set to true
    uvm_fault_buffer_info_t fault_buffer_info;

    // NUMA info, mainly for ATS
    uvm_numa_info_t numa_info;

    // Access counter buffer info. This is only valid if supports_access_counters is set to true
    uvm_access_counter_buffer_info_t access_counter_buffer_info;

    // Number of GPCs
    NvU32 gpc_count;

    // Number of TPCs
    NvU32 tpc_count;

    // Maximum number of TPCs per GPC
    NvU32 max_tpc_per_gpc;

    NvU32 max_subcontexts;

    // In order to service GPU faults, UVM must be able to obtain the VA
    // space for each reported fault. The fault packet contains the
    // instance_ptr of the channel that was bound when the SMs triggered
    // the fault. On fault any instance pointer in the TSG may be
    // reported. This is a problem on Volta, which allow different channels
    // in the TSG to be bound to different VA spaces in order to support
    // subcontexts. In order to be able to obtain the correct VA space, HW
    // provides the subcontext id (or VEID) in addition to the instance_ptr.
    //
    // Summary:
    //
    // 1) Channels in a TSG may be in different VA spaces, identified by their
    // subcontext ID.
    // 2) Different subcontext IDs may map to the same or different VA spaces.
    // 3) On fault, any instance pointer in the TSG may be reported. The
    // reported subcontext ID identifies which VA space within the TSG actually
    // encountered the fault.
    //
    // Thus, UVM needs to keep track of all the instance pointers that belong
    // to the same TSG. We use two tables:
    //
    // - instance_ptr_table (instance_ptr -> subctx_info) this table maps
    // instance pointers to the subcontext info descriptor for the channel. If
    // the channel belongs to a subcontext, this descriptor will contain all
    // the VA spaces for the subcontexts in the same TSG. If the channel does
    // not belong to a subcontext, it will only contain a pointer to its VA
    // space.
    // - tsg_table (tsg_id -> subctx_info): this table also stores the
    // subctx information, but in this case it is indexed by TSG ID. Thus,
    // when a new channel bound to a subcontext is registered, it will check
    // first in this table if the subcontext information descriptor for its TSG
    // already exists, otherwise it will create it. Channels not bound to
    // subcontexts will not use this table.
    //
    // The bottom half reads the tables under
    // isr.replayable_faults_handler.lock, but a separate lock is necessary
    // because entries are added and removed from the table under the va_space
    // lock, and we can't take isr.replayable_faults_handler.lock while holding
    // the va_space lock.
    struct radix_tree_root tsg_table;

    struct radix_tree_root instance_ptr_table;
    uvm_spinlock_t instance_ptr_table_lock;

    // This is set to true if the GPU belongs to an SLI group. Else, set to false.
    bool sli_enabled;

    // This is set to true if the GPU is a simulated/emulated device. Else, set to false.
    bool is_simulated;

    // Global statistics. These fields are per-GPU and most of them are only
    // updated during fault servicing, and can be safely incremented.
    struct
    {
        NvU64          num_replayable_faults;

        NvU64      num_non_replayable_faults;

        atomic64_t             num_pages_out;

        atomic64_t              num_pages_in;
    } stats;

#if UVM_IS_CONFIG_HMM()
    uvm_hmm_gpu_t hmm_gpu;
#endif



















    uvm_gpu_link_type_t sysmem_link;
};

struct uvm_gpu_peer_struct
{
    // Note: All the peer_caps fields in this global structure can be queried
    // if and only if the corresponding bit from "va_space.enabled_peers" bitmap
    // is set.

    // Peer Id associated with this device w.r.t. to a peer GPU.
    // Note: peerId (A -> B) != peerId (B -> A)
    // peer_id[0] from min(gpu_id_1, gpi_id_2) -> max(gpu_id_1, gpi_id_2)
    // peer_id[1] from max(gpu_id_1, gpi_id_2) -> min(gpu_id_1, gpi_id_2)
    NvU8 peer_ids[2];

    // 1 if peers have direct connection, 0 if P2P access is routed via an
    // intermediate processor. Currently, this represents GPUs connected to
    // different P9 sockets.
    NvU8 is_direct_peer : 1;

    // The link type between the peer GPUs, currently either PCIe or NVLINK1/2.
    // This field is used to determine the when this peer struct has been
    // initialized (link_type != UVM_GPU_LINK_INVALID). NVLink peers are
    // initialized at GPU registration time. PCIe peers are initialized when
    // the refcount below goes from 0 to 1.
    uvm_gpu_link_type_t link_type;

    // For PCIe, the number of times that this has been retained by a VA space.
    // For NVLINK this will always be 1.
    NvU64 ref_count;

    // Indexes of the optimal Copy Engines to use when writing between peers:
    //     -optimal_write_ce[0]: writes from min(gpu_id_1, gpu_id_2) to
    //      max(gpu_id_1, gpu_id_2)
    //     -optimal_write_ce[1]: writes from max(gpu_id_1, gpu_id_2) to
    //      min(gpu_id_1, gpu_id_2)
    // The indexes are valid only if the two GPUs are direct NvLink peers
    // (see uvm_gpu_to_gpu_optimal_write_ce)
    NvU32 optimal_write_ce[2];

    // This handle gets populated when enable_peer_access successfully creates
    // an NV50_P2P object. disable_peer_access resets the same on the object
    // deletion.
    NvHandle p2p_handle;
};

// Initialize global gpu state
NV_STATUS uvm_gpu_init(void);

// Deinitialize global state (called from module exit)
void uvm_gpu_exit(void);

NV_STATUS uvm_gpu_init_va_space(uvm_va_space_t *va_space);

void uvm_gpu_exit_va_space(uvm_va_space_t *va_space);


// Note that there is a uvm_gpu_get() function defined in uvm_global.h to break
// a circular dep between global and gpu modules.

// Get a gpu by uuid. This returns NULL if the GPU is not present. This is the
// general purpose call that should be used normally.
//
// LOCKING: requires the global lock to be held
uvm_gpu_t *uvm_gpu_get_by_uuid(const NvProcessorUuid *gpu_uuid);

// Same as uvm_gpu_get_by_uuid, except that this one does not assert that the
// caller is holding the global_lock. This is a narrower purpose function, and
// is only intended for use by the top-half ISR, or other very limited cases.
uvm_gpu_t *uvm_gpu_get_by_uuid_locked(const NvProcessorUuid *gpu_uuid);

// Retain a gpu by uuid
// Returns the retained uvm_gpu_t in gpu_out on success
// LOCKING: requires the global lock to be held
NV_STATUS uvm_gpu_retain_by_uuid_locked(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out);

// Like uvm_gpu_retain_by_uuid_locked, but takes and releases the global lock
// for the caller.
NV_STATUS uvm_gpu_retain_by_uuid(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out);

// Retain a gpu which is known to already be retained. Does NOT require the
// global lock to be held.
void uvm_gpu_retain(uvm_gpu_t *gpu);

// Release a gpu
// LOCKING: requires the global lock to be held
void uvm_gpu_release_locked(uvm_gpu_t *gpu);

// Like uvm_gpu_release_locked, but takes and releases the global lock for the
// caller.
void uvm_gpu_release(uvm_gpu_t *gpu);

// Helper which calls uvm_gpu_retain on each GPU in mask
void uvm_gpu_retain_mask(const uvm_processor_mask_t *mask);

// Helper which calls uvm_gpu_release_locked on each GPU in mask
void uvm_gpu_release_mask_locked(const uvm_processor_mask_t *mask);

// Like uvm_gpu_release_mask_locked, but takes and releases the global lock for
// the caller.
void uvm_gpu_release_mask(const uvm_processor_mask_t *mask);

static NvU64 uvm_gpu_retained_count(uvm_gpu_t *gpu)
{
    return atomic64_read(&gpu->retained_count);
}

// Decrease the refcount on the GPU object, and actually delete the object if the refcount hits
// zero.
void uvm_gpu_kref_put(uvm_gpu_t *gpu);

// Calculates peer table index using GPU ids.
NvU32 uvm_gpu_peer_table_index(uvm_gpu_id_t gpu_id1, uvm_gpu_id_t gpu_id2);

// Either retains an existing PCIe peer entry or creates a new one. In both
// cases the two GPUs are also each retained.
// LOCKING: requires the global lock to be held
NV_STATUS uvm_gpu_retain_pcie_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2);

// Releases a PCIe peer entry and the two GPUs.
// LOCKING: requires the global lock to be held
void uvm_gpu_release_pcie_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2);

// Get the aperture for local_gpu to use to map memory resident on remote_gpu.
// They must not be the same gpu.
uvm_aperture_t uvm_gpu_peer_aperture(uvm_gpu_t *local_gpu, uvm_gpu_t *remote_gpu);

// Get the processor id accessible by the given GPU for the given aperture.
uvm_processor_id_t uvm_gpu_get_processor_id_by_aperture(uvm_gpu_t *gpu, uvm_aperture_t aperture);

// Get the P2P capabilities between the gpus with the given indexes
uvm_gpu_peer_t *uvm_gpu_index_peer_caps(uvm_gpu_id_t gpu_id1, uvm_gpu_id_t gpu_id2);

// Get the P2P capabilities between the given gpus
static uvm_gpu_peer_t *uvm_gpu_peer_caps(const uvm_gpu_t *gpu_1, const uvm_gpu_t *gpu_2)
{
    return uvm_gpu_index_peer_caps(gpu_1->id, gpu_2->id);
}

static bool uvm_gpus_are_indirect_peers(uvm_gpu_t *gpu1, uvm_gpu_t *gpu2)
{
    uvm_gpu_peer_t *peer_caps = uvm_gpu_peer_caps(gpu1, gpu2);

    if (peer_caps->link_type != UVM_GPU_LINK_INVALID && !peer_caps->is_direct_peer) {
        UVM_ASSERT(gpu1->numa_info.enabled);
        UVM_ASSERT(gpu2->numa_info.enabled);
        return true;
    }

    return false;
}

// Get the index of the optimal Copy Engine to use when writing from source to 
// destination.
// Returns UVM_COPY_ENGINE_COUNT_MAX if the GPUs are not direct NvLink peers 
NvU32 uvm_gpu_to_gpu_optimal_write_ce(const uvm_gpu_t *src, const uvm_gpu_t *dst);

// Check for ECC errors
//
// Notably this check cannot be performed where it's not safe to call into RM.
NV_STATUS uvm_gpu_check_ecc_error(uvm_gpu_t *gpu);

// Check for ECC errors for all GPUs in a mask
NV_STATUS uvm_gpu_check_ecc_error_mask(uvm_processor_mask_t *gpus);

// Check for ECC errors without calling into RM
//
// Calling into RM is problematic in many places, this check is always safe to do.
// Returns NV_WARN_MORE_PROCESSING_REQUIRED if there might be an ECC error and
// it's required to call uvm_gpu_check_ecc_error() to be sure.
NV_STATUS uvm_gpu_check_ecc_error_no_rm(uvm_gpu_t *gpu);

// Map size bytes of contiguous sysmem on the GPU for physical access
//
// size has to be aligned to PAGE_SIZE.
//
// Returns the physical address of the pages that can be used to access them on
// the GPU.
NV_STATUS uvm_gpu_map_cpu_pages(uvm_gpu_t *gpu, struct page *page, size_t size, NvU64 *dma_address_out);

// Unmap num_pages pages previously mapped with uvm_gpu_map_cpu_pages().
void uvm_gpu_unmap_cpu_pages(uvm_gpu_t *gpu, NvU64 dma_address, size_t size);

static NV_STATUS uvm_gpu_map_cpu_page(uvm_gpu_t *gpu, struct page *page, NvU64 *dma_address_out)
{
    return uvm_gpu_map_cpu_pages(gpu, page, PAGE_SIZE, dma_address_out);
}

static void uvm_gpu_unmap_cpu_page(uvm_gpu_t *gpu, NvU64 dma_address)
{
    uvm_gpu_unmap_cpu_pages(gpu, dma_address, PAGE_SIZE);
}

static bool uvm_gpu_is_gk110_plus(uvm_gpu_t *gpu)
{
    return gpu->architecture >= NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK110;
}

// Returns whether the given address is within the GPU's maximum addressable VA
// range. Warning: This only checks whether the GPU's MMU can support the given
// address. Some HW units on that GPU might only support a smaller range.
//
// The GPU must be initialized before calling this function.
bool uvm_gpu_can_address(uvm_gpu_t *gpu, NvU64 addr);

static bool uvm_gpu_supports_eviction(uvm_gpu_t *gpu)
{
    // Eviction is supported only if the GPU supports replayable faults
    return gpu->replayable_faults_supported;
}

// Debug print of GPU properties
void uvm_gpu_print(uvm_gpu_t *gpu);

// Add the given instance pointer -> user_channel mapping to this GPU. The bottom
// half GPU page fault handler uses this to look up the VA space for GPU faults.
NV_STATUS uvm_gpu_add_user_channel(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel);
void uvm_gpu_remove_user_channel(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel);

// Looks up an entry added by uvm_gpu_add_user_channel. Return codes:
//  NV_OK                        Translation successful
//  NV_ERR_INVALID_CHANNEL       Entry's instance pointer was not found
//  NV_ERR_PAGE_TABLE_NOT_AVAIL  Entry's instance pointer is valid but the entry
//                               targets an invalid subcontext
//
// out_va_space is valid if NV_OK is returned, otherwise it's NULL. The caller
// is responsibile for ensuring that the returned va_space can't be destroyed,
// so these functions should only be called from the bottom half.
NV_STATUS uvm_gpu_fault_entry_to_va_space(uvm_gpu_t *gpu,
                                          uvm_fault_buffer_entry_t *fault,
                                          uvm_va_space_t **out_va_space);

NV_STATUS uvm_gpu_access_counter_entry_to_va_space(uvm_gpu_t *gpu,
                                                   uvm_access_counter_buffer_entry_t *entry,
                                                   uvm_va_space_t **out_va_space);

// Returns the physical address for use by accessing_gpu of a vidmem allocation
// on the peer owning_gpu. This address can be used for making PTEs on
// accessing_gpu, but not for copying between the two GPUs. For that, use
// uvm_gpu_peer_copy_address.
//
// TODO: Bug 2008200: Move this into PMM
uvm_gpu_phys_address_t uvm_gpu_peer_phys_address(uvm_gpu_t *accessing_gpu,
                                                 uvm_gpu_t *owning_gpu,
                                                 uvm_gpu_chunk_t *chunk);

// Returns the physical or virtual address for use by accessing_gpu to copy to/
// from a vidmem allocation on the peer owning_gpu. This may be different from
// uvm_gpu_peer_phys_address to handle CE limitations in addressing peer
// physical memory directly.
//
// TODO: Bug 2008200: Move this into PMM
uvm_gpu_address_t uvm_gpu_peer_copy_address(uvm_gpu_t *accessing_gpu, uvm_gpu_t *owning_gpu, uvm_gpu_chunk_t *chunk);

typedef enum
{
    UVM_GPU_SWIZZLE_OP_SWIZZLE,
    UVM_GPU_SWIZZLE_OP_DESWIZZLE,
    UVM_GPU_SWIZZLE_OP_COUNT
} uvm_gpu_swizzle_op_t;

// Convert the data format of the given big page physical address. The tracker
// parameter may be NULL. If not, it is an in/out parameter: the swizzle
// operation will acquire it, then replace it.
//
// This will only fail due to a global error.
NV_STATUS uvm_gpu_swizzle_phys(uvm_gpu_t *gpu,
                               NvU64 big_page_phys_address,
                               uvm_gpu_swizzle_op_t op,
                               uvm_tracker_t *tracker);

typedef enum
{
    UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT,
    UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
} uvm_gpu_buffer_flush_mode_t;

#endif // __UVM8_GPU_H__
