/*******************************************************************************
    Copyright (c) 2015-2017 NVIDIA Corporation

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

#include "nv_uvm_interface.h"
#include "uvm8_api.h"
#include "uvm8_channel.h"
#include "uvm8_global.h"
#include "uvm8_gpu.h"
#include "uvm8_gpu_semaphore.h"
#include "uvm8_hal.h"
#include "uvm8_procfs.h"
#include "uvm8_pmm_gpu.h"
#include "uvm8_pmm_sysmem.h"
#include "uvm8_va_space.h"
#include "uvm8_user_channel.h"
#include "uvm8_perf_events.h"
#include "uvm_common.h"
#include "ctrl2080mc.h"
#include "nv-kthread-q.h"
#include "uvm8_gpu_access_counters.h"

int uvm8_ats_mode = 0;
module_param(uvm8_ats_mode, int, S_IRUGO);
MODULE_PARM_DESC(uvm8_ats_mode, "Enable ATS (Address Translation Services) "
                                "UVM mode by setting this to 1");

static void remove_gpu(uvm_gpu_t *gpu);
static void disable_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2);
static NV_STATUS discover_nvlink_peers(uvm_gpu_t *gpu);
static void destroy_nvlink_peers(uvm_gpu_t *gpu);

static NV_STATUS get_gpu_info(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    UvmGpuInfo gpu_info = {{0}};
    char uuid_buffer[UVM_GPU_UUID_TEXT_BUFFER_LENGTH];

    status = uvm_rm_locked_call(nvUvmInterfaceGetGpuInfo(&gpu->uuid, &gpu_info));
    if (status != NV_OK)
        return status;

    gpu->architecture = gpu_info.gpuArch;
    gpu->implementation = gpu_info.gpuImplementation;

    gpu->host_class = gpu_info.hostClass;
    gpu->ce_class = gpu_info.ceClass;
    gpu->fault_buffer_class = gpu_info.faultBufferClass;
    gpu->access_counter_buffer_class = gpu_info.accessCounterBufferClass;
    gpu->max_subcontexts = gpu_info.maxSubctxCount;
    gpu->gpc_count = gpu_info.gpcCount;
    gpu->tpc_count = gpu_info.tpcCount;
    gpu->max_tpc_per_gpc = gpu_info.maxTpcPerGpc;

    gpu->sli_enabled = (gpu_info.subdeviceCount > 1);

    gpu->is_simulated = gpu_info.isSimulated;

    format_uuid_to_buffer(uuid_buffer, sizeof(uuid_buffer), &gpu->uuid);
    snprintf(gpu->name, sizeof(gpu->name), "ID %u: %s: %s", gpu->id, gpu_info.name, uuid_buffer);

    return status;
}

static NV_STATUS get_gpu_caps(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    UvmGpuCaps gpu_caps = {0};
    UvmGpuFbInfo fb_info = {0};

    status = uvm_rm_locked_call(nvUvmInterfaceQueryCaps(gpu->rm_address_space, &gpu_caps));
    if (status != NV_OK)
        return status;

    status = uvm_rm_locked_call(nvUvmInterfaceGetFbInfo(gpu->rm_address_space, &fb_info));
    if (status != NV_OK)
        return status;

    gpu->vidmem_size = ((NvU64)fb_info.heapSize + fb_info.reservedHeapSize) * 1024;
    gpu->vidmem_max_allocatable_address = fb_info.maxAllocatableAddress;

    memcpy(gpu->ce_caps, gpu_caps.copyEngineCaps, sizeof(gpu->ce_caps));

    gpu->ecc.enabled = gpu_caps.bEccEnabled;
    if (gpu->ecc.enabled) {
        gpu->ecc.hw_interrupt_tree_location = (volatile NvU32*)((char*)gpu_caps.eccReadLocation + gpu_caps.eccOffset);
        UVM_ASSERT(gpu->ecc.hw_interrupt_tree_location != NULL);
        gpu->ecc.mask = gpu_caps.eccMask;
        UVM_ASSERT(gpu->ecc.mask != 0);

        gpu->ecc.error_notifier = gpu_caps.eccErrorNotifier;
        UVM_ASSERT(gpu->ecc.error_notifier != NULL);
    }

    if (gpu_caps.sysmemLink == UVM_PEER_LINK_TYPE_PCIE)
        gpu->sysmem_link = UVM_GPU_LINK_PCIE;
    else if (gpu_caps.sysmemLink == UVM_PEER_LINK_TYPE_NVLINK_1)
        gpu->sysmem_link = UVM_GPU_LINK_NVLINK_1;
    else if (gpu_caps.sysmemLink == UVM_PEER_LINK_TYPE_NVLINK_2)
        gpu->sysmem_link = UVM_GPU_LINK_NVLINK_2;
    else
        UVM_ASSERT(0);












    if (gpu_caps.numaEnabled) {
        gpu->numa_info.enabled = true;
        gpu->numa_info.node_id = gpu_caps.numaNodeId;
        gpu->numa_info.system_memory_window_start = gpu_caps.systemMemoryWindowStart;
        gpu->numa_info.system_memory_window_end = gpu_caps.systemMemoryWindowStart + gpu_caps.systemMemoryWindowSize - 1;
    }
    else if (uvm8_ats_mode) {
        pr_info("UVM ATS is enabled but GPU %s does not have NUMA support\n", gpu->name);
        return NV_ERR_NOT_SUPPORTED;
    }

    return NV_OK;
}

static bool gpu_supports_uvm(uvm_gpu_t *gpu)
{
    // TODO: Bug 1757136: Add Linux SLI support. Until then, explicitly disable
    //       UVM on SLI.
    return !gpu->sli_enabled && gpu->architecture >= NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GK100;
}

bool uvm_gpu_can_address(uvm_gpu_t *gpu, NvU64 addr)
{
    NvU64 max_va;

    // Watch out for calling this too early in init
    UVM_ASSERT(gpu->address_space_tree.hal);
    UVM_ASSERT(gpu->address_space_tree.hal->num_va_bits() < 64);
    max_va = 1ULL << gpu->address_space_tree.hal->num_va_bits();

    // Despite not supporting a full 64-bit VA space, Pascal+ GPUs are capable
    // of accessing kernel pointers in various modes by applying the same upper-
    // bit checks that x86, ARM, and and Power processors do. We don't have an
    // immediate use case for that so we'll just let the below check fail if
    // addr falls in the upper bits which belong to kernel space.
    return addr < max_va;
}

static void
gpu_info_print_common(uvm_gpu_t *gpu, struct seq_file *s)
{
    NvU32 i;
    NvU64 num_pages_in;
    NvU64 num_pages_out;

    UVM_SEQ_OR_DBG_PRINT(s, "GPU %s\n", gpu->name);
    UVM_SEQ_OR_DBG_PRINT(s, "retained_count                         %llu\n",
                         uvm_gpu_retained_count(gpu));
    UVM_SEQ_OR_DBG_PRINT(s, "ecc                                    %s\n",
                         gpu->ecc.enabled ? "enabled" : "disabled");

    if (!uvm_procfs_is_debug_enabled())
        return;

    UVM_SEQ_OR_DBG_PRINT(s, "architecture                           0x%X\n", gpu->architecture);
    UVM_SEQ_OR_DBG_PRINT(s, "gpcs                                   %u\n", gpu->gpc_count);
    UVM_SEQ_OR_DBG_PRINT(s, "tpcs                                   %u\n", gpu->tpc_count);
    UVM_SEQ_OR_DBG_PRINT(s, "max_tpc_per_gpc                        %u\n", gpu->max_tpc_per_gpc);
    UVM_SEQ_OR_DBG_PRINT(s, "implementation                         0x%X\n", gpu->implementation);
    UVM_SEQ_OR_DBG_PRINT(s, "host_class                             0x%X\n", gpu->host_class);
    UVM_SEQ_OR_DBG_PRINT(s, "ce_class                               0x%X\n", gpu->ce_class);
    UVM_SEQ_OR_DBG_PRINT(s, "fault_buffer_class                     0x%X\n", gpu->fault_buffer_class);
    UVM_SEQ_OR_DBG_PRINT(s, "big_page_size                          %u\n", gpu->big_page.internal_size);
    UVM_SEQ_OR_DBG_PRINT(s, "big_page_swizzling                     %u\n", gpu->big_page.swizzling ? 1 : 0);
    UVM_SEQ_OR_DBG_PRINT(s, "rm_va_base                             0x%llx\n", gpu->rm_va_base);
    UVM_SEQ_OR_DBG_PRINT(s, "rm_va_size                             0x%llx\n", gpu->rm_va_size);
    UVM_SEQ_OR_DBG_PRINT(s, "vidmem_size                            %llu (%llu MBs)\n",
                         gpu->vidmem_size,
                         gpu->vidmem_size / (1024 * 1024));
    UVM_SEQ_OR_DBG_PRINT(s, "vidmem_max_allocatable                 0x%llx (%llu MBs)\n",
                         gpu->vidmem_max_allocatable_address,
                         gpu->vidmem_max_allocatable_address / (1024 * 1024));

    if (gpu->numa_info.enabled) {
        NvU64 window_size = gpu->numa_info.system_memory_window_end + 1 - gpu->numa_info.system_memory_window_start;
        UVM_SEQ_OR_DBG_PRINT(s, "numa_node_id                           %u\n", gpu->numa_info.node_id);
        UVM_SEQ_OR_DBG_PRINT(s, "system_memory_window_start             0x%llx\n",
                             gpu->numa_info.system_memory_window_start);
        UVM_SEQ_OR_DBG_PRINT(s, "system_memory_window_end               0x%llx\n",
                             gpu->numa_info.system_memory_window_end);
        UVM_SEQ_OR_DBG_PRINT(s, "system_memory_window_size              0x%llx (%llu MBs)\n",
                             window_size,
                             window_size / (1024 * 1024));
    }

    UVM_SEQ_OR_DBG_PRINT(s, "interrupts                             %llu\n", gpu->isr.interrupt_count);

    if (gpu->isr.replayable_faults.handling) {
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_bh                   %llu\n",
                             gpu->isr.replayable_faults.bottom_half_count);
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_buffer_entries       %u\n",
                             gpu->fault_buffer_info.replayable.max_faults);
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_cached_get           %u\n",
                             gpu->fault_buffer_info.replayable.cached_get);
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_cached_put           %u\n",
                             gpu->fault_buffer_info.replayable.cached_put);
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_get                  %u\n",
                             gpu->fault_buffer_hal->read_get(gpu));
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_put                  %u\n",
                             gpu->fault_buffer_hal->read_put(gpu));
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_fault_batch_size     %u\n",
                             gpu->fault_buffer_info.max_batch_size);
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_replay_policy        %s\n",
                             uvm_perf_fault_replay_policy_string(gpu->fault_buffer_info.replayable.replay_policy));
        UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults_num_faults           %llu\n",
                             gpu->stats.num_replayable_faults);
    }
    if (gpu->isr.non_replayable_faults.handling) {
        UVM_SEQ_OR_DBG_PRINT(s, "non_replayable_faults_bh               %llu\n",
                             gpu->isr.non_replayable_faults.bottom_half_count);
        UVM_SEQ_OR_DBG_PRINT(s, "non_replayable_faults_buffer_entries   %u\n",
                             gpu->fault_buffer_info.non_replayable.max_faults);
        UVM_SEQ_OR_DBG_PRINT(s, "non_replayable_faults_num_faults       %llu\n",
                             gpu->stats.num_non_replayable_faults);
    }

    if (gpu->isr.access_counters.handling) {
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_bh                     %llu\n",
                             gpu->isr.access_counters.bottom_half_count);
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_buffer_entries         %u\n",
                             gpu->access_counter_buffer_info.max_notifications);
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_cached_get             %u\n",
                             gpu->access_counter_buffer_info.cached_get);
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_cached_put             %u\n",
                             gpu->access_counter_buffer_info.cached_put);
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_get                    %u\n",
                             UVM_READ_ONCE(*gpu->access_counter_buffer_info.rm_info.pAccessCntrBufferGet));
        UVM_SEQ_OR_DBG_PRINT(s, "access_counters_put                    %u\n",
                             UVM_READ_ONCE(*gpu->access_counter_buffer_info.rm_info.pAccessCntrBufferPut));
    }

    num_pages_out = atomic64_read(&gpu->stats.num_pages_out);
    num_pages_in = atomic64_read(&gpu->stats.num_pages_in);

    UVM_SEQ_OR_DBG_PRINT(s, "migrated_pages_in                      %llu (%llu MB)\n", num_pages_in,
                         (num_pages_in * (NvU64)PAGE_SIZE) / (1024u * 1024u));
    UVM_SEQ_OR_DBG_PRINT(s, "migrated_pages_out                     %llu (%llu MB)\n", num_pages_out,
                         (num_pages_out * (NvU64)PAGE_SIZE) / (1024u * 1024u));

    UVM_SEQ_OR_DBG_PRINT(s, "supported_ces:\n");
    for (i = 0; i < UVM_COPY_ENGINE_COUNT_MAX; ++i) {
        if (!gpu->ce_caps[i].supported)
            continue;
        UVM_SEQ_OR_DBG_PRINT(s, " ce %u grce %u shared %u sysmem read %u sysmem write %u"
                                " sysmem %u nvlink p2p %u p2p %u\n",
                i,
                gpu->ce_caps[i].grce,
                gpu->ce_caps[i].shared,
                gpu->ce_caps[i].sysmemRead,
                gpu->ce_caps[i].sysmemWrite,
                gpu->ce_caps[i].sysmem,
                gpu->ce_caps[i].nvlinkP2p,
                gpu->ce_caps[i].p2p);
    }
}

static void
gpu_fault_stats_print_common(uvm_gpu_t *gpu, struct seq_file *s)
{
    NvU64 num_pages_in;
    NvU64 num_pages_out;

    if (!uvm_procfs_is_debug_enabled())
        return;

    num_pages_out = atomic64_read(&gpu->fault_buffer_info.replayable.stats.num_pages_out);
    num_pages_in = atomic64_read(&gpu->fault_buffer_info.replayable.stats.num_pages_in);

    UVM_SEQ_OR_DBG_PRINT(s, "replayable_faults      %llu\n", gpu->stats.num_replayable_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "duplicates             %llu\n", gpu->fault_buffer_info.replayable.stats.num_duplicate_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "faults_by_access_type:\n");
    UVM_SEQ_OR_DBG_PRINT(s, "  prefetch             %llu\n", gpu->fault_buffer_info.replayable.stats.num_prefetch_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  read                 %llu\n", gpu->fault_buffer_info.replayable.stats.num_read_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  write                %llu\n", gpu->fault_buffer_info.replayable.stats.num_write_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  atomic               %llu\n", gpu->fault_buffer_info.replayable.stats.num_atomic_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "migrations:\n");
    UVM_SEQ_OR_DBG_PRINT(s, "  num_pages_in         %llu (%llu MB)\n", num_pages_in,
                         (num_pages_in * (NvU64)PAGE_SIZE) / (1024u * 1024u));
    UVM_SEQ_OR_DBG_PRINT(s, "  num_pages_out        %llu (%llu MB)\n", num_pages_out,
                         (num_pages_out * (NvU64)PAGE_SIZE) / (1024u * 1024u));
    UVM_SEQ_OR_DBG_PRINT(s, "replays:\n");
    UVM_SEQ_OR_DBG_PRINT(s, "  start                %llu\n", gpu->fault_buffer_info.replayable.stats.num_replays);
    UVM_SEQ_OR_DBG_PRINT(s, "  start_ack_all        %llu\n", gpu->fault_buffer_info.replayable.stats.num_replays_ack_all);
    UVM_SEQ_OR_DBG_PRINT(s, "non_replayable_faults  %llu\n", gpu->stats.num_non_replayable_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "faults_by_access_type:\n");
    UVM_SEQ_OR_DBG_PRINT(s, "  read                 %llu\n", gpu->fault_buffer_info.non_replayable.stats.num_read_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  write                %llu\n", gpu->fault_buffer_info.non_replayable.stats.num_write_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  atomic               %llu\n", gpu->fault_buffer_info.non_replayable.stats.num_atomic_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "faults_by_addressing:\n");
    UVM_SEQ_OR_DBG_PRINT(s, "  virtual              %llu\n",
                         gpu->stats.num_non_replayable_faults - gpu->fault_buffer_info.non_replayable.stats.num_physical_faults);
    UVM_SEQ_OR_DBG_PRINT(s, "  physical             %llu\n", gpu->fault_buffer_info.non_replayable.stats.num_physical_faults);
}

void uvm_gpu_print(uvm_gpu_t *gpu)
{
    gpu_info_print_common(gpu, NULL);
}

static int
nv_procfs_read_gpu_info(struct seq_file *s, void *v)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)s->private;
    gpu_info_print_common(gpu, s);
    return 0;
}

static int
nv_procfs_read_gpu_fault_stats(struct seq_file *s, void *v)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)s->private;
    gpu_fault_stats_print_common(gpu, s);
    return 0;
}

NV_DEFINE_PROCFS_SINGLE_FILE(gpu_info);
NV_DEFINE_PROCFS_SINGLE_FILE(gpu_fault_stats);

static NV_STATUS init_procfs_dirs(uvm_gpu_t *gpu)
{
    // This needs to hold a gpu_id_t in decimal
    char gpu_dir_name[16];

    // This needs to hold a GPU UUID
    char symlink_name[UVM_GPU_UUID_TEXT_BUFFER_LENGTH];

    struct proc_dir_entry *gpu_base_dir_entry;
    if (!uvm_procfs_is_enabled())
        return NV_OK;

    gpu_base_dir_entry = uvm_procfs_get_gpu_base_dir();

    snprintf(gpu_dir_name, sizeof(gpu_dir_name), "%u", gpu->id);
    gpu->procfs.dir = NV_CREATE_PROC_DIR(gpu_dir_name, gpu_base_dir_entry);
    if (gpu->procfs.dir == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    // Create a symlink from UVM GPU UUID (UVM-GPU-...) to the UVM GPU ID
    format_uuid_to_buffer(symlink_name, sizeof(symlink_name), &gpu->uuid);
    gpu->procfs.dir_uuid_symlink = proc_symlink(symlink_name, gpu_base_dir_entry, gpu_dir_name);
    if (gpu->procfs.dir_uuid_symlink == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}

static void deinit_procfs_dirs(uvm_gpu_t *gpu)
{
    uvm_procfs_destroy_entry(gpu->procfs.dir_uuid_symlink);
    uvm_procfs_destroy_entry(gpu->procfs.dir);
}

static NV_STATUS init_procfs_files(uvm_gpu_t *gpu)
{
    gpu->procfs.info_file = NV_CREATE_PROC_FILE("info", gpu->procfs.dir, gpu_info, (void *)gpu);
    if (gpu->procfs.info_file == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    gpu->procfs.fault_stats_file = NV_CREATE_PROC_FILE("fault_stats", gpu->procfs.dir, gpu_fault_stats, (void *)gpu);
    if (gpu->procfs.fault_stats_file == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}

static void deinit_procfs_files(uvm_gpu_t *gpu)
{
    uvm_procfs_destroy_entry(gpu->procfs.info_file);
    uvm_procfs_destroy_entry(gpu->procfs.fault_stats_file);
}

static NV_STATUS init_semaphore_pool(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    uvm_gpu_t *other_gpu;

    status = uvm_gpu_semaphore_pool_create(gpu, &gpu->semaphore_pool);
    if (status != NV_OK)
        return status;

    for_each_global_gpu(other_gpu) {
        if (other_gpu == gpu)
            continue;
        status = uvm_gpu_semaphore_pool_map_gpu(other_gpu->semaphore_pool, gpu);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static void deinit_semaphore_pool(uvm_gpu_t *gpu)
{
    uvm_gpu_t *other_gpu;

    for_each_global_gpu(other_gpu) {
        if (other_gpu == gpu)
            continue;
        uvm_gpu_semaphore_pool_unmap_gpu(other_gpu->semaphore_pool, gpu);
    }

    uvm_gpu_semaphore_pool_destroy(gpu->semaphore_pool);
}

// Allocates a uvm_gpu_t*, assigns a gpu->id to it, but leaves all other initialization up to
// the caller.
static NV_STATUS alloc_gpu(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out)
{
    uvm_gpu_t *gpu;
    uvm_gpu_id_t id;
    uvm_gpu_id_t new_gpu_id;
    bool found_a_slot = false;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    // Find an unused slot:
    for_each_gpu_id(id) {
        gpu = uvm_gpu_get(id);

        if (gpu == NULL) {
            new_gpu_id = id;
            found_a_slot = true;
            break;
        }
    }

    if (!found_a_slot)
        return NV_ERR_INSUFFICIENT_RESOURCES;

    gpu = uvm_kvmalloc_zero(sizeof(*gpu));
    if (!gpu)
        return NV_ERR_NO_MEMORY;

    gpu->id = new_gpu_id;

    // Initialize enough of the gpu struct for remove_gpu to be called
    gpu->magic = UVM_GPU_MAGIC_VALUE;
    uvm_processor_uuid_copy(&gpu->uuid, gpu_uuid);
    uvm_mutex_init(&gpu->isr.replayable_faults.service_lock, UVM_LOCK_ORDER_ISR);
    uvm_mutex_init(&gpu->isr.non_replayable_faults.service_lock, UVM_LOCK_ORDER_ISR);
    uvm_mutex_init(&gpu->isr.access_counters.service_lock, UVM_LOCK_ORDER_ISR);
    uvm_spin_lock_init(&gpu->peer_gpus_lock, UVM_LOCK_ORDER_LEAF);
    uvm_spin_lock_irqsave_init(&gpu->isr.interrupts_lock, UVM_LOCK_ORDER_LEAF);
    uvm_spin_lock_init(&gpu->instance_ptr_table_lock, UVM_LOCK_ORDER_LEAF);
    uvm_init_radix_tree_preloadable(&gpu->instance_ptr_table);
    uvm_init_radix_tree_preloadable(&gpu->tsg_table);
    uvm_mutex_init(&gpu->big_page.staging.lock, UVM_LOCK_ORDER_SWIZZLE_STAGING);
    uvm_tracker_init(&gpu->big_page.staging.tracker);

    nv_kref_init(&gpu->gpu_kref);

    *gpu_out = gpu;

    return NV_OK;
}

static NV_STATUS configure_address_space(uvm_gpu_t *gpu)
{
    NV_STATUS status;
    NvU32 num_entries;
    NvU64 va_size;
    NvU64 va_per_entry;

    status = uvm_page_tree_init(gpu, gpu->big_page.internal_size, UVM_APERTURE_DEFAULT, &gpu->address_space_tree);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Initializing the page tree failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        return status;
    }

    num_entries = uvm_mmu_page_tree_entries(&gpu->address_space_tree, 0, UVM_PAGE_SIZE_AGNOSTIC);

    UVM_ASSERT(gpu->address_space_tree.hal->num_va_bits() < 64);
    va_size = 1ull << gpu->address_space_tree.hal->num_va_bits();
    va_per_entry = va_size / num_entries;

    // Make sure that RM's part of the VA is aligned to the VA covered by a
    // single top level PDE.
    UVM_ASSERT_MSG(gpu->rm_va_base % va_per_entry == 0,
            "va_base 0x%llx va_per_entry 0x%llx\n", gpu->rm_va_base, va_per_entry);
    UVM_ASSERT_MSG(gpu->rm_va_size % va_per_entry == 0,
            "va_size 0x%llx va_per_entry 0x%llx\n", gpu->rm_va_size, va_per_entry);

    status = uvm_rm_locked_call(nvUvmInterfaceSetPageDirectory(gpu->rm_address_space,
            uvm_page_tree_pdb(&gpu->address_space_tree)->addr.address, num_entries,
            uvm_page_tree_pdb(&gpu->address_space_tree)->addr.aperture == UVM_APERTURE_VID));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceSetPageDirectory() failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        return status;
    }
    gpu->rm_address_space_moved_to_page_tree = true;

    return NV_OK;
}

static void deconfigure_address_space(uvm_gpu_t *gpu)
{
    if (gpu->rm_address_space_moved_to_page_tree)
        uvm_rm_locked_call_void(nvUvmInterfaceUnsetPageDirectory(gpu->rm_address_space));

    if (gpu->address_space_tree.root)
        uvm_page_tree_deinit(&gpu->address_space_tree);
}

static NV_STATUS init_big_pages(uvm_gpu_t *gpu)
{
    NV_STATUS status;

    if (!gpu->big_page.swizzling)
        return NV_OK;

    status = uvm_mmu_create_big_page_identity_mappings(gpu);
    if (status != NV_OK)
        return status;

    status = uvm_pmm_gpu_alloc_kernel(&gpu->pmm,
                                      1,
                                      gpu->big_page.internal_size,
                                      UVM_PMM_ALLOC_FLAGS_NONE,
                                      &gpu->big_page.staging.chunk,
                                      &gpu->big_page.staging.tracker);
    if (status != NV_OK)
        return status;

    return NV_OK;
}

static void deinit_big_pages(uvm_gpu_t *gpu)
{
    if (!gpu->big_page.swizzling)
        return;

    (void)uvm_tracker_wait_deinit(&gpu->big_page.staging.tracker);
    uvm_pmm_gpu_free(&gpu->pmm, gpu->big_page.staging.chunk, NULL);
    uvm_mmu_destroy_big_page_identity_mappings(gpu);
}

// Add a new gpu and register it with RM
static NV_STATUS add_gpu(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out)
{
    NV_STATUS status;
    uvm_gpu_t *gpu;
    UvmGpuPlatformInfo gpu_platform_info = {0};
    UvmGpuAddressSpaceInfo gpu_address_space_info = {0};

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    status = uvm_rm_locked_call(nvUvmInterfaceRegisterGpu(gpu_uuid, &gpu_platform_info));
    if (status != NV_OK)
        return status;

    status = alloc_gpu(gpu_uuid, &gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to allocate a GPU object: %s\n", nvstatusToString(status));
        // Handle the clean up here as we didn't manage to get a uvm_gpu_t and cannot use remove_gpu()
        uvm_rm_locked_call_void(nvUvmInterfaceUnregisterGpu(gpu_uuid));
        return status;
    }

    // After this point all error clean up should be handled by remove_gpu()

    gpu->pci_dev = gpu_platform_info.pci_dev;
#if defined(NVCPU_PPC64LE) && defined(NV_PNV_PCI_GET_NPU_DEV_PRESENT)
    gpu->npu_dev = pnv_pci_get_npu_dev(gpu->pci_dev, 0);
#endif
    gpu->dma_addressable_start = gpu_platform_info.dma_addressable_start;
    gpu->dma_addressable_limit = gpu_platform_info.dma_addressable_limit;

    status = get_gpu_info(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to get GPU info: %s\n", nvstatusToString(status));
        goto error;
    }

    if (gpu->is_simulated)
        ++g_uvm_global.num_simulated_devices;

    if (!gpu_supports_uvm(gpu)) {
        UVM_DBG_PRINT("Register of non-UVM-capable GPU attempted: GPU %s\n", gpu->name);
        status = NV_ERR_NOT_SUPPORTED;
        goto error;
    }

    // Initialize the per-GPU procfs dirs as early as possible so that other
    // parts of the driver can add files in them as part of their per-GPU init.
    status = init_procfs_dirs(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init procfs dirs: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_hal_init_gpu(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init GPU hal: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    gpu->arch_hal->init_properties(gpu);
    uvm_mmu_init_gpu_peer_addresses(gpu);

    status = uvm_rm_locked_call(nvUvmInterfaceAddressSpaceCreate(g_uvm_global.rm_session_handle,
                                                                 &gpu->uuid,
                                                                 gpu->rm_va_base,
                                                                 gpu->rm_va_size,
                                                                 &gpu->rm_address_space,
                                                                 &gpu_address_space_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Creating RM address space failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    gpu->big_page.internal_size = gpu_address_space_info.bigPageSize;

    status = get_gpu_caps(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to get GPU caps: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_gpu_check_ecc_error(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Initial ECC error check failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_pmm_gpu_init(gpu, &gpu->pmm);
    if (status != NV_OK) {
        UVM_ERR_PRINT("PMM initialization failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_pmm_sysmem_mappings_init(gpu, &gpu->pmm_sysmem_mappings);
    if (status != NV_OK) {
        UVM_ERR_PRINT("CPU PMM MMIO initialization failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = init_semaphore_pool(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to initialize the semaphore pool: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_channel_manager_create(gpu, &gpu->channel_manager);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to initialize the channel manager: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = configure_address_space(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to configure the GPU address space: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = init_big_pages(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init big pages: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = init_procfs_files(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init procfs files: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_gpu_init_isr(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init ISR: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = uvm_hmm_device_register(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to register HMM device: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    status = discover_nvlink_peers(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to discover NVLINK peers: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    atomic64_set(&gpu->retained_count, 1);
    uvm_processor_mask_set(&g_uvm_global.retained_gpus, gpu->id);

    // Add the GPU to the GPU table.
    uvm_spin_lock_irqsave(&g_uvm_global.gpu_table_lock);

    g_uvm_global.gpus[uvm_gpu_index(gpu->id)] = gpu;

    // Although locking correctness does not, at this early point (before the GPU is visible in
    // the table) strictly require holding the gpu_table_lock in order to read
    // gpu->isr.replayable_faults.handling, nor to enable page fault interrupts (this
    // could have been done earlier), it is best to do it here, in order to avoid an interrupt
    // storm. That way, we take advantage of the spinlock_irqsave side effect of turning off
    // local CPU interrupts, as part of holding the gpu_table_lock. That means that the local CPU
    // won't receive any of these interrupts, until the GPU is safely added to the table (where the
    // top half ISR can find it).
    //
    // As usual with spinlock_irqsave behavior, *other* CPUs can still handle these interrupts,
    // but the local CPU will not be slowed down (interrupted) by such handling, and can
    // quickly release the gpu_table_lock, thus unblocking any other CPU's top half (which
    // waits for the gpu_table_lock).
    if (gpu->isr.replayable_faults.handling)
        gpu->fault_buffer_hal->enable_replayable_faults(gpu);

    if (gpu->isr.access_counters.handling)
        gpu->access_counter_buffer_hal->enable_access_counter_notifications(gpu);

    uvm_spin_unlock_irqrestore(&g_uvm_global.gpu_table_lock);

    *gpu_out = gpu;

    return NV_OK;

error:
    remove_gpu(gpu);

    return status;
}

// Remove all references the given GPU has to other GPUs, since one of those
// other GPUs is getting removed. This involves waiting for any unfinished
// trackers contained by this GPU.
static void remove_gpus_from_gpu(uvm_gpu_t *gpu)
{
    NV_STATUS status;

    // Sync the replay tracker since it inherits dependencies from the VA block
    // trackers.
    if (gpu->isr.replayable_faults.handling) {
        uvm_gpu_replayable_faults_isr_lock(gpu);
        status = uvm_tracker_wait(&gpu->fault_buffer_info.replayable.replay_tracker);
        uvm_gpu_replayable_faults_isr_unlock(gpu);

        if (status != NV_OK)
            UVM_ASSERT(status == uvm_global_get_status());
    }

    // Sync the clear_faulted tracker since it inherits dependencies from the
    // VA block trackers, too.
    if (gpu->isr.non_replayable_faults.handling) {
        uvm_gpu_non_replayable_faults_isr_lock(gpu);
        status = uvm_tracker_wait(&gpu->fault_buffer_info.non_replayable.clear_faulted_tracker);
        uvm_gpu_non_replayable_faults_isr_unlock(gpu);

        if (status != NV_OK)
            UVM_ASSERT(status == uvm_global_get_status());
    }

    uvm_mutex_lock(&gpu->big_page.staging.lock);
    status = uvm_tracker_wait(&gpu->big_page.staging.tracker);
    uvm_mutex_unlock(&gpu->big_page.staging.lock);
    if (status != NV_OK)
        UVM_ASSERT(status == uvm_global_get_status());

    // Sync all trackers in PMM
    uvm_pmm_gpu_sync(&gpu->pmm);
}

// Remove a gpu and unregister it from RM
// Note that this is also used in most error paths in add_gpu()
static void remove_gpu(uvm_gpu_t *gpu)
{
    uvm_gpu_t *other_gpu;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT_MSG(uvm_gpu_retained_count(gpu) == 0, "gpu_id %u retained_count %llu\n",
                   gpu->id, uvm_gpu_retained_count(gpu));

    // All channels should have been removed before the retained count went to 0
    UVM_ASSERT(radix_tree_empty(&gpu->instance_ptr_table));
    UVM_ASSERT(radix_tree_empty(&gpu->tsg_table));

    // Remove the GPU from the table.
    uvm_spin_lock_irqsave(&g_uvm_global.gpu_table_lock);

    g_uvm_global.gpus[uvm_gpu_index(gpu->id)] = NULL;
    uvm_spin_unlock_irqrestore(&g_uvm_global.gpu_table_lock);

    uvm_processor_mask_clear(&g_uvm_global.retained_gpus, gpu->id);

    // Stop scheduling new bottom halves
    uvm_gpu_disable_isr(gpu);

    // Remove any pointers to this GPU from other GPUs' trackers.
    for_each_global_gpu(other_gpu) {
        UVM_ASSERT(other_gpu != gpu);
        remove_gpus_from_gpu(other_gpu);
    }

    destroy_nvlink_peers(gpu);

    uvm_hmm_device_unregister(gpu);

    // Return ownership to RM
    uvm_gpu_deinit_isr(gpu);

    deinit_procfs_files(gpu);

    deinit_big_pages(gpu);

    // Wait for any deferred frees and their associated trackers to be finished
    // before tearing down channels.
    uvm_pmm_gpu_sync(&gpu->pmm);

    uvm_channel_manager_destroy(gpu->channel_manager);

    // Deconfigure the address space only after destroying all the channels as
    // in case any of them hit fatal errors, RM will assert that they are not
    // idle during nvUvmInterfaceUnsetPageDirectory() and that's an unnecessary
    // pain during development.
    deconfigure_address_space(gpu);

    deinit_semaphore_pool(gpu);

    uvm_pmm_sysmem_mappings_deinit(&gpu->pmm_sysmem_mappings);

    uvm_pmm_gpu_deinit(&gpu->pmm);

    if (gpu->rm_address_space != 0)
        uvm_rm_locked_call_void(nvUvmInterfaceAddressSpaceDestroy(gpu->rm_address_space));

    UVM_ASSERT(atomic64_read(&gpu->mapped_cpu_pages_size) == 0);

    // After calling nvUvmInterfaceUnregisterGpu() the reference to pci_dev may
    // not be valid any more so clear it ahead of time.
    gpu->pci_dev = NULL;
    uvm_rm_locked_call_void(nvUvmInterfaceUnregisterGpu(&gpu->uuid));

    deinit_procfs_dirs(gpu);

    if (gpu->is_simulated)
        --g_uvm_global.num_simulated_devices;

    uvm_gpu_kref_put(gpu);
}

// Do not not call this directly. It is called by nv_kref_put, when the GPU's ref count drops
// to zero.
static void uvm_gpu_destroy(nv_kref_t *nv_kref)
{
    uvm_gpu_t *gpu = container_of(nv_kref, uvm_gpu_t, gpu_kref);

    UVM_ASSERT_MSG(uvm_gpu_retained_count(gpu) == 0, "gpu_id %u retained_count %llu\n",
                   gpu->id, uvm_gpu_retained_count(gpu));

    gpu->magic = 0;

    uvm_kvfree(gpu);
}

void uvm_gpu_kref_put(uvm_gpu_t *gpu)
{
    nv_kref_put(&gpu->gpu_kref, uvm_gpu_destroy);
}

static void update_stats_gpu_fault_instance(uvm_gpu_t *gpu,
                                            const uvm_fault_buffer_entry_t *fault_entry,
                                            bool is_duplicate)
{
    if (!fault_entry->is_replayable) {
        switch (fault_entry->fault_access_type)
        {
            case UVM_FAULT_ACCESS_TYPE_READ:
                ++gpu->fault_buffer_info.non_replayable.stats.num_read_faults;
                break;
            case UVM_FAULT_ACCESS_TYPE_WRITE:
                ++gpu->fault_buffer_info.non_replayable.stats.num_write_faults;
                break;
            case UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK:
            case UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG:
                ++gpu->fault_buffer_info.non_replayable.stats.num_atomic_faults;
                break;
            default:
                UVM_ASSERT_MSG(false, "Invalid access type for non-replayable faults\n");
                break;
        }

        if (!fault_entry->is_virtual)
            ++gpu->fault_buffer_info.non_replayable.stats.num_physical_faults;

        ++gpu->stats.num_non_replayable_faults;

        return;
    }

    UVM_ASSERT(fault_entry->is_virtual);

    switch (fault_entry->fault_access_type)
    {
        case UVM_FAULT_ACCESS_TYPE_PREFETCH:
            ++gpu->fault_buffer_info.replayable.stats.num_prefetch_faults;
            break;
        case UVM_FAULT_ACCESS_TYPE_READ:
            ++gpu->fault_buffer_info.replayable.stats.num_read_faults;
            break;
        case UVM_FAULT_ACCESS_TYPE_WRITE:
            ++gpu->fault_buffer_info.replayable.stats.num_write_faults;
            break;
        case UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK:
        case UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG:
            ++gpu->fault_buffer_info.replayable.stats.num_atomic_faults;
            break;
        default:
            break;
    }
    if (is_duplicate || fault_entry->filtered)
        ++gpu->fault_buffer_info.replayable.stats.num_duplicate_faults;

    ++gpu->stats.num_replayable_faults;
}

static void update_stats_fault_cb(uvm_perf_event_t event_id, uvm_perf_event_data_t *event_data)
{
    uvm_gpu_t *gpu;
    const uvm_fault_buffer_entry_t *fault_entry, *fault_instance;

    UVM_ASSERT(event_id == UVM_PERF_EVENT_FAULT);

    if (event_data->fault.proc_id == UVM_CPU_ID)
        return;

    // The reported fault entry must be the "representative" fault entry
    UVM_ASSERT(!event_data->fault.gpu.buffer_entry->filtered);

    gpu = uvm_gpu_get(event_data->fault.proc_id);

    fault_entry = event_data->fault.gpu.buffer_entry;

    // Update the stats using the representative fault entry and the rest of instances
    update_stats_gpu_fault_instance(gpu, fault_entry, event_data->fault.gpu.is_duplicate);

    list_for_each_entry(fault_instance, &fault_entry->merged_instances_list, merged_instances_list)
        update_stats_gpu_fault_instance(gpu, fault_instance, event_data->fault.gpu.is_duplicate);
}

static void update_stats_migration_cb(uvm_perf_event_t event_id, uvm_perf_event_data_t *event_data)
{
    uvm_gpu_t *gpu_dst = NULL;
    uvm_gpu_t *gpu_src = NULL;
    NvU64 pages;
    bool is_fault;

    UVM_ASSERT(event_id == UVM_PERF_EVENT_MIGRATION);

    if (event_data->migration.dst != UVM_CPU_ID)
        gpu_dst = uvm_gpu_get(event_data->migration.dst);

    if (event_data->migration.src != UVM_CPU_ID)
        gpu_src = uvm_gpu_get(event_data->migration.src);

    if (!gpu_dst && !gpu_src)
        return;

    // Page prefetching is also triggered by faults
    is_fault = event_data->migration.cause == UVM_MAKE_RESIDENT_CAUSE_FAULT ||
               event_data->migration.cause == UVM_MAKE_RESIDENT_CAUSE_PREFETCH;

    pages = event_data->migration.bytes / PAGE_SIZE;
    UVM_ASSERT(event_data->migration.bytes % PAGE_SIZE == 0);
    UVM_ASSERT(pages > 0);

    if (gpu_dst) {
        atomic64_add(pages, &gpu_dst->stats.num_pages_in);
        if (is_fault)
            atomic64_add(pages, &gpu_dst->fault_buffer_info.replayable.stats.num_pages_in);
    }
    if (gpu_src) {
        atomic64_add(pages, &gpu_src->stats.num_pages_out);
        if (is_fault)
            atomic64_add(pages, &gpu_src->fault_buffer_info.replayable.stats.num_pages_out);
    }
}

NV_STATUS uvm_gpu_init(void)
{
    NV_STATUS status;
    status = uvm_hal_init_table();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_hal_init_table() failed: %s\n", nvstatusToString(status));
        return status;
    }

    return NV_OK;
}

void uvm_gpu_exit(void)
{
    uvm_gpu_t *gpu;
    uvm_gpu_id_t id;

    for_each_gpu_id(id) {
        gpu = uvm_gpu_get(id);
        UVM_ASSERT_MSG(gpu == NULL, "GPU still present: %s\n", gpu->name);
    }

    // CPU should never be in the retained GPUs mask
    UVM_ASSERT(!uvm_processor_mask_test(&g_uvm_global.retained_gpus, UVM_CPU_ID));
}

NV_STATUS uvm_gpu_init_va_space(uvm_va_space_t *va_space)
{
    NV_STATUS status;

    if (uvm_procfs_is_debug_enabled()) {
        status = uvm_perf_register_event_callback(&va_space->perf_events,
                                                  UVM_PERF_EVENT_FAULT,
                                                  update_stats_fault_cb);
        if (status != NV_OK)
            return status;

        status = uvm_perf_register_event_callback(&va_space->perf_events,
                                                  UVM_PERF_EVENT_MIGRATION,
                                                  update_stats_migration_cb);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

uvm_gpu_t *uvm_gpu_get_by_uuid_locked(const NvProcessorUuid *gpu_uuid)
{
    uvm_gpu_id_t id;

    for_each_gpu_id(id) {
        uvm_gpu_t *gpu = uvm_gpu_get(id);
        if (gpu && uvm_processor_uuid_eq(&gpu->uuid, gpu_uuid))
            return gpu;
    }

    return NULL;
}

uvm_gpu_t *uvm_gpu_get_by_uuid(const NvProcessorUuid *gpu_uuid)
{
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    return uvm_gpu_get_by_uuid_locked(gpu_uuid);
}

NV_STATUS uvm_gpu_retain_by_uuid_locked(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(gpu_uuid);

    if (gpu == NULL)
        status = add_gpu(gpu_uuid, &gpu);
    else
        atomic64_inc(&gpu->retained_count);

    *gpu_out = gpu;

    return status;
}

NV_STATUS uvm_gpu_retain_by_uuid(const NvProcessorUuid *gpu_uuid, uvm_gpu_t **gpu_out)
{
    NV_STATUS status;
    uvm_mutex_lock(&g_uvm_global.global_lock);
    status = uvm_gpu_retain_by_uuid_locked(gpu_uuid, gpu_out);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

void uvm_gpu_retain(uvm_gpu_t *gpu)
{
    UVM_ASSERT(uvm_gpu_retained_count(gpu) > 0);
    atomic64_inc(&gpu->retained_count);
}

void uvm_gpu_release_locked(uvm_gpu_t *gpu)
{
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(uvm_gpu_retained_count(gpu) > 0);

    if (atomic64_dec_and_test(&gpu->retained_count))
        remove_gpu(gpu);
}

void uvm_gpu_release(uvm_gpu_t *gpu)
{
    uvm_mutex_lock(&g_uvm_global.global_lock);
    uvm_gpu_release_locked(gpu);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
}

void uvm_gpu_retain_mask(const uvm_processor_mask_t *mask)
{
    uvm_gpu_t *gpu;
    for_each_gpu_in_mask(gpu, mask)
        uvm_gpu_retain(gpu);
}

void uvm_gpu_release_mask_locked(const uvm_processor_mask_t *mask)
{
    uvm_gpu_id_t gpu_id;
    // Do not use for_each_gpu_in_mask as it reads the GPU state and it might get destroyed
    for_each_gpu_id_in_mask(gpu_id, mask)
        uvm_gpu_release_locked(uvm_gpu_get(gpu_id));
}

void uvm_gpu_release_mask(const uvm_processor_mask_t *mask)
{
    uvm_mutex_lock(&g_uvm_global.global_lock);
    uvm_gpu_release_mask_locked(mask);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
}

// Note: Peer table is an upper triangular matrix packed into a flat array.
// This function converts an index of 2D array of size [N x N] into an index
// of upper triangular array of size [((N - 1) * ((N - 1) + 1)) / 2] which does not
// include diagonal elements.
NvU32 uvm_gpu_peer_table_index(uvm_gpu_id_t gpu_id_1, uvm_gpu_id_t gpu_id_2)
{
    NvU32 square_index, triangular_index;

    UVM_ASSERT(gpu_id_1 != gpu_id_2);

    // Calculate an index of 2D array by re-ordering indices to always point to the same entry.
    square_index = (uvm_gpu_index(min(gpu_id_1, gpu_id_2)) * UVM_MAX_GPUS) + uvm_gpu_index(max(gpu_id_1, gpu_id_2));

    // Calculate and subtract number of lower triangular matrix elements till the current row
    // (which includes diagonal elements) to get the correct index in an upper triangular matrix.
    // Note: As gpu_id can be [1, N), no extra logic is needed to calculate diagonal elements.
    triangular_index = square_index - SUM_FROM_0_TO_N(min(gpu_id_1, gpu_id_2));

    UVM_ASSERT(triangular_index < UVM_MAX_UNIQUE_GPU_PAIRS);

    return triangular_index;
}

static NV_STATUS service_interrupts(uvm_gpu_t *gpu)
{
    // Asking RM to service interrupts from top half interrupt handler would
    // very likely deadlock.
    UVM_ASSERT(!in_interrupt());

    return uvm_rm_locked_call(nvUvmInterfaceServiceDeviceInterruptsRM(gpu->rm_address_space));
}

NV_STATUS uvm_gpu_check_ecc_error_no_rm(uvm_gpu_t *gpu)
{
    // We may need to call service_interrupts() which cannot be done in the top
    // half interrupt handler so assert here as well to catch improper use as
    // early as possible.
    UVM_ASSERT(!in_interrupt());

    if (!gpu->ecc.enabled)
        return NV_OK;

    // Early out If a global ECC error is already set to not spam the logs with
    // the same error.
    if (uvm_global_get_status() == NV_ERR_ECC_ERROR)
        return NV_ERR_ECC_ERROR;

    if (*gpu->ecc.error_notifier) {
        UVM_ERR_PRINT("ECC error encountered, GPU %s\n", gpu->name);
        uvm_global_set_fatal_error(NV_ERR_ECC_ERROR);
        return NV_ERR_ECC_ERROR;
    }

    // RM hasn't seen an ECC error yet, check whether there is a pending
    // interrupt that might indicate one. We might get false positives because
    // the interrupt bits we read are not ECC-specific. They're just the
    // top-level bits for any interrupt on all engines which support ECC. On
    // Pascal for example, RM returns us a mask with the bits for GR, L2, and
    // FB, because any of those might raise an ECC interrupt. So if they're set
    // we have to ask RM to check whether it was really an ECC error (and a
    // double-bit ECC error at that), in which case it sets the notifier.
    if ((*gpu->ecc.hw_interrupt_tree_location & gpu->ecc.mask) == 0) {
        // No pending interrupts.
        return NV_OK;
    }

    // An interrupt that might mean an ECC error needs to be serviced, signal
    // that to the caller.
    return NV_WARN_MORE_PROCESSING_REQUIRED;
}

NV_STATUS uvm_gpu_check_ecc_error(uvm_gpu_t *gpu)
{
    NV_STATUS status = uvm_gpu_check_ecc_error_no_rm(gpu);

    if (status == NV_OK || status != NV_WARN_MORE_PROCESSING_REQUIRED)
        return status;

    // An interrupt that might mean an ECC error needs to be serviced.
    UVM_ASSERT(status == NV_WARN_MORE_PROCESSING_REQUIRED);

    status = service_interrupts(gpu);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Servicing interrupts failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        return status;
    }

    // After servicing interrupts the ECC error notifier should be current.
    if (*gpu->ecc.error_notifier) {
        UVM_ERR_PRINT("ECC error encountered, GPU %s\n", gpu->name);
        uvm_global_set_fatal_error(NV_ERR_ECC_ERROR);
        return NV_ERR_ECC_ERROR;
    }

    return NV_OK;
}

NV_STATUS uvm_gpu_check_ecc_error_mask(uvm_processor_mask_t *gpus)
{
    uvm_gpu_t *gpu;
    for_each_gpu_in_mask(gpu, gpus) {
        NV_STATUS status = uvm_gpu_check_ecc_error(gpu);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS get_p2p_caps(uvm_gpu_t *gpu_1,
                              uvm_gpu_t *gpu_2,
                              UvmGpuP2PCapsParams *p2p_caps_params)
{
    NV_STATUS status;
    uvmGpuAddressSpaceHandle rm_aspace1, rm_aspace2;

    rm_aspace1 = gpu_1->id < gpu_2->id ? gpu_1->rm_address_space : gpu_2->rm_address_space;
    rm_aspace2 = gpu_1->id > gpu_2->id ? gpu_1->rm_address_space : gpu_2->rm_address_space;

    memset(p2p_caps_params, 0, sizeof(*p2p_caps_params));
    status = uvm_rm_locked_call(nvUvmInterfaceGetP2PCaps(rm_aspace1, rm_aspace2, p2p_caps_params));
    if (status != NV_OK) {
        UVM_ERR_PRINT("failed to query P2P caps with error: %s, for GPU1:%s and GPU2:%s \n",
                       nvstatusToString(status),
                       gpu_1->name,
                       gpu_2->name);
        return status;
    }

    return NV_OK;
}

static NV_STATUS create_p2p_object(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2, NvHandle *p2p_handle)
{
    NV_STATUS status;
    uvmGpuAddressSpaceHandle rm_aspace1, rm_aspace2;

    rm_aspace1 = gpu_1->id < gpu_2->id ? gpu_1->rm_address_space : gpu_2->rm_address_space;
    rm_aspace2 = gpu_1->id > gpu_2->id ? gpu_1->rm_address_space : gpu_2->rm_address_space;

    *p2p_handle = 0;

    status = uvm_rm_locked_call(nvUvmInterfaceP2pObjectCreate(rm_aspace1, rm_aspace2, p2p_handle));
    if (status == NV_OK)
        UVM_ASSERT(p2p_handle);

    return status;
}

static NV_STATUS init_peer_access(uvm_gpu_t *gpu_1,
                                  uvm_gpu_t *gpu_2,
                                  const UvmGpuP2PCapsParams *p2p_caps_params,
                                  uvm_gpu_peer_t *peer_caps)
{
    NV_STATUS status;

    // check for peer-to-peer compatibility (PCI-E or NvLink).
    if (p2p_caps_params->p2pLink == UVM_PEER_LINK_TYPE_PCIE)
        peer_caps->link_type = UVM_GPU_LINK_PCIE;
    else if (p2p_caps_params->p2pLink == UVM_PEER_LINK_TYPE_NVLINK_1)
        peer_caps->link_type = UVM_GPU_LINK_NVLINK_1;
    else if (p2p_caps_params->p2pLink == UVM_PEER_LINK_TYPE_NVLINK_2)
        peer_caps->link_type = UVM_GPU_LINK_NVLINK_2;
    else
        return NV_ERR_NOT_SUPPORTED;

    // Initialize peer ids and establish peer mappings
    peer_caps->is_direct_peer = (p2p_caps_params->directAccess == NV_TRUE);

    if (peer_caps->is_direct_peer) {
        // Peer id from min(gpu_id_1, gpu_id_2) -> max(gpu_id_1, gpu_id_2)
        peer_caps->peer_ids[0] = p2p_caps_params->peerIds[0];

        // Peer id from max(gpu_id_1, gpu_id_2) -> min(gpu_id_1, gpu_id_2)
        peer_caps->peer_ids[1] = p2p_caps_params->peerIds[1];

        // Establish peer mappings from each GPU to the other. Indirect peers
        // do not require identity mappings since they use sysmem aperture to
        // communicate.
        status = uvm_mmu_create_peer_identity_mappings(gpu_1, gpu_2);
        if (status != NV_OK)
            return status;

        status = uvm_mmu_create_peer_identity_mappings(gpu_2, gpu_1);
        if (status != NV_OK)
            return status;

        uvm_spin_lock(&gpu_1->peer_gpus_lock);

        if (gpu_1->id < gpu_2->id)
            gpu_1->peer_gpus[peer_caps->peer_ids[0]] = gpu_2;
        else
            gpu_1->peer_gpus[peer_caps->peer_ids[1]] = gpu_2;

        uvm_spin_unlock(&gpu_1->peer_gpus_lock);
        uvm_spin_lock(&gpu_2->peer_gpus_lock);

        if (gpu_1->id < gpu_2->id)
            gpu_2->peer_gpus[peer_caps->peer_ids[1]] = gpu_1;
        else
            gpu_2->peer_gpus[peer_caps->peer_ids[0]] = gpu_1;

        uvm_spin_unlock(&gpu_2->peer_gpus_lock);

        if (peer_caps->link_type >= UVM_GPU_LINK_NVLINK_1) {
            const NvU32 ce_index_1 = p2p_caps_params->optimalNvlinkWriteCEs[0];
            const NvU32 ce_index_2 = p2p_caps_params->optimalNvlinkWriteCEs[1];

            UVM_ASSERT(ce_index_1 < UVM_COPY_ENGINE_COUNT_MAX);
            UVM_ASSERT(ce_index_2 < UVM_COPY_ENGINE_COUNT_MAX);

            peer_caps->optimal_write_ce[0] = ce_index_1;
            peer_caps->optimal_write_ce[1] = ce_index_2;
        }
    }
    else {
        UVM_ASSERT(gpu_1->numa_info.enabled);
        UVM_ASSERT(gpu_2->numa_info.enabled);

        status = uvm_pmm_gpu_indirect_peer_init(&gpu_1->pmm, gpu_2);
        if (status != NV_OK)
            return status;

        status = uvm_pmm_gpu_indirect_peer_init(&gpu_2->pmm, gpu_1);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS enable_pcie_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2)
{
    NV_STATUS status = NV_OK;
    UvmGpuP2PCapsParams p2p_caps_params;
    uvm_gpu_peer_t *peer_caps;
    NvHandle p2p_handle;

    UVM_ASSERT(gpu_1);
    UVM_ASSERT(gpu_2);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);
    UVM_ASSERT(peer_caps->link_type == UVM_GPU_LINK_INVALID);
    UVM_ASSERT(peer_caps->ref_count == 0);

    status = create_p2p_object(gpu_1, gpu_2, &p2p_handle);
    if (status != NV_OK)
        return status;

    // Store the handle in the global table.
    peer_caps->p2p_handle = p2p_handle;

    status = get_p2p_caps(gpu_1, gpu_2, &p2p_caps_params);
    if (status != NV_OK)
        goto cleanup;

    // Sanity checks
    UVM_ASSERT(p2p_caps_params.directAccess == NV_TRUE);
    UVM_ASSERT(p2p_caps_params.p2pLink == UVM_PEER_LINK_TYPE_PCIE);

    status = init_peer_access(gpu_1, gpu_2, &p2p_caps_params, peer_caps);
    if (status != NV_OK)
        goto cleanup;

    return NV_OK;

cleanup:
    disable_peer_access(gpu_1, gpu_2);
    return status;
}

static NV_STATUS enable_nvlink_peer_access(uvm_gpu_t *gpu_1,
                                           uvm_gpu_t *gpu_2,
                                           UvmGpuP2PCapsParams *p2p_caps_params)
{
    NV_STATUS status = NV_OK;
    NvHandle p2p_handle;
    uvm_gpu_peer_t *peer_caps;

    UVM_ASSERT(gpu_1);
    UVM_ASSERT(gpu_2);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);
    UVM_ASSERT(peer_caps->ref_count == 0);
    peer_caps->ref_count = 1;

    if (p2p_caps_params->directAccess) {
        // Create P2P object for direct NVLink peers
        status = create_p2p_object(gpu_1, gpu_2, &p2p_handle);
        if (status != NV_OK) {
            UVM_ERR_PRINT("failed to create a P2P object with error: %s, for GPU1:%s and GPU2:%s \n",
                           nvstatusToString(status),
                           gpu_1->name,
                           gpu_2->name);
            return status;
        }

        UVM_ASSERT(p2p_handle != 0);

        // Store the handle in the global table.
        peer_caps->p2p_handle = p2p_handle;

        // Update p2p caps after p2p object creation as it generates the peer
        // ids
        status = get_p2p_caps(gpu_1, gpu_2, p2p_caps_params);
        if (status != NV_OK)
            goto cleanup;
    }

    status = init_peer_access(gpu_1, gpu_2, p2p_caps_params, peer_caps);
    if (status != NV_OK)
        goto cleanup;

    return NV_OK;

cleanup:
    disable_peer_access(gpu_1, gpu_2);
    return status;
}

static NV_STATUS discover_nvlink_peers(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *other_gpu;

    UVM_ASSERT(gpu);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    for_each_global_gpu(other_gpu) {
        UvmGpuP2PCapsParams p2p_caps_params;

        if (other_gpu == gpu)
            continue;

        status = get_p2p_caps(gpu, other_gpu, &p2p_caps_params);
        if (status != NV_OK)
            goto cleanup;

        // PCIe peers need to be explicitly enabled via UvmEnablePeerAccess
        if (p2p_caps_params.p2pLink == UVM_PEER_LINK_TYPE_NONE || p2p_caps_params.p2pLink == UVM_PEER_LINK_TYPE_PCIE)
            continue;

        // Indirect peers are only supported when onlined as NUMA nodes, because
        // we want to use vm_insert_page and pci_map_page.
        if (!p2p_caps_params.directAccess && (!gpu->numa_info.enabled || !other_gpu->numa_info.enabled))
            continue;

        status = enable_nvlink_peer_access(gpu, other_gpu, &p2p_caps_params);
        if (status != NV_OK)
            goto cleanup;
    }

    return NV_OK;

cleanup:
    destroy_nvlink_peers(gpu);

    return status;
}

static void destroy_nvlink_peers(uvm_gpu_t *gpu)
{
    uvm_gpu_t *other_gpu;

    UVM_ASSERT(gpu);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    for_each_global_gpu(other_gpu) {
        uvm_gpu_peer_t *peer_caps;

        if (other_gpu == gpu)
            continue;

        peer_caps = uvm_gpu_peer_caps(gpu, other_gpu);

        // PCIe peers need to be explicitly destroyed via UvmDisablePeerAccess
        if (peer_caps->link_type == UVM_GPU_LINK_INVALID || peer_caps->link_type == UVM_GPU_LINK_PCIE)
            continue;

        disable_peer_access(gpu, other_gpu);
    }
}

NV_STATUS uvm_gpu_retain_pcie_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_peer_t *peer_caps;

    UVM_ASSERT(gpu_1);
    UVM_ASSERT(gpu_2);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);

    // Insert an entry into global peer table, if not present.
    if (peer_caps->link_type == UVM_GPU_LINK_INVALID) {
        UVM_ASSERT(peer_caps->ref_count == 0);

        status = enable_pcie_peer_access(gpu_1, gpu_2);
        if (status != NV_OK)
            return status;
    }
    else if (peer_caps->link_type != UVM_GPU_LINK_PCIE) {
        return NV_ERR_INVALID_DEVICE;
    }

    // GPUs can't be destroyed until their peer pairings have also been
    // destroyed.
    uvm_gpu_retain(gpu_1);
    uvm_gpu_retain(gpu_2);

    peer_caps->ref_count++;

    return status;
}

static void disable_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2)
{
    uvm_gpu_peer_t *peer_caps;
    NvHandle p2p_handle = 0;

    UVM_ASSERT(gpu_1);
    UVM_ASSERT(gpu_2);

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);
    p2p_handle = peer_caps->p2p_handle;

    if (peer_caps->is_direct_peer) {
        UVM_ASSERT(p2p_handle);

        uvm_mmu_destroy_peer_identity_mappings(gpu_1, gpu_2);
        uvm_mmu_destroy_peer_identity_mappings(gpu_2, gpu_1);

        uvm_rm_locked_call_void(nvUvmInterfaceP2pObjectDestroy(g_uvm_global.rm_session_handle, p2p_handle));

        uvm_spin_lock(&gpu_1->peer_gpus_lock);

        if (gpu_1->id < gpu_2->id)
            gpu_1->peer_gpus[peer_caps->peer_ids[0]] = NULL;
        else
            gpu_1->peer_gpus[peer_caps->peer_ids[1]] = NULL;

        uvm_spin_unlock(&gpu_1->peer_gpus_lock);
        uvm_spin_lock(&gpu_2->peer_gpus_lock);

        if (gpu_1->id < gpu_2->id)
            gpu_2->peer_gpus[peer_caps->peer_ids[1]] = NULL;
        else
            gpu_2->peer_gpus[peer_caps->peer_ids[0]] = NULL;

        uvm_spin_unlock(&gpu_2->peer_gpus_lock);
    }
    else {
        uvm_pmm_gpu_indirect_peer_destroy(&gpu_1->pmm, gpu_2);
        uvm_pmm_gpu_indirect_peer_destroy(&gpu_2->pmm, gpu_1);
    }

    // Flush the access counter buffer to avoid getting stale notifications for
    // accesses to GPUs to which peer access is being disabled. This is not
    // required if we are disabling automatic (NVLINK) peers on GPU unregister
    // (we use the retained count to detect that case).
    if (uvm_gpu_retained_count(gpu_1) != 0 && gpu_1->access_counters_supported)
        uvm_gpu_access_counter_buffer_flush(gpu_1);
    if (uvm_gpu_retained_count(gpu_2) != 0 && gpu_2->access_counters_supported)
        uvm_gpu_access_counter_buffer_flush(gpu_2);

    memset(peer_caps, 0, sizeof(*peer_caps));
}

void uvm_gpu_release_pcie_peer_access(uvm_gpu_t *gpu_1, uvm_gpu_t *gpu_2)
{
    uvm_gpu_peer_t *peer_caps;
    UVM_ASSERT(gpu_1);
    UVM_ASSERT(gpu_2);
    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    peer_caps = uvm_gpu_peer_caps(gpu_1, gpu_2);

    UVM_ASSERT(peer_caps->ref_count > 0);
    UVM_ASSERT(peer_caps->link_type == UVM_GPU_LINK_PCIE);
    peer_caps->ref_count--;

    if (peer_caps->ref_count == 0)
        disable_peer_access(gpu_1, gpu_2);

    uvm_gpu_release_locked(gpu_1);
    uvm_gpu_release_locked(gpu_2);
}

static uvm_aperture_t uvm_gpu_peer_caps_aperture(uvm_gpu_peer_t *peer_caps, uvm_gpu_t *local_gpu, uvm_gpu_t *remote_gpu)
{
    size_t peer_index;
    UVM_ASSERT(peer_caps->link_type != UVM_GPU_LINK_INVALID);

    // Indirect peers are accessed as sysmem addresses
    if (!peer_caps->is_direct_peer)
        return UVM_APERTURE_SYS;

    if (local_gpu->id < remote_gpu->id)
        peer_index = 0;
    else
        peer_index = 1;

    return UVM_APERTURE_PEER(peer_caps->peer_ids[peer_index]);
}

uvm_aperture_t uvm_gpu_peer_aperture(uvm_gpu_t *local_gpu, uvm_gpu_t *remote_gpu)
{
    uvm_gpu_peer_t *peer_caps = uvm_gpu_peer_caps(local_gpu, remote_gpu);
    return uvm_gpu_peer_caps_aperture(peer_caps, local_gpu, remote_gpu);
}

uvm_processor_id_t uvm_gpu_get_processor_id_by_aperture(uvm_gpu_t *gpu, uvm_aperture_t aperture)
{
    NvU32 peer_id;
    uvm_processor_id_t id;

    // TODO: Bug 1899622: On P9 systems with multiple CPU sockets, SYS aperture
    // is also reported for accesses to remote GPUs connected to a different CPU
    // NUMA domain. We will need to determine the actual processor id using the
    // reported physical address.
    if (aperture == UVM_APERTURE_SYS)
        return UVM_CPU_ID;
    else if (aperture == UVM_APERTURE_VID)
        return gpu->id;

    peer_id = UVM_APERTURE_PEER_ID(aperture);

    uvm_spin_lock(&gpu->peer_gpus_lock);

    if (gpu->peer_gpus[peer_id])
        id = gpu->peer_gpus[peer_id]->id;
    else
        id = UVM_MAX_PROCESSORS;

    uvm_spin_unlock(&gpu->peer_gpus_lock);

    return id;
}

uvm_gpu_peer_t *uvm_gpu_index_peer_caps(uvm_gpu_id_t gpu_id1, uvm_gpu_id_t gpu_id2)
{
    NvU32 table_index = uvm_gpu_peer_table_index(gpu_id1, gpu_id2);
    return &g_uvm_global.peers[table_index];
}

uvm_gpu_phys_address_t uvm_gpu_peer_phys_address(uvm_gpu_t *accessing_gpu,
                                                 uvm_gpu_t *owning_gpu,
                                                 uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_peer_t *peer_caps = uvm_gpu_peer_caps(accessing_gpu, owning_gpu);
    uvm_aperture_t aperture = uvm_gpu_peer_caps_aperture(peer_caps, accessing_gpu, owning_gpu);
    NvU64 addr;

    if (peer_caps->is_direct_peer)
        addr = chunk->address;
    else
        addr = uvm_pmm_gpu_indirect_peer_addr(&owning_gpu->pmm, chunk, accessing_gpu);

    return uvm_gpu_phys_address(aperture, addr);
}

uvm_gpu_address_t uvm_gpu_peer_copy_address(uvm_gpu_t *accessing_gpu, uvm_gpu_t *owning_gpu, uvm_gpu_chunk_t *chunk)
{
    uvm_gpu_peer_t *peer_caps = uvm_gpu_peer_caps(accessing_gpu, owning_gpu);

    UVM_ASSERT(peer_caps->link_type != UVM_GPU_LINK_INVALID);

    if (peer_caps->is_direct_peer) {
        NvU32 peer_id = UVM_APERTURE_PEER_ID(uvm_gpu_peer_caps_aperture(peer_caps, accessing_gpu, owning_gpu));
        UVM_ASSERT(accessing_gpu->peer_identity_mappings_supported);
        return uvm_gpu_address_virtual(accessing_gpu->peer_mappings[peer_id].base + chunk->address);
    }
    else {
        // Indirect peers are accessed as sysmem addresses, so they don't need
        // to use identity mappings.
        NvU64 addr = uvm_pmm_gpu_indirect_peer_addr(&owning_gpu->pmm, chunk, accessing_gpu);
        return uvm_gpu_address_physical(UVM_APERTURE_SYS, addr);
    }
}

NvU32 uvm_gpu_to_gpu_optimal_write_ce(const uvm_gpu_t *src, const uvm_gpu_t *dst)
{
    unsigned i;
    const uvm_gpu_peer_t *p2p_caps = uvm_gpu_peer_caps(src, dst);

    if (p2p_caps->link_type < UVM_GPU_LINK_NVLINK_1 || !p2p_caps->is_direct_peer)
        return UVM_COPY_ENGINE_COUNT_MAX;

    i = (src->id < dst->id) ? 0 : 1;
    return p2p_caps->optimal_write_ce[i];
}

static unsigned long instance_ptr_to_key(uvm_gpu_phys_address_t instance_ptr)
{
    NvU64 key;
    int is_sys = (instance_ptr.aperture == UVM_APERTURE_SYS);

    // Instance pointers must be 4k aligned and they must have either VID or SYS
    // apertures. Compress them as much as we can both to guarantee that the key
    // fits within 64 bits, and to make the table as shallow as possible.
    UVM_ASSERT(IS_ALIGNED(instance_ptr.address, UVM_PAGE_SIZE_4K));
    UVM_ASSERT(instance_ptr.aperture == UVM_APERTURE_VID || instance_ptr.aperture == UVM_APERTURE_SYS);

    key = (instance_ptr.address >> 11) | is_sys;
    UVM_ASSERT((unsigned long)key == key);

    return key;
}

static NV_STATUS gpu_add_user_channel_subctx_info(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_gpu_phys_address_t instance_ptr = user_channel->instance_ptr;
    int ret = 0;
    NV_STATUS status = NV_OK;
    uvm_user_channel_subctx_info_t *channel_subctx_info;
    uvm_user_channel_subctx_info_t *new_channel_subctx_info = NULL;
    uvm_va_space_t *va_space = user_channel->gpu_va_space->va_space;

    if (!user_channel->in_subctx)
        return NV_OK;

    // Pre-allocate a subcontext info descriptor out of the lock, in case we
    // need to add a new entry to the tree
    new_channel_subctx_info = uvm_kvmalloc_zero(sizeof(*new_channel_subctx_info));

    // Don't check for the result of the allocation since it is only needed
    // if the TSG has not been registered yet, and we do that under the lock
    // below
    if (new_channel_subctx_info) {
        new_channel_subctx_info->subctxs =
            uvm_kvmalloc_zero(sizeof(*new_channel_subctx_info->subctxs) * gpu->max_subcontexts);
    }

    // Pre-load the tree to allocate memory outside of the table lock. This
    // returns with preemption disabled.
    ret = radix_tree_preload(NV_UVM_GFP_FLAGS);
    if (ret != 0) {
        status = errno_to_nv_status(ret);
        goto exit_no_unlock;
    }

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    // Check if the subcontext information for the channel already exists
    channel_subctx_info =
        (uvm_user_channel_subctx_info_t *)radix_tree_lookup(&gpu->tsg_table, user_channel->tsg_id);

    if (!channel_subctx_info) {
        // We could not allocate the descriptor before taking the lock. Exiting
        if (!new_channel_subctx_info || !new_channel_subctx_info->subctxs) {
            status = NV_ERR_NO_MEMORY;
            goto exit_unlock;
        }

        // Insert the new subcontext information descriptor
        ret = radix_tree_insert(&gpu->tsg_table, user_channel->tsg_id, new_channel_subctx_info);
        UVM_ASSERT(ret == 0);

        channel_subctx_info = new_channel_subctx_info;
    }

    user_channel->subctx_info = channel_subctx_info;

    // Register the VA space of the channel subcontext info descriptor, or
    // check that the existing one matches the channel's
    if (channel_subctx_info->subctxs[user_channel->subctx_id].refcount++ > 0) {
        UVM_ASSERT_MSG(channel_subctx_info->subctxs[user_channel->subctx_id].va_space == va_space,
                       "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: expected VA space 0x%llx but got 0x%llx instead\n",
                       user_channel->hw_channel_id,
                       instance_ptr.address,
                       uvm_aperture_string(instance_ptr.aperture),
                       user_channel->subctx_id,
                       user_channel->tsg_id,
                       (NvU64)va_space,
                       (NvU64)channel_subctx_info->subctxs[user_channel->subctx_id].va_space);
        UVM_ASSERT_MSG(channel_subctx_info->subctxs[user_channel->subctx_id].va_space != NULL,
                       "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: VA space is NULL\n",
                       user_channel->hw_channel_id,
                       instance_ptr.address,
                       uvm_aperture_string(instance_ptr.aperture),
                       user_channel->subctx_id,
                       user_channel->tsg_id);
        UVM_ASSERT_MSG(channel_subctx_info->total_refcount > 0,
                       "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: TSG refcount is 0\n",
                       user_channel->hw_channel_id,
                       instance_ptr.address,
                       uvm_aperture_string(instance_ptr.aperture),
                       user_channel->subctx_id,
                       user_channel->tsg_id);
    }
    else {
        UVM_ASSERT_MSG(channel_subctx_info->subctxs[user_channel->subctx_id].va_space == NULL,
                       "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: expected VA space NULL but got 0x%llx instead\n",
                       user_channel->hw_channel_id,
                       instance_ptr.address,
                       uvm_aperture_string(instance_ptr.aperture),
                       user_channel->subctx_id,
                       user_channel->tsg_id,
                       (NvU64)channel_subctx_info->subctxs[user_channel->subctx_id].va_space);

        channel_subctx_info->subctxs[user_channel->subctx_id].va_space = va_space;
    }

    ++channel_subctx_info->total_refcount;

exit_unlock:
    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    // This re-enables preemption
    radix_tree_preload_end();

exit_no_unlock:
    // Remove the pre-allocated per-TSG subctx information struct if there was
    // some error or it was not used
    if (status != NV_OK || user_channel->subctx_info != new_channel_subctx_info) {
        if (new_channel_subctx_info)
            uvm_kvfree(new_channel_subctx_info->subctxs);

        uvm_kvfree(new_channel_subctx_info);
    }

    return status;
}

static void gpu_remove_user_channel_subctx_info(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_gpu_phys_address_t instance_ptr = user_channel->instance_ptr;
    uvm_user_channel_subctx_info_t *channel_subctx_info;
    uvm_va_space_t *va_space = user_channel->gpu_va_space->va_space;

    if (!user_channel->subctx_info)
        return;

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    // Channel subcontext info descriptor may not have been registered in
    // tsg_table since this function is called in some teardown paths during
    // channel creation
    channel_subctx_info = (uvm_user_channel_subctx_info_t *)radix_tree_lookup(&gpu->tsg_table, user_channel->tsg_id);
    UVM_ASSERT(channel_subctx_info == user_channel->subctx_info);

    UVM_ASSERT_MSG(channel_subctx_info->subctxs[user_channel->subctx_id].refcount > 0,
                   "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: SubCTX refcount is 0\n",
                   user_channel->hw_channel_id,
                   instance_ptr.address,
                   uvm_aperture_string(instance_ptr.aperture),
                   user_channel->subctx_id,
                   user_channel->tsg_id);

    UVM_ASSERT_MSG(channel_subctx_info->subctxs[user_channel->subctx_id].va_space == va_space,
                   "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: expected VA space 0x%llx but got 0x%llx instead\n",
                   user_channel->hw_channel_id,
                   instance_ptr.address,
                   uvm_aperture_string(instance_ptr.aperture),
                   user_channel->subctx_id,
                   user_channel->tsg_id,
                   (NvU64)va_space,
                   (NvU64)channel_subctx_info->subctxs[user_channel->subctx_id].va_space);

    UVM_ASSERT_MSG(channel_subctx_info->total_refcount > 0,
                   "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: TSG refcount is 0\n",
                   user_channel->hw_channel_id,
                   instance_ptr.address,
                   uvm_aperture_string(instance_ptr.aperture),
                   user_channel->subctx_id,
                   user_channel->tsg_id);

    // Decrement VA space refcount. If it gets to zero, unregister the pointer
    if (--channel_subctx_info->subctxs[user_channel->subctx_id].refcount == 0)
        channel_subctx_info->subctxs[user_channel->subctx_id].va_space = NULL;

    if (--channel_subctx_info->total_refcount == 0) {
        channel_subctx_info = (uvm_user_channel_subctx_info_t *)radix_tree_delete(&gpu->tsg_table, user_channel->tsg_id);
        UVM_ASSERT_MSG(channel_subctx_info == user_channel->subctx_info,
                       "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: subctx info found: 0x%llx, but expected: 0x%llx\n",
                       user_channel->hw_channel_id,
                       instance_ptr.address,
                       uvm_aperture_string(instance_ptr.aperture),
                       user_channel->subctx_id,
                       user_channel->tsg_id,
                       (NvU64)channel_subctx_info,
                       (NvU64)user_channel->subctx_info);
    }
    else {
        channel_subctx_info = NULL;
    }

    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    user_channel->subctx_info = NULL;

    // If the global channel_subctx_info refcount is zero, destroy it
    if (channel_subctx_info) {
        UVM_ASSERT(channel_subctx_info->total_refcount == 0);
        uvm_kvfree(channel_subctx_info->subctxs);
        uvm_kvfree(channel_subctx_info);
    }
}

static NV_STATUS gpu_add_user_channel_instance_ptr(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_gpu_phys_address_t instance_ptr = user_channel->instance_ptr;
    unsigned long instance_ptr_key = instance_ptr_to_key(instance_ptr);
    int ret = 0;

    // Pre-load the tree to allocate memory outside of the table lock. This
    // returns with preemption disabled.
    ret = radix_tree_preload(NV_UVM_GFP_FLAGS);
    if (ret != 0)
        return errno_to_nv_status(ret);

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    // Insert the instance_ptr -> user_channel mapping
    ret = radix_tree_insert(&gpu->instance_ptr_table, instance_ptr_key, user_channel);

    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    // This re-enables preemption
    radix_tree_preload_end();

    // Since we did the pre-load, and we shouldn't be adding duplicate entries
    UVM_ASSERT_MSG(ret == 0, "CH %u instance_ptr {0x%llx:%s} SubCTX %u in TSG %u: error %d",
                   user_channel->hw_channel_id,
                   instance_ptr.address,
                   uvm_aperture_string(instance_ptr.aperture),
                   user_channel->subctx_id,
                   user_channel->tsg_id,
                   ret);

    return NV_OK;
}

static void gpu_remove_user_channel_instance_ptr(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_user_channel_t *removed_user_channel;
    uvm_gpu_phys_address_t instance_ptr = user_channel->instance_ptr;
    unsigned long instance_ptr_key = instance_ptr_to_key(instance_ptr);

    if (!user_channel->is_instance_ptr_registered)
        return;

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    removed_user_channel = (uvm_user_channel_t *)radix_tree_delete(&gpu->instance_ptr_table, instance_ptr_key);

    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    UVM_ASSERT(removed_user_channel == user_channel);
}

NV_STATUS uvm_gpu_add_user_channel(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_va_space_t *va_space;
    NV_STATUS status;

    UVM_ASSERT(user_channel->rm_retained_channel);
    UVM_ASSERT(user_channel->gpu_va_space);
    va_space = user_channel->gpu_va_space->va_space;
    UVM_ASSERT(va_space);
    uvm_assert_rwsem_locked(&va_space->lock);

    status = gpu_add_user_channel_subctx_info(gpu, user_channel);
    if (status != NV_OK)
        return status;

    status = gpu_add_user_channel_instance_ptr(gpu, user_channel);
    if (status != NV_OK)
        gpu_remove_user_channel_subctx_info(gpu, user_channel);

    return status;
}

static uvm_user_channel_t *instance_ptr_to_user_channel(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr)
{
    unsigned long key = instance_ptr_to_key(instance_ptr);

    uvm_assert_spinlock_locked(&gpu->instance_ptr_table_lock);

    return (uvm_user_channel_t *)radix_tree_lookup(&gpu->instance_ptr_table, key);
}

static uvm_va_space_t *user_channel_and_subctx_to_va_space(uvm_user_channel_t *user_channel, NvU32 subctx_id)
{
    uvm_user_channel_subctx_info_t *channel_subctx_info;

    UVM_ASSERT(user_channel);
    UVM_ASSERT(user_channel->in_subctx);
    UVM_ASSERT(user_channel->subctx_info);

    uvm_assert_spinlock_locked(&user_channel->gpu->instance_ptr_table_lock);

    channel_subctx_info = user_channel->subctx_info;

    UVM_ASSERT_MSG(subctx_id < user_channel->gpu->max_subcontexts,
                   "instance_ptr {0x%llx:%s} in TSG %u. Invalid SubCTX %u\n",
                   user_channel->instance_ptr.address,
                   uvm_aperture_string(user_channel->instance_ptr.aperture),
                   user_channel->tsg_id,
                   subctx_id);
    UVM_ASSERT_MSG(channel_subctx_info->total_refcount > 0,
                   "instance_ptr {0x%llx:%s} in TSG %u: TSG refcount is 0\n",
                   user_channel->instance_ptr.address,
                   uvm_aperture_string(user_channel->instance_ptr.aperture),
                   user_channel->tsg_id);

    // A subcontext's refcount can be zero if that subcontext is torn down
    // uncleanly and work from that subcontext continues running with work from
    // other subcontexts.
    if (channel_subctx_info->subctxs[subctx_id].refcount == 0) {
        UVM_ASSERT(channel_subctx_info->subctxs[subctx_id].va_space == NULL);
    }
    else {
        UVM_ASSERT_MSG(channel_subctx_info->subctxs[subctx_id].va_space,
                       "instance_ptr {0x%llx:%s} in TSG %u: no VA space for SubCTX %u\n",
                       user_channel->instance_ptr.address,
                       uvm_aperture_string(user_channel->instance_ptr.aperture),
                       user_channel->tsg_id,
                       subctx_id);
    }

    return channel_subctx_info->subctxs[subctx_id].va_space;
}

NV_STATUS uvm_gpu_fault_entry_to_va_space(uvm_gpu_t *gpu, uvm_fault_buffer_entry_t *fault, uvm_va_space_t **out_va_space)
{
    uvm_user_channel_t *user_channel;
    NV_STATUS status = NV_OK;

    *out_va_space = NULL;

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    user_channel = instance_ptr_to_user_channel(gpu, fault->instance_ptr);
    if (!user_channel) {
        status = NV_ERR_INVALID_CHANNEL;
        goto exit_unlock;
    }

    // Faults from HUB clients will always report VEID 0 even if the channel
    // belongs a TSG with many subcontexts. Therefore, we cannot use the per-TSG
    // subctx table and we need to directly return the channel's VA space
    if (!user_channel->in_subctx || (fault->fault_source.client_type == UVM_FAULT_CLIENT_TYPE_HUB)) {
        UVM_ASSERT_MSG(fault->fault_source.ve_id == 0,
                       "Fault packet contains SubCTX %u for channel not in subctx\n",
                       fault->fault_source.ve_id);

        // We can safely access user_channel->gpu_va_space under the
        // instance_ptr_table_lock since gpu_va_space is set to NULL after this
        // function is called in uvm_user_channel_detach
        UVM_ASSERT(user_channel->gpu_va_space->va_space);
        *out_va_space = user_channel->gpu_va_space->va_space;
    }
    else {
        *out_va_space = user_channel_and_subctx_to_va_space(user_channel, fault->fault_source.ve_id);

        // Instance pointer is valid but the fault targets a non-existent
        // subcontext.
        if (!*out_va_space)
            status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
    }

exit_unlock:
    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    return status;
}

NV_STATUS uvm_gpu_access_counter_entry_to_va_space(uvm_gpu_t *gpu,
                                                   uvm_access_counter_buffer_entry_t *entry,
                                                   uvm_va_space_t **out_va_space)
{
    uvm_user_channel_t *user_channel;
    NV_STATUS status = NV_OK;

    *out_va_space = NULL;
    UVM_ASSERT(entry->address.is_virtual);

    uvm_spin_lock(&gpu->instance_ptr_table_lock);

    user_channel = instance_ptr_to_user_channel(gpu, entry->virtual_info.instance_ptr);
    if (!user_channel) {
        status = NV_ERR_INVALID_CHANNEL;
        goto exit_unlock;
    }

    if (!user_channel->in_subctx) {
        UVM_ASSERT_MSG(entry->virtual_info.ve_id == 0,
                       "Access counter packet contains SubCTX %u for channel not in subctx\n",
                       entry->virtual_info.ve_id);

        UVM_ASSERT(user_channel->gpu_va_space->va_space);
        *out_va_space = user_channel->gpu_va_space->va_space;
    }
    else {
        *out_va_space = user_channel_and_subctx_to_va_space(user_channel, entry->virtual_info.ve_id);
        if (!*out_va_space)
            status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
    }

exit_unlock:
    uvm_spin_unlock(&gpu->instance_ptr_table_lock);

    return status;
}

void uvm_gpu_remove_user_channel(uvm_gpu_t *gpu, uvm_user_channel_t *user_channel)
{
    uvm_va_space_t *va_space;

    UVM_ASSERT(user_channel->rm_retained_channel);
    UVM_ASSERT(user_channel->gpu_va_space);
    va_space = user_channel->gpu_va_space->va_space;
    UVM_ASSERT(va_space);
    uvm_assert_rwsem_locked_write(&va_space->lock);

    gpu_remove_user_channel_subctx_info(gpu, user_channel);
    gpu_remove_user_channel_instance_ptr(gpu, user_channel);
}

NV_STATUS uvm_gpu_swizzle_phys(uvm_gpu_t *gpu,
                               NvU64 big_page_phys_address,
                               uvm_gpu_swizzle_op_t op,
                               uvm_tracker_t *tracker)
{
    uvm_gpu_address_t staging_addr, phys_addr, identity_addr;
    uvm_push_t push;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(gpu->big_page.swizzling);
    UVM_ASSERT(IS_ALIGNED(big_page_phys_address, gpu->big_page.internal_size));

    uvm_mutex_lock(&gpu->big_page.staging.lock);

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_GPU_INTERNAL,
                                    &gpu->big_page.staging.tracker,
                                    &push,
                                    "%s phys 0x%llx",
                                    op == UVM_GPU_SWIZZLE_OP_SWIZZLE ? "Swizzling" : "Deswizzling",
                                    big_page_phys_address);
    if (status != NV_OK)
        goto out;

    uvm_push_acquire_tracker(&push, tracker);

    staging_addr  = uvm_gpu_address_physical(UVM_APERTURE_VID, gpu->big_page.staging.chunk->address);
    phys_addr     = uvm_gpu_address_physical(UVM_APERTURE_VID, big_page_phys_address);
    identity_addr = uvm_mmu_gpu_address_for_big_page_physical(phys_addr, gpu);

    // Note that these copies are dependent so they must not be pipelined. We
    // need the default MEMBAR_SYS in case we're going to map a peer GPU to the
    // newly-swizzled memory later.
    if (op == UVM_GPU_SWIZZLE_OP_SWIZZLE) {
        gpu->ce_hal->memcopy(&push, staging_addr, phys_addr, gpu->big_page.internal_size);
        gpu->ce_hal->memcopy(&push, identity_addr, staging_addr, gpu->big_page.internal_size);
    }
    else {
        gpu->ce_hal->memcopy(&push, staging_addr, identity_addr, gpu->big_page.internal_size);
        gpu->ce_hal->memcopy(&push, phys_addr, staging_addr, gpu->big_page.internal_size);
    }

    uvm_push_end(&push);

    uvm_tracker_overwrite_with_push(&gpu->big_page.staging.tracker, &push);

    if (tracker)
        uvm_tracker_overwrite_with_push(tracker, &push);

out:
    uvm_mutex_unlock(&gpu->big_page.staging.lock);
    return status;
}

void uvm_processor_uuid_from_id(NvProcessorUuid *uuid, uvm_processor_id_t id)
{
    if (id == UVM_CPU_ID) {
        memcpy(uuid, &NV_PROCESSOR_UUID_CPU_DEFAULT, sizeof(*uuid));
    }
    else {
        uvm_gpu_t *gpu = uvm_gpu_get(id);
        UVM_ASSERT(gpu);
        memcpy(uuid, &gpu->uuid, sizeof(*uuid));
    }
}

NV_STATUS uvm_gpu_map_cpu_pages(uvm_gpu_t *gpu, struct page *page, size_t size, NvU64 *dma_addr_out)
{
    NvU64 dma_addr = pci_map_page(gpu->pci_dev, page, 0, size, PCI_DMA_BIDIRECTIONAL);

    UVM_ASSERT(PAGE_ALIGNED(size));

    if (NV_PCI_DMA_MAPPING_ERROR(gpu->pci_dev, dma_addr))
        return NV_ERR_OPERATING_SYSTEM;

    if (dma_addr < gpu->dma_addressable_start || dma_addr + size - 1 > gpu->dma_addressable_limit) {
        pci_unmap_page(gpu->pci_dev, dma_addr, size, PCI_DMA_BIDIRECTIONAL);
        UVM_ERR_PRINT_RL("PCI mapped range [0x%llx, 0x%llx) not in the addressable range [0x%llx, 0x%llx), GPU %s\n",
                         dma_addr,
                         dma_addr + (NvU64)size,
                         gpu->dma_addressable_start,
                         gpu->dma_addressable_limit + 1,
                         gpu->name);
        return NV_ERR_INVALID_ADDRESS;
    }

    atomic64_add(size, &gpu->mapped_cpu_pages_size);

    // The GPU has its NV_PFB_XV_UPPER_ADDR register set by RM to
    // dma_addressable_start (in bifSetupDmaWindow_IMPL()) and hence when
    // referencing sysmem from the GPU, dma_addressable_start should be
    // subtracted from the DMA address we get from pci_map_page().
    dma_addr -= gpu->dma_addressable_start;

    // See Bug 1920398 for background and details about NVLink DMA address
    // transformations being applied here.
    if (gpu->npu_dev)
        dma_addr = nv_compress_nvlink_addr(dma_addr);

    *dma_addr_out = dma_addr;
    return NV_OK;
}

void uvm_gpu_unmap_cpu_pages(uvm_gpu_t *gpu, NvU64 dma_address, size_t size)
{
    UVM_ASSERT(PAGE_ALIGNED(size));

    if (gpu->npu_dev)
        dma_address = nv_expand_nvlink_addr(dma_address);
    dma_address += gpu->dma_addressable_start;
    pci_unmap_page(gpu->pci_dev, dma_address, size, PCI_DMA_BIDIRECTIONAL);
    atomic64_sub(size, &gpu->mapped_cpu_pages_size);
}

// This function implements the UvmRegisterGpu API call, as described in uvm.h. Notes:
//
// 1. The UVM VA space has a 1-to-1 relationship with an open instance of /dev/nvidia-uvm. That, in turn, has a 1-to-1
// relationship with a process, because the user-level UVM code (os-user-linux.c, for example) enforces an "open
// /dev/nvidia-uvm only once per process" policy. So a UVM VA space is very close to a process's VA space.
//
// If that user space code fails or is not used, then the relationship is no longer 1-to-1. That situation requires that
// this code should avoid crashing, leaking resources, exhibiting security holes, etc, but it does not have to provide
// correct UVM API behavior. Correct UVM API behavior requires doing the right things in user space before calling into
// the kernel.
//
// 2. The uvm_api*() routines are invoked directly from the top-level ioctl handler. They are considered "API routing
// routines", because they are responsible for providing the behavior that is described in the UVM user-to-kernel API
// documentation, in uvm.h.
//
// 3. A GPU VA space, which you'll see in other parts of the driver, is something different: there may be more than one
// GPU VA space within a process, and therefore within a UVM VA space.
//
NV_STATUS uvm_api_register_gpu(UVM_REGISTER_GPU_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return uvm_va_space_register_gpu(va_space, &params->gpu_uuid);
}

NV_STATUS uvm_api_unregister_gpu(UVM_UNREGISTER_GPU_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    return uvm_va_space_unregister_gpu(va_space, &params->gpu_uuid);
}

NV_STATUS uvm_api_register_gpu_va_space(UVM_REGISTER_GPU_VASPACE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_rm_user_object_t user_rm_va_space =
    {
        .rm_control_fd = params->rmCtrlFd,
        .user_client   = params->hClient,
        .user_object   = params->hVaSpace
    };
    return uvm_va_space_register_gpu_va_space(va_space, &user_rm_va_space, &params->gpuUuid);
}

NV_STATUS uvm_api_unregister_gpu_va_space(UVM_UNREGISTER_GPU_VASPACE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_va_space_unregister_gpu_va_space(va_space, &params->gpuUuid);
}

NV_STATUS uvm_api_pageable_mem_access_on_gpu(UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_t *gpu;

    uvm_va_space_down_read(va_space);
    gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->gpu_uuid);

    if (!gpu) {
        uvm_va_space_up_read(va_space);
        return NV_ERR_INVALID_DEVICE;
    }

    params->pageableMemAccess = NV_FALSE;

    // TODO: Bug 1938881: Currently, the driver checks for ATS support using the
    // user-provided uvm8_ats_mode module parameter. Update this check when
    // the parameter goes away.
    if (uvm8_ats_mode)
        params->pageableMemAccess = NV_TRUE;
    else
    if (uvm_hmm_is_enabled() && !(va_space->initialization_flags & UVM_INIT_FLAGS_DISABLE_HMM))
        params->pageableMemAccess = gpu->replayable_faults_supported;

    uvm_va_space_up_read(va_space);

    return NV_OK;
}

NV_STATUS uvm8_test_set_prefetch_filtering(UVM_TEST_SET_PREFETCH_FILTERING_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu = NULL;
    NV_STATUS status = NV_OK;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    switch (params->filtering_mode) {
        case UVM_TEST_PREFETCH_FILTERING_MODE_FILTER_ALL:
            gpu->arch_hal->disable_prefetch_faults(gpu);
            break;
        case UVM_TEST_PREFETCH_FILTERING_MODE_FILTER_NONE:
            gpu->arch_hal->enable_prefetch_faults(gpu);
            break;
        default:
            status = NV_ERR_INVALID_ARGUMENT;
            break;
    }

done:
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}
