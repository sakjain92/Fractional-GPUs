/*******************************************************************************
    Copyright (c) 2017 NVIDIA Corporation

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

#include "linux/sort.h"
#include "nv_uvm_interface.h"
#include "uvm8_gpu_access_counters.h"
#include "uvm8_global.h"
#include "uvm8_gpu.h"
#include "uvm8_hal.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_tools.h"
#include "uvm8_va_block.h"
#include "uvm8_pmm_sysmem.h"

#define UVM_PERF_ACCESS_COUNTER_BATCH_COUNT_MIN 1
#define UVM_PERF_ACCESS_COUNTER_BATCH_COUNT_DEFAULT 256

// Number of entries that are fetched from the GPU access counter notification
// buffer and serviced in batch
static unsigned uvm_perf_access_counter_batch_count = UVM_PERF_ACCESS_COUNTER_BATCH_COUNT_DEFAULT;
module_param(uvm_perf_access_counter_batch_count, uint, S_IRUGO);

// The GPU offers the following tracking granularities: 64K, 2M, 16M, 16G
//
// Use the largest granularity to minimize the number of access counter
// notifications. This is fine because we simply drop the notifications during
// normal operation, and tests override these values.
//
// TODO: Bug 1990466: we will set this value to 2MB in the release branches
// that implement access counter-guided migrations.
#define UVM_DEFAULT_ACCESS_COUNTER_GRANULARITY (16 * 1024 * 1024 * 1024ULL)
static const unsigned g_uvm_access_counter_threshold = 256;

// Each page in a tracked physical range may belong to a different VA Block. We
// preallocate an array of reverse map translations. However, access counter
// granularity can be set to up to 16G, which would require an array too large
// to hold all possible translations. Thus, we set an upper bound for reverse
// map translations, and we perform as many translation requests as needed to
// cover the whole tracked range.
#define UVM_MAX_TRANSLATION_SIZE (2 * 1024 * 1024ULL)

static NV_STATUS config_granularity_to_bytes(UVM_ACCESS_COUNTER_GRANULARITY granularity, NvU64 *bytes)
{
    switch (granularity) {
        case UVM_ACCESS_COUNTER_GRANULARITY_64K:
            *bytes = 64 * 1024ULL;
            break;
        case UVM_ACCESS_COUNTER_GRANULARITY_2M:
            *bytes = 2 * 1024 * 1024ULL;
            break;
        case UVM_ACCESS_COUNTER_GRANULARITY_16M:
            *bytes = 16 * 1024 * 1024ULL;
            break;
        case UVM_ACCESS_COUNTER_GRANULARITY_16G:
            *bytes = 16 * 1024 * 1024 * 1024ULL;
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    return NV_OK;
}

static NV_STATUS config_bytes_to_granularity(NvU64 bytes, UVM_ACCESS_COUNTER_GRANULARITY *granularity)
{
    switch (bytes) {
        case 64 * 1024ULL:
            *granularity = UVM_ACCESS_COUNTER_GRANULARITY_64K;
            break;
        case 2 * 1024 * 1024ULL:
            *granularity = UVM_ACCESS_COUNTER_GRANULARITY_2M;
            break;
        case 16 * 1024 * 1024ULL:
            *granularity = UVM_ACCESS_COUNTER_GRANULARITY_16M;
            break;
        case 16 * 1024 * 1024 * 1024ULL:
            *granularity = UVM_ACCESS_COUNTER_GRANULARITY_16G;
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    return NV_OK;
}

bool uvm_gpu_access_counters_pending(uvm_gpu_t *gpu)
{
    UVM_ASSERT(gpu->access_counters_supported);

    // Fast path 1: we left some notifications unserviced in the buffer in the last pass
    if (gpu->access_counter_buffer_info.cached_get != gpu->access_counter_buffer_info.cached_put)
        return true;

    // Fast path 2: read the valid bit of the notification buffer entry pointed by the cached get pointer
    if (!gpu->access_counter_buffer_hal->entry_is_valid(gpu, gpu->access_counter_buffer_info.cached_get)) {
        // Slow path: read the put pointer from the GPU register via BAR0 over PCIe
        gpu->access_counter_buffer_info.cached_put =
            UVM_READ_ONCE(*gpu->access_counter_buffer_info.rm_info.pAccessCntrBufferPut);

        // No interrupt pending
        if (gpu->access_counter_buffer_info.cached_get == gpu->access_counter_buffer_info.cached_put)
            return false;
    }

    return true;
}

// This function enables access counters with the given configuration and takes
// ownership from RM. The function also stores the new configuration within the
// uvm_gpu_t struct.
static NV_STATUS access_counters_take_ownership(uvm_gpu_t *gpu, UvmGpuAccessCntrConfig *config)
{
    NV_STATUS status;
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;
    NvU64 tracking_size = 0;

    UVM_ASSERT(gpu->access_counters_supported);

    status = uvm_rm_locked_call(nvUvmInterfaceEnableAccessCntr(gpu->rm_address_space,
                                                               &access_counters->rm_info,
                                                               config));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to enable access counter notification from RM: %s, GPU %s\n",
                      nvstatusToString(status), gpu->name);
        return status;
    }

    status = uvm_rm_locked_call(nvUvmInterfaceOwnAccessCntrIntr(g_uvm_global.rm_session_handle,
                                                                &access_counters->rm_info,
                                                                NV_TRUE));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to take access counter notification ownership from RM: %s, GPU %s\n",
                      nvstatusToString(status), gpu->name);
        return status;
    }

    // TODO: Clear all access counters at this point

    // Read current get/put pointers as this might not be the first time we
    // have taken control of the notify buffer since the GPU was initialized
    access_counters->cached_get = UVM_READ_ONCE(*access_counters->rm_info.pAccessCntrBufferGet);
    access_counters->cached_put = UVM_READ_ONCE(*access_counters->rm_info.pAccessCntrBufferPut);

    gpu->access_counter_buffer_info.current_config.rm = *config;

    // Precompute the maximum size to use in reverse map translations and the
    // number of translations that are required per access counter notification.
    status = config_granularity_to_bytes(config->mimcGranularity, &tracking_size);
    UVM_ASSERT(status == NV_OK);

    access_counters->current_config.mimc.translation_size = min(UVM_MAX_TRANSLATION_SIZE, tracking_size);
    access_counters->current_config.mimc.translations_per_counter =
        max(access_counters->current_config.mimc.translation_size / UVM_MAX_TRANSLATION_SIZE, 1ULL);

    status = config_granularity_to_bytes(config->momcGranularity, &tracking_size);
    UVM_ASSERT(status == NV_OK);

    access_counters->current_config.momc.translation_size = min(UVM_MAX_TRANSLATION_SIZE, tracking_size);
    access_counters->current_config.momc.translations_per_counter =
        max(access_counters->current_config.momc.translation_size / UVM_MAX_TRANSLATION_SIZE, 1ULL);

    return NV_OK;
}

static void access_counters_yield_ownership(uvm_gpu_t *gpu)
{
    NV_STATUS status;

    UVM_ASSERT(gpu->access_counters_supported);

    status = uvm_rm_locked_call(nvUvmInterfaceOwnAccessCntrIntr(g_uvm_global.rm_session_handle,
                                                                &gpu->access_counter_buffer_info.rm_info,
                                                                NV_FALSE));
    UVM_ASSERT(status == NV_OK);

    status = uvm_rm_locked_call(nvUvmInterfaceDisableAccessCntr(gpu->rm_address_space,
                                                                &gpu->access_counter_buffer_info.rm_info));
    UVM_ASSERT(status == NV_OK);
}


NV_STATUS uvm_gpu_init_access_counters(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    UvmGpuAccessCntrConfig config = {0};
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;
    uvm_access_counter_service_batch_context_t *batch_context = &access_counters->batch_service_context;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(gpu->access_counter_buffer_hal != NULL);

    status = uvm_rm_locked_call(nvUvmInterfaceInitAccessCntrInfo(gpu->rm_address_space,
                                                                 &access_counters->rm_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init notify buffer info from RM: %s, GPU %s\n",
                      nvstatusToString(status), gpu->name);

        // nvUvmInterfaceInitAccessCntrInfo may leave fields in rm_info
        // populated when it returns an error. Set the buffer handle to zero as
        // it is used by the deinitialization logic to determine if it was
        // correctly initialized.
        access_counters->rm_info.accessCntrBufferHandle = 0;
        goto fail;
    }

    UVM_ASSERT(access_counters->rm_info.bufferSize %
               gpu->access_counter_buffer_hal->entry_size(gpu) == 0);

    BUILD_BUG_ON(UVM_MAX_TRANSLATION_SIZE > UVM_DEFAULT_ACCESS_COUNTER_GRANULARITY);
    BUILD_BUG_ON(UVM_DEFAULT_ACCESS_COUNTER_GRANULARITY % UVM_MAX_TRANSLATION_SIZE != 0);

    status = config_bytes_to_granularity(UVM_DEFAULT_ACCESS_COUNTER_GRANULARITY, &config.mimcGranularity);
    UVM_ASSERT_MSG(status == NV_OK, "Unsupported access counter granularity %llu\n", UVM_DEFAULT_ACCESS_COUNTER_GRANULARITY);
    config.momcGranularity = config.mimcGranularity;

    config.mimcUseLimit = UVM_ACCESS_COUNTER_USE_LIMIT_FULL;
    config.momcUseLimit = UVM_ACCESS_COUNTER_USE_LIMIT_FULL;
    config.threshold = g_uvm_access_counter_threshold;

    gpu->access_counter_buffer_info.ignore_notifications = false;

    status = access_counters_take_ownership(gpu, &config);
    if (status != NV_OK)
        goto fail;

    access_counters->max_notifications = gpu->access_counter_buffer_info.rm_info.bufferSize /
                                         gpu->access_counter_buffer_hal->entry_size(gpu);

    // Check provided module parameter value
    access_counters->max_batch_size = max(uvm_perf_access_counter_batch_count,
                                          (NvU32)UVM_PERF_ACCESS_COUNTER_BATCH_COUNT_MIN);
    access_counters->max_batch_size = min(access_counters->max_batch_size,
                                          access_counters->max_notifications);

    if (access_counters->max_batch_size != uvm_perf_access_counter_batch_count) {
        pr_info("Invalid uvm_perf_access_counter_batch_count value on GPU %s: %u. Valid range [%u:%u] Using %u instead\n",
                gpu->name,
                uvm_perf_access_counter_batch_count,
                UVM_PERF_ACCESS_COUNTER_BATCH_COUNT_MIN,
                access_counters->max_notifications,
                access_counters->max_batch_size);
    }

    batch_context->notification_cache = uvm_kvmalloc_zero(gpu->access_counter_buffer_info.max_notifications *
                                                          sizeof(*batch_context->notification_cache));
    if (!batch_context->notification_cache) {
        status = NV_ERR_NO_MEMORY;
        goto fail;
    }

    batch_context->virt.notifications = uvm_kvmalloc_zero(access_counters->max_notifications *
                                                          sizeof(*batch_context->virt.notifications));
    if (!batch_context->virt.notifications) {
        status = NV_ERR_NO_MEMORY;
        goto fail;
    }

    batch_context->phys.notifications = uvm_kvmalloc_zero(access_counters->max_notifications *
                                                          sizeof(*batch_context->phys.notifications));
    if (!batch_context->phys.notifications) {
        status = NV_ERR_NO_MEMORY;
        goto fail;
    }

    batch_context->phys.translations = uvm_kvmalloc_zero((UVM_MAX_TRANSLATION_SIZE / PAGE_SIZE) *
                                                         sizeof(*batch_context->phys.translations));
    if (!batch_context->phys.translations) {
        status = NV_ERR_NO_MEMORY;
        goto fail;
    }

    return NV_OK;

fail:
    uvm_gpu_deinit_access_counters(gpu);

    return status;
}

void uvm_gpu_deinit_access_counters(uvm_gpu_t *gpu)
{
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;
    uvm_access_counter_service_batch_context_t *batch_context = &access_counters->batch_service_context;

    if (access_counters->rm_info.accessCntrBufferHandle) {
        NV_STATUS status;

        access_counters_yield_ownership(gpu);

        status = uvm_rm_locked_call(nvUvmInterfaceDestroyAccessCntrInfo(gpu->rm_address_space,
                                                                        &access_counters->rm_info));
        UVM_ASSERT(status == NV_OK);

        access_counters->rm_info.accessCntrBufferHandle = 0;
    }

    uvm_kvfree(batch_context->notification_cache);
    uvm_kvfree(batch_context->virt.notifications);
    uvm_kvfree(batch_context->phys.notifications);
    uvm_kvfree(batch_context->phys.translations);
    batch_context->notification_cache = NULL;
    batch_context->virt.notifications = NULL;
    batch_context->phys.notifications = NULL;
    batch_context->phys.translations = NULL;
}

static void access_counter_buffer_flush_locked(uvm_gpu_t *gpu, uvm_gpu_buffer_flush_mode_t flush_mode)
{
    NvU32 get;
    NvU32 put;
    uvm_spin_loop_t spin;
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;

    // TODO: Bug 1766600: right now uvm locks do not support the synchronization
    //       method used by top and bottom ISR. Add uvm lock assert when it's
    //       supported. Use plain mutex kernel utilities for now.
    UVM_ASSERT(gpu->access_counters_supported);
    UVM_ASSERT(mutex_is_locked(&gpu->isr.access_counters.service_lock.m));

    // Read PUT pointer from the GPU if requested
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT)
        access_counters->cached_put = UVM_READ_ONCE(*access_counters->rm_info.pAccessCntrBufferPut);

    get = access_counters->cached_get;
    put = access_counters->cached_put;

    while (get != put) {
        // Wait until valid bit is set
        UVM_SPIN_WHILE(!gpu->access_counter_buffer_hal->entry_is_valid(gpu, get), &spin);

        gpu->access_counter_buffer_hal->entry_clear_valid(gpu, get);
        ++get;
        if (get == access_counters->max_notifications)
            get = 0;
    }

    access_counters->cached_get = get;

    // Update get pointer on the GPU
    UVM_WRITE_ONCE(*access_counters->rm_info.pAccessCntrBufferGet, get);
}

void uvm_gpu_access_counter_buffer_flush(uvm_gpu_t *gpu)
{
    UVM_ASSERT(gpu->access_counters_supported);

    // Disables access counter interrupts and notification servicing servicing
    uvm_gpu_access_counters_isr_lock(gpu);

    access_counter_buffer_flush_locked(gpu, UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT);

    uvm_gpu_access_counters_isr_unlock(gpu);
}

static inline int cmp_access_counter_instance_ptr(const uvm_access_counter_buffer_entry_t *a,
                                                  const uvm_access_counter_buffer_entry_t *b)
{
    int result;

    result = uvm_gpu_phys_addr_cmp(a->virtual_info.instance_ptr, b->virtual_info.instance_ptr);
    // On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
    // map to a different VA space
    if (result != 0)
        return result;
    return UVM_CMP_DEFAULT(a->virtual_info.ve_id, b->virtual_info.ve_id);
}

// Sort comparator for pointers to GVA access counter notification buffer
// entries that sorts by instance pointer
static int cmp_sort_virt_notifications_by_instance_ptr(const void *_a, const void *_b)
{
    const uvm_access_counter_buffer_entry_t *a = *(const uvm_access_counter_buffer_entry_t **)_a;
    const uvm_access_counter_buffer_entry_t *b = *(const uvm_access_counter_buffer_entry_t **)_b;

    UVM_ASSERT(a->address.is_virtual);
    UVM_ASSERT(b->address.is_virtual);

    return cmp_access_counter_instance_ptr(a, b);
}

// Sort comparator for pointers to GPA access counter notification buffer
// entries that sorts by physical address' aperture
static int cmp_sort_phys_notifications_by_processor_id(const void *_a, const void *_b)
{
    const uvm_access_counter_buffer_entry_t *a = *(const uvm_access_counter_buffer_entry_t **)_a;
    const uvm_access_counter_buffer_entry_t *b = *(const uvm_access_counter_buffer_entry_t **)_b;

    UVM_ASSERT(!a->address.is_virtual);
    UVM_ASSERT(!b->address.is_virtual);

    return UVM_CMP_DEFAULT(a->physical_info.resident_id, b->physical_info.resident_id);
}

typedef enum
{
    // Stop at the first entry that is not ready yet
    NOTIFICATION_FETCH_MODE_READY,

    // Fetch all notifications in the buffer before PUT. Wait for all
    // notifications to become ready
    NOTIFICATION_FETCH_MODE_ALL,
} notification_fetch_mode_t;

static NvU32 fetch_access_counter_buffer_entries(uvm_gpu_t *gpu,
                                                 uvm_access_counter_service_batch_context_t *batch_context,
                                                 notification_fetch_mode_t fetch_mode)
{
    NvU32 get;
    NvU32 put;
    NvU32 i;
    uvm_access_counter_buffer_entry_t *notification_cache;
    uvm_spin_loop_t spin;
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;
    NvU32 last_instance_ptr_idx = 0;
    uvm_aperture_t last_aperture = UVM_APERTURE_PEER_MAX;

    // TODO: Bug 1766600: right now uvm locks do not support the synchronization
    //       method used by top and bottom ISR. Add uvm lock assert when it's
    //       supported. Use plain mutex kernel utilities for now.
    UVM_ASSERT(mutex_is_locked(&gpu->isr.access_counters.service_lock.m));
    UVM_ASSERT(gpu->access_counters_supported);

    notification_cache = batch_context->notification_cache;

    get = access_counters->cached_get;

    // Read put pointer from GPU and cache it
    if (get == access_counters->cached_put) {
        access_counters->cached_put =
            UVM_READ_ONCE(*access_counters->rm_info.pAccessCntrBufferPut);
    }

    put = access_counters->cached_put;

    batch_context->phys.num_notifications = 0;
    batch_context->virt.num_notifications = 0;

    batch_context->virt.is_single_instance_ptr = true;
    batch_context->phys.is_single_aperture = true;

    if (get == put)
        return 0;

    // Parse until get != put and have enough space to cache.
    for (i = 0; get != put; ++i) {
        // We cannot just wait for the last entry (the one pointed by put) to become valid, we have to do it
        // individually since entries can be written out of order
        UVM_SPIN_WHILE(!gpu->access_counter_buffer_hal->entry_is_valid(gpu, get), &spin) {
            // We have some entry to work on. Let's do the rest later.
            if (fetch_mode != NOTIFICATION_FETCH_MODE_ALL && i > 0)
                goto done;
        }

        // Prevent later accesses being moved above the read of the valid bit
        smp_mb__after_atomic();

        // Got valid bit set. Let's cache.
        gpu->access_counter_buffer_hal->parse_entry(gpu, get, &notification_cache[i]);

        // TODO: Align to tracking size

        if (notification_cache[i].address.is_virtual) {
            batch_context->virt.notifications[batch_context->virt.num_notifications++] = &notification_cache[i];

            if (batch_context->virt.is_single_instance_ptr) {
                if (batch_context->virt.num_notifications == 1) {
                    last_instance_ptr_idx = i;
                }
                else if (cmp_access_counter_instance_ptr(&notification_cache[last_instance_ptr_idx],
                                                         &notification_cache[i]) != 0) {
                    batch_context->virt.is_single_instance_ptr = false;
                }
            }
        }
        else {
            batch_context->phys.notifications[batch_context->phys.num_notifications++] = &notification_cache[i];

            notification_cache[i].physical_info.resident_id =
                uvm_gpu_get_processor_id_by_aperture(gpu, notification_cache[i].address.aperture);

            if (batch_context->phys.is_single_aperture) {
                if (batch_context->phys.num_notifications == 1)
                    last_aperture = notification_cache[i].address.aperture;
                else if (notification_cache[i].address.aperture != last_aperture)
                    batch_context->phys.is_single_aperture = false;
            }
        }

        if (notification_cache[i].counter_type == UVM_ACCESS_COUNTER_TYPE_MOMC)
            UVM_ASSERT(notification_cache[i].physical_info.resident_id == gpu->id);
        else
            UVM_ASSERT(notification_cache[i].physical_info.resident_id != gpu->id);

        ++get;
        if (get == access_counters->max_notifications)
            get = 0;
    }

done:
    access_counters->cached_get = get;

    // Update get pointer on the GPU
    UVM_WRITE_ONCE(*access_counters->rm_info.pAccessCntrBufferGet, get);

    return i;
}

static void translate_virt_notifications_instance_ptrs(uvm_gpu_t *gpu,
                                                       uvm_access_counter_service_batch_context_t *batch_context)
{
    NvU32 i;
    NV_STATUS status;

    for (i = 0; i < batch_context->virt.num_notifications; ++i) {
        uvm_access_counter_buffer_entry_t *current_entry = batch_context->virt.notifications[i];

        if (i == 0 ||
            cmp_access_counter_instance_ptr(current_entry, batch_context->virt.notifications[i - 1]) != 0) {
            // If instance_ptr is different, make a new translation. If the
            // translation fails then va_space will be NULL and the entry will
            // simply be ignored in subsequent processing.
            status = uvm_gpu_access_counter_entry_to_va_space(gpu,
                                                              current_entry,
                                                              &current_entry->virtual_info.va_space);
            if (status != NV_OK)
                UVM_ASSERT(current_entry->virtual_info.va_space == NULL);
        }
        else {
            current_entry->virtual_info.va_space = batch_context->virt.notifications[i - 1]->virtual_info.va_space;
        }
    }
}

// GVA notifications provide an instance_ptr and ve_id that can be directly
// translated to a VA space. In order to minimize translations, we sort the
// entries by instance_ptr.
static void preprocess_virt_notifications(uvm_gpu_t *gpu,
                                          uvm_access_counter_service_batch_context_t *batch_context)
{
    if (!batch_context->virt.is_single_instance_ptr) {
        // Sort by instance_ptr
        sort(batch_context->virt.notifications,
             batch_context->virt.num_notifications,
             sizeof(*batch_context->virt.notifications),
             cmp_sort_virt_notifications_by_instance_ptr,
             NULL);
    }

    translate_virt_notifications_instance_ptrs(gpu, batch_context);
}

static NV_STATUS service_virt_notifications(uvm_gpu_t *gpu,
                                            uvm_access_counter_service_batch_context_t *batch_context)
{
    preprocess_virt_notifications(gpu, batch_context);

    // TODO: Bug 1990466: Service virtual notifications. Entries with NULL
    // va_space are simply dropped.
    if (uvm_enable_builtin_tests) {
        NvU32 i;
        for (i = 0; i < batch_context->virt.num_notifications; ++i)
            uvm_tools_broadcast_access_counter(gpu, batch_context->virt.notifications[i]);
    }

    return NV_OK;
}

// GPA notifications provide a physical address and an aperture. Sort
// accesses by aperture to try to coalesce operations on the same target
// processor.
static void preprocess_phys_notifications(uvm_gpu_t *gpu,
                                          uvm_access_counter_service_batch_context_t *batch_context)
{
    if (!batch_context->phys.is_single_aperture) {
        // Sort by instance_ptr
        sort(batch_context->phys.notifications,
             batch_context->phys.num_notifications,
             sizeof(*batch_context->phys.notifications),
             cmp_sort_phys_notifications_by_processor_id,
             NULL);
    }
}

static void service_phys_notification(uvm_gpu_t *gpu,
                                      uvm_access_counter_service_batch_context_t *batch_context,
                                      uvm_access_counter_buffer_entry_t *current_entry)
{
    NvU64 address;
    NvU64 translation_index;
    uvm_access_counter_buffer_info_t *access_counters = &gpu->access_counter_buffer_info;
    const NvU64 translation_size = current_entry->counter_type == UVM_ACCESS_COUNTER_TYPE_MIMC?
                                       access_counters->current_config.mimc.translation_size :
                                       access_counters->current_config.momc.translation_size;
    const NvU64 translations_per_counter = current_entry->counter_type == UVM_ACCESS_COUNTER_TYPE_MIMC?
                                               access_counters->current_config.mimc.translations_per_counter :
                                               access_counters->current_config.momc.translations_per_counter;

    address = current_entry->address.address;

    for (translation_index = 0; translation_index < translations_per_counter; ++translation_index) {
        size_t index;
        size_t num_translations;

        // Obtain the virtual addresses of the pages within the reported
        // DMA range
        if (current_entry->physical_info.resident_id == UVM_CPU_ID) {
            num_translations = uvm_pmm_sysmem_mappings_dma_to_virt(&gpu->pmm_sysmem_mappings,
                                                                   address,
                                                                   translation_size,
                                                                   batch_context->phys.translations,
                                                                   translation_size / PAGE_SIZE);
        }
        else {
            uvm_gpu_t *resident_gpu = uvm_gpu_get(current_entry->physical_info.resident_id);

            // On P9 systems, the CPU accesses the reserved heap on vidmem via
            // coherent NVLINK mappings. This can trigger notifications that
            // fall outside of the allocatable address range. We just drop
            // them.
            if (address >= resident_gpu->vidmem_max_allocatable_address) {
                if (address == current_entry->address.address)
                    return;
                else
                    break;
            }

            num_translations = uvm_pmm_gpu_phys_to_virt(&resident_gpu->pmm,
                                                        address,
                                                        translation_size,
                                                        batch_context->phys.translations);
        }

        for (index = 0; index < num_translations; ++index) {
            uvm_reverse_map_t *reverse_map = &batch_context->phys.translations[index];
            uvm_va_block_release(reverse_map->va_block);
        }

        address += translation_size;
    }

    // TODO: Bug 1990466: Here we already have virtual addresses and
    // address spaces. Merge virtual and physical notification handling

    uvm_tools_broadcast_access_counter(gpu, current_entry);
}

// TODO: Bug 2018899: Add statistics for dropped access counter notifications
static NV_STATUS service_phys_notifications(uvm_gpu_t *gpu,
                                            uvm_access_counter_service_batch_context_t *batch_context)
{
    preprocess_phys_notifications(gpu, batch_context);

    if (uvm_enable_builtin_tests) {
        NvU32 i;
        for (i = 0; i < batch_context->phys.num_notifications; ++i) {
            uvm_access_counter_buffer_entry_t *current_entry = batch_context->phys.notifications[i];

            if (current_entry->physical_info.resident_id == UVM_MAX_PROCESSORS)
                continue;

            service_phys_notification(gpu, batch_context, current_entry);
        }
    }

    return NV_OK;
}

void uvm_gpu_service_access_counters(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;
    uvm_access_counter_service_batch_context_t *batch_context = &gpu->access_counter_buffer_info.batch_service_context;

    UVM_ASSERT(gpu->access_counters_supported);

    if (gpu->access_counter_buffer_info.ignore_notifications)
        return;

    while (1) {
        batch_context->num_cached_notifications = fetch_access_counter_buffer_entries(gpu,
                                                                                      batch_context,
                                                                                      NOTIFICATION_FETCH_MODE_READY);
        if (batch_context->num_cached_notifications == 0)
            break;

        ++batch_context->batch_id;

        status = service_virt_notifications(gpu, batch_context);
        if (status != NV_OK)
            break;

        status = service_phys_notifications(gpu, batch_context);
        if (status != NV_OK)
            break;
    }

    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing access counter notifications on GPU: %s\n", gpu->name);
}

static const NvU32 g_uvm_access_counters_threshold_max = (1 << 15) - 1;

static NV_STATUS access_counters_config_from_test_params(const UVM_TEST_RECONFIGURE_ACCESS_COUNTERS_PARAMS *params,
                                                         UvmGpuAccessCntrConfig *config)
{
    NvU64 tracking_size;
    memset(config, 0, sizeof(*config));

    if (params->threshold == 0 || params->threshold > g_uvm_access_counters_threshold_max)
        return NV_ERR_INVALID_ARGUMENT;

    if (config_granularity_to_bytes(params->mimc_granularity, &tracking_size) != NV_OK)
        return NV_ERR_INVALID_ARGUMENT;

    if (config_granularity_to_bytes(params->momc_granularity, &tracking_size) != NV_OK)
        return NV_ERR_INVALID_ARGUMENT;

    // Since values for granularity/use limit are shared between tests and
    // nv_uvm_types.h, the value will be checked in the call to
    // nvUvmInterfaceEnableAccessCntr
    config->mimcGranularity = params->mimc_granularity;
    config->momcGranularity = params->momc_granularity;

    config->mimcUseLimit = params->mimc_use_limit;
    config->momcUseLimit = params->momc_use_limit;

    config->threshold = params->threshold;

    return NV_OK;
}

NV_STATUS uvm8_test_reconfigure_access_counters(UVM_TEST_RECONFIGURE_ACCESS_COUNTERS_PARAMS *params,
                                                struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu = NULL;
    UvmGpuAccessCntrConfig config = {0};

    status = access_counters_config_from_test_params(params, &config);
    if (status != NV_OK)
        return status;

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    if (!gpu->access_counters_supported) {
        status = NV_ERR_NOT_SUPPORTED;
        goto exit_release_gpu;
    }

    // ISR lock ensures that interrupts are disabled and we own GET/PUT
    // registers and no other thread (nor the top half) will be able to
    // re-enable interrupts during reconfiguration
    uvm_gpu_access_counters_isr_lock(gpu);

    UVM_ASSERT(gpu->isr.access_counters.handling);

    // Disable counters
    access_counters_yield_ownership(gpu);

    // Re-enable with the new configuration
    status = access_counters_take_ownership(gpu, &config);

    uvm_gpu_access_counters_isr_unlock(gpu);

exit_release_gpu:
    uvm_gpu_release(gpu);

    return status;
}

NV_STATUS uvm8_test_reset_access_counters(UVM_TEST_RESET_ACCESS_COUNTERS_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu = NULL;
    uvm_push_t push;

    if (params->mode >= UVM_TEST_ACCESS_COUNTER_RESET_MODE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    if (params->mode == UVM_TEST_ACCESS_COUNTER_RESET_MODE_TARGETED &&
        params->counter_type >= UVM_TEST_ACCESS_COUNTER_TYPE_MAX) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    uvm_gpu_access_counters_isr_lock(gpu);

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &push,
                            "Reset access counters");
    if (status != NV_OK) {
        UVM_ERR_PRINT("Error creating push before resetting access counters: %s, GPU %s\n",
                      nvstatusToString(status), gpu->name);
        goto fail_unlock;
    }

    if (params->mode == UVM_TEST_ACCESS_COUNTER_RESET_MODE_ALL) {
        gpu->host_hal->access_counter_clear_all(&push);
    }
    else {
        uvm_access_counter_buffer_entry_t entry = { 0 };

        if (params->counter_type == UVM_TEST_ACCESS_COUNTER_TYPE_MIMC)
            entry.counter_type = UVM_ACCESS_COUNTER_TYPE_MIMC;
        else
            entry.counter_type = UVM_ACCESS_COUNTER_TYPE_MOMC;

        entry.bank = params->bank;
        entry.tag = params->tag;

        gpu->host_hal->access_counter_clear_targeted(&push, &entry);
    }

    status = uvm_push_end_and_wait(&push);

fail_unlock:
    uvm_gpu_access_counters_isr_unlock(gpu);

    uvm_gpu_release(gpu);

    return status;
}

NV_STATUS uvm8_test_set_ignore_access_counters(UVM_TEST_SET_IGNORE_ACCESS_COUNTERS_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu = NULL;

    status = uvm_gpu_retain_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK)
        return status;

    if (!gpu->access_counters_supported) {
        status = NV_ERR_NOT_SUPPORTED;
        goto exit_release_gpu;
    }

    uvm_gpu_access_counters_isr_lock(gpu);

    if (gpu->access_counter_buffer_info.ignore_notifications != params->ignore) {
        if (params->ignore == 0)
            access_counter_buffer_flush_locked(gpu, params->ignore);

        gpu->access_counter_buffer_info.ignore_notifications = params->ignore;
    }

    uvm_gpu_access_counters_isr_unlock(gpu);

exit_release_gpu:
    uvm_gpu_release(gpu);

    return NV_OK;
}
