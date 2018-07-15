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

#include "uvm8_channel.h"

#include "uvm8_global.h"
#include "uvm8_hal.h"
#include "uvm8_procfs.h"
#include "uvm8_push.h"
#include "uvm8_gpu_semaphore.h"
#include "uvm8_lock.h"
#include "uvm8_kvmalloc.h"

#include "nv_uvm_interface.h"
#include "cla06f.h"

#define UVM_CHANNEL_NUM_GPFIFO_ENTRIES_DEFAULT 1024
#define UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MIN 32
#define UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MAX (1024 * 1024)

static unsigned uvm_channel_num_gpfifo_entries = UVM_CHANNEL_NUM_GPFIFO_ENTRIES_DEFAULT;

#define UVM_CHANNEL_GPFIFO_LOC_DEFAULT "auto"

static char *uvm_channel_gpfifo_loc = UVM_CHANNEL_GPFIFO_LOC_DEFAULT;

#define UVM_CHANNEL_GPPUT_LOC_DEFAULT "auto"

static char *uvm_channel_gpput_loc = UVM_CHANNEL_GPPUT_LOC_DEFAULT;

#define UVM_CHANNEL_PUSHBUFFER_LOC_DEFAULT "auto"

static char *uvm_channel_pushbuffer_loc = UVM_CHANNEL_PUSHBUFFER_LOC_DEFAULT;

module_param(uvm_channel_num_gpfifo_entries, uint, S_IRUGO);
module_param(uvm_channel_gpfifo_loc, charp, S_IRUGO);
module_param(uvm_channel_gpput_loc, charp, S_IRUGO);
module_param(uvm_channel_pushbuffer_loc, charp, S_IRUGO);

static NV_STATUS manager_create_procfs_dirs(uvm_channel_manager_t *manager);
static NV_STATUS manager_create_procfs(uvm_channel_manager_t *manager);
static NV_STATUS channel_create_procfs(uvm_channel_t *channel);

typedef enum
{
    // Only remove completed GPFIFO entries from the pushbuffer
    UVM_CHANNEL_UPDATE_MODE_COMPLETED,

    // Remove all remaining GPFIFO entries from the pushbuffer, regardless of
    // whether they're actually done yet.
    UVM_CHANNEL_UPDATE_MODE_FORCE_ALL
} uvm_channel_update_mode_t;

// Update channel progress, completing up to max_to_complete entries
static NvU32 uvm_channel_update_progress_with_max(uvm_channel_t *channel,
                                                  NvU32 max_to_complete,
                                                  uvm_channel_update_mode_t mode)
{
    NvU32 gpu_get;
    NvU32 cpu_put;
    NvU32 completed_count = 0;
    NvU32 pending_gpfifos;

    NvU64 completed_value = uvm_channel_update_completed_value(channel);

    uvm_spin_lock(&channel->pool->lock);

    cpu_put = channel->cpu_put;
    gpu_get = channel->gpu_get;

    while (gpu_get != cpu_put && completed_count < max_to_complete) {
        uvm_gpfifo_entry_t *entry = &channel->gpfifo_entries[gpu_get];

        if (mode == UVM_CHANNEL_UPDATE_MODE_COMPLETED && entry->tracking_semaphore_value > completed_value)
            break;

        uvm_pushbuffer_mark_completed(channel->pool->manager->pushbuffer, entry);
        list_add_tail(&entry->push_info->available_list_node, &channel->available_push_infos);
        gpu_get = (gpu_get + 1) % channel->num_gpfifo_entries;
        ++completed_count;
    }

    channel->gpu_get = gpu_get;

    uvm_spin_unlock(&channel->pool->lock);

    if (cpu_put >= gpu_get)
        pending_gpfifos = cpu_put - gpu_get;
    else
        pending_gpfifos = channel->num_gpfifo_entries - gpu_get + cpu_put;

    return pending_gpfifos;
}

NvU32 uvm_channel_update_progress(uvm_channel_t *channel)
{
    // By default, don't complete too many entries at a time to spread the cost
    // of doing so across callers and avoid holding a spin lock for too long.
    return uvm_channel_update_progress_with_max(channel, 8, UVM_CHANNEL_UPDATE_MODE_COMPLETED);
}

// Update progress for all pending GPFIFO entries. This might take a longer time
// and should be only used in exceptional circumstances like when a channel
// error is encountered. Otherwise, uvm_chanel_update_progress() should be used.
static NvU32 channel_update_progress_all(uvm_channel_t *channel, uvm_channel_update_mode_t mode)
{
    return uvm_channel_update_progress_with_max(channel, channel->num_gpfifo_entries, mode);
}

NvU32 uvm_channel_update_progress_all(uvm_channel_t *channel)
{
    return channel_update_progress_all(channel, UVM_CHANNEL_UPDATE_MODE_COMPLETED);
}

NvU32 uvm_channel_manager_update_progress(uvm_channel_manager_t *channel_manager)
{
    NvU32 pending_gpfifos = 0;
    uvm_channel_t *channel;
    uvm_for_each_channel(channel, channel_manager)
        pending_gpfifos += uvm_channel_update_progress(channel);

    return pending_gpfifos;
}

static bool is_channel_available(uvm_channel_t *channel)
{
    NvU32 next_put;

    uvm_assert_spinlock_locked(&channel->pool->lock);

    next_put = (channel->cpu_put + channel->current_pushes_count + 1) % channel->num_gpfifo_entries;

    return (next_put != channel->gpu_get);
}

static bool try_claim_channel(uvm_channel_t *channel)
{
    bool claimed = false;

    uvm_spin_lock(&channel->pool->lock);

    if (is_channel_available(channel)) {
        ++channel->current_pushes_count;
        claimed = true;
    }

    uvm_spin_unlock(&channel->pool->lock);

    return claimed;
}

// Reserve a channel in the given manager index range
static NV_STATUS channel_reserve_in_range(uvm_channel_manager_t *manager,
                                          unsigned start,
                                          unsigned end,
                                          uvm_channel_t **channel_out)
{
    uvm_channel_t *channel;
    uvm_spin_loop_t spin;

    uvm_for_each_channel_in_range(channel, manager, start, end) {
        // TODO: Bug 1764953: Prefer idle/less busy channels
        if (try_claim_channel(channel)) {
            *channel_out = channel;
            return NV_OK;
        }
    }

    uvm_spin_loop_init(&spin);
    while (1) {
        uvm_for_each_channel_in_range(channel, manager, start, end) {
            NV_STATUS status;

            uvm_channel_update_progress(channel);

            if (try_claim_channel(channel)) {
                *channel_out = channel;
                return NV_OK;
            }

            status = uvm_channel_check_errors(channel);
            if (status != NV_OK)
                return status;

            UVM_SPIN_LOOP(&spin);
        }
    }

    UVM_ASSERT_MSG(0, "Cannot get here?!\n");
    return NV_ERR_GENERIC;
}

static NV_STATUS channel_reserve_ce(uvm_channel_manager_t *channel_manager,
                                    NvU32 ce_index,
                                    uvm_channel_t **channel_out)
{
    unsigned start, end;

    UVM_ASSERT(ce_index < UVM_COPY_ENGINE_COUNT_MAX);

    start = channel_manager->channel_pools[ce_index].channel_index;
    end = start + UVM_CHANNELS_PER_COPY_ENGINE;

    return channel_reserve_in_range(channel_manager, start, end, channel_out);
}

NV_STATUS uvm_channel_reserve_type(uvm_channel_manager_t *channel_manager,
                                   uvm_channel_type_t type,
                                   uvm_channel_t **channel_out)
{
    unsigned ce_index;

    if (type == UVM_CHANNEL_TYPE_ANY) 
        return channel_reserve_in_range(channel_manager, 0, channel_manager->num_channels, channel_out);

    ce_index = channel_manager->ce_to_use_by_type[type];
    return channel_reserve_ce(channel_manager, ce_index, channel_out);
}

NV_STATUS uvm_channel_reserve_gpu_to_gpu(uvm_channel_manager_t *channel_manager,
                                         uvm_gpu_t *dst_gpu,
                                         uvm_channel_t **channel_out)
{
    const uvm_gpu_t *src_gpu = channel_manager->gpu;
    NvU32 ce_index = uvm_gpu_to_gpu_optimal_write_ce(src_gpu, dst_gpu);

    // No recommended CE for the given pair, use default
    if (ce_index == UVM_COPY_ENGINE_COUNT_MAX)
        ce_index = channel_manager->ce_to_use_by_type[UVM_CHANNEL_TYPE_GPU_TO_GPU];

    return channel_reserve_ce(channel_manager, ce_index, channel_out);
}

NV_STATUS uvm_channel_manager_wait(uvm_channel_manager_t *manager)
{
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;

    if (uvm_channel_manager_update_progress(manager) == 0)
        return uvm_channel_manager_check_errors(manager);

    uvm_spin_loop_init(&spin);
    while (uvm_channel_manager_update_progress(manager) > 0 && status == NV_OK) {
        UVM_SPIN_LOOP(&spin);
        status = uvm_channel_manager_check_errors(manager);
    }

    return status;
}

static NvU32 channel_get_available_push_info_index(uvm_channel_t *channel)
{
    uvm_push_info_t *push_info;

    uvm_spin_lock(&channel->pool->lock);

    push_info = list_first_entry_or_null(&channel->available_push_infos, uvm_push_info_t, available_list_node);
    UVM_ASSERT(push_info != NULL);
    UVM_ASSERT(push_info->on_complete == NULL && push_info->on_complete_data == NULL);
    list_del(&push_info->available_list_node);

    uvm_spin_unlock(&channel->pool->lock);

    return push_info - channel->push_infos;
}

NV_STATUS uvm_channel_begin_push(uvm_channel_t *channel, uvm_push_t *push)
{
    NV_STATUS status;
    uvm_channel_manager_t *manager;

    UVM_ASSERT(channel);
    UVM_ASSERT(push);

    manager = channel->pool->manager;

    status = uvm_pushbuffer_begin_push(manager->pushbuffer, push);
    if (status != NV_OK)
        return status;

    push->channel = channel;
    push->channel_tracking_value = 0;
    push->push_info_index = channel_get_available_push_info_index(channel);

    return NV_OK;
}

void uvm_channel_end_push(uvm_push_t *push)
{
    uvm_channel_t *channel = push->channel;
    uvm_channel_manager_t *channel_manager = channel->pool->manager;
    uvm_gpu_t *gpu = channel_manager->gpu;
    uvm_pushbuffer_t *pushbuffer = channel_manager->pushbuffer;
    uvm_gpu_semaphore_t *semaphore = &channel->tracking_sem.semaphore;
    uvm_gpfifo_entry_t *entry;
    NvU64 new_tracking_value;
    NvU32 new_payload;
    NvU32 push_size;
    NvU64 pushbuffer_va;
    NvU32 cpu_put;
    NvU32 new_cpu_put;
    NvU64 *gpfifo_entry;

    BUILD_BUG_ON(sizeof(*gpfifo_entry) != NVA06F_GP_ENTRY__SIZE);

    uvm_spin_lock(&channel->pool->lock);

    new_tracking_value = ++channel->tracking_sem.queued_value;
    new_payload = (NvU32)new_tracking_value;

    gpu->ce_hal->semaphore_release(push, semaphore, new_payload);

    push_size = uvm_push_get_size(push);
    UVM_ASSERT_MSG(push_size <= UVM_MAX_PUSH_SIZE, "push size %u\n", push_size);

    cpu_put = channel->cpu_put;
    new_cpu_put = (cpu_put + 1) % channel->num_gpfifo_entries;
    gpfifo_entry = (NvU64*)channel->channel_info.gpFifoEntries + cpu_put;

    entry = &channel->gpfifo_entries[cpu_put];
    entry->tracking_semaphore_value = new_tracking_value;
    entry->pushbuffer_offset = uvm_pushbuffer_get_offset_for_push(pushbuffer, push);
    entry->pushbuffer_size = push_size;
    entry->push_info = &channel->push_infos[push->push_info_index];
    push->push_info_index = -1;
    pushbuffer_va = uvm_pushbuffer_get_gpu_va_for_push(pushbuffer, push);

    UVM_ASSERT(channel->current_pushes_count > 0);
    --channel->current_pushes_count;

    gpu->host_hal->set_gpfifo_entry(gpfifo_entry, pushbuffer_va, push_size);

    // Need to make sure all the pushbuffer and the GPFIFO entries writes
    // complete before updating GPPUT. We also don't want any reads to be moved
    // after the GPPut write as the GPU might modify the data they read as soon
    // as the GPPut write happens.
    mb();

    channel->cpu_put = new_cpu_put;
    gpu->host_hal->write_gpu_put(channel, new_cpu_put);

    uvm_pushbuffer_end_push(pushbuffer, push, entry);

    // The moment the channel is unlocked uvm_channel_update_progress_with_max()
    // may notice the GPU work to be completed and hence all state tracking the
    // push must be updated before that. Notably uvm_pushbuffer_end_push() has
    // to be called first.
    uvm_spin_unlock(&channel->pool->lock);

    // This is borrowed from CUDA as it supposedly fixes perf issues on some systems,
    // comment from CUDA:
    // This fixes throughput-related performance problems, e.g. bugs 626179, 593841
    // This may be related to bug 124888, which GL works around by doing a clflush.
    wmb();

    push->channel_tracking_value = new_tracking_value;
}

NV_STATUS uvm_channel_reserve(uvm_channel_t *channel)
{
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;

    if (try_claim_channel(channel))
        return NV_OK;

    uvm_channel_update_progress(channel);

    uvm_spin_loop_init(&spin);
    while (!try_claim_channel(channel) && status == NV_OK) {
        UVM_SPIN_LOOP(&spin);
        status = uvm_channel_check_errors(channel);
        uvm_channel_update_progress(channel);
    }

    return status;
}

// Get the first pending GPFIFO entry, if any.
// This doesn't stop the entry from being reused.
static uvm_gpfifo_entry_t *uvm_channel_get_first_pending_entry(uvm_channel_t *channel)
{
    uvm_gpfifo_entry_t *entry = NULL;
    NvU32 pending_count = channel_update_progress_all(channel, UVM_CHANNEL_UPDATE_MODE_COMPLETED);

    if (pending_count == 0)
        return NULL;

    uvm_spin_lock(&channel->pool->lock);

    if (channel->gpu_get != channel->cpu_put)
        entry = &channel->gpfifo_entries[channel->gpu_get];

    uvm_spin_unlock(&channel->pool->lock);

    return entry;
}

NV_STATUS uvm_channel_get_status(uvm_channel_t *channel)
{
    uvm_gpu_t *gpu;
    NvNotification *errorNotifier = channel->channel_info.errorNotifier;
    if (errorNotifier->status == 0)
        return NV_OK;

    // In case we hit a channel error, check the ECC error notifier as well so
    // that a more precise ECC error can be returned in case there is indeed an
    // ECC error.
    //
    // Notably this might be racy depending on the ordering of the notifications,
    // but we can't always call RM to service interrupts from this context.
    gpu = uvm_channel_get_gpu(channel);
    if (gpu->ecc.enabled && *gpu->ecc.error_notifier)
        return NV_ERR_ECC_ERROR;

    return NV_ERR_RC_ERROR;
}

uvm_gpfifo_entry_t *uvm_channel_get_fatal_entry(uvm_channel_t *channel)
{
    UVM_ASSERT(uvm_channel_get_status(channel) != NV_OK);

    return uvm_channel_get_first_pending_entry(channel);
}

NV_STATUS uvm_channel_check_errors(uvm_channel_t *channel)
{
    uvm_gpfifo_entry_t *fatal_entry;
    NV_STATUS status = uvm_channel_get_status(channel);

    if (status == NV_OK)
        return NV_OK;

    UVM_ERR_PRINT("Detected a channel error, channel %s GPU %s\n", channel->name, uvm_channel_get_gpu(channel)->name);

    fatal_entry = uvm_channel_get_fatal_entry(channel);
    if (fatal_entry != NULL) {
        uvm_push_info_t *push_info = fatal_entry->push_info;
        UVM_ERR_PRINT("Channel error likely caused by push '%s' started at %s:%d in %s()\n",
                push_info->description, push_info->filename, push_info->line, push_info->function);
    }

    uvm_global_set_fatal_error(status);
    return status;
}

NV_STATUS uvm_channel_manager_check_errors(uvm_channel_manager_t *channel_manager)
{
    NV_STATUS status = uvm_global_get_status();
    uvm_channel_t *channel;

    if (status != NV_OK)
        return status;

    uvm_for_each_channel(channel, channel_manager) {
        status = uvm_channel_check_errors(channel);
        if (status != NV_OK)
            return status;
    }

    return status;
}

uvm_gpu_semaphore_t *uvm_channel_get_tracking_semaphore(uvm_channel_t *channel)
{
    return &channel->tracking_sem.semaphore;
}

bool uvm_channel_is_value_completed(uvm_channel_t *channel, NvU64 value)
{
    return uvm_gpu_tracking_semaphore_is_value_completed(&channel->tracking_sem, value);
}

NvU64 uvm_channel_update_completed_value(uvm_channel_t *channel)
{
    return uvm_gpu_tracking_semaphore_update_completed_value(&channel->tracking_sem);
}

static void destroy_channel(uvm_channel_manager_t *manager, uvm_channel_t *channel);

static unsigned channel_pool_get_ce_index(const uvm_channel_pool_t *pool)
{
    return pool - pool->manager->channel_pools;
}

static NV_STATUS create_channel(uvm_channel_pool_t *pool, bool with_procfs, uvm_channel_t *channel)
{
    NV_STATUS status;
    uvmGpuCopyEngineHandle ce_handle;
    uvm_channel_manager_t *manager = pool->manager;
    uvm_gpu_t *gpu = manager->gpu;
    UvmGpuChannelAllocParams channel_alloc_params;
    unsigned int i;
    const unsigned ce_index = channel_pool_get_ce_index(pool);

    UVM_ASSERT(channel != NULL);

    channel->pool = pool;
    manager->num_channels++;
    INIT_LIST_HEAD(&channel->available_push_infos);
    channel->tools.pending_event_count = 0;
    INIT_LIST_HEAD(&channel->tools.channel_list_node);

    status = uvm_gpu_tracking_semaphore_alloc(gpu->semaphore_pool, &channel->tracking_sem);
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_gpu_tracking_semaphore_alloc() failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    memset(&channel_alloc_params, 0, sizeof(channel_alloc_params));
    channel_alloc_params.numGpFifoEntries = manager->conf.num_gpfifo_entries;
    channel_alloc_params.gpFifoLoc        = manager->conf.gpfifo_loc;
    channel_alloc_params.gpPutLoc         = manager->conf.gpput_loc;

    status = uvm_rm_locked_call(nvUvmInterfaceChannelAllocate(gpu->rm_address_space,
                                                              &channel->handle,
                                                              &channel_alloc_params,
                                                              &channel->channel_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceChannelAllocate() failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
        goto error;
    }

    channel->num_gpfifo_entries = manager->conf.num_gpfifo_entries;

    snprintf(channel->name, sizeof(channel->name), "ID %u (0x%x) CE %u",
            channel->channel_info.hwChannelId, channel->channel_info.hwChannelId,
            ce_index);

    status = uvm_rm_locked_call(nvUvmInterfaceCopyEngineAlloc(channel->handle,
                                                              ce_index,
                                                              &ce_handle,
                                                              &channel->channel_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceCopyEngineAlloc(ce_index=%u) failed: %s, channel %s GPU %s\n",
                ce_index, nvstatusToString(status), channel->name, gpu->name);
        goto error;
    }

    channel->gpfifo_entries = uvm_kvmalloc_zero(sizeof(*channel->gpfifo_entries) * channel->num_gpfifo_entries);
    if (channel->gpfifo_entries == NULL) {
        status = NV_ERR_NO_MEMORY;
        goto error;
    }

    channel->push_infos = uvm_kvmalloc_zero(sizeof(*channel->push_infos) * channel->num_gpfifo_entries);
    if (channel->push_infos == NULL) {
        status = NV_ERR_NO_MEMORY;
        goto error;
    }

    for (i = 0; i < channel->num_gpfifo_entries; i++)
        list_add_tail(&channel->push_infos[i].available_list_node, &channel->available_push_infos);

    if (with_procfs) {
        status = channel_create_procfs(channel);
        if (status != NV_OK)
            goto error;
    }

    return status;

error:
    destroy_channel(manager, channel);

    return status;
}

static void destroy_channel(uvm_channel_manager_t *manager, uvm_channel_t *channel)
{
    UVM_ASSERT(manager->num_channels > 0);
    UVM_ASSERT(uvm_channel_get_index(manager, channel) == manager->num_channels - 1);

    if (channel->tracking_sem.queued_value > 0) {
        // The channel should have been idled before being destroyed, unless an
        // error was triggered. We need to check both error cases (global and
        // channel) to handle the UVM_TEST_CHANNEL_SANITY unit test.
        if (uvm_global_get_status() == NV_OK && uvm_channel_get_status(channel) == NV_OK)
            UVM_ASSERT(uvm_gpu_tracking_semaphore_is_completed(&channel->tracking_sem));

        // Remove all remaining GPFIFOs from their pushbuffer chunk, since the
        // pushbuffer has a longer lifetime.
        channel_update_progress_all(channel, UVM_CHANNEL_UPDATE_MODE_FORCE_ALL);
    }

    uvm_procfs_destroy_entry(channel->procfs.pushes);
    uvm_procfs_destroy_entry(channel->procfs.info);
    uvm_procfs_destroy_entry(channel->procfs.dir);

    uvm_kvfree(channel->push_infos);

    uvm_kvfree(channel->gpfifo_entries);

    if (channel->handle)
        uvm_rm_locked_call_void(nvUvmInterfaceChannelDestroy(channel->handle));

    uvm_gpu_tracking_semaphore_free(&channel->tracking_sem);

    UVM_ASSERT(list_empty(&channel->tools.channel_list_node));
    UVM_ASSERT(channel->tools.pending_event_count == 0);

    manager->num_channels--;
}

static NV_STATUS create_channel_pool(uvm_channel_manager_t *channel_manager, unsigned ce_index, bool with_procfs)
{
    unsigned i;
    unsigned start, end;
    uvm_channel_pool_t *pool = channel_manager->channel_pools + ce_index;
    uvm_channel_t *channel;

    pool->manager = channel_manager;
    pool->channel_index = channel_manager->num_channels;

    uvm_spin_lock_init(&pool->lock, UVM_LOCK_ORDER_CHANNEL);

    start = pool->channel_index;
    end = start + UVM_CHANNELS_PER_COPY_ENGINE;

    // Avoid uvm_for_each_channel_in_range because channel creation modifies the number of channels
    for (i = start; i < end; i++) {
        NV_STATUS status;

        channel = channel_manager->channels + i;
        status = create_channel(pool, with_procfs, channel);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

uvm_channel_t *uvm_channel_manager_find_available_channel(uvm_channel_manager_t *channel_manager)
{
    uvm_channel_t *channel;

    uvm_for_each_channel(channel, channel_manager) {
        bool available;

        uvm_spin_lock(&channel->pool->lock);
        available = is_channel_available(channel);
        uvm_spin_unlock(&channel->pool->lock);

        if (available)
            return channel;
    }
    return NULL;
}

static NV_STATUS init_channels(uvm_channel_manager_t *manager)
{
    uvm_channel_t *channel;
    uvm_gpu_t *gpu = manager->gpu;

    uvm_for_each_channel(channel, manager) {
        NV_STATUS status;
        uvm_push_t push;

        status = uvm_push_begin_on_channel(channel, &push, "Init channel");
        if (status != NV_OK) {
            UVM_ERR_PRINT("Failed to begin push on channel: %s, GPU %s\n", nvstatusToString(status), gpu->name);
            return status;
        }
        gpu->ce_hal->init(&push);
        gpu->host_hal->init(&push);
        status = uvm_push_end_and_wait(&push);
        if (status != NV_OK) {
            UVM_ERR_PRINT("Channel init failed: %s, GPU %s\n", nvstatusToString(status), gpu->name);
            return status;
        }
    }

    return NV_OK;
}

static bool ce_usable_for_channel_type(uvm_channel_type_t type, const UvmGpuCopyEngineCaps *cap)
{
    if (!cap->supported || cap->grce)
        return false;

    switch (type) {
        case UVM_CHANNEL_TYPE_CPU_TO_GPU:
        case UVM_CHANNEL_TYPE_GPU_TO_CPU:
            return cap->sysmem;
        case UVM_CHANNEL_TYPE_GPU_INTERNAL:
        case UVM_CHANNEL_TYPE_MEMOPS:
            return true;
        case UVM_CHANNEL_TYPE_GPU_TO_GPU:
            return cap->p2p;
        default:
            UVM_ASSERT_MSG(false, "Unexpected channel type 0x%x\n", type);
            return false;
    }
}

// Returns negative if the first CE should be considered better than the second
static int compare_ce_for_channel_type(const uvm_gpu_t *gpu,
                                       uvm_channel_type_t type,
                                       NvU32 ce_index1,
                                       NvU32 ce_index2,
                                       NvU32 *usage_count)
{
    const UvmGpuCopyEngineCaps *cap1 = &gpu->ce_caps[ce_index1];
    const UvmGpuCopyEngineCaps *cap2 = &gpu->ce_caps[ce_index2];

    UVM_ASSERT(ce_usable_for_channel_type(type, cap1));
    UVM_ASSERT(ce_usable_for_channel_type(type, cap2));
    UVM_ASSERT(ce_index1 < UVM_COPY_ENGINE_COUNT_MAX);
    UVM_ASSERT(ce_index2 < UVM_COPY_ENGINE_COUNT_MAX);
    UVM_ASSERT(ce_index1 != ce_index2);

    switch (type) {
        case UVM_CHANNEL_TYPE_CPU_TO_GPU:
            // For CPU to GPU fast sysmem read is the most important
            if (cap1->sysmemRead != cap2->sysmemRead)
                return cap2->sysmemRead - cap1->sysmemRead;

            // Prefer not to take up the CEs for nvlink P2P
            if (cap1->nvlinkP2p != cap2->nvlinkP2p)
                return cap1->nvlinkP2p - cap2->nvlinkP2p;

            break;

        case UVM_CHANNEL_TYPE_GPU_TO_CPU:
            // For GPU to CPU fast sysmem write is the most important
            if (cap1->sysmemWrite != cap2->sysmemWrite)
                return cap2->sysmemWrite - cap1->sysmemWrite;

            // Prefer not to take up the CEs for nvlink P2P
            if (cap1->nvlinkP2p != cap2->nvlinkP2p)
                return cap1->nvlinkP2p - cap2->nvlinkP2p;

            break;

        case UVM_CHANNEL_TYPE_GPU_TO_GPU:
            // Prefer the LCE with the most PCEs
            {
                int pce_diff = (int)hweight32(cap2->cePceMask) - (int)hweight32(cap1->cePceMask);

                if (pce_diff != 0)
                    return pce_diff;
            }

            break;

        case UVM_CHANNEL_TYPE_GPU_INTERNAL:
            // We want the max possible bandwidth for CEs used for GPU_INTERNAL,
            // for now assume that the number of PCEs is a good measure.
            // TODO: Bug 1735254: Add a direct CE query for local FB bandwidth
            {
                int pce_diff = (int)hweight32(cap2->cePceMask) - (int)hweight32(cap1->cePceMask);

                if (pce_diff != 0)
                    return pce_diff;
            }

            // Leave P2P CEs to the GPU_TO_GPU channel type, when possible
            if (cap1->nvlinkP2p != cap2->nvlinkP2p)
                return cap1->nvlinkP2p - cap2->nvlinkP2p;

            break;

        case UVM_CHANNEL_TYPE_MEMOPS:
            // For MEMOPS we mostly care about  latency which should be better
            // with less used CEs (although we only know about our own usage and
            // not system-wide) so just break out to get the default ordering
            // which prioritizes usage count.
            break;

        default:
            UVM_ASSERT_MSG(false, "Unexpected channel type 0x%x\n", type);
            return 0;
    }

    // By default, prefer less used CEs (within the UVM driver at least)
    if (usage_count[ce_index1] != usage_count[ce_index2])
        return usage_count[ce_index1] - usage_count[ce_index2];

    // And CEs that don't share PCEs
    if (cap1->shared != cap2->shared)
        return cap1->shared - cap2->shared;

    // Last resort, just order by index
    return ce_index1 - ce_index2;
}
 
// Pick default CE for the given channel type, and increment its usage
// count. This function also sets the i-th bit in usable_ce_mask if the 
// CE with index i can be used for transfers of the specified type. 
//
// Both the usage count array and the CE mask must be initialized prior 
// to calling this function 
static NV_STATUS pick_ce_for_channel_type(uvm_channel_manager_t *manager,
                                          uvm_channel_type_t type,
                                          NvU32 *usage_count,
                                          long unsigned *usable_ce_mask)
{
    NvU32 i;
    NvU32 best_ce = UVM_COPY_ENGINE_COUNT_MAX;
    uvm_gpu_t *gpu = manager->gpu;

    for (i = 0; i < UVM_COPY_ENGINE_COUNT_MAX; ++i) {
        UvmGpuCopyEngineCaps *cap = &gpu->ce_caps[i];
        if (!ce_usable_for_channel_type(type, cap))
            continue;

        __set_bit(i, usable_ce_mask);

        if (best_ce == UVM_COPY_ENGINE_COUNT_MAX) {
            best_ce = i;
            continue;
        }

        if (compare_ce_for_channel_type(gpu, type, i, best_ce, usage_count) < 0)
            best_ce = i;
    }

    if (best_ce == UVM_COPY_ENGINE_COUNT_MAX) {
        UVM_ERR_PRINT("Failed to find a suitable CE for channel type %s\n", uvm_channel_type_to_string(type));
        return NV_ERR_NOT_SUPPORTED;
    }

    ++usage_count[best_ce];
    manager->ce_to_use_by_type[type] = best_ce;
    return NV_OK;
}

static NV_STATUS channel_manager_pick_copy_engines(uvm_channel_manager_t *manager, long unsigned *usable_ce_mask)
{
    unsigned i;

    // Per CE usage count so far
    NvU32 usage_count[UVM_COPY_ENGINE_COUNT_MAX] = {0};
    uvm_channel_type_t types[] = {UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                  UVM_CHANNEL_TYPE_GPU_TO_CPU,
                                  UVM_CHANNEL_TYPE_GPU_INTERNAL,
                                  UVM_CHANNEL_TYPE_GPU_TO_GPU,
                                  UVM_CHANNEL_TYPE_MEMOPS};

   // The order of picking CEs for each type matters as it's affected by the 
   // usage count of each CE and it increases every time a CE is selected. 
   // MEMOPS has the least priority as it only cares about low usage of the
   // CE to improve latency
    for (i = 0; i < ARRAY_SIZE(types); ++i) {
        NV_STATUS status = pick_ce_for_channel_type(manager, types[i], usage_count, usable_ce_mask);

        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static bool is_string_valid_location(const char *loc)
{
    return strcmp(uvm_channel_gpfifo_loc, "sys") == 0 ||
           strcmp(uvm_channel_gpfifo_loc, "vid") == 0 ||
           strcmp(uvm_channel_gpfifo_loc, "auto") == 0;
}

static UVM_BUFFER_LOCATION string_to_buffer_location(const char *loc)
{
    UVM_ASSERT(is_string_valid_location(loc));

    if (strcmp(loc, "sys") == 0)
        return UVM_BUFFER_LOCATION_SYS;
    else if (strcmp(loc, "vid") == 0)
        return UVM_BUFFER_LOCATION_VID;
    else
        return UVM_BUFFER_LOCATION_DEFAULT;
}

static const char *buffer_location_to_string(UVM_BUFFER_LOCATION loc)
{
    if (loc == UVM_BUFFER_LOCATION_SYS)
        return "sys";
    else if (loc == UVM_BUFFER_LOCATION_VID)
        return "vid";
    else if (loc == UVM_BUFFER_LOCATION_DEFAULT)
        return "auto";

    UVM_ASSERT_MSG(false, "Invalid buffer locationvalue %d\n", loc);
    return NULL;
}

static void init_channel_manager_conf(uvm_channel_manager_t *manager)
{
    const char *gpfifo_loc_value;
    const char *gpput_loc_value;
    const char *pushbuffer_loc_value;

    uvm_gpu_t *gpu = manager->gpu;

    // 1- Number of GPFIFO entries
    manager->conf.num_gpfifo_entries = uvm_channel_num_gpfifo_entries;

    if (uvm_channel_num_gpfifo_entries < UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MIN)
        manager->conf.num_gpfifo_entries = UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MIN;
    else if (uvm_channel_num_gpfifo_entries > UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MAX)
        manager->conf.num_gpfifo_entries = UVM_CHANNEL_NUM_GPFIFO_ENTRIES_MAX;

    if (!is_power_of_2(manager->conf.num_gpfifo_entries))
        manager->conf.num_gpfifo_entries = UVM_CHANNEL_NUM_GPFIFO_ENTRIES_DEFAULT;

    if (manager->conf.num_gpfifo_entries != uvm_channel_num_gpfifo_entries) {
        pr_info("Invalid value for uvm_channel_num_gpfifo_entries = %u, using %u instead\n",
                uvm_channel_num_gpfifo_entries,
                manager->conf.num_gpfifo_entries);
    }

    // 2- Pushbuffer location
    manager->conf.pushbuffer_loc = UVM_BUFFER_LOCATION_SYS;

    pushbuffer_loc_value = uvm_channel_pushbuffer_loc;
    if (!is_string_valid_location(pushbuffer_loc_value)) {
        pushbuffer_loc_value = UVM_CHANNEL_PUSHBUFFER_LOC_DEFAULT;
        pr_info("Invalid value for uvm_channel_pushbuffer_loc = %s, using %s instead\n",
                uvm_channel_pushbuffer_loc,
                pushbuffer_loc_value);
    }

    // Override the default value if requested by the user
    if (strcmp(pushbuffer_loc_value, "vid") == 0)
        manager->conf.pushbuffer_loc = UVM_BUFFER_LOCATION_VID;

    // 3- GPFIFO/GPPut location
    // Only support the knobs for GPFIFO/GPPut on Volta+
    if (!gpu->gpfifo_in_vidmem_supported) {
        manager->conf.gpfifo_loc = UVM_BUFFER_LOCATION_DEFAULT;
        manager->conf.gpput_loc = UVM_BUFFER_LOCATION_DEFAULT;

        return;
    }

    gpfifo_loc_value = uvm_channel_gpfifo_loc;
    if (!is_string_valid_location(gpfifo_loc_value)) {
        gpfifo_loc_value = UVM_CHANNEL_GPFIFO_LOC_DEFAULT;
        pr_info("Invalid value for uvm_channel_gpfifo_loc = %s, using %s instead\n",
                uvm_channel_gpfifo_loc,
                gpfifo_loc_value);
    }

    gpput_loc_value = uvm_channel_gpput_loc;
    if (!is_string_valid_location(gpput_loc_value)) {
        gpput_loc_value = UVM_CHANNEL_GPPUT_LOC_DEFAULT;
        pr_info("Invalid value for uvm_channel_gpput_loc = %s, using %s instead\n",
                uvm_channel_gpput_loc,
                gpput_loc_value);
    }

    // By default we place GPFIFO and GPPUT on vidmem as it potentially has
    // lower latency.
    manager->conf.gpfifo_loc = UVM_BUFFER_LOCATION_VID;
    manager->conf.gpput_loc = UVM_BUFFER_LOCATION_VID;

    // TODO: Bug 1766129: However, this will likely be different on P9 systems.
    // Leaving GPFIFO on sysmem for now. GPPut on sysmem is not supported in
    // production, so we keep it on vidmem, too.
    if (gpu->sysmem_link >= UVM_GPU_LINK_NVLINK_2)
        manager->conf.gpfifo_loc = UVM_BUFFER_LOCATION_SYS;

    // Override defaults
    if (string_to_buffer_location(gpfifo_loc_value) != UVM_BUFFER_LOCATION_DEFAULT)
        manager->conf.gpfifo_loc = string_to_buffer_location(gpfifo_loc_value);

    if (string_to_buffer_location(gpput_loc_value) != UVM_BUFFER_LOCATION_DEFAULT) {
        manager->conf.gpput_loc = string_to_buffer_location(gpput_loc_value);

        if (manager->conf.gpput_loc == UVM_BUFFER_LOCATION_SYS)
            pr_info("CAUTION: allocating GPPut in sysmem is NOT supported and may crash your system.\n");
    }
}

NV_STATUS uvm_channel_manager_create_common(uvm_gpu_t *gpu, bool with_procfs, uvm_channel_manager_t **channel_manager_out)
{
    NV_STATUS status = NV_OK;
    uvm_channel_manager_t *channel_manager;
    NvU32 i;

    // Mask containing the indexes of the Copy Engines that can be used for at
    // least one channel type
    long unsigned usable_ce_mask = 0;

    BUILD_BUG_ON(UVM_COPY_ENGINE_COUNT_MAX > sizeof(usable_ce_mask) * 8);

    channel_manager = uvm_kvmalloc_zero(sizeof(*channel_manager));
    if (!channel_manager)
        return NV_ERR_NO_MEMORY;

    channel_manager->gpu = gpu;
    init_channel_manager_conf(channel_manager);
    status = uvm_pushbuffer_create_common(channel_manager, with_procfs, &channel_manager->pushbuffer);
    if (status != NV_OK)
        goto error;

    if (with_procfs) {
        status = manager_create_procfs_dirs(channel_manager);
        if (status != NV_OK)
            goto error;
    }

    status = channel_manager_pick_copy_engines(channel_manager, &usable_ce_mask);
    if (status != NV_OK)
        goto error;

    // We create a channel for every CE that is usable for at least one channel
    // type, even if it has not been selected as the default CE for any type.
    // As more information is discovered (for example, a pair of peer GPUs is
    // added) we may start using the previously idle channels.
    for_each_set_bit(i, &usable_ce_mask, UVM_COPY_ENGINE_COUNT_MAX) {
        status = create_channel_pool(channel_manager, i, with_procfs);
        if (status != NV_OK)
            goto error;
    }

    status = init_channels(channel_manager);
    if (status != NV_OK)
        goto error;

    if (with_procfs) {
        status = manager_create_procfs(channel_manager);
        if (status != NV_OK)
            goto error;
    }

    *channel_manager_out = channel_manager;

    return status;

error:
    uvm_channel_manager_destroy(channel_manager);
    return status;
}

void uvm_channel_manager_destroy(uvm_channel_manager_t *channel_manager)
{
    if (channel_manager == NULL)
        return;

    uvm_procfs_destroy_entry(channel_manager->procfs.pending_pushes);

    while (channel_manager->num_channels)
        destroy_channel(channel_manager, channel_manager->channels + channel_manager->num_channels - 1);

    uvm_procfs_destroy_entry(channel_manager->procfs.channels_dir);

    uvm_pushbuffer_destroy(channel_manager->pushbuffer);

    uvm_kvfree(channel_manager);
}

const char *uvm_channel_type_to_string(uvm_channel_type_t channel_type)
{
    BUILD_BUG_ON(UVM_CHANNEL_TYPE_COUNT != 6);

    switch (channel_type) {
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_CPU_TO_GPU);
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_GPU_TO_CPU);
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_GPU_INTERNAL);
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_MEMOPS);
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_GPU_TO_GPU);
        UVM_ENUM_STRING_CASE(UVM_CHANNEL_TYPE_ANY);
        UVM_ENUM_STRING_DEFAULT();
    }
}

static void uvm_channel_print_info(uvm_channel_t *channel, struct seq_file *s)
{
    uvm_channel_manager_t *manager = channel->pool->manager;
    UVM_SEQ_OR_DBG_PRINT(s, "Channel %s\n", channel->name);

    uvm_spin_lock(&channel->pool->lock);

    UVM_SEQ_OR_DBG_PRINT(s, "completed          %llu\n", uvm_channel_update_completed_value(channel));
    UVM_SEQ_OR_DBG_PRINT(s, "queued             %llu\n", channel->tracking_sem.queued_value);
    UVM_SEQ_OR_DBG_PRINT(s, "GPFIFO count       %u\n", channel->num_gpfifo_entries);
    UVM_SEQ_OR_DBG_PRINT(s, "GPFIFO location    %s\n", buffer_location_to_string(manager->conf.gpfifo_loc));
    UVM_SEQ_OR_DBG_PRINT(s, "GPPUT location     %s\n", buffer_location_to_string(manager->conf.gpput_loc));
    UVM_SEQ_OR_DBG_PRINT(s, "get                %u\n", channel->gpu_get);
    UVM_SEQ_OR_DBG_PRINT(s, "put                %u\n", channel->cpu_put);
    UVM_SEQ_OR_DBG_PRINT(s, "Semaphore GPU VA   0x%llx\n", uvm_gpu_semaphore_get_gpu_va(&channel->tracking_sem.semaphore,
                                                                                        uvm_channel_get_gpu(channel)));

    uvm_spin_unlock(&channel->pool->lock);
}

// Print all pending pushes and up to finished_pushes_count completed if their
// GPFIFO entries haven't been reused yet.
static void channel_print_pushes(uvm_channel_t *channel, NvU32 finished_pushes_count, struct seq_file *seq)
{
    NvU32 gpu_get;
    NvU32 cpu_put;

    NvU64 completed_value = uvm_channel_update_completed_value(channel);

    uvm_spin_lock(&channel->pool->lock);

    cpu_put = channel->cpu_put;

    for (gpu_get = channel->gpu_get; gpu_get != cpu_put; gpu_get = (gpu_get + 1) % channel->num_gpfifo_entries) {
        uvm_gpfifo_entry_t *entry = &channel->gpfifo_entries[gpu_get];
        uvm_push_info_t *push_info = entry->push_info;

        if (entry->tracking_semaphore_value + finished_pushes_count <= completed_value)
            continue;

        UVM_SEQ_OR_DBG_PRINT(seq, " %s push '%s' started at %s:%d in %s() releasing value %llu\n",
                entry->tracking_semaphore_value <= completed_value ? "finished" : "pending",
                push_info->description, push_info->filename, push_info->line, push_info->function,
                entry->tracking_semaphore_value);
    }
    uvm_spin_unlock(&channel->pool->lock);
}

void uvm_channel_print_pending_pushes(uvm_channel_t *channel)
{
    channel_print_pushes(channel, 0, NULL);
}

void uvm_channel_manager_print_pending_pushes(uvm_channel_manager_t *manager, struct seq_file *seq)
{
    uvm_channel_t *channel;

    uvm_for_each_channel(channel, manager) {
        UVM_SEQ_OR_DBG_PRINT(seq, "Channel %s, pending pushes:\n", channel->name);

        channel_print_pushes(channel, 0, seq);
    }
}

static NV_STATUS manager_create_procfs_dirs(uvm_channel_manager_t *manager)
{
    uvm_gpu_t *gpu = manager->gpu;

    // The channel manager procfs files are debug only
    if (!uvm_procfs_is_debug_enabled())
        return NV_OK;

    manager->procfs.channels_dir = NV_CREATE_PROC_DIR("channels", gpu->procfs.dir);
    if (manager->procfs.channels_dir == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}

static int nv_procfs_read_manager_pending_pushes(struct seq_file *s, void *v)
{
    uvm_channel_manager_t *manager = (uvm_channel_manager_t *)s->private;
    uvm_channel_manager_print_pending_pushes(manager, s);
    return 0;
}

NV_DEFINE_PROCFS_SINGLE_FILE(manager_pending_pushes);

static NV_STATUS manager_create_procfs(uvm_channel_manager_t *manager)
{
    uvm_gpu_t *gpu = manager->gpu;

    // The channel manager procfs files are debug only
    if (!uvm_procfs_is_debug_enabled())
        return NV_OK;

    manager->procfs.pending_pushes = NV_CREATE_PROC_FILE("pending_pushes", gpu->procfs.dir, manager_pending_pushes, (void *)manager);
    if (manager->procfs.pending_pushes == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}

static int nv_procfs_read_channel_info(struct seq_file *s, void *v)
{
    uvm_channel_t *channel = (uvm_channel_t *)s->private;
    uvm_channel_print_info(channel, s);
    return 0;
}

NV_DEFINE_PROCFS_SINGLE_FILE(channel_info);

static int nv_procfs_read_channel_pushes(struct seq_file *s, void *v)
{
    uvm_channel_t *channel = (uvm_channel_t *)s->private;
    // Include up to 5 finished pushes for some context
    channel_print_pushes(channel, 5, s);
    return 0;
}

NV_DEFINE_PROCFS_SINGLE_FILE(channel_pushes);

static NV_STATUS channel_create_procfs(uvm_channel_t *channel)
{
    char channel_dirname[16];
    uvm_channel_manager_t *manager = channel->pool->manager;

    // The channel procfs files are debug only
    if (!uvm_procfs_is_debug_enabled())
        return NV_OK;

    snprintf(channel_dirname, sizeof(channel_dirname), "%u", channel->channel_info.hwChannelId);

    channel->procfs.dir = NV_CREATE_PROC_DIR(channel_dirname, manager->procfs.channels_dir);
    if (channel->procfs.dir == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    channel->procfs.info = NV_CREATE_PROC_FILE("info", channel->procfs.dir, channel_info, (void *)channel);
    if (channel->procfs.info == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    channel->procfs.pushes = NV_CREATE_PROC_FILE("pushes", channel->procfs.dir, channel_pushes, (void *)channel);
    if (channel->procfs.info == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    return NV_OK;
}
