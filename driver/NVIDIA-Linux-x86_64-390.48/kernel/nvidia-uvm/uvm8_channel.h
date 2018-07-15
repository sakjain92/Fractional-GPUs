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

#ifndef __UVM8_CHANNEL_H__
#define __UVM8_CHANNEL_H__

#include "nv_uvm_types.h"
#include "uvm8_forward_decl.h"
#include "uvm8_gpu_semaphore.h"
#include "uvm8_pushbuffer.h"
#include "uvm8_tracker.h"

// TODO: Bug 1764958: Tweak this after we can run and benchmark real workloads.
// Likely also using different number of channels for different pools.
#define UVM_CHANNELS_PER_COPY_ENGINE 2

// Maximum number of channels to be created 
#define UVM_CHANNEL_COUNT_MAX (UVM_COPY_ENGINE_COUNT_MAX * UVM_CHANNELS_PER_COPY_ENGINE)

//
// UVM channels
//
// A channel manager is created as part of the GPU addition. This involves
// creating channels for each of the supported types (uvm_channel_type_t) in
// separate channel pools possibly using different CE instances in the HW. Each
// channel has a uvm_gpu_tracking_semaphore_t and a set of uvm_gpfifo_entry_t
// (one per each HW GPFIFO entry) allowing to track completion of pushes on the
// channel.
//
// Beginning a push on a channel implies reserving a GPFIFO entry in that
// channel and hence there can only be as many on-going pushes per channel as
// there are free GPFIFO entries. This ensures that ending a push won't have to
// wait for a GPFIFO entry to free up.
//

// Channel types
//
// Remember to update uvm_channel_type_to_string() in uvm8_channel.c when adding new types.
typedef enum
{
    // CPU to GPU copies
    UVM_CHANNEL_TYPE_CPU_TO_GPU,

    // GPU to CPU copies
    UVM_CHANNEL_TYPE_GPU_TO_CPU,

    // Memsets and copies within the GPU
    UVM_CHANNEL_TYPE_GPU_INTERNAL,

    // Memops and small memsets/copies for writing PTEs
    UVM_CHANNEL_TYPE_MEMOPS,

    // GPU to GPU peer copies
    UVM_CHANNEL_TYPE_GPU_TO_GPU,

    // Any channel type
    // Can be used as the type with uvm_push_begin*() to pick any channel
    UVM_CHANNEL_TYPE_ANY,

    UVM_CHANNEL_TYPE_COUNT
} uvm_channel_type_t;

struct uvm_gpfifo_entry_struct
{
    // Offset of the pushbuffer in the pushbuffer allocation used by this entry
    NvU32 pushbuffer_offset;

    // Size of the pushbuffer used for this entry
    NvU32 pushbuffer_size;

    // List node used by the pushbuffer tracking
    struct list_head pending_list_node;

    // Channel tracking semaphore value that indicates completion of this entry
    NvU64 tracking_semaphore_value;

    // Push info for the pending push that used this GPFIFO entry
    uvm_push_info_t *push_info;
};

// A channel pool is a set of channels that use the same (logical) Copy Engine
typedef struct
{
    // Owning channel manager
    uvm_channel_manager_t *manager;

    // Index (within manager->channels) of the first channel in the pool
    unsigned channel_index;

    // Lock protecting the state of channels in the pool
    uvm_spinlock_t lock;
} uvm_channel_pool_t;

struct uvm_channel_struct
{
    // Owning pool
    uvm_channel_pool_t *pool;

    // The channel type and HW channel ID as string for easy debugging and logs
    char name[64];

    // Array of gpfifo entries, one per each HW GPFIFO
    uvm_gpfifo_entry_t *gpfifo_entries;

    // Number of GPFIFO entries in gpfifo_entries
    NvU32 num_gpfifo_entries;

    // Latest GPFIFO entry submitted to the GPU
    // Updated when new pushes are submitted to the GPU in
    // uvm_channel_end_push().
    NvU32 cpu_put;

    // Latest GPFIFO entry completed by the GPU
    // Updated by uvm_channel_update_progress() after checking pending GPFIFOs
    // for completion.
    NvU32 gpu_get;

    // Number of currently on-going pushes on this channel
    // A new push is only allowed to begin on the channel if there is a free
    // GPFIFO entry for it.
    NvU32 current_pushes_count;

    // Array of uvm_push_info_t for all pending pushes on the channel
    uvm_push_info_t *push_infos;

    // List of uvm_push_info_entry_t that are currently available. A push info entry is not available if it has been
    // assigned to a push (uvm_push_begin), and the GPFIFO entry associated with the push has not been marked as
    // completed.
    struct list_head available_push_infos;

    // GPU tracking semaphore tracking the work in the channel
    // Each push on the channel increments the semaphore, see
    // uvm_channel_end_push().
    uvm_gpu_tracking_semaphore_t tracking_sem;

    // UVM-RM interface handle
    uvmGpuChannelHandle handle;

    // Channel state that UVM-RM interface populates, includes the GPFIFO, error
    // notifier etc.
    UvmGpuChannelPointers channel_info;

    struct
    {
        struct proc_dir_entry *dir;
        struct proc_dir_entry *info;
        struct proc_dir_entry *pushes;
    } procfs;

    // Information managed by the tools event notification mechanism. Mainly
    // used to keep a list of channels with pending events, which is needed
    // to collect the timestamps of asynchronous operations.
    struct
    {
        struct list_head channel_list_node;
        NvU32 pending_event_count;
    } tools;
};

struct uvm_channel_manager_struct
{
    // The owning GPU
    uvm_gpu_t *gpu;

    // The pushbuffer used for all pushes done with this channel manager
    uvm_pushbuffer_t *pushbuffer;

    // Array of channel pools, one per copy engine
    // Depending on GPU topology, some of the pools may never be initialized,
    // because the associated LCEs cannot be used for any transfer of interest.
    uvm_channel_pool_t channel_pools[UVM_COPY_ENGINE_COUNT_MAX];

    // Channel array. Only the first num_channels have been created.
    // Channels belonging to the same pool are located contiguously in the array. 
    uvm_channel_t channels[UVM_CHANNEL_COUNT_MAX];

    // Number of created channels that have not been destroyed. 
    unsigned num_channels;

    // Array of CE indices to be used by each channel type by default.
    // Initialized in channel_manager_pick_copy_engines()
    //
    // Transfers of a given type may use a CE different from that with index
    // ce_to_use_by_type[type]. For example, transfers between NvLink GPU
    // peers may ignore the default CE and use a dedicated one instead.
    NvU32 ce_to_use_by_type[UVM_CHANNEL_TYPE_COUNT];

    struct
    {
        struct proc_dir_entry *channels_dir;
        struct proc_dir_entry *pending_pushes;
    } procfs;

    struct
    {
        NvU32 num_gpfifo_entries;
        UVM_BUFFER_LOCATION gpfifo_loc;
        UVM_BUFFER_LOCATION gpput_loc;
        UVM_BUFFER_LOCATION pushbuffer_loc;
    } conf;
};

// Create a channel manager for the GPU
//
// If with_procfs is true, also create the procfs entries for the pushbuffer.
// This is needed because some tests create temporary channel manager, but only
// only a single one can have its procfs entries created currently.
NV_STATUS uvm_channel_manager_create_common(uvm_gpu_t *gpu, bool with_procfs, uvm_channel_manager_t **manager_out);

// Create a channel manager for the GPU with procfs
static NV_STATUS uvm_channel_manager_create(uvm_gpu_t *gpu, uvm_channel_manager_t **manager_out)
{
    return uvm_channel_manager_create_common(gpu, true, manager_out);
}

// Create a channel manager for the GPU without procfs
static NV_STATUS uvm_channel_manager_create_no_procfs(uvm_gpu_t *gpu, uvm_channel_manager_t **manager_out)
{
    return uvm_channel_manager_create_common(gpu, false, manager_out);
}

// Destroy the channel manager
void uvm_channel_manager_destroy(uvm_channel_manager_t *channel_manager);

// Get the current status of the channel
// Returns NV_OK if the channel is in a good state and NV_ERR_RC_ERROR
// otherwise. Notably this never sets the global fatal error.
NV_STATUS uvm_channel_get_status(uvm_channel_t *channel);

// Check for channel errors
// Checks for channel errors by calling uvm_channel_get_status(). If an error
// occurred, sets the global fatal error and prints errors.
NV_STATUS uvm_channel_check_errors(uvm_channel_t *channel);

// Check errors on all channels in the channel manager
// Also includes uvm_global_get_status
NV_STATUS uvm_channel_manager_check_errors(uvm_channel_manager_t *channel_manager);

// Retrieve the GPFIFO entry that caused a channel error
// The channel has to be in error state prior to calling this function.
uvm_gpfifo_entry_t *uvm_channel_get_fatal_entry(uvm_channel_t *channel);

// Update progress of a specific channel
// Returns the number of still pending GPFIFO entries for that channel.
// Notably some of the pending GPFIFO entries might be already completed, but
// the update early-outs after completing a fixed number of them to spread the
// cost of the updates across calls.
NvU32 uvm_channel_update_progress(uvm_channel_t *channel);

// Update progress of all channels
// Returns the number of still pending GPFIFO entries for all channels.
// Notably some of the pending GPFIFO entries might be already completed, but
// the update early-outs after completing a fixed number of them to spread the
// cost of the updates across calls.
NvU32 uvm_channel_manager_update_progress(uvm_channel_manager_t *channel_manager);

// Wait for all channels to idle
// It waits for anything that is running, but doesn't prevent new work from
// beginning.
NV_STATUS uvm_channel_manager_wait(uvm_channel_manager_t *manager);

// Get the GPU tracking semaphore
uvm_gpu_semaphore_t *uvm_channel_get_tracking_semaphore(uvm_channel_t *channel);

// Channel's index within the manager's channel array.
static unsigned uvm_channel_get_index(const uvm_channel_manager_t *channel_manager, const uvm_channel_t *channel)
{
    return channel - channel_manager->channels;
}

// Check whether the channel completed a value
bool uvm_channel_is_value_completed(uvm_channel_t *channel, NvU64 value);

// Update and get the latest completed value by the channel
NvU64 uvm_channel_update_completed_value(uvm_channel_t *channel);

// Select and reserve a channel with the specified type for a push
// Channel type can be UVM_CHANNEL_TYPE_ANY to reserve any channel.
NV_STATUS uvm_channel_reserve_type(uvm_channel_manager_t *manager,
                                   uvm_channel_type_t type,
                                   uvm_channel_t **channel_out);

// Select and reserve a channel for a transfer from channel_manager->gpu to
// dst_gpu.
NV_STATUS uvm_channel_reserve_gpu_to_gpu(uvm_channel_manager_t *channel_manager,
                                         uvm_gpu_t *dst_gpu,
                                         uvm_channel_t **channel_out);

// Reserve a specific channel for a push
NV_STATUS uvm_channel_reserve(uvm_channel_t *channel);

// Find a channel that's available at the moment.
// Only really useful in tests.
uvm_channel_t *uvm_channel_manager_find_available_channel(uvm_channel_manager_t *channel_manager);

// Begin a push on a previously reserved channel
// Should be used by uvm_push_*() only.
NV_STATUS uvm_channel_begin_push(uvm_channel_t *channel, uvm_push_t *push);

// End a push
// Should be used by uvm_push_end() only.
void uvm_channel_end_push(uvm_push_t *push);

const char *uvm_channel_type_to_string(uvm_channel_type_t channel_type);

void uvm_channel_print_pending_pushes(uvm_channel_t *channel);

static uvm_gpu_t *uvm_channel_get_gpu(uvm_channel_t *channel)
{
    return channel->pool->manager->gpu;
}

NvU32 uvm_channel_update_progress_all(uvm_channel_t *channel);

// Helper to get the channel at the given index
// Returns NULL if index is greater or equal than the number of channels.
static uvm_channel_t *uvm_channel_get(uvm_channel_manager_t *manager, unsigned index)
{
    const unsigned num_channels = manager->num_channels;

    return (index < num_channels) ? manager->channels + index : NULL;
}

// Helper to get the successor of a given channel.
// Returns NULL if there is no successor, or the successor's index is equal
// or greater than the outer limit
static uvm_channel_t *uvm_channel_get_next(uvm_channel_manager_t *manager, uvm_channel_t *channel, unsigned outer)
{
    const unsigned next_index = uvm_channel_get_index(manager, channel) + 1;

    return (next_index < outer) ? uvm_channel_get(manager, next_index) : NULL;
}

static uvm_channel_t *uvm_channel_first(uvm_channel_manager_t *manager)
{
    return uvm_channel_get(manager, 0);
}

// Helper to iterate over the channels in a certain range.
// If the iterator body adds or removes channels to the manager, the behavior
// is undefined
#define uvm_for_each_channel_in_range(channel, manager, start, outer)          \
    for (channel = uvm_channel_get((manager), (start));                        \
         channel != NULL;                                                      \
         channel = uvm_channel_get_next((manager), channel, (outer)))

// Helper to iterate over all the channels.
#define uvm_for_each_channel(channel, manager) \
    uvm_for_each_channel_in_range(channel, (manager), 0, (manager)->num_channels)
#endif // __UVM8_CHANNEL_H__
