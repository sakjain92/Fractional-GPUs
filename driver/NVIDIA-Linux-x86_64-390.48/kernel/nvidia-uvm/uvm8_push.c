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

#include "uvm8_push.h"
#include "uvm8_channel.h"
#include "uvm8_hal.h"
#include "uvm8_kvmalloc.h"
#include "uvm_linux.h"

// This parameter enables push description tracking in push info. It's enabled
// by default for debug and develop builds and disabled for release builds.
static int uvm_debug_enable_push_desc = UVM_IS_DEBUG() || UVM_IS_DEVELOP();
module_param(uvm_debug_enable_push_desc, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_debug_enable_push_desc, "Enable push description tracking");

void uvm_push_set_flag(uvm_push_t *push, uvm_push_flag_t flag)
{
    UVM_ASSERT_MSG(flag < UVM_PUSH_FLAG_COUNT, "flag %u\n", (unsigned)flag);

    __set_bit(flag, push->flags);
}

bool uvm_push_get_and_reset_flag(uvm_push_t *push, uvm_push_flag_t flag)
{
    UVM_ASSERT_MSG(flag < UVM_PUSH_FLAG_COUNT, "flag %u\n", (unsigned)flag);

    return __test_and_clear_bit(flag, push->flags);
}

static NV_STATUS push_begin_on_channel_common(uvm_channel_t *channel, uvm_push_t *push)
{
    memset(push, 0, sizeof(*push));

    push->gpu = uvm_channel_get_gpu(channel);

    return uvm_channel_begin_push(channel, push);
}

NV_STATUS __uvm_push_begin_on_channel(uvm_channel_t *channel, uvm_push_t *push)
{
    NV_STATUS status;

    // Wait to claim a specific channel
    status = uvm_channel_reserve(channel);
    if (status != NV_OK)
        return status;

    return push_begin_on_channel_common(channel, push);
}

NV_STATUS __uvm_push_begin_acquire(uvm_channel_manager_t *manager,
                                   uvm_channel_type_t channel_type,
                                   uvm_gpu_t *dst_gpu,
                                   uvm_tracker_t *tracker,
                                   uvm_push_t *push)
{
    NV_STATUS status;
    uvm_channel_t *channel;

    // Pick a channel and reserve a GPFIFO entry
    // TODO: Bug 1764953: use the dependencies in the tracker to pick a channel
    //       in a smarter way.
    if (dst_gpu == NULL)
        status = uvm_channel_reserve_type(manager, channel_type, &channel);
    else
        status = uvm_channel_reserve_gpu_to_gpu(manager, dst_gpu, &channel);

    if (status != NV_OK)
        return status;

    UVM_ASSERT(channel);

    status = push_begin_on_channel_common(channel, push);
    if (status != NV_OK)
        return status;

    uvm_push_acquire_tracker(push, tracker);

    return NV_OK;
}

bool uvm_push_info_is_tracking_descriptions()
{
    return uvm_debug_enable_push_desc != 0;
}

void __uvm_push_fill_info(uvm_push_t *push, const char *filename, const char *function, int line, const char *format, va_list args)
{
    uvm_channel_t *channel = push->channel;
    uvm_push_info_t *push_info;

    channel = push->channel;

    push_info = uvm_push_info_from_push(push);

    push_info->filename = kbasename(filename);
    push_info->function = function;
    push_info->line = line;

    if (uvm_push_info_is_tracking_descriptions())
        vsnprintf(push_info->description, sizeof(push_info->description), format, args);
}

void uvm_push_end(uvm_push_t *push)
{
    uvm_push_flag_t flag;
    uvm_channel_end_push(push);

    flag = find_first_bit(push->flags, UVM_PUSH_FLAG_COUNT);

    // All flags should be reset by the end of the push
    UVM_ASSERT_MSG(flag == UVM_PUSH_FLAG_COUNT, "first flag set %d\n", flag);
}

void uvm_push_acquire_tracker_entry(uvm_push_t *push, uvm_tracker_entry_t *tracker_entry)
{
    uvm_channel_t *channel = push->channel;
    uvm_channel_manager_t *channel_manager = channel->pool->manager;
    uvm_gpu_t *gpu = channel_manager->gpu;

    UVM_ASSERT(push != NULL);
    UVM_ASSERT(tracker_entry != NULL);

    if (tracker_entry->channel == NULL)
        return;

    if (channel == tracker_entry->channel)
        return;

    gpu->host_hal->semaphore_acquire(push, uvm_channel_get_tracking_semaphore(tracker_entry->channel), (NvU32)tracker_entry->value);
}

void uvm_push_acquire_tracker(uvm_push_t *push, uvm_tracker_t *tracker)
{
    uvm_tracker_entry_t *entry;

    UVM_ASSERT(push != NULL);

    if (tracker == NULL)
        return;

    uvm_tracker_remove_completed(tracker);

    for_each_tracker_entry(entry, tracker)
        uvm_push_acquire_tracker_entry(push, entry);
}

void uvm_push_get_tracker_entry(uvm_push_t *push, uvm_tracker_entry_t *entry)
{
    UVM_ASSERT(push->channel_tracking_value != 0);
    UVM_ASSERT(push->channel != NULL);

    entry->channel = push->channel;
    entry->value = push->channel_tracking_value;
}

NV_STATUS uvm_push_wait(uvm_push_t *push)
{
    uvm_tracker_entry_t entry;
    uvm_push_get_tracker_entry(push, &entry);

    return uvm_tracker_wait_for_entry(&entry);
}

NV_STATUS uvm_push_end_and_wait(uvm_push_t *push)
{
    uvm_push_end(push);

    return uvm_push_wait(push);
}

NV_STATUS uvm_push_begin_fake(uvm_gpu_t *gpu, uvm_push_t *push)
{
    memset(push, 0, sizeof(*push));
    push->begin = (NvU32 *)uvm_kvmalloc(UVM_MAX_PUSH_SIZE);
    if (!push->begin)
        return NV_ERR_NO_MEMORY;

    push->next = push->begin;
    push->gpu = gpu;

    return NV_OK;
}

void uvm_push_end_fake(uvm_push_t *push)
{
    uvm_kvfree(push->begin);
    push->begin = NULL;
}

size_t uvm_push_inline_data_size(uvm_push_inline_data_t *data)
{
    return data->next_data - (char*)(data->push->next + 1);
}

void uvm_push_inline_data_begin(uvm_push_t *push, uvm_push_inline_data_t *data)
{
    data->push = push;
    // +1 for the NOOP method inserted at inline_data_end()
    data->next_data = (char*)(push->next + 1);
}

void *uvm_push_inline_data_get(uvm_push_inline_data_t *data, size_t size)
{
    void *buffer = data->next_data;

    UVM_ASSERT_MSG(uvm_push_get_size(data->push) + uvm_push_inline_data_size(data) + UVM_METHOD_SIZE + size <= UVM_MAX_PUSH_SIZE,
            "push size %u inline data size %zu new data size %zu max push %u\n",
            uvm_push_get_size(data->push), uvm_push_inline_data_size(data), size, UVM_MAX_PUSH_SIZE);
    UVM_ASSERT_MSG(uvm_push_inline_data_size(data) + size <= UVM_PUSH_INLINE_DATA_MAX_SIZE,
            "inline data size %zu new data size %zu max %u\n",
            uvm_push_inline_data_size(data), size, UVM_PUSH_INLINE_DATA_MAX_SIZE);

    data->next_data += size;

    return buffer;
}

void *uvm_push_inline_data_get_aligned(uvm_push_inline_data_t *data, size_t size, size_t alignment)
{
    NvU64 next_ptr = (NvU64)(uintptr_t)data->next_data;
    size_t offset = 0;
    char *buffer;

    UVM_ASSERT_MSG(IS_ALIGNED(alignment, UVM_METHOD_SIZE), "alignment %zu\n", alignment);

    offset = UVM_ALIGN_UP(next_ptr, alignment) - next_ptr;

    buffer = (char *)uvm_push_inline_data_get(data, size + offset);
    return buffer + offset;
}

uvm_gpu_address_t uvm_push_inline_data_end(uvm_push_inline_data_t *data)
{
    uvm_push_t *push = data->push;
    uvm_gpu_address_t inline_data_addr;

    // Round up the inline data size to the method size
    size_t noop_size = roundup(uvm_push_inline_data_size(data), UVM_METHOD_SIZE);

    if (push->channel) {
        NvU64 data_gpu_va = uvm_pushbuffer_get_gpu_va_for_push(push->channel->pool->manager->pushbuffer, push);
        // offset of where the inline data started and + 1 for the noop method
        data_gpu_va += (push->next - push->begin + 1) * UVM_METHOD_SIZE;

        inline_data_addr = uvm_gpu_address_virtual(data_gpu_va);
    }
    else {
        // Fake push, just return the CPU address.
        inline_data_addr = uvm_gpu_address_virtual((NvU64)(unsigned long)(push->next + 1));
    }

    // This will place a noop right before the inline data that was written.
    // Plus UVM_METHOD_SIZE for the noop method itself.
    uvm_push_get_gpu(push)->host_hal->noop(push, noop_size + UVM_METHOD_SIZE);

    return inline_data_addr;
}

void *uvm_push_get_single_inline_buffer_aligned(uvm_push_t *push, size_t size, size_t alignment, uvm_gpu_address_t *gpu_address)
{
    uvm_push_inline_data_t data;
    void *buffer;

    uvm_push_inline_data_begin(push, &data);
    buffer = uvm_push_inline_data_get_aligned(&data, size, alignment);
    *gpu_address = uvm_push_inline_data_end(&data);

    gpu_address->address = UVM_ALIGN_UP(gpu_address->address, alignment);

    return buffer;
}

void *uvm_push_get_single_inline_buffer(uvm_push_t *push, size_t size, uvm_gpu_address_t *gpu_address)
{
    return uvm_push_get_single_inline_buffer_aligned(push, size, UVM_METHOD_SIZE, gpu_address);
}

NvU64 *uvm_push_timestamp(uvm_push_t *push)
{
    uvm_gpu_t *gpu = uvm_push_get_gpu(push);
    const size_t timestamp_size = 16;
    NvU64 *timestamp;
    uvm_gpu_address_t address;

    timestamp = (NvU64 *)uvm_push_get_single_inline_buffer_aligned(push, timestamp_size, timestamp_size, &address);
    // Timestamp is in the second half of the 16 byte semaphore release
    timestamp += 1;

    gpu->ce_hal->semaphore_timestamp(push, address.address);

    return timestamp;
}
