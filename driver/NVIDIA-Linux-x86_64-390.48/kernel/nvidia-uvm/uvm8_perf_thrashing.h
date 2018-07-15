/*******************************************************************************
    Copyright (c) 2016 NVIDIA Corporation

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

#ifndef __UVM8_PERF_THRASHING_H__
#define __UVM8_PERF_THRASHING_H__

#include "uvm_linux.h"
#include "uvm8_forward_decl.h"
#include "uvm8_processors.h"
#include "uvm8_va_block_types.h"

typedef enum
{
    // No thrashing detected
    UVM_PERF_THRASHING_HINT_TYPE_NONE     = 0,

    // Map remotely to avoid future faults (does not help with revocations due
    // to system-wide atomics)
    UVM_PERF_THRASHING_HINT_TYPE_PIN      = 1,

    // Throttle execution of the calling processor (this can be implemented by
    // sleeping or handing other faults)
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE = 2,

    // TODO: Bug 1765618 [UVM8] Implement read-duplication
    // Add a thrashing hint type to read-duplicate a page when it is being
    // accessed read-only from different processors
} uvm_perf_thrashing_hint_type_t;

typedef struct
{
    uvm_perf_thrashing_hint_type_t type;

    union
    {
        struct
        {
            // Map to this processor, which must be accessible, at least, from the calling
            // processor
            uvm_processor_id_t residency;

            // Processors to be mapped, when possible, to the new residency
            uvm_processor_mask_t processors;
        } pin;

        struct
        {
            // Absolute timestamp in ns after which the throttled processor is
            // allowed to start servicing faults on the thrashing page.
            NvU64 end_time_stamp;
        } throttle;
    };
} uvm_perf_thrashing_hint_t;

// Tunables for thrashing detection and prevention
extern bool g_uvm_perf_thrashing_enable;
extern unsigned g_uvm_perf_thrashing_threshold;
extern NvU64 g_uvm_perf_thrashing_lapse_ns;
extern NvU64 g_uvm_perf_thrashing_nap_ns;
extern NvU64 g_uvm_perf_thrashing_epoch_ns;

// Obtain a hint to prevent thrashing on the page with given address
uvm_perf_thrashing_hint_t uvm_perf_thrashing_get_hint(uvm_va_block_t *va_block, NvU64 address,
                                                      uvm_processor_id_t requester);

// Obtain a pointer to a mask with the processors that are thrashing on the given page. This function
// assumes that thrashing has been just reported on the page. It will fail otherwise.
uvm_processor_mask_t *uvm_perf_thrashing_get_thrashing_processors(uvm_va_block_t *va_block, NvU64 address);

const uvm_page_mask_t *uvm_perf_thrashing_get_thrashing_pages(uvm_va_block_t *va_block);

// Returns true if any page in the block is thrashing, or false otherwise
bool uvm_perf_thrashing_is_block_thrashing(uvm_va_block_t *va_block);

// When the preferred location changes, we unmap pinned pages so that
// processors will fault again and the policy will see the new value. It is
// responsibility of the caller to retry if the function returns
// NV_WARN_MORE_PROCESSING_REQUIRED
NV_STATUS uvm_perf_thrashing_change_preferred_location(uvm_va_block_t *va_block,
                                                       uvm_processor_id_t new_preferred_location,
                                                       uvm_processor_id_t old_preferred_location);

// Global initialization/cleanup functions
NV_STATUS uvm_perf_thrashing_init(void);
void uvm_perf_thrashing_exit(void);

// VA space Initialization/cleanup functions
NV_STATUS uvm_perf_thrashing_load(uvm_va_space_t *va_space);
void uvm_perf_thrashing_unload(uvm_va_space_t *va_space);

#endif
