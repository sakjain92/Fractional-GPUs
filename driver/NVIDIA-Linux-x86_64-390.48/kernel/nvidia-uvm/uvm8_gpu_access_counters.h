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

#ifndef __UVM8_GPU_ACCESS_COUNTERS_H__
#define __UVM8_GPU_ACCESS_COUNTERS_H__

#include "uvm_common.h"
#include "uvm8_forward_decl.h"
#include "uvm8_test_ioctl.h"

NV_STATUS uvm_gpu_init_access_counters(uvm_gpu_t *gpu);
void uvm_gpu_deinit_access_counters(uvm_gpu_t *gpu);
bool uvm_gpu_access_counters_pending(uvm_gpu_t *gpu);

void uvm_gpu_service_access_counters(uvm_gpu_t *gpu);

void uvm_gpu_access_counter_buffer_flush(uvm_gpu_t *gpu);

NV_STATUS uvm8_test_reconfigure_access_counters(UVM_TEST_RECONFIGURE_ACCESS_COUNTERS_PARAMS *params,
                                                struct file *filp);
NV_STATUS uvm8_test_reset_access_counters(UVM_TEST_RESET_ACCESS_COUNTERS_PARAMS *params, struct file *filp);
NV_STATUS uvm8_test_set_ignore_access_counters(UVM_TEST_SET_IGNORE_ACCESS_COUNTERS_PARAMS *params, struct file *filp);

#endif // __UVM8_GPU_ACCESS_COUNTERS_H__
