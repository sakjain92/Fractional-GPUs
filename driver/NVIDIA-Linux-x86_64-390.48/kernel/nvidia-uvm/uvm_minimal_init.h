/*******************************************************************************
    Copyright (c) 2013 NVIDIA Corporation

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

#ifndef _UVM_MINIMAL_INIT_H
#define _UVM_MINIMAL_INIT_H

// This file includes as little information and dependencies as possible, in order
// to support the uvm_unsupported.c case (which ideally is nearly self-sufficient),
// while avoiding the need to duplicate any definitions in uvm_unsupported.c.

#include "uvmtypes.h"
#include "uvm_linux_ioctl.h"

#define NVIDIA_UVM_DEVICE_NAME          "nvidia-uvm"

enum {
    NVIDIA_UVM_PRIMARY_MINOR_NUMBER = 0,
    NVIDIA_UVM_TOOLS_MINOR_NUMBER   = 1,
    // to ensure backward-compatiblity and correct counting, please insert any
    // new minor devices just above the following field:
    NVIDIA_UVM_NUM_MINOR_DEVICES
};

// Avoid directly pulling in Linux kernel files, because each user of this
// file will do that.
struct file;

NV_STATUS uvm_api_initialize(UVM_INITIALIZE_PARAMS *params, struct file *filp);

#endif // _UVM_MINIMAL_INIT_H
