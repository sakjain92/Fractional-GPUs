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

#include "uvm_linux.h"
#include "uvm8_perf_heuristics.h"
#include "uvm8_perf_thrashing.h"
#include "uvm8_perf_prefetch.h"

NV_STATUS uvm_perf_heuristics_init()
{
    NV_STATUS status;

    status = uvm_perf_thrashing_init();
    if (status != NV_OK)
        return status;

    status = uvm_perf_prefetch_init();
    if (status != NV_OK)
        return status;

    return NV_OK;
}

void uvm_perf_heuristics_exit()
{
    uvm_perf_prefetch_exit();
    uvm_perf_thrashing_exit();
}

NV_STATUS uvm_perf_heuristics_load(uvm_va_space_t *va_space)
{
    NV_STATUS status;

    status = uvm_perf_thrashing_load(va_space);
    if (status != NV_OK)
        return status;
    status = uvm_perf_prefetch_load(va_space);
    if (status != NV_OK)
        return status;

    return NV_OK;
}

void uvm_perf_heuristics_unload(uvm_va_space_t *va_space)
{
    uvm_perf_prefetch_unload(va_space);
    uvm_perf_thrashing_unload(va_space);
}
