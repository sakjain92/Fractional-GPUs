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

#include "uvm8_hal.h"
#include "uvm8_push.h"
#include "clc0b5.h"

void uvm_hal_pascal_ce_offset_out(uvm_push_t *push, NvU64 offset_out)
{
    NV_PUSH_2U(C0B5, OFFSET_OUT_UPPER, HWVALUE(C0B5, OFFSET_OUT_UPPER, UPPER, NvOffset_HI32(offset_out)),
                     OFFSET_OUT_LOWER, HWVALUE(C0B5, OFFSET_OUT_LOWER, VALUE, NvOffset_LO32(offset_out)));
}

void uvm_hal_pascal_ce_offset_in_out(uvm_push_t *push, NvU64 offset_in, NvU64 offset_out)
{
    NV_PUSH_4U(C0B5, OFFSET_IN_UPPER,  HWVALUE(C0B5, OFFSET_IN_UPPER,  UPPER, NvOffset_HI32(offset_in)),
                     OFFSET_IN_LOWER,  HWVALUE(C0B5, OFFSET_IN_LOWER,  VALUE, NvOffset_LO32(offset_in)),
                     OFFSET_OUT_UPPER, HWVALUE(C0B5, OFFSET_OUT_UPPER, UPPER, NvOffset_HI32(offset_out)),
                     OFFSET_OUT_LOWER, HWVALUE(C0B5, OFFSET_OUT_LOWER, VALUE, NvOffset_LO32(offset_out)));
}
