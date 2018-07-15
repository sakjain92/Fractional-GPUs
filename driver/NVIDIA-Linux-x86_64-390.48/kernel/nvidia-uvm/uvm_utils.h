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

//
// uvm_utils.h
//
// This file contains declarations for "utility" routines, each of which works
// on all supported operating systems.
//
//

#ifndef _UVM_UTILS_H
#define _UVM_UTILS_H

#include "uvmtypes.h"
//
// This sort of routine is portable between OSes. Actually printing something is
// less portable, so routines such as print_uuid() will be found in the
// OS-specific files.
//
// Please see the .c file for detailed documentation.
//
#define UVM_GPU_UUID_TEXT_BUFFER_LENGTH (8+16*2+4+1)

int format_uuid_to_buffer(char * buffer, unsigned bufferLength, const NvProcessorUuid * pGpuUuid);

#endif // _UVM_UTILS_H

