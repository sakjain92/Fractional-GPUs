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

#ifdef __linux__
#include "uvm_linux.h"
#endif

#include "uvm_utils.h"

//
// uvm_utils.c
//
// This file contains code that works on all supported operating systems.
//

//
// This formats a GPU UUID, in a UVM-friendly way. That is, nearly the same as
// what nvidia-smi reports.  It will always prefix the UUID with UVM-GPU
// so that we know that we have a real, binary formatted UUID that will
// work in the UVM APIs.
//
// It comes out like this:
//
//     UVM-GPU-d802726c-df8d-a3c3-ec53-48bdec201c27
//
//  This routine will always null-terminate the string for you. This is true
//  even if the buffer was too small!
//
//  Return value is the number of non-null characters written.
//
// Note that if you were to let the NV2080_CTRL_CMD_GPU_GET_GID_INFO command
// return it's default format, which is ascii, not binary, then you would get
// this back:
//
//     GPU-d802726c-df8d-a3c3-ec53-48bdec201c27
//
//  ...which is actually a character string, and won't work for UVM API calls.
//  So it's very important to be able to see the difference.
//
//       

static char UvmDigitToHex(unsigned value)
{
    if (value >= 10)
        return value - 10 + 'a';
    else
        return value + '0';
}

int format_uuid_to_buffer(char * buffer, unsigned bufferLength, const NvProcessorUuid * pUuidStruct)
{
    char * str = buffer+8;
    unsigned i, dashMask = 1 << 4 | 1 << 6 | 1 << 8 | 1 << 10; 

    memcpy(buffer, "UVM-GPU-", 8);
    if (bufferLength < (8 /*prefix*/+ 16 * 2 /*digits*/ + 4 * 1 /*dashes*/ + 1 /*null*/)) 
        return *buffer = 0;

    for (i = 0; i < 16; i++)
    {
        *str ++ = UvmDigitToHex(pUuidStruct->uuid[i] >> 4);
        *str ++ = UvmDigitToHex(pUuidStruct->uuid[i] & 0xF);

        if (dashMask & (1 << (i+1)))
            *str++ = '-';
    }
    *str = 0;
    return (int)(str-buffer);
}
