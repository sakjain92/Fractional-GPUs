/*******************************************************************************
    Copyright (c) 2016, 2016 NVIDIA Corporation

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

#ifndef _UVM8_HMM_H_
#define _UVM8_HMM_H_

#include "nvtypes.h"
#include "uvm8_forward_decl.h"
#include "uvm_linux.h"

#if UVM_IS_CONFIG_HMM()
typedef struct
{
    struct hmm_device *device;
} uvm_hmm_gpu_t;

typedef struct
{
    struct hmm_mirror mirror;
} uvm_hmm_va_space_t;
#endif

#if UVM_IS_CONFIG_HMM()
    // Tells whether HMM is enabled, driver-wide.
    // If it is not enabled, all of the functions below are no-ops.
    bool uvm_hmm_is_enabled(void);

    // Initializes HMM support globally for this driver instance.
    void uvm_hmm_init(void);

    // Creates a new HMM device and binds it to the GPU.
    // The HMM device will manage page migrations associated with this GPU.
    NV_STATUS uvm_hmm_device_register(uvm_gpu_t *gpu);

    // Releases the HMM device for this GPU.
    void uvm_hmm_device_unregister(uvm_gpu_t *gpu);

    // Starts mirroring the HMM CPU table to the VA space.
    // Retains current->mm.
    // Locking: mmap_sem must be held in write mode.
    //          va_space lock must be held in write mode.
    NV_STATUS uvm_hmm_mirror_register(uvm_va_space_t *va_space);

    // Stops mirroring the HMM CPU table to the VA space.
    // Releases mm_struct retained by uvm_hmm_mirror_register.
    // Locking: va_space lock must be held in write mode.
    void uvm_hmm_mirror_unregister(uvm_va_space_t *va_space);

#else
    static bool uvm_hmm_is_enabled(void)
    {
        return false;
    }

    static void uvm_hmm_init(void)
    {
    }

    static NV_STATUS uvm_hmm_device_register(uvm_gpu_t *gpu)
    {
        return NV_OK;
    }

    static void uvm_hmm_device_unregister(uvm_gpu_t *gpu)
    {
    }

    static NV_STATUS uvm_hmm_mirror_register(uvm_va_space_t *va_space)
    {
        return NV_OK;
    }

    static void uvm_hmm_mirror_unregister(uvm_va_space_t *va_space)
    {
    }

#endif

#endif  // _UVM8_HMM_H_
