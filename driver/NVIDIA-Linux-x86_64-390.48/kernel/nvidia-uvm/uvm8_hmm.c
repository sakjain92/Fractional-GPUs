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

#include "uvm8_hmm.h"

// You have to opt in, in order to use HMM. Once "HMM bringup" is complete,
// this should be reversed, so that HMM is enabled by default. See below
// (uvm_hmm_is_enabled) for further details on enabling HMM.
static int uvm_hmm = 0;
module_param(uvm_hmm, int, S_IRUGO);
MODULE_PARM_DESC(uvm_hmm, "Enable (1) or disable (0) HMM mode. Default: 0. "
                          "Ignored if CONFIG_HMM is not set, or if NEXT settings conflict with HMM.");

#if UVM_IS_CONFIG_HMM()

#include "uvm8_gpu.h"
#include "uvm8_va_space.h"

// You need all of these things, in order to actually run HMM:
//
//     1) An HMM kernel, with CONFIG_HMM set.
//
//     2) UVM Kernel module parameter set: uvm_hmm=1
//
//     3) ATS must not be enabled
//
bool uvm_hmm_is_enabled(void)
{
    return (uvm_hmm != 0) && (uvm8_ats_mode == 0);
}

// If ATS support is enabled, then HMM will be disabled, even if HMM was
// specifically requested via uvm_hmm kernel module parameter. Detect that case
// and print a warning to the unsuspecting developer.
void uvm_hmm_init(void)
{
    if ((uvm_hmm != 0) && (uvm8_ats_mode != 0)) {
        UVM_ERR_PRINT("uvm_hmm=%d (HMM was requested), ATS mode is also enabled, which is incompatible with HMM, "
                      "so HMM remains disabled\n", uvm_hmm);
    }
}

NV_STATUS uvm_hmm_device_register(uvm_gpu_t *gpu)
{
    struct hmm_device *device;

    if (!uvm_hmm_is_enabled())
        return NV_OK;

    device = hmm_device_new(gpu);

    if (IS_ERR_OR_NULL(device)) {
        if (IS_ERR(device))
            return errno_to_nv_status(PTR_ERR(device));
        else
            return NV_ERR_NO_MEMORY;
    }

    gpu->hmm_gpu.device = device;

    return NV_OK;
}

void uvm_hmm_device_unregister(uvm_gpu_t *gpu)
{
    if (!uvm_hmm_is_enabled())
        return;

    if (!gpu->hmm_gpu.device)
        return;

    hmm_device_put(gpu->hmm_gpu.device);

    gpu->hmm_gpu.device = NULL;
}

static void mirror_sync_cpu_device_pagetables(struct hmm_mirror *mirror,
                                              enum hmm_update_type update,
                                              unsigned long start,
                                              unsigned long end)
{
    // TODO: Bug 1750144: Implement this
}

static const struct hmm_mirror_ops mirror_ops = {
    .sync_cpu_device_pagetables = &mirror_sync_cpu_device_pagetables,
};

NV_STATUS uvm_hmm_mirror_register(uvm_va_space_t *va_space)
{
    int ret;

    if (!uvm_hmm_is_enabled())
        return NV_OK;

    uvm_assert_mmap_sem_locked_write(&current->mm->mmap_sem);
    uvm_assert_rwsem_locked_write(&va_space->lock);

    UVM_ASSERT_MSG(!(va_space->initialization_flags & UVM_INIT_FLAGS_DISABLE_HMM),
                   "uvm_hmm_mirror_register called, but HMM is disabled at runtime");

    va_space->hmm_va_space.mirror.ops = &mirror_ops;

    ret = hmm_mirror_register(&va_space->hmm_va_space.mirror, current->mm);
    if (ret != 0)
        return errno_to_nv_status(ret);

    return NV_OK;
}

void uvm_hmm_mirror_unregister(uvm_va_space_t *va_space)
{
    if (!uvm_hmm_is_enabled())
        return;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!va_space->hmm_va_space.mirror.hmm)
        return;

    hmm_mirror_unregister(&va_space->hmm_va_space.mirror);
}

#endif // UVM_IS_CONFIG_HMM()
