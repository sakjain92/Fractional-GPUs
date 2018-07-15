/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_MODESET_INTERFACE_H_
#define _NV_MODESET_INTERFACE_H_

/*
 * This file defines the interface between the nvidia and
 * nvidia-modeset UNIX kernel modules.
 *
 * The nvidia-modeset kernel module calls the nvidia kernel module's
 * nvidia_get_rm_ops() function to get the RM API function pointers
 * which it will need.
 */

#include "nvstatus.h"
#include "cpuopsys.h" /* NVCPU_IS_X86 */

#include "nv-gpu-info.h"

/*
 * nvidia_modeset_rm_ops_t::op gets assigned a function pointer from
 * core RM, which uses the calling convention of arguments on the
 * stack.  Make sure the caller doesn't place arguments in registers.
 */
#if NVCPU_IS_X86
  #define NV_MODESET_INTERFACE_API_CALL __attribute__((regparm(0)))
#else
  #define NV_MODESET_INTERFACE_API_CALL
#endif

/*
 * nvidia_stack_s is defined in nv.h, which pulls in a lot of other
 * dependencies.  The nvidia-modeset kernel module doesn't need to
 * dereference the nvidia_stack_s pointer, so just treat is as an
 * opaque pointer for purposes of this API definition.
 */
typedef struct nvidia_stack_s *nvidia_modeset_stack_ptr;

/*
 * Callback functions from the RM OS interface layer into the NVKMS OS interface
 * layer.
 *
 * These functions should be called without the RM lock held, using the kernel's
 * native calling convention.
 */
typedef struct {
    /*
     * Suspend & resume callbacks.  Note that these are called once per GPU.
     */
    void (*suspend)(NvU32 gpu_id);
    void (*resume)(NvU32 gpu_id);
} nvidia_modeset_callbacks_t;

/*
 * The RM API entry points which the nvidia-modeset kernel module should
 * call in the nvidia kernel module.
 */

typedef struct {
    /*
     * The nvidia-modeset kernel module should assign version_string
     * before passing the structure to the nvidia kernel module, so
     * that a version match can be confirmed: it is not supported to
     * mix nvidia and nvidia-modeset kernel modules from different
     * releases.
     */
    const char *version_string;

    /*
     * Return system information.
     */
    struct {
        /* Availability of write combining support for video memory */
        NvBool allow_write_combining;
    } system_info;

    /*
     * Allocate and free an nvidia_stack_t to pass into
     * nvidia_modeset_rm_ops_t::op().  An nvidia_stack_t must only be
     * used by one thread at a time.
     *
     * Note that on architectures where an alternate stack is not
     * used, alloc_stack() will set sp=NULL even when it returns 0
     * (success).  I.e., check the return value, not the sp value.
     */
    int (*alloc_stack)(nvidia_modeset_stack_ptr *sp);
    void (*free_stack)(nvidia_modeset_stack_ptr sp);

    /*
     * Enumerate list of gpus probed by nvidia driver.
     *
     * gpu_info is an array of NVIDIA_MAX_GPUS elements. The number of GPUs
     * in the system is returned.
     */
    NvU32 (*enumerate_gpus)(nv_gpu_info_t *gpu_info);

    /*
     * {open,close}_gpu() raise and lower the reference count of the
     * specified GPU.  This is equivalent to opening and closing a
     * /dev/nvidiaN device file from user-space.
     */
    int (*open_gpu)(NvU32 gpu_id, nvidia_modeset_stack_ptr sp);
    void (*close_gpu)(NvU32 gpu_id, nvidia_modeset_stack_ptr sp);

    void NV_MODESET_INTERFACE_API_CALL (*op)
        (nvidia_modeset_stack_ptr sp, void *ops_cmd);

    int (*set_callbacks)(const nvidia_modeset_callbacks_t *cb);

} nvidia_modeset_rm_ops_t;

NV_STATUS nvidia_get_rm_ops(nvidia_modeset_rm_ops_t *rm_ops);

#endif /* _NV_MODESET_INTERFACE_H_ */
