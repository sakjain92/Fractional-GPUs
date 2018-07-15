/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include "nv-modeset-interface.h"

#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"
#include "nvstatus.h"
#include "nv.h"

static const nvidia_modeset_callbacks_t *nv_modeset_callbacks;

static int nvidia_modeset_rm_ops_alloc_stack(nvidia_stack_t **sp)
{
    return nv_kmem_cache_alloc_stack(sp);
}

static void nvidia_modeset_rm_ops_free_stack(nvidia_stack_t *sp)
{
    if (sp != NULL)
    {
        nv_kmem_cache_free_stack(sp);
    }
}

static int nvidia_modeset_set_callbacks(const nvidia_modeset_callbacks_t *cb)
{
    if ((nv_modeset_callbacks != NULL && cb != NULL) ||
        (nv_modeset_callbacks == NULL && cb == NULL))
    {
        return -EINVAL;
    }

    nv_modeset_callbacks = cb;
    return 0;
}

void nvidia_modeset_suspend(NvU32 gpuId)
{
    if (nv_modeset_callbacks)
    {
        nv_modeset_callbacks->suspend(gpuId);
    }
}

void nvidia_modeset_resume(NvU32 gpuId)
{
    if (nv_modeset_callbacks)
    {
        nv_modeset_callbacks->resume(gpuId);
    }
}

static NvU32 nvidia_modeset_enumerate_gpus(nv_gpu_info_t *gpu_info)
{
    nv_linux_state_t *nvl;
    unsigned int count;

    LOCK_NV_LINUX_DEVICES();

    count = 0;

    for (nvl = nv_linux_devices; nvl != NULL; nvl = nvl->next)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);

        /*
         * The gpu_info[] array has NV_MAX_GPUS elements.  Fail if there
         * are more GPUs than that.
         */
        if (count >= NV_MAX_GPUS) {
            nv_printf(NV_DBG_WARNINGS, "NVRM: More than %d GPUs found.",
                      NV_MAX_GPUS);
            count = 0;
            break;
        }

        gpu_info[count].gpu_id = nv->gpu_id;

        gpu_info[count].pci_info.domain   = nv->pci_info.domain;
        gpu_info[count].pci_info.bus      = nv->pci_info.bus;
        gpu_info[count].pci_info.slot     = nv->pci_info.slot;
        gpu_info[count].pci_info.function = nv->pci_info.function;

        gpu_info[count].os_dev_ptr = nvl->dev;

        count++;
    }

    UNLOCK_NV_LINUX_DEVICES();

    return count;
}

NV_STATUS nvidia_get_rm_ops(nvidia_modeset_rm_ops_t *rm_ops)
{
    const nvidia_modeset_rm_ops_t local_rm_ops = {
        .version_string = NV_VERSION_STRING,
        .system_info    = {
            .allow_write_combining = NV_FALSE,
        },
        .alloc_stack    = nvidia_modeset_rm_ops_alloc_stack,
        .free_stack     = nvidia_modeset_rm_ops_free_stack,
        .enumerate_gpus = nvidia_modeset_enumerate_gpus,
        .open_gpu       = nvidia_dev_get,
        .close_gpu      = nvidia_dev_put,
        .op             = rm_kernel_rmapi_op, /* provided by nv-kernel.o */
        .set_callbacks  = nvidia_modeset_set_callbacks,
    };

    if (strcmp(rm_ops->version_string, NV_VERSION_STRING) != 0)
    {
        rm_ops->version_string = NV_VERSION_STRING;
        return NV_ERR_GENERIC;
    }

    *rm_ops = local_rm_ops;

    if (NV_ALLOW_WRITE_COMBINING(NV_MEMORY_TYPE_FRAMEBUFFER)) {
        rm_ops->system_info.allow_write_combining = NV_TRUE;
    }

    return NV_OK;
}

EXPORT_SYMBOL(nvidia_get_rm_ops);
