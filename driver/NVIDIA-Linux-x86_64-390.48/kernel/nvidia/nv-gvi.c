
/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"
#include "nv-reg.h"
#include "rmil.h"

irqreturn_t nv_gvi_kern_isr(
    int   irq,
    void *arg
#if !defined(NV_IRQ_HANDLER_T_PRESENT) || (NV_IRQ_HANDLER_T_ARGUMENT_COUNT == 3)
    ,struct pt_regs *regs
#endif
)
{
    nv_linux_state_t *nvl = (void *) arg;
    nv_state_t *nv = NV_STATE_PTR(nvl);
    NvU32 need_to_run_bottom_half = 0;
    BOOL ret = TRUE;

    ret = rm_gvi_isr(nvl->sp[NV_DEV_STACK_ISR], nv, &need_to_run_bottom_half);
    if (need_to_run_bottom_half && !(nv->flags & NV_FLAG_GVI_IN_SUSPEND))
    {
        NV_WORKQUEUE_SCHEDULE(&nvl->work.task);
    }

    return IRQ_RETVAL(ret);
}

void nv_gvi_kern_bh(
    NV_WORKQUEUE_DATA_T *data
)
{
    nv_state_t *nv;
    nv_work_t *work = NV_WORKQUEUE_UNPACK_DATA(data);
    nv_linux_state_t *nvl = (nv_linux_state_t *)work->data;
    nv  = NV_STATE_PTR(nvl);

    rm_gvi_bh(nvl->sp[NV_DEV_STACK_ISR_BH], nv);
}

#if defined(NV_PM_SUPPORT_DEVICE_DRIVER_MODEL)

int nv_gvi_kern_suspend(
    struct pci_dev *dev,
    pm_message_t state
)
{
    nv_state_t *nv;
    nv_linux_state_t *lnv = NULL;
    int status = NV_OK;
    nvidia_stack_t *sp = NULL;

    nv_printf(NV_DBG_INFO, "NVGVI: Begin suspending GVI device!\n");
    lnv = pci_get_drvdata(dev);

    if ((!lnv) || (lnv->dev != dev))
    {
        nv_printf(NV_DBG_WARNINGS, "NVGVI: PM: invalid device!\n");
        return -1;
    }

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return -1;
    }

    nv = NV_STATE_PTR(lnv);

    status = rm_shutdown_gvi_device(sp, nv);
    if (status != 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVGVI: failed to stop gvi!\n");
        goto failed;
    }
        
    nv->flags |= NV_FLAG_GVI_IN_SUSPEND;

    NV_WORKQUEUE_FLUSH();

    status = rm_gvi_suspend(sp, nv);
    if (status != 0)
    {
        nv->flags &= ~NV_FLAG_GVI_IN_SUSPEND;
        nv_printf(NV_DBG_ERRORS, "NVGVI: failed to suspend gvi!\n");
        goto failed;
    }

    nv_printf(NV_DBG_INFO, "NVGVI: End suspending GVI device!\n");
failed:    
    nv_kmem_cache_free_stack(sp);
    return status;
}

int nv_gvi_kern_resume(
    struct pci_dev *dev
)
{
    nv_state_t *nv;
    nv_linux_state_t *lnv = NULL;
    nvidia_stack_t *sp = NULL;
    int status = NV_OK;

    nv_printf(NV_DBG_INFO, "NVGVI: Begin resuming GVI device!\n");
    lnv = pci_get_drvdata(dev);

    if ((!lnv) || (lnv->dev != dev))
    {
        nv_printf(NV_DBG_WARNINGS, "NVGVI: PM: invalid device!\n");
        return -1;
    }

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return -1;
    }

    nv = NV_STATE_PTR(lnv);
    status = rm_gvi_resume(sp, nv);
    if (status == NV_OK)
        nv->flags &= ~NV_FLAG_GVI_IN_SUSPEND;
    nv_kmem_cache_free_stack(sp);
    return status;
}

#endif
