/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"
#include "nv-frontend.h"
#include "nv-pci-table.h"

#define MODULE_BASE_NAME "nvidia"
#define MODULE_INSTANCE_NUMBER 0
#define MODULE_INSTANCE_STRING ""
#define MULTIPLE_KERNEL_MODULES NV_FALSE
#define MODULE_NAME MODULE_BASE_NAME MODULE_INSTANCE_STRING

extern struct pci_error_handlers nv_pci_error_handlers;

/* instance-specific variables */

const int nv_multiple_kernel_modules = MULTIPLE_KERNEL_MODULES;
const unsigned int nv_module_instance = MODULE_INSTANCE_NUMBER;
const char *nv_device_name = MODULE_NAME;
const char *nvidia_stack_cache_name = MODULE_NAME "_stack_cache";
const char *nvidia_pte_cache_name = MODULE_NAME "_pte_cache";
const char *nvidia_p2p_page_cache_name = MODULE_NAME "_p2p_page_cache";

/* These structs are statically initialized with instance-specific strings */

struct pci_driver nv_pci_driver = {
    .name     = MODULE_NAME,
    .id_table = nv_pci_table,
    .probe    = nvidia_probe,
    .remove   = nvidia_remove,
#if defined(NV_PM_SUPPORT_DEVICE_DRIVER_MODEL)
    .suspend  = nvidia_suspend,
    .resume   = nvidia_resume,
#endif
#if defined(NV_PCI_ERROR_RECOVERY)
    .err_handler = &nv_pci_error_handlers,
#endif
};

/* character device entry points*/
nvidia_module_t nv_fops = {
    .owner       = THIS_MODULE,
    .module_name = MODULE_NAME,
    .open        = nvidia_open,
    .close       = nvidia_close,
    .ioctl       = nvidia_ioctl,
    .mmap        = nvidia_mmap,
    .poll        = nvidia_poll,
};

/*
 * Helper function that can be called from common source files, since
 * pci_register_driver depends on KBUILD_MODNAME, which is not defined when
 * building C files that are shared between multiple modules.
 */

int
nv_pci_register_driver(
    struct pci_driver *driver
)
{
    return pci_register_driver(driver);
}
