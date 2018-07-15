/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include <linux/pci.h>
#include "nv-frontend.h"

extern const int nv_multiple_kernel_modules;
extern unsigned const int nv_module_instance;
extern const char *nv_device_name;
extern const char *nvidia_stack_cache_name;
extern const char *nvidia_pte_cache_name;
extern const char *nvidia_p2p_page_cache_name;
extern struct pci_driver nv_pci_driver;
extern nvidia_module_t nv_fops;

int nv_pci_register_driver(struct pci_driver *);
