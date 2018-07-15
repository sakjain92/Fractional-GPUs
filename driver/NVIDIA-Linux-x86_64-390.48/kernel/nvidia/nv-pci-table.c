/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include <linux/kernel.h>
#include <linux/module.h>

#include "nv-pci-table.h"

struct pci_device_id nv_pci_table[] = {
    {
        .vendor      = PCI_VENDOR_ID_NVIDIA,
        .device      = PCI_ANY_ID,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_DISPLAY_VGA << 8),
        .class_mask  = ~0
    },
    {
        .vendor      = PCI_VENDOR_ID_NVIDIA,
        .device      = PCI_ANY_ID,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_DISPLAY_3D << 8),
        .class_mask  = ~0
    },
    {
        .vendor      = PCI_VENDOR_ID_NVIDIA,
        .device      = 0x0e00,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_MULTIMEDIA_OTHER << 8),
        .class_mask  = ~0
    },
    { }
};

MODULE_DEVICE_TABLE(pci, nv_pci_table);
