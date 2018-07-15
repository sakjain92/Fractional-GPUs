/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2013 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#define  __NO_VERSION__
#include "nv-misc.h"

#include "os-interface.h"
#include "nv-linux.h"

void* NV_API_CALL os_pci_init_handle(
    NvU32 domain,
    NvU8  bus,
    NvU8  slot,
    NvU8  function,
    NvU16 *vendor,
    NvU16 *device
)
{
    struct pci_dev *dev;
    unsigned int devfn = PCI_DEVFN(slot, function);

    if (!NV_MAY_SLEEP())
        return NULL;

    dev = NV_GET_DOMAIN_BUS_AND_SLOT(domain, bus, devfn);
    if (dev != NULL)
    {
        if (vendor) *vendor = dev->vendor;
        if (device) *device = dev->device;
        NV_PCI_DEV_PUT(dev); /* XXX Fix me! (hotplug) */
    }
    return (void *) dev;
}

NV_STATUS NV_API_CALL os_pci_read_byte(
    void *handle,
    NvU32 offset,
    NvU8 *pReturnValue
)
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
    {
        *pReturnValue = 0xff;
        return NV_ERR_NOT_SUPPORTED;
    }
    pci_read_config_byte( (struct pci_dev *) handle, offset, pReturnValue);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_pci_read_word(
    void *handle,
    NvU32 offset,
    NvU16 *pReturnValue
)
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
    {
        *pReturnValue = 0xffff;
        return NV_ERR_NOT_SUPPORTED;
    }
    pci_read_config_word( (struct pci_dev *) handle, offset, pReturnValue);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_pci_read_dword(
    void *handle,
    NvU32 offset,
    NvU32 *pReturnValue
) 
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
    {
        *pReturnValue = 0xffffffff;
        return NV_ERR_NOT_SUPPORTED;
    }
    pci_read_config_dword( (struct pci_dev *) handle, offset, pReturnValue);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_pci_write_byte(
    void *handle,
    NvU32 offset,
    NvU8 value
)
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
        return NV_ERR_NOT_SUPPORTED;

    pci_write_config_byte( (struct pci_dev *) handle, offset, value);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_pci_write_word(
    void *handle,
    NvU32 offset,
    NvU16 value
)
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
        return NV_ERR_NOT_SUPPORTED;

    pci_write_config_word( (struct pci_dev *) handle, offset, value);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_pci_write_dword(
    void *handle,
    NvU32 offset,
    NvU32 value
)
{
    if (offset >= NV_PCIE_CFG_MAX_OFFSET)
        return NV_ERR_NOT_SUPPORTED;

    pci_write_config_dword( (struct pci_dev *) handle, offset, value);
    return NV_OK;
}

NvBool NV_API_CALL os_pci_remove_supported(void)
{
#if defined NV_PCI_STOP_AND_REMOVE_BUS_DEVICE
    return NV_TRUE;
#else
    return NV_FALSE;
#endif
}

void NV_API_CALL os_pci_remove(
    void *handle
)
{
#if defined(NV_PCI_STOP_AND_REMOVE_BUS_DEVICE)
    NV_PCI_STOP_AND_REMOVE_BUS_DEVICE(handle);
#elif defined(DEBUG)
    nv_printf(NV_DBG_ERRORS,
            "NVRM: %s() is called even though NV_PCI_STOP_AND_REMOVE_BUS_DEVICE is not defined\n",
            __FUNCTION__);
    os_dbg_breakpoint();
#endif
}
