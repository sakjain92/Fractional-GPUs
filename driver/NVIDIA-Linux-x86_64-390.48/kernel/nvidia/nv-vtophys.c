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

NvU64 NV_API_CALL nv_get_kern_phys_address(NvU64 address)
{
    /* make sure this address is a kernel virtual address */
#if defined(DEBUG) && !defined(CONFIG_X86_4G)
    if (access_ok(VERIFY_READ, address, sizeof(address)))
    {
        nv_printf(NV_DBG_WARNINGS,
            "NVRM: user address passed to %s: 0x%llx!\n",
            __FUNCTION__, address);
        return 0;
    }
#endif

    /* direct-mapped kernel address */
    if (virt_addr_valid(address))
        return __pa(address);

    nv_printf(NV_DBG_ERRORS,
        "NVRM: can't translate address in %s()!\n", __FUNCTION__);
    return 0;
}

