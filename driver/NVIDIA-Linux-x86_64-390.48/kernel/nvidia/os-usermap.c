/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2011 by NVIDIA Corporation.  All rights reserved.  All
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

void* NV_API_CALL os_map_user_space(
    NvU64   start,
    NvU64   size_bytes,
    NvU32   mode,
    NvU32   protect,
    void  **priv_data
)
{
    return (void *)(NvUPtr)start;
}

void NV_API_CALL os_unmap_user_space(
    void  *address,
    NvU64  size,
    void  *priv_data
)
{
}

NV_STATUS NV_API_CALL os_match_mmap_offset(
    void  *pAllocPrivate,
    NvU64  offset,
    NvU64 *pPageIndex
)
{
    nv_alloc_t *at = pAllocPrivate;
    NvU64 i;

    for (i = 0; i < at->num_pages; i++)
    {
        if (NV_ALLOC_MAPPING_CONTIG(at->flags))
        {
            if (offset == (at->page_table[0]->phys_addr + (i * PAGE_SIZE)))
            {
                *pPageIndex = i;
                return NV_OK;
            }
        }
        else
        {
            if (offset == at->page_table[i]->phys_addr)
            {
                *pPageIndex = i;
                return NV_OK;
            }
        }
    }

    return NV_ERR_OBJECT_NOT_FOUND;
}
