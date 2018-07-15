/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2014 by NVIDIA Corporation.  All rights reserved.  All
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
#include "nv-frontend.h"

/* minor number table */
extern nvidia_module_t *nv_minor_num_table[];

void nv_user_map_init(void)
{
}

int nv_user_map_register(
    NvU64 address,
    NvU64 size
)
{
    return 0;
}

void nv_user_map_unregister(
    NvU64 address,
    NvU64 size
)
{
}

static nv_file_private_t* nv_get_file_private(NvU32 fd, nv_state_t *nv)
{
    struct file *pFile = NULL;
    dev_t rdev = 0;
    nv_file_private_t *nvfp = NULL;
    int i;
    int rc = -1;

    pFile = fget(fd);
    if (pFile == NULL)
        return NULL;

    if (!NV_FILE_INODE(pFile))
        goto done;

    rdev = (NV_FILE_INODE(pFile))->i_rdev;

    /* Note it is not safe to interpret the file private data
     * as an nv_file_private_t until it passes this check.
     */
    if (MAJOR(rdev) != NV_MAJOR_DEVICE_NUMBER)
        goto done;

    if (NV_IS_CTL_DEVICE(nv))
    {
        /* Validate minor number for Nvidia control device */
        if (MINOR(rdev) != NV_CONTROL_DEVICE_MINOR)
            goto done;
    }
    else
    {
        /* Validate minor number for Nvidia device */
        for (i = 0; i <= NV_FRONTEND_CONTROL_DEVICE_MINOR_MIN; i++) {
            if ((nv_minor_num_table[i] != NULL) && (MINOR(rdev) == i))
            {
                rc = 0;
                break;
            }
        }

        if (rc != 0)
            goto done;
    }

    nvfp = NV_GET_FILE_PRIVATE(pFile);
    if(nvfp == NULL)
        goto done;

done:
    /*
     * fget() incremented the struct file's reference count, which
     * needs to be balanced with a call to fput(). It is safe to
     * decrement the reference count before returning filp->private_data
     * because we are holding the GPUs lock which prevents freeing the file out.
     */
    fput(pFile);
    return nvfp;
}

NV_STATUS NV_API_CALL nv_add_mapping_context_to_file(
    nv_state_t *nv,
    nv_usermap_access_params_t *nvuap,
    NvU32       prot,
    void       *pAllocPriv,
    NvU64       pageIndex,
    NvU32       fd
)
{
    NV_STATUS status = NV_OK;
    nv_alloc_mapping_context_t *nvamc = NULL;
    nv_file_private_t *nvfp = NULL;

    nvfp = nv_get_file_private(fd, nv);
    if (nvfp == NULL)
        return NV_ERR_INVALID_ARGUMENT;

    nvamc = &nvfp->mmap_context;

    if (nvamc->bValid)
        return NV_ERR_STATE_IN_USE;

    if (NV_IS_CTL_DEVICE(nv))
    {
        nvamc->alloc = pAllocPriv;
        nvamc->page_index = pageIndex;
    }
    else
    {
        nvamc->mmap_start = nvuap->mmap_start;
        nvamc->mmap_size = nvuap->mmap_size;
        nvamc->access_start = nvuap->access_start;
        nvamc->access_size = nvuap->access_size;
        nvamc->remap_prot_extra = nvuap->remap_prot_extra;
    }

    nvamc->prot = prot;
    nvamc->bValid = NV_TRUE;
    return status;
}

NV_STATUS NV_API_CALL nv_alloc_user_mapping(
    nv_state_t *nv,
    void       *pAllocPrivate,
    NvU64       pageIndex,
    NvU32       pageOffset,
    NvU64       size,
    NvU32       protect,
    NvU64      *pUserAddress,
    void      **ppPrivate
)
{
    nv_alloc_t *at = pAllocPrivate;

    if (NV_ALLOC_MAPPING_CONTIG(at->flags))
        *pUserAddress = (at->page_table[0]->phys_addr + (pageIndex * PAGE_SIZE) + pageOffset);
    else
        *pUserAddress = (at->page_table[pageIndex]->phys_addr + pageOffset);

    return NV_OK;
}

NV_STATUS NV_API_CALL nv_free_user_mapping(
    nv_state_t *nv,
    void       *pAllocPrivate,
    NvU64       userAddress,
    void       *pPrivate
)
{
    return NV_OK;
}

/*
 * This function adjust the {mmap,access}_{start,size} to reflect platform-specific
 * mechanisms for isolating mappings at a finer granularity than the OS_PAGE_SIZE
 */
NV_STATUS NV_API_CALL nv_get_usermap_access_params(
    nv_state_t *nv,
    nv_usermap_access_params_t *nvuap
)
{
    NV_STATUS rmStatus;
    NvBool page_isolation_required;

    nvuap->remap_prot_extra = 0;

    rmStatus = rm_gpu_need_4k_page_isolation(nv, &page_isolation_required);
    if (rmStatus != NV_OK)
        return -EINVAL;

    /*
     * Do verification and cache encoding based on the original
     * (ostensibly smaller) mmap request, since accesses should be
     * restricted to that range.
     */
    if (page_isolation_required)
    {
#if defined(NV_4K_PAGE_ISOLATION_PRESENT)
        NvU64 addr = nvuap->addr;
        NvU64 size = nvuap->size;

        if (NV_4K_PAGE_ISOLATION_REQUIRED(addr, size))
        {
            nvuap->remap_prot_extra = NV_PROT_4K_PAGE_ISOLATION;
            nvuap->access_start = (NvU64)NV_4K_PAGE_ISOLATION_ACCESS_START(addr);
            nvuap->access_size = NV_4K_PAGE_ISOLATION_ACCESS_LEN(addr, size);
            nvuap->mmap_start = (NvU64)NV_4K_PAGE_ISOLATION_MMAP_ADDR(addr);
            nvuap->mmap_size = NV_4K_PAGE_ISOLATION_MMAP_LEN(size);
        }
#endif
    }

    return rmStatus;
}
