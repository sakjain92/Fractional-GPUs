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

extern nv_cpu_type_t nv_cpu_type;

/*
 * The 'struct vm_operations' open() callback is called by the Linux
 * kernel when the parent VMA is split or copied, close() when the
 * current VMA is about to be deleted.
 *
 * We implement these callbacks to keep track of the number of user
 * mappings of system memory allocations. This was motivated by a
 * subtle interaction problem between the driver and the kernel with
 * respect to the bookkeeping of pages marked reserved and later
 * mapped with mmap().
 *
 * Traditionally, the Linux kernel ignored reserved pages, such that
 * when they were mapped via mmap(), the integrity of their usage
 * counts depended on the reserved bit being set for as long as user
 * mappings existed.
 *
 * Since we mark system memory pages allocated for DMA reserved and
 * typically map them with mmap(), we need to ensure they remain
 * reserved until the last mapping has been torn down. This worked
 * correctly in most cases, but in a few, the RM API called into the
 * RM to free memory before calling munmap() to unmap it.
 *
 * In the past, we allowed nv_free_pages() to remove the 'at' from
 * the parent device's allocation list in this case, but didn't
 * release the underlying pages until the last user mapping had been
 * destroyed:
 *
 * In nvidia_vma_release(), we freed any resources associated with
 * the allocation (IOMMU mappings, etc.) and cleared the
 * underlying pages' reserved bits, but didn't free them. The kernel
 * was expected to do this.
 *
 * This worked in practise, but made dangerous assumptions about the
 * kernel's behavior and could fail in some cases. We now handle
 * this case differently (see below).
 */
static void
nvidia_vma_open(struct vm_area_struct *vma)
{
    nv_alloc_t *at = NV_VMA_PRIVATE(vma);

    NV_PRINT_VMA(NV_DBG_MEMINFO, vma);

    if (at != NULL)
    {
        NV_ATOMIC_INC(at->usage_count);

        NV_PRINT_AT(NV_DBG_MEMINFO, at);
    }
}

/*
 * (see above for additional information)
 *
 * If the 'at' usage count drops to zero with the updated logic, the
 * the allocation is recorded in the free list of the private
 * data associated with the file pointer; nvidia_close() uses this
 * list to perform deferred free operations when the parent file
 * descriptor is closed. This will typically happen when the process
 * exits.
 *
 * Since this is technically a workaround to handle possible fallout
 * from misbehaving clients, we additionally print a warning.
 */
static void
nvidia_vma_release(struct vm_area_struct *vma)
{
    nv_alloc_t *at = NV_VMA_PRIVATE(vma);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(NV_VMA_FILE(vma));
    static int count = 0;

    NV_PRINT_VMA(NV_DBG_MEMINFO, vma);

    if (at != NULL && nv_alloc_release(nvfp, at))
    {
        if ((at->pid == os_get_current_process()) &&
            (count++ < NV_MAX_RECURRING_WARNING_MESSAGES))
        {
            nv_printf(NV_DBG_MEMINFO,
                "NVRM: VM: %s: late unmap, comm: %s, 0x%p\n",
                __FUNCTION__, current->comm, at);
        }
    }
}

#if defined(NV_VM_OPERATIONS_STRUCT_HAS_ACCESS)
static int
nvidia_vma_access(
    struct vm_area_struct *vma,
    unsigned long addr,
    void *buffer,
    int length,
    int write
)
{
    nv_alloc_t *at = NULL;
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(NV_VMA_FILE(vma));
    nv_state_t *nv = NV_STATE_PTR(nvfp->nvptr);
    NvU32 pageIndex, pageOffset;
    void *kernel_mapping;
    const nv_alloc_mapping_context_t *mmap_context = &nvfp->mmap_context;
    NvU64 offset;

    pageIndex = ((addr - vma->vm_start) >> PAGE_SHIFT);
    pageOffset = (addr & ~PAGE_MASK);

    if (!mmap_context->bValid)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: VM: invalid mmap context\n");
        return -EINVAL;
    }

    offset = mmap_context->mmap_start;

    if (nv->flags & NV_FLAG_CONTROL)
    {
        at = NV_VMA_PRIVATE(vma);

        /*
         * at can be NULL for peer IO mem.
         */
        if (!at)
            return -EINVAL;

        if (pageIndex >= at->num_pages)
            return -EINVAL;

        kernel_mapping = (void *)(at->page_table[pageIndex]->virt_addr + pageOffset);
    }
    else if (IS_FB_OFFSET(nv, offset, length))
    {
        addr = (offset & PAGE_MASK);
        kernel_mapping = os_map_kernel_space(addr, PAGE_SIZE, NV_MEMORY_UNCACHED, NV_MEMORY_TYPE_SYSTEM);
        if (kernel_mapping == NULL)
            return -ENOMEM;

        kernel_mapping = ((char *)kernel_mapping + pageOffset);
    }
    else
        return -EINVAL;

    length = NV_MIN(length, (int)(PAGE_SIZE - pageOffset));

    if (write)
        memcpy(kernel_mapping, buffer, length);
    else
        memcpy(buffer, kernel_mapping, length);

    if (at == NULL)
    {
        kernel_mapping = ((char *)kernel_mapping - pageOffset);
        os_unmap_kernel_space(kernel_mapping, PAGE_SIZE);
    }

    return length;
}
#endif

struct vm_operations_struct nv_vm_ops = {
    .open   = nvidia_vma_open,
    .close  = nvidia_vma_release,
#if defined(NV_VM_OPERATIONS_STRUCT_HAS_ACCESS)
    .access = nvidia_vma_access,
#endif
};

int nv_encode_caching(
    pgprot_t *prot,
    NvU32     cache_type,
    NvU32     memory_type
)
{
    pgprot_t tmp;

    if (prot == NULL)
    {
        tmp = __pgprot(0);
        prot = &tmp;
    }

    switch (cache_type)
    {
        case NV_MEMORY_UNCACHED_WEAK:
#if defined(NV_PGPROT_UNCACHED_WEAK)
            *prot = NV_PGPROT_UNCACHED_WEAK(*prot);
            break;
#endif
        case NV_MEMORY_UNCACHED:
            /*!
             * On Tegra 3 (A9), we cannot have the device type bits set on
             * any BAR mappings.
             */
            *prot = ((memory_type == NV_MEMORY_TYPE_SYSTEM) ||
                (nv_cpu_type == NV_CPU_TYPE_ARM_A9)) ?
                    NV_PGPROT_UNCACHED(*prot) :
                    NV_PGPROT_UNCACHED_DEVICE(*prot);
            break;
#if defined(NV_PGPROT_WRITE_COMBINED) && \
    defined(NV_PGPROT_WRITE_COMBINED_DEVICE)
        case NV_MEMORY_WRITECOMBINED:
            if (NV_ALLOW_WRITE_COMBINING(memory_type))
            {
                /*!
                 * On Tegra 3 (A9), we cannot have the device type bits set on
                 * any BAR mappings.
                 */
                *prot = ((memory_type == NV_MEMORY_TYPE_FRAMEBUFFER) &&
                    (nv_cpu_type != NV_CPU_TYPE_ARM_A9)) ?
                        NV_PGPROT_WRITE_COMBINED_DEVICE(*prot) :
                        NV_PGPROT_WRITE_COMBINED(*prot);
                break;
            }

            /*
             * If WC support is unavailable, we need to return an error
             * code to the caller, but need not print a warning.
             *
             * For frame buffer memory, callers are expected to use the
             * UC- memory type if we report WC as unsupported, which
             * translates to the effective memory type WC if a WC MTRR
             * exists or else UC.
             */
            return 1;
#endif
        case NV_MEMORY_CACHED:
            if (NV_ALLOW_CACHING(memory_type))
                break;
        default:
            nv_printf(NV_DBG_ERRORS,
                "NVRM: VM: cache type %d not supported for memory type %d!\n",
                cache_type, memory_type);
            return 1;
    }
    return 0;
}

int static nvidia_mmap_peer_io(
    struct vm_area_struct *vma,
    nv_alloc_t *at,
    NvU64 page_index,
    NvU64 pages
)
{
    int ret;
    NvU64 start;
    NvU64 size;

    BUG_ON(!NV_ALLOC_MAPPING_CONTIG(at->flags));

    start = at->page_table[page_index]->phys_addr;
    size = pages * PAGE_SIZE;

    ret = nv_io_remap_page_range(vma, start, size, 0);

    return ret;
}

int static nvidia_mmap_sysmem(
    struct vm_area_struct *vma,
    nv_alloc_t *at,
    NvU64 page_index,
    NvU64 pages
)
{
    NvU64 j;
    int ret = 0;
    unsigned long start = 0;

    NV_ATOMIC_INC(at->usage_count);

    start = vma->vm_start;
    for (j = page_index; j < (page_index + pages); j++)
    {
#if defined(NV_VM_INSERT_PAGE_PRESENT)
#if defined(NV_VGPU_KVM_BUILD)
        if (NV_ALLOC_MAPPING_GUEST(at->flags))
        {
            ret = nv_remap_page_range(vma, start, at->page_table[j]->phys_addr,
                                      PAGE_SIZE, vma->vm_page_prot);
        }
        else
#endif
        {
            ret = NV_VM_INSERT_PAGE(vma, start,
                                    NV_GET_PAGE_STRUCT(at->page_table[j]->phys_addr));
        }
        if (ret)
#else
        if (nv_remap_page_range(vma, start, at->page_table[j]->phys_addr,
                                PAGE_SIZE, vma->vm_page_prot) != 0)
#endif
        {
            NV_ATOMIC_DEC(at->usage_count);
            return -EAGAIN;
        }
        start += PAGE_SIZE;
    }

    return ret;
}

int nvidia_mmap_helper(
    nv_state_t *nv,
    nv_file_private_t *nvfp,
    nvidia_stack_t *sp,
    struct vm_area_struct *vma,
    void *vm_priv
)
{
    NvU32 prot = 0;
    const nv_alloc_mapping_context_t *mmap_context = &nvfp->mmap_context;

    if (nvfp == NULL)
        return NV_ERR_INVALID_ARGUMENT;

    /*
     * If mmap context is not valid on this file descriptor, this mapping wasn't
     * previously validated with the RM so it must be rejected.
     */
    if (!mmap_context->bValid)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: VM: invalid mmap\n");
        return -EINVAL;
    }

    NV_PRINT_VMA(NV_DBG_MEMINFO, vma);

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, NV_MAY_SLEEP());

    NV_VMA_PRIVATE(vma) = vm_priv;

    prot = mmap_context->prot;

    /*
     * Nvidia device node(nvidia#) maps device's BAR memory,
     * Nvidia control node(nvidiactrl) maps system memory.
     */
    if (!NV_IS_CTL_DEVICE(nv))
    {
        NvU32 remap_prot_extra = mmap_context->remap_prot_extra;
        NvU64 mmap_start = mmap_context->mmap_start;
        NvU64 mmap_length = mmap_context->mmap_size;
        NvU64 access_start = mmap_context->access_start;
        NvU64 access_len = mmap_context->access_size;

        if (IS_REG_OFFSET(nv, access_start, access_len))
        {
            if (nv_encode_caching(&vma->vm_page_prot, NV_MEMORY_UNCACHED,
                        NV_MEMORY_TYPE_REGISTERS))
            {
                return -ENXIO;
            }
        }
        else if (IS_FB_OFFSET(nv, access_start, access_len))
        {
            if (IS_UD_OFFSET(nv, access_start, access_len))
            {
                if (nv_encode_caching(&vma->vm_page_prot, NV_MEMORY_UNCACHED,
                            NV_MEMORY_TYPE_FRAMEBUFFER))
                {
                    return -ENXIO;
                }
            }
            else
            {
                if (nv_encode_caching(&vma->vm_page_prot,
                        NV_MEMORY_WRITECOMBINED, NV_MEMORY_TYPE_FRAMEBUFFER))
                {
                    if (nv_encode_caching(&vma->vm_page_prot,
                            NV_MEMORY_UNCACHED_WEAK, NV_MEMORY_TYPE_FRAMEBUFFER))
                    {
                        return -ENXIO;
                    }
                }
            }
        }
#if defined(NVCPU_PPC64LE) && defined(NV_VM_INSERT_PAGE_PRESENT)
        // 
        // This path is similar to the sysmem mapping code. 
        // TODO: Refactor is needed as part of bug#2001704.
        // Use pfn_valid to determine whether the physical address has
        // backing struct page. This is used to isolate P8 from P9. 
        //
        if (!IS_REG_OFFSET(nv, access_start, access_len) && 
            (pfn_valid(PFN_DOWN(mmap_start))))
        { 
            NvU64 j; 
            NvU64 start, addr;
            unsigned int pages;

            pages = NV_VMA_SIZE(vma) >> PAGE_SHIFT;
            start = vma->vm_start;
            addr  = mmap_start;
            
            // Needed for the linux kernel for mapping compound pages
            vma->vm_flags |= VM_MIXEDMAP;

            for (j = 0; j < pages; j++)
            {
                if (vm_insert_page(vma, start, NV_GET_PAGE_STRUCT(addr)) != 0)
                {
                    return -EAGAIN;
                }
                start += PAGE_SIZE;

                // Assuming contiguity. Need to be changed for bug #1945056
                addr  += PAGE_SIZE;
            }
        }
        else
#endif
        {
            if (nv_io_remap_page_range(vma, mmap_start, mmap_length,
                    remap_prot_extra) != 0)
            {
                return -EAGAIN;
            }
        }

        vma->vm_flags |= VM_IO;
    }
    else
    {
        int ret;
        nv_alloc_t *at;
        NvU64 page_index;
        NvU64 pages;
        NvU64 mmap_size;

        at = (nv_alloc_t *)mmap_context->alloc;
        page_index = mmap_context->page_index;

        if (nv_encode_caching(&vma->vm_page_prot,
                              NV_ALLOC_MAPPING(at->flags),
                              NV_MEMORY_TYPE_SYSTEM))
        {
            return -ENXIO;
        }

        mmap_size = NV_VMA_SIZE(vma);
        pages = mmap_size >> PAGE_SHIFT;

        if ((page_index + pages) > at->num_pages)
        {
            return -ERANGE;
        }

        /*
         * Callers that pass in non-NULL VMA private data must never reach this
         * code. They should be mapping on a non-control node.
         */
        BUG_ON(NV_VMA_PRIVATE(vma));

        if (NV_ALLOC_MAPPING_PEER_IO(at->flags))
        {
            /*
             * There is no need to keep 'peer IO at' alive till vma_release like
             * 'sysmem at' because there are no security concerns where a client
             * could free RM allocated sysmem before unmapping it. Hence, vm_ops
             * are NOP, and at->usage_count is never being used.
             */
            NV_VMA_PRIVATE(vma) = NULL;

            ret = nvidia_mmap_peer_io(vma, at, page_index, pages);

            BUG_ON(NV_VMA_PRIVATE(vma));
        }
        else
        {
            NV_VMA_PRIVATE(vma) = at;

            ret = nvidia_mmap_sysmem(vma, at, page_index, pages);
        }

        if (ret)
        {
            return ret;
        }

        NV_PRINT_AT(NV_DBG_MEMINFO, at);

        vma->vm_flags |= (VM_IO | VM_LOCKED | VM_RESERVED);
        vma->vm_flags |= (VM_DONTEXPAND | VM_DONTDUMP);
    }

    if ((prot & NV_PROTECT_WRITEABLE) == 0)
    {
        vma->vm_page_prot = NV_PGPROT_READ_ONLY(vma->vm_page_prot);
        vma->vm_flags &= ~VM_WRITE;
        vma->vm_flags &= ~VM_MAYWRITE;
    }

    vma->vm_ops = &nv_vm_ops;

    return 0;
}

int nvidia_mmap(
    struct file *file,
    struct vm_area_struct *vma
)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_FILEP(file);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    nvidia_stack_t *sp = NULL;
    int status;

    //
    // Do not allow mmap operation if this is a fd into
    // which a rm-object has been exported.
    //
    if (nvfp->hExportedRmObject != 0)
    {
        return -EINVAL;
    }

    down(&nvfp->fops_sp_lock[NV_FOPS_STACK_INDEX_MMAP]);

    sp = nvfp->fops_sp[NV_FOPS_STACK_INDEX_MMAP];

    status = nvidia_mmap_helper(nv, nvfp, sp, vma, NULL);

    up(&nvfp->fops_sp_lock[NV_FOPS_STACK_INDEX_MMAP]);

    return status;
}
