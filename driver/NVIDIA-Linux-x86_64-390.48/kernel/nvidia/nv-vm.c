/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2013 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include "nv-misc.h"
#include "os-interface.h"
#include "nv.h"
#include "nv-linux.h"

static inline void nv_set_contig_memory_uc(nvidia_pte_t *page_ptr, NvU32 num_pages)
{
    if (nv_update_memory_types)
    {
#if defined(NV_SET_MEMORY_UC_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        unsigned long addr = (unsigned long)page_address(page);
        set_memory_uc(addr, num_pages);
#elif defined(NV_SET_PAGES_UC_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        set_pages_uc(page, num_pages);
#elif defined(NV_CHANGE_PAGE_ATTR_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        pgprot_t prot = PAGE_KERNEL_NOCACHE;
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
        pgprot_val(prot) &= __nv_supported_pte_mask;
#endif
        change_page_attr(page, num_pages, prot);
#endif
    }
}


static inline void nv_set_contig_memory_wb(nvidia_pte_t *page_ptr, NvU32 num_pages)
{
    if (nv_update_memory_types)
    {
#if defined(NV_SET_MEMORY_UC_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        unsigned long addr = (unsigned long)page_address(page);
        set_memory_wb(addr, num_pages);
#elif defined(NV_SET_PAGES_UC_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        set_pages_wb(page, num_pages);
#elif defined(NV_CHANGE_PAGE_ATTR_PRESENT)
        struct page *page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
        pgprot_t prot = PAGE_KERNEL;
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
        pgprot_val(prot) &= __nv_supported_pte_mask;
#endif
        change_page_attr(page, num_pages, prot);
#endif
    }
}

static inline int nv_set_memory_array_type_present(NvU32 type)
{
    switch (type)
    {
#if defined(NV_SET_MEMORY_ARRAY_UC_PRESENT)
        case NV_MEMORY_UNCACHED:
            return 1;
        case NV_MEMORY_WRITEBACK:
            return 1;
#endif
        default:
            return 0;
    }
}

static inline void nv_set_memory_array_type(
    unsigned long *pages,
    NvU32 num_pages,
    NvU32 type
)
{
    switch (type)
    {
#if defined(NV_SET_MEMORY_ARRAY_UC_PRESENT)
        case NV_MEMORY_UNCACHED:
            set_memory_array_uc(pages, num_pages);
            break;
        case NV_MEMORY_WRITEBACK:
            set_memory_array_wb(pages, num_pages);
            break;
#endif
        default:
            nv_printf(NV_DBG_ERRORS,
                "NVRM: %s(): type %d unimplemented\n",
                __FUNCTION__, type);
            break;
    }
}

static inline void nv_set_contig_memory_type(
    nvidia_pte_t *page_ptr,
    NvU32 num_pages,
    NvU32 type
)
{
    if (nv_update_memory_types)
    {
        switch (type)
        {
            case NV_MEMORY_UNCACHED:
                nv_set_contig_memory_uc(page_ptr, num_pages);
                break;
            case NV_MEMORY_WRITEBACK:
                nv_set_contig_memory_wb(page_ptr, num_pages);
                break;
            default:
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: %s(): type %d unimplemented\n",
                    __FUNCTION__, type);
        }
    }
}

static inline void nv_set_memory_type(nv_alloc_t *at, NvU32 type)
{
    NvU32 i;
    NV_STATUS status;
    unsigned long *pages;
    nvidia_pte_t *page_ptr;
    struct page *page;

    if (nv_update_memory_types)
    {
        pages = NULL;

        if (nv_set_memory_array_type_present(type))
        {
            status = os_alloc_mem((void **)&pages,
                        at->num_pages * sizeof(unsigned long));
            if (status != NV_OK)
                pages = NULL;
        }

        if (pages)
        {
            for (i = 0; i < at->num_pages; i++)
            {
                page_ptr = at->page_table[i];
                page = NV_GET_PAGE_STRUCT(page_ptr->phys_addr);
                pages[i] = (unsigned long)page_address(page);
            }
            nv_set_memory_array_type(pages, at->num_pages, type);
            os_free_mem(pages);
        }
        else
        {
            for (i = 0; i < at->num_pages; i++)
                nv_set_contig_memory_type(at->page_table[i], 1, type);
        }
    }
}

/*
 * Cache flushes and TLB invalidation
 *
 * Allocating new pages, we may change their kernel mappings' memory types
 * from cached to UC to avoid cache aliasing. One problem with this is
 * that cache lines may still contain data from these pages and there may
 * be then stale TLB entries.
 *
 * Linux kernels prior to 2.6.25 fail to reliably flush caches on all or
 * some CPUS. The NVIDIA Linux graphics driver implements nv_flush_cache()
 * to perform heavy-weight flush/invalidation operations for older kernels
 * to avoid problems due to stale cache lines and/or TLB entries.
 *
 * Linux kernels which include tlbflush.h reliably flush caches when the
 * NVIDIA Linux graphics driver calls change_page_attr(...) after allocating
 * memory.  
 */

#if defined(NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL) || \
  defined(NV_CONFIG_PREEMPT_RT) || defined(NV_ASM_TLBFLUSH_H_PRESENT)
#define NV_REQUIRE_HEAVY_WEIGHT_FLUSH 0
#else
#define NV_REQUIRE_HEAVY_WEIGHT_FLUSH 1
#endif

#if NV_REQUIRE_HEAVY_WEIGHT_FLUSH
static void nv_flush_cache(void *p)
{
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    unsigned long reg0, reg1, reg2;
#endif

    CACHE_FLUSH();

    // flush global TLBs
#if defined(NVCPU_X86)
    asm volatile("movl %%cr4, %0;  \n"
                 "movl %0, %2;     \n"
                 "andl $~0x80, %0; \n"
                 "movl %0, %%cr4;  \n"
                 "movl %%cr3, %1;  \n"
                 "movl %1, %%cr3;  \n"
                 "movl %2, %%cr4;  \n"
                 : "=&r" (reg0), "=&r" (reg1), "=&r" (reg2)
                 : : "memory");
#elif defined(NVCPU_X86_64)
    asm volatile("movq %%cr4, %0;  \n"
                 "movq %0, %2;     \n"
                 "andq $~0x80, %0; \n"
                 "movq %0, %%cr4;  \n"
                 "movq %%cr3, %1;  \n"
                 "movq %1, %%cr3;  \n"
                 "movq %2, %%cr4;  \n"
                 : "=&r" (reg0), "=&r" (reg1), "=&r" (reg2)
                 : : "memory");
#endif
}
#endif

static inline void nv_flush_caches(void)
{
#if NV_REQUIRE_HEAVY_WEIGHT_FLUSH
    NV_ON_EACH_CPU(nv_flush_cache, NULL);
#endif
}

static NvU64 nv_get_max_sysmem_address(void)
{
    NvU64 global_max_pfn = 0ULL;
    int node_id;

    NV_FOR_EACH_ONLINE_NODE(node_id)
    {
        global_max_pfn = max(global_max_pfn, nv_node_end_pfn(node_id));
    }

    return ((global_max_pfn + 1) << PAGE_SHIFT) - 1;
}

static unsigned int nv_compute_gfp_mask(
    nv_alloc_t *at
)
{
    unsigned int gfp_mask = NV_GFP_KERNEL;
    struct pci_dev *dev = at->dev;
#if !(defined(CONFIG_X86_UV) && defined(NV_CONFIG_X86_UV))
    NvU64 max_sysmem_address = nv_get_max_sysmem_address();
    if (dev->dma_mask < max_sysmem_address)
    {
        gfp_mask = NV_GFP_DMA32;
    }
#endif
#if defined(__GFP_NORETRY)
    gfp_mask |= __GFP_NORETRY;
#endif
#if defined(__GFP_ZERO)
    if (at->flags & NV_ALLOC_TYPE_ZEROED)
        gfp_mask |= __GFP_ZERO;
#endif
#if defined(__GFP_THISNODE)
    if (at->flags & NV_ALLOC_TYPE_NODE0)
        gfp_mask |= __GFP_THISNODE;
#endif

    return gfp_mask;
}

/*
 * This function is needed for allocating contiguous physical memory in xen 
 * dom0. Because of the use of xen sw iotlb in xen dom0, memory allocated by 
 * NV_GET_FREE_PAGES may not be machine contiguous when size is more than 
 * 1 page. nv_alloc_coherent_pages() will give us machine contiguous memory.
 * Even though we get dma_address directly in this function, we will 
 * still call pci_map_page() later to get dma address. This is fine as it 
 * will return the same machine address.
 */
static NV_STATUS nv_alloc_coherent_pages(
    nv_state_t *nv,
    nv_alloc_t *at
)
{
    nvidia_pte_t *page_ptr;
    NvU32 i;
    unsigned int gfp_mask;
    unsigned long virt_addr = 0;
    dma_addr_t bus_addr;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    struct pci_dev *dev = nvl->dev;

    gfp_mask = nv_compute_gfp_mask(at);

    virt_addr = (unsigned long)dma_alloc_coherent(&dev->dev,
                                                  at->num_pages * PAGE_SIZE,
                                                  &bus_addr,
                                                  gfp_mask | __GFP_COMP);
    if (!virt_addr)
    {
        nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %s: failed to allocate memory\n", __FUNCTION__);
        return NV_ERR_NO_MEMORY;
    }

    for (i = 0; i < at->num_pages; i++)
    {
        page_ptr = at->page_table[i];

        page_ptr->virt_addr = virt_addr + i * PAGE_SIZE;
        page_ptr->phys_addr = virt_to_phys((void *)page_ptr->virt_addr);
        page_ptr->dma_addr  = bus_addr + i * PAGE_SIZE;
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
    {
        nv_set_contig_memory_type(at->page_table[0],
                                  at->num_pages,
                                  NV_MEMORY_UNCACHED);
        nv_flush_caches();
    }

    return NV_OK;
}

static void nv_free_coherent_pages(
    nv_alloc_t *at
)
{
    nvidia_pte_t *page_ptr;
    struct pci_dev *dev = at->dev;

    page_ptr = at->page_table[0];

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
    {
        nv_set_contig_memory_type(at->page_table[0],
                                  at->num_pages,
                                  NV_MEMORY_WRITEBACK);
    }

    dma_free_coherent(&dev->dev, at->num_pages * PAGE_SIZE,
                      (void *)page_ptr->virt_addr, page_ptr->dma_addr);

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_flush_caches();
}

NV_STATUS nv_alloc_contig_pages(
    nv_state_t *nv,
    nv_alloc_t *at
)
{
    NV_STATUS status;
    nvidia_pte_t *page_ptr;
    NvU32 i, j;
    unsigned int gfp_mask;
    unsigned long virt_addr = 0;
    NvU64 phys_addr;
    struct pci_dev *dev = at->dev;

    nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %s: %u pages\n", __FUNCTION__, at->num_pages);

    if (os_is_xen_dom0())
        return nv_alloc_coherent_pages(nv, at);

    at->order = get_order(at->num_pages * PAGE_SIZE);
    gfp_mask = nv_compute_gfp_mask(at);

    if (at->flags & NV_ALLOC_TYPE_NODE0)
    {
        NV_ALLOC_PAGES_NODE(virt_addr, 0, at->order, gfp_mask);
    }
    else
    {
        NV_GET_FREE_PAGES(virt_addr, at->order, gfp_mask);
    }
    if (virt_addr == 0)
    {
        nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %s: failed to allocate memory\n", __FUNCTION__);
        return NV_ERR_NO_MEMORY;
    }
#if !defined(__GFP_ZERO)
    if (at->flags & NV_ALLOC_TYPE_ZEROED)
        memset((void *)virt_addr, 0, (at->num_pages * PAGE_SIZE));
#endif

    for (i = 0; i < at->num_pages; i++, virt_addr += PAGE_SIZE)
    {
        phys_addr = nv_get_kern_phys_address(virt_addr);
        if (phys_addr == 0)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: VM: %s: failed to look up physical address\n",
                __FUNCTION__);
            status = NV_ERR_OPERATING_SYSTEM;
            goto failed;
        }

        page_ptr = at->page_table[i];
        page_ptr->phys_addr = phys_addr;
        page_ptr->page_count = NV_GET_PAGE_COUNT(page_ptr);
        page_ptr->virt_addr = virt_addr;
        page_ptr->dma_addr = nv_phys_to_dma(dev, page_ptr->phys_addr);

        NV_LOCK_PAGE(page_ptr);
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
    {
        nv_set_contig_memory_type(at->page_table[0],
                                  at->num_pages,
                                  NV_MEMORY_UNCACHED);
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_flush_caches();

    return NV_OK;

failed:
    if (i > 0)
    {
        for (j = 0; j < i; j++)
            NV_UNLOCK_PAGE(at->page_table[j]);
    }

    page_ptr = at->page_table[0];
    NV_FREE_PAGES(page_ptr->virt_addr, at->order);

    return status;
}

void nv_free_contig_pages(
    nv_alloc_t *at
)
{
    nvidia_pte_t *page_ptr;
    unsigned int i;

    nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %s: %u pages\n", __FUNCTION__, at->num_pages);

    if (os_is_xen_dom0())
        return nv_free_coherent_pages(at);

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
    {
        nv_set_contig_memory_type(at->page_table[0],
                                  at->num_pages,
                                  NV_MEMORY_WRITEBACK);
    }

    for (i = 0; i < at->num_pages; i++)
    {
        page_ptr = at->page_table[i];

        if (NV_GET_PAGE_COUNT(page_ptr) != page_ptr->page_count)
        {
            static int count = 0;
            if (count++ < NV_MAX_RECURRING_WARNING_MESSAGES)
            {
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: VM: %s: page count != initial page count (%u,%u)\n",
                    __FUNCTION__, NV_GET_PAGE_COUNT(page_ptr),
                    page_ptr->page_count);
            }
        }
        NV_UNLOCK_PAGE(page_ptr);
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_flush_caches();

    page_ptr = at->page_table[0];

    NV_FREE_PAGES(page_ptr->virt_addr, at->order);
}

NV_STATUS nv_alloc_system_pages(
    nv_state_t *nv,
    nv_alloc_t *at
)
{
    NV_STATUS status;
    nvidia_pte_t *page_ptr;
    NvU32 i, j;
    unsigned int gfp_mask;
    unsigned long virt_addr = 0;
    NvU64 phys_addr;
    struct pci_dev *dev = at->dev;

    nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %u: %u pages\n", __FUNCTION__, at->num_pages);

    gfp_mask = nv_compute_gfp_mask(at);

    for (i = 0; i < at->num_pages; i++)
    {
        if (at->flags & NV_ALLOC_TYPE_NODE0)
        {
            NV_ALLOC_PAGES_NODE(virt_addr, 0, 0, gfp_mask);
        }
        else
        {
            NV_GET_FREE_PAGES(virt_addr, 0, gfp_mask);
        }
        if (virt_addr == 0)
        {
            nv_printf(NV_DBG_MEMINFO,
                "NVRM: VM: %s: failed to allocate memory\n", __FUNCTION__);
            status = NV_ERR_NO_MEMORY;
            goto failed;
        }
#if !defined(__GFP_ZERO)
        if (at->flags & NV_ALLOC_TYPE_ZEROED)
            memset((void *)virt_addr, 0, PAGE_SIZE);
#endif

        phys_addr = nv_get_kern_phys_address(virt_addr);
        if (phys_addr == 0)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: VM: %s: failed to look up physical address\n",
                __FUNCTION__);
            NV_FREE_PAGES(virt_addr, 0);
            status = NV_ERR_OPERATING_SYSTEM;
            goto failed;
        }

#if defined(_PAGE_NX)
        if (((_PAGE_NX & pgprot_val(PAGE_KERNEL)) != 0) &&
                (phys_addr < 0x400000))
        {
            nv_printf(NV_DBG_SETUP,
                "NVRM: VM: %s: discarding page @ 0x%llx\n",
                __FUNCTION__, phys_addr);
            --i;
            continue;
        }
#endif

        page_ptr = at->page_table[i];
        page_ptr->phys_addr = phys_addr;
        page_ptr->page_count = NV_GET_PAGE_COUNT(page_ptr);
        page_ptr->virt_addr = virt_addr;
        page_ptr->dma_addr = nv_phys_to_dma(dev, page_ptr->phys_addr);

        NV_LOCK_PAGE(page_ptr);
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_set_memory_type(at, NV_MEMORY_UNCACHED);

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_flush_caches();

    return NV_OK;

failed:
    if (i > 0)
    {
        for (j = 0; j < i; j++)
        {
            page_ptr = at->page_table[j];
            NV_UNLOCK_PAGE(page_ptr);
            NV_FREE_PAGES(page_ptr->virt_addr, 0);
        }
    }

    return status;
}

void nv_free_system_pages(
    nv_alloc_t *at
)
{
    nvidia_pte_t *page_ptr;
    unsigned int i;

    nv_printf(NV_DBG_MEMINFO,
            "NVRM: VM: %s: %u pages\n", __FUNCTION__, at->num_pages);

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_set_memory_type(at, NV_MEMORY_WRITEBACK);

    for (i = 0; i < at->num_pages; i++)
    {
        page_ptr = at->page_table[i];

        if (NV_GET_PAGE_COUNT(page_ptr) != page_ptr->page_count)
        {
            static int count = 0;
            if (count++ < NV_MAX_RECURRING_WARNING_MESSAGES)
            {
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: VM: %s: page count != initial page count (%u,%u)\n",
                    __FUNCTION__, NV_GET_PAGE_COUNT(page_ptr),
                    page_ptr->page_count);
            }
        }
        NV_UNLOCK_PAGE(page_ptr);
        NV_FREE_PAGES(page_ptr->virt_addr, 0);
    }

    if (!NV_ALLOC_MAPPING_CACHED(at->flags))
        nv_flush_caches();
}

NvUPtr nv_vm_map_pages(
    struct page **pages,
    NvU32 count,
    NvBool cached
)
{
    NvUPtr virt_addr = 0;

    if (!NV_MAY_SLEEP())
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: %s: can't map %d pages, invalid context!\n",
                  __FUNCTION__, count);
        os_dbg_breakpoint();
        return virt_addr;
    }

    virt_addr = nv_vmap(pages, count, cached);
    return virt_addr;
}

void nv_vm_unmap_pages(
    NvUPtr virt_addr,
    NvU32 count
)
{
    if (!NV_MAY_SLEEP())
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: %s: can't unmap %d pages at 0x%0llx, "
                  "invalid context!\n", __FUNCTION__, count, virt_addr);
        os_dbg_breakpoint();
        return;
    }

    nv_vunmap(virt_addr, count);
}
