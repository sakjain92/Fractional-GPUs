/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2015 by NVIDIA Corporation.  All rights reserved.  All
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

NV_STATUS   nv_create_dma_map_scatterlist (nv_dma_map_t *dma_map);
void        nv_destroy_dma_map_scatterlist(nv_dma_map_t *dma_map);
NV_STATUS   nv_map_dma_map_scatterlist    (nv_dma_map_t *dma_map);
void        nv_unmap_dma_map_scatterlist  (nv_dma_map_t *dma_map);
static void nv_dma_unmap_contig           (nv_dma_map_t *dma_map);
static void nv_dma_unmap_scatterlist      (nv_dma_map_t *dma_map);

static NV_STATUS nv_dma_map_contig(
    nv_state_t *nv,
    nv_dma_map_t *dma_map,
    NvU64 *va
)
{
    *va = pci_map_page(dma_map->dev, dma_map->pages[0], 0,
            dma_map->page_count * PAGE_SIZE, PCI_DMA_BIDIRECTIONAL);
    if (NV_PCI_DMA_MAPPING_ERROR(dma_map->dev, *va))
    {
        return NV_ERR_OPERATING_SYSTEM;
    }

    dma_map->mapping.contig.dma_addr = *va;

    if (!IS_DMA_ADDRESSABLE(nv, *va) ||
        !IS_DMA_ADDRESSABLE(nv, *va + (dma_map->page_count * PAGE_SIZE - 1)))
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: DMA address not in addressable range of device "
                "%04x:%02x:%02x (0x%llx-0x%llx, 0x%llx-0x%llx)\n",
                NV_PCI_DOMAIN_NUMBER(dma_map->dev),
                NV_PCI_BUS_NUMBER(dma_map->dev),
                NV_PCI_SLOT_NUMBER(dma_map->dev),
                *va, *va + (dma_map->page_count * PAGE_SIZE - 1),
                nv->dma_addressable_start,
                nv->dma_addressable_limit);
        nv_dma_unmap_contig(dma_map);
        return NV_ERR_INVALID_ADDRESS;
    }

    return NV_OK;
}

static void nv_dma_unmap_contig(nv_dma_map_t *dma_map)
{
    pci_unmap_page(dma_map->dev, dma_map->mapping.contig.dma_addr,
            dma_map->page_count * PAGE_SIZE, PCI_DMA_BIDIRECTIONAL);
}

static void nv_fill_scatterlist
(
    struct scatterlist *sgl,
    struct page **pages,
    unsigned int page_count
)
{
    unsigned int i;
    struct scatterlist *sg;
#if defined(for_each_sg)
    for_each_sg(sgl, sg, page_count, i)
    {
        sg_set_page(sg, pages[i], PAGE_SIZE, 0);
    }
#else
    for (i = 0; i < page_count; i++)
    {
        sg = &(sgl)[i];
        sg->page = pages[i];
        sg->length = PAGE_SIZE;
        sg->offset = 0;
    }
#endif
}

NV_STATUS nv_create_dma_map_scatterlist(nv_dma_map_t *dma_map)
{
    /*
     * We need to split our mapping into at most 4GB - PAGE_SIZE chunks.
     * The Linux kernel stores the length (and offset) of a scatter-gather
     * segment as an unsigned int, so it will overflow if we try to do
     * anything larger.
     */
    NV_STATUS status;
    nv_dma_submap_t *submap;
    NvU32 i;
    NvU64 allocated_size = 0;
    NvU64 num_submaps = dma_map->page_count + NV_DMA_SUBMAP_MAX_PAGES - 1;
    NvU64 total_size = dma_map->page_count << PAGE_SHIFT;

    /* 
     * This turns into 64-bit division, which the ARMv7 kernel doesn't provide
     * implicitly. Instead, we need to use the platform's do_div() to perform
     * the division.
     */
    do_div(num_submaps, NV_DMA_SUBMAP_MAX_PAGES);

    WARN_ON(NvU64_HI32(num_submaps) != 0);

    dma_map->mapping.discontig.submap_count = NvU64_LO32(num_submaps);
    
    status = os_alloc_mem((void **)&dma_map->mapping.discontig.submaps,
        sizeof(nv_dma_submap_t) * dma_map->mapping.discontig.submap_count);
    if (status != NV_OK)
    {
        return status;
    }

    os_mem_set((void *)dma_map->mapping.discontig.submaps, 0,
        sizeof(nv_dma_submap_t) * dma_map->mapping.discontig.submap_count);

    NV_FOR_EACH_DMA_SUBMAP(dma_map, submap, i)
    {
        NvU64 submap_size = NV_MIN(NV_DMA_SUBMAP_MAX_PAGES << PAGE_SHIFT,
                                   total_size - allocated_size);

        submap->page_count = (NvU32)(submap_size >> PAGE_SHIFT);

        status = NV_ALLOC_DMA_SUBMAP_SCATTERLIST(dma_map, submap, i);
        if (status != NV_OK)
        {
            submap->page_count = 0;
            break;
        }

#if !defined(NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT) || \
    defined(NV_DOM0_KERNEL_PRESENT)
        {
            NvU64 page_idx = NV_DMA_SUBMAP_IDX_TO_PAGE_IDX(i);
            nv_fill_scatterlist(NV_DMA_SUBMAP_SCATTERLIST(submap),
                &dma_map->pages[page_idx], submap->page_count);
        }
#endif

        allocated_size += submap_size;
    }

    WARN_ON(allocated_size != total_size);

    if (status != NV_OK)
    {
        nv_destroy_dma_map_scatterlist(dma_map);
    }

    return status;
}

NV_STATUS nv_map_dma_map_scatterlist(nv_dma_map_t *dma_map)
{
    NV_STATUS status = NV_OK;
    nv_dma_submap_t *submap;
    NvU64 i;

    NV_FOR_EACH_DMA_SUBMAP(dma_map, submap, i)
    {
        submap->sg_map_count = pci_map_sg(dma_map->dev,
                NV_DMA_SUBMAP_SCATTERLIST(submap),
                NV_DMA_SUBMAP_SCATTERLIST_LENGTH(submap),
                PCI_DMA_BIDIRECTIONAL);
        if (submap->sg_map_count == 0)
        {
            status = NV_ERR_OPERATING_SYSTEM;
            break;
        }
    }

    if (status != NV_OK)
    {
        nv_unmap_dma_map_scatterlist(dma_map);
    }

    return status;
}

void nv_unmap_dma_map_scatterlist(nv_dma_map_t *dma_map)
{
    nv_dma_submap_t *submap;
    NvU64 i;

    NV_FOR_EACH_DMA_SUBMAP(dma_map, submap, i)
    {
        if (submap->sg_map_count == 0)
        {
            break;
        }

        pci_unmap_sg(dma_map->dev, NV_DMA_SUBMAP_SCATTERLIST(submap),
                NV_DMA_SUBMAP_SCATTERLIST_LENGTH(submap),
                PCI_DMA_BIDIRECTIONAL);
    }
}

void nv_destroy_dma_map_scatterlist(nv_dma_map_t *dma_map)
{
    nv_dma_submap_t *submap;
    NvU64 i;

    NV_FOR_EACH_DMA_SUBMAP(dma_map, submap, i)
    {
        if (submap->page_count == 0)
        {
            break;
        }

        NV_FREE_DMA_SUBMAP_SCATTERLIST(submap);
    }

    os_free_mem(dma_map->mapping.discontig.submaps);
}

void nv_load_dma_map_scatterlist(
    nv_dma_map_t *dma_map,
    NvU64 *va_array
)
{
    unsigned int i, j;
    struct scatterlist *sg;
    nv_dma_submap_t *submap;
    NvU64 sg_addr, sg_off, sg_len, k, l = 0;

    NV_FOR_EACH_DMA_SUBMAP(dma_map, submap, i)
    {
        NV_FOR_EACH_DMA_SUBMAP_SG(submap, sg, j)
        {
            /*
             * It is possible for pci_map_sg() to merge scatterlist entries, so
             * make sure we account for that here.
             */
            for (sg_addr = sg_dma_address(sg), sg_len = sg_dma_len(sg),
                    sg_off = 0, k = 0;
                 (sg_off < sg_len) && (k < submap->page_count);
                 sg_off += PAGE_SIZE, l++, k++)
            {
                va_array[l] = sg_addr + sg_off;
            }
        }
    }
}

static NV_STATUS nv_dma_map_scatterlist(
    nv_state_t   *nv,
    nv_dma_map_t *dma_map,
    NvU64        *va_array
)
{
    NV_STATUS status;
    NvU64 i;

    status = nv_create_dma_map_scatterlist(dma_map);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: Failed to allocate DMA mapping scatterlist!\n");
        return status;
    }

    status = nv_map_dma_map_scatterlist(dma_map);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: failed to create a DMA mapping!\n");
        nv_destroy_dma_map_scatterlist(dma_map);
        return status;
    }

    nv_load_dma_map_scatterlist(dma_map, va_array);

    for (i = 0; i < dma_map->page_count; i++)
    {
        if (!IS_DMA_ADDRESSABLE(nv, va_array[i]))
        {
            nv_printf(NV_DBG_ERRORS,
                    "NVRM: DMA address not in addressable range of device "
                    "%04x:%02x:%02x (0x%llx, 0x%llx-0x%llx)\n",
                    NV_PCI_DOMAIN_NUMBER(dma_map->dev),
                    NV_PCI_BUS_NUMBER(dma_map->dev),
                    NV_PCI_SLOT_NUMBER(dma_map->dev),
                    va_array[i], nv->dma_addressable_start,
                    nv->dma_addressable_limit);
            nv_dma_unmap_scatterlist(dma_map);
            return NV_ERR_INVALID_ADDRESS;
        }
    }

    return NV_OK;
}

static void nv_dma_unmap_scatterlist(nv_dma_map_t *dma_map)
{
    nv_unmap_dma_map_scatterlist(dma_map);
    nv_destroy_dma_map_scatterlist(dma_map);
}

static void nv_dma_nvlink_addr_compress
(
    nv_state_t *nv,
    NvU64      *va_array,
    NvU64       page_count,
    NvBool      contig
)
{
#if defined(NVCPU_PPC64LE)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvU64 addr = 0;
    NvU64 i;

    /*
     * On systems that support NVLink sysmem links, apply the required address
     * compression scheme when links are trained. Otherwise check that PCIe and
     * NVLink DMA mappings are equivalent as per requirements of Bug 1920398.
     */
    if (nvl->npu == NULL)
    {
        return;
    }

    if (nv->nvlink_sysmem_links_enabled)
    {
        for (i = 0; i < (contig ? 1 : page_count); i++)
        {
            va_array[i] = nv_compress_nvlink_addr(va_array[i]);
        }

        return;
    }

    for (i = 0; i < (contig ? 1 : page_count); i++)
    {
        addr = nv_compress_nvlink_addr(va_array[i]);
        if (WARN_ONCE(va_array[i] != addr,
                      "unexpected DMA address compression (0x%llx, 0x%llx)\n",
                      va_array[i], addr))
        {
            break;
        }
    }
#endif
}

static void nv_dma_nvlink_addr_decompress
(
    nv_state_t *nv,
    NvU64      *va_array,
    NvU64       page_count,
    NvBool      contig
)
{
#if defined(NVCPU_PPC64LE)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvU64 i;

    if (nvl->npu == NULL)
    {
        return;
    }

    if (nv->nvlink_sysmem_links_enabled)
    {
        for (i = 0; i < (contig ? 1 : page_count); i++)
        {
            va_array[i] = nv_expand_nvlink_addr(va_array[i]);
        }
    }
#endif
}

NV_STATUS NV_API_CALL nv_dma_map_pages(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *va_array,
    NvBool      contig,
    void      **priv
)
{
    NV_STATUS status;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nv_dma_map_t *dma_map = NULL;

    if (priv == NULL)
    {
        /*
         * IOMMU path has not been implemented yet to handle
         * anything except a nv_dma_map_t as the priv argument.
         */
        return NV_ERR_NOT_SUPPORTED;
    }

    if (page_count > os_get_num_phys_pages())
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: DMA mapping request too large!\n");
        return NV_ERR_INVALID_REQUEST;
    }

    status = os_alloc_mem((void **)&dma_map, sizeof(nv_dma_map_t));
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: Failed to allocate nv_dma_map_t!\n");
        return status;
    }

    dma_map->dev = nvl->dev;
    dma_map->pages = *priv;
    dma_map->page_count = page_count;
    dma_map->contiguous = contig;

    if (dma_map->page_count > 1 && !dma_map->contiguous)
    {
        dma_map->mapping.discontig.submap_count = 0;
        status = nv_dma_map_scatterlist(nv, dma_map, va_array);
    }
    else
    {
        /* 
         * Force single-page mappings to be contiguous to avoid scatterlist
         * overhead.
         */
        dma_map->contiguous = NV_TRUE;

        status = nv_dma_map_contig(nv, dma_map, va_array);
    }

    if (status != NV_OK)
    {
        os_free_mem(dma_map);
    }
    else
    {
        *priv = dma_map;
        nv_dma_nvlink_addr_compress(nv, va_array, dma_map->page_count,
                                    dma_map->contiguous);
    }

    return status;
}

NV_STATUS NV_API_CALL nv_dma_unmap_pages(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *va_array,
    void      **priv
)
{
    nv_dma_map_t *dma_map;

    if (priv == NULL)
    {
        /*
         * IOMMU path has not been implemented yet to handle
         * anything except a nv_dma_map_t as the priv argument.
         */
        return NV_ERR_NOT_SUPPORTED;
    }

    dma_map = *priv;

    if (page_count > os_get_num_phys_pages())
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: DMA unmapping request too large!\n");
        return NV_ERR_INVALID_REQUEST;
    }

    if (page_count != dma_map->page_count)
    {
        nv_printf(NV_DBG_WARNINGS,
                "NVRM: Requested to DMA unmap %llu pages, but there are %llu "
                "in the mapping\n", page_count, dma_map->page_count);
        return NV_ERR_INVALID_REQUEST;
    }

    *priv = dma_map->pages;

    if (dma_map->contiguous)
    {
        nv_dma_unmap_contig(dma_map);
    }
    else
    {
        nv_dma_unmap_scatterlist(dma_map);
    }

    os_free_mem(dma_map);

    return NV_OK;
}

/*
 * Wrappers used for DMA-remapping an nv_alloc_t during transition to more
 * generic interfaces.
 */
NV_STATUS NV_API_CALL nv_dma_map_alloc
(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *va_array,
    NvBool      contig,
    void      **priv
)
{
    NV_STATUS status;
    NvU64 i;
    nv_alloc_t *at = *priv;
    struct page **pages = NULL;
    NvU64 pages_size = sizeof(struct page *) * (contig ? 1 : page_count);

    /*
     * Convert the nv_alloc_t into a struct page * array for
     * nv_dma_map_pages().
     */
    status = os_alloc_mem((void **)&pages, pages_size);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: Failed to allocate page array for DMA mapping!\n");
        return status;
    }

    os_mem_set(pages, 0, pages_size);

    if (at != NULL)
    {
        WARN_ON(page_count != at->num_pages);

        if (NV_ALLOC_MAPPING_USER(at->flags))
        {
            pages[0] = at->user_pages[0];
            if (!contig)
            {
                for (i = 1; i < page_count; i++)
                {
                    pages[i] = at->user_pages[i];
                }
            }
        }
    }

    if ((at == NULL) || !(NV_ALLOC_MAPPING_USER(at->flags)))
    {
        pages[0] = NV_GET_PAGE_STRUCT(va_array[0]);
        if (!contig)
        {
            for (i = 1; i < page_count; i++)
            {
                pages[i] = NV_GET_PAGE_STRUCT(va_array[i]);
            }
        }
    }

    *priv = pages;
    status = nv_dma_map_pages(nv, page_count, va_array, contig, priv);
    if (status != NV_OK)
    {
        *priv = at;
        os_free_mem(pages);
    }

    return status;
}

NV_STATUS NV_API_CALL nv_dma_unmap_alloc
(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *va_array,
    void      **priv
)
{
    NV_STATUS status = nv_dma_unmap_pages(nv, page_count, va_array, priv);
    if (status != NV_OK)
    {
        /*
         * If nv_dma_unmap_pages() fails, we hit an assert condition and the
         * priv argument won't be the page array we allocated in
         * nv_dma_map_alloc(), so we skip the free here. But note that since
         * this is an assert condition it really should never happen.
         */
        return status;
    }

    /* Free the struct page * array allocated by nv_dma_map_alloc() */
    os_free_mem(*priv);

    return NV_OK;
}

/* DMA-map a peer PCI device's BAR for peer access. */
NV_STATUS NV_API_CALL nv_dma_map_peer
(
    nv_state_t *nv,
    nv_state_t *peer,
    NvU8        bar_index,
    NvU64       page_count,
    NvU64      *va
)
{
    nv_linux_state_t *nvlpeer = NV_GET_NVL_FROM_NV_STATE(peer);
    struct resource *res;
    NV_STATUS status;

    BUG_ON(bar_index >= NV_GPU_NUM_BARS);
    res = &nvlpeer->dev->resource[bar_index];
    if (res->start == 0)
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: resource %u not valid for GPU " NV_PCI_DEV_FMT,
                bar_index, NV_PCI_DEV_FMT_ARGS(peer));
        return NV_ERR_INVALID_REQUEST;
    }

    if ((*va < res->start) || ((*va + (page_count * PAGE_SIZE)) > res->end))
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: mapping requested (start = 0x%llx, page_count = 0x%llx)"
                " outside of resource bounds (start = 0x%llx, end = 0x%llx)\n",
                *va, page_count, res->start, res->end);
        return NV_ERR_INVALID_REQUEST;
    }

    status = nv_dma_map_mmio(nv, page_count, va);
    if (status == NV_ERR_NOT_SUPPORTED)
    {
        /*
         * Best effort - can't map through the iommu but at least try to
         * convert to a bus address.
         */
        NvU64 offset = *va - res->start;
        *va = nv_pci_bus_address(nvlpeer, bar_index) + offset;
        status = NV_OK;
    }

    return status;
}

void NV_API_CALL nv_dma_unmap_peer
(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64       va
)
{
    /*
     * It's safe to always call nv_dma_unmap_mmio() here. If nv_dma_map_peer()
     * succeeded, then either we didn't call dma_map_resource() or we did
     * and it succeeded. nv_dma_unmap_mmio() handles both cases.
     */
    nv_dma_unmap_mmio(nv, page_count, va);
}

/* DMA-map another anonymous device's MMIO region for peer access. */
NV_STATUS NV_API_CALL nv_dma_map_mmio
(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *va
)
{
#if defined(NV_DMA_MAP_RESOURCE_PRESENT)
    NvU64 mmio_addr;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    const struct dma_map_ops *ops = get_dma_ops(&nvl->dev->dev);

    /* 
     * The default implementation passes through the source address
     * without failing. However, we may want to try something else
     * so we need to be able to differentiate between no map_resource
     * implementation and no change to the mapped address.
     */
    if (!ops->map_resource)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    BUG_ON(!va);

    mmio_addr = *va;

    *va = dma_map_resource(&nvl->dev->dev, mmio_addr, page_count * PAGE_SIZE,
                           DMA_BIDIRECTIONAL, 0);
    if (dma_mapping_error(&nvl->dev->dev, *va))
    {
        nv_printf(NV_DBG_ERRORS,
                "NVRM: failed to DMA map MMIO range [0x%llx-0x%llx]\n",
                mmio_addr, mmio_addr + page_count * PAGE_SIZE - 1);
        return NV_ERR_OPERATING_SYSTEM;
    }

    nv_dma_nvlink_addr_compress(nv, va, page_count, NV_TRUE);

    return NV_OK;
#else
    return NV_ERR_NOT_SUPPORTED;
#endif
}

void NV_API_CALL nv_dma_unmap_mmio
(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64       va
)
{
#if defined(NV_DMA_MAP_RESOURCE_PRESENT)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    const struct dma_map_ops *ops = get_dma_ops(&nvl->dev->dev);

    /* Make sure we only unmap what we mapped */
    if (!ops->map_resource)
    {
        return;
    }

    nv_dma_nvlink_addr_decompress(nv, &va, page_count, NV_TRUE);

    dma_unmap_resource(&nvl->dev->dev, va, page_count * PAGE_SIZE,
                       DMA_BIDIRECTIONAL, 0);
#endif
}
