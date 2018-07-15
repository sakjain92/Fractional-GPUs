/*******************************************************************************
    Copyright (c) 2017 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

/*
 * nv-ibmnpu.c - interface with the ibmnpu (IBM NVLink Processing Unit) "module"
 */
#include "nv-linux.h"

#if defined(NVCPU_PPC64LE)
#include "nv-ibmnpu.h"

#include "nvlink_common.h"
#include "nvlink_errors.h"
#include "nvlink_proto.h"

/*
 * GPU device memory can be exposed to the kernel as NUMA node memory via the
 * IBMNPU devices associated with the GPU. The platform firmware will specify
 * the parameters of where the memory lives in the system address space via
 * firmware properties on the IBMNPU devices. These properties specify what
 * memory can be accessed through the IBMNPU device, and the driver can online
 * a GPU device's memory into the range accessible by its associated IBMNPU
 * devices.
 *
 * This function calls over to the IBMNPU driver to query the parameters from
 * firmware, and validates that the resulting parameters are acceptable.
 */
static void nv_init_ibmnpu_numa_info(nv_state_t *nv)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nv_numa_info_t *numa_info = &nvl->npu->numa_info;
    struct pci_dev *npu_dev = nvl->npu->devs[0];
    NvU64 spa, gpa, aper_size;

    NV_ATOMIC_SET(numa_info->status, NV_IOCTL_NUMA_STATUS_DISABLED);

    /*
     * Terminology:
     * - system physical address (spa): 47-bit NVIDIA physical address, which
     *      is the CPU real address with the NVLink address compression scheme
     *      already applied in firmware.
     * - guest physical address (gpa): 56-bit physical address as seen by the
     *      operating system. This is the base address that we should use for
     *      onlining device memory.
     */
    numa_info->node_id = ibmnpu_device_get_memory_config(npu_dev, &spa, &gpa,
                                                         &aper_size);
    if (numa_info->node_id == NUMA_NO_NODE)
    {
        nv_printf(NV_DBG_SETUP,
            "NVRM: no NUMA memory aperture found for GPU device "
            NV_PCI_DEV_FMT "\n", NV_PCI_DEV_FMT_ARGS(nv));
        return;
    }

    /* Validate that the compressed system physical address is not too wide */
    if (spa & (~(BIT_ULL(nv_volta_dma_addr_size) - 1)))
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: invalid NUMA memory system pa 0x%llx"
            " on IBM-NPU device %04x:%02x:%02x.%u\n",
            spa, NV_PCI_DOMAIN_NUMBER(npu_dev), NV_PCI_BUS_NUMBER(npu_dev),
            NV_PCI_SLOT_NUMBER(npu_dev), PCI_FUNC(npu_dev->devfn));
        goto invalid_numa_config;
    }

    /*
     * Validate that the guest physical address is aligned to 128GB.
     * This alignment requirement comes from the Volta address space
     * size on POWER9.
     */
    if (!IS_ALIGNED(gpa, BIT_ULL(nv_volta_addr_space_width)))
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: invalid alignment in NUMA memory guest pa 0x%llx"
            " on IBM-NPU device %04x:%02x:%02x.%u\n",
            gpa, NV_PCI_DOMAIN_NUMBER(npu_dev), NV_PCI_BUS_NUMBER(npu_dev),
            NV_PCI_SLOT_NUMBER(npu_dev), PCI_FUNC(npu_dev->devfn));
        goto invalid_numa_config;
    }

    /* Validate that the aperture can map all of the device's framebuffer */
    if (aper_size < nv->fb->size)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: insufficient NUMA memory aperture size 0x%llx"
            " on IBM-NPU device %04x:%02x:%02x.%u (0x%llx required)\n",
            aper_size, NV_PCI_DOMAIN_NUMBER(npu_dev),
            NV_PCI_BUS_NUMBER(npu_dev), NV_PCI_SLOT_NUMBER(npu_dev),
            PCI_FUNC(npu_dev->devfn), nv->fb->size);
        goto invalid_numa_config;
    }

    numa_info->compr_sys_phys_addr = spa;
    numa_info->guest_phys_addr = gpa;

    if (NVreg_EnableUserNUMAManagement)
    {
        NV_ATOMIC_SET(numa_info->status, NV_IOCTL_NUMA_STATUS_OFFLINE);
    }
    else
    {
        nv_printf(NV_DBG_SETUP, "NVRM: user-mode NUMA onlining disabled.\n");
    }

    nv_printf(NV_DBG_SETUP,
        "NVRM: " NV_PCI_DEV_FMT " NUMA memory aperture: "
        "[spa = 0x%llx, gpa = 0x%llx, aper_size = 0x%llx]\n",
        NV_PCI_DEV_FMT_ARGS(nv), spa, gpa, aper_size);

    return;

invalid_numa_config:
    nv_printf(NV_DBG_ERRORS,
        "NVRM: NUMA memory aperture for GPU device " NV_PCI_DEV_FMT
        " disabled due to invalid firmware configuration\n",
        NV_PCI_DEV_FMT_ARGS(nv));
    numa_info->node_id = NUMA_NO_NODE;
}

void nv_init_ibmnpu_info(nv_state_t *nv)
{
#if defined(NV_PNV_PCI_GET_NPU_DEV_PRESENT)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    struct pci_dev *npu_dev = pnv_pci_get_npu_dev(nvl->dev, 0);
    NvU8 dev_count;

    if (!npu_dev)
    {
        return;
    }

    if (os_alloc_mem((void **)&nvl->npu, sizeof(nv_ibmnpu_info_t)) != NV_OK)
    {
        return;
    }

    /* Find any other IBMNPU devices attached to this GPU */
    for (nvl->npu->devs[0] = npu_dev, dev_count = 1;
         dev_count < NV_MAX_ATTACHED_IBMNPUS; dev_count++)
    {
        nvl->npu->devs[dev_count] = pnv_pci_get_npu_dev(nvl->dev, dev_count);
        if (!nvl->npu->devs[dev_count])
        {
            break;
        }
    }

    nvl->npu->dev_count = dev_count;

    nvl->npu->relaxed_ordering_enabled = NV_FALSE;

    /*
     * If we run out of space for IBMNPU devices, NV_MAX_ATTACHED_IBMNPUS will
     * need to be bumped.
     */
    WARN_ON((dev_count == NV_MAX_ATTACHED_IBMNPUS) &&
            pnv_pci_get_npu_dev(nvl->dev, dev_count));

    ibmnpu_device_get_genregs_info(npu_dev, &nvl->npu->genregs);

    if (nvl->npu->genregs.size > 0)
    {
        nv_printf(NV_DBG_SETUP,
            "NVRM: IBM-NPU device %04x:%02x:%02x.%u associated with GPU device "
            NV_PCI_DEV_FMT " has a generation register space 0x%llx-0x%llx\n",
            NV_PCI_DOMAIN_NUMBER(npu_dev), NV_PCI_BUS_NUMBER(npu_dev),
            NV_PCI_SLOT_NUMBER(npu_dev), PCI_FUNC(npu_dev->devfn),
            NV_PCI_DEV_FMT_ARGS(nv),
            nvl->npu->genregs.start_addr,
            nvl->npu->genregs.start_addr + nvl->npu->genregs.size - 1);
    }
    else
    {
        nv_printf(NV_DBG_SETUP,
            "NVRM: IBM-NPU device %04x:%02x:%02x.%u associated with GPU device "
            NV_PCI_DEV_FMT " does not support generation registers\n",
            NV_PCI_DOMAIN_NUMBER(npu_dev), NV_PCI_BUS_NUMBER(npu_dev),
            NV_PCI_SLOT_NUMBER(npu_dev), PCI_FUNC(npu_dev->devfn),
            NV_PCI_DEV_FMT_ARGS(nv));
    }

    nv_init_ibmnpu_numa_info(nv);
#endif
}

void nv_destroy_ibmnpu_info(nv_state_t *nv)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    if (nvl->npu != NULL)
    {
        os_free_mem(nvl->npu);
    }
}

int nv_init_ibmnpu_devices(nv_state_t *nv)
{
    NvU8 i;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (!nvl->npu)
    {
        return 0;
    }

    for (i = 0; i < nvl->npu->dev_count; i++)
    {
        nv_printf(NV_DBG_SETUP,
            "NVRM: initializing IBM-NPU device %04x:%02x:%02x.%u"
            " associated with GPU device " NV_PCI_DEV_FMT "\n",
            NV_PCI_DOMAIN_NUMBER(nvl->npu->devs[i]),
            NV_PCI_BUS_NUMBER(nvl->npu->devs[i]),
            NV_PCI_SLOT_NUMBER(nvl->npu->devs[i]),
            PCI_FUNC(nvl->npu->devs[i]->devfn),
            NV_PCI_DEV_FMT_ARGS(nv));

        if (ibmnpu_init_device(nvl->npu->devs[i]) != NVL_SUCCESS)
        {
            return -EIO;
        }
    }

    return 0;
}

NV_STATUS NV_API_CALL nv_get_ibmnpu_genreg_info(nv_state_t *nv, NvU64 *base,
                                                NvU64 *size)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (nvl->npu == NULL || nvl->npu->genregs.size == 0)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    *base = nvl->npu->genregs.start_addr;
    *size = nvl->npu->genregs.size;

    return NV_OK;
}

NV_STATUS NV_API_CALL nv_get_ibmnpu_relaxed_ordering_mode(nv_state_t *nv,
                                                          NvBool *mode)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (nvl->npu == NULL || nvl->npu->genregs.size == 0)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    *mode = nvl->npu->relaxed_ordering_enabled ||
            NVreg_EnableIBMNPURelaxedOrderingMode;

    return NV_OK;
}

#else

void nv_init_ibmnpu_info(nv_state_t *nv)
{
}

void nv_destroy_ibmnpu_info(nv_state_t *nv)
{
}

int nv_init_ibmnpu_devices(nv_state_t *nv)
{
    return 0;
}

NV_STATUS NV_API_CALL nv_get_ibmnpu_genreg_info(nv_state_t *nv, NvU64 *base,
                                                NvU64 *size)
{
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS NV_API_CALL nv_get_ibmnpu_relaxed_ordering_mode(nv_state_t *nv,
                                                          NvBool *mode)
{
    return NV_ERR_NOT_SUPPORTED;
}

#endif
