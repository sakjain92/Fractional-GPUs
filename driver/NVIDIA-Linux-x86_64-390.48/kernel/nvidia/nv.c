/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"
#include "nv-p2p.h"
#include "nv-reg.h"
#include "rmil.h"
#include "nv-instance.h"

#include "nvlink_proto.h"

#if defined(NV_UVM_ENABLE)
#include "nv_uvm_interface.h"
#endif

#if defined(NV_VGPU_KVM_BUILD)
#include "nv-vgpu-vfio-interface.h"
#endif

#include "nv-frontend.h"
#include "nv-hypervisor.h"
#include "nv-ibmnpu.h"
#include "nv-kthread-q.h"
#include "nv-pat.h"

/*
 * The module information macros for Linux single-module builds
 * are present in nv-frontend.c.
 */

#if (NV_BUILD_MODULE_INSTANCES != 0)
#if defined(MODULE_LICENSE)
MODULE_LICENSE("NVIDIA");
#endif
#if defined(MODULE_INFO)
MODULE_INFO(supported, "external");
#endif
#if defined(MODULE_VERSION)
MODULE_VERSION(NV_VERSION_STRING);
#endif
#ifdef MODULE_ALIAS_CHARDEV_MAJOR
MODULE_ALIAS_CHARDEV_MAJOR(NV_MAJOR_DEVICE_NUMBER);
#endif
#endif

#include "conftest/patches.h"

/*
 * our global state; one per device
 */

static NvU32 num_nv_devices = 0;
NvU32 num_probed_nv_devices = 0;

NvU32 nv_assign_gpu_count = 0;
nv_pci_info_t nv_assign_gpu_pci_info[NV_MAX_DEVICES];

nv_linux_state_t *nv_linux_devices;

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
NvU64 __nv_supported_pte_mask = ~_PAGE_NX;
#endif

/*
 * And one for the control device
 */

nv_linux_state_t nv_ctl_device = { { 0 } };
wait_queue_head_t nv_ctl_waitqueue;

#if defined(NV_CHANGE_PAGE_ATTR_BUG_PRESENT)
static const char *__cpgattr_warning = \
    "Your Linux kernel has known problems in its implementation of\n"
    "the change_page_attr() kernel interface.\n\n"
    "The NVIDIA graphics driver will attempt to work around these\n"
    "problems, but system stability may be adversely affected.\n"
    "It is recommended that you update to Linux 2.6.11 (or a newer\n"
    "Linux kernel release).\n";

static const char *__cpgattr_warning_2 = \
    "Your Linux kernel's version and architecture indicate that it\n"
    "may have an implementation of the change_page_attr() kernel\n"
    "kernel interface known to have problems. The NVIDIA graphics\n"
    "driver made an attempt to determine whether your kernel is\n"
    "affected, but could not. It will assume the interface does not\n"
    "work correctly and attempt to employ workarounds.\n"
    "This may adversely affect system stability.\n"
    "It is recommended that you update to Linux 2.6.11 (or a newer\n"
    "Linux kernel release).\n";
#endif

static int nv_mmconfig_failure_detected = 0;
static const char *__mmconfig_warning = \
    "Your current system configuration has known problems when\n"
    "accessing PCI Configuration Space that can lead to accesses\n"
    "to the PCI Configuration Space of the wrong PCI device. This\n"
    "is known to cause instabilities with the NVIDIA graphics driver.\n\n"
    "Please see the MMConfig section in the readme for more information\n"
    "on how to work around this problem.\n";

#if (defined(NVCPU_X86) || defined(NVCPU_X86_64))
static int nv_fbdev_failure_detected = 0;
static const char *__fbdev_warning = \
    "Your system is not currently configured to drive a VGA console\n"
    "on the primary VGA device. The NVIDIA Linux graphics driver\n"
    "requires the use of a text-mode VGA console. Use of other console\n"
    "drivers including, but not limited to, vesafb, may result in\n"
    "corruption and stability problems, and is not supported.\n";
#endif

#define NV_UPDATE_MEMORY_TYPES_DEFAULT 1

int nv_update_memory_types = NV_UPDATE_MEMORY_TYPES_DEFAULT;

nv_cpu_type_t nv_cpu_type = NV_CPU_TYPE_UNKNOWN;

void *nvidia_p2p_page_t_cache;
static void *nvidia_pte_t_cache;
void *nvidia_stack_t_cache;
static nvidia_stack_t *__nv_init_sp;

static int nv_tce_bypass_mode = NV_TCE_BYPASS_MODE_DEFAULT;

struct semaphore nv_linux_devices_lock;

static NvTristate nv_chipset_is_io_coherent = NV_TRISTATE_INDETERMINATE;

static int nv_use_threaded_interrupts = 0;

// allow an easy way to convert all debug printfs related to events
// back and forth between 'info' and 'errors'
#if defined(NV_DBG_EVENTS)
#define NV_DBG_EVENTINFO NV_DBG_ERRORS
#else
#define NV_DBG_EVENTINFO NV_DBG_INFO
#endif

//
// Attempt to determine if we are running into the MMCONFIG coherency
// issue and, if so, warn the user and stop attempting to verify
// and correct the BAR values (see NV_CHECK_PCI_CONFIG_SPACE()), so
// that we do not do more harm than good.
//
#define NV_CHECK_MMCONFIG_FAILURE(nv,bar,value)                            \
    {                                                                      \
        nv_linux_state_t *nvl;                                             \
        LOCK_NV_LINUX_DEVICES();                                           \
        for (nvl = nv_linux_devices; nvl != NULL;  nvl = nvl->next)        \
        {                                                                  \
            nv_state_t *nv_tmp = NV_STATE_PTR(nvl);                        \
            if (((nv) != nv_tmp) &&                                        \
                (nv_tmp->bars[(bar)].bus_address == (value)))              \
            {                                                              \
                nv_printf(NV_DBG_ERRORS, "NVRM: %s", __mmconfig_warning);  \
                nv_procfs_add_warning("mmconfig", __mmconfig_warning);     \
                nv_mmconfig_failure_detected = 1;                          \
                UNLOCK_NV_LINUX_DEVICES();                                 \
                return;                                                    \
            }                                                              \
        }                                                                  \
        UNLOCK_NV_LINUX_DEVICES();                                         \
    }

static void
verify_pci_bars(
    nv_state_t  *nv,
    void        *dev_handle
)
{
    NvU32 bar, bar_hi, bar_lo;

    //
    // If an MMCONFIG specific failure was detected, skip the
    // PCI BAR verification to avoid overwriting the BAR(s)
    // of a given device with those of this GPU. See above for
    // more information.
    //
    if (nv_mmconfig_failure_detected)
        return;

    for (bar = 0; bar < NV_GPU_NUM_BARS; bar++)
    {
        nv_aperture_t *tmp = &nv->bars[bar];

        bar_lo = bar_hi = 0;
        if (tmp->offset == 0)
            continue;

        os_pci_read_dword(dev_handle, tmp->offset, &bar_lo);

        if ((bar_lo & NVRM_PCICFG_BAR_ADDR_MASK)
                != NvU64_LO32(tmp->bus_address))
        {
            nv_printf(NV_DBG_USERERRORS,
                "NVRM: BAR%u(L) is 0x%08x, will restore to 0x%08llx.\n",
                bar, bar_lo, NvU64_LO32(tmp->bus_address) |
                             (bar_lo & ~NVRM_PCICFG_BAR_ADDR_MASK));

            NV_CHECK_MMCONFIG_FAILURE(nv, bar,
                    (bar_lo & NVRM_PCICFG_BAR_ADDR_MASK));

            os_pci_write_dword(dev_handle, tmp->offset,
                NvU64_LO32(tmp->bus_address));
        }

        if ((bar_lo & NVRM_PCICFG_BAR_MEMTYPE_MASK)
                != NVRM_PCICFG_BAR_MEMTYPE_64BIT)
            continue;

        os_pci_read_dword(dev_handle, (tmp->offset + 4), &bar_hi);

        if (bar_hi != NvU64_HI32(tmp->bus_address))
        {
            nv_printf(NV_DBG_USERERRORS,
                "NVRM: BAR%u(H) is 0x%08x, will restore to 0x%08llx.\n",
                bar, bar_hi, NvU64_HI32(tmp->bus_address));

            os_pci_write_dword(dev_handle, (tmp->offset + 4),
                    NvU64_HI32(tmp->bus_address));
        }
    }
}

void nv_check_pci_config_space(nv_state_t *nv, BOOL check_the_bars)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    unsigned short cmd = 0, flag = 0;

    pci_read_config_word(nvl->dev, PCI_COMMAND, &cmd);
    if (!(cmd & PCI_COMMAND_MASTER))
    {
        nv_printf(NV_DBG_USERERRORS, "NVRM: restoring bus mastering!\n");
        cmd |= PCI_COMMAND_MASTER;
        flag = 1;
    }

    if (!(cmd & PCI_COMMAND_MEMORY))
    {
        nv_printf(NV_DBG_USERERRORS, "NVRM: restoring MEM access!\n");
        cmd |= PCI_COMMAND_MEMORY;
        flag = 1;
    }

    if (cmd & PCI_COMMAND_SERR)
    {
        nv_printf(NV_DBG_USERERRORS, "NVRM: clearing SERR enable bit!\n");
        cmd &= ~PCI_COMMAND_SERR;
        flag = 1;
    }

    if (cmd & PCI_COMMAND_INTX_DISABLE)
    {
        nv_printf(NV_DBG_USERERRORS, "NVRM: clearing INTx disable bit!\n");
        cmd &= ~PCI_COMMAND_INTX_DISABLE;
        flag = 1;
    }

    if (flag)
        pci_write_config_word(nvl->dev, PCI_COMMAND, cmd);

    if (check_the_bars && NV_MAY_SLEEP() && !(nv->flags & NV_FLAG_PASSTHRU))
        verify_pci_bars(nv, nvl->dev);
}

void NV_API_CALL nv_verify_pci_config(
    nv_state_t *nv,
    BOOL        check_the_bars
)
{
    nv_linux_state_t *nvl;
    nvidia_stack_t *sp;

    if ((nv)->flags & NV_FLAG_USE_BAR0_CFG)
    {
        nvl = NV_GET_NVL_FROM_NV_STATE(nv);
        sp = nvl->sp[NV_DEV_STACK_PCI_CFGCHK];

        rm_check_pci_config_space(sp, nv,
                check_the_bars, FALSE, NV_MAY_SLEEP());
    }
    else
        nv_check_pci_config_space(nv, NV_MAY_SLEEP());
}

/***
 *** STATIC functions, only in this file
 ***/

/* nvos_ functions.. do not take a state device parameter  */
static int      nvos_count_devices(nvidia_stack_t *);

static nv_alloc_t  *nvos_create_alloc(struct pci_dev *, int);
static int          nvos_free_alloc(nv_alloc_t *);

/* lock-related functions that should only be called from this file */
static NvBool nv_lock_init_locks(nvidia_stack_t *sp, nv_state_t *nv);
static void   nv_lock_destroy_locks(nvidia_stack_t *sp, nv_state_t *nv);


/***
 *** EXPORTS to Linux Kernel
 ***/

static long          nvidia_unlocked_ioctl  (struct file *, unsigned int, unsigned long);
static void          nvidia_isr_tasklet_bh  (unsigned long);
static irqreturn_t   nvidia_isr_kthread_bh  (int, void *);
static irqreturn_t   nvidia_isr_common_bh   (void *);
static void          nvidia_isr_bh_unlocked (void *);
#if !defined(NV_IRQ_HANDLER_T_PRESENT) || (NV_IRQ_HANDLER_T_ARGUMENT_COUNT == 3)
static irqreturn_t   nvidia_isr             (int, void *, struct pt_regs *);
#else
static irqreturn_t   nvidia_isr             (int, void *);
#endif

static int           nvidia_ctl_open        (struct inode *, struct file *);
static int           nvidia_ctl_close       (struct inode *, struct file *);

#if defined(NV_PCI_ERROR_RECOVERY)
static pci_ers_result_t nvidia_pci_error_detected   (struct pci_dev *, enum pci_channel_state);
static pci_ers_result_t nvidia_pci_mmio_enabled     (struct pci_dev *);

struct pci_error_handlers nv_pci_error_handlers = {
    .error_detected = nvidia_pci_error_detected,
    .mmio_enabled   = nvidia_pci_mmio_enabled,
};
#endif

/***
 *** see nv.h for functions exported to other parts of resman
 ***/

/***
 *** STATIC functions
 ***/

static
nv_alloc_t *nvos_create_alloc(
    struct pci_dev *dev,
    int num_pages
)
{
    nv_alloc_t *at;
    unsigned int pt_size, i;

    NV_KMALLOC(at, sizeof(nv_alloc_t));
    if (at == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate alloc info\n");
        return NULL;
    }

    memset(at, 0, sizeof(nv_alloc_t));

    at->dev = dev;
    pt_size = num_pages *  sizeof(nvidia_pte_t *);
    if (os_alloc_mem((void **)&at->page_table, pt_size) != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate page table\n");
        NV_KFREE(at, sizeof(nv_alloc_t));
        return NULL;
    }

    memset(at->page_table, 0, pt_size);
    at->num_pages = num_pages;
    NV_ATOMIC_SET(at->usage_count, 0);

    for (i = 0; i < at->num_pages; i++)
    {
        at->page_table[i] = NV_KMEM_CACHE_ALLOC(nvidia_pte_t_cache);
        if (at->page_table[i] == NULL)
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: failed to allocate page table entry\n");
            nvos_free_alloc(at);
            return NULL;
        }
        memset(at->page_table[i], 0, sizeof(nvidia_pte_t));
    }

    at->pid = os_get_current_process();

    return at;
}

static
int nvos_free_alloc(
    nv_alloc_t *at
)
{
    unsigned int i;

    if (at == NULL)
        return -1;

    if (NV_ATOMIC_READ(at->usage_count))
        return 1;

    for (i = 0; i < at->num_pages; i++)
    {
        if (at->page_table[i] != NULL)
            NV_KMEM_CACHE_FREE(at->page_table[i], nvidia_pte_t_cache);
    }
    os_free_mem(at->page_table);

    NV_KFREE(at, sizeof(nv_alloc_t));

    return 0;
}

NvU8 nv_find_pci_capability(struct pci_dev *dev, NvU8 capability)
{
    u16 status = 0;
    u8  cap_ptr = 0, cap_id = 0xff;

    pci_read_config_word(dev, PCI_STATUS, &status);
    status &= PCI_STATUS_CAP_LIST;
    if (!status)
        return 0;

    switch (dev->hdr_type) {
        case PCI_HEADER_TYPE_NORMAL:
        case PCI_HEADER_TYPE_BRIDGE:
            pci_read_config_byte(dev, PCI_CAPABILITY_LIST, &cap_ptr);
            break;
        default:
            return 0;
    }

    do {
        cap_ptr &= 0xfc;
        pci_read_config_byte(dev, cap_ptr + PCI_CAP_LIST_ID, &cap_id);
        if (cap_id == capability)
            return cap_ptr;
        pci_read_config_byte(dev, cap_ptr + PCI_CAP_LIST_NEXT, &cap_ptr);
    } while (cap_ptr && cap_id != 0xff);

    return 0;
}

/*!
 * @brief This function accepts pci information corresponding to a GPU
 * and returns a reference to the nv_linux_state_t corresponding to that GPU.
 *
 * @param[in] domain            Pci domain number for the GPU to be found.
 * @param[in] bus               Pci bus number for the GPU to be found.
 * @param[in] slot              Pci slot number for the GPU to be found.
 * @param[in] function          Pci function number for the GPU to be found.
 *
 * @return Pointer to nv_linux_state_t for the GPU if it is found, or NULL otherwise.
 */
nv_linux_state_t * find_pci(NvU16 domain, NvU8 bus, NvU8 slot, NvU8 function)
{
    nv_linux_state_t *nvl = NULL;

    LOCK_NV_LINUX_DEVICES();

    for (nvl = nv_linux_devices; nvl != NULL; nvl = nvl->next)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);

        if (nv->pci_info.domain == domain &&
            nv->pci_info.bus == bus &&
            nv->pci_info.slot == slot &&
            nv->pci_info.function == function)
        {
            break;
        }
    }

    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

static void nv_state_init_gpu_uuid_cache(nv_state_t* nv)
{
    memset(&nv->nv_gpu_uuid_cache, 0x0, sizeof(nv->nv_gpu_uuid_cache));
}

static void nv_state_set_gpu_uuid_cache(nv_state_t* nv, const NvU8* uuid)
{
    if (uuid)
    {
        if (nv->nv_gpu_uuid_cache.bValid)
        {
            WARN_ON(memcmp(&nv->nv_gpu_uuid_cache.uuid, uuid, GPU_UUID_LEN));
        }
        else
        {
            memcpy(&nv->nv_gpu_uuid_cache.uuid, uuid, GPU_UUID_LEN);
            nv->nv_gpu_uuid_cache.bValid = NV_TRUE;
        }
    }
}

static const NvU8* nv_state_get_gpu_uuid_cache(nv_state_t* nv)
{
    if (nv->nv_gpu_uuid_cache.bValid)
    {
        return nv->nv_gpu_uuid_cache.uuid;
    }
    else
    {
        return NULL;
    }
}

#if defined(NV_CHANGE_PAGE_ATTR_BUG_PRESENT)
#define NV_PGD_OFFSET(address, kernel, mm)              \
   ({                                                   \
        struct mm_struct *__mm = (mm);                  \
        pgd_t *__pgd;                                   \
        if (!kernel)                                    \
            __pgd = pgd_offset(__mm, address);          \
        else                                            \
            __pgd = pgd_offset_k(address);              \
        __pgd;                                          \
    })

#define NV_PGD_PRESENT(pgd)                             \
   ({                                                   \
         if ((pgd != NULL) &&                           \
             (pgd_bad(*pgd) || pgd_none(*pgd)))         \
            /* static */ pgd = NULL;                    \
         pgd != NULL;                                   \
    })

#if defined(pmd_offset_map)
#define NV_PMD_OFFSET(address, pgd)                     \
   ({                                                   \
        pmd_t *__pmd;                                   \
        __pmd = pmd_offset_map(pgd, address);           \
   })
#define NV_PMD_UNMAP(pmd) pmd_unmap(pmd);
#else
#if defined(P4D_SHIFT)
/* 5-level page tables */
#define NV_PMD_OFFSET(address, pgd)                     \
   ({                                                   \
        pmd_t *__pmd = NULL;                            \
        p4d_t *__p4d;                                   \
        pud_t *__pud;                                   \
        __p4d = p4d_offset(pgd, address);               \
        if ((__p4d != NULL) &&                          \
            !(p4d_bad(*__p4d) || p4d_none(*__p4d))) {   \
            __pud = pud_offset(__p4d, address);         \
                                                        \
            if ((__pud != NULL) &&                      \
                !(pud_bad(*__pud) || pud_none(*__pud))) \
                __pmd = pmd_offset(__pud, address);     \
        }                                               \
        __pmd;                                          \
    })
#elif defined(PUD_SHIFT)
/* 4-level page tables */
#define NV_PMD_OFFSET(address, pgd)                     \
   ({                                                   \
        pmd_t *__pmd = NULL;                            \
        pud_t *__pud;                                   \
        __pud = pud_offset(pgd, address);               \
        if ((__pud != NULL) &&                          \
            !(pud_bad(*__pud) || pud_none(*__pud)))     \
            __pmd = pmd_offset(__pud, address);         \
        __pmd;                                          \
    })
#else
/* 3-level page tables */
#define NV_PMD_OFFSET(address, pgd)                     \
   ({                                                   \
        pmd_t *__pmd;                                   \
        __pmd = pmd_offset(pgd, address);               \
    })
#endif
#define NV_PMD_UNMAP(pmd)
#endif

#define NV_PMD_PRESENT(pmd)                             \
   ({                                                   \
        if ((pmd != NULL) &&                            \
            (pmd_bad(*pmd) || pmd_none(*pmd)))          \
        {                                               \
            NV_PMD_UNMAP(pmd);                          \
            pmd = NULL; /* mark invalid */              \
        }                                               \
        pmd != NULL;                                    \
    })

#if defined(pte_offset_atomic)
#define NV_PTE_OFFSET(address, pmd)                     \
   ({                                                   \
        pte_t *__pte;                                   \
        __pte = pte_offset_atomic(pmd, address);        \
        NV_PMD_UNMAP(pmd); __pte;                       \
    })
#define NV_PTE_UNMAP(pte) pte_kunmap(pte);
#elif defined(pte_offset)
#define NV_PTE_OFFSET(address, pmd)                     \
   ({                                                   \
        pte_t *__pte;                                   \
        __pte = pte_offset(pmd, address);               \
        NV_PMD_UNMAP(pmd); __pte;                       \
    })
#define NV_PTE_UNMAP(pte)
#else
#define NV_PTE_OFFSET(address, pmd)                     \
   ({                                                   \
        pte_t *__pte;                                   \
        __pte = pte_offset_map(pmd, address);           \
        NV_PMD_UNMAP(pmd); __pte;                       \
    })
#define NV_PTE_UNMAP(pte) pte_unmap(pte);
#endif

#define NV_PTE_PRESENT(pte)                             \
   ({                                                   \
        if ((pte != NULL) && !pte_present(*pte))        \
        {                                               \
            NV_PTE_UNMAP(pte);                          \
            pte = NULL; /* mark invalid */              \
        }                                               \
        pte != NULL;                                    \
    })

#define NV_PTE_VALUE(pte)                               \
   ({                                                   \
        unsigned long __pte_value = pte_val(*pte);      \
        NV_PTE_UNMAP(pte);                              \
        __pte_value;                                    \
    })
/*
 * nv_verify_cpa_interface() - determine if the change_page_attr() large page
 * management accounting bug known to exist in early Linux/x86-64 kernels
 * is present in this kernel.
 *
 * There's really no good way to determine if change_page_attr() is working
 * correctly. We can't reliably use change_page_attr() on Linux/x86-64 2.6
 * kernels < 2.6.11: if we run into the accounting bug, the Linux kernel will
 * trigger a BUG() if we attempt to restore the WB memory type of a page
 * originally part of a large page.
 *
 * So if we can successfully allocate such a page, change its memory type to
 * UC and check if the accounting was done correctly, we can determine if
 * the change_page_attr() interface can be used safely.
 *
 * Return values:
 *    0 - test passed, the change_page_attr() interface works
 *    1 - test failed, the status is unclear
 *   -1 - test failed, the change_page_attr() interface is broken
 */

static inline pte_t *check_large_page(unsigned long vaddr)
{
    pgd_t *pgd = NULL;
    pmd_t *pmd = NULL;

    pgd = NV_PGD_OFFSET(vaddr, 1, NULL);
    if (!NV_PGD_PRESENT(pgd))
        return NULL;

    pmd = NV_PMD_OFFSET(vaddr, pgd);
    if (!pmd || pmd_none(*pmd))
        return NULL;

    if (!pmd_large(*pmd))
        return NULL;

    return (pte_t *) pmd;
}

#define CPA_FIXED_MAX_ALLOCS 500

int nv_verify_cpa_interface(void)
{
    unsigned int i, size;
    unsigned long large_page = 0;
    unsigned long *vaddr_list;
    size = sizeof(unsigned long) * CPA_FIXED_MAX_ALLOCS;

    NV_KMALLOC(vaddr_list, size);
    if (!vaddr_list)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: nv_verify_cpa_interface: failed to allocate "
            "page table\n");
        return 1;
    }

    memset(vaddr_list, 0, size);

    /* try to track down an allocation from a 2M page. */
    for (i = 0; i < CPA_FIXED_MAX_ALLOCS; i++)
    {
        vaddr_list[i] =  __get_free_page(GFP_KERNEL);
        if (!vaddr_list[i])
            continue;

#if defined(_PAGE_NX)
        if ((pgprot_val(PAGE_KERNEL) & _PAGE_NX) &&
                virt_to_phys((void *)vaddr_list[i]) < 0x400000)
            continue;
#endif

        if (check_large_page(vaddr_list[i]) != NULL)
        {
            large_page = vaddr_list[i];
            vaddr_list[i] = 0;
            break;
        }
    }

    for (i = 0; i < CPA_FIXED_MAX_ALLOCS; i++)
    {
        if (vaddr_list[i])
            free_page(vaddr_list[i]);
    }
    NV_KFREE(vaddr_list, size);

    if (large_page)
    {
        struct page *page = virt_to_page(large_page);
        struct page *kpte_page;
        pte_t *kpte;
        unsigned long kpte_val;
        pgprot_t prot;

        // lookup a pointer to our pte
        kpte = check_large_page(large_page);
        kpte_val = pte_val(*kpte);
        kpte_page = virt_to_page(((unsigned long)kpte) & PAGE_MASK);

        prot = PAGE_KERNEL_NOCACHE;
        pgprot_val(prot) &= __nv_supported_pte_mask;

        // this should split the large page
        change_page_attr(page, 1, prot);

        // broken kernels may get confused after splitting the page and
        // restore the page before returning to us. detect that case.
        if (((pte_val(*kpte) & ~_PAGE_NX) == kpte_val) &&
            (pte_val(*kpte) & _PAGE_PSE))
        {
            if ((pte_val(*kpte) & _PAGE_NX) &&
                    (__nv_supported_pte_mask & _PAGE_NX) == 0)
                clear_bit(_PAGE_BIT_NX, kpte);
            // don't change the page back, as it's already been reverted
            put_page(kpte_page);
            free_page(large_page);
            return -1;  // yep, we're broken
        }

        // ok, now see if our bookkeeping is broken
        if (page_count(kpte_page) != 0)
            return -1;  // yep, we're broken

        prot = PAGE_KERNEL;
        pgprot_val(prot) &= __nv_supported_pte_mask;

        // everything's ok!
        change_page_attr(page, 1, prot);
        free_page(large_page);
        return 0;
    }

    return 1;
}
#endif /* defined(NV_CHANGE_PAGE_ATTR_BUG_PRESENT) */

int __init nvidia_init_module(void)
{
    NV_STATUS status;
    int rc;
    NvU32 count, data, i;
    nv_state_t *nv = NV_STATE_PTR(&nv_ctl_device);
    nvidia_stack_t *sp = NULL;

    if (nv_multiple_kernel_modules)
    {
        nv_printf(NV_DBG_INFO, "NVRM: nvidia module instance %d\n",
                  nv_module_instance);
    }

    nv_user_map_init();

    rc = nv_heap_create();
    if (rc < 0)
    {
        goto failed6;
    }

    rc = nv_mem_pool_create();
    if (rc < 0)
    {
        goto failed6;
    }

    nv_memdbg_init();

    nvidia_stack_t_cache = NV_KMEM_CACHE_CREATE(nvidia_stack_cache_name,
                                                nvidia_stack_t);
    if (nvidia_stack_t_cache == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: stack cache allocation failed!\n");
        rc = -ENOMEM;
        goto failed6;
    }

    rc = nv_kmem_cache_alloc_stack(&sp);
    if (rc != 0)
    {
        NV_KMEM_CACHE_DESTROY(nvidia_stack_t_cache);
        goto failed6;
    }

    rc = nvlink_core_init();
    if (rc < 0)
    {
        nv_printf(NV_DBG_INFO, "NVRM: Nvlink Core init failed\n");
    }
    else
    {
#if defined(NVCPU_PPC64LE)
        rc = ibmnpu_init();
        if (rc < 0)
        {
            nv_printf(NV_DBG_INFO, "NVRM: Ibmnpu init failed\n");
        }
#endif










    }

    if (!rm_init_rm(sp))
    {
        nv_kmem_cache_free_stack(sp);
        NV_KMEM_CACHE_DESTROY(nvidia_stack_t_cache);
        nv_printf(NV_DBG_ERRORS, "NVRM: rm_init_rm() failed!\n");
        rc = -EIO;
        goto failed5;
    }

    // init the nvidia control device
    nv->os_state = (void *) &nv_ctl_device;
    if (!nv_lock_init_locks(sp, nv))
    {
        rc = -ENOMEM;
        goto failed4;
    }
    nv_state_init_gpu_uuid_cache(nv);

    count = nvos_count_devices(sp);
    if (count == 0)
    {
        if (NV_IS_ASSIGN_GPU_PCI_INFO_SPECIFIED())
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: The requested GPU assignments are invalid. Please ensure\n"
                "NVRM: that the GPUs you wish to assign to this kernel module\n"
                "NVRM: are present and available.\n");
        }
        else
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: No NVIDIA graphics adapter found!\n");
        }
        rc = -ENODEV;
        goto failed4;
    }

    nv_linux_devices = NULL;
    NV_INIT_MUTEX(&nv_linux_devices_lock);

    /*
     * Determine the TCE bypass mode here so it can be used during device
     * probe. This must happen after the call to rm_init_rm(), because the
     * registry must be initialized.
     *
     * Also determine whether we should allow user-mode NUMA onlining of
     * device memory.
     */
    if (NVCPU_IS_PPC64LE)
    {
        status = rm_read_registry_dword(sp, nv,
                "NVreg", NV_REG_TCE_BYPASS_MODE, &data);
        if ((status == NV_OK) && ((int)data != NV_TCE_BYPASS_MODE_DEFAULT))
        {
            nv_tce_bypass_mode = data;
        }

        if (NVreg_EnableUserNUMAManagement)
        {
            /* Force on the core RM regkey to match */
            status = rm_write_registry_dword(sp, nv,
                    "NVreg", "RMNumaOnlining", 1);
            WARN_ON(status != NV_OK);
        }
    }

    rc = nv_register_chrdev((void *)&nv_fops);
    if (rc < 0)
        goto failed4;

    /* create /proc/driver/nvidia/... */
    rc = nv_register_procfs();
    if (rc < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to register procfs!\n");
        rc = -ENODEV;
        goto failed3;
    }

    if (nv_pci_register_driver(&nv_pci_driver) < 0)
    {
        rc = -ENODEV;
        nv_printf(NV_DBG_ERRORS, "NVRM: No NVIDIA graphics adapter found!\n");
        goto failed2;
    }

    // Init the per GPU based registry keys.
    nv_parse_per_device_option_string(sp);

    if (num_probed_nv_devices != count)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: The NVIDIA probe routine was not called for %d device(s).\n",
            count - num_probed_nv_devices);
        nv_printf(NV_DBG_ERRORS,
            "NVRM: This can occur when a driver such as: \n"
            "NVRM: nouveau, rivafb, nvidiafb or rivatv "
#if (NV_BUILD_MODULE_INSTANCES != 0)
            "NVRM: or another NVIDIA kernel module "
#endif
            "\nNVRM: was loaded and obtained ownership of the NVIDIA device(s).\n");
        nv_printf(NV_DBG_ERRORS,
            "NVRM: Try unloading the conflicting kernel module (and/or\n"
            "NVRM: reconfigure your kernel without the conflicting\n"
            "NVRM: driver(s)), then try loading the NVIDIA kernel module\n"
            "NVRM: again.\n");
    }

    if (num_probed_nv_devices == 0)
    {
        rc = -ENODEV;
        nv_printf(NV_DBG_ERRORS, "NVRM: No NVIDIA graphics adapter probed!\n");
        goto failed1;
    }

    if (num_probed_nv_devices != num_nv_devices)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: The NVIDIA probe routine failed for %d device(s).\n",
            num_probed_nv_devices - num_nv_devices);
    }

    if (num_nv_devices == 0)
    {
        rc = -ENODEV;
        nv_printf(NV_DBG_ERRORS,
            "NVRM: None of the NVIDIA graphics adapters were initialized!\n");
        goto failed1;
    }

#ifdef NV_REQUEST_THREADED_IRQ_PRESENT
    /*
     * You only get this choice if your kernel is new enough to have the
     * request_threaded_irq() routine. Otherwise, nv_use_threaded_interrupts == 0
     * and must stay that way.
     */
    status = rm_read_registry_dword(sp, nv,
                 "NVreg", NV_REG_USE_THREADED_INTERRUPTS, &data);
    if (status == NV_OK)
    {
        nv_use_threaded_interrupts = (int)data;
    }
#else

    // Overwrite the regkey to zero, so that core RM (which calls
    // osReadRegistryDword() to read regkeys) doesn't think that
    // threaded interrupts are enabled.
    data = 0;
    rm_write_registry_dword(sp, nv, "NVreg",
                            NV_REG_USE_THREADED_INTERRUPTS, data);

#endif

    /*
     * TODO: Bug 1758006: remove this once RM settles on a final choice for
     *       interrupt bottom halves (tasklets vs. kernel threads).
     */
    if (nv_use_threaded_interrupts)
        nv_printf(NV_DBG_ERRORS, "NVRM: loading %s (using threaded interrupts)\n", pNVRM_ID);
    else
        nv_printf(NV_DBG_ERRORS, "NVRM: loading %s (using tasklets)\n", pNVRM_ID);

    for (i = 0; __nv_patches[i].short_description; i++)
    {
        if (i == 0)
            nv_printf(NV_DBG_ERRORS, "NVRM: Applied patches:\n");

        // Report patches via one-based, rather than zero-based numbering:
        nv_printf(NV_DBG_ERRORS,
            "NVRM:    Patch #%d: %s\n", i + 1, __nv_patches[i].short_description);
    }

    nvidia_pte_t_cache = NV_KMEM_CACHE_CREATE(nvidia_pte_cache_name,
                                              nvidia_pte_t);
    if (nvidia_pte_t_cache == NULL)
    {
        rc = -ENOMEM;
        nv_printf(NV_DBG_ERRORS, "NVRM: pte cache allocation failed\n");
        goto failed;
    }

    if (!nv_multiple_kernel_modules)
    {
        nvidia_p2p_page_t_cache = NV_KMEM_CACHE_CREATE(nvidia_p2p_page_cache_name,
                                                       nvidia_p2p_page_t);
        if (nvidia_p2p_page_t_cache == NULL)
        {
            rc = -ENOMEM;
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: p2p page cache allocation failed\n");
            goto failed;
        }
    }

#if (defined(NVCPU_X86_64) || (defined(NVCPU_X86) && defined(CONFIG_X86_PAE)))
    if (boot_cpu_has(X86_FEATURE_NX))
    {
        NvU32 __eax, __edx;
        rdmsr(MSR_EFER, __eax, __edx);
        if ((__eax & EFER_NX) != 0)
            __nv_supported_pte_mask |= _PAGE_NX;
    }
    if (_PAGE_NX != ((NvU64)1<<63))
    {
        /*
         * Make sure we don't strip software no-execute
         * bits from PAGE_KERNEL(_NOCACHE) before calling
         * change_page_attr().
         */
        __nv_supported_pte_mask |= _PAGE_NX;
    }
#endif

    /*
     * Give users an opportunity to disable the driver's use of
     * the change_page_attr(), set_pages_{uc,wb}() and set_memory_{uc,wb}() kernel
     * interfaces.
     */
    status = rm_read_registry_dword(sp, nv,
            "NVreg", NV_REG_UPDATE_MEMORY_TYPES, &data);
    if ((status == NV_OK) && ((int)data != ~0))
    {
        nv_update_memory_types = data;
    }

#if defined(NV_CHANGE_PAGE_ATTR_BUG_PRESENT)
    /*
     * Unless we explicitely detect that the change_page_attr()
     * inteface is fixed, disable usage of the interface on
     * this kernel. Notify the user of this problem using the
     * driver's /proc warnings interface (read by the installer
     * and the bug report script).
     */
    else
    {
        rc = nv_verify_cpa_interface();
        if (rc < 0)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: %s", __cpgattr_warning);
            nv_procfs_add_warning("change_page_attr", __cpgattr_warning);
            nv_update_memory_types = 0;
        }
        else if (rc != 0)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: %s", __cpgattr_warning_2);
            nv_procfs_add_warning("change_page_attr", __cpgattr_warning_2);
            nv_update_memory_types = 0;
        }
    }
#endif /* defined(NV_CHANGE_PAGE_ATTR_BUG_PRESENT) */

#if defined(NVCPU_X86_64) && defined(CONFIG_IA32_EMULATION) && \
  !defined(NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL)
    rm_register_compatible_ioctls(sp);
#endif

    rc = nv_init_pat_support(sp);
    if (rc < 0)
        goto failed;

#if defined(NV_UVM_ENABLE)
    rc = nv_uvm_init();
    if (rc != 0)
        goto failed;
#endif

    __nv_init_sp = sp;

    return 0;

failed:
#if defined(NV_UVM_ENABLE)
    nv_uvm_exit();
#endif

    if (nvidia_p2p_page_t_cache != NULL)
        NV_KMEM_CACHE_DESTROY(nvidia_p2p_page_t_cache);

    if (nvidia_pte_t_cache != NULL)
        NV_KMEM_CACHE_DESTROY(nvidia_pte_t_cache);

failed1:
    pci_unregister_driver(&nv_pci_driver);

failed2:
    nv_unregister_procfs();

failed3:
    nv_unregister_chrdev((void *)&nv_fops);

failed4:
    nv_lock_destroy_locks(sp, nv);
    rm_shutdown_rm(sp);

    nv_kmem_cache_free_stack(sp);
    NV_KMEM_CACHE_DESTROY(nvidia_stack_t_cache);

failed5:





#if defined(NVCPU_PPC64LE)
    ibmnpu_exit();
#endif

    nvlink_core_exit();

failed6:
    nv_mem_pool_destroy();
    nv_heap_destroy();

    return rc;
}

void nvidia_exit_module(void)
{
    nvidia_stack_t *sp = __nv_init_sp;
    nv_state_t *nv = NV_STATE_PTR(&nv_ctl_device);

    nv_printf(NV_DBG_INFO, "NVRM: nvidia_exit_module\n");

#if defined(NV_UVM_ENABLE)
    nv_uvm_exit();
#endif





#if defined(NVCPU_PPC64LE)
    ibmnpu_exit();
#endif

    nvlink_core_exit();

    pci_unregister_driver(&nv_pci_driver);

    /* remove /proc/driver/nvidia/... */
    nv_unregister_procfs();

    nv_unregister_chrdev((void *)&nv_fops);

    nv_lock_destroy_locks(sp, nv);

    // Shutdown the resource manager
    rm_shutdown_rm(sp);

#if defined(NVCPU_X86_64) && defined(CONFIG_IA32_EMULATION) && \
  !defined(NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL)
    rm_unregister_compatible_ioctls(sp);
#endif

    nv_teardown_pat_support();

    nv_memdbg_exit();

    if (!nv_multiple_kernel_modules)
    {
        NV_KMEM_CACHE_DESTROY(nvidia_p2p_page_t_cache);
    }
    NV_KMEM_CACHE_DESTROY(nvidia_pte_t_cache);

    nv_kmem_cache_free_stack(sp);
    NV_KMEM_CACHE_DESTROY(nvidia_stack_t_cache);

    nv_mem_pool_destroy();
    nv_heap_destroy();
}


/*
 * Module entry and exit functions for Linux single-module builds
 * are present in nv-frontend.c.
 */

#if (NV_BUILD_MODULE_INSTANCES != 0)
module_init(nvidia_init_module);
module_exit(nvidia_exit_module);
#endif

void *nv_alloc_file_private(void)
{
    nv_file_private_t *nvfp;
    unsigned int i;

    NV_KMALLOC(nvfp, sizeof(nv_file_private_t));
    if (!nvfp)
        return NULL;

    memset(nvfp, 0, sizeof(nv_file_private_t));

    for (i = 0; i < NV_FOPS_STACK_INDEX_COUNT; ++i)
    {
        NV_INIT_MUTEX(&nvfp->fops_sp_lock[i]);
    }
    init_waitqueue_head(&nvfp->waitqueue);
    NV_SPIN_LOCK_INIT(&nvfp->fp_lock);

    return nvfp;
}

void nv_free_file_private(nv_file_private_t *nvfp)
{
    nvidia_event_t *nvet;

    if (nvfp == NULL)
        return;

    for (nvet = nvfp->event_head; nvet != NULL; nvet = nvfp->event_head)
    {
        nvfp->event_head = nvfp->event_head->next;
        NV_KFREE(nvet, sizeof(nvidia_event_t));
    }
    NV_KFREE(nvfp, sizeof(nv_file_private_t));
}


static int nv_is_control_device(
    struct inode *inode
)
{
    return (minor((inode)->i_rdev) == \
            (NV_CONTROL_DEVICE_MINOR - nv_module_instance));
}

/*
 * Search the global list of nv devices for the one with the given minor device
 * number. If found, nvl is returned with nvl->ldata_lock taken.
 */
static nv_linux_state_t *find_minor(NvU32 minor)
{
    nv_linux_state_t *nvl;

    LOCK_NV_LINUX_DEVICES();
    nvl = nv_linux_devices;
    while (nvl != NULL)
    {
        if (nvl->minor_num == minor)
        {
            down(&nvl->ldata_lock);
            break;
        }
        nvl = nvl->next;
    }

    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

/*
 * Search the global list of nv devices for the one with the given gpu_id.
 * If found, nvl is returned with nvl->ldata_lock taken.
 */
static nv_linux_state_t *find_gpu_id(NvU32 gpu_id)
{
    nv_linux_state_t *nvl;

    LOCK_NV_LINUX_DEVICES();
    nvl = nv_linux_devices;
    while (nvl != NULL)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);
        if (nv->gpu_id == gpu_id)
        {
            down(&nvl->ldata_lock);
            break;
        }
        nvl = nvl->next;
    }

    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

/*
 * Search the global list of nv devices for the one with the given UUID. Devices
 * with missing UUID information are ignored. If found, nvl is returned with
 * nvl->ldata_lock taken.
 */
static nv_linux_state_t *find_uuid(const NvU8 *uuid)
{
    nv_linux_state_t *nvl = NULL;
    nv_state_t *nv;
    const NvU8 *dev_uuid;

    LOCK_NV_LINUX_DEVICES();

    for (nvl = nv_linux_devices; nvl; nvl = nvl->next)
    {
        nv = NV_STATE_PTR(nvl);
        down(&nvl->ldata_lock);
        dev_uuid = nv_state_get_gpu_uuid_cache(nv);
        if (dev_uuid && memcmp(dev_uuid, uuid, GPU_UUID_LEN) == 0)
            goto out;
        up(&nvl->ldata_lock);
    }

out:
    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

/*
 * Search the global list of nv devices. The search logic is:
 *
 * 1) If any device has the given UUID, return it
 *
 * 2) If no device has the given UUID but at least one non-GVI device is missing
 *    its UUID (for example because rm_init_adapter has not run on it yet),
 *    return that device.
 *
 * 3) If no device has the given UUID and all UUIDs are present, return NULL.
 *
 * In cases 1 and 2, nvl is returned with nvl->ldata_lock taken.
 *
 * The reason for this weird logic is because UUIDs aren't always available. See
 * bug 1642200.
 */
static nv_linux_state_t *find_uuid_candidate(const NvU8 *uuid)
{
    nv_linux_state_t *nvl = NULL;
    nv_state_t *nv;
    const NvU8 *dev_uuid;
    int use_missing;
    int has_missing = 0;

    LOCK_NV_LINUX_DEVICES();

    /*
     * Take two passes through the list. The first pass just looks for the UUID.
     * The second looks for the target or missing UUIDs. It would be nice if
     * this could be done in a single pass by remembering which nvls are missing
     * UUIDs, but we have to hold the nvl lock after we check for the UUID.
     */
    for (use_missing = 0; use_missing <= 1; use_missing++)
    {
        for (nvl = nv_linux_devices; nvl; nvl = nvl->next)
        {
            nv = NV_STATE_PTR(nvl);
            down(&nvl->ldata_lock);
            dev_uuid = nv_state_get_gpu_uuid_cache(nv);
            if (dev_uuid)
            {
                /* Case 1: If a device has the given UUID, return it */
                if (memcmp(dev_uuid, uuid, GPU_UUID_LEN) == 0)
                    goto out;
            }
            else if (!NV_IS_GVI_DEVICE(nv))
            {
                /* Case 2: If no device has the given UUID but at least one non-
                 * GVI device is missing its UUID, return that device. */
                if (use_missing)
                    goto out;
                has_missing = 1;
            }
            up(&nvl->ldata_lock);
        }

        /* Case 3: If no device has the given UUID and all UUIDs are present,
         * return NULL. */
        if (!has_missing)
            break;
    }

out:
    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

static void nv_dev_free_stacks(nv_linux_state_t *nvl)
{
    NvU32 i;
    for (i = 0; i < NV_DEV_STACK_COUNT; i++)
    {
        if (nvl->sp[i])
        {
            nv_kmem_cache_free_stack(nvl->sp[i]);
            nvl->sp[i] = NULL;
        }
    }
}

static int nv_dev_alloc_stacks(nv_linux_state_t *nvl)
{
    NvU32 i;
    int rc;

    for (i = 0; i < NV_DEV_STACK_COUNT; i++)
    {
        rc = nv_kmem_cache_alloc_stack(&nvl->sp[i]);
        if (rc != 0)
        {
            nv_dev_free_stacks(nvl);
            return rc;
        }
    }

    return 0;
}

static int validate_numa_start_state(nv_linux_state_t *nvl)
{
    int rc = 0;
    int numa_status = nv_get_numa_status(nvl);

    if (numa_status != NV_IOCTL_NUMA_STATUS_DISABLED)
    {
        if (nv_ctl_device.numa_memblock_size == 0)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: numa memblock size of zero "
                      "found during device start");
            rc = -EINVAL;
        }
    }

    return rc;
}

/*
 * Brings up the device on the first file open. Assumes nvl->ldata_lock is held.
 */
static int nv_start_device(nv_state_t *nv, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
    NvU32 msi_config = 0;
#endif
    int rc = 0;
    NvBool kthread_init = NV_FALSE;

    rc = validate_numa_start_state(nvl);
    if (rc != 0)
    {
        goto failed;
    }

    if (nv->pci_info.device_id == 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: open of nonexistent "
                  "device bearing minor number %d\n", nvl->minor_num);
        rc = -ENXIO;
        goto failed;
    }

    rc = nv_init_ibmnpu_devices(nv);
    if (rc != 0)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: failed to initialize ibmnpu devices attached to device bearing minor number %d\n",
            nvl->minor_num);
        goto failed;
    }

    if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
    {
        rc = nv_dev_alloc_stacks(nvl);
        if (rc != 0)
            goto failed;
    }

#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
    if (!NV_IS_GVI_DEVICE(nv))
    {
        if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
        {
            rm_read_registry_dword(sp, nv, "NVreg", NV_REG_ENABLE_MSI,
                                   &msi_config);
            if ((msi_config == 1) &&
                    (nv_find_pci_capability(nvl->dev, PCI_CAP_ID_MSI)))
            {
                if (nv_find_pci_capability(nvl->dev, PCI_CAP_ID_PM))
                {
                    rc = pci_enable_msi(nvl->dev);
                    if (rc == 0)
                    {
                        nv->interrupt_line = nvl->dev->irq;
                        nv->flags |= NV_FLAG_USES_MSI;
                    }
                    else
                    {
                        nv->flags &= ~NV_FLAG_USES_MSI;
                        if (nvl->dev->irq != 0)
                        {
                            nv_printf(NV_DBG_ERRORS,
                                      "NVRM: failed to enable MSI. "
                                      "Therefore, using PCIe virtual-wire interrupts.\n");
                        }
                    }
                }
                else
                {
                    // Bug 200117372: Work around for pci_disable_device enable_cnt
                    // related bug in the Linux kernel
                    nv->flags &= ~NV_FLAG_USES_MSI;
                    nv_printf(NV_DBG_INFO,
                              "NVRM: Failed to query the PCI power management capability.\n"
                              "NVRM: Not enabling MSI interrupts.\n");
                }
            }
        }
    }
#endif

    if ((!(nv->flags & NV_FLAG_USES_MSI)) && (nvl->dev->irq == 0))
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: No interrupts of any type are available. Cannot use this GPU.\n");
        rc = -EIO;
        goto failed;
    }

    if (NV_IS_GVI_DEVICE(nv))
    {
        rc = request_irq(nv->interrupt_line, nv_gvi_kern_isr, IRQF_SHARED,
                         nv_device_name, (void *)nvl);
        if (rc == 0)
        {
            nvl->work.data = (void *)nvl;
            NV_WORKQUEUE_INIT(&nvl->work.task, nv_gvi_kern_bh,
                              (void *)&nvl->work);
            rm_init_gvi_device(sp, nv);
            goto done;
        }
    }
    else
    {
        rc = 0;
        if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
        {
            if (nv_use_threaded_interrupts)
            {
#ifdef NV_REQUEST_THREADED_IRQ_PRESENT
                rc = request_threaded_irq(nv->interrupt_line, nvidia_isr,
                                          nvidia_isr_kthread_bh, IRQF_SHARED,
                                          nv_device_name, (void *)nvl);
#endif
            }
            else
            {
                rc = request_irq(nv->interrupt_line, nvidia_isr, IRQF_SHARED,
                                 nv_device_name, (void *)nvl);
            }
        }
    }
    if (rc != 0)
    {
        if ((nv->interrupt_line != 0) && (rc == -EBUSY))
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: Tried to get IRQ %d, but another driver\n",
                (unsigned int) nv->interrupt_line);
            nv_printf(NV_DBG_ERRORS, "NVRM: has it and is not sharing it.\n");
            nv_printf(NV_DBG_ERRORS, "NVRM: You may want to verify that no audio driver");
            nv_printf(NV_DBG_ERRORS, " is using the IRQ.\n");
        }
        nv_printf(NV_DBG_ERRORS, "NVRM: request_irq() failed (%d)\n", rc);
        goto failed;
    }

    if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE) && !nv_use_threaded_interrupts)
    {
        tasklet_init(&nvl->tasklet, nvidia_isr_tasklet_bh, (NvUPtr)NV_STATE_PTR(nvl));
    }

    if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
    {
        rc = os_alloc_mutex(&nvl->isr_bh_unlocked_mutex);
        if (rc != 0)
            goto failed;
        nv_kthread_q_item_init(&nvl->bottom_half_q_item, nvidia_isr_bh_unlocked, (void *)nv);
        rc = nv_kthread_q_init(&nvl->bottom_half_q, nv_device_name);
        if (rc != 0)
            goto failed;
        kthread_init = NV_TRUE;
    }

    if (!rm_init_adapter(sp, nv))
    {
        if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE) && !nv_use_threaded_interrupts)
        {
            tasklet_kill(&nvl->tasklet);
        }
        free_irq(nv->interrupt_line, (void *) nvl);
        nv_printf(NV_DBG_ERRORS, "NVRM: rm_init_adapter failed "
                  "for device bearing minor number %d\n", nvl->minor_num);
        rc = -EIO;
        goto failed;
    }

    if (!NV_IS_GVI_DEVICE(nv))
    {
        NvU8 *uuid;
        if (rm_get_gpu_uuid_raw(sp, nv, &uuid, NULL) == NV_OK)
        {
            nv_state_set_gpu_uuid_cache(nv, uuid);

#if defined(NV_UVM_ENABLE)
            nv_uvm_notify_start_device(uuid);
#endif

            os_free_mem(uuid);
        }
    }

done:
    nv->flags |= NV_FLAG_OPEN;
    return 0;

failed:
#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
    if (nv->flags & NV_FLAG_USES_MSI)
    {
        nv->flags &= ~NV_FLAG_USES_MSI;
        NV_PCI_DISABLE_MSI(nvl->dev);
    }
#endif
    if (kthread_init && !(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
        nv_kthread_q_stop(&nvl->bottom_half_q);

    if (nvl->isr_bh_unlocked_mutex)
    {
        os_free_mutex(nvl->isr_bh_unlocked_mutex);
        nvl->isr_bh_unlocked_mutex = NULL;
    }

    nv_dev_free_stacks(nvl);
    return rc;
}

/*
 * Makes sure the device is ready for operations and increases nvl->usage_count.
 * Assumes nvl->ldata_lock is held.
 */
static int nv_open_device(nv_state_t *nv, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    int rc;

    if (IS_VGX_HYPER())
    {
        /* fail open if GPU is being unbound */
        if (nv->flags & NV_FLAG_UNBIND_LOCK)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: open on device %04x:%02x:%02x.0"
                      " failed as GPU is locked for unbind operation\n",
                      nv->pci_info.domain, nv->pci_info.bus, nv->pci_info.slot);
            return -ENODEV;
        }
    }

    nv_printf(NV_DBG_INFO, "NVRM: opening device bearing minor number %d\n",
              nvl->minor_num);

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, NV_MAY_SLEEP());

    if ( ! (nv->flags & NV_FLAG_OPEN))
    {
        /* Sanity check: !NV_FLAG_OPEN requires usage_count == 0 */
        if (NV_ATOMIC_READ(nvl->usage_count) != 0)
        {
            nv_printf(NV_DBG_ERRORS,
                      "NVRM: Minor device %u is referenced without being open!\n",
                      nvl->minor_num);
            WARN_ON(1);
            return -EBUSY;
        }

        rc = nv_start_device(nv, sp);
        if (rc != 0)
            return rc;
    }

    NV_ATOMIC_INC(nvl->usage_count);
    return 0;
}

/*
** nvidia_open
**
** nv driver open entry point.  Sessions are created here.
*/
int
nvidia_open(
    struct inode *inode,
    struct file *file
)
{
    nv_state_t *nv = NULL;
    nv_linux_state_t *nvl = NULL;
    NvU32 minor_num;
    int rc = 0;
    nv_file_private_t *nvfp = NULL;
    nvidia_stack_t *sp = NULL;
    unsigned int i;
    unsigned int k;

    nv_printf(NV_DBG_INFO, "NVRM: nvidia_open...\n");

    nvfp = nv_alloc_file_private();
    if (nvfp == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate file private!\n");
        return -ENOMEM;
    }

    rc = nv_kmem_cache_alloc_stack(&sp);
    if (rc != 0)
    {
        nv_free_file_private(nvfp);
        return rc;
    }

    for (i = 0; i < NV_FOPS_STACK_INDEX_COUNT; ++i)
    {
        rc = nv_kmem_cache_alloc_stack(&nvfp->fops_sp[i]);
        if (rc != 0)
        {
            nv_kmem_cache_free_stack(sp);
            for (k = 0; k < i; ++k)
            {
                nv_kmem_cache_free_stack(nvfp->fops_sp[k]);
            }
            nv_free_file_private(nvfp);
            return rc;
        }
    }

    /* what device are we talking about? */
    minor_num = NV_DEVICE_MINOR_NUMBER(inode);

    nvfp->minor_num = minor_num;

    NV_SET_FILE_PRIVATE(file, nvfp);
    nvfp->sp = sp;

    /* for control device, just jump to its open routine */
    /* after setting up the private data */
    if (nv_is_control_device(inode))
    {
        rc = nvidia_ctl_open(inode, file);
        if (rc != 0)
            goto failed;
        return rc;
    }

    /* Takes nvl->ldata_lock */
    nvl = find_minor(minor_num);
    if (!nvl)
    {
        rc = -ENODEV;
        goto failed;
    }

    nvfp->nvptr = nvl;
    nv = NV_STATE_PTR(nvl);

    rc = nv_open_device(nv, sp);
    /* Fall-through on error */

    up(&nvl->ldata_lock);
failed:
    if (rc != 0)
    {
        if (nvfp != NULL)
        {
            nv_kmem_cache_free_stack(sp);
            for (i = 0; i < NV_FOPS_STACK_INDEX_COUNT; ++i)
            {
                nv_kmem_cache_free_stack(nvfp->fops_sp[i]);
            }
            nv_free_file_private(nvfp);
            NV_SET_FILE_PRIVATE(file, NULL);
        }
    }

    return rc;
}

static void validate_numa_shutdown_state(nv_linux_state_t *nvl)
{
    int numa_status = nv_get_numa_status(nvl);
    WARN_ON((numa_status != NV_IOCTL_NUMA_STATUS_OFFLINE) &&
            (numa_status != NV_IOCTL_NUMA_STATUS_DISABLED));
}

static void nv_shutdown_adapter(nvidia_stack_t *sp,
                                nv_state_t *nv,
                                nv_linux_state_t *nvl)
{
    validate_numa_shutdown_state(nvl);

    rm_disable_adapter(sp, nv);
    if (!nv_use_threaded_interrupts)
    {
        tasklet_kill(&nvl->tasklet);
    }

    // It's safe to call nv_kthread_q_stop even if queue is not initialized
    nv_kthread_q_stop(&nvl->bottom_half_q);
    if (nvl->isr_bh_unlocked_mutex)
    {
        os_free_mutex(nvl->isr_bh_unlocked_mutex);
        nvl->isr_bh_unlocked_mutex = NULL;
    }

    free_irq(nv->interrupt_line, (void *)nvl);
    if (nv->flags & NV_FLAG_USES_MSI)
        NV_PCI_DISABLE_MSI(nvl->dev);
    rm_shutdown_adapter(sp, nv);
}

/*
 * Tears down the device on the last file close. Assumes nvl->ldata_lock is
 * held.
 */
static void nv_stop_device(nv_state_t *nv, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    static int persistence_mode_notice_logged;

    if (NV_IS_GVI_DEVICE(nv))
    {
        rm_shutdown_gvi_device(sp, nv);
        NV_WORKQUEUE_FLUSH();
        free_irq(nv->interrupt_line, (void *)nvl);
    }
    else
    {
#if defined(NV_UVM_ENABLE)
        {
            const NvU8* uuid;
            // Inform UVM before disabling adapter. Use cached copy
            uuid = nv_state_get_gpu_uuid_cache(nv);
            if (uuid != NULL)
            {
                // this function cannot fail
                nv_uvm_notify_stop_device(uuid);
            }
        }
#endif
        if (nv->flags & NV_FLAG_PERSISTENT_SW_STATE)
        {
            rm_disable_adapter(sp, nv);
        }
        else
        {
            nv_shutdown_adapter(sp, nv, nvl);
        }
    }

    if (!(nv->flags & NV_FLAG_PERSISTENT_SW_STATE))
    {
        nv_dev_free_stacks(nvl);
    }

    if ((nv->flags & NV_FLAG_PERSISTENT_SW_STATE) &&
        (!persistence_mode_notice_logged) && (!IS_VGX_HYPER()))
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: Persistence mode is deprecated and"
                  " will be removed in a future release. Please use"
                  " nvidia-persistenced instead.\n");
        persistence_mode_notice_logged  = 1;
    }

    /* leave INIT flag alone so we don't reinit every time */
    nv->flags &= ~NV_FLAG_OPEN;
}

/*
 * Decreases nvl->usage_count, stopping the device when it reaches 0. Assumes
 * nvl->ldata_lock is held.
 */
static void nv_close_device(nv_state_t *nv, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (NV_ATOMIC_READ(nvl->usage_count) == 0)
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: Attempting to close unopened minor device %u!\n",
                  nvl->minor_num);
        WARN_ON(1);
        return;
    }

    if (NV_ATOMIC_DEC_AND_TEST(nvl->usage_count))
        nv_stop_device(nv, sp);
}

/*
** nvidia_close
**
** Master driver close entry point.
*/

int
nvidia_close(
    struct inode *inode,
    struct file *file
)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_FILEP(file);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    nvidia_stack_t *sp = nvfp->sp;
    unsigned int i;
    NvBool bRemove = NV_FALSE;

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, NV_MAY_SLEEP());

    /* for control device, just jump to its open routine */
    /* after setting up the private data */
    if (nv_is_control_device(inode))
        return nvidia_ctl_close(inode, file);

    nv_printf(NV_DBG_INFO, "NVRM: nvidia_close on device "
              "bearing minor number %d\n", NV_DEVICE_MINOR_NUMBER(inode));

    rm_free_unused_clients(sp, nv, nvfp);

    down(&nvl->ldata_lock);
    nv_close_device(nv, sp);
    bRemove = (NV_ATOMIC_READ(nvl->usage_count) == 0) &&
                rm_get_device_remove_flag(sp, nv->gpu_id);
    up(&nvl->ldata_lock);

    for (i = 0; i < NV_FOPS_STACK_INDEX_COUNT; ++i)
    {
        nv_kmem_cache_free_stack(nvfp->fops_sp[i]);
    }

    nv_free_file_private(nvfp);
    NV_SET_FILE_PRIVATE(file, NULL);

#if defined(NV_PCI_STOP_AND_REMOVE_BUS_DEVICE)
    if (bRemove)
    {
        NV_PCI_STOP_AND_REMOVE_BUS_DEVICE(nvl->dev);
    }
#endif

    nv_kmem_cache_free_stack(sp);

    return 0;
}

unsigned int
nvidia_poll(
    struct file *file,
    poll_table  *wait
)
{
    unsigned int mask = 0;
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    unsigned long eflags;

    //
    // File descriptors with an attached memory descriptor are intended
    // for use only via mmap(). poll() operations are forbidden on file
    // descritors to which memory has been attached or rm-object has
    // been exported.
    //
    if (nvfp->fd_memdesc.bValid || nvfp->hExportedRmObject != 0)
    {
        return POLLERR;
    }

    if ((file->f_flags & O_NONBLOCK) == 0)
        poll_wait(file, &nvfp->waitqueue, wait);

    NV_SPIN_LOCK_IRQSAVE(&nvfp->fp_lock, eflags);

    if ((nvfp->event_head != NULL) || nvfp->event_pending)
    {
        mask = (POLLPRI | POLLIN);
        nvfp->event_pending = FALSE;
    }

    NV_SPIN_UNLOCK_IRQRESTORE(&nvfp->fp_lock, eflags);

    return mask;
}

#define NV_CTL_DEVICE_ONLY(nv)                 \
{                                              \
    if (((nv)->flags & NV_FLAG_CONTROL) == 0)  \
    {                                          \
        status = -EINVAL;                      \
        goto done;                             \
    }                                          \
}

#define NV_ACTUAL_DEVICE_ONLY(nv)              \
{                                              \
    if (((nv)->flags & NV_FLAG_CONTROL) != 0)  \
    {                                          \
        status = -EINVAL;                      \
        goto done;                             \
    }                                          \
}

/*
 * Fills the ci array with the state of num_entries devices. Returns -EINVAL if
 * num_entries isn't big enough to hold all available devices.
 */
static int nvidia_read_card_info(nv_ioctl_card_info_t *ci, size_t num_entries)
{
    nv_state_t *nv;
    nv_linux_state_t *nvl;
    size_t i = 0;
    int rc = 0;

    /* Clear each card's flags field the lazy way */
    memset(ci, 0, num_entries * sizeof(ci[0]));

    LOCK_NV_LINUX_DEVICES();

    if (num_entries < num_nv_devices)
    {
        rc = -EINVAL;
        goto out;
    }

    for (nvl = nv_linux_devices; nvl && i < num_entries; nvl = nvl->next)
    {
        nv = NV_STATE_PTR(nvl);
        if (nv->pci_info.device_id)
        {
            ci[i].flags              = NV_IOCTL_CARD_INFO_FLAG_PRESENT;
            ci[i].pci_info.domain    = nv->pci_info.domain;
            ci[i].pci_info.bus       = nv->pci_info.bus;
            ci[i].pci_info.slot      = nv->pci_info.slot;
            ci[i].pci_info.vendor_id = nv->pci_info.vendor_id;
            ci[i].pci_info.device_id = nv->pci_info.device_id;
            ci[i].gpu_id             = nv->gpu_id;
            ci[i].interrupt_line     = nv->interrupt_line;
            ci[i].reg_address        = nv->regs->cpu_address;
            ci[i].reg_size           = nv->regs->size;
            ci[i].fb_address         = nv->fb->cpu_address;
            ci[i].fb_size            = nv->fb->size;
            ci[i].minor_number       = nvl->minor_num;
            i++;
        }
    }

out:
    UNLOCK_NV_LINUX_DEVICES();
    return rc;
}

int
nvidia_ioctl(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long i_arg)
{
    NV_STATUS rmStatus;
    int status = 0;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_FILEP(file);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    nvidia_stack_t *sp = NULL;
    nv_ioctl_xfer_t ioc_xfer;
    void *arg_ptr = (void *) i_arg;
    void *arg_copy = NULL;
    size_t arg_size;
    int arg_cmd;

    nv_printf(NV_DBG_INFO, "NVRM: ioctl(0x%x, 0x%x, 0x%x)\n",
        _IOC_NR(cmd), (unsigned int) i_arg, _IOC_SIZE(cmd));

    down(&nvfp->fops_sp_lock[NV_FOPS_STACK_INDEX_IOCTL]);
    sp = nvfp->fops_sp[NV_FOPS_STACK_INDEX_IOCTL];

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, NV_MAY_SLEEP());

    arg_size = _IOC_SIZE(cmd);
    arg_cmd  = _IOC_NR(cmd);

    //
    // File descriptors with an attached memory descriptor are intended
    // for use only via mmap(). ioctl() operations are forbidden on file
    // descritors to which memory has been attached or rm-object has
    // been exported.
    //
    if (nvfp->fd_memdesc.bValid || nvfp->hExportedRmObject != 0)
    {
        status = -EINVAL;
        goto done;
    }

    if (arg_cmd == NV_ESC_IOCTL_XFER_CMD)
    {
        if (arg_size != sizeof(nv_ioctl_xfer_t))
        {
            nv_printf(NV_DBG_ERRORS,
                    "NVRM: invalid ioctl XFER structure size!\n");
            status = -EINVAL;
            goto done;
        }

        if (NV_COPY_FROM_USER(&ioc_xfer, arg_ptr, sizeof(ioc_xfer)))
        {
            nv_printf(NV_DBG_ERRORS,
                    "NVRM: failed to copy in ioctl XFER data!\n");
            status = -EFAULT;
            goto done;
        }

        arg_cmd  = ioc_xfer.cmd;
        arg_size = ioc_xfer.size;
        arg_ptr  = NvP64_VALUE(ioc_xfer.ptr);

        if (arg_size > NV_ABSOLUTE_MAX_IOCTL_SIZE)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: invalid ioctl XFER size!\n");
            status = -EINVAL;
            goto done;
        }
    }

    NV_KMALLOC(arg_copy, arg_size);
    if (arg_copy == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate ioctl memory\n");
        status = -ENOMEM;
        goto done;
    }

    if (NV_COPY_FROM_USER(arg_copy, arg_ptr, arg_size))
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to copy in ioctl data!\n");
        status = -EFAULT;
        goto done;
    }

    switch (arg_cmd)
    {
        case NV_ESC_QUERY_DEVICE_INTR:
        {
            nv_ioctl_query_device_intr *query_intr = arg_copy;

            NV_ACTUAL_DEVICE_ONLY(nv);

            if (!nv->regs->map)
            {
                status = -EINVAL;
                goto done;
            }

            query_intr->intrStatus =
                *(nv->regs->map + (NV_RM_DEVICE_INTR_ADDRESS >> 2));
            query_intr->status = NV_OK;
            break;
        }

        /* pass out info about the card */
        case NV_ESC_CARD_INFO:
        {
            nv_ioctl_rm_api_old_version_t *rm_api;
            size_t num_arg_devices = arg_size / sizeof(nv_ioctl_card_info_t);
            NvU32 major, minor, patch;

            NV_CTL_DEVICE_ONLY(nv);

            /* the first element of card info passed from the client will have
             * the rm_api_version_magic value to show that the client is new
             * enough to support versioning. If the client is too old to
             * support versioning, our mmap interfaces are probably different
             * enough to cause serious damage.
             * just copy in the one dword to check.
             */
            if (arg_size < sizeof(rm_api->magic))
            {
                status = -EINVAL;
                goto done;
            }

            rm_api = arg_copy;
            switch (rm_api->magic)
            {
                case NV_RM_API_OLD_VERSION_MAGIC_REQ:
                case NV_RM_API_OLD_VERSION_MAGIC_LAX_REQ:
                case NV_RM_API_OLD_VERSION_MAGIC_OVERRIDE_REQ:
                    /* the client is using the old major-minor-patch
                     * API version check; reject it.
                     */
                    if (arg_size < sizeof(*rm_api))
                    {
                        major = 0;
                        minor = 0;
                        patch = 0;
                    }
                    else
                    {
                        major = rm_api->major;
                        minor = rm_api->minor;
                        patch = rm_api->patch;
                    }
                    nv_printf(NV_DBG_ERRORS,
                              "NVRM: API mismatch: the client has the version %d.%d-%d, but\n"
                              "NVRM: this kernel module has the version %s.  Please\n"
                              "NVRM: make sure that this kernel module and all NVIDIA driver\n"
                              "NVRM: components have the same version.\n",
                              major, minor, patch, NV_VERSION_STRING);
                    status = -EINVAL;
                    goto done;

                case NV_RM_API_OLD_VERSION_MAGIC_IGNORE:
                    /* the client is telling us to ignore the old
                     * version scheme; it will do a version check via
                     * NV_ESC_CHECK_VERSION_STR
                     */
                    break;
                default:
                    nv_printf(NV_DBG_ERRORS,
                        "NVRM: client does not support versioning!!\n");
                    status = -EINVAL;
                    goto done;
            }

            status = nvidia_read_card_info(arg_copy, num_arg_devices);
            break;
        }

        case NV_ESC_ATTACH_GPUS_TO_FD:
        {
            size_t num_arg_gpus = arg_size / sizeof(NvU32);
            size_t i;

            NV_CTL_DEVICE_ONLY(nv);

            if (num_arg_gpus == 0 || nvfp->num_attached_gpus != 0)
            {
                status = -EINVAL;
                goto done;
            }

            NV_KMALLOC(nvfp->attached_gpus, sizeof(NvU32) * num_arg_gpus);
            if (nvfp->attached_gpus == NULL)
            {
                status = -ENOMEM;
                goto done;
            }
            memcpy(nvfp->attached_gpus, arg_copy, sizeof(NvU32) * num_arg_gpus);
            nvfp->num_attached_gpus = num_arg_gpus;

            for (i = 0; i < nvfp->num_attached_gpus; i++)
            {
                if (nvfp->attached_gpus[i] == 0)
                {
                    continue;
                }

                nvidia_dev_get(nvfp->attached_gpus[i], sp);
            }

            break;
        }

        case NV_ESC_CHECK_VERSION_STR:
        {
            NV_CTL_DEVICE_ONLY(nv);

            rmStatus = rm_perform_version_check(sp, arg_copy, arg_size);
            status = ((rmStatus == NV_OK) ? 0 : -EINVAL);
            break;
        }

        case NV_ESC_SYS_PARAMS:
        {
            nv_ioctl_sys_params_t *api = arg_copy;

            NV_CTL_DEVICE_ONLY(nv);

            if (arg_size != sizeof(nv_ioctl_sys_params_t))
            {
                status = -EINVAL;
                goto done;
            }

            /* numa_memblock_size should only be set once */
            if (nvl->numa_memblock_size == 0)
            {
                nvl->numa_memblock_size = api->memblock_size;
            }
            else
            {
                status = (nvl->numa_memblock_size == api->memblock_size) ?
                    0 : -EBUSY;
                goto done;
            }
            break;
        }

        case NV_ESC_NUMA_INFO:
        {
            nv_ioctl_numa_info_t *api = arg_copy;

            NV_ACTUAL_DEVICE_ONLY(nv);

            if (arg_size != sizeof(nv_ioctl_numa_info_t))
            {
                status = -EINVAL;
                goto done;
            }

            rm_get_gpu_numa_info(sp, nv,
                &(api->nid),
                &(api->numa_mem_addr),
                &(api->numa_mem_size));

            api->status = nv_get_numa_status(nvl);
            api->memblock_size = nv_ctl_device.numa_memblock_size;
            break;
        }

        case NV_ESC_SET_NUMA_STATUS:
        {
            nv_ioctl_set_numa_status_t *api = arg_copy;
            rmStatus = NV_OK;

            if (!NV_IS_SUSER())
            {
                status = -EACCES;
                goto done;
            }

            NV_ACTUAL_DEVICE_ONLY(nv);

            if (arg_size != sizeof(nv_ioctl_set_numa_status_t))
            {
                status = -EINVAL;
                goto done;
            }

            if (nv_get_numa_status(nvl) != api->status)
            {
                if (api->status == NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS)
                {
                    /*
                     * If this call fails, it indicates that RM
                     * is not ready to offline memory, and we should keep
                     * the current NUMA status of ONLINE.
                     */
                    rmStatus = rm_gpu_numa_offline(sp, nv);
                }

                if (rmStatus == NV_OK)
                {
                    status = nv_set_numa_status(nvl, api->status);
                    if (status < 0)
                    {
                        goto done;
                    }
                    if (api->status == NV_IOCTL_NUMA_STATUS_ONLINE)
                    {
                        rmStatus = rm_gpu_numa_online(sp, nv);
                    }
                }

                status = (rmStatus == NV_OK) ?  status : -EBUSY;
            }

            break;
        }

        default:
            rmStatus = rm_ioctl(sp, nv, nvfp, arg_cmd, arg_copy, arg_size);
            status = ((rmStatus == NV_OK) ? 0 : -EINVAL);
            break;
    }

done:
    up(&nvfp->fops_sp_lock[NV_FOPS_STACK_INDEX_IOCTL]);

    if (arg_copy != NULL)
    {
        if (status != -EFAULT)
        {
            if (NV_COPY_TO_USER(arg_ptr, arg_copy, arg_size))
            {
                nv_printf(NV_DBG_ERRORS, "NVRM: failed to copy out ioctl data\n");
                status = -EFAULT;
            }
        }
        NV_KFREE(arg_copy, arg_size);
    }

    return status;
}

static long
nvidia_unlocked_ioctl(
    struct file *file,
    unsigned int cmd,
    unsigned long i_arg
)
{
    return nvidia_ioctl(NV_FILE_INODE(file), file, cmd, i_arg);
}

/*
 * driver receives an interrupt
 *    if someone waiting, then hand it off.
 */
static irqreturn_t
nvidia_isr(
    int   irq,
    void *arg
#if !defined(NV_IRQ_HANDLER_T_PRESENT) || (NV_IRQ_HANDLER_T_ARGUMENT_COUNT == 3)
    ,struct pt_regs *regs
#endif
)
{
    nv_linux_state_t *nvl = (void *) arg;
    nv_state_t *nv = NV_STATE_PTR(nvl);
    NvU32 need_to_run_bottom_half_gpu_lock_held = 0;
    BOOL rm_handled = FALSE,  uvm_handled = FALSE, rm_fault_handling_needed = FALSE;
    NvU32 rm_serviceable_fault_cnt = 0;

    rm_gpu_copy_mmu_faults_unlocked(nvl->sp[NV_DEV_STACK_ISR], nv, &rm_serviceable_fault_cnt);
    rm_fault_handling_needed = (rm_serviceable_fault_cnt != 0);

#if defined (NV_UVM_ENABLE)
    if (!NV_IS_GVI_DEVICE(nv))
    {
        //
        // Returns NV_OK if the UVM driver handled the interrupt
        //
        // Returns NV_ERR_NO_INTR_PENDING if the interrupt is not for
        // the UVM driver.
        //
        // Returns NV_WARN_MORE_PROCESSING_REQUIRED if the UVM top-half ISR was
        // unable to get its lock(s), due to other (UVM) threads holding them.
        //
        // RM can normally treat NV_WARN_MORE_PROCESSING_REQUIRED the same as
        // NV_ERR_NO_INTR_PENDING, but in some cases the extra information may
        // be helpful.
        //
        if (nv_uvm_event_interrupt(nv_state_get_gpu_uuid_cache(nv)) == NV_OK)
            uvm_handled = TRUE;
    }
#endif

    rm_handled = rm_isr(nvl->sp[NV_DEV_STACK_ISR], nv,
                        &need_to_run_bottom_half_gpu_lock_held);

    if (need_to_run_bottom_half_gpu_lock_held)
    {
        if (nv_use_threaded_interrupts)
            return IRQ_WAKE_THREAD;

        tasklet_schedule(&nvl->tasklet);
    }
    else
    {
        //
        // If rm_isr does not need to run a bottom half and mmu_faults_copied
        // indicates that bottom half is needed, then we enqueue a kthread based
        // bottom half rather than tasklet as this specific bottom_half will acquire
        // GPU lock
        //
        if (rm_fault_handling_needed)
            nv_kthread_q_schedule_q_item(&nvl->bottom_half_q, &nvl->bottom_half_q_item);
    }

    return IRQ_RETVAL(rm_handled || uvm_handled || rm_fault_handling_needed);
}

static irqreturn_t
nvidia_isr_kthread_bh(
    int irq,
    void *data
)
{
    return nvidia_isr_common_bh(data);
}

static void
nvidia_isr_tasklet_bh(
    unsigned long data
)
{
    nvidia_isr_common_bh((void*)data);
}

static irqreturn_t
nvidia_isr_common_bh(
    void *data
)
{
    nv_state_t *nv = (nv_state_t *) data;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nvidia_stack_t *sp = nvl->sp[NV_DEV_STACK_ISR_BH];

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, FALSE, FALSE);
    rm_isr_bh(sp, nv);
    return IRQ_HANDLED;
}

static void
nvidia_isr_bh_unlocked(
    void * args
)
{
    nv_state_t *nv = (nv_state_t *) args;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nvidia_stack_t *sp;
    NV_STATUS status;

    //
    // Synchronize kthreads servicing unlocked bottom half as they
    // share same pre-allocated stack for alt-stack
    //
    status = os_acquire_mutex(nvl->isr_bh_unlocked_mutex);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s: Unable to take bottom_half mutex!\n",
                  __FUNCTION__);
        WARN_ON(1);
    }

    sp = nvl->sp[NV_DEV_STACK_ISR_BH_UNLOCKED];

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, FALSE, FALSE);
    rm_isr_bh_unlocked(sp, nv);
    os_release_mutex(nvl->isr_bh_unlocked_mutex);
}

static void
nvidia_rc_timer_callback(
    struct nv_timer *nv_timer
)
{
    nv_linux_state_t *nvl = container_of(nv_timer, nv_linux_state_t, rc_timer);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nvidia_stack_t *sp = nvl->sp[NV_DEV_STACK_TIMER];

    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, FALSE);

    if (rm_run_rc_callback(sp, nv) == NV_OK)
    {
        // set another timeout 1 sec in the future:
        mod_timer(&nvl->rc_timer.kernel_timer, jiffies + HZ);
    }
}

/*
** nvidia_ctl_open
**
** nv control driver open entry point.  Sessions are created here.
*/
static int
nvidia_ctl_open(
    struct inode *inode,
    struct file *file
)
{
    nv_linux_state_t *nvl = &nv_ctl_device;
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    static int count = 0;

    nv_printf(NV_DBG_INFO, "NVRM: nvidia_ctl_open\n");

    down(&nvl->ldata_lock);

    /* save the nv away in file->private_data */
    nvfp->nvptr = nvl;

    if (NV_ATOMIC_READ(nvl->usage_count) == 0)
    {
        init_waitqueue_head(&nv_ctl_waitqueue);

        nv->flags |= (NV_FLAG_OPEN | NV_FLAG_CONTROL);

        if ((nv_acpi_init() < 0) &&
            (count++ < NV_MAX_RECURRING_WARNING_MESSAGES))
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: failed to register with the ACPI subsystem!\n");
        }
    }

    NV_ATOMIC_INC(nvl->usage_count);
    up(&nvl->ldata_lock);

    return 0;
}


/*
** nvidia_ctl_close
*/
static int
nvidia_ctl_close(
    struct inode *inode,
    struct file *file
)
{
    nv_alloc_t *at, *next;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_FILEP(file);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    nv_file_private_t *nvfp = NV_GET_FILE_PRIVATE(file);
    nvidia_stack_t *sp = nvfp->sp;
    static int count = 0;
    unsigned int i;

    nv_printf(NV_DBG_INFO, "NVRM: nvidia_ctl_close\n");

    //
    // If a rm-object exported into this file descriptor the get it free.
    //
    if (nvfp->hExportedRmObject != 0)
    {
        rm_free_exported_object(sp, nvfp->hExportedRmObject);
    }

    down(&nvl->ldata_lock);
    if (NV_ATOMIC_DEC_AND_TEST(nvl->usage_count))
    {
        nv->flags &= ~NV_FLAG_OPEN;

        if ((nv_acpi_uninit() < 0) &&
            (count++ < NV_MAX_RECURRING_WARNING_MESSAGES))
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: failed to unregister from the ACPI subsystem!\n");
        }
    }
    up(&nvl->ldata_lock);

    rm_free_unused_clients(sp, nv, nvfp);

    if (nvfp->free_list != NULL)
    {
        at = nvfp->free_list;
        while (at != NULL)
        {
            next = at->next;
            if (at->pid == os_get_current_process())
                NV_PRINT_AT(NV_DBG_MEMINFO, at);
            nv_free_pages(nv, at->num_pages,
                          NV_ALLOC_MAPPING_CONTIG(at->flags),
                          NV_ALLOC_MAPPING(at->flags),
                          (void *)at);
            at = next;
        }
    }

    if (nvfp->num_attached_gpus != 0)
    {
        size_t i;

        for (i = 0; i < nvfp->num_attached_gpus; i++)
        {
            if (nvfp->attached_gpus[i] != 0)
                nvidia_dev_put(nvfp->attached_gpus[i], sp);
        }

        NV_KFREE(nvfp->attached_gpus, sizeof(NvU32) * nvfp->num_attached_gpus);
        nvfp->num_attached_gpus = 0;
    }

    for (i = 0; i < NV_FOPS_STACK_INDEX_COUNT; ++i)
    {
        nv_kmem_cache_free_stack(nvfp->fops_sp[i]);
    }

    nv_free_file_private(nvfp);
    NV_SET_FILE_PRIVATE(file, NULL);

    nv_kmem_cache_free_stack(sp);

    return 0;
}


void NV_API_CALL
nv_set_dma_address_size(
    nv_state_t  *nv,
    NvU32       phys_addr_bits
)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    /*
     * The only scenario in which we definitely should not update the DMA mask
     * is on POWER, when using TCE bypass mode (see nv_get_dma_start_address()
     * for details), since the meaning of the DMA mask is overloaded in that
     * case.
     */
    if (!nvl->tce_bypass_enabled)
    {
        NvU64 new_mask = (((NvU64)1) << phys_addr_bits) - 1;
        pci_set_dma_mask(nvl->dev, new_mask);
    }
}

static NvUPtr
nv_map_guest_pages(nv_alloc_t *at,
                   NvU64 address,
                   NvU32 page_count,
                   NvU32 page_idx)
{
    struct page **pages;
    NvU32 j;
    NvUPtr virt_addr;

    NV_KMALLOC(pages, sizeof(struct page *) * page_count);
    if (pages == NULL)
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: failed to allocate vmap() page descriptor table!\n");
        return 0;
    }

    for (j = 0; j < page_count; j++)
    {
        pages[j] = NV_GET_PAGE_STRUCT(at->page_table[page_idx+j]->phys_addr);
    }

    virt_addr = nv_vm_map_pages(pages, page_count,
        NV_ALLOC_MAPPING_CACHED(at->flags));
    NV_KFREE(pages, sizeof(struct page *) * page_count);

    return virt_addr;
}

NV_STATUS NV_API_CALL
nv_alias_pages(
    nv_state_t *nv,
    NvU32 page_cnt,
    NvU32 contiguous,
    NvU32 cache_type,
    NvU64 guest_id,
    NvU64 *pte_array,
    void **priv_data
)
{
    nv_alloc_t *at;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvU32 i=0;
    nvidia_pte_t *page_ptr = NULL;

    at = nvos_create_alloc(nvl->dev, page_cnt);

    if (at == NULL)
    {
        return NV_ERR_NO_MEMORY;
    }

    at->flags = nv_alloc_init_flags(cache_type, contiguous, 0);
    at->flags |= NV_ALLOC_TYPE_GUEST;

    at->order = get_order(at->num_pages * PAGE_SIZE);

    for (i=0; i < at->num_pages; ++i)
    {
        page_ptr = at->page_table[i];

        if (contiguous && i>0)
        {
            page_ptr->dma_addr = pte_array[0] + (i << PAGE_SHIFT);
        }
        else
        {
            page_ptr->dma_addr  = pte_array[i];
        }

        page_ptr->phys_addr = page_ptr->dma_addr;

        /* aliased pages will be mapped on demand. */
        page_ptr->virt_addr = 0x0;
    }

    at->guest_id = guest_id;
    *priv_data = at;
    NV_ATOMIC_INC(at->usage_count);

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    return NV_OK;
}

/*
 *   This creates a dummy nv_alloc_t for peer IO mem, so that it can
 *   be mapped using NvRmMapMemory.
 */
NV_STATUS NV_API_CALL nv_register_peer_io_mem(
    nv_state_t *nv,
    NvU64      *phys_addr,
    NvU64       page_count,
    void      **priv_data
)
{
    nv_alloc_t *at;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvU64 i;
    NvU64 addr;

    at = nvos_create_alloc(nvl->dev, page_count);

    if (at == NULL)
        return NV_ERR_NO_MEMORY;

    // IO regions should be uncached and contiguous
    at->flags = nv_alloc_init_flags(0, 1, 0);
    at->flags |= NV_ALLOC_TYPE_PEER_IO;

    at->order = get_order(at->num_pages * PAGE_SIZE);

    addr = phys_addr[0];

    for (i = 0; i < page_count; i++)
    {
        at->page_table[i]->phys_addr = addr;
        addr += PAGE_SIZE;
    }

    // No struct page array exists for this memory.
    at->user_pages = NULL;

    *priv_data = at;

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    return NV_OK;
}

void NV_API_CALL nv_unregister_peer_io_mem(
    nv_state_t *nv,
    void       *priv_data
)
{
    nv_alloc_t *at = priv_data;

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    nvos_free_alloc(at);
}

/*
 * By registering user pages, we create a dummy nv_alloc_t for it, so that the
 * rest of the RM can treat it like any other alloc.
 *
 * This also converts the page array to an array of physical addresses.
 */
NV_STATUS NV_API_CALL nv_register_user_pages(
    nv_state_t *nv,
    NvU64       page_count,
    NvU64      *phys_addr,
    void      **priv_data
)
{
    nv_alloc_t *at;
    NvU64 i;
    struct page **user_pages = *priv_data;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nvidia_pte_t *page_ptr;

    nv_printf(NV_DBG_MEMINFO, "NVRM: VM: nv_register_user_pages: 0x%x\n", page_count);

    at = nvos_create_alloc(nvl->dev, page_count);

    if (at == NULL)
    {
        return NV_ERR_NO_MEMORY;
    }

    /*
     * Anonymous memory currently must be write-back cacheable, and we can't
     * enforce contiguity.
     */
    at->flags = nv_alloc_init_flags(1, 0, 0);
    at->flags |= NV_ALLOC_TYPE_USER;

    at->order = get_order(at->num_pages * PAGE_SIZE);

    for (i = 0; i < page_count; i++)
    {
        /*
         * We only assign the physical address and not the DMA address, since
         * this allocation hasn't been DMA-mapped yet.
         */
        page_ptr = at->page_table[i];
        page_ptr->phys_addr = page_to_phys(user_pages[i]);

        phys_addr[i] = page_ptr->phys_addr;
    }

    /* Save off the user pages array to be restored later */
    at->user_pages = user_pages;
    *priv_data = at;

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    return NV_OK;
}

NV_STATUS NV_API_CALL nv_unregister_user_pages(
    nv_state_t *nv,
    NvU64       page_count,
    void      **priv_data
)
{
    nv_alloc_t *at = *priv_data;

    nv_printf(NV_DBG_MEMINFO, "NVRM: VM: nv_unregister_user_pages: 0x%x\n", page_count);

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    WARN_ON(!NV_ALLOC_MAPPING_USER(at->flags));

    /* Restore the user pages array for the caller to handle */
    *priv_data = at->user_pages;

    nvos_free_alloc(at);

    return NV_OK;
}

void* NV_API_CALL nv_alloc_kernel_mapping(
    nv_state_t *nv,
    void       *pAllocPrivate,
    NvU64       pageIndex,
    NvU32       pageOffset,
    NvU64       size,
    void      **pPrivate
)
{
    nv_alloc_t *at = pAllocPrivate;
    NvU32 j, page_count;
    NvUPtr virt_addr;
    struct page **pages;
    NvBool isUserAllocatedMem;

    //
    // For User allocated memory (like ErrorNotifier's) which is NOT allocated
    // nor owned by RM, the RM driver just stores the physical address
    // corresponding to that memory and does not map it until required.
    // In that case, in page tables the virt_addr == 0, so first we need to map
    // those pages to obtain virtual address.
    //
    isUserAllocatedMem = NV_ALLOC_MAPPING_USER(at->flags) &&
                        !at->page_table[pageIndex]->virt_addr &&
                         at->page_table[pageIndex]->phys_addr;

    //
    // User memory may NOT have kernel VA. So check this and fallback to else
    // case to create one.
    //
    if (((size + pageOffset) <= PAGE_SIZE) &&
         !NV_ALLOC_MAPPING_GUEST(at->flags) && !NV_ALLOC_MAPPING_ALIASED(at->flags) &&
         !isUserAllocatedMem)
    {
        *pPrivate = NULL;
        return (void *)(at->page_table[pageIndex]->virt_addr + pageOffset);
    }
    else
    {
        size += pageOffset;
        page_count = (size >> PAGE_SHIFT) + ((size & ~NV_PAGE_MASK) ? 1 : 0);

        if (NV_ALLOC_MAPPING_GUEST(at->flags))
        {
            virt_addr = nv_map_guest_pages(at,
                                           nv->bars[NV_GPU_BAR_INDEX_REGS].cpu_address,
                                           page_count, pageIndex);
        }
        else
        {
            NV_KMALLOC(pages, sizeof(struct page *) * page_count);
            if (pages == NULL)
            {
                nv_printf(NV_DBG_ERRORS,
                          "NVRM: failed to allocate vmap() page descriptor table!\n");
                return NULL;
            }

            for (j = 0; j < page_count; j++)
              pages[j] = NV_GET_PAGE_STRUCT(at->page_table[pageIndex+j]->phys_addr);

            virt_addr = nv_vm_map_pages(pages, page_count,
                NV_ALLOC_MAPPING_CACHED(at->flags));
            NV_KFREE(pages, sizeof(struct page *) * page_count);
        }

        if (virt_addr == 0)
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: failed to map pages!\n");
            return NULL;
        }

        *pPrivate = (void *)(NvUPtr)page_count;
        return (void *)(virt_addr + pageOffset);
    }

    return NULL;
}

NV_STATUS NV_API_CALL nv_free_kernel_mapping(
    nv_state_t *nv,
    void       *pAllocPrivate,
    void       *address,
    void       *pPrivate
)
{
    nv_alloc_t *at = pAllocPrivate;
    NvUPtr virt_addr;
    NvU32 page_count;

    virt_addr = ((NvUPtr)address & NV_PAGE_MASK);
    page_count = (NvUPtr)pPrivate;

    if (NV_ALLOC_MAPPING_GUEST(at->flags))
    {
        nv_iounmap((void *)virt_addr, (page_count * PAGE_SIZE));
    }
    else if (pPrivate != NULL)
    {
        nv_vm_unmap_pages(virt_addr, page_count);
    }

    return NV_OK;
}

NV_STATUS NV_API_CALL nv_alloc_pages(
    nv_state_t *nv,
    NvU32       page_count,
    NvBool      contiguous,
    NvU32       cache_type,
    NvBool      zeroed,
    NvU64      *pte_array,
    void      **priv_data
)
{
    nv_alloc_t *at;
    NV_STATUS status = NV_ERR_NO_MEMORY;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvBool will_remap = nv_requires_dma_remap(nv);
    NvU32 i, memory_type;

    nv_printf(NV_DBG_MEMINFO, "NVRM: VM: nv_alloc_pages: %d pages\n", page_count);
    nv_printf(NV_DBG_MEMINFO, "NVRM: VM:    contig %d  cache_type %d\n",
        contiguous, cache_type);

    memory_type = NV_MEMORY_TYPE_SYSTEM;

    if (nv_encode_caching(NULL, cache_type, memory_type))
        return NV_ERR_NOT_SUPPORTED;

    at = nvos_create_alloc(nvl->dev, page_count);
    if (at == NULL)
        return NV_ERR_NO_MEMORY;

    at->flags = nv_alloc_init_flags(cache_type, contiguous, zeroed);

#if defined(NVCPU_PPC64LE)
    /*
     * Starting on Power9 systems, DMA addresses for NVLink are no longer the
     * same as used over PCIe. There is an address compression scheme required
     * for NVLink ONLY which impacts the upper address bits of the DMA address.
     *
     * This divergence between PCIe and NVLink DMA mappings breaks assumptions
     * in the driver where during initialization we allocate system memory
     * for the GPU to access over PCIe before NVLink is trained -- and some of
     * these mappings persist on the GPU. If these persistent mappings are not
     * equivalent they will cause invalid DMA accesses from the GPU once we
     * switch to NVLink.
     *
     * To work around this we limit all system memory allocations from the driver
     * during the period before NVLink is enabled to be from NUMA node 0 (CPU 0)
     * which has a CPU real address with the upper address bits (above bit 42)
     * set to 0. Effectively making the PCIe and NVLink DMA mappings equivalent
     * allowing persistent system memory mappings already programmed on the GPU
     * to remain valid after NVLink is enabled.
     *
     * See Bug 1920398 for more details.
     */
    if (nvl->npu && !(nv->nvlink_sysmem_links_enabled))
        at->flags |= NV_ALLOC_TYPE_NODE0;
#endif

    if (NV_ALLOC_MAPPING_CONTIG(at->flags))
        status = nv_alloc_contig_pages(nv, at);
    else
        status = nv_alloc_system_pages(nv, at);

    if (status != NV_OK)
        goto failed;

    for (i = 0; i < ((contiguous) ? 1 : page_count); i++)
    {
        /*
         * The contents of the pte_array[] depend on whether or not this device
         * requires DMA-remapping. If it does, it should be the phys addresses
         * used by the DMA-remapping paths, otherwise it should be the actual
         * address that the device should use for DMA (which, confusingly, may
         * be different than the CPU physical address, due to a static DMA
         * offset).
         */
        if (will_remap)
        {
            pte_array[i] = at->page_table[i]->phys_addr;
        }
        else
        {
            pte_array[i] = nv_phys_to_dma(nvl->dev,
                at->page_table[i]->phys_addr);
        }
    }

    *priv_data = at;
    NV_ATOMIC_INC(at->usage_count);

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    return NV_OK;

failed:
    nvos_free_alloc(at);

    return status;
}

NV_STATUS NV_API_CALL nv_free_pages(
    nv_state_t *nv,
    NvU32 page_count,
    NvBool contiguous,
    NvU32 cache_type,
    void *priv_data
)
{
    NV_STATUS rmStatus = NV_OK;
    nv_alloc_t *at = priv_data;

    nv_printf(NV_DBG_MEMINFO, "NVRM: VM: nv_free_pages: 0x%x\n", page_count);

    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    /*
     * If the 'at' usage count doesn't drop to zero here, not all of
     * the user mappings have been torn down in time - we can't
     * safely free the memory. We report success back to the RM, but
     * defer the actual free operation until later.
     *
     * This is described in greater detail in the comments above the
     * nvidia_vma_(open|release)() callbacks in nv-mmap.c.
     */
    if (!NV_ATOMIC_DEC_AND_TEST(at->usage_count))
        return NV_OK;

    if (!NV_ALLOC_MAPPING_GUEST(at->flags))
    {
        if (NV_ALLOC_MAPPING_CONTIG(at->flags))
            nv_free_contig_pages(at);
        else
            nv_free_system_pages(at);
    }

    nvos_free_alloc(at);

    return rmStatus;
}

static NvBool nv_lock_init_locks
(
    nvidia_stack_t *sp,
    nv_state_t *nv
)
{
    nv_linux_state_t *nvl;
    nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    NV_INIT_MUTEX(&nvl->ldata_lock);

    NV_ATOMIC_SET(nvl->usage_count, 0);

    if (!rm_init_event_locks(sp, nv))
        return NV_FALSE;

    return NV_TRUE;
}

static void nv_lock_destroy_locks
(
    nvidia_stack_t *sp,
    nv_state_t *nv
)
{
    rm_destroy_event_locks(sp, nv);
}

void NV_API_CALL nv_post_event(
    nv_state_t *nv,
    nv_event_t *event,
    NvHandle    handle,
    NvU32       index,
    NvBool      data_valid
)
{
    nv_file_private_t *nvfp = event->file;
    unsigned long eflags;
    nvidia_event_t *nvet;

    NV_SPIN_LOCK_IRQSAVE(&nvfp->fp_lock, eflags);

    if (data_valid)
    {
        NV_KMALLOC_ATOMIC(nvet, sizeof(nvidia_event_t));
        if (nvet == NULL)
        {
            NV_SPIN_UNLOCK_IRQRESTORE(&nvfp->fp_lock, eflags);
            return;
        }

        if (nvfp->event_tail != NULL)
            nvfp->event_tail->next = nvet;
        if (nvfp->event_head == NULL)
            nvfp->event_head = nvet;
        nvfp->event_tail = nvet;
        nvet->next = NULL;

        nvet->event = *event;
        nvet->event.hObject = handle;
        nvet->event.index = index;
    }

    nvfp->event_pending = TRUE;

    NV_SPIN_UNLOCK_IRQRESTORE(&nvfp->fp_lock, eflags);

    wake_up_interruptible(&nvfp->waitqueue);
}

nv_fd_memdesc_t* NV_API_CALL nv_get_fd_memdesc(
    void       *file
)
{
    nv_file_private_t *nvfp = file;

    if (!file)
        return NULL;

    return &(nvfp->fd_memdesc);
}

NV_STATUS NV_API_CALL nv_add_fd_memdesc_to_fd(
    NvU32                  fd,
    const nv_fd_memdesc_t *pFdMemdesc
)
{
    struct file *filp = NULL;
    nv_file_private_t *nvfp = NULL;
    dev_t rdev = 0;
    NV_STATUS status = NV_ERR_INVALID_ARGUMENT;

    filp = fget(fd);

    if (filp == NULL)
    {
        return status;
    }
    if (NV_FILE_INODE(filp))
    {
        rdev = (NV_FILE_INODE(filp))->i_rdev;
    }
    else
    {
        goto done;
    }
    // Only allow adding the fd memdesc if the struct file is an open of
    // the NVIDIA control device file. Note it is not safe to interpret
    // the file private data as an nv_file_private_t until it passes this check.
    if ((MAJOR(rdev) != NV_MAJOR_DEVICE_NUMBER) ||
        (MINOR(rdev) != NV_CONTROL_DEVICE_MINOR))
    {
        goto done;
    }

    nvfp = NV_GET_FILE_PRIVATE(filp);
    if (!nvfp->fd_memdesc.bValid)
    {
        nvfp->fd_memdesc = *pFdMemdesc;
        status = NV_OK;
    }
    else
    {
        status = NV_ERR_STATE_IN_USE;
    }

done:

    fput(filp);

    return status;
}

NV_STATUS NV_API_CALL nv_export_rm_object_to_fd(
    NvHandle  hExportedRmObject,
    NvS32     fd
)
{
    struct file *filp = NULL;
    nv_file_private_t *nvfp = NULL;
    dev_t rdev = 0;
    NV_STATUS status = NV_OK;

    filp = fget(fd);

    if (filp == NULL || !NV_FILE_INODE(filp))
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    rdev = (NV_FILE_INODE(filp))->i_rdev;

    if ((MAJOR(rdev) != NV_MAJOR_DEVICE_NUMBER) ||
        (MINOR(rdev) != NV_CONTROL_DEVICE_MINOR))
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    nvfp = NV_GET_FILE_PRIVATE(filp);

    if (nvfp->hExportedRmObject != 0)
    {
        status = NV_ERR_STATE_IN_USE;
        goto done;
    }

    nvfp->hExportedRmObject = hExportedRmObject;

done:

    if (filp != NULL)
    {
        fput(filp);
    }

    return status;
}

NV_STATUS NV_API_CALL nv_import_rm_object_from_fd(
    NvS32     fd,
    NvHandle *pExportedObject
)
{
    struct file *filp = NULL;
    nv_file_private_t *nvfp = NULL;
    dev_t rdev = 0;
    NV_STATUS status = NV_OK;

    filp = fget(fd);

    if (filp == NULL || !NV_FILE_INODE(filp))
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    rdev = (NV_FILE_INODE(filp))->i_rdev;

    if ((MAJOR(rdev) != NV_MAJOR_DEVICE_NUMBER) ||
        (MINOR(rdev) != NV_CONTROL_DEVICE_MINOR))
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    nvfp = NV_GET_FILE_PRIVATE(filp);

    if (nvfp->hExportedRmObject == 0)
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done;
    }

    *pExportedObject = nvfp->hExportedRmObject;

done:

    if (filp != NULL)
    {
        fput(filp);
    }

    return status;
}

int NV_API_CALL nv_get_event(
    nv_state_t *nv,
    void       *file,
    nv_event_t *event,
    NvU32      *pending
)
{
    nv_file_private_t *nvfp = file;
    nvidia_event_t *nvet;
    unsigned long eflags;

    NV_SPIN_LOCK_IRQSAVE(&nvfp->fp_lock, eflags);

    nvet = nvfp->event_head;
    if (nvet == NULL)
    {
        NV_SPIN_UNLOCK_IRQRESTORE(&nvfp->fp_lock, eflags);
        return NV_ERR_GENERIC;
    }

    *event = nvet->event;

    if (nvfp->event_tail == nvet)
        nvfp->event_tail = NULL;
    nvfp->event_head = nvet->next;

    *pending = (nvfp->event_head != NULL);

    NV_SPIN_UNLOCK_IRQRESTORE(&nvfp->fp_lock, eflags);

    NV_KFREE(nvet, sizeof(nvidia_event_t));

    return NV_OK;
}

int NV_API_CALL nv_start_rc_timer(
    nv_state_t *nv
)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (nv->rc_timer_enabled)
        return -1;

    nv_printf(NV_DBG_INFO, "NVRM: initializing rc timer\n");

    nv_timer_setup(&nvl->rc_timer, nvidia_rc_timer_callback);

    nv->rc_timer_enabled = 1;

    // set the timeout for 1 second in the future:
    mod_timer(&nvl->rc_timer.kernel_timer, jiffies + HZ);

    nv_printf(NV_DBG_INFO, "NVRM: rc timer initialized\n");

    return 0;
}

int NV_API_CALL nv_stop_rc_timer(
    nv_state_t *nv
)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (!nv->rc_timer_enabled)
        return -1;

    nv_printf(NV_DBG_INFO, "NVRM: stopping rc timer\n");
    nv->rc_timer_enabled = 0;
    del_timer_sync(&nvl->rc_timer.kernel_timer);
    nv_printf(NV_DBG_INFO, "NVRM: rc timer stopped\n");

    return 0;
}

static void
nvos_validate_assigned_gpus(struct pci_dev *dev)
{
    NvU32 i;

    if (NV_IS_ASSIGN_GPU_PCI_INFO_SPECIFIED())
    {
        for (i = 0; i < nv_assign_gpu_count; i++)
        {
            if ((nv_assign_gpu_pci_info[i].domain == NV_PCI_DOMAIN_NUMBER(dev)) &&
                (nv_assign_gpu_pci_info[i].bus == NV_PCI_BUS_NUMBER(dev)) &&
                (nv_assign_gpu_pci_info[i].slot == NV_PCI_SLOT_NUMBER(dev)))
            {
                nv_assign_gpu_pci_info[i].valid = NV_TRUE;
                return;
            }
        }
    }
}

/* make sure the pci_driver called probe for all of our devices.
 * we've seen cases where rivafb claims the device first and our driver
 * doesn't get called.
 */
static int
nvos_count_devices(nvidia_stack_t *sp)
{
    struct pci_dev *dev;
    int count = 0;

    dev = NV_PCI_GET_CLASS(PCI_CLASS_DISPLAY_VGA << 8, NULL);
    while (dev)
    {
        if ((dev->vendor == 0x10de) && (dev->device >= 0x20) &&
            !rm_is_legacy_device(sp, dev->device, dev->subsystem_vendor,
                                 dev->subsystem_device, TRUE))
        {
            count++;
            nvos_validate_assigned_gpus(dev);
        }
        dev = NV_PCI_GET_CLASS(PCI_CLASS_DISPLAY_VGA << 8, dev);
    }

    dev = NV_PCI_GET_CLASS(PCI_CLASS_DISPLAY_3D << 8, NULL);
    while (dev)
    {
        if ((dev->vendor == 0x10de) && (dev->device >= 0x20) &&
            !rm_is_legacy_device(sp, dev->device, dev->subsystem_vendor,
                                 dev->subsystem_device, TRUE))
        {
            count++;
            nvos_validate_assigned_gpus(dev);
        }
        dev = NV_PCI_GET_CLASS(PCI_CLASS_DISPLAY_3D << 8, dev);
    }

    dev = NV_PCI_GET_CLASS(PCI_CLASS_MULTIMEDIA_OTHER << 8, NULL);
    while (dev)
    {
        if ((dev->vendor == 0x10de) && (dev->device == 0x0e00))
            count++;
        dev = NV_PCI_GET_CLASS(PCI_CLASS_MULTIMEDIA_OTHER << 8, dev);
    }

    if (NV_IS_ASSIGN_GPU_PCI_INFO_SPECIFIED())
    {
        NvU32 i;

        for (i = 0; i < nv_assign_gpu_count; i++)
        {
            if (nv_assign_gpu_pci_info[i].valid == NV_TRUE)
                count++;
        }
    }

    return count;
}

NvBool nvos_is_chipset_io_coherent(void)
{
    if (nv_chipset_is_io_coherent == NV_TRISTATE_INDETERMINATE)
    {
        nvidia_stack_t *sp = NULL;
        if (nv_kmem_cache_alloc_stack(&sp) != 0)
        {
            nv_printf(NV_DBG_ERRORS,
              "NVRM: cannot allocate stack for platform coherence check callback \n");
            WARN_ON(1);
            return NV_FALSE;
        }

        nv_chipset_is_io_coherent = rm_is_chipset_io_coherent(sp);

        nv_kmem_cache_free_stack(sp);
    }

    return nv_chipset_is_io_coherent;
}

static BOOL nv_treat_missing_irq_as_error(void)
{
#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
    return (nv_get_hypervisor_type() != OS_HYPERVISOR_HYPERV);
#else
    return TRUE;
#endif
}

/* find nvidia devices and set initial state */
int
nvidia_probe
(
    struct pci_dev *dev,
    const struct pci_device_id *id_table
)
{
    nv_state_t *nv = NULL;
    nv_linux_state_t *nvl = NULL;
    unsigned int i, j;
    int flags = 0;
    nvidia_stack_t *sp = NULL;

    nv_printf(NV_DBG_SETUP, "NVRM: probing 0x%x 0x%x, class 0x%x\n",
        dev->vendor, dev->device, dev->class);

    if ((dev->class == (PCI_CLASS_MULTIMEDIA_OTHER << 8)) &&
        (dev->device == 0x0e00))
    {
        flags = NV_FLAG_GVI;
    }

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return -1;
    }

    if (!(flags & NV_FLAG_GVI))
    {
        if ((dev->vendor != 0x10de) || (dev->device < 0x20) ||
            ((dev->class != (PCI_CLASS_DISPLAY_VGA << 8)) &&
             (dev->class != (PCI_CLASS_DISPLAY_3D << 8))) ||
            rm_is_legacy_device(sp, dev->device, dev->subsystem_vendor,
                                dev->subsystem_device, FALSE))
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: ignoring the legacy GPU %04x:%02x:%02x.%x\n",
                      NV_PCI_DOMAIN_NUMBER(dev), NV_PCI_BUS_NUMBER(dev), NV_PCI_SLOT_NUMBER(dev),
                      PCI_FUNC(dev->devfn));
            goto failed;
        }
    }

    if (NV_IS_ASSIGN_GPU_PCI_INFO_SPECIFIED())
    {
        for (i = 0; i < nv_assign_gpu_count; i++)
        {
            if (((nv_assign_gpu_pci_info[i].domain == NV_PCI_DOMAIN_NUMBER(dev)) &&
                 (nv_assign_gpu_pci_info[i].bus == NV_PCI_BUS_NUMBER(dev)) &&
                 (nv_assign_gpu_pci_info[i].slot == NV_PCI_SLOT_NUMBER(dev))) &&
                (nv_assign_gpu_pci_info[i].valid))
                break;
        }

        if (i == nv_assign_gpu_count)
        {
            goto failed;
        }
    }

    num_probed_nv_devices++;

    if (pci_enable_device(dev) != 0)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: pci_enable_device failed, aborting\n");
        goto failed;
    }

    if ((dev->irq == 0) && nv_treat_missing_irq_as_error())
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: Can't find an IRQ for your NVIDIA card!\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: Please check your BIOS settings.\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: [Plug & Play OS] should be set to NO\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: [Assign IRQ to VGA] should be set to YES \n");
        goto failed;
    }

    for (i = 0; i < (NV_GPU_NUM_BARS - 1); i++)
    {
        if (NV_PCI_RESOURCE_VALID(dev, i))
        {
#if defined(NV_PCI_MAX_MMIO_BITS_SUPPORTED)
            if ((NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64) &&
                ((NV_PCI_RESOURCE_START(dev, i) >> NV_PCI_MAX_MMIO_BITS_SUPPORTED)))
            {
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: This is a 64-bit BAR mapped above %dGB by the system\n"
                    "NVRM: BIOS or the %s kernel. This PCI I/O region assigned\n"
                    "NVRM: to your NVIDIA device is not supported by the kernel.\n"
                    "NVRM: BAR%d is %dM @ 0x%llx (PCI:%04x:%02x:%02x.%x)\n",
                    (1 << (NV_PCI_MAX_MMIO_BITS_SUPPORTED - 30)),
                    NV_KERNEL_NAME, i,
                    (NV_PCI_RESOURCE_SIZE(dev, i) >> 20),
                    (NvU64)NV_PCI_RESOURCE_START(dev, i),
                    NV_PCI_DOMAIN_NUMBER(dev),
                    NV_PCI_BUS_NUMBER(dev), NV_PCI_SLOT_NUMBER(dev),
                    PCI_FUNC(dev->devfn));
                goto failed;
            }
#endif
            if ((NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64) &&
                (NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_PREFETCH))
            {
                struct pci_dev *bridge = dev->bus->self;
                NvU32 base_upper, limit_upper;

                if (bridge == NULL)
                    continue;

                pci_read_config_dword(dev, NVRM_PCICFG_BAR_OFFSET(i) + 4,
                        &base_upper);
                if (base_upper == 0)
                    continue;

                pci_read_config_dword(bridge, PCI_PREF_BASE_UPPER32,
                        &base_upper);
                pci_read_config_dword(bridge, PCI_PREF_LIMIT_UPPER32,
                        &limit_upper);

                if ((base_upper != 0) && (limit_upper != 0))
                    continue;

                nv_printf(NV_DBG_ERRORS,
                    "NVRM: This is a 64-bit BAR mapped above 4GB by the system\n"
                    "NVRM: BIOS or the %s kernel, but the PCI bridge\n"
                    "NVRM: immediately upstream of this GPU does not define\n"
                    "NVRM: a matching prefetchable memory window.\n",
                    NV_KERNEL_NAME);
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: This may be due to a known Linux kernel bug.  Please\n"
                    "NVRM: see the README section on 64-bit BARs for additional\n"
                    "NVRM: information.\n");
                goto failed;
            }
                continue;
        }
        nv_printf(NV_DBG_ERRORS,
            "NVRM: This PCI I/O region assigned to your NVIDIA device is invalid:\n"
            "NVRM: BAR%d is %dM @ 0x%llx (PCI:%04x:%02x:%02x.%x)\n", i,
            (NV_PCI_RESOURCE_SIZE(dev, i) >> 20),
            (NvU64)NV_PCI_RESOURCE_START(dev, i),
            NV_PCI_DOMAIN_NUMBER(dev),
            NV_PCI_BUS_NUMBER(dev), NV_PCI_SLOT_NUMBER(dev), PCI_FUNC(dev->devfn));
#if defined(NVCPU_X86)
        if ((NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64) &&
            ((NV_PCI_RESOURCE_START(dev, i) >> PAGE_SHIFT) > 0xfffffULL))
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: This is a 64-bit BAR mapped above 4GB by the system\n"
                "NVRM: BIOS or the Linux kernel.  The NVIDIA Linux/x86\n"
                "NVRM: graphics driver and other system software components\n"
                "NVRM: do not support this configuration.\n");
        }
        else
#endif
        if (NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: This is a 64-bit BAR, which some operating system\n"
                "NVRM: kernels and other system software components are known\n"
                "NVRM: to handle incorrectly.  Please see the README section\n"
                "NVRM: on 64-bit BARs for more information.\n");
        }
        else
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: The system BIOS may have misconfigured your GPU.\n");
        }
        goto failed;
    }

    if (!request_mem_region(NV_PCI_RESOURCE_START(dev, NV_GPU_BAR_INDEX_REGS),
                            NV_PCI_RESOURCE_SIZE(dev, NV_GPU_BAR_INDEX_REGS), nv_device_name))
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: request_mem_region failed for %dM @ 0x%llx. This can\n"
            "NVRM: occur when a driver such as rivatv is loaded and claims\n"
            "NVRM: ownership of the device's registers.\n",
            (NV_PCI_RESOURCE_SIZE(dev, NV_GPU_BAR_INDEX_REGS) >> 20),
            (NvU64)NV_PCI_RESOURCE_START(dev, NV_GPU_BAR_INDEX_REGS));
        goto failed;
    }

    NV_KMALLOC(nvl, sizeof(nv_linux_state_t));
    if (nvl == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate memory\n");
        goto err_not_supported;
    }

    os_mem_set(nvl, 0, sizeof(nv_linux_state_t));

    nv  = NV_STATE_PTR(nvl);

    pci_set_drvdata(dev, (void *)nvl);

    /* default to 32-bit PCI bus address space */
    dev->dma_mask = 0xffffffffULL;

    nvl->dev               = dev;

    nv->pci_info.vendor_id = dev->vendor;
    nv->pci_info.device_id = dev->device;
    nv->subsystem_id       = dev->subsystem_device;
    nv->subsystem_vendor   = dev->subsystem_vendor;
    nv->os_state           = (void *) nvl;
    nv->pci_info.domain    = NV_PCI_DOMAIN_NUMBER(dev);
    nv->pci_info.bus       = NV_PCI_BUS_NUMBER(dev);
    nv->pci_info.slot      = NV_PCI_SLOT_NUMBER(dev);
    nv->handle             = dev;
    nv->flags             |= flags;

    if (!nv_lock_init_locks(sp, nv))
    {
        goto err_not_supported;
    }

    for (i = 0, j = 0; i < NVRM_PCICFG_NUM_BARS && j < NV_GPU_NUM_BARS; i++)
    {
        if ((NV_PCI_RESOURCE_VALID(dev, i)) &&
            (NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_SPACE)
                == PCI_BASE_ADDRESS_SPACE_MEMORY)
        {
            NvU32 bar = 0;
            nv->bars[j].offset = NVRM_PCICFG_BAR_OFFSET(i);
            pci_read_config_dword(dev, nv->bars[j].offset, &bar);
            nv->bars[j].bus_address = (bar & PCI_BASE_ADDRESS_MEM_MASK);
            if (NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64)
            {
                pci_read_config_dword(dev, nv->bars[j].offset + 4, &bar);
                nv->bars[j].bus_address |= (((NvU64)bar) << 32);
            }
            nv->bars[j].cpu_address = NV_PCI_RESOURCE_START(dev, i);
            nv->bars[j].strapped_size = NV_PCI_RESOURCE_SIZE(dev, i);
            nv->bars[j].size = nv->bars[j].strapped_size;
            j++;
        }
    }
    nv->regs = &nv->bars[NV_GPU_BAR_INDEX_REGS];
    nv->fb   = &nv->bars[NV_GPU_BAR_INDEX_FB];

    nv->interrupt_line = dev->irq;

    nv_init_ibmnpu_info(nv);

    pci_set_master(dev);

#if defined(CONFIG_VGA_ARB) && !defined(NVCPU_PPC64LE)
#if defined(VGA_DEFAULT_DEVICE)
    vga_tryget(VGA_DEFAULT_DEVICE, VGA_RSRC_LEGACY_MASK);
#endif
    vga_set_legacy_decoding(dev, VGA_RSRC_NONE);
#endif

    if (rm_get_cpu_type(sp, &nv_cpu_type) != NV_OK)
        nv_printf(NV_DBG_ERRORS, "NVRM: error retrieving cpu type\n");

    if (NV_IS_GVI_DEVICE(nv))
    {
        if (!rm_gvi_init_private_state(sp, nv))
        {
            nv_printf(NV_DBG_ERRORS, "NVGVI: rm_init_gvi_private_state() failed!\n");
            goto err_not_supported;
        }

        if (rm_gvi_attach_device(sp, nv) != NV_OK)
        {
            rm_gvi_free_private_state(sp, nv);
            goto err_not_supported;
        }
    }
    else
    {
        NV_CHECK_PCI_CONFIG_SPACE(sp, nv, FALSE, TRUE, NV_MAY_SLEEP());

        if ((rm_is_supported_device(sp, nv)) != NV_OK)
            goto err_not_supported;

        if (!rm_init_private_state(sp, nv))
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: rm_init_private_state() failed!\n");
            goto err_zero_dev;
        }
    }

    nv_printf(NV_DBG_INFO,
              "NVRM: PCI:%04x:%02x:%02x.%x (%04x:%04x): BAR0 @ 0x%llx (%lluMB)\n",
              nv->pci_info.domain, nv->pci_info.bus, nv->pci_info.slot,
              PCI_FUNC(dev->devfn), nv->pci_info.vendor_id, nv->pci_info.device_id,
              nv->regs->cpu_address, (nv->regs->size >> 20));
    nv_printf(NV_DBG_INFO,
              "NVRM: PCI:%04x:%02x:%02x.%x (%04x:%04x): BAR1 @ 0x%llx (%lluMB)\n",
              nv->pci_info.domain, nv->pci_info.bus, nv->pci_info.slot,
              PCI_FUNC(dev->devfn), nv->pci_info.vendor_id, nv->pci_info.device_id,
              nv->fb->cpu_address, (nv->fb->size >> 20));

    num_nv_devices++;

    for (i = 0; i < NV_GPU_NUM_BARS; i++)
    {
        if (nv->bars[i].size != 0)
        {
            if (nv_user_map_register(nv->bars[i].cpu_address,
                nv->bars[i].strapped_size) != 0)
            {
                nv_printf(NV_DBG_ERRORS,
                          "NVRM: failed to register usermap for BAR %u!\n", i);
                for (j = 0; j < i; j++)
                {
                    nv_user_map_unregister(nv->bars[j].cpu_address,
                                           nv->bars[j].strapped_size);
                }
                goto err_zero_dev;
            }
        }
    }

    /*
     * The newly created nvl object is added to the nv_linux_devices global list
     * only after all the initialization operations for that nvl object are
     * completed, so as to protect against simultaneous lookup operations which
     * may discover a partially initialized nvl object in the list
     */
    LOCK_NV_LINUX_DEVICES();
    if (nv_linux_devices == NULL)
        nv_linux_devices = nvl;
    else
    {
        nv_linux_state_t *tnvl;
        for (tnvl = nv_linux_devices; tnvl->next != NULL;  tnvl = tnvl->next);
        tnvl->next = nvl;
    }
    UNLOCK_NV_LINUX_DEVICES();
    if (nvidia_frontend_add_device((void *)&nv_fops, nvl) != 0)
        goto err_zero_dev;

#if defined(NV_PM_VT_SWITCH_REQUIRED_PRESENT)
    pm_vt_switch_required(&nvl->dev->dev, NV_TRUE);
#endif

    nv_procfs_add_gpu(nvl);

#if defined(NV_VGPU_KVM_BUILD)
    if (nvidia_vgpu_vfio_probe(nvl->dev) != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: Failed to register device to vGPU VFIO module");
        goto err_zero_dev;
    }
#endif

    nv_kmem_cache_free_stack(sp);

    return 0;

err_zero_dev:
    rm_free_private_state(sp, nv);
err_not_supported:
    nv_destroy_ibmnpu_info(nv);
    nv_lock_destroy_locks(sp, nv);
    if (nvl != NULL)
    {
        NV_KFREE(nvl, sizeof(nv_linux_state_t));
    }
    release_mem_region(NV_PCI_RESOURCE_START(dev, NV_GPU_BAR_INDEX_REGS),
                       NV_PCI_RESOURCE_SIZE(dev, NV_GPU_BAR_INDEX_REGS));
    NV_PCI_DISABLE_DEVICE(dev);
    pci_set_drvdata(dev, NULL);
failed:
    nv_kmem_cache_free_stack(sp);
    return -1;
}

void
nvidia_remove(struct pci_dev *dev)
{
    nv_linux_state_t *nvl = NULL;
    nv_state_t *nv;
    nvidia_stack_t *sp = NULL;
    NvU32 i;

    nv_printf(NV_DBG_SETUP, "NVRM: removing GPU %04x:%02x:%02x.%x\n",
              NV_PCI_DOMAIN_NUMBER(dev), NV_PCI_BUS_NUMBER(dev),
              NV_PCI_SLOT_NUMBER(dev), PCI_FUNC(dev->devfn));

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return;
    }

    LOCK_NV_LINUX_DEVICES();
    nvl = pci_get_drvdata(dev);
    if (!nvl || (nvl->dev != dev))
    {
        goto done;
    }

    /* Sanity check: A removed device shouldn't have a non-zero usage_count */
    if (NV_ATOMIC_READ(nvl->usage_count) != 0)
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: Attempting to remove minor device %u with non-zero usage count!\n",
                  nvl->minor_num);
        WARN_ON(1);

        /* We can't continue without corrupting state, so just hang to give the
         * user some chance to do something about this before reboot */
        while (1)
            os_schedule();
    }

    nv = NV_STATE_PTR(nvl);
    if (nvl == nv_linux_devices)
        nv_linux_devices = nvl->next;
    else
    {
        nv_linux_state_t *tnvl;
        for (tnvl = nv_linux_devices; tnvl->next != nvl;  tnvl = tnvl->next);
        tnvl->next = nvl->next;
    }

    /* Remove proc entry for this GPU */
    nv_procfs_remove_gpu(nvl);

    down(&nvl->ldata_lock);
    UNLOCK_NV_LINUX_DEVICES();

#if defined(NV_PM_VT_SWITCH_REQUIRED_PRESENT)
    pm_vt_switch_unregister(&dev->dev);
#endif

#if defined(NV_VGPU_KVM_BUILD)
    nvidia_vgpu_vfio_remove(nvl->dev);
#endif

   /* Update the frontend data structures */
    nvidia_frontend_remove_device((void *)&nv_fops, nvl);

#if defined(NV_UVM_ENABLE)
    if (!NV_IS_GVI_DEVICE(nv))
    {
        NvU8 *uuid;
        // Inform UVM before disabling adapter
        if(rm_get_gpu_uuid_raw(sp, nv, &uuid, NULL) == NV_OK)
        {
            // this function cannot fail
            nv_uvm_notify_stop_device(uuid);
            // get_uuid allocates memory for this call free it here
            os_free_mem(uuid);
        }
    }
#endif

    if ((nv->flags & NV_FLAG_PERSISTENT_SW_STATE) || (nv->flags & NV_FLAG_OPEN))
    {
        if (nv->flags & NV_FLAG_PERSISTENT_SW_STATE)
        {
            rm_disable_gpu_state_persistence(sp, nv);
        }
        nv_shutdown_adapter(sp, nv, nvl);
        nv_dev_free_stacks(nvl);
    }

    nv_destroy_ibmnpu_info(nv);

    nv_lock_destroy_locks(sp, nv);

    num_probed_nv_devices--;

    pci_set_drvdata(dev, NULL);

    if (NV_IS_GVI_DEVICE(nv))
    {
        NV_WORKQUEUE_FLUSH();
        rm_gvi_detach_device(sp, nv);
        rm_gvi_free_private_state(sp, nv);
    }
    else
    {
        for (i = 0; i < NV_GPU_NUM_BARS; i++)
        {
            if (nv->bars[i].size != 0)
            {
                nv_user_map_unregister(nv->bars[i].cpu_address,
                                       nv->bars[i].strapped_size);
            }
        }
        rm_i2c_remove_adapters(sp, nv);
        rm_free_private_state(sp, nv);
    }
    release_mem_region(NV_PCI_RESOURCE_START(dev, NV_GPU_BAR_INDEX_REGS),
                       NV_PCI_RESOURCE_SIZE(dev, NV_GPU_BAR_INDEX_REGS));
    NV_PCI_DISABLE_DEVICE(dev);
    num_nv_devices--;

    NV_KFREE(nvl, sizeof(nv_linux_state_t));
    nv_kmem_cache_free_stack(sp);
    return;

done:
    UNLOCK_NV_LINUX_DEVICES();
    nv_kmem_cache_free_stack(sp);
}


#if defined(NV_PM_SUPPORT_DEVICE_DRIVER_MODEL)

static int
nv_power_management(
    struct pci_dev *dev,
    u32 pci_state,
    u32 power_state
)
{
    nv_state_t *nv;
    nv_linux_state_t *nvl = NULL;
    int status = NV_OK;
    nvidia_stack_t *sp = NULL;

    nv_printf(NV_DBG_INFO, "NVRM: nv_power_management: %d\n", pci_state);
    nvl = pci_get_drvdata(dev);

    if (!nvl || (nvl->dev != dev))
    {
        nv_printf(NV_DBG_WARNINGS, "NVRM: PM: invalid device!\n");
        return -1;
    }

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return -1;
    }

    nv = NV_STATE_PTR(nvl);
    NV_CHECK_PCI_CONFIG_SPACE(sp, nv, TRUE, TRUE, NV_MAY_SLEEP());

    switch (pci_state)
    {
         case PCI_D3hot:
            nv_printf(NV_DBG_INFO, "NVRM: ACPI: received suspend event\n");
            status = rm_power_management(sp, nv, 0, power_state);

            if (!nv_use_threaded_interrupts)
            {
                tasklet_kill(&nvl->tasklet);
            }

            // It's safe to call nv_kthread_q_stop even if queue is not initialized
            nv_kthread_q_stop(&nvl->bottom_half_q);

            nv_disable_pat_support();
            nv_pci_save_state(dev);
            break;

        case PCI_D0:
            nv_printf(NV_DBG_INFO, "NVRM: ACPI: received resume event\n");
            nv_pci_restore_state(dev);
            nv_enable_pat_support();

            if (!nv_use_threaded_interrupts)
            {
                tasklet_init(&nvl->tasklet, nvidia_isr_tasklet_bh, (NvUPtr)NV_STATE_PTR(nvl));
            }

            nv_kthread_q_item_init(&nvl->bottom_half_q_item, nvidia_isr_bh_unlocked, (void *)nv);
            status = nv_kthread_q_init(&nvl->bottom_half_q, nv_device_name);
            if (status != NV_OK)
                break;
            status = rm_power_management(sp, nv, 0, power_state);
            break;

        default:
            nv_printf(NV_DBG_WARNINGS, "NVRM: PM: unsupported event: %d\n", pci_state);
            status = -1;
    }

    nv_kmem_cache_free_stack(sp);

    if (status != NV_OK)
        nv_printf(NV_DBG_ERRORS, "NVRM: PM: failed event: %d\n", pci_state);

    return status;
}

int
nvidia_suspend(
    struct pci_dev *dev,
    pm_message_t state
)
{
    int pci_state = -1;
    u32 power_state;
    nv_state_t *nv;
    nv_linux_state_t *nvl = NULL;

    nvl = pci_get_drvdata(dev);

    if (!nvl || (nvl->dev != dev))
    {
        nv_printf(NV_DBG_WARNINGS, "NVRM: PM: invalid device!\n");
        return -1;
    }

    nv = NV_STATE_PTR(nvl);

    if (NV_IS_GVI_DEVICE(nv))
    {
        return nv_gvi_kern_suspend(dev, state);
    }

    nvidia_modeset_suspend(nv->gpu_id);

#if !defined(NV_PM_MESSAGE_T_PRESENT)
    pci_state = state;
#elif defined(NV_PCI_CHOOSE_STATE_PRESENT)
    pci_state = PCI_D3hot;
#endif

    power_state = NV_PM_ACPI_STANDBY;

#if defined(NV_PM_MESSAGE_T_HAS_EVENT)
    if (state.event == PM_EVENT_FREEZE) /* for hibernate */
        power_state = NV_PM_ACPI_HIBERNATE;
#endif

    return nv_power_management(dev, pci_state, power_state);
}

int
nvidia_resume(
    struct pci_dev *dev
)
{
    nv_state_t *nv;
    nv_linux_state_t *nvl = NULL;
    int ret;

    nvl = pci_get_drvdata(dev);

    if (!nvl || (nvl->dev != dev))
    {
        nv_printf(NV_DBG_WARNINGS, "NVRM: PM: invalid device!\n");
        return -1;
    }

    nv = NV_STATE_PTR(nvl);

    if (NV_IS_GVI_DEVICE(nv))
    {
        return nv_gvi_kern_resume(dev);
    }

    ret = nv_power_management(dev, PCI_D0, NV_PM_ACPI_RESUME);

    if (ret >= 0)
    {
        nvidia_modeset_resume(nv->gpu_id);
    }

    return ret;
}

#endif /* defined(NV_PM_SUPPORT_DEVICE_DRIVER_MODEL) */

#if defined(NV_PCI_ERROR_RECOVERY)
static pci_ers_result_t
nvidia_pci_error_detected(
    struct pci_dev *dev,
    enum pci_channel_state error
)
{
    nv_linux_state_t *nvl = pci_get_drvdata(dev);

    if ((nvl == NULL) || (nvl->dev != dev))
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s: invalid device!\n", __FUNCTION__);
        return PCI_ERS_RESULT_NONE;
    }

    /*
     * Tell Linux to continue recovery of the device. The kernel will enable
     * MMIO for the GPU and call the mmio_enabled callback.
     */
    return PCI_ERS_RESULT_CAN_RECOVER;
}

static pci_ers_result_t
nvidia_pci_mmio_enabled(
    struct pci_dev *dev
)
{
    NV_STATUS         status = NV_OK;
    nv_stack_t       *sp = NULL;
    nv_linux_state_t *nvl = pci_get_drvdata(dev);
    nv_state_t       *nv = NULL;

    if ((nvl == NULL) || (nvl->dev != dev))
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s: invalid device!\n", __FUNCTION__);
        goto done;
    }

    nv = NV_STATE_PTR(nvl);

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s: failed to allocate stack!\n",
            __FUNCTION__);
        goto done;
    }

    nv_printf(NV_DBG_ERRORS,
        "NVRM: A fatal error was detected on GPU " NV_PCI_DEV_FMT "\n.",
        NV_PCI_DEV_FMT_ARGS(nv));

    /*
     * MMIO should be re-enabled now. If we still get bad reads, there's
     * likely something wrong with the adapter itself that will require a
     * reset. This should let us know whether the GPU has completely fallen
     * off the bus or just did something the host didn't like.
     */
    status = rm_is_supported_device(sp, nv);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: The kernel has enabled MMIO for the device,\n"
            "NVRM: but it still appears unreachable. The device\n"
            "NVRM: will not function properly until it is reset.\n");
    }

    status = rm_log_gpu_crash(sp, nv);
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: failed to log crash data for GPU " NV_PCI_DEV_FMT " (0x%x)\n",
            NV_PCI_DEV_FMT_ARGS(nv), status);
        goto done;
    }

done:
    if (sp != NULL)
    {
        nv_kmem_cache_free_stack(sp);
    }

    /*
     * Tell Linux to abandon recovery of the device. The kernel might be able
     * to recover the device, but RM and clients don't yet support that.
     */
    return PCI_ERS_RESULT_DISCONNECT;
}
#endif

nv_state_t* NV_API_CALL nv_get_adapter_state(
    NvU32 domain,
    NvU8  bus,
    NvU8  slot
)
{
    nv_linux_state_t *nvl;

    LOCK_NV_LINUX_DEVICES();
    for (nvl = nv_linux_devices; nvl != NULL;  nvl = nvl->next)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);
        if (nv->pci_info.domain == domain && nv->pci_info.bus == bus
            && nv->pci_info.slot == slot)
        {
            UNLOCK_NV_LINUX_DEVICES();
            return nv;
        }
    }
    UNLOCK_NV_LINUX_DEVICES();

    return NULL;
}

nv_state_t* NV_API_CALL nv_get_ctl_state(void)
{
    return NV_STATE_PTR(&nv_ctl_device);
}

NV_STATUS NV_API_CALL nv_log_error(
    nv_state_t *nv,
    NvU32       error_number,
    const char *format,
    va_list    ap
)
{
    NV_STATUS status = NV_OK;
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    nv_report_error(nvl->dev, error_number, format, ap);
#if defined(CONFIG_CRAY_XT)
    status = nvos_forward_error_to_cray(nvl->dev, error_number,
                format, ap);
#endif

    return status;
}

NvU64 NV_API_CALL nv_get_dma_start_address(
    nv_state_t *nv
)
{
    NvU64 start = 0;
#if defined(NVCPU_PPC64LE)
    nv_linux_state_t      *nvl;
    struct pci_dev        *dev;
    dma_addr_t             dma_addr;
    NvU64                  saved_dma_mask;

    /*
     * If TCE bypass is disabled via a module parameter, then just return
     * the default (which is 0).
     *
     * Otherwise, the DMA start address only needs to be set once, and it
     * won't change afterward. Just return the cached value if asked again,
     * to avoid the kernel printing redundant messages to the kernel
     * log when we call pci_set_dma_mask().
     */
    nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    if ((nv_tce_bypass_mode == NV_TCE_BYPASS_MODE_DISABLE) ||
        (nvl->tce_bypass_enabled))
    {
        return nv->dma_addressable_start;
    }

    dev = nvl->dev;

    /*
     * Linux on IBM POWER8 offers 2 different DMA set-ups, sometimes
     * referred to as "windows".
     *
     * The "default window" provides a 2GB region of PCI address space
     * located below the 32-bit line. The IOMMU is used to provide a
     * "rich" mapping--any page in system memory can be mapped at an
     * arbitrary address within this window. The mappings are dynamic
     * and pass in and out of being as pci_map*()/pci_unmap*() calls
     * are made.
     *
     * Dynamic DMA Windows (sometimes "Huge DDW") provides a linear
     * mapping of the system's entire physical address space at some
     * fixed offset above the 59-bit line. IOMMU is still used, and
     * pci_map*()/pci_unmap*() are still required, but mappings are
     * static. They're effectively set up in advance, and any given
     * system page will always map to the same PCI bus address. I.e.
     *   physical 0x00000000xxxxxxxx => PCI 0x08000000xxxxxxxx
     *
     * This driver does not support the 2G default window because
     * of its limited size, and for reasons having to do with UVM.
     *
     * Linux on POWER8 will only provide the DDW-style full linear
     * mapping when the driver claims support for 64-bit DMA addressing
     * (a pre-requisite because the PCI addresses used in this case will
     * be near the top of the 64-bit range). The linear mapping
     * is not available in all system configurations.
     *
     * Detect whether the linear mapping is present by claiming
     * 64-bit support and then mapping physical page 0. For historical
     * reasons, Linux on POWER8 will never map a page to PCI address 0x0.
     * In the "default window" case page 0 will be mapped to some
     * non-zero address below the 32-bit line.  In the
     * DDW/linear-mapping case, it will be mapped to address 0 plus
     * some high-order offset.
     *
     * If the linear mapping is present and sane then return the offset
     * as the starting address for all DMA mappings.
     */
    saved_dma_mask = dev->dma_mask;
    if (pci_set_dma_mask(dev, DMA_BIT_MASK(64)) != 0)
    {
        goto done;
    }

    dma_addr = pci_map_single(dev, NULL, 1, DMA_BIDIRECTIONAL);
    if (pci_dma_mapping_error(dev, dma_addr))
    {
        pci_set_dma_mask(dev, saved_dma_mask);
        goto done;
    }

    pci_unmap_single(dev, dma_addr, 1, DMA_BIDIRECTIONAL);

    /*
     * From IBM: "For IODA2, native DMA bypass or KVM TCE-based implementation
     * of full 64-bit DMA support will establish a window in address-space
     * with the high 14 bits being constant and the bottom up-to-50 bits
     * varying with the mapping."
     *
     * Unfortunately, we don't have any good interfaces or definitions from
     * the kernel to get information about the DMA offset assigned by OS.
     * However, we have been told that the offset will be defined by the top
     * 14 bits of the address, and bits 40-49 will not vary for any DMA
     * mappings until 1TB of system memory is surpassed; this limitation is
     * essential for us to function properly since our current GPUs only
     * support 40 physical address bits. We are in a fragile place where we
     * need to tell the OS that we're capable of 64-bit addressing, while
     * relying on the assumption that the top 24 bits will not vary in this
     * case.
     *
     * The way we try to compute the window, then, is mask the trial mapping
     * against the DMA capabilities of the device. That way, devices with
     * greater addressing capabilities will only take the bits it needs to
     * define the window.
     */
    if ((dma_addr & DMA_BIT_MASK(32)) != 0)
    {
        /*
         * Huge DDW not available - page 0 mapped to non-zero address below
         * the 32-bit line.
         */
        nv_printf(NV_DBG_WARNINGS,
            "NVRM: DMA window limited by platform\n");
        pci_set_dma_mask(dev, saved_dma_mask);
        goto done;
    }
    else if ((dma_addr & saved_dma_mask) != 0)
    {
        NvU64 memory_size = os_get_num_phys_pages() * PAGE_SIZE;
        if ((dma_addr & ~saved_dma_mask) !=
            ((dma_addr + memory_size) & ~saved_dma_mask))
        {
            /*
             * The physical window straddles our addressing limit boundary,
             * e.g., for an adapter that can address up to 1TB, the window
             * crosses the 40-bit limit so that the lower end of the range
             * has different bits 63:40 than the higher end of the range.
             * We can only handle a single, static value for bits 63:40, so
             * we must fall back here.
             */
            nv_printf(NV_DBG_WARNINGS,
                "NVRM: DMA window limited by memory size\n");
            pci_set_dma_mask(dev, saved_dma_mask);
            goto done;
        }
    }

    nvl->tce_bypass_enabled = NV_TRUE;
    start = dma_addr & ~(saved_dma_mask);

    /* Update the coherent mask to match */
    dma_set_coherent_mask(&dev->dev, dev->dma_mask);

done:
#endif
    return start;
}

NV_STATUS NV_API_CALL nv_set_primary_vga_status(
    nv_state_t *nv
)
{
    /* IORESOURCE_ROM_SHADOW wasn't added until 2.6.10 */
#if defined(IORESOURCE_ROM_SHADOW)
    nv_linux_state_t *nvl;
    struct pci_dev *dev;

    nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    dev = nvl->dev;

    nv->primary_vga = ((NV_PCI_RESOURCE_FLAGS(dev, PCI_ROM_RESOURCE) &
        IORESOURCE_ROM_SHADOW) == IORESOURCE_ROM_SHADOW);
    return NV_OK;
#else
    return NV_ERR_NOT_SUPPORTED;
#endif
}

NV_STATUS NV_API_CALL nv_pci_trigger_recovery(
     nv_state_t *nv
)
{
    NV_STATUS status = NV_ERR_NOT_SUPPORTED;
#if defined(NV_PCI_ERROR_RECOVERY)
    nv_linux_state_t *nvl       = NV_GET_NVL_FROM_NV_STATE(nv);

    /*
     * Calling readl() on PPC64LE will allow the kernel to check its state for
     * the device and update it accordingly. This needs to be done before
     * checking if the PCI channel is offline, so that we don't check stale
     * state.
     *
     * This will also kick off the recovery process for the device.
     */
    if (NV_PCI_ERROR_RECOVERY_ENABLED())
    {
        if (readl(nv->regs->map) == 0xFFFFFFFF)
        {
            if (pci_channel_offline(nvl->dev))
            {
                nv_printf(NV_DBG_ERRORS,
                    "NVRM: PCI channel for device " NV_PCI_DEV_FMT " is offline\n",
                    NV_PCI_DEV_FMT_ARGS(nv));
                status = NV_OK;
            }
        }
    }
#endif
    return status;
}

NvBool NV_API_CALL nv_requires_dma_remap(
    nv_state_t *nv
)
{
    NvBool dma_remap = NV_FALSE;
#if !defined(NVCPU_ARM)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    dma_remap = !nv_dma_maps_swiotlb(nvl->dev);
#endif
    return dma_remap;
}

/*
 * Intended for use by external kernel modules to list nvidia gpu ids.
 */
NvBool nvidia_get_gpuid_list(NvU32 *gpu_ids, NvU32 *gpu_count)
{
    nv_linux_state_t *nvl;
    unsigned int count;
    NvBool ret = NV_TRUE;

    LOCK_NV_LINUX_DEVICES();

    count = 0;
    for (nvl = nv_linux_devices; nvl != NULL; nvl = nvl->next)
        count++;

    if (*gpu_count == 0)
    {
        goto done;
    }
    else if ((*gpu_count) < count)
    {
        ret = NV_FALSE;
        goto done;
    }

    count = 0;
    for (nvl = nv_linux_devices; nvl != NULL; nvl = nvl->next)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);
        gpu_ids[count++] = nv->gpu_id;
    }


done:

    *gpu_count = count;

    UNLOCK_NV_LINUX_DEVICES();

    return ret;
}

/*
 * Kernel-level analog to nvidia_open, intended for use by external
 * kernel modules. This increments the ref count of the device with
 * the given gpu_id and makes sure the device has been initialized.
 *
 * Returns -ENODEV if the given gpu_id does not exist.
 */
int nvidia_dev_get(NvU32 gpu_id, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl;
    int rc;

    /* Takes nvl->ldata_lock */
    nvl = find_gpu_id(gpu_id);
    if (!nvl)
        return -ENODEV;

    rc = nv_open_device(NV_STATE_PTR(nvl), sp);

    up(&nvl->ldata_lock);
    return rc;
}

/*
 * Kernel-level analog to nvidia_close, intended for use by external
 * kernel modules. This decrements the ref count of the device with
 * the given gpu_id, potentially tearing it down.
 */
void nvidia_dev_put(NvU32 gpu_id, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl;

    /* Takes nvl->ldata_lock */
    nvl = find_gpu_id(gpu_id);
    if (!nvl)
        return;

    nv_close_device(NV_STATE_PTR(nvl), sp);
    up(&nvl->ldata_lock);
}

/*
 * Like nvidia_dev_get but uses UUID instead of gpu_id. Note that this may
 * trigger initialization and teardown of unrelated devices to look up their
 * UUIDs.
 */
int nvidia_dev_get_uuid(const NvU8 *uuid, nvidia_stack_t *sp)
{
    nv_state_t *nv = NULL;
    nv_linux_state_t *nvl = NULL;
    const NvU8 *dev_uuid;
    int rc = 0;

    /* Takes nvl->ldata_lock */
    nvl = find_uuid_candidate(uuid);
    while (nvl)
    {
        nv = NV_STATE_PTR(nvl);

        /*
         * If the device is missing its UUID, this call exists solely so
         * rm_get_gpu_uuid_raw will be called and we can inspect the UUID.
         */
        rc = nv_open_device(nv, sp);
        if (rc != 0)
            goto out;

        /* The UUID should always be present following nv_open_device */
        dev_uuid = nv_state_get_gpu_uuid_cache(nv);
        WARN_ON(!dev_uuid);
        if (dev_uuid && memcmp(dev_uuid, uuid, GPU_UUID_LEN) == 0)
            break;

        /* No match, try again. */
        nv_close_device(nv, sp);
        up(&nvl->ldata_lock);
        nvl = find_uuid_candidate(uuid);
    }

    if (nvl)
        rc = 0;
    else
        rc = -ENODEV;

out:
    if (nvl)
        up(&nvl->ldata_lock);
    return rc;
}

/*
 * Like nvidia_dev_put but uses UUID instead of gpu_id.
 */
void nvidia_dev_put_uuid(const NvU8 *uuid, nvidia_stack_t *sp)
{
    nv_linux_state_t *nvl;

    /* Takes nvl->ldata_lock */
    nvl = find_uuid(uuid);
    if (!nvl)
        return;

    nv_close_device(NV_STATE_PTR(nvl), sp);
    up(&nvl->ldata_lock);
}

int nvidia_dev_get_pci_info(const NvU8 *uuid, struct pci_dev **pci_dev_out, NvU64 *dma_start, NvU64 *dma_limit)
{
    nv_linux_state_t *nvl;

    /* Takes nvl->ldata_lock */
    nvl = find_uuid(uuid);
    if (!nvl)
        return -ENODEV;

    *pci_dev_out = nvl->dev;
    *dma_start = nvl->nv_state.dma_addressable_start;
    *dma_limit = nvl->nv_state.dma_addressable_limit;

    up(&nvl->ldata_lock);

    return 0;
}

NV_STATUS NV_API_CALL nv_get_device_memory_config(
    nv_state_t *nv,
    NvU32 *compr_addr_sys_phys,
    NvU32 *addr_guest_phys,
    NvU32 *addr_width,
    NvU32 *granularity,
    NvS32 *node_id
)
{
    NV_STATUS status = NV_ERR_NOT_SUPPORTED;
#if defined(NVCPU_PPC64LE)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    nv_numa_info_t *numa_info;

    if (!nv_numa_info_valid(nvl))
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    numa_info = &nvl->npu->numa_info;

    *node_id = numa_info->node_id;

    *compr_addr_sys_phys =
        numa_info->compr_sys_phys_addr >> nv_volta_addr_space_width;
    *addr_guest_phys =
        numa_info->guest_phys_addr >> nv_volta_addr_space_width;

    *addr_width = nv_volta_dma_addr_size - nv_volta_addr_space_width;
    *granularity = nv_volta_addr_space_width;

    status = NV_OK;
#endif
    return status;
}

void NV_API_CALL nv_warn_about_vesafb(void)
{
#if (defined(NVCPU_X86) || defined(NVCPU_X86_64))
    if (!nv_fbdev_failure_detected)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s", __fbdev_warning);
        nv_procfs_add_warning("fbdev", __fbdev_warning);
    }
    nv_fbdev_failure_detected = 1;
#endif
}

#if defined(NVCPU_PPC64LE)

NV_STATUS NV_API_CALL nv_get_nvlink_line_rate(
    nv_state_t *nvState,
    NvU32      *linerate
)
{
#if defined(NV_PNV_PCI_GET_NPU_DEV_PRESENT) && defined(NV_OF_GET_PROPERTY_PRESENT)

    nv_linux_state_t *nvl;
    struct pci_dev   *npuDev;
    NvU32            *pSpeedPtr = NULL;
    NvU32            speed;
    int              len;

    if (nvState != NULL)
        nvl = NV_GET_NVL_FROM_NV_STATE(nvState);
    else
        return NV_ERR_INVALID_ARGUMENT;

    if (!nvl->npu)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    npuDev = nvl->npu->devs[0];
    if (!npuDev->dev.of_node)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: %s: OF Node not found in IBM-NPU device node\n",
                  __FUNCTION__);
        return NV_ERR_NOT_SUPPORTED;
    }

    pSpeedPtr = (NvU32 *) of_get_property(npuDev->dev.of_node, "ibm,nvlink-speed", &len);

    if (pSpeedPtr)
    {
        speed = (NvU32) be32_to_cpup(pSpeedPtr);
    }
    else
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    if (!speed)
    {
        return NV_ERR_NOT_SUPPORTED;
    }
    else
    {
        *linerate = speed;
    }

    return NV_OK;

#endif

    return NV_ERR_NOT_SUPPORTED;
}

#endif

#if defined(NV_BACKLIGHT_DEVICE_REGISTER_PRESENT)
static int nv_update_backlight_status(struct backlight_device *bd)
{
    nvidia_stack_t *sp;
    nv_state_t *nv = bl_get_data(bd);
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    NvU32 displayId = nvl->backlight.displayId;
    NV_STATUS status;
    int rc;

    rc = nv_kmem_cache_alloc_stack(&sp);
    if (rc != 0)
    {
        return rc;
    }

    status = rm_set_backlight(sp, nv, displayId, bd->props.brightness);
    if (status != NV_OK)
    {
        rc = -EINVAL;
    }

    nv_kmem_cache_free_stack(sp);
    return rc;
}

static int nv_get_backlight_brightness(struct backlight_device *bd)
{
    nvidia_stack_t *sp;
    nv_state_t *nv = bl_get_data(bd);
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    const NvU32 displayId = nvl->backlight.displayId;
    NV_STATUS status;
    NvU32 brightness;
    int rc;

    rc = nv_kmem_cache_alloc_stack(&sp);
    if (rc != 0)
    {
        return -1;
    }

    status = rm_get_backlight(sp, nv, displayId, &brightness);

    nv_kmem_cache_free_stack(sp);

    if (status == NV_OK)
    {
        return brightness;
    }

    return -1;
}

static const struct backlight_ops nvidia_backlight_ops = {
    .update_status = nv_update_backlight_status,
    .get_brightness = nv_get_backlight_brightness,
};
#endif


void NV_API_CALL nv_register_backlight(
    nv_state_t *nv,
    NvU32 displayId,
    NvU32 currentBrightness
)
{
#if defined(NV_BACKLIGHT_DEVICE_REGISTER_PRESENT)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    char name[11];
    struct backlight_properties props = {
        .brightness = currentBrightness,
        .max_brightness = 100,
# if defined(NV_BACKLIGHT_PROPERTIES_TYPE_PRESENT)
        .type = BACKLIGHT_RAW,
# endif
    };

    snprintf(name, sizeof(name), "nvidia_%d", nvl->minor_num);
    name[sizeof(name) - 1] = '\0';

    nvl->backlight.dev = backlight_device_register(name, &nvl->dev->dev, nv,
                                                   &nvidia_backlight_ops,
                                                   &props);
    nvl->backlight.displayId = displayId;
#endif
}

void NV_API_CALL nv_unregister_backlight(
    nv_state_t *nv
)
{
#if defined(NV_BACKLIGHT_DEVICE_REGISTER_PRESENT)
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);

    if (nvl->backlight.dev)
    {
        backlight_device_unregister(nvl->backlight.dev);
    }
#endif
}
