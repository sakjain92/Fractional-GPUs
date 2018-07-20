/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2001-2016 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_LINUX_H_
#define _NV_LINUX_H_

#include "nvstatus.h"
#include "nv-misc.h"
#include "nv.h"
#include "conftest.h"

#include "nv-lock.h"
#include "nv-pgprot.h"
#include "nv-mm.h"
#include "os-interface.h"
#include "nv-timer.h"

#define NV_KERNEL_NAME "Linux"

#ifndef AUTOCONF_INCLUDED
#if defined(NV_GENERATED_AUTOCONF_H_PRESENT)
#include <generated/autoconf.h>
#else
#include <linux/autoconf.h>
#endif
#endif

#if defined(NV_GENERATED_UTSRELEASE_H_PRESENT)
  #include <generated/utsrelease.h>
#endif

#if defined(NV_GENERATED_COMPILE_H_PRESENT)
  #include <generated/compile.h>
#endif

#include <linux/version.h>
#include <linux/utsname.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 9)
#error "This driver does not support kernels older than 2.6.9!"
#elif LINUX_VERSION_CODE < KERNEL_VERSION(2, 7, 0)
#  define KERNEL_2_6
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(3, 0, 0)
#  define KERNEL_3
#else
#error "This driver does not support development kernels!"
#endif

#if defined (CONFIG_SMP) && !defined (__SMP__)
#define __SMP__
#endif

#if defined (CONFIG_MODVERSIONS) && !defined (MODVERSIONS)
#  define MODVERSIONS
#endif

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kmod.h>
#include <asm/bug.h>

#include <linux/mm.h>

#if !defined(VM_RESERVED)
#define VM_RESERVED    0x00000000
#endif
#if !defined(VM_DONTEXPAND)
#define VM_DONTEXPAND  0x00000000
#endif
#if !defined(VM_DONTDUMP)
#define VM_DONTDUMP    0x00000000
#endif

#include <linux/init.h>             /* module_init, module_exit         */
#include <linux/types.h>            /* pic_t, size_t, __u32, etc        */
#include <linux/errno.h>            /* error codes                      */
#include <linux/list.h>             /* circular linked list             */
#include <linux/stddef.h>           /* NULL, offsetof                   */
#include <linux/hashtable.h>	    /* hash table                       */
#include <linux/wait.h>             /* wait queues                      */
#include <linux/string.h>           /* strchr(), strpbrk()              */

#include <linux/ctype.h>            /* isspace(), etc                   */
#include <linux/console.h>          /* acquire_console_sem(), etc       */
#include <linux/cpufreq.h>          /* cpufreq_get                      */

#include <linux/slab.h>             /* kmalloc, kfree, etc              */
#include <linux/vmalloc.h>          /* vmalloc, vfree, etc              */

#include <linux/poll.h>             /* poll_wait                        */
#include <linux/delay.h>            /* mdelay, udelay                   */

#include <linux/sched.h>            /* suser(), capable() replacement   */

/*
 * sched.h was refactored with this commit (as part of Linux 4.11)
 *   2017-03-03  1827adb11ad26b2290dc9fe2aaf54976b2439865
 */
#if defined(NV_LINUX_SCHED_SIGNAL_H_PRESENT)
#include <linux/sched/signal.h>     /* task_lock(), task_unlock()       */
#endif

#if defined(NV_LINUX_SCHED_TASK_H_PRESENT)
#include <linux/sched/task.h>       /* task_lock(), task_unlock()       */
#endif

/* task and signal-related items, for kernels < 4.11: */
#include <linux/sched.h>            /* task_lock(), task_unlock()       */

#include <linux/moduleparam.h>      /* module_param()                   */
#include <asm/tlbflush.h>           /* flush_tlb(), flush_tlb_all()     */
#include <asm/kmap_types.h>         /* page table entry lookup          */

#include <linux/pci.h>              /* pci_find_class, etc              */
#include <linux/interrupt.h>        /* tasklets, interrupt helpers      */
#include <linux/timer.h>
#include <linux/file.h>             /* fget(), fput()                   */
#include <linux/rbtree.h>
#include <linux/cpu.h>              /* CPU hotplug support              */

#include <asm/div64.h>              /* do_div()                         */
#if defined(NV_ASM_SYSTEM_H_PRESENT)
#include <asm/system.h>             /* cli, sli, save_flags             */
#endif
#include <asm/io.h>                 /* ioremap, virt_to_phys            */
#include <asm/uaccess.h>            /* access_ok                        */
#include <asm/page.h>               /* PAGE_OFFSET                      */
#include <asm/pgtable.h>            /* pte bit definitions              */

#include "nv-list-helpers.h"

/*
 * Use current->cred->euid, instead of calling current_euid().
 * The latter can pull in the GPL-only debug_lockdep_rcu_enabled()
 * symbol when CONFIG_PROVE_RCU.  That is only used for debugging.
 *
 * The Linux kernel relies on the assumption that only the current process
 * is permitted to change its cred structure. Therefore, current_euid()
 * does not require the RCU's read lock on current->cred.
 */
#if defined(NV_TASK_STRUCT_HAS_CRED)
#define NV_CURRENT_EUID() (__kuid_val(current->cred->euid))
#else
#define NV_CURRENT_EUID() (__kuid_val(current->euid))
#endif

#if !defined(NV_KUID_T_PRESENT)
typedef uid_t kuid_t;

static inline uid_t __kuid_val(kuid_t uid)
{
    return uid;
}
#endif

#if defined(NVCPU_X86_64) && !defined(HAVE_COMPAT_IOCTL)
#include <linux/syscalls.h>         /* sys_ioctl()                      */
#include <linux/ioctl32.h>          /* register_ioctl32_conversion()    */
#endif

#if !defined(NV_FILE_OPERATIONS_HAS_IOCTL) && \
  !defined(NV_FILE_OPERATIONS_HAS_UNLOCKED_IOCTL)
#error "struct file_operations compile test likely failed!"
#endif

#if defined(CONFIG_VGA_ARB)
#include <linux/vgaarb.h>
#endif

#if defined(NV_VM_INSERT_PAGE_PRESENT)
#include <linux/pagemap.h>
#include <linux/dma-mapping.h>
#endif

#if defined(CONFIG_SWIOTLB) && defined(NVCPU_AARCH64)
#include <linux/swiotlb.h>
#endif

#if defined(NV_SCATTERLIST_HAS_PAGE_LINK)
#include <linux/scatterlist.h>
#endif

#include <linux/completion.h>
#include <linux/highmem.h>

#include <linux/workqueue.h>        /* workqueue                        */
#include <nv-kthread-q.h>           /* kthread based queue              */

#if defined(NV_LINUX_EFI_H_PRESENT)
#include <linux/efi.h>              /* efi_enabled                      */
#endif
#if defined(NV_LINUX_SCREEN_INFO_H_PRESENT)
#include <linux/screen_info.h>      /* screen_info                      */
#else
#include <linux/tty.h>              /* screen_info                      */
#endif

#if !defined(CONFIG_PCI)
#warning "Attempting to build driver for a platform with no PCI support!"
#include <asm-generic/pci-dma-compat.h>
#endif

#if defined(NV_EFI_ENABLED_PRESENT) && defined(NV_EFI_ENABLED_ARGUMENT_COUNT)
#if (NV_EFI_ENABLED_ARGUMENT_COUNT == 1)
#define NV_EFI_ENABLED() efi_enabled(EFI_BOOT)
#else
#error "NV_EFI_ENABLED_ARGUMENT_COUNT value unrecognized!"
#endif
#elif (defined(NV_EFI_ENABLED_PRESENT) || defined(efi_enabled))
#define NV_EFI_ENABLED() efi_enabled
#else
#define NV_EFI_ENABLED() 0
#endif

#if defined(CONFIG_CRAY_XT)
#include <cray/cray_nvidia.h>
NV_STATUS nvos_forward_error_to_cray(struct pci_dev *, NvU32,
        const char *, va_list);
#endif

#if defined(NVCPU_PPC64LE) && defined(CONFIG_EEH)
#include <asm/eeh.h>
#define NV_PCI_ERROR_RECOVERY_ENABLED() eeh_enabled()
#define NV_PCI_ERROR_RECOVERY
#endif

#if defined(NV_ASM_SET_MEMORY_H_PRESENT)
#include <asm/set_memory.h>
#endif

#if defined(NV_SET_MEMORY_UC_PRESENT)
#undef NV_SET_PAGES_UC_PRESENT
#undef NV_CHANGE_PAGE_ATTR_PRESENT
#elif defined(NV_SET_PAGES_UC_PRESENT)
#undef NV_CHANGE_PAGE_ATTR_PRESENT
#endif

#if !defined(NVCPU_FAMILY_ARM) && !defined(NVCPU_PPC64LE)
#if !defined(NV_SET_MEMORY_UC_PRESENT) && !defined(NV_SET_PAGES_UC_PRESENT) && \
  !defined(NV_CHANGE_PAGE_ATTR_PRESENT)
#error "This driver requires the ability to change memory types!"
#endif
#endif

/*
 * Using change_page_attr() on early Linux/x86-64 2.6 kernels may
 * result in a BUG() being triggered. The underlying problem
 * actually exists on multiple architectures and kernels, but only
 * the above check for the condition and trigger a BUG().
 *
 * Note that this is a due to a bug in the Linux kernel, not an
 * NVIDIA driver bug.
 *
 * We therefore need to determine at runtime if change_page_attr()
 * can be used safely on these kernels.
 */
#if defined(NV_CHANGE_PAGE_ATTR_PRESENT) && defined(NVCPU_X86_64) && \
  (LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 11))
#define NV_CHANGE_PAGE_ATTR_BUG_PRESENT
#endif

/*
 * Traditionally, CONFIG_XEN indicated that the target kernel was
 * built exclusively for use under a Xen hypervisor, requiring
 * modifications to or disabling of a variety of NVIDIA graphics
 * driver code paths. As of the introduction of CONFIG_PARAVIRT
 * and support for Xen hypervisors within the CONFIG_PARAVIRT_GUEST
 * architecture, CONFIG_XEN merely indicates that the target
 * kernel can run under a Xen hypervisor, but not that it will.
 *
 * If CONFIG_XEN and CONFIG_PARAVIRT are defined, the old Xen
 * specific code paths are disabled. If the target kernel executes
 * stand-alone, the NVIDIA graphics driver will work fine. If the
 * kernels executes under a Xen (or other) hypervisor, however, the
 * NVIDIA graphics driver has no way of knowing and is unlikely
 * to work correctly.
 */
#if defined(CONFIG_XEN) && !defined(CONFIG_PARAVIRT)
#include <asm/maddr.h>
#include <xen/interface/memory.h>
#define NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL
#endif

#include "nv-procfs.h"

#ifdef CONFIG_KDB
#include <linux/kdb.h>
#include <asm/kdb.h>
#endif

#if defined(CONFIG_X86_REMOTE_DEBUG)
#include <linux/gdb.h>
#endif

#if defined(DEBUG) && defined(CONFIG_KGDB) && \
    defined(NVCPU_FAMILY_ARM)
#include <asm/kgdb.h>
#endif

#if (defined(NVCPU_X86) || defined(NVCPU_X86_64)) && \
  !defined(NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL)
#define NV_ENABLE_PAT_SUPPORT
#endif

#define NV_PAT_MODE_DISABLED    0
#define NV_PAT_MODE_KERNEL      1
#define NV_PAT_MODE_BUILTIN     2

extern int nv_pat_mode;

#if defined(CONFIG_HOTPLUG_CPU)
#define NV_ENABLE_HOTPLUG_CPU
#include <linux/notifier.h>         /* struct notifier_block, etc       */
#endif

#if (defined(CONFIG_I2C) || defined(CONFIG_I2C_MODULE))
#include <linux/i2c.h>
#endif

#if defined(CONFIG_ACPI)
#include <linux/acpi.h>
#if defined(NV_ACPI_DEVICE_OPS_HAS_MATCH) || defined(ACPI_VIDEO_HID)
#define NV_LINUX_ACPI_EVENTS_SUPPORTED 1
#endif
#endif

#if (defined(CONFIG_IPMI_HANDLER) || defined(CONFIG_IPMI_HANDLER_MODULE))
#include <linux/ipmi.h>
#endif

#if defined(NV_PCI_DMA_MAPPING_ERROR_PRESENT)
#if (NV_PCI_DMA_MAPPING_ERROR_ARGUMENT_COUNT == 2)
#define NV_PCI_DMA_MAPPING_ERROR(dev, addr) \
    pci_dma_mapping_error(dev, addr)
#elif (NV_PCI_DMA_MAPPING_ERROR_ARGUMENT_COUNT == 1)
#define NV_PCI_DMA_MAPPING_ERROR(dev, addr) \
    pci_dma_mapping_error(addr)
#else
#error "NV_PCI_DMA_MAPPING_ERROR_ARGUMENT_COUNT value unrecognized!"
#endif
#elif defined(NV_VM_INSERT_PAGE_PRESENT)
#error "NV_PCI_DMA_MAPPING_ERROR() undefined!"
#endif

#if defined(NV_LINUX_ACPI_EVENTS_SUPPORTED)
#if (NV_ACPI_WALK_NAMESPACE_ARGUMENT_COUNT == 6)
#define NV_ACPI_WALK_NAMESPACE(type, args...) acpi_walk_namespace(type, args)
#elif (NV_ACPI_WALK_NAMESPACE_ARGUMENT_COUNT == 7)
#define NV_ACPI_WALK_NAMESPACE(type, start_object, max_depth, \
        user_function, args...) \
    acpi_walk_namespace(type, start_object, max_depth, \
            user_function, NULL, args)
#else
#error "NV_ACPI_WALK_NAMESPACE_ARGUMENT_COUNT value unrecognized!"
#endif
#endif

#if defined(CONFIG_PREEMPT_RT) || defined(CONFIG_PREEMPT_RT_FULL)
#define NV_CONFIG_PREEMPT_RT 1
#endif

#if defined(NVCPU_X86)
#ifndef write_cr4
#define write_cr4(x) __asm__ ("movl %0,%%cr4" :: "r" (x));
#endif

#ifndef read_cr4
#define read_cr4()                                  \
 ({                                                 \
      unsigned int __cr4;                           \
      __asm__ ("movl %%cr4,%0" : "=r" (__cr4));     \
      __cr4;                                        \
  })
#endif

#ifndef wbinvd
#define wbinvd() __asm__ __volatile__("wbinvd" ::: "memory");
#endif
#endif /* defined(NVCPU_X86) */

#if defined(NV_WRITE_CR4_PRESENT)
#define NV_READ_CR4()       read_cr4()
#define NV_WRITE_CR4(cr4)   write_cr4(cr4)
#else
#define NV_READ_CR4()       __read_cr4()
#define NV_WRITE_CR4(cr4)   __write_cr4(cr4)
#endif

#ifndef get_cpu
#define get_cpu() smp_processor_id()
#define put_cpu()
#endif

#if !defined(unregister_hotcpu_notifier)
#define unregister_hotcpu_notifier unregister_cpu_notifier
#endif
#if !defined(register_hotcpu_notifier)
#define register_hotcpu_notifier register_cpu_notifier
#endif

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
#if !defined(pmd_large)
#define pmd_large(_pmd) \
    ((pmd_val(_pmd) & (_PAGE_PSE|_PAGE_PRESENT)) == (_PAGE_PSE|_PAGE_PRESENT))
#endif
#endif /* defined(NVCPU_X86) || defined(NVCPU_X86_64) */

#define NV_GET_PAGE_COUNT(page_ptr) \
  ((unsigned int)page_count(NV_GET_PAGE_STRUCT(page_ptr->phys_addr)))
#define NV_GET_PAGE_FLAGS(page_ptr) \
  (NV_GET_PAGE_STRUCT(page_ptr->phys_addr)->flags)
#define NV_LOCK_PAGE(ptr_ptr) \
  SetPageReserved(NV_GET_PAGE_STRUCT(page_ptr->phys_addr))
#define NV_UNLOCK_PAGE(page_ptr) \
  ClearPageReserved(NV_GET_PAGE_STRUCT(page_ptr->phys_addr))

#if !defined(__GFP_COMP)
#define __GFP_COMP 0
#endif

#if !defined(DEBUG) && defined(__GFP_NOWARN)
#define NV_GFP_KERNEL (GFP_KERNEL | __GFP_NOWARN)
#define NV_GFP_ATOMIC (GFP_ATOMIC | __GFP_NOWARN)
#else
#define NV_GFP_KERNEL (GFP_KERNEL)
#define NV_GFP_ATOMIC (GFP_ATOMIC)
#endif

#if defined(GFP_DMA32)
/*
 * GFP_DMA32 is similar to GFP_DMA, but instructs the Linux zone
 * allocator to allocate memory from the first 4GB on platforms
 * such as Linux/x86-64; the alternative is to use an IOMMU such
 * as the one implemented with the K8 GART, if available.
 */
#define NV_GFP_DMA32 (NV_GFP_KERNEL | GFP_DMA32)
#else
#define NV_GFP_DMA32 (NV_GFP_KERNEL)
#endif

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
#define CACHE_FLUSH()  asm volatile("wbinvd":::"memory")
#define WRITE_COMBINE_FLUSH() asm volatile("sfence":::"memory")
#elif defined(NVCPU_FAMILY_ARM)
#if defined(NVCPU_ARM)
    static inline void nv_flush_cache_cpu(void *info)
    {
        __cpuc_flush_kern_all();
    }
#define CACHE_FLUSH()            nv_flush_cache_cpu(NULL)
#define CACHE_FLUSH_ALL()        NV_ON_EACH_CPU(nv_flush_cache_cpu, NULL)
#if defined(NV_OUTER_FLUSH_ALL_PRESENT)
#define OUTER_FLUSH_ALL()               outer_flush_all()
#endif
#if defined(CONFIG_OUTER_CACHE)
#define OUTER_FLUSH_RANGE(start, end)   outer_flush_range((start),(end))
#endif
#define WRITE_COMBINE_FLUSH()    { dsb(); outer_sync(); }
#elif defined(NVCPU_AARCH64)
    static inline void nv_flush_cache_cpu(void *info)
    {
        flush_cache_all();
    }
#define CACHE_FLUSH()            nv_flush_cache_cpu(NULL)
#define CACHE_FLUSH_ALL()        NV_ON_EACH_CPU(nv_flush_cache_cpu, NULL)
#define WRITE_COMBINE_FLUSH()    mb()
#endif
#elif defined(NVCPU_PPC64LE)
#define CACHE_FLUSH()            asm volatile("sync":::"memory")
#define WRITE_COMBINE_FLUSH()    asm volatile("sync":::"memory")
#endif

#if defined(NVCPU_FAMILY_ARM) || defined(NVCPU_PPC64LE)
#define NV_ALLOW_WRITE_COMBINING(mt)    1
#elif (defined(NVCPU_X86) || defined(NVCPU_X86_64))
#if defined(NV_ENABLE_PAT_SUPPORT)
#define NV_ALLOW_WRITE_COMBINING(mt)    \
    ((nv_pat_mode != NV_PAT_MODE_DISABLED) && \
     ((mt) != NV_MEMORY_TYPE_REGISTERS))
#else
#define NV_ALLOW_WRITE_COMBINING(mt)    0
#endif
#endif

#if !defined(IRQF_SHARED)
#define IRQF_SHARED SA_SHIRQ
#endif

#define NV_MAX_RECURRING_WARNING_MESSAGES 10

#ifndef NVWATCH

/* various memory tracking/debugging techniques
 * disabled for retail builds, enabled for debug builds
 */

// allow an easy way to convert all debug printfs related to memory
// management back and forth between 'info' and 'errors'
#if defined(NV_DBG_MEM)
#define NV_DBG_MEMINFO NV_DBG_ERRORS
#else
#define NV_DBG_MEMINFO NV_DBG_INFO
#endif

#define NV_MEM_TRACKING_PAD_SIZE(size) \
    (size) = NV_ALIGN_UP((size + sizeof(void *)), sizeof(void *))

#define NV_MEM_TRACKING_HIDE_SIZE(ptr, size)            \
    if ((ptr != NULL) && (*(ptr) != NULL))              \
    {                                                   \
        NvU8 *__ptr;                                    \
        *(unsigned long *) *(ptr) = (size);             \
        __ptr = *(ptr); __ptr += sizeof(void *);        \
        *(ptr) = (void *) __ptr;                        \
    }
#define NV_MEM_TRACKING_RETRIEVE_SIZE(ptr, size)        \
    {                                                   \
        NvU8 *__ptr = (ptr); __ptr -= sizeof(void *);   \
        (ptr) = (void *) __ptr;                         \
        (size) = *(unsigned long *) (ptr);              \
    }

/* keep track of memory usage */
#include "nv-memdbg.h"

static inline void *nv_vmalloc(unsigned long size)
{
    void *ptr = __vmalloc(size, GFP_KERNEL, PAGE_KERNEL);
    if (ptr)
        NV_MEMDBG_ADD(ptr, size);
    return ptr;
}

static inline void nv_vfree(void *ptr, NvU32 size)
{
    NV_MEMDBG_REMOVE(ptr, size);
    vfree(ptr);
}

static inline void *nv_ioremap(NvU64 phys, NvU64 size)
{
    void *ptr = ioremap(phys, size);
    if (ptr)
        NV_MEMDBG_ADD(ptr, size);
    return ptr;
}

static inline void *nv_ioremap_nocache(NvU64 phys, NvU64 size)
{
    void *ptr = ioremap_nocache(phys, size);
    if (ptr)
        NV_MEMDBG_ADD(ptr, size);
    return ptr;
}

static inline void *nv_ioremap_cache(NvU64 phys, NvU64 size)
{
#if defined(NV_IOREMAP_CACHE_PRESENT)
    void *ptr = ioremap_cache(phys, size);
    if (ptr)
        NV_MEMDBG_ADD(ptr, size);
    return ptr;
#else
    return nv_ioremap(phys, size);
#endif
}

static inline void *nv_ioremap_wc(NvU64 phys, NvU64 size)
{
#if defined(NV_IOREMAP_WC_PRESENT)
    void *ptr = ioremap_wc(phys, size);
    if (ptr)
        NV_MEMDBG_ADD(ptr, size);
    return ptr;
#else
    return nv_ioremap_nocache(phys, size);
#endif
}

static inline void nv_iounmap(void *ptr, NvU64 size)
{
    NV_MEMDBG_REMOVE(ptr, size);
    iounmap(ptr);
}

#define NV_KMALLOC(ptr, size) \
    { \
        (ptr) = kmalloc(size, NV_GFP_KERNEL); \
        if (ptr) \
            NV_MEMDBG_ADD(ptr, size); \
    }

#define NV_KMALLOC_ATOMIC(ptr, size) \
    { \
        (ptr) = kmalloc(size, NV_GFP_ATOMIC); \
        if (ptr) \
            NV_MEMDBG_ADD(ptr, size); \
    }


#define NV_KFREE(ptr, size) \
    { \
        NV_MEMDBG_REMOVE(ptr, size); \
        kfree((void *) (ptr)); \
    }

#define NV_ALLOC_PAGES_NODE(ptr, nid, order, gfp_mask) \
    { \
        (ptr) = (unsigned long)page_address(alloc_pages_node(nid, gfp_mask, order)); \
    }

#define NV_GET_FREE_PAGES(ptr, order, gfp_mask)      \
    {                                                \
        (ptr) = __get_free_pages(gfp_mask, order);   \
    }

#define NV_FREE_PAGES(ptr, order)                    \
    {                                                \
        free_pages(ptr, order);                      \
    }

extern NvBool nvos_is_chipset_io_coherent(void);

static inline NvUPtr nv_vmap(struct page **pages, NvU32 page_count,
    NvBool cached)
{
    void *ptr;
    pgprot_t prot = PAGE_KERNEL;
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    prot = cached ? PAGE_KERNEL : PAGE_KERNEL_NOCACHE;
#elif defined(NVCPU_FAMILY_ARM)
    if (!nvos_is_chipset_io_coherent())
        prot = cached ? PAGE_KERNEL : NV_PGPROT_UNCACHED(PAGE_KERNEL);
#endif
    /* All memory cached in PPC64LE; can't honor 'cached' input. */
    ptr = vmap(pages, page_count, VM_MAP, prot);
    if (ptr)
        NV_MEMDBG_ADD(ptr, page_count * PAGE_SIZE);
    return (NvUPtr)ptr;
}

static inline void nv_vunmap(NvUPtr vaddr, NvU32 page_count)
{
    vunmap((void *)vaddr);
    NV_MEMDBG_REMOVE((void *)vaddr, page_count * PAGE_SIZE);
}

#endif /* !defined NVWATCH */

#if defined(NV_SMP_CALL_FUNCTION_PRESENT)
#if (NV_SMP_CALL_FUNCTION_ARGUMENT_COUNT == 4)
#define NV_SMP_CALL_FUNCTION(func, info)                     \
    ({                                                       \
        int __ret = smp_call_function(func, info, 1, 1);     \
        __ret;                                               \
     })
#elif (NV_SMP_CALL_FUNCTION_ARGUMENT_COUNT == 3)
#define NV_SMP_CALL_FUNCTION(func, info)                     \
    ({                                                       \
        int __ret = smp_call_function(func, info, 1);        \
        __ret;                                               \
     })
#else
#error "NV_SMP_CALL_FUNCTION_ARGUMENT_COUNT value unrecognized!"
#endif
#elif defined(CONFIG_SMP)
#error "NV_SMP_CALL_FUNCTION() undefined (smp_call_function() unavailable)!"
#else
// Non-SMP case:
#define NV_SMP_CALL_FUNCTION(func, info)                     \
    ({                                                       \
        func(info);                                          \
        0;                                                   \
     })
#endif

#if defined(NV_ON_EACH_CPU_PRESENT)
#if (NV_ON_EACH_CPU_ARGUMENT_COUNT == 4)
#define NV_ON_EACH_CPU(func, info)                     \
    ({                                                 \
        int __ret = on_each_cpu(func, info, 1, 1);     \
        __ret;                                         \
     })
#elif (NV_ON_EACH_CPU_ARGUMENT_COUNT == 3)
#define NV_ON_EACH_CPU(func, info)                     \
    ({                                                 \
        int __ret = on_each_cpu(func, info, 1);        \
        __ret;                                         \
     })
#else
#error "NV_ON_EACH_CPU_ARGUMENT_COUNT value unrecognized!"
#endif
#elif defined(CONFIG_SMP)
#error "NV_ON_EACH_CPU() undefined (on_each_cpu() unavailable)!"
#else
// Non-SMP case:
#define NV_ON_EACH_CPU(func, info)                     \
    ({                                                 \
        func(info);                                    \
        0;                                             \
     })
#endif

#if defined(NV_GET_NUM_PHYSPAGES_PRESENT)
#define NV_NUM_PHYSPAGES                get_num_physpages()
#else
#define NV_NUM_PHYSPAGES                num_physpages
#endif
#define NV_GET_CURRENT_PROCESS()        current->tgid
#define NV_IN_ATOMIC()                  in_atomic()
#define NV_LOCAL_BH_DISABLE()           local_bh_disable()
#define NV_LOCAL_BH_ENABLE()            local_bh_enable()
#define NV_COPY_TO_USER(to, from, n)    copy_to_user(to, from, n)
#define NV_COPY_FROM_USER(to, from, n)  copy_from_user(to, from, n)

#define NV_IS_SUSER()                   capable(CAP_SYS_ADMIN)
#define NV_PCI_DEVICE_NAME(dev)         ((dev)->pretty_name)
#define NV_CLI()                        local_irq_disable()
#define NV_SAVE_FLAGS(eflags)           local_save_flags(eflags)
#define NV_RESTORE_FLAGS(eflags)        local_irq_restore(eflags)
#define NV_MAY_SLEEP()                  (!irqs_disabled() && !in_interrupt() && !NV_IN_ATOMIC())
#define NV_MODULE_PARAMETER(x)          module_param(x, int, 0)
#define NV_MODULE_STRING_PARAMETER(x)   module_param(x, charp, 0)
#undef  MODULE_PARM

#define NV_NUM_CPUS()                   num_possible_cpus()

static inline dma_addr_t nv_phys_to_dma(struct pci_dev *dev, NvU64 pa)
{
#if defined(NV_PHYS_TO_DMA_PRESENT)
    return phys_to_dma(&dev->dev, pa);
#elif defined(NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL)
    return phys_to_machine(pa);
#else
    return (dma_addr_t)pa;
#endif
}

#define NV_GET_PAGE_STRUCT(phys_page) virt_to_page(__va(phys_page))
#define NV_VMA_PGOFF(vma)             ((vma)->vm_pgoff)
#define NV_VMA_SIZE(vma)              ((vma)->vm_end - (vma)->vm_start)
#define NV_VMA_OFFSET(vma)            (((NvU64)(vma)->vm_pgoff) << PAGE_SHIFT)
#define NV_VMA_PRIVATE(vma)           ((vma)->vm_private_data)
#define NV_VMA_FILE(vma)              ((vma)->vm_file)

#define NV_DEVICE_MINOR_NUMBER(x)     minor((x)->i_rdev)
#define NV_CONTROL_DEVICE_MINOR       255

#define NV_PCI_DISABLE_DEVICE(dev)                            \
    {                                                         \
        NvU16 __cmd[2];                                       \
        pci_read_config_word((dev), PCI_COMMAND, &__cmd[0]);  \
        pci_disable_device(dev);                              \
        pci_read_config_word((dev), PCI_COMMAND, &__cmd[1]);  \
        __cmd[1] |= PCI_COMMAND_MEMORY;                       \
        pci_write_config_word((dev), PCI_COMMAND,             \
                (__cmd[1] | (__cmd[0] & PCI_COMMAND_IO)));    \
    }

#define NV_PCI_RESOURCE_START(dev, bar) pci_resource_start(dev, (bar))
#define NV_PCI_RESOURCE_SIZE(dev, bar)  pci_resource_len(dev, (bar))
#define NV_PCI_RESOURCE_FLAGS(dev, bar) pci_resource_flags(dev, (bar))

#if defined(NVCPU_X86)
#define NV_PCI_RESOURCE_VALID(dev, bar)                                      \
    ((NV_PCI_RESOURCE_START(dev, bar) != 0) &&                               \
     (NV_PCI_RESOURCE_SIZE(dev, bar) != 0) &&                                \
     (!((NV_PCI_RESOURCE_FLAGS(dev, bar) & PCI_BASE_ADDRESS_MEM_TYPE_64) &&  \
       ((NV_PCI_RESOURCE_START(dev, bar) >> PAGE_SHIFT) > 0xfffffULL))))
#else
#define NV_PCI_RESOURCE_VALID(dev, bar)                                      \
    ((NV_PCI_RESOURCE_START(dev, bar) != 0) &&                               \
     (NV_PCI_RESOURCE_SIZE(dev, bar) != 0))
#endif

#if defined(NV_PCI_DOMAIN_NR_PRESENT)
#define NV_PCI_DOMAIN_NUMBER(dev)     (NvU32)pci_domain_nr(dev->bus)
#else
#define NV_PCI_DOMAIN_NUMBER(dev)     (0)
#endif
#define NV_PCI_BUS_NUMBER(dev)        (dev)->bus->number
#define NV_PCI_DEVFN(dev)             (dev)->devfn
#define NV_PCI_SLOT_NUMBER(dev)       PCI_SLOT(NV_PCI_DEVFN(dev))

#if defined(NV_PCI_GET_CLASS_PRESENT)
#define NV_PCI_DEV_PUT(dev)                    pci_dev_put(dev)
#define NV_PCI_GET_DEVICE(vendor,device,from)  pci_get_device(vendor,device,from)
#define NV_PCI_GET_CLASS(class,from)           pci_get_class(class,from)
#else
#define NV_PCI_DEV_PUT(dev)
#define NV_PCI_GET_DEVICE(vendor,device,from)  pci_find_device(vendor,device,from)
#define NV_PCI_GET_CLASS(class,from)           pci_find_class(class,from)
#endif

#if defined(CONFIG_X86_UV) && defined(NV_CONFIG_X86_UV)
#define NV_GET_DOMAIN_BUS_AND_SLOT(domain,bus,devfn)                        \
   ({                                                                       \
        struct pci_dev *__dev = NULL;                                       \
        while ((__dev = NV_PCI_GET_DEVICE(PCI_VENDOR_ID_NVIDIA,             \
                    PCI_ANY_ID, __dev)) != NULL)                            \
        {                                                                   \
            if ((NV_PCI_DOMAIN_NUMBER(__dev) == domain) &&                  \
                (NV_PCI_BUS_NUMBER(__dev) == bus) &&                        \
                (NV_PCI_DEVFN(__dev) == devfn))                             \
            {                                                               \
                break;                                                      \
            }                                                               \
        }                                                                   \
        if (__dev == NULL)                                                  \
        {                                                                   \
            while ((__dev = NV_PCI_GET_CLASS((PCI_CLASS_BRIDGE_HOST << 8),  \
                        __dev)) != NULL)                                    \
            {                                                               \
                if ((NV_PCI_DOMAIN_NUMBER(__dev) == domain) &&              \
                    (NV_PCI_BUS_NUMBER(__dev) == bus) &&                    \
                    (NV_PCI_DEVFN(__dev) == devfn))                         \
                {                                                           \
                    break;                                                  \
                }                                                           \
            }                                                               \
        }                                                                   \
        if (__dev == NULL)                                                  \
        {                                                                   \
            while ((__dev = NV_PCI_GET_CLASS((PCI_CLASS_BRIDGE_PCI << 8),   \
                        __dev)) != NULL)                                    \
            {                                                               \
                if ((NV_PCI_DOMAIN_NUMBER(__dev) == domain) &&              \
                    (NV_PCI_BUS_NUMBER(__dev) == bus) &&                    \
                    (NV_PCI_DEVFN(__dev) == devfn))                         \
                {                                                           \
                    break;                                                  \
                }                                                           \
            }                                                               \
        }                                                                   \
        if (__dev == NULL)                                                  \
        {                                                                   \
            while ((__dev = NV_PCI_GET_DEVICE(PCI_ANY_ID, PCI_ANY_ID,       \
                            __dev)) != NULL)                                \
            {                                                               \
                if ((NV_PCI_DOMAIN_NUMBER(__dev) == domain) &&              \
                    (NV_PCI_BUS_NUMBER(__dev) == bus) &&                    \
                    (NV_PCI_DEVFN(__dev) == devfn))                         \
                {                                                           \
                    break;                                                  \
                }                                                           \
            }                                                               \
        }                                                                   \
        __dev;                                                              \
    })
#elif defined(NV_PCI_GET_DOMAIN_BUS_AND_SLOT_PRESENT)
#define NV_GET_DOMAIN_BUS_AND_SLOT(domain,bus, devfn) \
    pci_get_domain_bus_and_slot(domain, bus, devfn)
#elif defined(NV_PCI_GET_CLASS_PRESENT)
#define NV_GET_DOMAIN_BUS_AND_SLOT(domain,bus,devfn)               \
   ({                                                              \
        struct pci_dev *__dev = NULL;                              \
        while ((__dev = NV_PCI_GET_DEVICE(PCI_ANY_ID, PCI_ANY_ID,  \
                    __dev)) != NULL)                               \
        {                                                          \
            if ((NV_PCI_DOMAIN_NUMBER(__dev) == domain) &&         \
                (NV_PCI_BUS_NUMBER(__dev) == bus) &&               \
                (NV_PCI_DEVFN(__dev) == devfn))                    \
            {                                                      \
                break;                                             \
            }                                                      \
        }                                                          \
        __dev;                                                     \
    })
#else
#define NV_GET_DOMAIN_BUS_AND_SLOT(domain,bus,devfn) pci_find_slot(bus,devfn)
#endif

#if defined(NV_PCI_STOP_AND_REMOVE_BUS_DEVICE_PRESENT)  // introduced in 3.4.9
#define NV_PCI_STOP_AND_REMOVE_BUS_DEVICE(dev) pci_stop_and_remove_bus_device(dev)
#elif defined(NV_PCI_REMOVE_BUS_DEVICE_PRESENT) // introduced in 2.6
#define NV_PCI_STOP_AND_REMOVE_BUS_DEVICE(dev) pci_remove_bus_device(dev)
#endif

#define NV_PRINT_AT(nv_debug_level,at)                                           \
    {                                                                            \
        nv_printf(nv_debug_level,                                                \
            "NVRM: VM: %s:%d: 0x%p, %d page(s), count = %d, flags = 0x%08x, "    \
            "page_table = 0x%p\n",  __FUNCTION__, __LINE__, at,                  \
            at->num_pages, NV_ATOMIC_READ(at->usage_count),                      \
            at->flags, at->page_table);                                          \
    }

#define NV_PRINT_VMA(nv_debug_level,vma)                                                 \
    {                                                                                    \
        nv_printf(nv_debug_level,                                                        \
            "NVRM: VM: %s:%d: 0x%lx - 0x%lx, 0x%08x bytes @ 0x%016llx, 0x%p, 0x%p\n",    \
            __FUNCTION__, __LINE__, vma->vm_start, vma->vm_end, NV_VMA_SIZE(vma),        \
            NV_VMA_OFFSET(vma), NV_VMA_PRIVATE(vma), NV_VMA_FILE(vma));                  \
    }

/*
 * On Linux 2.6, we support both APM and ACPI power management. On Linux
 * 2.4, we support APM, only. ACPI support has been back-ported to the
 * Linux 2.4 kernel, but the Linux 2.4 driver model is not sufficient for
 * full ACPI support: it may work with some systems, but not reliably
 * enough for us to officially support this configuration.
 *
 * We support two Linux kernel power managment interfaces: the original
 * pm_register()/pm_unregister() on Linux 2.4 and the device driver model
 * backed PCI driver power management callbacks introduced with Linux
 * 2.6.
 *
 * The code below determines which interface to support on this kernel
 * version, if any; if built for Linux 2.6, it will also determine if the
 * kernel comes with ACPI or APM power management support.
 */
#if defined(CONFIG_PM)
#define NV_PM_SUPPORT_DEVICE_DRIVER_MODEL
#if (defined(CONFIG_APM) || defined(CONFIG_APM_MODULE)) && !defined(CONFIG_ACPI)
#define NV_PM_SUPPORT_NEW_STYLE_APM
#endif
#endif

/*
 * On Linux 2.6 kernels >= 2.6.11, the PCI subsystem provides a new
 * interface that allows PCI drivers to determine the correct power state
 * for a given system power state; our suspend/resume callbacks now use
 * this interface and operate on PCI power state defines.
 *
 * Define these new PCI power state #define's here for compatibility with
 * older Linux 2.6 kernels.
 */
#if !defined(PCI_D0)
#define PCI_D0 PM_SUSPEND_ON
#define PCI_D3hot PM_SUSPEND_MEM
#endif

#if !defined(NV_PM_MESSAGE_T_PRESENT)
typedef u32 pm_message_t;
#endif

#ifndef minor
# define minor(x) MINOR(x)
#endif

#if defined(cpu_relax)
#define NV_CPU_RELAX() cpu_relax()
#else
#define NV_CPU_RELAX() barrier()
#endif

#ifndef IRQ_RETVAL
typedef void irqreturn_t;
#define IRQ_RETVAL(a)
#endif

#ifndef NV_REQUEST_THREADED_IRQ_PRESENT
#define IRQ_WAKE_THREAD (1 << 1)
#endif

#if !defined(PCI_COMMAND_SERR)
#define PCI_COMMAND_SERR            0x100
#endif
#if !defined(PCI_COMMAND_INTX_DISABLE)
#define PCI_COMMAND_INTX_DISABLE    0x400
#endif

#ifndef PCI_CAP_ID_EXP
#define PCI_CAP_ID_EXP 0x10
#endif

/*
 * On Linux on PPC64LE enable basic support for Linux PCI error recovery (see
 * Documentation/PCI/pci-error-recovery.txt). Currently RM only supports error
 * notification and data collection, not actual recovery of the device.
 */
#if defined(NVCPU_PPC64LE) && defined(CONFIG_EEH)
#include <asm/eeh.h>
#define NV_PCI_ERROR_RECOVERY
#endif

/*
 * Early 2.6 kernels have acquire_console_sem, but from 2.6.38+ it was
 * renamed to console_lock.
 */
#if defined(NV_ACQUIRE_CONSOLE_SEM_PRESENT)
#define NV_ACQUIRE_CONSOLE_SEM() acquire_console_sem()
#define NV_RELEASE_CONSOLE_SEM() release_console_sem()
#elif defined(NV_CONSOLE_LOCK_PRESENT)
#define NV_ACQUIRE_CONSOLE_SEM() console_lock()
#define NV_RELEASE_CONSOLE_SEM() console_unlock()
#else
#error "console lock api unrecognized!."
#endif

/*
 * If the host OS has page sizes larger than 4KB, we may have a security
 * problem. Registers are typically grouped in 4KB pages, but if there are
 * larger pages, then the smallest userspace mapping possible (e.g., a page)
 * may give more access than intended to the user.
 *
 * The kernel may have a workaround for this, by providing a method to isolate
 * a single 4K page in a given mapping.
 */
#if (PAGE_SIZE > NV_RM_PAGE_SIZE)
#define NV_4K_PAGE_ISOLATION_REQUIRED(addr, size)                             \
    (((size) <= NV_RM_PAGE_SIZE) &&                                           \
        (((addr) >> NV_RM_PAGE_SHIFT) ==                                      \
            (((addr) + (size) - 1) >> NV_RM_PAGE_SHIFT)))
#if defined(NVCPU_PPC64LE) && defined(NV_PAGE_4K_PFN)
#define NV_4K_PAGE_ISOLATION_PRESENT
#define NV_4K_PAGE_ISOLATION_MMAP_ADDR(addr)                                  \
    ((NvP64)((void*)(((addr) >> NV_RM_PAGE_SHIFT) << PAGE_SHIFT)))
#define NV_4K_PAGE_ISOLATION_MMAP_LEN(size)     PAGE_SIZE
#define NV_4K_PAGE_ISOLATION_ACCESS_START(addr)                               \
    ((NvP64)((void*)((addr) & ~NV_RM_PAGE_MASK)))
#define NV_4K_PAGE_ISOLATION_ACCESS_LEN(addr, size)                           \
    ((((addr) & NV_RM_PAGE_MASK) + size + NV_RM_PAGE_MASK) &                  \
        ~NV_RM_PAGE_MASK)
#define NV_PROT_4K_PAGE_ISOLATION NV_PAGE_4K_PFN
#else
#error "The kernel provides no known mechanism to isolate 4KB page mappings!"
#endif
#endif

#if defined(NV_VM_INSERT_PAGE_PRESENT)
#define NV_VM_INSERT_PAGE(vma, addr, page) \
    vm_insert_page(vma, addr, page)
#endif

static inline int nv_remap_page_range(struct vm_area_struct *vma,
    unsigned long virt_addr, NvU64 phys_addr, NvU64 size, pgprot_t prot)
{
    int ret = -1;
#if defined(NV_REMAP_PFN_RANGE_PRESENT)
#if defined(NV_4K_PAGE_ISOLATION_PRESENT) && defined(NV_PROT_4K_PAGE_ISOLATION)
    if ((size == PAGE_SIZE) &&
        ((pgprot_val(prot) & NV_PROT_4K_PAGE_ISOLATION) != 0))
    {
        /*
         * remap_4k_pfn() hardcodes the length to a single OS page, and checks
         * whether applying the page isolation workaround will cause PTE
         * corruption (in which case it will fail, and this is an unsupported
         * configuration).
         */
#if defined(NV_HASH__REMAP_4K_PFN_PRESENT)
        ret = hash__remap_4k_pfn(vma, virt_addr, (phys_addr >> PAGE_SHIFT), prot);
#else
        ret = remap_4k_pfn(vma, virt_addr, (phys_addr >> PAGE_SHIFT), prot);
#endif
    }
    else
#endif
    {
        ret = remap_pfn_range(vma, virt_addr, (phys_addr >> PAGE_SHIFT), size,
            prot);
    }
#else
    /*
     * remap_page_range() was replaced by remap_pfn_range() in 2.6.10, but we
     * still support 2.6.9.
     */
    ret = remap_page_range(vma, virt_addr, phys_addr, size, prot);
#endif /* defined(NV_REMAP_PFN_RANGE_PRESENT) */
    return ret;
}

static inline int nv_io_remap_page_range(struct vm_area_struct *vma,
    NvU64 phys_addr, NvU64 size, NvU32 extra_prot)
{
    int ret = -1;
#if !defined(NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL)
    ret = nv_remap_page_range(vma, vma->vm_start, phys_addr, size,
        __pgprot(pgprot_val(vma->vm_page_prot) | extra_prot));
#else
    ret = io_remap_pfn_range(vma, vma->vm_start, (phys_addr >> PAGE_SHIFT),
        size, __pgprot(pgprot_val(vma->vm_page_prot) | extra_prot));
#endif
    return ret;
}

#define NV_PAGE_MASK    (NvU64)(long)PAGE_MASK

extern void *nvidia_stack_t_cache;

// Changed in 2.6.23 via commit 20c2df83d25c6a95affe6157a4c9cac4cf5ffaac
#if (NV_KMEM_CACHE_CREATE_ARGUMENT_COUNT == 5)
#define NV_KMEM_CACHE_CREATE_FULL(name, size, align, flags, ctor) \
    kmem_cache_create(name, size, align, flags, ctor)

#else
#define NV_KMEM_CACHE_CREATE_FULL(name, size, align, flags, ctor) \
    kmem_cache_create(name, size, align, flags, ctor, NULL)
#endif

#define NV_KMEM_CACHE_CREATE(name, type)    \
    NV_KMEM_CACHE_CREATE_FULL(name, sizeof(type), 0, 0, NULL)

#define NV_KMEM_CACHE_DESTROY(kmem_cache)   \
    kmem_cache_destroy(kmem_cache)

#define NV_KMEM_CACHE_ALLOC(kmem_cache)     \
    kmem_cache_alloc(kmem_cache, GFP_KERNEL)
#define NV_KMEM_CACHE_FREE(ptr, kmem_cache) \
    kmem_cache_free(kmem_cache, ptr)

static inline int nv_kmem_cache_alloc_stack(nvidia_stack_t **stack)
{
    nvidia_stack_t *sp = NULL;
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    sp = NV_KMEM_CACHE_ALLOC(nvidia_stack_t_cache);
    if (sp == NULL)
        return -ENOMEM;
    sp->size = sizeof(sp->stack);
    sp->top = sp->stack + sp->size;
#endif
    *stack = sp;
    return 0;
}

static inline void nv_kmem_cache_free_stack(nvidia_stack_t *stack)
{
#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    NV_KMEM_CACHE_FREE(stack, nvidia_stack_t_cache);
#endif
}

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
/*
 * RAM is cached on Linux by default, we can assume there's
 * nothing to be done here. This is not the case for the
 * other memory spaces: we will have made an attempt to add
 * a WC MTRR for the frame buffer.
 *
 * If a WC MTRR is present, we can't satisfy the WB mapping
 * attempt here, since the achievable effective memory
 * types in that case are WC and UC, if not it's typically
 * UC (MTRRdefType is UC); we could only satisfy WB mapping
 * requests with a WB MTRR.
 */
#define NV_ALLOW_CACHING(mt)            ((mt) == NV_MEMORY_TYPE_SYSTEM)
#else
#define NV_ALLOW_CACHING(mt)            ((mt) != NV_MEMORY_TYPE_REGISTERS)
#endif

typedef struct nvidia_pte_s {
    NvU64           phys_addr;
    unsigned long   virt_addr;
    NvU64           dma_addr;
#ifdef CONFIG_XEN
    unsigned int    guest_pfn;
#endif
    unsigned int    page_count;
} nvidia_pte_t;

typedef struct nv_alloc_s {
    struct nv_alloc_s *next;
    struct pci_dev    *dev;
    atomic_t       usage_count;
    unsigned int   flags;
    unsigned int   num_pages;
    unsigned int   order;
    unsigned int   size;
    nvidia_pte_t **page_table;          /* list of physical pages allocated */
    unsigned int   pid;
    struct page  **user_pages;
    NvU64         guest_id;             /* id of guest VM */
} nv_alloc_t;

#define NV_ALLOC_TYPE_PCI               (1<<0)
#define NV_ALLOC_TYPE_CONTIG            (1<<2)
#define NV_ALLOC_TYPE_GUEST             (1<<3)
#define NV_ALLOC_TYPE_ZEROED            (1<<4)
#define NV_ALLOC_TYPE_ALIASED           (1<<5)
#define NV_ALLOC_TYPE_USER              (1<<6)
#define NV_ALLOC_TYPE_NODE0             (1<<7)
#define NV_ALLOC_TYPE_PEER_IO           (1<<8)

#define NV_ALLOC_MAPPING_SHIFT      16
#define NV_ALLOC_MAPPING(flags)     (((flags)>>NV_ALLOC_MAPPING_SHIFT)&0xff)
#define NV_ALLOC_ENC_MAPPING(flags) ((flags)<<NV_ALLOC_MAPPING_SHIFT)

#define NV_ALLOC_MAPPING_CACHED(flags)        (NV_ALLOC_MAPPING(flags) == NV_MEMORY_CACHED)
#define NV_ALLOC_MAPPING_UNCACHED(flags)      (NV_ALLOC_MAPPING(flags) == NV_MEMORY_UNCACHED)
#define NV_ALLOC_MAPPING_WRITECOMBINED(flags) (NV_ALLOC_MAPPING(flags) == NV_MEMORY_WRITECOMBINED)

#define NV_ALLOC_MAPPING_CONTIG(flags)            ((flags) & NV_ALLOC_TYPE_CONTIG)
#define NV_ALLOC_MAPPING_GUEST(flags)             ((flags) & NV_ALLOC_TYPE_GUEST)
#define NV_ALLOC_MAPPING_ALIASED(flags)           ((flags) & NV_ALLOC_TYPE_ALIASED)
#define NV_ALLOC_MAPPING_USER(flags)              ((flags) & NV_ALLOC_TYPE_USER)
#define NV_ALLOC_MAPPING_PEER_IO(flags)           ((flags) & NV_ALLOC_TYPE_PEER_IO)

static inline NvU32 nv_alloc_init_flags(int cached, int contiguous, int zeroed)
{
    NvU32 flags = NV_ALLOC_ENC_MAPPING(cached);
    flags |= NV_ALLOC_TYPE_PCI;
    if (contiguous)
        flags |= NV_ALLOC_TYPE_CONTIG;
    if (zeroed)
        flags |= NV_ALLOC_TYPE_ZEROED;
#if defined(NVCPU_FAMILY_ARM)
    if (!NV_ALLOC_MAPPING_CACHED(flags))
        flags |= NV_ALLOC_TYPE_ALIASED;
#endif
    return flags;
}

static inline NvBool nv_dma_maps_swiotlb(struct pci_dev *dev)
{
    NvBool swiotlb_in_use = NV_FALSE;
#if defined(CONFIG_SWIOTLB)
  #if defined(NV_DMA_OPS_PRESENT) || defined(NV_GET_DMA_OPS_PRESENT)
    /*
     * We only use the 'dma_ops' symbol on older x86_64 kernels; later kernels,
     * including those for other architectures, have converged on the
     * get_dma_ops() interface.
     */
    #if defined(NV_GET_DMA_OPS_PRESENT)
      #if defined(NV_DMA_MAP_OPS_PRESENT)
    const struct dma_map_ops *ops = get_dma_ops(&dev->dev);
      #else
    const struct dma_mapping_ops *ops = get_dma_ops(&dev->dev);
      #endif
    #else
    const struct dma_mapping_ops *ops = dma_ops;
    #endif
    #if defined(NV_DMA_MAP_OPS_PRESENT)
    /*
     * The switch from dma_mapping_ops -> dma_map_ops coincided with the
     * switch from swiotlb_map_sg -> swiotlb_map_sg_attrs.
     */
      #if defined(NVCPU_AARCH64) && \
          defined(NV_NONCOHERENT_SWIOTLB_DMA_OPS_PRESENT)
    /* AArch64 exports these symbols directly */
    swiotlb_in_use = ((ops == &noncoherent_swiotlb_dma_ops) ||
                      (ops == &coherent_swiotlb_dma_ops));
      #else
    swiotlb_in_use = (ops->map_sg == swiotlb_map_sg_attrs);
      #endif
    #else
    swiotlb_in_use = (ops->map_sg == swiotlb_map_sg);
    #endif
  #elif defined(NVCPU_X86_64)
    /*
     * Fallback for old 2.6 kernels - if the DMA operations infrastructure
     * isn't in place, use the swiotlb flag. Before dma_ops was added, this
     * flag used to be exported. It still exists in modern kernels but is no
     * longer exported.
     */
    swiotlb_in_use = (swiotlb == 1);
  #endif
#endif

    return swiotlb_in_use;
}

/*
 * TODO: Bug 1522381 will allow us to move these mapping relationships into
 *       common code.
 */

/*
 * Bug 1606851: the Linux kernel scatterlist code doesn't work for regions
 * greater than or equal to 4GB, due to regular use of unsigned int
 * throughout. So we need to split our mappings into 4GB-minus-1-page-or-less
 * chunks and manage them separately.
 */
typedef struct nv_dma_submap_s {
    NvU32 page_count;
    NvU32 sg_map_count;
#if defined(NV_SG_TABLE_PRESENT)
    struct sg_table sgt;
#else
    struct scatterlist *sgl;
#endif
} nv_dma_submap_t;

typedef struct nv_dma_map_s {
    struct page **pages;
    NvU64 page_count;
    NvBool contiguous;

    union
    {
        struct
        {
            NvU32 submap_count;
            nv_dma_submap_t *submaps;
        } discontig;

        struct
        {
            NvU64 dma_addr;
        } contig;
    } mapping;

    struct pci_dev *dev;
} nv_dma_map_t;

#define NV_FOR_EACH_DMA_SUBMAP(dm, sm, i)                                     \
    for (i = 0, sm = &dm->mapping.discontig.submaps[0];                       \
         i < dm->mapping.discontig.submap_count;                              \
         i++, sm = &dm->mapping.discontig.submaps[i])

#define NV_DMA_SUBMAP_MAX_PAGES           ((NvU32)(NV_U32_MAX >> PAGE_SHIFT))
#define NV_DMA_SUBMAP_IDX_TO_PAGE_IDX(s)  (s * NV_DMA_SUBMAP_MAX_PAGES)

#if defined(NV_SG_TABLE_PRESENT)
#define NV_DMA_SUBMAP_SCATTERLIST(sm)           sm->sgt.sgl
#define NV_DMA_SUBMAP_SCATTERLIST_LENGTH(sm)    sm->sgt.orig_nents
#else
#define NV_DMA_SUBMAP_SCATTERLIST(sm)           sm->sgl
#define NV_DMA_SUBMAP_SCATTERLIST_LENGTH(sm)    sm->page_count
#endif

#if defined(for_each_sg)
    #define NV_FOR_EACH_DMA_SUBMAP_SG(sm, sg, i)                              \
        for_each_sg(NV_DMA_SUBMAP_SCATTERLIST((sm)), (sg),                    \
            (sm)->sg_map_count, (i))
#else
    #define NV_FOR_EACH_DMA_SUBMAP_SG(sm, sg, i)                              \
        for ((i) = 0, (sg) = &NV_DMA_SUBMAP_SCATTERLIST(sm)[0];               \
             (i) < (sm)->sg_map_count;                                        \
             (sg) = &NV_DMA_SUBMAP_SCATTERLIST(sm)[++i])
#endif

/*
 * DO NOT use sg_alloc_table_from_pages on Xen Server, even if it's available.
 * This will glom multiple pages into a single sg element, which
 * xen_swiotlb_map_sg_attrs may try to route to the SWIOTLB. We must only use
 * single-page sg elements on Xen Server.
 */
#if defined(NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT) && \
    !defined(NV_DOM0_KERNEL_PRESENT)
    #define NV_ALLOC_DMA_SUBMAP_SCATTERLIST(dm, sm, i)                        \
        ((sg_alloc_table_from_pages(&sm->sgt,                                 \
            &dm->pages[NV_DMA_SUBMAP_IDX_TO_PAGE_IDX(i)],                     \
            sm->page_count, 0,                                                \
            sm->page_count * PAGE_SIZE, NV_GFP_KERNEL) == 0) ? NV_OK :        \
                NV_ERR_OPERATING_SYSTEM)

    #define NV_FREE_DMA_SUBMAP_SCATTERLIST(sm)  sg_free_table(&sm->sgt)

#else /* !defined(NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT) ||
          defined(NV_DOM0_KERNEL_PRESENT) */
    #if defined(NV_SG_TABLE_PRESENT)
        #if defined(NV_SG_ALLOC_TABLE_PRESENT)
            #define NV_ALLOC_DMA_SUBMAP_SCATTERLIST(dm, sm, i)                \
                ((sg_alloc_table(&sm->sgt, sm->page_count, NV_GFP_KERNEL)) == \
                    0 ? NV_OK : NV_ERR_OPERATING_SYSTEM)

            #define NV_FREE_DMA_SUBMAP_SCATTERLIST(sm)  sg_free_table(&sm->sgt)

        #else
            #error "No known allocation function for sg_table present!"
        #endif
    #else /* !defined(NV_SG_TABLE_PRESENT) */
        #define NV_ALLOC_DMA_SUBMAP_SCATTERLIST(dm, sm, i)                    \
            os_alloc_mem((void **)&sm->sgl,                                    \
                sm->page_count * sizeof(struct scatterlist))

        #define NV_FREE_DMA_SUBMAP_SCATTERLIST(sm)  os_free_mem(sm->sgl)

    #endif /* defined(NV_SG_TABLE_PRESENT) */

#endif /* defined(NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT) &&
          !defined(NV_DOM0_KERNEL_PRESENT) */

typedef struct work_struct nv_task_t;

typedef struct nv_ibmnpu_info nv_ibmnpu_info_t;

typedef struct nv_work_s {
    nv_task_t task;
    void *data;
} nv_work_t;

#define NV_WORKQUEUE_SCHEDULE(work) schedule_work(work)
#define NV_WORKQUEUE_FLUSH()                           \
    flush_scheduled_work();
#if (NV_INIT_WORK_ARGUMENT_COUNT == 2)
#define NV_WORKQUEUE_INIT(tq,handler,data)             \
    {                                                  \
        struct work_struct __work =                    \
            __WORK_INITIALIZER(*(tq), handler);        \
        *(tq) = __work;                                \
    }
#define NV_WORKQUEUE_DATA_T nv_task_t
#define NV_WORKQUEUE_UNPACK_DATA(tq)                   \
    container_of((tq), nv_work_t, task)
#elif (NV_INIT_WORK_ARGUMENT_COUNT == 3)
#define NV_WORKQUEUE_INIT(tq,handler,data)             \
    {                                                  \
        struct work_struct __work =                    \
            __WORK_INITIALIZER(*(tq), handler, data);  \
        *(tq) = __work;                                \
    }
#define NV_WORKQUEUE_DATA_T void
#define NV_WORKQUEUE_UNPACK_DATA(tq) (nv_work_t *)(tq)
#else
#error "NV_INIT_WORK_ARGUMENT_COUNT value unrecognized!"
#endif

#define NV_MAX_REGISTRY_KEYS_LENGTH   512

typedef enum
{
    NV_DEV_STACK_TIMER,
    NV_DEV_STACK_ISR,
    NV_DEV_STACK_ISR_BH,
    NV_DEV_STACK_ISR_BH_UNLOCKED,
    NV_DEV_STACK_PCI_CFGCHK,
    NV_DEV_STACK_COUNT
} nvidia_linux_dev_stack_t;

/* linux-specific version of old nv_state_t */
/* this is a general os-specific state structure. the first element *must* be
   the general state structure, for the generic unix-based code */
typedef struct nv_linux_state_s {
    nv_state_t nv_state;
    atomic_t usage_count;

    struct pci_dev *dev;

    /* IBM-NPU info associated with this GPU */
    nv_ibmnpu_info_t *npu;

    nvidia_stack_t *sp[NV_DEV_STACK_COUNT];

    char registry_keys[NV_MAX_REGISTRY_KEYS_LENGTH];

    /* keep track of any pending bottom halfes */
    struct tasklet_struct tasklet;
    nv_work_t work;

    /* get a timer callback every second */
    struct nv_timer rc_timer;

    /* lock for linux-specific data, not used by core rm */
    struct semaphore ldata_lock;

    /* proc directory information */
    struct proc_dir_entry *proc_dir;

    NvU32 minor_num;
    struct nv_linux_state_s *next;

    /* DRM private information */
    struct drm_device *drm;

    /* kthread based bottom half servicing queue and elements */
    nv_kthread_q_t bottom_half_q;
    nv_kthread_q_item_t bottom_half_q_item;

    /* Lock for unlocked bottom half protecting common allocated stack */
    void *isr_bh_unlocked_mutex;

    NvBool tce_bypass_enabled;

    NvU64 numa_memblock_size;

    struct {
        struct backlight_device *dev;
        NvU32 displayId;
    } backlight;
} nv_linux_state_t;

extern nv_linux_state_t *nv_linux_devices;

/*
 * Macros to protect operations on nv_linux_devices list
 * Lock acquisition order while using the nv_linux_devices list
 * 1. LOCK_NV_LINUX_DEVICES()
 * 2. Traverse the list
 *    If the list is traversed to search for an element say nvl,
 *    acquire the nvl->ldata_lock before step 3
 * 3. UNLOCK_NV_LINUX_DEVICES()
 * 4. Release nvl->ldata_lock after any read/write access to the
 *    nvl element is complete
 */
extern struct semaphore nv_linux_devices_lock;
#define LOCK_NV_LINUX_DEVICES()     down(&nv_linux_devices_lock)
#define UNLOCK_NV_LINUX_DEVICES()   up(&nv_linux_devices_lock)

#if defined(NV_LINUX_ACPI_EVENTS_SUPPORTED)
/*
 * acpi data storage structure
 *
 * This structure retains the pointer to the device,
 * and any other baggage we want to carry along
 *
 */
#define NV_MAXNUM_DISPLAY_DEVICES 8

typedef struct
{
    acpi_handle dev_handle;
    int dev_id;
} nv_video_t;

typedef struct
{
    nvidia_stack_t *sp;
    struct acpi_device *device;

    nv_video_t pNvVideo[NV_MAXNUM_DISPLAY_DEVICES];

    int notify_handler_installed;
    int default_display_mask;
} nv_acpi_t;

#endif

/*
 * file-private data
 * hide a pointer to our data structures in a file-private ptr
 * there are times we need to grab this data back from the file
 * data structure..
 */

typedef struct nvidia_event
{
    struct nvidia_event *next;
    nv_event_t event;
} nvidia_event_t;

typedef enum
{
    NV_FOPS_STACK_INDEX_MMAP,
    NV_FOPS_STACK_INDEX_IOCTL,
    NV_FOPS_STACK_INDEX_PROCFS,
    NV_FOPS_STACK_INDEX_COUNT
} nvidia_entry_point_index_t;

typedef struct
{
    nvidia_stack_t *sp;
    nvidia_stack_t *fops_sp[NV_FOPS_STACK_INDEX_COUNT];
    struct semaphore fops_sp_lock[NV_FOPS_STACK_INDEX_COUNT];
    nv_alloc_t *free_list;
    void *nvptr;
    void *proc_data;
    void *data;
    nvidia_event_t *event_head, *event_tail;
    // This field is only applicable for the control device
    // it should not be used for the per-GPU special files.
    nv_fd_memdesc_t fd_memdesc;
    int event_pending;
    nv_spinlock_t fp_lock;
    wait_queue_head_t waitqueue;
    off_t off;
    NvU32 minor_num;
    NvHandle hExportedRmObject;
    NvU32 *attached_gpus;
    size_t num_attached_gpus;
    nv_alloc_mapping_context_t mmap_context;
} nv_file_private_t;

#define NV_SET_FILE_PRIVATE(filep,data) ((filep)->private_data = (data))
#define NV_GET_FILE_PRIVATE(filep) ((nv_file_private_t *)(filep)->private_data)

/* for the card devices */
#define NV_GET_NVL_FROM_FILEP(filep)    (NV_GET_FILE_PRIVATE(filep)->nvptr)
#define NV_GET_NVL_FROM_NV_STATE(nv)    ((nv_linux_state_t *)nv->os_state)

#define NV_STATE_PTR(nvl)   &(((nv_linux_state_t *)(nvl))->nv_state)


#define NV_ATOMIC_READ(data)            atomic_read(&(data))
#define NV_ATOMIC_SET(data,val)         atomic_set(&(data), (val))
#define NV_ATOMIC_INC(data)             atomic_inc(&(data))
#define NV_ATOMIC_DEC(data)             atomic_dec(&(data))
#define NV_ATOMIC_DEC_AND_TEST(data)    atomic_dec_and_test(&(data))

#if defined(NV_ATOMIC_LONG_PRESENT)
    typedef atomic_long_t nv_atomic_long_t;
    #define NV_ATOMIC_LONG_READ(data)           atomic_long_read(&(data))
    #define NV_ATOMIC_LONG_SET(data,val)        atomic_long_set(&(data), (val))
    #define NV_ATOMIC_LONG_INC(data)            atomic_long_inc(&(data))
    #define NV_ATOMIC_LONG_DEC(data)            atomic_long_dec(&(data))
#else
    /* atomic_long_t is not present, so we have to use the 32- or 64-bit native
       type, which is guaranteed to be present */
    #if BITS_PER_LONG == 64
        typedef atomic64_t nv_atomic_long_t;
        #define NV_ATOMIC_LONG_READ(data)       atomic64_read(&(data))
        #define NV_ATOMIC_LONG_SET(data,val)    atomic64_set(&(data), (val))
        #define NV_ATOMIC_LONG_INC(data)        atomic64_inc(&(data))
        #define NV_ATOMIC_LONG_DEC(data)        atomic64_dec(&(data))
    #else
        /* BITS_PER_LONG != 64 */
        typedef atomic_t nv_atomic_long_t;
        #define NV_ATOMIC_LONG_READ(data)       atomic_read(&(data))
        #define NV_ATOMIC_LONG_SET(data,val)    atomic_set(&(data), (val))
        #define NV_ATOMIC_LONG_INC(data)        atomic_inc(&(data))
        #define NV_ATOMIC_LONG_DEC(data)        atomic_dec(&(data))
    #endif
#endif /* NV_ATOMIC_LONG_PRESENT */

#if (defined(CONFIG_X86_LOCAL_APIC) || defined(NVCPU_FAMILY_ARM) || \
     defined(NVCPU_PPC64LE)) && \
    (defined(CONFIG_PCI_MSI) || defined(CONFIG_PCI_USE_VECTOR))
#define NV_LINUX_PCIE_MSI_SUPPORTED
#endif

#if !defined(NV_LINUX_PCIE_MSI_SUPPORTED) || !defined(CONFIG_PCI_MSI)
#define NV_PCI_DISABLE_MSI(dev)
#else
#define NV_PCI_DISABLE_MSI(dev)         pci_disable_msi(dev)
#endif

#if (NV_PCI_SAVE_STATE_ARGUMENT_COUNT == 2)
/*
 * On Linux 2.6.9, pci_(save|restore)_state() requires a storage buffer to be
 * passed in by the caller. This went away in 2.6.10, so until we can drop
 * this support, use the original config space storage buffer for this call.
 */
static inline void nv_pci_save_state(struct pci_dev *dev)
{
    nv_linux_state_t *nvl = pci_get_drvdata(dev);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    pci_save_state(dev, &nv->pci_cfg_space[0]);
}

static inline void nv_pci_restore_state(struct pci_dev *dev)
{
    nv_linux_state_t *nvl = pci_get_drvdata(dev);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    pci_restore_state(dev, &nv->pci_cfg_space[0]);
}
#else
#define nv_pci_save_state(dev)      pci_save_state(dev)
#define nv_pci_restore_state(dev)   pci_restore_state(dev)
#endif

#define NV_PCIE_CFG_MAX_OFFSET 0x1000

#if defined(NVCPU_PPC64LE)
/*
 * It is OK to stub this for PowerPC platforms because the conditions
 * below with X servers does not apply to PPC.
 */
#define NV_CHECK_PCI_CONFIG_SPACE(sp,nv,cb,as,mb) 
#else
/*
 * Verify that access to PCI configuration space wasn't modified
 * by a third party.  Unfortunately, some X servers disable
 * memory access in PCI configuration space at various times (such
 * as when restoring initial PCI configuration space settings
 * during VT switches or when driving multiple GPUs), which may
 * cause BAR[n] reads/writes to fail and/or inhibit GPU initiator
 * functionality.
 */
#define NV_CHECK_PCI_CONFIG_SPACE(sp,nv,cb,as,mb)                   \
    {                                                               \
        if (!NV_IS_GVI_DEVICE(nv) &&                                \
            (((nv)->flags & NV_FLAG_SKIP_CFG_CHECK) == 0) &&        \
            (((nv)->flags & NV_FLAG_CONTROL) == 0))                 \
        {                                                           \
            if ((nv)->flags & NV_FLAG_USE_BAR0_CFG)                 \
                rm_check_pci_config_space(sp, nv, cb, as, mb);      \
            else                                                    \
                nv_check_pci_config_space(nv, cb);                  \
        }                                                           \
    }

#endif

extern int nv_update_memory_types;

extern NvU32 NVreg_EnableUserNUMAManagement;

extern NvU32 NVreg_EnableIBMNPURelaxedOrderingMode;

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
/*
 * On Linux/x86-64 (and recent Linux/x86) kernels, the PAGE_KERNEL
 * and PAGE_KERNEL_NOCACHE protection bit masks include _PAGE_NX
 * to indicate that the no-execute protection page feature is used
 * for the page in question.
 *
 * We need to be careful to mask out _PAGE_NX when the host system
 * doesn't support this feature or when it's disabled: the kernel
 * may not do this in its implementation of the change_page_attr()
 * interface.
 */
#ifndef X86_FEATURE_NX
#define X86_FEATURE_NX (1*32+20)
#endif
#ifndef boot_cpu_has
#define boot_cpu_has(x) test_bit(x, boot_cpu_data.x86_capability)
#endif
#ifndef MSR_EFER
#define MSR_EFER 0xc0000080
#endif
#ifndef EFER_NX
#define EFER_NX (1 << 11)
#endif
#ifndef _PAGE_NX
#define _PAGE_NX ((NvU64)1 << 63)
#endif
extern NvU64 __nv_supported_pte_mask;
#endif

#include "nv-proto.h"

extern nv_pci_info_t nv_assign_gpu_pci_info[NV_MAX_DEVICES];
extern NvU32 nv_assign_gpu_count;

#define NV_IS_ASSIGN_GPU_PCI_INFO_SPECIFIED()     \
    (!((nv_assign_gpu_pci_info[0].domain == 0) && \
       (nv_assign_gpu_pci_info[0].bus == 0) &&    \
       (nv_assign_gpu_pci_info[0].slot == 0)))

#if defined(NV_FILE_HAS_INODE)
#define NV_FILE_INODE(file) (file)->f_inode
#else
#define NV_FILE_INODE(file) (file)->f_dentry->d_inode
#endif

/* Stub out UVM in multi-RM builds */

#if (NV_BUILD_MODULE_INSTANCES != 0)
#undef NV_UVM_ENABLE
#endif

#if defined(NV_DOM0_KERNEL_PRESENT) || defined(NV_VGPU_KVM_BUILD)
#define NV_VGX_HYPER
#if defined(NV_XEN_IOEMU_INJECT_MSI)
#include <xen/ioemu.h>
#endif
#endif

#if defined(NV_FOR_EACH_ONLINE_NODE_PRESENT)
#define NV_FOR_EACH_ONLINE_NODE(nid)   for_each_online_node(nid)
#else
#define NV_FOR_EACH_ONLINE_NODE(nid)   for (nid = 0; nid < MAX_NUMNODES; nid++)
#endif

static inline NvU64 nv_node_end_pfn(int nid)
{
#if defined(NV_NODE_END_PFN_PRESENT)
    return node_end_pfn(nid);
#else
    return NODE_DATA(nid)->node_start_pfn + node_spanned_pages(nid);
#endif
}

static inline NvU64 nv_pci_bus_address(nv_linux_state_t *nvl, NvU8 bar_index)
{
    NvU64 bus_addr = 0;
#if defined(NV_PCI_BUS_ADDRESS_PRESENT)
    bus_addr = pci_bus_address(nvl->dev, bar_index);
#elif defined(CONFIG_PCI)
    struct pci_bus_region region;

    pcibios_resource_to_bus(nvl->dev, &region, &nvl->dev->resource[bar_index]);
    bus_addr = region.start;
#endif
    return bus_addr;
}

/*
 * Decrements the usage count of the allocation, and moves the allocation to
 * the given nvfp's free list if the usage count drops to zero.
 *
 * Returns NV_TRUE if the allocation is moved to the nvfp's free list.
 */
static inline NvBool nv_alloc_release(nv_file_private_t *nvfp, nv_alloc_t *at)
{
    NV_PRINT_AT(NV_DBG_MEMINFO, at);

    if (NV_ATOMIC_DEC_AND_TEST(at->usage_count))
    {
        NV_ATOMIC_INC(at->usage_count);

        at->next = nvfp->free_list;
        nvfp->free_list = at;
        return NV_TRUE;
    }

    return NV_FALSE;
}

/*
 * RB_EMPTY_ROOT was added in 2.6.18 by this commit:
 *   2006-06-21  dd67d051529387f6e44d22d1d5540ef281965fdd
 */
#if !defined(RB_EMPTY_ROOT)
#define RB_EMPTY_ROOT(root) ((root)->rb_node == NULL)
#endif

/*
 * Starting on Power9 systems, DMA addresses for NVLink are no longer
 * the same as used over PCIe.
 *
 * Power9 supports a 56-bit Real Address. This address range is compressed
 * when accessed over NVLink to allow the GPU to access all of memory using
 * its 47-bit Physical address.
 *
 * If there is an NPU device present on the system, it implies that NVLink
 * sysmem links are present and we need to apply the required address
 * conversion for NVLink within the driver.
 *
 * See Bug 1920398 for further background and details.
 *
 * Note, a deviation from the documented compression scheme is that the 
 * upper address bits (i.e. bit 56-63) instead of being set to zero are
 * preserved during NVLink address compression so the orignal PCIe DMA
 * address can be reconstructed on expansion. These bits can be safely
 * ignored on NVLink since they are truncated by the GPU.
 *
 * Bug 1968345: As a performance enhancement it is the responsibility of
 * the caller on PowerPC platforms to check for presence of an NPU device
 * before the address transformation is applied.
 */
static inline NvU64 nv_compress_nvlink_addr(NvU64 addr)
{
    NvU64 addr47 = addr;

#if defined(NVCPU_PPC64LE)
    addr47 = addr & ((1ULL << 43) - 1);
    addr47 |= (addr & (0x3ULL << 45)) >> 2;
    WARN_ON(addr47 & (1ULL << 44));
    addr47 |= (addr & (0x3ULL << 49)) >> 4;
    addr47 |= addr & ~((1ULL << 56) - 1);
#endif

    return addr47;
}

static inline NvU64 nv_expand_nvlink_addr(NvU64 addr47)
{
    NvU64 addr = addr47;

#if defined(NVCPU_PPC64LE)
    addr = addr47 & ((1ULL << 43) - 1);
    addr |= (addr47 & (3ULL << 43)) << 2;
    addr |= (addr47 & (3ULL << 45)) << 4;
    addr |= addr47 & ~((1ULL << 56) - 1);
#endif

    return addr;
}

#if defined(NV_BACKLIGHT_DEVICE_REGISTER_PRESENT)
#include <linux/backlight.h>
#endif

#endif  /* _NV_LINUX_H_ */
