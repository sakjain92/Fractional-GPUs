/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


/*
 * Os interface definitions needed by os-interface.c
 */

#ifndef _OS_INTERFACE_H_
#define _OS_INTERFACE_H_

/******************* Operating System Interface Routines *******************\
*                                                                           *
* Module: os-interface.h                                                    *
*       Included by os.h                                                    *
*       Operating system wrapper functions used to abstract the OS.         *
*                                                                           *
\***************************************************************************/

#include <stdarg.h>

#include "nvipmi.h"

/*
 * Define away Microsoft compiler extensions when possible
 */

#define __stdcall
#define far
#define PASCAL

/*
 * Make sure that arguments to and from the core resource manager
 * are passed and expected on the stack. define duplicated in nv.h
 */
#if !defined(NV_API_CALL)
#if defined(NVCPU_X86)
#if defined(__use_altstack__)
#define NV_API_CALL __attribute__((regparm(0),altstack(false)))
#else
#define NV_API_CALL __attribute__((regparm(0)))
#endif
#elif defined(NVCPU_X86_64) && defined(__use_altstack__)
#define NV_API_CALL __attribute__((altstack(false)))
#else
#define NV_API_CALL
#endif
#endif /* !defined(NV_API_CALL) */

typedef struct
{
    NvU32  os_major_version;
    NvU32  os_minor_version;
    NvU32  os_build_number;
    char * os_build_version_str;
    char * os_build_date_plus_str;
}os_version_info;

/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for OS interface.
 *
 * ---------------------------------------------------------------------------
 */

NvU32       NV_API_CALL  os_get_page_size            (void);
NvU64       NV_API_CALL  os_get_page_mask            (void);
NvU8        NV_API_CALL  os_get_page_shift           (void);
NvU64       NV_API_CALL  os_get_num_phys_pages       (void);
NV_STATUS   NV_API_CALL  os_alloc_mem                (void **, NvU64);
void        NV_API_CALL  os_free_mem                 (void *);
NV_STATUS   NV_API_CALL  os_get_current_time         (NvU32 *, NvU32 *);
void        NV_API_CALL  os_get_current_tick         (NvU64 *);
NV_STATUS   NV_API_CALL  os_delay                    (NvU32);
NV_STATUS   NV_API_CALL  os_delay_us                 (NvU32);
NvU64       NV_API_CALL  os_get_cpu_frequency        (void);
NvU32       NV_API_CALL  os_get_current_process      (void);
void        NV_API_CALL  os_get_current_process_name (char *, NvU32);
NvU32       NV_API_CALL  os_get_current_pasid        (void);
NV_STATUS   NV_API_CALL  os_get_current_thread       (NvU64 *);
char*       NV_API_CALL  os_string_copy              (char *, const char *);
NvU32       NV_API_CALL  os_string_length            (const char *);
NvU32       NV_API_CALL  os_strtoul                  (const char *, char **, NvU32);
NvS32       NV_API_CALL  os_string_compare           (const char *, const char *);
NvS32       NV_API_CALL  os_snprintf                 (char *, NvU32, const char *, ...);
void        NV_API_CALL  os_log_error                (const char *, va_list);
NvU8*       NV_API_CALL  os_mem_copy                 (NvU8 *, const NvU8 *, NvU32);
NV_STATUS   NV_API_CALL  os_memcpy_from_user         (void *, const void *, NvU32);
NV_STATUS   NV_API_CALL  os_memcpy_to_user           (void *, const void *, NvU32);
void*       NV_API_CALL  os_mem_set                  (void *, NvU8, NvU32);
NvS32       NV_API_CALL  os_mem_cmp                  (const NvU8 *, const NvU8 *, NvU32);
void*       NV_API_CALL  os_pci_init_handle          (NvU32, NvU8, NvU8, NvU8, NvU16 *, NvU16 *);
NV_STATUS   NV_API_CALL  os_pci_read_byte            (void *, NvU32, NvU8 *);
NV_STATUS   NV_API_CALL  os_pci_read_word            (void *, NvU32, NvU16 *);
NV_STATUS   NV_API_CALL  os_pci_read_dword           (void *, NvU32, NvU32 *);
NV_STATUS   NV_API_CALL  os_pci_write_byte           (void *, NvU32, NvU8);
NV_STATUS   NV_API_CALL  os_pci_write_word           (void *, NvU32, NvU16);
NV_STATUS   NV_API_CALL  os_pci_write_dword          (void *, NvU32, NvU32);
NvBool      NV_API_CALL  os_pci_remove_supported     (void);
void        NV_API_CALL  os_pci_remove               (void *);
void*       NV_API_CALL  os_map_kernel_space         (NvU64, NvU64, NvU32, NvU32);
void        NV_API_CALL  os_unmap_kernel_space       (void *, NvU64);
void        NV_API_CALL  os_unmap_kernel_numa        (void *, NvU64);
void*       NV_API_CALL  os_map_user_space           (NvU64, NvU64, NvU32, NvU32, void **);
void        NV_API_CALL  os_unmap_user_space         (void *, NvU64, void *);
NV_STATUS   NV_API_CALL  os_flush_cpu_cache          (void);
NV_STATUS   NV_API_CALL  os_flush_cpu_cache_all      (void);
NV_STATUS   NV_API_CALL  os_flush_user_cache         (NvU64, NvU64, NvU64, NvU64, NvU32);
void        NV_API_CALL  os_flush_cpu_write_combine_buffer(void);
NvU8        NV_API_CALL  os_io_read_byte             (NvU32);
NvU16       NV_API_CALL  os_io_read_word             (NvU32);
NvU32       NV_API_CALL  os_io_read_dword            (NvU32);
void        NV_API_CALL  os_io_write_byte            (NvU32, NvU8);
void        NV_API_CALL  os_io_write_word            (NvU32, NvU16);
void        NV_API_CALL  os_io_write_dword           (NvU32, NvU32);
BOOL        NV_API_CALL  os_is_administrator         (PHWINFO);
void        NV_API_CALL  os_dbg_init                 (void);
void        NV_API_CALL  os_dbg_breakpoint           (void);
void        NV_API_CALL  os_dbg_set_level            (NvU32);
NvU32       NV_API_CALL  os_get_cpu_count            (void);
NvU32       NV_API_CALL  os_get_cpu_number           (void);
NV_STATUS   NV_API_CALL  os_disable_console_access   (void);
NV_STATUS   NV_API_CALL  os_enable_console_access    (void);
NV_STATUS   NV_API_CALL  os_registry_init            (void);
NV_STATUS   NV_API_CALL  os_schedule                 (void);
NV_STATUS   NV_API_CALL  os_alloc_spinlock           (void **);
void        NV_API_CALL  os_free_spinlock            (void *);
NvU64       NV_API_CALL  os_acquire_spinlock         (void *);
void        NV_API_CALL  os_release_spinlock         (void *, NvU64);
NV_STATUS   NV_API_CALL  os_get_address_space_info   (NvU64 *, NvU64 *, NvU64 *, NvU64 *);
NV_STATUS   NV_API_CALL  os_queue_work_item          (void *);
NV_STATUS   NV_API_CALL  os_flush_work_queue         (void);
void        NV_API_CALL  os_register_compatible_ioctl    (NvU32, NvU32);
void        NV_API_CALL  os_unregister_compatible_ioctl  (NvU32, NvU32);
NV_STATUS   NV_API_CALL  os_alloc_mutex              (void **);
void        NV_API_CALL  os_free_mutex               (void *);
NV_STATUS   NV_API_CALL  os_acquire_mutex            (void *);
NV_STATUS   NV_API_CALL  os_cond_acquire_mutex       (void *);
void        NV_API_CALL  os_release_mutex            (void *);
void*       NV_API_CALL  os_alloc_semaphore          (NvU32);
void        NV_API_CALL  os_free_semaphore           (void *);
NV_STATUS   NV_API_CALL  os_acquire_semaphore        (void *);
NV_STATUS   NV_API_CALL  os_release_semaphore        (void *);
BOOL        NV_API_CALL  os_semaphore_may_sleep      (void);
NV_STATUS   NV_API_CALL  os_get_version_info         (os_version_info*);
BOOL        NV_API_CALL  os_is_isr                   (void);
BOOL        NV_API_CALL  os_pat_supported            (void);
void        NV_API_CALL  os_dump_stack               (void);
BOOL        NV_API_CALL  os_is_efi_enabled           (void);
BOOL        NV_API_CALL  os_iommu_is_snooping_enabled(void);
NvBool      NV_API_CALL  os_is_xen_dom0              (void);
NvBool      NV_API_CALL  os_is_vgx_hyper             (void);
NV_STATUS   NV_API_CALL  os_inject_vgx_msi           (NvU16, NvU64, NvU32);
NvBool      NV_API_CALL  os_is_grid_supported        (void);
void        NV_API_CALL  os_get_screen_info          (NvU64 *, NvU16 *, NvU16 *, NvU16 *, NvU16 *);
void        NV_API_CALL  os_bug_check                (NvU32, const char *);
NV_STATUS   NV_API_CALL  os_lock_user_pages          (void *, NvU64, void **);
NV_STATUS   NV_API_CALL  os_lookup_user_io_memory    (void *, NvU64, NvU64 **);
NV_STATUS   NV_API_CALL  os_unlock_user_pages        (NvU64, void *);
NV_STATUS   NV_API_CALL  os_match_mmap_offset        (void *, NvU64, NvU64 *);
NV_STATUS   NV_API_CALL  os_get_euid                 (NvU32 *);
NV_STATUS   NV_API_CALL  os_get_smbios_header        (NvU64 *pSmbsAddr);
NV_STATUS   NV_API_CALL  os_get_acpi_rsdp_from_uefi  (NvU32 *);
void        NV_API_CALL  os_add_record_for_crashLog  (void *, NvU32);
void        NV_API_CALL  os_delete_record_for_crashLog (void *);
NV_STATUS   NV_API_CALL  os_call_vgpu_vfio           (void *, NvU32);
NV_STATUS   NV_API_CALL  os_numa_memblock_size(NvU64 *);
NV_STATUS   NV_API_CALL  os_numa_online_memory(NvS32, NvU64, NvU64, NvU64, NvU64, NvBool);
void        NV_API_CALL  os_numa_offline_memory(NvS32);
NV_STATUS   NV_API_CALL  os_alloc_pages_node         (NvS32, NvU32, NvU32, NvU64 *);
NV_STATUS   NV_API_CALL  os_get_page                 (NvU64 address);
NV_STATUS   NV_API_CALL  os_put_page                 (NvU64 address);
void        NV_API_CALL  os_free_pages_phys          (NvU64, NvU32);
NV_STATUS   NV_API_CALL  os_ipmi_connect             (NvU32 devIndex, NvU8 myAddr, void **ppOsPriv);
void        NV_API_CALL  os_ipmi_disconnect          (void *pOsPriv);
NV_STATUS   NV_API_CALL  os_ipmi_send_receive_cmd    (void *pOsPriv, nvipmi_req_resp_t *pReq);

/*
 * ---------------------------------------------------------------------------
 *
 * Debug macros.
 *
 * ---------------------------------------------------------------------------
 */

#if !defined(DBG_LEVEL_INFO)
/*
 * Debug Level values
 */
#define DBG_LEVEL_INFO          0x0   /* For informational debug trace info */
#define DBG_LEVEL_SETUPINFO     0x1   /* For informational debug setup info */
#define DBG_LEVEL_USERERRORS    0x2   /* For debug info on app level errors */ 
#define DBG_LEVEL_WARNINGS      0x3   /* For RM debug warning info          */
#define DBG_LEVEL_ERRORS        0x4   /* For RM debug error info            */
#endif

#define NV_DBG_INFO       0x0
#define NV_DBG_SETUP      0x1
#define NV_DBG_USERERRORS 0x2
#define NV_DBG_WARNINGS   0x3
#define NV_DBG_ERRORS     0x4


void NV_API_CALL  out_string(const char *str);
int  NV_API_CALL  nv_printf(NvU32 debuglevel, const char *printf_format, ...);

/*
 * those NV_MEMORY_* and NV_PROTECT_* have dup defines in osCore.h
 */
#define NV_MEMORY_TYPE_SYSTEM       0
// #define NV_MEMORY_TYPE_AGP          1
#define NV_MEMORY_TYPE_REGISTERS    2
#define NV_MEMORY_TYPE_FRAMEBUFFER  3
#define NV_MEMORY_TYPE_INSTANCE     4
#define NV_MEMORY_TYPE_DEVICE_MMIO  5  /*  All kinds of MMIO referred by NVRM e.g. BARs and MCFG of device */

#define NV_MEMORY_NONCONTIGUOUS     0
#define NV_MEMORY_CONTIGUOUS        1

#define NV_MEMORY_CACHED            0
#define NV_MEMORY_UNCACHED          1
#define NV_MEMORY_WRITECOMBINED     2
#define NV_MEMORY_WRITETHRU         3
#define NV_MEMORY_WRITEPROTECT      4
#define NV_MEMORY_WRITEBACK         5
#define NV_MEMORY_DEFAULT           6
#define NV_MEMORY_UNCACHED_WEAK     7
#define NV_MEMORY_NUMA              8

#define NV_PROTECT_READABLE   1
#define NV_PROTECT_WRITEABLE  2
#define NV_PROTECT_READ_WRITE (NV_PROTECT_READABLE | NV_PROTECT_WRITEABLE)

#define OS_UNIX_FLUSH_USER_CACHE       1
#define OS_UNIX_INVALIDATE_USER_CACHE  2
#define OS_UNIX_FLUSH_INVALIDATE_USER_CACHE (OS_UNIX_FLUSH_USER_CACHE | OS_UNIX_INVALIDATE_USER_CACHE)

/* in some cases, the os may have a different page size, but the
 * hardware (fb, regs, etc) still address and "think" in 4k
 * pages. make sure we can mask and twiddle with these addresses when
 * PAGE_SIZE isn't what we want.
 */
#define OS_PAGE_SIZE                (os_get_page_size())
#define OS_PAGE_MASK                (os_get_page_mask())
#define OS_PAGE_SHIFT               (os_get_page_shift())

#define IS_VGX_HYPER()              os_is_vgx_hyper()
#define IS_GRID_SUPPORTED()         os_is_grid_supported()

#endif /* _OS_INTERFACE_H_ */
