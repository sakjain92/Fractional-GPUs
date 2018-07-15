/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_PROTO_H_
#define _NV_PROTO_H_

#include "nv-misc.h"

int         nv_acpi_init                (void);
int         nv_acpi_uninit              (void);

#if !defined(NV_IRQ_HANDLER_T_PRESENT) || (NV_IRQ_HANDLER_T_ARGUMENT_COUNT == 3)
irqreturn_t nv_gvi_kern_isr             (int, void *, struct pt_regs *);
#else
irqreturn_t nv_gvi_kern_isr             (int, void *);
#endif

#if (NV_INIT_WORK_ARGUMENT_COUNT == 3)
void        nv_gvi_kern_bh              (void *);
#else
void        nv_gvi_kern_bh              (struct work_struct *);
#endif

#if defined(NV_PM_SUPPORT_DEVICE_DRIVER_MODEL)
int         nv_gvi_kern_suspend         (struct pci_dev *, pm_message_t);
int         nv_gvi_kern_resume          (struct pci_dev *);
#endif

int         nv_register_chrdev          (void *);
void        nv_unregister_chrdev        (void *);

NvU8        nv_find_pci_capability      (struct pci_dev *, NvU8);
void *      nv_alloc_file_private       (void);
void        nv_free_file_private        (nv_file_private_t *);

void        nv_check_pci_config_space   (nv_state_t *, BOOL);

int         nv_register_procfs          (void);
void        nv_unregister_procfs        (void);
void        nv_procfs_add_warning       (const char *, const char *);
int         nv_procfs_add_gpu           (nv_linux_state_t *);
void        nv_procfs_remove_gpu        (nv_linux_state_t *);

int         nvidia_mmap                 (struct file *, struct vm_area_struct *);
int         nvidia_mmap_helper          (nv_state_t *, nv_file_private_t *, nvidia_stack_t *, struct vm_area_struct *, void *);
int         nv_encode_caching           (pgprot_t *, NvU32, NvU32);

void        nv_user_map_init            (void);
int         nv_user_map_register        (NvU64, NvU64);
void        nv_user_map_unregister      (NvU64, NvU64);

int         nv_heap_create              (void);
void        nv_heap_destroy             (void);
int         nv_mem_pool_create          (void);
void        nv_mem_pool_destroy         (void);
void *      nv_mem_pool_alloc_pages     (NvU32);
void        nv_mem_pool_free_pages      (void *, NvU32);

NvUPtr      nv_vm_map_pages             (struct page **, NvU32, NvBool);
void        nv_vm_unmap_pages           (NvUPtr, NvU32);

NV_STATUS   nv_alloc_contig_pages       (nv_state_t *, nv_alloc_t *);
void        nv_free_contig_pages        (nv_alloc_t *);
NV_STATUS   nv_alloc_system_pages       (nv_state_t *, nv_alloc_t *);
void        nv_free_system_pages        (nv_alloc_t *);

int         nv_uvm_init                 (void);
void        nv_uvm_exit                 (void);
void        nv_uvm_notify_start_device  (const NvU8 *uuid);
void        nv_uvm_notify_stop_device   (const NvU8 *uuid);
NV_STATUS   nv_uvm_event_interrupt      (const NvU8 *uuid);

/* Move these to nv.h once implemented by other UNIX platforms */
NvBool      nvidia_get_gpuid_list       (NvU32 *gpu_ids, NvU32 *gpu_count);
int         nvidia_dev_get              (NvU32, nvidia_stack_t *);
void        nvidia_dev_put              (NvU32, nvidia_stack_t *);
int         nvidia_dev_get_uuid         (const NvU8 *, nvidia_stack_t *);
void        nvidia_dev_put_uuid         (const NvU8 *, nvidia_stack_t *);
int         nvidia_dev_get_pci_info     (const NvU8 *, struct pci_dev **, NvU64 *, NvU64 *);

int           nvidia_open           (struct inode *, struct file *);
int           nvidia_close          (struct inode *, struct file *);
unsigned int  nvidia_poll           (struct file *, poll_table *);
int           nvidia_ioctl          (struct inode *, struct file *, unsigned int, unsigned long);

int           nvidia_probe          (struct pci_dev *, const struct pci_device_id *);
void          nvidia_remove         (struct pci_dev *);

int           nvidia_suspend        (struct pci_dev *, pm_message_t);
int           nvidia_resume         (struct pci_dev *);

void          nvidia_modeset_suspend    (NvU32 gpuId);
void          nvidia_modeset_resume     (NvU32 gpuId);

NV_STATUS     nv_parse_per_device_option_string(nvidia_stack_t *sp);
nv_linux_state_t *  find_pci(NvU16 domain, NvU8 bus, NvU8 slot, NvU8 function);
void nv_report_error(struct pci_dev *dev, NvU32 error_number, const char *format, va_list ap);
#endif /* _NV_PROTO_H_ */
