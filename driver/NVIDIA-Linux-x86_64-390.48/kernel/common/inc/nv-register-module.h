/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2013 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _NV_REGISTER_MODULE_H_
#define _NV_REGISTER_MODULE_H_

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/poll.h>

#include "nvtypes.h"

typedef struct nvidia_module_s {
    struct module *owner;

    /* nvidia0, nvidia1 ..*/
    const char *module_name;

    /* module instance */
    NvU32 instance;

    /* file operations */
    int (*open)(struct inode *, struct file *filp);
    int (*close)(struct inode *, struct file *filp);
    int (*mmap)(struct file *filp, struct vm_area_struct *vma);
    int (*ioctl)(struct inode *, struct file * file, unsigned int cmd, unsigned long arg);
    unsigned int (*poll)(struct file * file, poll_table *wait);

} nvidia_module_t;

int nvidia_register_module(nvidia_module_t *);
int nvidia_unregister_module(nvidia_module_t *);

#endif
