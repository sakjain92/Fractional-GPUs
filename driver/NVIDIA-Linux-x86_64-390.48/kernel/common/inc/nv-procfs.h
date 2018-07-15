/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */
#ifndef _NV_PROCFS_H
#define _NV_PROCFS_H

#include "conftest.h"

#ifdef CONFIG_PROC_FS
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

/*
 * Allow procfs to create file to exercise error forwarding.
 * This is supported by CRAY platforms.
 */
#if defined(CONFIG_CRAY_XT)
#define EXERCISE_ERROR_FORWARDING NV_TRUE
#else
#define EXERCISE_ERROR_FORWARDING NV_FALSE
#endif

#define IS_EXERCISE_ERROR_FORWARDING_ENABLED() (EXERCISE_ERROR_FORWARDING)

#if defined(NV_PROC_DIR_ENTRY_HAS_OWNER)
#define NV_SET_PROC_ENTRY_OWNER(entry) ((entry)->owner = THIS_MODULE)
#else
#define NV_SET_PROC_ENTRY_OWNER(entry)
#endif

#if defined(NV_PROC_CREATE_DATA_PRESENT)
# define NV_CREATE_PROC_ENTRY(name,mode,parent,fops,__data) \
    proc_create_data(name, mode, parent, fops, __data)
#else
# define NV_CREATE_PROC_ENTRY(name,mode,parent,fops,__data) \
   ({                                                       \
        struct proc_dir_entry *__entry;                     \
        __entry = create_proc_entry(name, mode, parent);    \
        if (__entry != NULL)                                \
        {                                                   \
            NV_SET_PROC_ENTRY_OWNER(__entry);               \
            __entry->proc_fops = fops;                      \
            __entry->data = (__data);                       \
        }                                                   \
        __entry;                                            \
    })
#endif

#define NV_CREATE_PROC_FILE(filename,parent,__name,__data)               \
   ({                                                                    \
        struct proc_dir_entry *__entry;                                  \
        int mode = (S_IFREG | S_IRUGO);                                  \
        const struct file_operations *fops = &nv_procfs_##__name##_fops; \
        if (fops->write != 0)                                            \
            mode |= S_IWUSR;                                             \
        __entry = NV_CREATE_PROC_ENTRY(filename, mode, parent, fops,     \
            __data);                                                     \
        __entry;                                                         \
    })

/*
 * proc_mkdir_mode exists in Linux 2.6.9, but isn't exported until Linux 3.0.
 * Use the older interface instead unless the newer interface is necessary.
 */
#if defined(NV_PROC_REMOVE_PRESENT)
# define NV_PROC_MKDIR_MODE(name, mode, parent)                \
    proc_mkdir_mode(name, mode, parent)
#else
# define NV_PROC_MKDIR_MODE(name, mode, parent)                \
   ({                                                          \
        struct proc_dir_entry *__entry;                        \
        __entry = create_proc_entry(name, mode, parent);       \
        if (__entry != NULL)                                   \
            NV_SET_PROC_ENTRY_OWNER(__entry);                  \
        __entry;                                               \
    })
#endif

#define NV_CREATE_PROC_DIR(name,parent)                        \
   ({                                                          \
        struct proc_dir_entry *__entry;                        \
        int mode = (S_IFDIR | S_IRUGO | S_IXUGO);              \
        __entry = NV_PROC_MKDIR_MODE(name, mode, parent);      \
        __entry;                                               \
    })

#if defined(NV_PDE_DATA_PRESENT)
# define NV_PDE_DATA(inode) PDE_DATA(inode)
#else
# define NV_PDE_DATA(inode) PDE(inode)->data
#endif

#if defined(NV_PROC_REMOVE_PRESENT)
# define NV_REMOVE_PROC_ENTRY(entry)                           \
    proc_remove(entry);
#else
# define NV_REMOVE_PROC_ENTRY(entry)                           \
    remove_proc_entry(entry->name, entry->parent);
#endif

#define NV_DEFINE_PROCFS_SINGLE_FILE(__name)                                  \
    static int nv_procfs_open_##__name(                                       \
        struct inode *inode,                                                  \
        struct file *filep                                                    \
    )                                                                         \
    {                                                                         \
        return single_open(filep, nv_procfs_read_##__name,                    \
            NV_PDE_DATA(inode));                                              \
    }                                                                         \
                                                                              \
    static const struct file_operations nv_procfs_##__name##_fops = {         \
        .owner      = THIS_MODULE,                                            \
        .open       = nv_procfs_open_##__name,                                \
        .read       = seq_read,                                               \
        .llseek     = seq_lseek,                                              \
        .release    = single_release,                                         \
    };

#endif  /* CONFIG_PROC_FS */

#endif /* _NV_PROCFS_H */

