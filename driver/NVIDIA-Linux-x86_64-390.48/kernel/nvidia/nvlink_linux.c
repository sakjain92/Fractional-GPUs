/*******************************************************************************
    Copyright (c) 2015-2016 NVidia Corporation

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

#include "conftest.h"

#include "nvlink_common.h"
#include "nvlink_linux.h"
#include "nvlink_errors.h"
#include "nvlink_export.h"
#include "nv-linux.h"
#include "nv-procfs.h"

#define MAX_ERROR_STRING           512

#define NV_MAX_ISR_DELAY_US           20000
#define NV_MAX_ISR_DELAY_MS           (NV_MAX_ISR_DELAY_US / 1000)

#define NV_MSECS_PER_JIFFIE         (1000 / HZ)
#define NV_MSECS_TO_JIFFIES(msec)   ((msec) * HZ / 1000)
#define NV_USECS_PER_JIFFIE         (1000000 / HZ)
#define NV_USECS_TO_JIFFIES(usec)   ((usec) * HZ / 1000000)

/* NVLink character device driver is supported from 2.6.16 onwards */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 16)
#include <linux/module.h>
#include <linux/major.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/string.h>
#include <linux/mutex.h>

typedef struct
{
    struct mutex    lock;
    NvBool          initialized;
    struct cdev     cdev;
    dev_t           devno;
    int             opened;
    int             major_devnum;
} _nvlink_drvctx;


// nvlink driver local state
static _nvlink_drvctx nvlink_drvctx;

#define NVLINK_PROCFS_DIR "driver/nvidia-nvlink"

static struct proc_dir_entry *nvlink_procfs_dir = NULL;

#if defined(CONFIG_PROC_FS)
    static int nvlink_is_procfs_available = 1;
#else
    static int nvlink_is_procfs_available = 0;
#endif

static struct proc_dir_entry *nvlink_permissions = NULL;

static int nv_procfs_read_permissions(struct seq_file *s, void *v)
{
    // Restrict device node permissions to root-only - 0600.
    seq_printf(s, "%s: %u\n", "DeviceFileMode", 384);

    return 0;
}

NV_DEFINE_PROCFS_SINGLE_FILE(permissions);

static void nvlink_permissions_exit(void)
{
    if (!nvlink_permissions)
    {
        return;
    }

    NV_REMOVE_PROC_ENTRY(nvlink_permissions);
    nvlink_permissions = NULL;
}

static int nvlink_permissions_init(void)
{
    if (!nvlink_procfs_dir)
    {
        return -EACCES;
    }

    nvlink_permissions = NV_CREATE_PROC_FILE("permissions",
                                             nvlink_procfs_dir,
                                             permissions,
                                             NULL);
    if (!nvlink_permissions)
    {
        return -EACCES;
    }

    return 0;
}

static void nvlink_procfs_exit(void)
{
    nvlink_permissions_exit();

    if (!nvlink_procfs_dir)
    {
        return;
    }

    NV_REMOVE_PROC_ENTRY(nvlink_procfs_dir);
    nvlink_procfs_dir = NULL;
}

static int nvlink_procfs_init(void)
{
    int rc = 0;

    if (!nvlink_is_procfs_available)
    {
        return -EACCES;
    }

    nvlink_procfs_dir = NV_CREATE_PROC_DIR(NVLINK_PROCFS_DIR, NULL);
    if (!nvlink_procfs_dir)
    {
        return -EACCES;
    }

    rc = nvlink_permissions_init();
    if (rc < 0)
    {
        goto cleanup;
    }

    return 0;

cleanup:

    nvlink_procfs_exit();

    return rc;
}

static int nvlink_fops_open(struct inode *inode, struct file *filp)
{
    int rc = 0;

    nvlink_print(NVLINK_DBG_INFO, "nvlink driver open\n");

    mutex_lock(&nvlink_drvctx.lock);

    // nvlink lib driver is currently exclusive open.
    if (nvlink_drvctx.opened)
    {
        rc = -EBUSY;
        goto open_error;
    }
    // mark our state as opened
    nvlink_drvctx.opened = NV_TRUE;

open_error:
    mutex_unlock(&nvlink_drvctx.lock);
    return rc;
}

static int nvlink_fops_release(struct inode *inode, struct file *filp)
{
    nvlink_print(NVLINK_DBG_INFO, "nvlink driver close\n");

    mutex_lock(&nvlink_drvctx.lock);

    // mark the device as not opened
    nvlink_drvctx.opened = NV_FALSE;

    mutex_unlock(&nvlink_drvctx.lock);

    return 0;
}

static int nvlink_fops_ioctl(struct inode *inode,
                             struct file *filp,
                             unsigned int cmd,
                             unsigned long arg)
{
    nvlink_ioctrl_params ctrl_params = {0};
    int param_size = _IOC_SIZE(cmd);
    void *param_buf = NULL;
    NvlStatus ret_val = 0;
    int rc = 0;

    // no buffer for simple _IO types
    if (param_size)
    {
        // allocate a buffer to hold user input
        param_buf = kzalloc(param_size, GFP_KERNEL);
        if (NULL == param_buf) 
        {
            rc = -ENOMEM;
            goto nvlink_ioctl_fail;
        }

        // copy user input to kernel buffers. Simple _IOR() ioctls can skip this step.
        if (_IOC_DIR(cmd) & _IOC_WRITE)
        {
            // copy user input to local buffer
            if (copy_from_user(param_buf, (const void *)arg, param_size))
            {
                rc = -EFAULT;
                goto nvlink_ioctl_fail;
            }
        }
    }

    ctrl_params.cmd = _IOC_NR(cmd);
    ctrl_params.buf = param_buf;
    ctrl_params.size = param_size;

    ret_val = nvlink_lib_ioctl_ctrl(&ctrl_params);
    if (NVL_SUCCESS != ret_val)
    {
        rc = -EINVAL;
        goto nvlink_ioctl_fail;
    }

    //  no copy for write-only ioctl
    if ((param_size) && (_IOC_DIR(cmd) & _IOC_READ))
    {
        if (copy_to_user((void *)arg, ctrl_params.buf, ctrl_params.size))
        {
            rc = -EFAULT;
            goto nvlink_ioctl_fail;
        }
    }

nvlink_ioctl_fail:
    if (param_buf) 
    {
        kfree(param_buf);
    }
    return rc;
}

#if defined(NV_FILE_HAS_INODE)
#define NV_FILE_INODE(file) (file)->f_inode
#else
#define NV_FILE_INODE(file) (file)->f_dentry->d_inode
#endif

static long nvlink_fops_unlocked_ioctl(struct file *file,
                                       unsigned int cmd,
                                       unsigned long arg)
{
    return nvlink_fops_ioctl(NV_FILE_INODE(file), file, cmd, arg);
}


static const struct file_operations nvlink_fops = {
    .owner           = THIS_MODULE,
    .open            = nvlink_fops_open,
    .release         = nvlink_fops_release,
#if defined(NV_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl           = nvlink_fops_ioctl,   
#endif    
#if defined(NV_FILE_OPERATIONS_HAS_UNLOCKED_IOCTL)
    .unlocked_ioctl  = nvlink_fops_unlocked_ioctl,
#endif
};

int __init nvlink_core_init(void)
{
    NvlStatus ret_val;
    int rc;

    if (NV_TRUE == nvlink_drvctx.initialized)
    {
        nvlink_print(NVLINK_DBG_ERRORS, "nvlink core interface already initialized\n");
        return -EBUSY;
    }

    mutex_init(&nvlink_drvctx.lock);

    ret_val = nvlink_lib_initialize();
    if (NVL_SUCCESS != ret_val)
    {
        nvlink_print(NVLINK_DBG_ERRORS,  "Failed to initialize driver : %d\n", ret_val);
        rc = -ENODEV;
        goto nvlink_lib_initialize_fail;
    }

    rc = alloc_chrdev_region(&nvlink_drvctx.devno, 0, NVLINK_NUM_MINOR_DEVICES,
                             NVLINK_DEVICE_NAME);
    if (rc < 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS, "alloc_chrdev_region failed: %d\n", rc);
        goto alloc_chrdev_region_fail;
    }

    nvlink_drvctx.major_devnum = MAJOR(nvlink_drvctx.devno);
    nvlink_print(NVLINK_DBG_INFO, "Nvlink Core is being initialized, major device number %d\n",
                 nvlink_drvctx.major_devnum);

    cdev_init(&nvlink_drvctx.cdev, &nvlink_fops);
    nvlink_drvctx.cdev.owner = THIS_MODULE;
    rc = cdev_add(&nvlink_drvctx.cdev, nvlink_drvctx.devno, NVLINK_NUM_MINOR_DEVICES);
    if (rc < 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS, " Unable to create cdev\n");
        goto cdev_add_fail;
    }

    rc = nvlink_procfs_init();
    if (rc < 0)
    {
        goto procfs_init_fail;
    }

    nvlink_drvctx.initialized = NV_TRUE;

    return 0;

procfs_init_fail:
    cdev_del(&nvlink_drvctx.cdev);

cdev_add_fail:
    unregister_chrdev_region(nvlink_drvctx.devno, NVLINK_NUM_MINOR_DEVICES);

alloc_chrdev_region_fail:
    nvlink_lib_unload();

nvlink_lib_initialize_fail:
    mutex_destroy(&nvlink_drvctx.lock);
    return rc;
}

void nvlink_core_exit(void)
{
    if (NV_FALSE == nvlink_drvctx.initialized)
    {
        return;
    }

    nvlink_procfs_exit();

    cdev_del(&nvlink_drvctx.cdev);

    unregister_chrdev_region(nvlink_drvctx.devno, NVLINK_NUM_MINOR_DEVICES);

    nvlink_lib_unload();

    mutex_destroy(&nvlink_drvctx.lock);

    nvlink_print(NVLINK_DBG_INFO, "Unregistered the Nvlink Core, major device number %d\n",
                 nvlink_drvctx.major_devnum);
}

#else
/* initialize NVLink Core library without any character device support */
int __init nvlink_core_init(void)
{
    return nvlink_lib_initialize();
}
void nvlink_core_exit(void)
{
    nvlink_lib_unload();
}
#endif /* LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 16) */


/* OS specific APIs implemented for NVLink library */
static void dbg_breakpoint(void)
{
#if defined(DEBUG)
  #if defined(CONFIG_X86_REMOTE_DEBUG) || defined(CONFIG_KGDB) || defined(CONFIG_XMON)
    #if defined(NVCPU_X86) || defined(NVCPU_X86_64)
        __asm__ __volatile__ ("int $3");
    #elif defined(NVCPU_ARM)
        __asm__ __volatile__ (".word %c0" :: "i" (KGDB_COMPILED_BREAK));
    #elif defined(NVCPU_AARCH64)
        # warning "Need to implement dbg_breakpoint() for aarch64"
    #elif defined(NVCPU_PPC64LE)
        __asm__ __volatile__ ("trap");
    #endif // NVCPU_X86 || NVCPU_X86_64
  #elif defined(CONFIG_KDB)
      KDB_ENTER();
  #endif // CONFIG_X86_REMOTE_DEBUG || CONFIG_KGDB || CONFIG_XMON
#endif // DEBUG
}

void 
NVLINK_API_CALL
nvlink_print
(
    const char *file,
    int         line,
    const char *function,
    int         log_level,
    const char *fmt,
    ...
)
{
    va_list arglist;
    char    nv_string[MAX_ERROR_STRING];
    char   *sys_log_level;

    switch (log_level) {
    case NVLINK_DBG_LEVEL_INFO:
        sys_log_level = KERN_INFO;
        break;
    case NVLINK_DBG_LEVEL_SETUP:
        sys_log_level = KERN_DEBUG;
        break;
    case NVLINK_DBG_LEVEL_USERERRORS:
        sys_log_level = KERN_NOTICE;
        break;
    case NVLINK_DBG_LEVEL_WARNINGS:
        sys_log_level = KERN_WARNING;
        break;
    case NVLINK_DBG_LEVEL_ERRORS:
        sys_log_level = KERN_ERR;
        break;
    default:
        sys_log_level = KERN_INFO;
        break;
    }

    va_start(arglist, fmt);
    vsnprintf(nv_string, sizeof(nv_string), fmt, arglist);
    va_end(arglist);

    nv_string[sizeof(nv_string) - 1] = '\0';
    printk("%snvidia-nvlink: %s", sys_log_level, nv_string);
}

void * NVLINK_API_CALL nvlink_malloc(NvLength size)
{
   return kmalloc(size, GFP_KERNEL);
}

void NVLINK_API_CALL nvlink_free(void *ptr)
{
    return kfree(ptr);
}

char * NVLINK_API_CALL nvlink_strcpy(char *dest, const char *src)
{
    return strcpy(dest, src);
}

int NVLINK_API_CALL nvlink_strcmp(const char *dest, const char *src)
{
    return strcmp(dest, src);
}

NvLength NVLINK_API_CALL nvlink_strlen(const char *s)
{
    return strlen(s);
}

int NVLINK_API_CALL nvlink_snprintf(char *dest, NvLength size, const char *fmt, ...)
{
    va_list arglist;
    int chars_written;

    va_start(arglist, fmt);
    chars_written = vsnprintf(dest, size, fmt, arglist);
    va_end(arglist);

    return chars_written;
}

int NVLINK_API_CALL nvlink_memRd32(const volatile void * address)
{
    return (*(const volatile unsigned int*)(address));
}

void NVLINK_API_CALL nvlink_memWr32(volatile void *address, unsigned int data)
{
    (*(volatile unsigned int *)(address)) = data;
}

int NVLINK_API_CALL nvlink_memRd64(const volatile void * address)
{
    return (*(const volatile unsigned long long *)(address));
}

void NVLINK_API_CALL nvlink_memWr64(volatile void *address, unsigned long long data)
{
    (*(volatile unsigned long long *)(address)) = data;
}

void * NVLINK_API_CALL nvlink_memset(void *dest, int value, NvLength size)
{
     return memset(dest, value, size);
}

void * NVLINK_API_CALL nvlink_memcpy(void *dest, void *src, NvLength size)
{
    return memcpy(dest, src, size);
}

static NvBool nv_timer_less_than
(
    const struct timeval *a,
    const struct timeval *b
)
{
    return (a->tv_sec == b->tv_sec) ? (a->tv_usec < b->tv_usec) 
                                    : (a->tv_sec < b->tv_sec);
}

static void nv_timeradd
(
    const struct timeval    *a,
    const struct timeval    *b,
    struct timeval          *result
)
{
    result->tv_sec = a->tv_sec + b->tv_sec;
    result->tv_usec = a->tv_usec + b->tv_usec;
    while (result->tv_usec >= 1000000)
    {
        ++result->tv_sec;
        result->tv_usec -= 1000000;
    }
}

static void nv_timersub
(
    const struct timeval    *a,
    const struct timeval    *b,
    struct timeval          *result
)
{
    result->tv_sec = a->tv_sec - b->tv_sec;
    result->tv_usec = a->tv_usec - b->tv_usec;
    while (result->tv_usec < 0)
    {
        --(result->tv_sec);
        result->tv_usec += 1000000;
    }
}

/*
 * Sleep for specified milliseconds. Yields the CPU to scheduler.
 */
void NVLINK_API_CALL nvlink_sleep(unsigned int ms)
{
    unsigned long us;
    unsigned long jiffies;
    unsigned long mdelay_safe_msec;
    struct timeval tm_end, tm_aux;

    do_gettimeofday(&tm_aux);

    if (in_irq() && (ms > NV_MAX_ISR_DELAY_MS))
    {
        return;
    }

    if (irqs_disabled() || in_interrupt() || in_atomic())
    {
        mdelay(ms);
        return;
    }

    us = ms * 1000;
    tm_end.tv_usec = us;
    tm_end.tv_sec = 0;
    nv_timeradd(&tm_aux, &tm_end, &tm_end);

    /* do we have a full jiffie to wait? */
    jiffies = NV_USECS_TO_JIFFIES(us);

    if (jiffies)
    {
        //
        // If we have at least one full jiffy to wait, give up
        // up the CPU; since we may be rescheduled before
        // the requested timeout has expired, loop until less
        // than a jiffie of the desired delay remains.
        //
        current->state = TASK_INTERRUPTIBLE;
        do
        {
            schedule_timeout(jiffies);
            do_gettimeofday(&tm_aux);
            if (nv_timer_less_than(&tm_aux, &tm_end))
            {
                nv_timersub(&tm_end, &tm_aux, &tm_aux);
                us = tm_aux.tv_usec + tm_aux.tv_sec * 1000000;
            }
            else
            {
                us = 0;
            }
        } 
        while ((jiffies = NV_USECS_TO_JIFFIES(us)) != 0);
    }

    if (us > 1000)
    {
        mdelay_safe_msec = us / 1000;
        mdelay(mdelay_safe_msec);
        us %= 1000;
    }
    if (us)
    {
        udelay(us);
    }
}

void NVLINK_API_CALL nvlink_assert(int cond)
{
    if ((cond) == 0x0)
    {
        nvlink_print(NVLINK_DBG_ERRORS, "Assertion failed!\n");
        dbg_breakpoint();
    }
}

void * NVLINK_API_CALL nvlink_allocLock()
{
    struct semaphore *sema;

    sema = nvlink_malloc(sizeof(*sema));
    if (sema == NULL)
    {
        nvlink_print(NVLINK_DBG_ERRORS, "Failed to allocate sema!\n");
        return NULL;
    }
    sema_init(sema, 1);

    return sema;
}

void NVLINK_API_CALL nvlink_acquireLock(void *hLock)
{
    down(hLock);
}

void NVLINK_API_CALL nvlink_releaseLock(void *hLock)
{
    up(hLock);
}

void NVLINK_API_CALL nvlink_freeLock(void *hLock)
{
    if (NULL == hLock)
    {
        return;
    }

    NVLINK_FREE(hLock);
}

NvBool NVLINK_API_CALL nvlink_isLockOwner(void *hLock)
{
    return NV_TRUE;
}

