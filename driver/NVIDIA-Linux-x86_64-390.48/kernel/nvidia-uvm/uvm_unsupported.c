/*******************************************************************************
    Copyright (c) 2015, 2016 NVIDIA Corporation

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

// This file provides a tiny kernel module that implements a "not supported" version
// of the UVM character device driver. The following character device file operations
// are supported:
//
//      .open:    reports success, but does nothing
//
//      .release: reports success, but does nothing
//
//      .unlocked_ioctl, compat_ioctl: reports success. Supports the following ioctl commands:
//
//             UVM_INITIALIZE: returns an embedded NV_STATUS value that indicates that this
//                             call (and therefore, UVM in general, is not supported).
//
//             UVM_DEINITIALIZE: reports success, but does nothing.

#include "uvmtypes.h"
#include "uvm_linux_ioctl.h"
#include "uvm_minimal_init.h"
#include "conftest.h"

#include <linux/module.h>
#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/fs.h>

#if defined(NV_LINUX_PRINTK_H_PRESENT)
    #include <linux/printk.h>
#endif

static dev_t g_uvm_base_dev;
static struct cdev g_uvm_cdev;

static NV_STATUS uvm_unsupported_initialize(UVM_INITIALIZE_PARAMS *params,
                                            struct file *filp)
{
    // The UVM_ROUTE_CMD framework translates function return values
    // into params.rmStatus values:
    return NV_ERR_NOT_SUPPORTED;
}

static long uvm_unsupported_unlocked_ioctl(struct file *filp, unsigned int cmd,
                                           unsigned long arg)
{
    // The following macro is only intended for use in this routine. That's why
    // it is declared inside the function (even though, of course, the
    // preprocessor ignores such scoping).
    #define UVM_ROUTE_CMD(cmd, function_name)                               \
        case cmd:                                                           \
        {                                                                   \
            cmd##_PARAMS params;                                            \
            if (copy_from_user(&params, (void __user*)arg, sizeof(params))) \
                return -EFAULT;                                             \
                                                                            \
            params.rmStatus = function_name(&params, filp);                 \
            if (copy_to_user((void __user*)arg, &params, sizeof(params)))   \
                return -EFAULT;                                             \
                                                                            \
            return 0;                                                       \
        }

        switch (cmd)
        {
            case UVM_DEINITIALIZE:
                return 0;

            UVM_ROUTE_CMD(UVM_INITIALIZE, uvm_unsupported_initialize);
        }
    #undef UVM_ROUTE_CMD

    return -EINVAL;
}

static int uvm_unsupported_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static int uvm_unsupported_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static const struct file_operations uvm_unsupported_fops =
{
    .open            = uvm_unsupported_open,
    .release         = uvm_unsupported_release,

#if defined(NV_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl           = uvm_unsupported_unlocked_ioctl,
#endif
#if defined(NV_FILE_OPERATIONS_HAS_UNLOCKED_IOCTL)
    .unlocked_ioctl  = uvm_unsupported_unlocked_ioctl,
#endif

#if NVCPU_IS_X86_64 && defined(NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL)
    .compat_ioctl    = uvm_unsupported_unlocked_ioctl,
#endif
    .owner           = THIS_MODULE,
};

static int __init uvm_unsupported_module_init(void)
{
    dev_t uvm_dev;

    int ret = alloc_chrdev_region(&g_uvm_base_dev,
                                  0,
                                  NVIDIA_UVM_NUM_MINOR_DEVICES,
                                  NVIDIA_UVM_DEVICE_NAME);
    if (ret != 0) {
        printk(KERN_ERR "nvidia-uvm-unsupported: alloc_chrdev_region failed, "
                        "therefore, failing module_init. %d\n", ret);
        goto error;
    }

    uvm_dev = MKDEV(MAJOR(g_uvm_base_dev), NVIDIA_UVM_PRIMARY_MINOR_NUMBER);
    cdev_init(&g_uvm_cdev, &uvm_unsupported_fops);
    g_uvm_cdev.owner = THIS_MODULE;

    ret = cdev_add(&g_uvm_cdev, uvm_dev, 1);
    if (ret != 0) {
        printk(KERN_ERR "cdev_add (major %u, minor %u) failed: %d\n",
               MAJOR(uvm_dev), MINOR(uvm_dev), ret);

        goto error;
    }

    printk(KERN_ERR "Loaded a UVM driver shell (unsupported mode) major device number %d\n",
           MAJOR(g_uvm_base_dev));

    return 0;

error:
    unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);

    return ret;
}

static void __exit uvm_unsupported_exit(void)
{
    cdev_del(&g_uvm_cdev);

    unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);

    printk(KERN_ERR "Unloaded the UVM driver shell (unsupported mode) major device number %d\n",
           MAJOR(g_uvm_base_dev));
}

module_init(uvm_unsupported_module_init);
module_exit(uvm_unsupported_exit);

MODULE_LICENSE("MIT");
MODULE_INFO(supported, "external");

