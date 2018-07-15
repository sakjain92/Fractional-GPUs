/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_ATOMIC_MODESET_AVAILABLE)

#include "nvidia-drm-gem-nvkms-memory.h"
#include "nvidia-drm-ioctl.h"

#include "nv-mm.h"

static void __nv_drm_gem_nvkms_memory_free(struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_device *nv_dev = nv_gem->nv_dev;
    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory =
        to_nv_nvkms_memory(nv_gem);

    if (nv_nvkms_memory->dumb_buffer) {
        if (nv_nvkms_memory->pWriteCombinedIORemapAddress != NULL) {
            iounmap(nv_nvkms_memory->pWriteCombinedIORemapAddress);
        }

        nvKms->unmapMemory(nv_dev->pDevice,
                           nv_nvkms_memory->pMemory,
                           NVKMS_KAPI_MAPPING_TYPE_USER,
                           nv_nvkms_memory->pPhysicalAddress);
    }

    /* Free NvKmsKapiMemory handle associated with this gem object */

    nvKms->freeMemory(nv_dev->pDevice, nv_nvkms_memory->pMemory);

    nv_drm_free(nv_nvkms_memory);
}

const struct nv_drm_gem_object_funcs nv_gem_nvkms_memory_ops = {
    .free = __nv_drm_gem_nvkms_memory_free,
};

int nv_drm_dumb_create(
    struct drm_file *file_priv,
    struct drm_device *dev, struct drm_mode_create_dumb *args)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory;
    int ret = 0;

    args->pitch = roundup(args->width * ((args->bpp + 7) >> 3),
                          nv_dev->pitchAlignment);

    args->size = args->height * args->pitch;

    /* Core DRM requires gem object size to be aligned with PAGE_SIZE */

    args->size = roundup(args->size, PAGE_SIZE);

    if ((nv_nvkms_memory =
            nv_drm_calloc(1, sizeof(*nv_nvkms_memory))) == NULL) {
        ret = -ENOMEM;
        goto fail;
    }

    if ((nv_nvkms_memory->pMemory =
            nvKms->allocateMemory(nv_dev->pDevice, args->size)) == NULL) {
        ret = -ENOMEM;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate NvKmsKapiMemory for dumb object of size %llu",
            args->size);
        goto nvkms_alloc_memory_failed;
    }

    if (!nvKms->mapMemory(nv_dev->pDevice,
                          nv_nvkms_memory->pMemory,
                          NVKMS_KAPI_MAPPING_TYPE_USER,
                          &nv_nvkms_memory->pPhysicalAddress)) {
        ret = -ENOMEM;

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to map NvKmsKapiMemory 0x%p",
            nv_nvkms_memory->pMemory);
        goto nvkms_map_memory_failed;
    }

    nv_nvkms_memory->pWriteCombinedIORemapAddress = ioremap_wc(
        (uintptr_t)nv_nvkms_memory->pPhysicalAddress,
        args->size);

    nv_nvkms_memory->dumb_buffer = true;

    nv_drm_gem_object_init(nv_dev,
                           &nv_nvkms_memory->base,
                           &nv_gem_nvkms_memory_ops,
                           args->size,
                           false);

    return nv_drm_gem_handle_create_drop_reference(file_priv,
                                                   &nv_nvkms_memory->base,
                                                   &args->handle);

nvkms_map_memory_failed:

    nvKms->freeMemory(nv_dev->pDevice, nv_nvkms_memory->pMemory);

nvkms_alloc_memory_failed:
    nv_drm_free(nv_nvkms_memory);

fail:
    return ret;
}

int nv_drm_gem_import_nvkms_memory_ioctl(struct drm_device *dev,
                                         void *data, struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_gem_import_nvkms_memory_params *p = data;
    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory;
    int ret;

    if (!drm_core_check_feature(dev, DRIVER_MODESET)) {
        ret = -EINVAL;
        goto failed;
    }

    if ((nv_nvkms_memory =
            nv_drm_calloc(1, sizeof(*nv_nvkms_memory))) == NULL) {
        ret = -ENOMEM;
        goto failed;
    }

    nv_nvkms_memory->pMemory =
        nvKms->importMemory(nv_dev->pDevice,
                            p->mem_size,
                            p->nvkms_params_ptr,
                            p->nvkms_params_size);

    if (nv_nvkms_memory->pMemory == NULL) {
        ret = -EINVAL;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to import NVKMS memory to GEM object");
        goto nvkms_import_memory_failed;
    }

    nv_nvkms_memory->pPhysicalAddress = NULL;
    nv_nvkms_memory->pWriteCombinedIORemapAddress = NULL;

    nv_drm_gem_object_init(nv_dev,
                           &nv_nvkms_memory->base,
                           &nv_gem_nvkms_memory_ops,
                           p->mem_size,
                           false);

    return nv_drm_gem_handle_create_drop_reference(filep,
                                                   &nv_nvkms_memory->base,
                                                   &p->handle);

nvkms_import_memory_failed:
    nv_drm_free(nv_nvkms_memory);

failed:
    return ret;
}

int nv_drm_dumb_map_offset(struct drm_file *file,
                           struct drm_device *dev, uint32_t handle,
                           uint64_t *offset)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory;
    int ret = -EINVAL;

    if ((nv_nvkms_memory = nv_drm_gem_object_nvkms_memory_lookup(
                    dev,
                    file,
                    handle)) == NULL) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lookup gem object for mapping: 0x%08x",
            handle);
        goto done;
    }

    if (!nv_nvkms_memory->dumb_buffer) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Invalid gem object type for mapping: 0x%08x",
            handle);
        goto done;
    }

    ret = nv_drm_gem_create_mmap_offset(&nv_nvkms_memory->base, offset);

done:
    if (nv_nvkms_memory != NULL) {
        nv_drm_gem_object_unreference_unlocked(&nv_nvkms_memory->base);
    }

    return ret;
}

/* XXX Move these vma operations to os layer */

static int __nv_drm_vma_fault(struct vm_area_struct *vma,
                              struct vm_fault *vmf)
{
    unsigned long address = nv_page_fault_va(vmf);
    struct drm_gem_object *gem = vma->vm_private_data;
    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory = to_nv_nvkms_memory(
        to_nv_gem_object(gem));
    unsigned long page_offset, pfn;
    int ret = -EINVAL;

    pfn = (unsigned long)(uintptr_t)nv_nvkms_memory->pPhysicalAddress;
    pfn >>= PAGE_SHIFT;

    page_offset = vmf->pgoff - drm_vma_node_start(&gem->vma_node);

    ret = vm_insert_pfn(vma, address, pfn + page_offset);

    switch (ret) {
        case 0:
        case -EBUSY:
            /*
             * EBUSY indicates that another thread already handled
             * the faulted range.
             */
            return VM_FAULT_NOPAGE;
        case -ENOMEM:
            return VM_FAULT_OOM;
        default:
            WARN_ONCE(1, "Unhandled error in %s: %d\n", __FUNCTION__, ret);
            break;
    }

    return VM_FAULT_SIGBUS;
}

/*
 * Note that nv_drm_vma_fault() can be called for different or same
 * ranges of the same drm_gem_object simultaneously.
 */

#if defined(NV_VM_OPS_FAULT_REMOVED_VMA_ARG)
static int nv_drm_vma_fault(struct vm_fault *vmf)
{
    return __nv_drm_vma_fault(vmf->vma, vmf);
}
#else
static int nv_drm_vma_fault(struct vm_area_struct *vma,
                                struct vm_fault *vmf)
{
    return __nv_drm_vma_fault(vma, vmf);
}
#endif

const struct vm_operations_struct nv_drm_gem_vma_ops = {
    .open  = drm_gem_vm_open,
    .fault = nv_drm_vma_fault,
    .close = drm_gem_vm_close,
};

#endif
