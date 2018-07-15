/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#include "nvidia-drm-conftest.h" /* NV_DRM_ATOMIC_MODESET_AVAILABLE */

#if defined(NV_DRM_ATOMIC_MODESET_AVAILABLE)

#include "nvidia-drm-priv.h"
#include "nvidia-drm-ioctl.h"
#include "nvidia-drm-fb.h"
#include "nvidia-drm-utils.h"
#include "nvidia-drm-gem.h"

#include <drm/drm_crtc_helper.h>

static void nv_drm_framebuffer_destroy(struct drm_framebuffer *fb)
{
    struct nv_drm_device *nv_dev = to_nv_device(fb->dev);

    struct nv_drm_framebuffer *nv_fb = to_nv_framebuffer(fb);

    /* Unreference gem object */

    nv_drm_gem_object_unreference_unlocked(&nv_fb->nv_nvkms_memory->base);

    /* Cleaup core framebuffer object */

    drm_framebuffer_cleanup(fb);

    /* Free NvKmsKapiSurface associated with this framebuffer object */

    nvKms->destroySurface(nv_dev->pDevice, nv_fb->pSurface);

    /* Free framebuffer */

    nv_drm_free(nv_fb);
}

static int
nv_drm_framebuffer_create_handle(struct drm_framebuffer *fb,
                                 struct drm_file *file, unsigned int *handle)
{
    struct nv_drm_framebuffer *nv_fb = to_nv_framebuffer(fb);

    return nv_drm_gem_handle_create(file,
                                    &nv_fb->nv_nvkms_memory->base,
                                    handle);
}

static struct drm_framebuffer_funcs nv_framebuffer_funcs = {
    .destroy       = nv_drm_framebuffer_destroy,
    .create_handle = nv_drm_framebuffer_create_handle,
};

struct drm_framebuffer *nv_drm_internal_framebuffer_create(
    struct drm_device *dev,
    struct drm_file *file,
    struct drm_mode_fb_cmd2 *cmd)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    struct nv_drm_framebuffer *nv_fb;

    struct nv_drm_gem_nvkms_memory *nv_nvkms_memory;

    enum NvKmsSurfaceMemoryFormat format;

    int ret;

    if (!drm_format_to_nvkms_format(cmd->pixel_format, &format)) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Unsupported drm pixel format 0x%08x", cmd->pixel_format);
        return ERR_PTR(-EINVAL);
    }

    /*
     * In case of planar formats, this ioctl allows up to 4 buffer objects with
     * offsets and pitches per plane.
     *
     * We don't support any planar format, pick up first buffer only.
     */

    if ((nv_nvkms_memory = nv_drm_gem_object_nvkms_memory_lookup(
                    dev,
                    file,
                    cmd->handles[0])) == NULL) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to find gem object of type nvkms memory");
        return ERR_PTR(-ENOENT);
    }

    /* Allocate memory for the framebuffer object */

    nv_fb = nv_drm_calloc(1, sizeof(*nv_fb));

    if (nv_fb == NULL) {
        ret = -ENOMEM;

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate memory for framebuffer obejct");
        goto failed_fb_create;
    }

    nv_fb->nv_nvkms_memory = nv_nvkms_memory;

    /* Fill out framebuffer metadata from the userspace fb creation request */

    drm_helper_mode_fill_fb_struct(
        #if defined(NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_DEV_ARG)
        dev,
        #endif
        &nv_fb->base,
        cmd);

    /* Initialize the base framebuffer object and add it to drm subsystem */

    ret = drm_framebuffer_init(dev, &nv_fb->base, &nv_framebuffer_funcs);

    if (ret != 0) {
        NV_DRM_DEV_LOG_ERR(nv_dev, "Failed to initialize framebuffer object");
        goto failed_fb_init;
    }

    /* Create NvKmsKapiSurface */

    nv_fb->pSurface = nvKms->createSurface(
        nv_dev->pDevice, nv_nvkms_memory->pMemory,
        format, nv_fb->base.width, nv_fb->base.height, nv_fb->base.pitches[0]);

    if (nv_fb->pSurface == NULL) {
        ret = -EINVAL;

        NV_DRM_DEV_LOG_ERR(nv_dev, "Failed to create NvKmsKapiSurface");
        goto failed_nvkms_create_surface;
    }

    return &nv_fb->base;

failed_nvkms_create_surface:

    drm_framebuffer_cleanup(&nv_fb->base);

failed_fb_init:

    nv_drm_free(nv_fb);

failed_fb_create:

    nv_drm_gem_object_unreference_unlocked(&nv_nvkms_memory->base);

    return ERR_PTR(ret);
}

#endif
