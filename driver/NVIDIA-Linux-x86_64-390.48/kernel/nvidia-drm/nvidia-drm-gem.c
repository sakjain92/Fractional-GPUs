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

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#include "nvidia-drm-priv.h"
#include "nvidia-drm-ioctl.h"
#include "nvidia-drm-prime-fence.h"
#include "nvidia-drm-gem.h"

void nv_drm_gem_free(struct drm_gem_object *gem)
{
    struct drm_device *dev = gem->dev;
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    WARN_ON(!mutex_is_locked(&dev->struct_mutex));

    /* Cleanup core gem object */

    drm_gem_object_release(&nv_gem->base);

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
    reservation_object_fini(&nv_gem->resv);
#endif

    nv_gem->ops->free(nv_gem);
}

struct dma_buf *nv_drm_gem_prime_export(struct drm_device *dev,
                                        struct drm_gem_object *gem, int flags)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (!nv_gem->prime) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Gem object 0x%p is not suitable to export", gem);
        return ERR_PTR(-EINVAL);
    }

    return drm_gem_prime_export(dev, gem, flags);
}

struct sg_table *nv_drm_gem_prime_get_sg_table(struct drm_gem_object *gem)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_get_sg_table != NULL) {
        return nv_gem->ops->prime_get_sg_table(nv_gem);
    }

    return ERR_PTR(-ENOTSUPP);
}

void *nv_drm_gem_prime_vmap(struct drm_gem_object *gem)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_vmap != NULL) {
        return nv_gem->ops->prime_vmap(nv_gem);
    }

    return ERR_PTR(-ENOTSUPP);
}

void nv_drm_gem_prime_vunmap(struct drm_gem_object *gem, void *address)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_vunmap != NULL) {
        nv_gem->ops->prime_vunmap(nv_gem, address);
    }
}

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
struct reservation_object* nv_drm_gem_prime_res_obj(struct drm_gem_object *obj)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(obj);

    return &nv_gem->resv;
}
#endif

#endif /* NV_DRM_AVAILABLE */
