/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __NVIDIA_DRM_GEM_H__
#define __NVIDIA_DRM_GEM_H__

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#include "nvidia-drm-priv.h"

#include <drm/drmP.h>
#include "nvkms-kapi.h"

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)

#include "nvidia-dma-fence-helper.h"

#endif

struct nv_drm_gem_object;

struct nv_drm_gem_object_funcs {
    void (*free)(struct nv_drm_gem_object *nv_gem);
    struct sg_table *(*prime_get_sg_table)(struct nv_drm_gem_object *nv_gem);
    void *(*prime_vmap)(struct nv_drm_gem_object *nv_gem);
    void (*prime_vunmap)(struct nv_drm_gem_object *nv_gem, void *address);
};

struct nv_drm_gem_object {
    struct drm_gem_object base;

    struct nv_drm_device *nv_dev;
    const struct nv_drm_gem_object_funcs *ops;

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
    struct reservation_object resv;
#endif

    bool prime:1;
};

static inline struct nv_drm_gem_object *to_nv_gem_object(
    struct drm_gem_object *gem)
{
    if (gem != NULL) {
        return container_of(gem, struct nv_drm_gem_object, base);
    }

    return NULL;
}

static inline int nv_drm_gem_handle_create_drop_reference(
    struct drm_file *file_priv,
    struct nv_drm_gem_object *nv_gem,
    uint32_t *handle)
{
    int ret = drm_gem_handle_create(file_priv, &nv_gem->base, handle);

    /* drop reference from allocate - handle holds it now */

    drm_gem_object_unreference_unlocked(&nv_gem->base);

    return ret;
}

static inline
void nv_drm_gem_object_init(struct nv_drm_device *nv_dev,
                            struct nv_drm_gem_object *nv_gem,
                            const struct nv_drm_gem_object_funcs * const ops,
                            size_t size,
                            bool prime)
{
    struct drm_device *dev = nv_dev->dev;

    nv_gem->nv_dev = nv_dev;
    nv_gem->prime = prime;
    nv_gem->ops = ops;

    /* Initialize the gem object */

    drm_gem_private_object_init(dev, &nv_gem->base, size);

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
    reservation_object_init(&nv_gem->resv);
#endif
}

static inline int nv_drm_gem_create_mmap_offset(
    struct nv_drm_gem_object *nv_gem,
    uint64_t *offset)
{
    int ret;

    if ((ret = drm_gem_create_mmap_offset(&nv_gem->base)) < 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_gem->nv_dev,
            "drm_gem_create_mmap_offset failed with error code %d",
            ret);
        goto done;
    }

    *offset = drm_vma_node_offset_addr(&nv_gem->base.vma_node);

done:

    return ret;
}

void nv_drm_gem_free(struct drm_gem_object *gem);

struct dma_buf *nv_drm_gem_prime_export(struct drm_device *dev,
                                        struct drm_gem_object *gem, int flags);

static inline struct nv_drm_gem_object *nv_drm_gem_object_lookup(
    struct drm_device *dev,
    struct drm_file *filp,
    u32 handle)
{
    #if defined(NV_DRM_GEM_OBJECT_LOOKUP_PRESENT)
        #if (NV_DRM_GEM_OBJECT_LOOKUP_ARGUMENT_COUNT == 3)
            return to_nv_gem_object(drm_gem_object_lookup(dev, filp, handle));
        #elif (NV_DRM_GEM_OBJECT_LOOKUP_ARGUMENT_COUNT == 2)
            return to_nv_gem_object(drm_gem_object_lookup(filp, handle));
        #else
            #error "Unknow arguments count of drm_gem_object_lookup()"
        #endif
    #else
        #error "drm_gem_object_lookup() is not defined"
    #endif
}

static inline void
nv_drm_gem_object_unreference_unlocked(struct nv_drm_gem_object *nv_gem)
{
    drm_gem_object_unreference_unlocked(&nv_gem->base);
}

static inline void
nv_drm_gem_object_unreference(struct nv_drm_gem_object *nv_gem)
{
    drm_gem_object_unreference(&nv_gem->base);
}

static inline int nv_drm_gem_handle_create(struct drm_file *filp,
                                           struct nv_drm_gem_object *nv_gem,
                                           uint32_t *handle)
{
    return drm_gem_handle_create(filp, &nv_gem->base, handle);
}

struct sg_table *nv_drm_gem_prime_get_sg_table(struct drm_gem_object *gem);

void *nv_drm_gem_prime_vmap(struct drm_gem_object *gem);

void nv_drm_gem_prime_vunmap(struct drm_gem_object *gem, void *address);

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
struct reservation_object* nv_drm_gem_prime_res_obj(struct drm_gem_object *obj);
#endif

#endif /* NV_DRM_AVAILABLE */

#endif /* __NVIDIA_DRM_GEM_H__ */
