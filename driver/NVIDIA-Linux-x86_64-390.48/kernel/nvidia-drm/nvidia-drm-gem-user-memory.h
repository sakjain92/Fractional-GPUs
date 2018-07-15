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

#ifndef __NVIDIA_DRM_GEM_USER_MEMORY_H__
#define __NVIDIA_DRM_GEM_USER_MEMORY_H__

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#include "nvidia-drm-gem.h"

struct nv_drm_gem_user_memory {
    struct nv_drm_gem_object base;
    struct page **pages;
    unsigned long pages_count;
};

static inline struct nv_drm_gem_user_memory *to_nv_user_memory(
    struct nv_drm_gem_object *nv_gem)
{
    if (nv_gem != NULL) {
        return container_of(nv_gem, struct nv_drm_gem_user_memory, base);
    }

    return NULL;
}

int nv_drm_gem_import_userspace_memory_ioctl(struct drm_device *dev,
                                             void *data, struct drm_file *filep);

#endif

#endif /* __NVIDIA_DRM_GEM_USER_MEMORY_H__ */
