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

#ifndef __NVIDIA_DRM_HELPER_H__
#define __NVIDIA_DRM_HELPER_H__

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#include <drm/drmP.h>

/*
 * drm_dev_unref() has been added and drm_dev_free() removed by commit -
 *
 *      2014-01-29: 099d1c290e2ebc3b798961a6c177c3aef5f0b789
 */
static inline void nv_drm_dev_free(struct drm_device *dev)
{
#if defined(NV_DRM_DEV_UNREF_PRESENT)
    drm_dev_unref(dev);
#else
    drm_dev_free(dev);
#endif
}

#if defined(NV_DRM_ATOMIC_MODESET_AVAILABLE)

#include <drm/drm_atomic.h>
#include <drm/drm_atomic_helper.h>

int nv_drm_atomic_set_mode_for_crtc(struct drm_crtc_state *state,
                                    struct drm_display_mode *mode);

void nv_drm_atomic_clean_old_fb(struct drm_device *dev,
                                unsigned plane_mask,
                                int ret);

int nv_drm_atomic_helper_disable_all(struct drm_device *dev,
                                     struct drm_modeset_acquire_ctx *ctx);

static inline int nv_drm_atomic_helper_set_config(
    struct drm_mode_set *set,
    struct drm_modeset_acquire_ctx *ctx)
{
#if defined(NV_DRM_ATOMIC_HELPER_SET_CONFIG_PRESENT)
    #if defined(NV_DRM_ATOMIC_HELPER_SET_CONFIG_HAS_CTX_ARG)
        return drm_atomic_helper_set_config(set, ctx);
    #else
        return drm_atomic_helper_set_config(set);
    #endif
#else
    #error "drm_atomic_helper_set_config not present!"
#endif
}

/*
 * for_each_connector_in_state(), for_each_crtc_in_state() and
 * for_each_plane_in_state() were added by kernel commit
 * df63b9994eaf942afcdb946d27a28661d7dfbf2a which was Signed-off-by:
 *      Ander Conselvan de Oliveira <ander.conselvan.de.oliveira@intel.com>
 *      Daniel Vetter <daniel.vetter@ffwll.ch>
 *
 * for_each_connector_in_state(), for_each_crtc_in_state() and
 * for_each_plane_in_state() were copied from
 *      include/drm/drm_atomic.h @
 *      21a01abbe32a3cbeb903378a24e504bfd9fe0648
 * which has the following copyright and license information:
 *
 * Copyright (C) 2014 Red Hat
 * Copyright (C) 2014 Intel Corp.
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
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 * Rob Clark <robdclark@gmail.com>
 * Daniel Vetter <daniel.vetter@ffwll.ch>
 */

/**
 * nv_drm_for_each_connector_in_state - iterate over all connectors in an
 * atomic update
 * @__state: &struct drm_atomic_state pointer
 * @connector: &struct drm_connector iteration cursor
 * @connector_state: &struct drm_connector_state iteration cursor
 * @__i: int iteration cursor, for macro-internal use
 *
 * This iterates over all connectors in an atomic update. Note that before the
 * software state is committed (by calling drm_atomic_helper_swap_state(), this
 * points to the new state, while afterwards it points to the old state. Due to
 * this tricky confusion this macro is deprecated.
 */
#if !defined(NV_DRM_OLD_ATOMIC_STATE_ITERATORS_PRESENT)
#define nv_drm_for_each_connector_in_state(__state,                         \
                                           connector, connector_state, __i) \
       for ((__i) = 0;                                                      \
            (__i) < (__state)->num_connector &&                             \
            ((connector) = (__state)->connectors[__i].ptr,                  \
            (connector_state) = (__state)->connectors[__i].state, 1);       \
            (__i)++)                                                        \
               for_each_if (connector)
#else
#define nv_drm_for_each_connector_in_state(__state,                         \
                                           connector, connector_state, __i) \
    for_each_connector_in_state(__state, connector, connector_state, __i)
#endif


/**
 * nv_drm_for_each_crtc_in_state - iterate over all CRTCs in an atomic update
 * @__state: &struct drm_atomic_state pointer
 * @crtc: &struct drm_crtc iteration cursor
 * @crtc_state: &struct drm_crtc_state iteration cursor
 * @__i: int iteration cursor, for macro-internal use
 *
 * This iterates over all CRTCs in an atomic update. Note that before the
 * software state is committed (by calling drm_atomic_helper_swap_state(), this
 * points to the new state, while afterwards it points to the old state. Due to
 * this tricky confusion this macro is deprecated.
 */
#if !defined(NV_DRM_OLD_ATOMIC_STATE_ITERATORS_PRESENT)
#define nv_drm_for_each_crtc_in_state(__state, crtc, crtc_state, __i) \
       for ((__i) = 0;                                                \
            (__i) < (__state)->dev->mode_config.num_crtc &&           \
            ((crtc) = (__state)->crtcs[__i].ptr,                      \
            (crtc_state) = (__state)->crtcs[__i].state, 1);           \
            (__i)++)                                                  \
               for_each_if (crtc_state)
#else
#define nv_drm_for_each_crtc_in_state(__state, crtc, crtc_state, __i) \
    for_each_crtc_in_state(__state, crtc, crtc_state, __i)
#endif

/**
 * nv_drm_for_each_plane_in_state - iterate over all planes in an atomic update
 * @__state: &struct drm_atomic_state pointer
 * @plane: &struct drm_plane iteration cursor
 * @plane_state: &struct drm_plane_state iteration cursor
 * @__i: int iteration cursor, for macro-internal use
 *
 * This iterates over all planes in an atomic update. Note that before the
 * software state is committed (by calling drm_atomic_helper_swap_state(), this
 * points to the new state, while afterwards it points to the old state. Due to
 * this tricky confusion this macro is deprecated.
 */
#if !defined(NV_DRM_OLD_ATOMIC_STATE_ITERATORS_PRESENT)
#define nv_drm_for_each_plane_in_state(__state, plane, plane_state, __i) \
       for ((__i) = 0;                                                   \
            (__i) < (__state)->dev->mode_config.num_total_plane &&       \
            ((plane) = (__state)->planes[__i].ptr,                       \
            (plane_state) = (__state)->planes[__i].state, 1);            \
            (__i)++)                                                     \
               for_each_if (plane_state)
#else
#define nv_drm_for_each_plane_in_state(__state, plane, plane_state, __i) \
    for_each_plane_in_state(__state, plane, plane_state, __i)
#endif

static inline struct drm_crtc *nv_drm_crtc_find(struct drm_device *dev,
    uint32_t id)
{
#if defined(NV_DRM_MODE_OBJECT_FIND_HAS_FILE_PRIV_ARG)
    return drm_crtc_find(dev, NULL /* file_priv */, id);
#else
    return drm_crtc_find(dev, id);
#endif
}

static inline struct drm_encoder *nv_drm_encoder_find(struct drm_device *dev,
    uint32_t id)
{
#if defined(NV_DRM_MODE_OBJECT_FIND_HAS_FILE_PRIV_ARG)
    return drm_encoder_find(dev, NULL /* file_priv */, id);
#else
    return drm_encoder_find(dev, id);
#endif
}

#endif /* defined(NV_DRM_ATOMIC_MODESET_AVAILABLE) */

#endif /* defined(NV_DRM_AVAILABLE) */

#endif /* __NVIDIA_DRM_HELPER_H__ */
