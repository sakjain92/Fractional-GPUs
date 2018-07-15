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

#include "nvidia-drm-helper.h"
#include "nvidia-drm-priv.h"
#include "nvidia-drm-crtc.h"
#include "nvidia-drm-connector.h"
#include "nvidia-drm-encoder.h"
#include "nvidia-drm-utils.h"
#include "nvidia-drm-fb.h"
#include "nvidia-drm-ioctl.h"

#include <drm/drm_crtc_helper.h>
#include <drm/drm_plane_helper.h>

#include <drm/drm_atomic.h>
#include <drm/drm_atomic_helper.h>

static const u32 nv_default_supported_plane_drm_formats[] = {
    DRM_FORMAT_ARGB1555,
    DRM_FORMAT_XRGB1555,
    DRM_FORMAT_RGB565,
    DRM_FORMAT_ARGB8888,
    DRM_FORMAT_XRGB8888,
    DRM_FORMAT_ABGR2101010,
    DRM_FORMAT_XBGR2101010,
};

static const u32 nv_supported_cursor_plane_drm_formats[] = {
    DRM_FORMAT_ARGB1555,
    DRM_FORMAT_ARGB8888,
};

static void nv_drm_plane_destroy(struct drm_plane *plane)
{
    /* plane->state gets freed here */
    drm_plane_cleanup(plane);

    nv_drm_free(plane);
}

static inline void
plane_req_config_disable(struct NvKmsKapiPlaneRequestedConfig *req_config)
{
    req_config->config.surface = NULL;
    req_config->flags.surfaceChanged = NV_TRUE;
}

static void
plane_req_config_update(struct drm_plane_state *plane_state,
                        struct NvKmsKapiPlaneRequestedConfig *req_config)
{
    struct NvKmsKapiPlaneConfig old_config = req_config->config;

    if (plane_state->fb == NULL) {
        plane_req_config_disable(req_config);
        return;
    }

    *req_config = (struct NvKmsKapiPlaneRequestedConfig) {
        .config = {
            .surface = to_nv_framebuffer(plane_state->fb)->pSurface,

            /* Source values are 16.16 fixed point */
            .srcX = plane_state->src_x >> 16,
            .srcY = plane_state->src_y >> 16,
            .srcWidth  = plane_state->src_w >> 16,
            .srcHeight = plane_state->src_h >> 16,

            .dstX = plane_state->crtc_x,
            .dstY = plane_state->crtc_y,
            .dstWidth  = plane_state->crtc_w,
            .dstHeight = plane_state->crtc_h,
        },
    };

    /*
     * Unconditionally mark the surface as changed, even if nothing changed,
     * so that we always get a flip event: a DRM client may flip with
     * the same surface and wait for a flip event.
     */
    req_config->flags.surfaceChanged = NV_TRUE;

    if (old_config.surface == NULL &&
        old_config.surface != req_config->config.surface) {
        req_config->flags.srcXYChanged = NV_TRUE;
        req_config->flags.srcWHChanged = NV_TRUE;
        req_config->flags.dstXYChanged = NV_TRUE;
        req_config->flags.dstWHChanged = NV_TRUE;
        return;
    }

    req_config->flags.srcXYChanged =
        old_config.srcX != req_config->config.srcX ||
        old_config.srcY != req_config->config.srcY;

    req_config->flags.srcWHChanged =
        old_config.srcWidth != req_config->config.srcWidth ||
        old_config.srcHeight != req_config->config.srcHeight;

    req_config->flags.dstXYChanged =
        old_config.dstX != req_config->config.dstX ||
        old_config.dstY != req_config->config.dstY;

    req_config->flags.dstWHChanged =
        old_config.dstWidth != req_config->config.dstWidth ||
        old_config.dstHeight != req_config->config.dstHeight;
}

static int nv_drm_plane_atomic_check(struct drm_plane *plane,
                                     struct drm_plane_state *plane_state)
{
    int i;
    struct drm_crtc *crtc;
    struct drm_crtc_state *crtc_state;
    NvKmsKapiPlaneType type;

    if (NV_DRM_WARN(!drm_plane_type_to_nvkms_plane_type(plane->type, &type))) {
        goto done;
    }

    nv_drm_for_each_crtc_in_state(plane_state->state, crtc, crtc_state, i) {
        struct nv_drm_crtc_state *nv_crtc_state = to_nv_crtc_state(crtc_state);
        struct NvKmsKapiHeadRequestedConfig *head_req_config =
            &nv_crtc_state->req_config;
        struct NvKmsKapiPlaneRequestedConfig *plane_requested_config =
            &head_req_config->planeRequestedConfig[type];

        if (plane->state->crtc == crtc &&
            plane->state->crtc != plane_state->crtc) {
            plane_req_config_disable(plane_requested_config);
            continue;
        }

        if (plane_state->crtc == crtc) {
            plane_req_config_update(plane_state,
                                    plane_requested_config);
        }
    }

done:
    return 0;
}

static void nv_drm_plane_atomic_update(struct drm_plane *plane,
                                       struct drm_plane_state *old_state)
{
}

static void nv_drm_plane_atomic_disable(struct drm_plane *plane,
                                        struct drm_plane_state *old_state)
{
}

static const struct drm_plane_funcs nv_plane_funcs = {
    .update_plane           = drm_atomic_helper_update_plane,
    .disable_plane          = drm_atomic_helper_disable_plane,
    .destroy                = nv_drm_plane_destroy,
    .reset                  = drm_atomic_helper_plane_reset,
    .atomic_duplicate_state = drm_atomic_helper_plane_duplicate_state,
    .atomic_destroy_state   = drm_atomic_helper_plane_destroy_state,
};

static const struct drm_plane_helper_funcs nv_plane_helper_funcs = {
    .atomic_check   = nv_drm_plane_atomic_check,
    .atomic_update  = nv_drm_plane_atomic_update,
    .atomic_disable = nv_drm_plane_atomic_disable,
};

static void nv_drm_crtc_destroy(struct drm_crtc *crtc)
{
    struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

    drm_crtc_cleanup(crtc);

    nv_drm_free(nv_crtc);
}

static inline void
__nv_drm_atomic_helper_crtc_destroy_state(struct drm_crtc *crtc,
                                          struct drm_crtc_state *crtc_state)
{
#if defined(NV_DRM_ATOMIC_HELPER_CRTC_DESTROY_STATE_HAS_CRTC_ARG)
    __drm_atomic_helper_crtc_destroy_state(crtc, crtc_state);
#else
    __drm_atomic_helper_crtc_destroy_state(crtc_state);
#endif
}

static inline void nv_drm_crtc_duplicate_req_head_modeset_config(
    const struct NvKmsKapiHeadRequestedConfig *old,
    struct NvKmsKapiHeadRequestedConfig *new)
{
    uint32_t i;

    /*
     * Do not duplicate fields like 'modeChanged' flags expressing delta changed
     * in new configuration with respect to previous/old configuration because
     * there is no change in new configuration yet with respect
     * to older one!
     */
    *new = (struct NvKmsKapiHeadRequestedConfig) {
        .modeSetConfig = old->modeSetConfig,
    };

    for (i = 0; i < ARRAY_SIZE(old->planeRequestedConfig); i++) {
        new->planeRequestedConfig[i] = (struct NvKmsKapiPlaneRequestedConfig) {
            .config = old->planeRequestedConfig[i].config,
        };
    }
}

/**
 * nv_drm_atomic_crtc_duplicate_state - crtc state duplicate hook
 * @crtc: DRM crtc
 *
 * Allocate and accosiate flip state with DRM crtc state, this flip state will
 * be getting consumed at the time of atomic update commit to hardware by
 * nv_drm_atomic_helper_commit_tail().
 */
static struct drm_crtc_state*
nv_drm_atomic_crtc_duplicate_state(struct drm_crtc *crtc)
{
    struct nv_drm_crtc_state *nv_state = nv_drm_calloc(1, sizeof(*nv_state));

    if (nv_state == NULL) {
        return NULL;
    }

    __drm_atomic_helper_crtc_duplicate_state(crtc, &nv_state->base);

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    if ((nv_state->nv_flip =
            nv_drm_calloc(1, sizeof(*(nv_state->nv_flip)))) == NULL) {
        __nv_drm_atomic_helper_crtc_destroy_state(crtc, &nv_state->base);
        nv_drm_free(nv_state);
        return NULL;
    }

    INIT_LIST_HEAD(&nv_state->nv_flip->list_entry);
#endif

    nv_drm_crtc_duplicate_req_head_modeset_config(
        &(to_nv_crtc_state(crtc->state)->req_config),
        &nv_state->req_config);

    return &nv_state->base;
}

/**
 * nv_drm_atomic_crtc_destroy_state - crtc state destroy hook
 * @crtc: DRM crtc
 * @state: DRM crtc state object to destroy
 *
 * Destroy flip state associated with the given crtc state if it haven't get
 * consumed because failure of atomic commit.
 */
static void nv_drm_atomic_crtc_destroy_state(struct drm_crtc *crtc,
                                             struct drm_crtc_state *state)
{
    struct nv_drm_crtc_state *nv_state = to_nv_crtc_state(state);

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    if (nv_state->nv_flip != NULL) {
        nv_drm_free(nv_state->nv_flip);
        nv_state->nv_flip = NULL;
    }
#endif

    __nv_drm_atomic_helper_crtc_destroy_state(crtc, &nv_state->base);

    nv_drm_free(nv_state);
}

static struct drm_crtc_funcs nv_crtc_funcs = {
    .set_config             = drm_atomic_helper_set_config,
    .page_flip              = drm_atomic_helper_page_flip,
    .reset                  = drm_atomic_helper_crtc_reset,
    .destroy                = nv_drm_crtc_destroy,
    .atomic_duplicate_state = nv_drm_atomic_crtc_duplicate_state,
    .atomic_destroy_state   = nv_drm_atomic_crtc_destroy_state,
};

/*
 * In kernel versions before the addition of
 * drm_crtc_state::connectors_changed, connector changes were
 * reflected in drm_crtc_state::mode_changed.
 */
static inline bool
nv_drm_crtc_state_connectors_changed(struct drm_crtc_state *crtc_state)
{
#if defined(NV_DRM_CRTC_STATE_HAS_CONNECTORS_CHANGED)
    return crtc_state->connectors_changed;
#else
    return crtc_state->mode_changed;
#endif
}

static int head_modeset_config_attach_connector(
    struct nv_drm_connector *nv_connector,
    struct NvKmsKapiHeadModeSetConfig *head_modeset_config)
{
    struct nv_drm_encoder *nv_encoder = nv_connector->nv_detected_encoder;

    if (NV_DRM_WARN(nv_encoder == NULL ||
                    head_modeset_config->numDisplays >=
                        ARRAY_SIZE(head_modeset_config->displays))) {
        return -EINVAL;
    }
    head_modeset_config->displays[head_modeset_config->numDisplays++] =
        nv_encoder->hDisplay;
    return 0;
}

/**
 * nv_drm_crtc_atomic_check() can fail after it has modified
 * the 'nv_drm_crtc_state::req_config', that is fine becase 'nv_drm_crtc_state'
 * will be discarded if ->atomic_check() fails.
 */
static int nv_drm_crtc_atomic_check(struct drm_crtc *crtc,
                                    struct drm_crtc_state *crtc_state)
{
    struct nv_drm_crtc_state *nv_crtc_state = to_nv_crtc_state(crtc_state);
    struct NvKmsKapiHeadRequestedConfig *req_config =
        &nv_crtc_state->req_config;
    int ret = 0;

    if (crtc_state->mode_changed) {
        drm_mode_to_nvkms_display_mode(&crtc_state->mode,
                                       &req_config->modeSetConfig.mode);
        req_config->flags.modeChanged = NV_TRUE;
    }

    if (nv_drm_crtc_state_connectors_changed(crtc_state)) {
        struct NvKmsKapiHeadModeSetConfig *config = &req_config->modeSetConfig;
        struct drm_connector *connector;
        struct drm_connector_state *connector_state;
        int j;

        config->numDisplays = 0;

        memset(config->displays, 0, sizeof(config->displays));

        req_config->flags.displaysChanged = NV_TRUE;

        nv_drm_for_each_connector_in_state(crtc_state->state,
                                           connector, connector_state, j) {
            if (connector_state->crtc != crtc) {
                continue;
            }

            if ((ret = head_modeset_config_attach_connector(
                            to_nv_connector(connector),
                            config)) != 0) {
                return ret;
            }
        }
    }

    if (crtc_state->active_changed) {
        req_config->modeSetConfig.bActive = crtc_state->active;
        req_config->flags.activeChanged = NV_TRUE;
    }

    return ret;
}

static bool
nv_drm_crtc_mode_fixup(struct drm_crtc *crtc,
                       const struct drm_display_mode *mode,
                       struct drm_display_mode *adjusted_mode)
{
    return true;
}

static void nv_drm_crtc_prepare(struct drm_crtc *crtc)
{

}

static void nv_drm_crtc_commit(struct drm_crtc *crtc)
{

}

static void nv_drm_crtc_disable(struct drm_crtc *crtc)
{

}

#ifdef NV_DRM_CRTC_HELPER_FUNCS_HAS_ATOMIC_ENABLE
static void nv_drm_crtc_atomic_enable(struct drm_crtc *crtc,
                                      struct drm_crtc_state *old_crtc_state)
{

}
#else
static void nv_drm_crtc_enable(struct drm_crtc *crtc)
{

}
#endif

static const struct drm_crtc_helper_funcs nv_crtc_helper_funcs = {
    .atomic_check = nv_drm_crtc_atomic_check,
    .prepare    = nv_drm_crtc_prepare,
    .commit     = nv_drm_crtc_commit,
#ifdef NV_DRM_CRTC_HELPER_FUNCS_HAS_ATOMIC_ENABLE
    .atomic_enable = nv_drm_crtc_atomic_enable,
#else
    .enable     = nv_drm_crtc_enable,
#endif
    .disable    = nv_drm_crtc_disable,
    .mode_fixup = nv_drm_crtc_mode_fixup,
};

static struct drm_plane*
nv_drm_plane_create(struct drm_device *dev,
                    enum drm_plane_type plane_type,
                    const u32 formats[], unsigned int formats_count)
{
    struct drm_plane *plane = NULL;
    int ret = -ENOMEM;

    if ((plane = nv_drm_calloc(1, sizeof(*plane))) == NULL) {
        goto failed;
    }

    plane->state = nv_drm_calloc(1, sizeof(*plane->state));
    if (plane->state == NULL) {
        goto failed_state_alloc;
    }

    plane->state->plane = plane;

    /*
     * Possible_crtcs is zero here because drm_crtc_init_with_planes() will
     * assign the plane's possible_crtcs after the crtc is successfully
     * initialized.
     */
    ret = drm_universal_plane_init(
        dev,
        plane, 0 /* possible_crtcs */, &nv_plane_funcs,
        formats, formats_count,
#if defined(NV_DRM_UNIVERSAL_PLANE_INIT_HAS_FORMAT_MODIFIERS_ARG)
        NULL,
#endif
        plane_type
#if defined(NV_DRM_UNIVERSAL_PLANE_INIT_HAS_NAME_ARG)
        , NULL
#endif
        );

    if (ret != 0) {
        goto failed_plane_init;
    }

    drm_plane_helper_add(plane, &nv_plane_helper_funcs);

    return plane;

failed_plane_init:
    nv_drm_free(plane->state);

failed_state_alloc:
    nv_drm_free(plane);

failed:
    return ERR_PTR(ret);
}

/*
 * Add drm crtc for given head and supported enum NvKmsSurfaceMemoryFormats.
 */
static struct drm_crtc *__nv_drm_crtc_create(struct nv_drm_device *nv_dev,
                                             struct drm_plane *primary_plane,
                                             struct drm_plane *cursor_plane,
                                             unsigned int head)
{
    struct nv_drm_crtc *nv_crtc = NULL;
    int ret = -ENOMEM;

    if ((nv_crtc = nv_drm_calloc(1, sizeof(*nv_crtc))) == NULL) {
        goto failed;
    }

    nv_crtc->base.state = nv_drm_calloc(1, sizeof(*nv_crtc->base.state));
    if (nv_crtc->base.state == NULL) {
        goto failed_state_alloc;
    }
    nv_crtc->base.state->crtc = &nv_crtc->base;

    nv_crtc->head = head;
#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    INIT_LIST_HEAD(&nv_crtc->flip_list);
    spin_lock_init(&nv_crtc->flip_lock);
#endif

    ret = drm_crtc_init_with_planes(nv_dev->dev,
                                    &nv_crtc->base,
                                    primary_plane, cursor_plane,
                                    &nv_crtc_funcs
#if defined(NV_DRM_CRTC_INIT_WITH_PLANES_HAS_NAME_ARG)
                                    , NULL
#endif
                                    );

    if (ret != 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to init crtc for head %u with planes", head);
        goto failed_init_crtc;
    }

    /* Add crtc to drm sub-system */

    drm_crtc_helper_add(&nv_crtc->base, &nv_crtc_helper_funcs);

    return &nv_crtc->base;

failed_init_crtc:
    nv_drm_free(nv_crtc->base.state);

failed_state_alloc:
    nv_drm_free(nv_crtc);

failed:
    return ERR_PTR(ret);
}

void nv_drm_enumerate_crtcs_and_planes(struct nv_drm_device *nv_dev,
                                       unsigned int num_heads)
{
    unsigned int i;

    for (i = 0; i < num_heads; i++) {
        struct drm_plane *primary_plane = NULL, *cursor_plane = NULL;

        primary_plane = nv_drm_plane_create(
            nv_dev->dev,
            DRM_PLANE_TYPE_PRIMARY,
            nv_default_supported_plane_drm_formats,
            ARRAY_SIZE(nv_default_supported_plane_drm_formats));

        if (IS_ERR(primary_plane)) {
            NV_DRM_DEV_LOG_ERR(
                nv_dev,
                "Failed to create primary plane for head %u, error = %ld",
                i, PTR_ERR(primary_plane));
        }

        cursor_plane = nv_drm_plane_create(
            nv_dev->dev,
            DRM_PLANE_TYPE_CURSOR,
            nv_supported_cursor_plane_drm_formats,
            ARRAY_SIZE(nv_supported_cursor_plane_drm_formats));

        if (IS_ERR(cursor_plane)) {
            NV_DRM_DEV_LOG_ERR(
                nv_dev,
                "Failed to create cursor plane for head %u, error = %ld",
                i, PTR_ERR(cursor_plane));
        }

        if (primary_plane != NULL) {
            struct drm_crtc *crtc =
                __nv_drm_crtc_create(nv_dev,
                                     primary_plane, cursor_plane,
                                     i);

            if (IS_ERR(crtc)) {
                NV_DRM_DEV_LOG_ERR(
                    nv_dev,
                    "Failed to add DRM CRTC for head %u, error = %ld",
                    i, PTR_ERR(crtc));
            }
        }
    }
}

int nv_drm_get_crtc_crc32_ioctl(struct drm_device *dev,
                                void *data, struct drm_file *filep)
{
    struct drm_nvidia_get_crtc_crc32_params *params = data;
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_crtc *crtc = NULL;
    struct nv_drm_crtc *nv_crtc = NULL;
    NvU32 crc32 = 0;
    int ret = 0;

    if (!drm_core_check_feature(dev, DRIVER_MODESET)) {
        ret = -ENOENT;
        goto done;
    }

    crtc = nv_drm_crtc_find(dev, params->crtc_id);
    if (!crtc) {
        ret = -ENOENT;
        goto done;
    }

    nv_crtc = to_nv_crtc(crtc);

    if (!nvKms->getCRC32(nv_dev->pDevice, nv_crtc->head, &crc32)) {
        ret = -ENODEV;
    }

    params->crc32 = crc32;

done:
    return ret;
}

#endif
