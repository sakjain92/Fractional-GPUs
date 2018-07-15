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

#ifndef __NVIDIA_DRM_CRTC_H__
#define __NVIDIA_DRM_CRTC_H__

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_ATOMIC_MODESET_AVAILABLE)

#include <drm/drmP.h>
#include "nvtypes.h"
#include "nvkms-kapi.h"

struct nv_drm_crtc {
    NvU32 head;

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    /**
     * @flip_list:
     *
     * List of flips pending to get proccessed by __nv_drm_handle_flip_event().
     * Protected by @flip_lock.
     */
    struct list_head flip_list;

    /**
     * @flip_lock:
     *
     * Spinlock to protect @flip_list.
     */
    spinlock_t flip_lock;
#else
    atomic_t has_pending_commit;
    atomic_t has_pending_flip_event;
#endif

    struct drm_crtc base;
};

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
/**
 * struct nv_drm_flip - flip state
 *
 * This state is getting used to consume DRM completion event associated
 * with each crtc state from atomic commit.
 *
 * Function nv_drm_atomic_helper_commit_tail() consumes DRM completion
 * event, save it into flip state associated with crtc and queue flip state into
 * crtc's flip list and commits atomic update to hardware.
 */
struct nv_drm_flip {
    /**
     * @event:
     *
     * Optional pointer to a DRM event to signal upon completion of
     * the state update.
     */
    struct drm_pending_vblank_event *event;

    /**
     * @list_entry:
     *
     * Entry on the per-CRTC &nv_drm_crtc.flip_list. Protected by
     * &nv_drm_crtc.flip_lock.
     */
    struct list_head list_entry;
};
#endif

struct nv_drm_crtc_state {
    /**
     * @base:
     *
     * Base DRM crtc state object for this.
     */
    struct drm_crtc_state base;

    /**
     * @head_req_config:
     *
     * Requested head's modeset configuration corresponding to this crtc state.
     */
    struct NvKmsKapiHeadRequestedConfig req_config;

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    /**
     * @nv_flip:
     *
     * Flip state associated with this crtc state, gets allocated
     * by nv_drm_atomic_crtc_duplicate_state(), on successful commit it gets
     * consumed and queued into flip list by nv_drm_atomic_helper_commit_tail()
     * and finally gets destroyed by __nv_drm_handle_flip_event() after getting
     * processed.
     *
     * In case of failure of atomic commit, this flip state getting destroyed by
     * nv_drm_atomic_crtc_destroy_state().
     */
    struct nv_drm_flip *nv_flip;
#endif
};

static inline struct nv_drm_crtc_state *to_nv_crtc_state(struct drm_crtc_state *state)
{
    return container_of(state, struct nv_drm_crtc_state, base);
}

static inline struct nv_drm_crtc *to_nv_crtc(struct drm_crtc *crtc)
{
    if (crtc == NULL) {
        return NULL;
    }
    return container_of(crtc, struct nv_drm_crtc, base);
}

/*
 * CRTCs are static objects, list does not change once after initialization and
 * before teardown of device. Initialization/teardown paths are single
 * threaded, so no locking required.
 */
static inline
struct nv_drm_crtc *nv_drm_crtc_lookup(struct nv_drm_device *nv_dev, NvU32 head)
{
    struct drm_crtc *crtc;
    list_for_each_entry(crtc, &nv_dev->dev->mode_config.crtc_list, head) {
        struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

        if (nv_crtc->head == head)  {
            return nv_crtc;
        }
    }
    return NULL;
}

void nv_drm_enumerate_crtcs_and_planes(struct nv_drm_device *nv_dev,
                                       unsigned int num_heads);

int nv_drm_get_crtc_crc32_ioctl(struct drm_device *dev,
                                void *data, struct drm_file *filep);

#endif /* NV_DRM_ATOMIC_MODESET_AVAILABLE */

#endif /* __NVIDIA_DRM_CRTC_H__ */
