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
#include "nvidia-drm-modeset.h"
#include "nvidia-drm-crtc.h"
#include "nvidia-drm-os-interface.h"
#include "nvidia-drm-helper.h"

#include <drm/drm_atomic.h>
#include <drm/drm_atomic_helper.h>
#include <drm/drm_crtc.h>

struct nv_drm_atomic_state {
    struct NvKmsKapiRequestedModeSetConfig config;
    struct drm_atomic_state base;
};

static inline struct nv_drm_atomic_state *to_nv_atomic_state(
    struct drm_atomic_state *state)
{
    return container_of(state, struct nv_drm_atomic_state, base);
}

struct drm_atomic_state *nv_drm_atomic_state_alloc(struct drm_device *dev)
{
    struct nv_drm_atomic_state *nv_state =
            nv_drm_calloc(1, sizeof(*nv_state));

    if (nv_state == NULL || drm_atomic_state_init(dev, &nv_state->base) < 0) {
        nv_drm_free(nv_state);
        return NULL;
    }

    return &nv_state->base;
}

void nv_drm_atomic_state_clear(struct drm_atomic_state *state)
{
    drm_atomic_state_default_clear(state);
}

void nv_drm_atomic_state_free(struct drm_atomic_state *state)
{
    struct nv_drm_atomic_state *nv_state =
                    to_nv_atomic_state(state);
    drm_atomic_state_default_release(state);
    nv_drm_free(nv_state);
}

/**
 * nv_drm_atomic_commit - validate/commit modeset config
 * @dev: DRM device
 * @state: atomic state tracking atomic update
 * @commit: commit/check modeset config associated with atomic update
 *
 * @state tracks atomic update and modeset objects affected
 * by the atomic update, but the state of the modeset objects it contains
 * depends on the current stage of the update.
 * At the commit stage, the proposed state is already stored in the current
 * state, and @state contains old state for all affected modeset objects.
 * At the check/validation stage, @state contains the proposed state for
 * all affected objects.
 *
 * Sequence of atomic update -
 *   1. The check/validation of proposed atomic state,
 *   2. Do any other steps that might fail,
 *   3. Put the proposed state into the current state pointers,
 *   4. Actually commit the hardware state,
 *   5. Cleanup old state.
 *
 * The function nv_drm_atomic_apply_modeset_config() is getting called
 * at stages (1) and (4).
 */
static int
nv_drm_atomic_apply_modeset_config(struct drm_device *dev,
                                   struct drm_atomic_state *state,
                                   bool commit)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct NvKmsKapiRequestedModeSetConfig *requested_config =
        &(to_nv_atomic_state(state)->config);
    struct drm_crtc *crtc;
    struct drm_crtc_state *crtc_state;
    int i;

    memset(requested_config, 0, sizeof(*requested_config));

    /* Loop over affected crtcs and construct NvKmsKapiRequestedModeSetConfig */
    nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
        /*
         * When commiting a state, the new state is already stored in
         * crtc->state. When checking a proposed state, the proposed state is
         * stored in crtc_state.
         */
        struct drm_crtc_state *new_crtc_state =
                               commit ? crtc->state : crtc_state;
        struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

        requested_config->headRequestedConfig[nv_crtc->head] =
            to_nv_crtc_state(new_crtc_state)->req_config;

        requested_config->headsMask |= 1 << nv_crtc->head;
    }

    if (!nvKms->applyModeSetConfig(nv_dev->pDevice,
                                   requested_config, commit)) {
        return -EINVAL;
    }

    return 0;
}

int nv_drm_atomic_check(struct drm_device *dev,
                        struct drm_atomic_state *state)
{
    int ret = 0;

    if ((ret = drm_atomic_helper_check(dev, state)) != 0) {
        goto done;
    }

    ret = nv_drm_atomic_apply_modeset_config(dev,
                                             state, false /* commit */);

done:
    return ret;
}

#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
/**
 * nv_drm_atomic_helper_commit_tail - hook to commit atomic state to hardware
 * @state: Old DRM atomic state
 *
 * This hook is getting invoked from commit_work() which gets called or
 * scheduled from drm_atomic_helper_commit(). This function has been implemented
 * to flush write combined dumb buffers, queue flips and commit atomic state
 * to hardware using NvKmsKapi.
 */
void nv_drm_atomic_helper_commit_tail(struct drm_atomic_state *state)
{
    struct drm_device *dev = state->dev;
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    int i;
    struct drm_crtc *crtc;
    struct drm_crtc_state *crtc_state;
    int ret;

    if (nvKms->systemInfo.bAllowWriteCombining) {
        /*
         * XXX This call is required only if dumb buffer is going
         * to be presented.
         */
         nv_drm_write_combine_flush();
    }

    nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
        struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);
        struct nv_drm_crtc_state *nv_crtc_state = to_nv_crtc_state(crtc->state);
        struct nv_drm_flip *nv_flip = nv_crtc_state->nv_flip;

        nv_crtc_state->nv_flip  = NULL;

        if (!crtc->state->active && !crtc_state->active) {
            nv_drm_free(nv_flip);
            continue;
        }

        nv_flip->event = crtc->state->event;
        crtc->state->event = NULL;

        spin_lock(&nv_crtc->flip_lock);
        list_add(&nv_flip->list_entry, &nv_crtc->flip_list);
        spin_unlock(&nv_crtc->flip_lock);
    }

    if ((ret = nv_drm_atomic_apply_modeset_config(
                    dev,
                    state, true /* commit */)) != 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to apply atomic modeset.  Error code: %d",
            ret);
    }

    drm_atomic_helper_commit_hw_done(state);
}

/**
 * __nv_drm_handle_flip_event - handle flip occurred event
 * @nv_crtc: crtc on which flip has been occurred
 *
 * This handler dequeue a first nv_drm_flip from the crtc's flips list, process
 * it to single DRM about completion of state update and free nv_drm_flip.
 */
static void __nv_drm_handle_flip_event(struct nv_drm_crtc *nv_crtc)
{
    struct drm_crtc *crtc = &nv_crtc->base;
    struct drm_device *dev = crtc->dev;

    struct nv_drm_flip *nv_flip = NULL;

    spin_lock(&nv_crtc->flip_lock);
    if ((nv_flip =
            list_first_entry_or_null(&nv_crtc->flip_list,
                                     struct nv_drm_flip, list_entry)) == NULL) {
        spin_unlock(&nv_crtc->flip_lock);
        WARN_ON(1);
        return;
    }
    list_del(&nv_flip->list_entry);
    spin_unlock(&nv_crtc->flip_lock);

    spin_lock(&dev->event_lock);
    if (nv_flip->event != NULL) {
        drm_crtc_send_vblank_event(crtc, nv_flip->event);
    }
    spin_unlock(&dev->event_lock);

    nv_drm_free(nv_flip);
}

#else

struct nv_drm_atomic_commit_task {
    struct drm_device *dev;
    struct drm_atomic_state *state;

    struct work_struct work;
};

static void nv_drm_atomic_commit_task_callback(struct work_struct *work)
{
    struct nv_drm_atomic_commit_task *nv_commit_task =
        container_of(work, struct nv_drm_atomic_commit_task, work);
    struct drm_device *dev = nv_commit_task->dev;
    struct drm_atomic_state *state = nv_commit_task->state;
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    int i;
    struct drm_crtc *crtc;
    struct drm_crtc_state *crtc_state;
    int ret;

    if (nvKms->systemInfo.bAllowWriteCombining) {
        /*
         * XXX This call is required only if dumb buffer is going
         * to be presented.
         */
         nv_drm_write_combine_flush();
    }

    if ((ret = nv_drm_atomic_apply_modeset_config(
                    dev,
                    state, true /* commit */)) != 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to apply atomic modeset.  Error code: %d",
            ret);
    }

    nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
        struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

        if (wait_event_timeout(
                nv_dev->pending_flip_queue,
                !atomic_read(&nv_crtc->has_pending_flip_event),
                3 * HZ /* 3 second */) == 0) {
            NV_DRM_DEV_LOG_ERR(
                nv_dev,
                "Flip event timeout on head %u", nv_crtc->head);
        }

        atomic_set(&nv_crtc->has_pending_commit, false);
        wake_up_all(&nv_dev->pending_commit_queue);
    }

#if defined(NV_DRM_ATOMIC_STATE_FREE)
    drm_atomic_state_free(state);
#else
    drm_atomic_state_put(state);
#endif

    nv_drm_free(nv_commit_task);
}

static int nv_drm_atomic_commit_internal(
    struct drm_device *dev,
    struct drm_atomic_state *state,
    bool nonblock)
{
    int ret = 0;

    int i;
    struct drm_crtc *crtc = NULL;
    struct drm_crtc_state *crtc_state = NULL;

    struct nv_drm_atomic_commit_task *nv_commit_task = NULL;

    struct NvKmsKapiRequestedModeSetConfig *requested_config = NULL;

    nv_commit_task = nv_drm_calloc(1, sizeof(*nv_commit_task));

    if (nv_commit_task == NULL) {
        ret = -ENOMEM;
        goto failed;
    }

    /*
     * Not required to convert convert drm_atomic_state to
     * NvKmsKapiRequestedModeSetConfig because it has been already
     * happened in nv_drm_atomic_check().
     *
     * Core DRM guarantees to call into nv_drm_atomic_check() before
     * calling into nv_drm_atomic_commit().
     */
    requested_config = &(to_nv_atomic_state(state)->config);

    /*
     * drm_mode_config_funcs::atomic_commit() mandates to return -EBUSY
     * for nonblocking commit if previous updates (commit tasks/flip event) are
     * pending. In case of blocking commits it mandates to wait for previous
     * updates to complete.
     */
    if (!nonblock) {
        /*
         * Serialize commits and flip events on crtc, in order to avoid race
         * condition between two/more nvKms->applyModeSetConfig() on single
         * crtc and generate flip events in correct order.
         */
        nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
            struct nv_drm_device *nv_dev = to_nv_device(dev);
            struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

            if (wait_event_timeout(
                    nv_dev->pending_flip_queue,
                    !atomic_read(&nv_crtc->has_pending_flip_event),
                    3 * HZ /* 3 second */) == 0) {
                ret = -EBUSY;
                goto failed;
            }

            if (wait_event_timeout(
                    nv_dev->pending_commit_queue,
                    !atomic_read(&nv_crtc->has_pending_commit),
                    3 * HZ /* 3 second */) == 0) {
                ret = -EBUSY;
                goto failed;
            }
        }
    } else {
        nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
            struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

            if (atomic_read(&nv_crtc->has_pending_commit) ||
                atomic_read(&nv_crtc->has_pending_flip_event)) {
                ret = -EBUSY;
                goto failed;
            }
        }
    }

    /*
     * Mark all affected crtcs which will have pending commits and/or
     * flip events.
     */

    nv_drm_for_each_crtc_in_state(state, crtc, crtc_state, i) {
        struct nv_drm_crtc *nv_crtc = to_nv_crtc(crtc);

        atomic_set(&nv_crtc->has_pending_commit, true);

        if (!crtc->state->active && !crtc_state->active) {
            continue;
        }

        atomic_set(&nv_crtc->has_pending_flip_event, true);
    }

    drm_atomic_helper_swap_state(dev, state);

    INIT_WORK(&nv_commit_task->work,
              nv_drm_atomic_commit_task_callback);

    nv_commit_task->dev = dev;
    nv_commit_task->state = state;

    if (nonblock) {
        schedule_work(&nv_commit_task->work);
    } else {
        nv_drm_atomic_commit_task_callback(&nv_commit_task->work);
    }

    return 0;

failed:

    nv_drm_free(nv_commit_task);

    return ret;
}

static void __nv_drm_handle_flip_event(struct nv_drm_crtc *nv_crtc)
{
    struct drm_crtc *crtc = &nv_crtc->base;
    struct drm_crtc_state *crtc_state = crtc->state;

    struct drm_device *dev = crtc->dev;
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    spin_lock(&dev->event_lock);
    if (crtc_state->event != NULL) {
        drm_crtc_send_vblank_event(crtc, crtc_state->event);
    }
    crtc_state->event = NULL;
    spin_unlock(&dev->event_lock);

    WARN_ON(!atomic_read(&nv_crtc->has_pending_flip_event));
    atomic_set(&nv_crtc->has_pending_flip_event, false);
    wake_up_all(&nv_dev->pending_flip_queue);
}

#endif /* NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE */

int nv_drm_atomic_commit(struct drm_device *dev,
                             struct drm_atomic_state *state, bool nonblock)
{
#if defined(NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE)
    return drm_atomic_helper_commit(dev, state, nonblock);
#else
    return nv_drm_atomic_commit_internal(dev, state, nonblock);
#endif
}

void nv_drm_handle_flip_occurred(struct nv_drm_device *nv_dev,
                                 NvU32 head, NvKmsKapiPlaneType plane)
{
    struct nv_drm_crtc *nv_crtc = nv_drm_crtc_lookup(nv_dev, head);

    if (NV_DRM_WARN(nv_crtc == NULL)) {
        return;
    }

    switch (plane) {
        case NVKMS_KAPI_PLANE_PRIMARY:
            __nv_drm_handle_flip_event(nv_crtc);
            break;
        case NVKMS_KAPI_PLANE_OVERLAY:
            /* TODO */
        case NVKMS_KAPI_PLANE_CURSOR:
        default:
            BUG_ON(1);
    }
}

#endif
