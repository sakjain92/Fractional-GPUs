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
#include "nvidia-drm-utils.h"

struct NvKmsKapiConnectorInfo*
nvkms_get_connector_info(struct NvKmsKapiDevice *pDevice,
                         NvKmsKapiConnector hConnector)
{
    struct NvKmsKapiConnectorInfo *connectorInfo =
                           nv_drm_calloc(1, sizeof(*connectorInfo));

    if (connectorInfo == NULL) {
        return ERR_PTR(-ENOMEM);
    }

    if (!nvKms->getConnectorInfo(pDevice, hConnector, connectorInfo)) {
        nv_drm_free(connectorInfo);

        return ERR_PTR(-EINVAL);
    }

    return connectorInfo;
}

int
nvkms_connector_signal_to_drm_encoder_signal(NvKmsConnectorSignalFormat format)
{
    switch (format) {
        default:
        case NVKMS_CONNECTOR_SIGNAL_FORMAT_UNKNOWN:
            return DRM_MODE_ENCODER_NONE;
        case NVKMS_CONNECTOR_SIGNAL_FORMAT_TMDS:
        case NVKMS_CONNECTOR_SIGNAL_FORMAT_DP:
            return DRM_MODE_ENCODER_TMDS;
        case NVKMS_CONNECTOR_SIGNAL_FORMAT_LVDS:
            return DRM_MODE_ENCODER_LVDS;
        case NVKMS_CONNECTOR_SIGNAL_FORMAT_VGA:
            return DRM_MODE_ENCODER_DAC;
    }
}

int nvkms_connector_type_to_drm_connector_type(NvKmsConnectorType type,
                                               NvBool internal)
{
    switch (type) {
        default:
        case NVKMS_CONNECTOR_TYPE_UNKNOWN:
            return DRM_MODE_CONNECTOR_Unknown;
        case NVKMS_CONNECTOR_TYPE_DP:
            return
                internal ?
                DRM_MODE_CONNECTOR_eDP : DRM_MODE_CONNECTOR_DisplayPort;
        case NVKMS_CONNECTOR_TYPE_HDMI:
            return DRM_MODE_CONNECTOR_HDMIA;
        case NVKMS_CONNECTOR_TYPE_DVI_D:
            return DRM_MODE_CONNECTOR_DVID;
        case NVKMS_CONNECTOR_TYPE_DVI_I:
            return DRM_MODE_CONNECTOR_DVII;
        case NVKMS_CONNECTOR_TYPE_LVDS:
            return DRM_MODE_CONNECTOR_LVDS;
        case NVKMS_CONNECTOR_TYPE_VGA:
            return DRM_MODE_CONNECTOR_VGA;
    }
}

void
nvkms_display_mode_to_drm_mode(const struct NvKmsKapiDisplayMode *displayMode,
                               struct drm_display_mode *mode)
{
    mode->vrefresh    = (displayMode->timings.refreshRate + 500) / 1000; /* In Hz */

    mode->clock       = (displayMode->timings.pixelClockHz + 500) / 1000; /* In Hz */

    mode->hdisplay    = displayMode->timings.hVisible;
    mode->hsync_start = displayMode->timings.hSyncStart;
    mode->hsync_end   = displayMode->timings.hSyncEnd;
    mode->htotal      = displayMode->timings.hTotal;
    mode->hskew       = displayMode->timings.hSkew;

    mode->vdisplay    = displayMode->timings.vVisible;
    mode->vsync_start = displayMode->timings.vSyncStart;
    mode->vsync_end   = displayMode->timings.vSyncEnd;
    mode->vtotal      = displayMode->timings.vTotal;

    if (displayMode->timings.flags.interlaced) {
        mode->flags |= DRM_MODE_FLAG_INTERLACE;
    }

    if (displayMode->timings.flags.doubleScan) {
        mode->flags |= DRM_MODE_FLAG_DBLSCAN;
    }

    if (displayMode->timings.flags.hSyncPos) {
        mode->flags |= DRM_MODE_FLAG_PHSYNC;
    }

    if (displayMode->timings.flags.hSyncNeg) {
        mode->flags |= DRM_MODE_FLAG_NHSYNC;
    }

    if (displayMode->timings.flags.vSyncPos) {
        mode->flags |= DRM_MODE_FLAG_PVSYNC;
    }

    if (displayMode->timings.flags.vSyncNeg) {
        mode->flags |= DRM_MODE_FLAG_NVSYNC;
    }

    mode->width_mm  = displayMode->timings.widthMM;
    mode->height_mm = displayMode->timings.heightMM;

    if (strlen(displayMode->name) != 0) {
        memcpy(
            mode->name, displayMode->name,
            min(sizeof(mode->name), sizeof(displayMode->name)));

        mode->name[sizeof(mode->name) - 1] = '\0';
    } else {
        drm_mode_set_name(mode);
    }
}

bool drm_format_to_nvkms_format(u32 format,
                                enum NvKmsSurfaceMemoryFormat *nvkms_format)
{
    switch (format) {
        case DRM_FORMAT_ARGB1555:
            *nvkms_format = NvKmsSurfaceMemoryFormatA1R5G5B5;
            return true;
        case DRM_FORMAT_XRGB1555:
            *nvkms_format = NvKmsSurfaceMemoryFormatX1R5G5B5;
            return true;
        case DRM_FORMAT_RGB565:
            *nvkms_format = NvKmsSurfaceMemoryFormatR5G6B5;
            return true;
        case DRM_FORMAT_ARGB8888:
            *nvkms_format = NvKmsSurfaceMemoryFormatA8R8G8B8;
            return true;
        case DRM_FORMAT_XRGB8888:
            *nvkms_format = NvKmsSurfaceMemoryFormatX8R8G8B8;
            return true;
        case DRM_FORMAT_ABGR2101010:
            *nvkms_format = NvKmsSurfaceMemoryFormatA2B10G10R10;
            return true;
        case DRM_FORMAT_XBGR2101010:
            *nvkms_format = NvKmsSurfaceMemoryFormatX2B10G10R10;
            return true;
    }

    return false;
}

void drm_mode_to_nvkms_display_mode(const struct drm_display_mode *src,
                                    struct NvKmsKapiDisplayMode *dst)
{
    dst->timings.refreshRate  = src->vrefresh * 1000;

    dst->timings.pixelClockHz = src->clock * 1000; /* In Hz */

    dst->timings.hVisible   = src->hdisplay;
    dst->timings.hSyncStart = src->hsync_start;
    dst->timings.hSyncEnd   = src->hsync_end;
    dst->timings.hTotal     = src->htotal;
    dst->timings.hSkew      = src->hskew;

    dst->timings.vVisible   = src->vdisplay;
    dst->timings.vSyncStart = src->vsync_start;
    dst->timings.vSyncEnd   = src->vsync_end;
    dst->timings.vTotal     = src->vtotal;

    if (src->flags & DRM_MODE_FLAG_INTERLACE) {
        dst->timings.flags.interlaced = NV_TRUE;
    } else {
        dst->timings.flags.interlaced = NV_FALSE;
    }

    if (src->flags & DRM_MODE_FLAG_DBLSCAN) {
        dst->timings.flags.doubleScan = NV_TRUE;
    } else {
        dst->timings.flags.doubleScan = NV_FALSE;
    }

    if (src->flags & DRM_MODE_FLAG_PHSYNC) {
        dst->timings.flags.hSyncPos = NV_TRUE;
    } else {
        dst->timings.flags.hSyncPos = NV_FALSE;
    }

    if (src->flags & DRM_MODE_FLAG_NHSYNC) {
        dst->timings.flags.hSyncNeg = NV_TRUE;
    } else {
       dst->timings.flags.hSyncNeg = NV_FALSE;
    }

    if (src->flags & DRM_MODE_FLAG_PVSYNC) {
        dst->timings.flags.vSyncPos = NV_TRUE;
    } else {
        dst->timings.flags.vSyncPos = NV_FALSE;
    }

    if (src->flags & DRM_MODE_FLAG_NVSYNC) {
        dst->timings.flags.vSyncNeg = NV_TRUE;
    } else {
        dst->timings.flags.vSyncNeg = NV_FALSE;
    }

    dst->timings.widthMM  = src->width_mm;
    dst->timings.heightMM = src->height_mm;

    memcpy(dst->name, src->name, min(sizeof(dst->name), sizeof(src->name)));
}

bool drm_plane_type_to_nvkms_plane_type(enum drm_plane_type src,
                                        NvKmsKapiPlaneType *type)
{
    switch (src) {
        default:
            return false;
        case DRM_PLANE_TYPE_OVERLAY:
            *type = NVKMS_KAPI_PLANE_OVERLAY;
            break;
        case DRM_PLANE_TYPE_PRIMARY:
            *type = NVKMS_KAPI_PLANE_PRIMARY;
            break;
        case DRM_PLANE_TYPE_CURSOR:
            *type = NVKMS_KAPI_PLANE_CURSOR;
            break;
    }

    return true;
}

#endif
