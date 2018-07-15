/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(NVKMS_API_TYPES_H)
#define NVKMS_API_TYPES_H

#include "nvtypes.h"

#define NVKMS_MAX_SUBDEVICES                  8

#define NVKMS_LEFT                            0
#define NVKMS_RIGHT                           1
#define NVKMS_MAX_EYES                        2

#define NVKMS_MAIN_PLANE                      0
#define NVKMS_OVERLAY_PLANE                   1
#define NVKMS_MAX_PLANES_PER_HEAD             2
#define NVKMS_ALL_PLANES_MASK                (BIT(NVKMS_MAIN_PLANE) | \
                                              BIT(NVKMS_OVERLAY_PLANE))

#define NVKMS_DP_ADDRESS_STRING_LENGTH        64

typedef NvU32 NvKmsDeviceHandle;
typedef NvU32 NvKmsDispHandle;
typedef NvU32 NvKmsConnectorHandle;
typedef NvU32 NvKmsSurfaceHandle;
typedef NvU32 NvKmsFrameLockHandle;
typedef NvU32 NvKmsDeferredRequestFifoHandle;
typedef NvU32 NvKmsSwapGroupHandle;

struct NvKmsSize {
    NvU16 width;
    NvU16 height;
};

struct NvKmsPoint {
    NvU16 x;
    NvU16 y;
};

struct NvKmsSignedPoint {
    NvS16 x;
    NvS16 y;
};

struct NvKmsRect {
    NvU16 x;
    NvU16 y;
    NvU16 width;
    NvU16 height;
};

/*
 * A 3x3 row-major matrix.
 *
 * The elements are 32-bit single-precision IEEE floating point values.  The
 * floating point bit pattern should be stored in NvU32s to be passed into the
 * kernel.
 */
struct NvKmsMatrix {
    NvU32 m[3][3];
};

typedef enum {
    NVKMS_CONNECTOR_TYPE_DP      = 0,
    NVKMS_CONNECTOR_TYPE_VGA     = 1,
    NVKMS_CONNECTOR_TYPE_DVI_I   = 2,
    NVKMS_CONNECTOR_TYPE_DVI_D   = 3,
    NVKMS_CONNECTOR_TYPE_ADC     = 4,
    NVKMS_CONNECTOR_TYPE_LVDS    = 5,
    NVKMS_CONNECTOR_TYPE_HDMI    = 6,
    NVKMS_CONNECTOR_TYPE_UNKNOWN = 7,
    NVKMS_CONNECTOR_TYPE_MAX     = NVKMS_CONNECTOR_TYPE_UNKNOWN,
} NvKmsConnectorType;

static inline
const char *NvKmsConnectorTypeString(const NvKmsConnectorType connectorType)
{
    switch (connectorType) {
        case NVKMS_CONNECTOR_TYPE_DP:    return "DP";
        case NVKMS_CONNECTOR_TYPE_VGA:   return "VGA";
        case NVKMS_CONNECTOR_TYPE_DVI_I: return "DVI-I";
        case NVKMS_CONNECTOR_TYPE_DVI_D: return "DVI-D";
        case NVKMS_CONNECTOR_TYPE_ADC:   return "ADC";
        case NVKMS_CONNECTOR_TYPE_LVDS:  return "LVDS";
        case NVKMS_CONNECTOR_TYPE_HDMI:  return "HDMI";
        default: break;
    }
    return "Unknown";
}

typedef enum {
    NVKMS_CONNECTOR_SIGNAL_FORMAT_VGA     = 0,
    NVKMS_CONNECTOR_SIGNAL_FORMAT_LVDS    = 1,
    NVKMS_CONNECTOR_SIGNAL_FORMAT_TMDS    = 2,
    NVKMS_CONNECTOR_SIGNAL_FORMAT_DP      = 3,
    NVKMS_CONNECTOR_SIGNAL_FORMAT_UNKNOWN = 4,
    NVKMS_CONNECTOR_SIGNAL_FORMAT_MAX     =
      NVKMS_CONNECTOR_SIGNAL_FORMAT_UNKNOWN,
} NvKmsConnectorSignalFormat;

enum NvKmsSurfaceMemoryFormat {
    NvKmsSurfaceMemoryFormatI8          = 0,
    NvKmsSurfaceMemoryFormatA1R5G5B5    = 1,
    NvKmsSurfaceMemoryFormatX1R5G5B5    = 2,
    NvKmsSurfaceMemoryFormatR5G6B5      = 3,
    NvKmsSurfaceMemoryFormatA8R8G8B8    = 4,
    NvKmsSurfaceMemoryFormatX8R8G8B8    = 5,
    NvKmsSurfaceMemoryFormatA2B10G10R10 = 6,
    NvKmsSurfaceMemoryFormatX2B10G10R10 = 7,
    NvKmsSurfaceMemoryFormatA8B8G8R8    = 8,
    NvKmsSurfaceMemoryFormatX8B8G8R8    = 9,
};

static inline NvU8 nvKmsSurfaceMemoryFormatToBytesPerPixel(
    enum NvKmsSurfaceMemoryFormat format)
{
    switch (format)
    {
        case NvKmsSurfaceMemoryFormatI8:
            return 1;
        case NvKmsSurfaceMemoryFormatA1R5G5B5:
        case NvKmsSurfaceMemoryFormatX1R5G5B5:
        case NvKmsSurfaceMemoryFormatR5G6B5:
            return 2;
        case NvKmsSurfaceMemoryFormatA8R8G8B8:
        case NvKmsSurfaceMemoryFormatX8R8G8B8:
        case NvKmsSurfaceMemoryFormatA8B8G8R8:
        case NvKmsSurfaceMemoryFormatX8B8G8R8:
        case NvKmsSurfaceMemoryFormatA2B10G10R10:
        case NvKmsSurfaceMemoryFormatX2B10G10R10:
            return 4;
    }

    return 0;
}

static inline NvU8 nvKmsSurfaceMemoryFormatToBitsPerPixel(
    enum NvKmsSurfaceMemoryFormat format)
{
    return nvKmsSurfaceMemoryFormatToBytesPerPixel(format) * 8;
}

static inline NvU8 nvKmsSurfaceMemoryFormatToDepth
(
    enum NvKmsSurfaceMemoryFormat format
)
{
    switch (format)
    {
        case NvKmsSurfaceMemoryFormatI8:
            return 8;
        case NvKmsSurfaceMemoryFormatX1R5G5B5:
            return 15;
        case NvKmsSurfaceMemoryFormatA1R5G5B5:
        case NvKmsSurfaceMemoryFormatR5G6B5:
            return 16;
        case NvKmsSurfaceMemoryFormatX8R8G8B8:
        case NvKmsSurfaceMemoryFormatX8B8G8R8:
            return 24;
        case NvKmsSurfaceMemoryFormatX2B10G10R10:
            return 30;
        case NvKmsSurfaceMemoryFormatA8R8G8B8:
        case NvKmsSurfaceMemoryFormatA8B8G8R8:
        case NvKmsSurfaceMemoryFormatA2B10G10R10:
            return 32;
    }

    return 0;
}

static inline const char *NvKmsSurfaceMemoryFormatToString
(
    const enum NvKmsSurfaceMemoryFormat format
)
{
    switch (format) {
        default:
            /* Default return "Unknown" */
            break;

        case NvKmsSurfaceMemoryFormatI8:
            return "I8";
        case NvKmsSurfaceMemoryFormatA1R5G5B5:
            return "A1R5G5B5";
        case NvKmsSurfaceMemoryFormatX1R5G5B5:
            return "X1R5G5B5";
        case NvKmsSurfaceMemoryFormatR5G6B5:
            return "R5G6B5";
        case NvKmsSurfaceMemoryFormatA8R8G8B8:
            return "A8R8G8B8";
        case NvKmsSurfaceMemoryFormatX8R8G8B8:
            return "X8R8G8B8";
        case NvKmsSurfaceMemoryFormatA2B10G10R10:
            return "A2B10G10R10";
        case NvKmsSurfaceMemoryFormatX2B10G10R10:
            return "X2B10G10R10";
        case NvKmsSurfaceMemoryFormatA8B8G8R8:
            return "A8B8G8R8";
        case NvKmsSurfaceMemoryFormatX8B8G8R8:
            return "X8B8G8R8";
    }

    return "Unknown";
}

/*!
 * Description of Notifiers and Semaphores (Non-isochronous (NISO) surfaces).
 *
 * When flipping, the client can optionally specify a notifier and/or
 * a semaphore to use with the flip.  The surfaces used for these
 * should be registered with NVKMS to get an NvKmsSurfaceHandle.
 *
 * NvKmsNIsoSurface::offsetInWords indicates the starting location, in
 * 32-bit words, within the surface where EVO should write the
 * notifier or semaphore.  Note that only the first 4096 bytes of a
 * surface can be used by semaphores or notifiers; offsetInWords must
 * allow for the semaphore or notifier to be written within the first
 * 4096 bytes of the surface.  I.e., this must be satisfied:
 *
 *   ((offsetInWords * 4) + elementSizeInBytes) <= 4096
 *
 * Where elementSizeInBytes is:
 *
 *  if NISO_FORMAT_FOUR_WORD*, elementSizeInBytes = 16
 *  if NISO_FORMAT_LEGACY,
 *    if overlay && notifier, elementSizeInBytes = 16
 *    else, elementSizeInBytes = 4
 *
 *  Note that different GPUs support different semaphore and notifier formats.
 *  Check NvKmsAllocDeviceReply::validNIsoFormatMask to determine which are
 *  valid for the given device.
 *
 *  Note also that FOUR_WORD and FOUR_WORD_NVDISPLAY are the same size, but
 *  FOUR_WORD uses a format compatible with display class 907[ce], and
 *  FOUR_WORD_NVDISPLAY uses a format compatible with c37e (actually defined by
 *  the NV_DISP_NOTIFIER definition in clc37d.h).
 */
enum NvKmsNIsoFormat {
    NVKMS_NISO_FORMAT_LEGACY,
    NVKMS_NISO_FORMAT_FOUR_WORD,
    NVKMS_NISO_FORMAT_FOUR_WORD_NVDISPLAY,
};

enum NvKmsEventType {
    NVKMS_EVENT_TYPE_DPY_CHANGED,
    NVKMS_EVENT_TYPE_DYNAMIC_DPY_CONNECTED,
    NVKMS_EVENT_TYPE_DYNAMIC_DPY_DISCONNECTED,
    NVKMS_EVENT_TYPE_DPY_ATTRIBUTE_CHANGED,
    NVKMS_EVENT_TYPE_FRAMELOCK_ATTRIBUTE_CHANGED,
    NVKMS_EVENT_TYPE_DISP_ATTRIBUTE_CHANGED,
    NVKMS_EVENT_TYPE_FLIP_OCCURRED,
};

/*
 * Base super sample (FOS) capabilities.  Keep in this order (we rely on this
 * when falling back)
 */
enum NvKmsSuperSample {
    NVKMS_SUPER_SAMPLE_X1,
    NVKMS_SUPER_SAMPLE_X4,
    NVKMS_SUPER_SAMPLE_X8
};

struct NvKmsUsageBounds {
    struct {
        NvBool usable;
        NvBool distRenderUsable;
        NvU8 depth;
        enum NvKmsSuperSample superSample;
    } base;

    struct {
        NvBool usable;
        NvU8 depth;
    } overlay;
};

/*
 * A 3x4 row-major colorspace conversion matrix.
 *
 * The output color C' is the CSC matrix M times the column vector
 * [ R, G, B, 1 ].
 *
 * Each entry in the matrix is a signed 2's-complement fixed-point number with
 * 3 integer bits and 16 fractional bits.
 */
struct NvKmsCscMatrix {
    NvSFXP16_16 m[3][4];
};

#define NVKMS_IDENTITY_CSC_MATRIX   \
    (struct NvKmsCscMatrix){{       \
        { 0x10000, 0, 0, 0 },       \
        { 0, 0x10000, 0, 0 },       \
        { 0, 0, 0x10000, 0 }        \
    }}

#endif /* NVKMS_API_TYPES_H */
