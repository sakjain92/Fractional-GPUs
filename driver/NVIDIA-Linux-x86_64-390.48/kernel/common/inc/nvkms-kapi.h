/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#if !defined(__NVKMS_KAPI_H__)

#include "nvtypes.h"

#include "nv-gpu-info.h"
#include "nvkms-api-types.h"

#define __NVKMS_KAPI_H__

/*
 * On Linux-x86, the kernel's function calling convention may pass
 * parameters in registers.  Force functions called to and from core
 * NVKMS to pass parameters on the stack.
 */
#if NVCPU_IS_X86
  #define NVKMS_KAPI_CALL __attribute__((regparm(0)))
#else
  #define NVKMS_KAPI_CALL
#endif

#define NVKMS_KAPI_MAX_HEADS           4

#define NVKMS_KAPI_MAX_CONNECTORS     16
#define NVKMS_KAPI_MAX_CLONE_DISPLAYS 16

#define NVKMS_KAPI_EDID_BUFFER_SIZE   2048

#define NVKMS_KAPI_MODE_NAME_LEN      32

/**
 * \defgroup Objects
 * @{
 */

struct NvKmsKapiDevice;
struct NvKmsKapiMemory;
struct NvKmsKapiSurface;
struct NvKmsKapiChannelEvent;

typedef NvU32 NvKmsKapiConnector;
typedef NvU32 NvKmsKapiDisplay;

/** @} */

/**
 * \defgroup FuncPtrs
 * @{
 */

typedef void NVKMS_KAPI_CALL NvKmsChannelEventProc(void *dataPtr, NvU32 dataU32);

/** @} */

/**
 * \defgroup Structs
 * @{
 */

struct NvKmsKapiDisplayModeTimings {

    NvU32 refreshRate;
    NvU32 pixelClockHz;
    NvU32 hVisible;
    NvU32 hSyncStart;
    NvU32 hSyncEnd;
    NvU32 hTotal;
    NvU32 hSkew;
    NvU32 vVisible;
    NvU32 vSyncStart;
    NvU32 vSyncEnd;
    NvU32 vTotal;

    struct {

        NvU32 interlaced : 1;
        NvU32 doubleScan : 1;
        NvU32 hSyncPos   : 1;
        NvU32 hSyncNeg   : 1;
        NvU32 vSyncPos   : 1;
        NvU32 vSyncNeg   : 1;

    } flags;

    NvU32 widthMM;
    NvU32 heightMM;

};

struct NvKmsKapiDisplayMode {
    struct NvKmsKapiDisplayModeTimings timings;
    char name[NVKMS_KAPI_MODE_NAME_LEN];
};

struct NvKmsKapiDeviceResourcesInfo {

    NvU32 numHeads;

    NvU32        numConnectors;
    NvKmsKapiConnector connectorHandles[NVKMS_KAPI_MAX_CONNECTORS];

    struct {

        NvU32 minWidthInPixels;
        NvU32 maxWidthInPixels;

        NvU32 minHeightInPixels;
        NvU32 maxHeightInPixels;

        NvU32 maxCursorSizeInPixels;

        NvU32 pitchAlignment;

    } caps;

};

typedef enum NvKmsKapiPlaneTypeRec {
    NVKMS_KAPI_PLANE_PRIMARY = 0,
    NVKMS_KAPI_PLANE_CURSOR  = 1,
    NVKMS_KAPI_PLANE_OVERLAY = 2,
    NVKMS_KAPI_PLANE_MAX     = 3,
} NvKmsKapiPlaneType;

#define NVKMS_KAPI_PLANE_MASK(planeType) (1 << (planeType))

typedef enum NvKmsKapiMappingTypeRec {
    NVKMS_KAPI_MAPPING_TYPE_USER   = 1,
    NVKMS_KAPI_MAPPING_TYPE_KERNEL = 2,
} NvKmsKapiMappingType;

struct NvKmsKapiConnectorInfo {

    NvKmsKapiConnector handle;

    NvU32 physicalIndex;

    NvU32 headMask;

    NvKmsConnectorSignalFormat signalFormat;
    NvKmsConnectorType         type;

    /*
     * List of connectors, not possible to serve together with this connector
     * becase they are competing for same resources.
     */
    NvU32        numIncompatibleConnectors;
    NvKmsKapiConnector incompatibleConnectorHandles[NVKMS_KAPI_MAX_CONNECTORS];

};

struct NvKmsKapiStaticDisplayInfo {

    NvKmsKapiDisplay handle;

    NvKmsKapiConnector connectorHandle;

    /* Set for DisplayPort MST displays (dynamic displays) */
    char dpAddress[NVKMS_DP_ADDRESS_STRING_LENGTH];

    NvBool internal;

    /* List of potential sibling display for cloning */
    NvU32  numPossibleClones;
    NvKmsKapiDisplay possibleCloneHandles[NVKMS_KAPI_MAX_CLONE_DISPLAYS];

};

struct NvKmsKapiPlaneConfig {
    struct NvKmsKapiSurface *surface;

    NvU16 srcX, srcY;
    NvU16 srcWidth, srcHeight;

    NvU16 dstX, dstY;
    NvU16 dstWidth, dstHeight;
};

struct NvKmsKapiPlaneRequestedConfig {
    struct NvKmsKapiPlaneConfig config;
    struct {
        NvBool surfaceChanged : 1;
        NvBool srcXYChanged   : 1;
        NvBool srcWHChanged   : 1;
        NvBool dstXYChanged   : 1;
        NvBool dstWHChanged   : 1;
    } flags;
};

struct NvKmsKapiHeadModeSetConfig {
    /*
     * DRM distinguishes between the head state "enabled" (the specified
     * configuration for the head is valid, its resources are allocated,
     * etc, but the head may not necessarily be currently driving pixels
     * to its output resource) and the head state "active" (the head is
     * "enabled" _and_ the head is actively driving pixels to its output
     * resource).
     *
     * This distinction is for DPMS:
     *
     *  DPMS On  : enabled=true, active=true
     *  DPMS Off : enabled=true, active=false
     *
     * "Enabled" state is indicated by numDisplays != 0.
     * "Active" state is indicated by bActive == true.
     */
    NvBool bActive;

    NvU32  numDisplays;
    NvKmsKapiDisplay displays[NVKMS_KAPI_MAX_CLONE_DISPLAYS];

    struct NvKmsKapiDisplayMode mode;
};

struct NvKmsKapiHeadRequestedConfig {
    struct NvKmsKapiHeadModeSetConfig modeSetConfig;
    struct {
        NvBool activeChanged   : 1;
        NvBool displaysChanged : 1;
        NvBool modeChanged     : 1;
    } flags;

    struct NvKmsKapiPlaneRequestedConfig
        planeRequestedConfig[NVKMS_KAPI_PLANE_MAX];
};

struct NvKmsKapiRequestedModeSetConfig {
    NvU32 headsMask;
    struct NvKmsKapiHeadRequestedConfig
        headRequestedConfig[NVKMS_KAPI_MAX_HEADS];
};

struct NvKmsKapiEventDisplayChanged {
    NvKmsKapiDisplay display;
};

struct NvKmsKapiEventDynamicDisplayConnected {
    NvKmsKapiDisplay display;
};

struct NvKmsKapiEventFlipOccurred {
    NvU32 head;
    NvKmsKapiPlaneType plane;
};

struct NvKmsKapiEvent {
    enum NvKmsEventType type;

    struct NvKmsKapiDevice  *device;

    void *privateData;

    union {
        struct NvKmsKapiEventDisplayChanged displayChanged;
        struct NvKmsKapiEventDynamicDisplayConnected dynamicDisplayConnected;
        struct NvKmsKapiEventFlipOccurred flipOccurred;
    } u;
};

struct NvKmsKapiAllocateDeviceParams {
    /* [IN] GPU ID obtained from enumerateGpus() */
    NvU32 gpuId;

    /* [IN] Private data of device allocator */
    void *privateData;
    /* [IN] Event callback */
    void (*eventCallback)(const struct NvKmsKapiEvent *event);
};

struct NvKmsKapiDynamicDisplayParams {
    /* [IN] Display Handle returned by getDisplays() */
    NvKmsKapiDisplay handle;

    /* [OUT] Connection status */
    NvU32 connected;

    /* [IN/OUT] EDID of connected monitor/ Input to override EDID */
    struct {
        NvU16  bufferSize;
        NvU8   buffer[NVKMS_KAPI_EDID_BUFFER_SIZE];
    } edid;

    /* [IN] Set true to override EDID */
    NvBool overrideEdid;

    /* [IN] Set true to force connected status */
    NvBool forceConnected;

    /* [IN] Set true to force disconnect status */
    NvBool forceDisconnected;
};

struct NvKmsKapiFunctionsTable {

    /*!
     * NVIDIA Driver version string.
     */
    const char *versionString;

    /*!
     * System Information.
     */
    struct {
        /* Availability of write combining support for video memory */
        NvBool bAllowWriteCombining;
    } systemInfo;

    /*!
     * Enumerate the available physical GPUs that can be used with NVKMS.
     *
     * \param [out]  gpuInfo  The information of the enumerated GPUs.
     *                        It is an array of NVIDIA_MAX_GPUS elements.
     *
     * \return  Count of enumerated gpus.
     */
    NvU32 NVKMS_KAPI_CALL (*enumerateGpus)(nv_gpu_info_t *gpuInfo);

    /*!
     * Allocate an NVK device using which you can query/allocate resources on
     * GPU and do modeset.
     *
     * \param [in]  params  Parameters required for device allocation.
     *
     * \return  An valid device handle on success, NULL on failure.
     */
    struct NvKmsKapiDevice* NVKMS_KAPI_CALL (*allocateDevice)
    (
        const struct NvKmsKapiAllocateDeviceParams *params
    );

    /*!
     * Frees a device allocated by allocateDevice() and all its resources.
     *
     * \param [in]  device  A device returned by allocateDevice().
     *                      This function is a no-op if device is not valid.
     */
    void NVKMS_KAPI_CALL (*freeDevice)(struct NvKmsKapiDevice *device);

    /*!
     * Grab ownership of device, ownership is required to do modeset.
     *
     * \param [in]  device  A device returned by allocateDevice().
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*grabOwnership)(struct NvKmsKapiDevice *device);

    /*!
     * Release ownership of device.
     *
     * \param [in]  device  A device returned by allocateDevice().
     */
    void NVKMS_KAPI_CALL (*releaseOwnership)(struct NvKmsKapiDevice *device);

    /*!
     * Registers for notification, via
     * NvKmsKapiAllocateDeviceParams::eventCallback, of the events specified
     * in interestMask.
     *
     * This call does nothing if eventCallback is NULL when NvKmsKapiDevice
     * is allocated.
     *
     * Supported events are DPY_CHANGED and DYNAMIC_DPY_CONNECTED.
     *
     * \param [in]  device        A device returned by allocateDevice().
     *
     * \param [in]  interestMask  A mask of events requested to listen.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*declareEventInterest)
    (
        const struct NvKmsKapiDevice *device,
        const NvU32 interestMask
    );

    /*!
     * Retrieve various static resources like connector, head etc. present on
     * device and capacities.
     *
     * \param [in]      device  A device allocated using allocateDevice().
     *
     * \param [in/out]  info    A pointer to an NvKmsKapiDeviceResourcesInfo
     *                          struct that the call will fill out with number
     *                          of resources and their handles.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getDeviceResourcesInfo)
    (
        struct NvKmsKapiDevice *device,
        struct NvKmsKapiDeviceResourcesInfo *info
    );

    /*!
     * Retrieve the number of displays on a device and an array of handles to
     * those displays.
     *
     * \param [in]      device          A device allocated using
     *                                  allocateDevice().
     *
     * \param [in/out]  displayCount    The caller should set this to the size
     *                                  of the displayHandles array it passed
     *                                  in. The function will set it to the
     *                                  number of displays returned, or the
     *                                  total number of displays on the device
     *                                  if displayHandles is NULL or array size
     *                                  of less than number of number of displays.
     *
     * \param [out]     displayHandles  An array of display handles with
     *                                  displayCount entries.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getDisplays)
    (
        struct NvKmsKapiDevice *device,
        NvU32 *numDisplays, NvKmsKapiDisplay *displayHandles
    );

    /*!
     * Retrieve information about a specified connector.
     *
     * \param [in]  device      A device allocated using allocateDevice().
     *
     * \param [in]  connector   Which connector to query, handle return by
     *                          getDeviceResourcesInfo().
     *
     * \param [out] info        A pointer to an NvKmsKapiConnectorInfo struct
     *                          that the call will fill out with information
     *                          about connector.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getConnectorInfo)
    (
        struct NvKmsKapiDevice *device,
        NvKmsKapiConnector connector, struct NvKmsKapiConnectorInfo *info
    );

    /*!
     * Retrieve information about a specified display.
     *
     * \param [in]  device    A device allocated using allocateDevice().
     *
     * \param [in]  display   Which connector to query, handle return by
     *                        getDisplays().
     *
     * \param [out] info      A pointer to an NvKmsKapiStaticDisplayInfo struct
     *                        that the call will fill out with information
     *                        about display.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getStaticDisplayInfo)
    (
        struct NvKmsKapiDevice *device,
        NvKmsKapiDisplay display, struct NvKmsKapiStaticDisplayInfo *info
    );

    /*!
     * Detect/force connection status/EDID of display.
     *
     * \param [in/out]  params    Parameters containing display
     *                            handle, EDID and flags to force connection
     *                            status.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getDynamicDisplayInfo)
    (
        struct NvKmsKapiDevice *device,
        struct NvKmsKapiDynamicDisplayParams *params
    );

    /*!
     * Allocate some unformatted memory of the specified size.
     *
     * This function allocates displayable (Usually vidmem) memory on the
     * specified GPU. It should be suitable for mapping on the CPU as a pitch
     * linear surface.
     *
     * \param [in] device  A device allocated using allocateDevice().
     *
     * \param [in] size    Size, in bytes, of the memory to allocate.
     *
     * \return An valid memory handle on success, NULL on failure.
     */
    struct NvKmsKapiMemory* NVKMS_KAPI_CALL (*allocateMemory)
    (
        struct NvKmsKapiDevice *device, NvU64 size
    );

    /*!
     * Import some unformatted memory of the specified size.
     *
     * This function accepts a driver-specific parameter structure representing
     * memory allocated elsewhere and imports it to a NVKMS KAPI memory object
     * of the specified size.
     *
     * \param [in] device  A device allocated using allocateDevice().  The
     *                     memory being imported must have been allocated
     *                     against the same physical device this device object
     *                     represents.
     *
     * \param [in] size    Size, in bytes, of the memory being imported.
     *
     * \param [in] nvKmsParamsUser Userspace pointer to driver-specific
     *                             parameters describing the memory object being
     *                             imported.
     *
     * \param [in] nvKmsParamsSize Size of the driver-specific parameter struct.
     *
     * \return A valid memory handle on success, NULL on failure.
     */
    struct NvKmsKapiMemory* NVKMS_KAPI_CALL (*importMemory)
    (
        struct NvKmsKapiDevice *device, NvU64 size,
        NvU64 nvKmsParamsUser,
        NvU64 nvKmsParamsSize
    );

    /*!
     * Free memory allocated using allocateMemory()
     *
     * \param [in] device  A device allocated using allocateDevice().
     *
     * \param [in] memory  Memory allocated using allocateMemory().
     *
     * \return NV_TRUE on success, NV_FALSE if memory is in use.
     */
    void NVKMS_KAPI_CALL (*freeMemory)
    (
        struct NvKmsKapiDevice *device, struct NvKmsKapiMemory *memory
    );

    /*!
     * Create MMIO mappings for a memory object allocated using
     * allocateMemory().
     *
     * \param [in]  device           A device allocated using allocateDevice().
     *
     * \param [in]  memory           Memory allocated using allocateMemory()
     *
     * \param [in]  type             Userspace or kernelspace mapping
     *
     * \param [out] ppLinearAddress  The MMIO address where memory object is
     *                               mapped.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*mapMemory)
    (
        const struct NvKmsKapiDevice *device,
        const struct NvKmsKapiMemory *memory, NvKmsKapiMappingType type,
        void **ppLinearAddress
    );

    /*!
     * Destroy MMIO mappings created for a memory object allocated using
     * allocateMemory().
     *
     * \param [in]  device           A device allocated using allocateDevice().
     *
     * \param [in]  memory           Memory allocated using allocateMemory()
     *
     * \param [in]  type             Userspace or kernelspace mapping
     *
     * \param [in]  pLinearAddress   The MMIO address return by mapMemory()
     */
    void NVKMS_KAPI_CALL (*unmapMemory)
    (
        const struct NvKmsKapiDevice *device,
        const struct NvKmsKapiMemory *memory, NvKmsKapiMappingType type,
        const void *pLinearAddress
    );

    /*!
     * Create a formatted surface from an NvKmsKapiMemory object.
     *
     * \param [in]  device  A device allocated using allocateDevice().
     *
     * \param [in]  memory  Memory allocated using allocateMemory()
     *
     * \param [in]  format  The format used to interpret the memory contents of
     *                      the surface being created.
     *
     * \param [in]  width   Width of the surface, in pixels.
     *
     * \param [in]  height  Height of the surface, in pixels.
     *
     * \param [in]  pitch   Byte pitch of the surface.
     *
     * \return  An valid surface handle on success.  NULL on failure.
     */
    struct NvKmsKapiSurface* NVKMS_KAPI_CALL (*createSurface)
    (
        struct NvKmsKapiDevice *device,
        struct NvKmsKapiMemory *memory, enum NvKmsSurfaceMemoryFormat format,
        NvU32 width, NvU32 height, NvU32 pitch
    );

    /*!
     * Destroy a surface created by createSurface().
     *
     * \param [in]  device   A device allocated using allocateDevice().
     *
     * \param [in]  surface  A surface created using createSurface()
     */
    void NVKMS_KAPI_CALL (*destroySurface)
    (
        struct NvKmsKapiDevice *device, struct NvKmsKapiSurface *surface
    );

    /*!
     * Enumerate the mode timings available on a given display.
     *
     * \param [in]   device     A device allocated using allocateDevice().
     *
     * \param [in]   display    A display handle returned by  getDisplays().
     *
     * \param [in]   modeIndex  A mode index (Any integer >= 0).
     *
     * \param [out]  mode       A pointer to an NvKmsKapiDisplayMode struct that
     *                          the call will fill out with mode-timings of mode
     *                          at index modeIndex.
     *
     * \param [out]  valid      Returns TRUE in this param if mode-timings of
     *                          mode at index modeIndex are valid on display.
     *
     * \return Value >= 1 if more modes are available, 0 if no more modes are
     *         available, and Value < 0 on failure.
     */
    int NVKMS_KAPI_CALL (*getDisplayMode)
    (
        struct NvKmsKapiDevice *device,
        NvKmsKapiDisplay display, NvU32 modeIndex,
        struct NvKmsKapiDisplayMode *mode, NvBool *valid
    );

    /*!
     * Validate given mode timings available on a given display.
     *
     * \param [in]  device   A device allocated using allocateDevice().
     *
     * \param [in]  display  A display handle returned by  getDisplays().
     *
     * \param [in]  mode     A pointer to an NvKmsKapiDisplayMode struct that
     *                       filled with mode-timings to validate.
     *
     * \return NV_TRUE if mode-timings are valid, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*validateDisplayMode)
    (
        struct NvKmsKapiDevice *device,
        NvKmsKapiDisplay display, const struct NvKmsKapiDisplayMode *mode
    );

    /*!
     * Apply a mode configuration to the device.
     *
     * Client can describe damaged part of configuration but still it is must
     * to describe entire configuration.
     *
     * \param [in]  device            A device allocated using allocateDevice().
     *
     * \param [in]  requestedConfig   Parameters describing a device-wide
     *                                display configuration.
     *
     * \param [in]  commit            If set to 0 them call will only validate
     *                                mode configuration, will not apply it.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*applyModeSetConfig)
    (
        struct NvKmsKapiDevice *device,
        const struct NvKmsKapiRequestedModeSetConfig *requestedConfig,
        const NvBool commit
    );

    /*!
     * Return status of flip.
     *
     * \param  [in]  device   A device allocated using allocateDevice().
     *
     * \param  [in]  head     A head returned by getDeviceResourcesInfo().
     *
     * \param  [in]  plane    A plane type.
     *
     * \param  [out] pending  Return TRUE if head has pending flip for
     *                        given plane.
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getFlipPendingStatus)
    (
        const struct NvKmsKapiDevice *device,
        const NvU32 head,
        const NvKmsKapiPlaneType plane,
        NvBool *pending
    );

    /*!
     * Allocate an event callback.
     *
     * \param [in]  device          A device allocated using allocateDevice().
     *
     * \param [in]  proc            Function pointer to call when triggered.
     *
     * \param [in]  data            Argument to pass into function.
     *
     * \param [in] nvKmsParamsUser  Userspace pointer to driver-specific
     *                              parameters describing the event callback
     *                              being created.
     *
     * \param [in] nvKmsParamsSize  Size of the driver-specific parameter struct.
     *
     * \return struct NvKmsKapiChannelEvent* on success, NULL on failure.
     */
    struct NvKmsKapiChannelEvent* NVKMS_KAPI_CALL (*allocateChannelEvent)
    (
        struct NvKmsKapiDevice *device,
        NvKmsChannelEventProc *proc,
        void *data,
        NvU64 nvKmsParamsUser,
        NvU64 nvKmsParamsSize
    );

    /*!
     * Free an event callback.
     *
     * \param [in]  device  A device allocated using allocateDevice().
     *
     * \param [in]  cb      struct NvKmsKapiChannelEvent* returned from
     *                      allocateChannelEvent()
     */
    void NVKMS_KAPI_CALL (*freeChannelEvent)
    (
        struct NvKmsKapiDevice *device,
        struct NvKmsKapiChannelEvent *cb
    );

    /*!
     * Get 32-bit CRC value for the last contents presented on the specified
     * head.
     *
     * \param [in]  device  A device allocated using allocateDevice().
     *
     * \param [in]  head    A head returned by getDeviceResourcesInfo().
     *
     * \param [out] crc32   The CRC32 generated from the content currently
     *                      presented onto the given head
     *
     * \return NV_TRUE on success, NV_FALSE on failure.
     */
    NvBool NVKMS_KAPI_CALL (*getCRC32)
    (
        struct NvKmsKapiDevice *device,
        NvU32 head,
        NvU32 *crc32
    );
};

/** @} */

/**
 * \defgroup Functions
 * @{
 */

NvBool NVKMS_KAPI_CALL nvKmsKapiGetFunctionsTable
(
    struct NvKmsKapiFunctionsTable *funcsTable
);

/** @} */

#endif /* defined(__NVKMS_KAPI_H__) */
