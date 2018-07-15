/*
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#if !defined(NVKMS_IOCTL_H)
#define NVKMS_IOCTL_H

#include "nvtypes.h"

/*!
 * Some of the NVKMS ioctl parameter data structures are quite large
 * and would exceed the parameter size constraints on at least SunOS.
 *
 * Redirect ioctls through a level of indirection: user-space assigns
 * NvKmsIoctlParams with the real command, size, and pointer, and
 * passes the NvKmsIoctlParams through the ioctl.
 */

struct NvKmsIoctlParams {
    NvU32 cmd;
    NvU32 size;
    NvU64 address NV_ALIGN_BYTES(8);
};

#define NVKMS_IOCTL_MAGIC 'm'
#define NVKMS_IOCTL_CMD 0

#define NVKMS_IOCTL_IOWR \
    _IOWR(NVKMS_IOCTL_MAGIC, NVKMS_IOCTL_CMD, struct NvKmsIoctlParams)

/*!
 * User-space pointers are always passed to NVKMS in an NvU64.
 * This user-space address is eventually passed into the platform's
 * copyin/copyout functions, in a void* argument.
 *
 * This utility function converts from an NvU64 to a pointer.
 */

static inline void *nvKmsNvU64ToPointer(NvU64 value)
{
    return (void *)(NvUPtr)value;
}

/*!
 * Before casting the NvU64 to a void*, check that casting to a pointer
 * size within the kernel does not lose any precision in the current
 * environment.
 */
static inline NvBool nvKmsNvU64AddressIsSafe(NvU64 address)
{
    return address == (NvU64)(NvUPtr)address;
}

#endif /* NVKMS_IOCTL_H */
