/*
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __NV_KMS_H__
#define __NV_KMS_H__

#include "nvtypes.h"
#include <stddef.h> /* size_t */

#include "nvkms-kapi.h"

/*
 * On Linux-x86, the kernel's function calling convention may pass
 * parameters in registers.  Force functions called to and from core
 * NVKMS to pass parameters on the stack.
 */
#if NVCPU_IS_X86
  #define NVKMS_API_CALL __attribute__((regparm(0)))
#else
  #define NVKMS_API_CALL
#endif

typedef struct nvkms_per_open nvkms_per_open_handle_t;

typedef void NVKMS_API_CALL nvkms_procfs_out_string_func_t(void *data,
                                                           const char *str);

typedef void NVKMS_API_CALL nvkms_procfs_proc_t(void *data,
                                                char *buffer, size_t size,
                                                nvkms_procfs_out_string_func_t *outString);

typedef struct {
    const char *name;
    nvkms_procfs_proc_t *func;
} nvkms_procfs_file_t;

enum NvKmsClientType {
    NVKMS_CLIENT_USER_SPACE,
    NVKMS_CLIENT_KERNEL_SPACE,
};

NvBool NVKMS_API_CALL nvKmsIoctl(
    void *pOpenVoid,
    NvU32 cmd,
    NvU64 paramsAddress,
    const size_t paramSize);

void NVKMS_API_CALL nvKmsClose(void *pOpenVoid);

void* NVKMS_API_CALL nvKmsOpen(
    NvU32 pid,
    enum NvKmsClientType clientType,
    nvkms_per_open_handle_t *pOpenKernel);

void NVKMS_API_CALL nvKmsModuleLoad(void);

void NVKMS_API_CALL nvKmsModuleUnload(void);

void NVKMS_API_CALL nvKmsSuspend(NvU32 gpuId);
void NVKMS_API_CALL nvKmsResume(NvU32 gpuId);

void NVKMS_API_CALL nvKmsGetProcFiles(const nvkms_procfs_file_t **ppProcFiles);

void NVKMS_API_CALL nvKmsKapiHandleEventQueueChange
(
    struct NvKmsKapiDevice *device
);

NvBool NVKMS_API_CALL nvKmsKapiGetFunctionsTableInternal
(
    struct NvKmsKapiFunctionsTable *funcsTable
);

#endif /* __NV_KMS_H__ */
