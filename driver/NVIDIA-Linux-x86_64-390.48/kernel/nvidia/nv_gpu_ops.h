/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


/*
 * nv_gpu_ops.h
 *
 * This file defines the interface between the common RM layer
 * and the OS specific platform layers. (Currently supported 
 * are Linux and KMD)
 *
 */

#ifndef _NV_GPU_OPS_H_
#define _NV_GPU_OPS_H_
#include "nvgputypes.h"
#include "nv_uvm_types.h"

typedef struct gpuSession      *gpuSessionHandle;
typedef struct gpuAddressSpace *gpuAddressSpaceHandle;
typedef struct gpuChannel      *gpuChannelHandle;
typedef struct gpuObject       *gpuObjectHandle;

//
// Default Page Size if left "0" because in RM BIG page size is default & there
// are multiple BIG page sizes in RM. These defines are used as flags to "0" 
// should be OK when user is not sure which pagesize allocation it wants
//
#define PAGE_SIZE_DEFAULT    0x0
#define PAGE_SIZE_4K         0x1000
#define PAGE_SIZE_64K        0x10000
#define PAGE_SIZE_128K       0x20000
#define PAGE_SIZE_2M         0x200000

// Note: Do not modify these structs as nv_uvm_types.h holds
// another definition of the same. Modifying one of the copies will lead
// to struct member mismatch. Note, compiler does not catch such errors.
// TODO: Nuke these structs and add a typedef for UvmGpuTypes*.
struct gpuVaAllocInfo
{
    NvU64    vaStart;                    // Needs to be alinged to pagesize
    NvBool   bFixedAddressAllocate;      // rangeBegin & rangeEnd both included
    NvU32    pageSize;                   // default is where both 4k and 64k page tables will be allocated. 
};

struct gpuMapInfo
{
   NvBool      bPteFlagReadOnly;
   NvBool      bPteFlagAtomic;
   NvBool      bPteFlagsValid;
   NvBool      bApertureIsVid;
   NvBool      bIsContiguous;
   NvU32       pageSize;
};

struct gpuPmaAllocationOptions
{
    NvU32 flags;
    NvU32 minimumSpeed;         // valid if flags & PMA_ALLOCATE_SPECIFY_MININUM_SPEED
    NvU64 physBegin, physEnd;   // valid if flags & PMA_ALLOCATE_SPECIFY_ADDRESS_RANGE
    NvU32 regionId;             // valid if flags & PMA_ALLOCATE_SPECIFY_REGION_ID
    NvU64 alignment;            // valid if flags & PMA_ALLOCATE_FORCE_ALIGNMENT
};

struct gpuChannelPhysInfo
{
    NvU64 pdb;
    NvBool bPdbLocVidmem;
    NvU64 instPtr;
    NvBool bInstPtrLocVidmem;
    NvP64 memHandle; // RM memDesc handle to inst ptr
};

typedef struct gpuRetainedChannel_struct gpuRetainedChannel;

NV_STATUS nvGpuOpsCreateSession(gpuSessionHandle *session);

void nvGpuOpsDestroySession(gpuSessionHandle session);

NV_STATUS nvGpuOpsAddressSpaceCreate(gpuSessionHandle session,
                                     const NvProcessorUuid *gpuUuid,
                                     unsigned long long vaBase,
                                     unsigned long long vaSize,
                                     gpuAddressSpaceHandle *vaSpace,
                                     UvmGpuAddressSpaceInfo *vaSpaceInfo);

NV_STATUS nvGpuOpsGetP2PCaps(gpuAddressSpaceHandle vaSpace1,
                             gpuAddressSpaceHandle vaSpace2,
                             getP2PCapsParams *p2pCaps);

NV_STATUS nvGpuOpsAddressSpaceDestroy(gpuAddressSpaceHandle vaSpace);

// nvGpuOpsMemoryAllocGpuPa and nvGpuOpsFreePhysical were added to support UVM driver
// when PMA was not ready. These should not be used anymore and will be nuked soon.
NV_STATUS nvGpuOpsMemoryAllocGpuPa (struct gpuAddressSpace * vaSpace,
    NvLength length, NvU64 *gpuOffset, gpuAllocInfo * allocInfo);

void nvGpuOpsFreePhysical(struct gpuAddressSpace * vaSpace, NvU64 paOffset);

NV_STATUS nvGpuOpsMemoryAllocFb (gpuAddressSpaceHandle vaSpace,
    NvLength length, NvU64 *gpuOffset, gpuAllocInfo * allocInfo);

NV_STATUS nvGpuOpsMemoryAllocSys (gpuAddressSpaceHandle vaSpace,
    NvLength length, NvU64 *gpuOffset, gpuAllocInfo * allocInfo);

NV_STATUS nvGpuOpsPmaAllocPages(void *pPma,
                                NvLength pageCount,
                                NvU32 pageSize,
                                struct gpuPmaAllocationOptions *pPmaAllocOptions,
                                NvU64 *pPages);

void nvGpuOpsPmaFreePages(void *pPma,
                          NvU64 *pPages,
                          NvLength pageCount,
                          NvU32 pageSize,
                          NvU32 flags);

NV_STATUS nvGpuOpsPmaPinPages(void *pPma,
                              NvU64 *pPages,
                              NvLength pageCount,
                              NvU32 pageSize);

NV_STATUS nvGpuOpsPmaUnpinPages(void *pPma,
                                NvU64 *pPages,
                                NvLength pageCount,
                                NvU32 pageSize);

NV_STATUS nvGpuOpsChannelAllocate(gpuAddressSpaceHandle vaSpace,
                                  gpuChannelHandle  *channelHandle,
                                  const gpuChannelAllocParams *params,
                                  gpuChannelInfo *channelInfo);

NV_STATUS nvGpuOpsMemoryReopen(struct gpuAddressSpace *vaSpace,
     NvHandle hSrcClient, NvHandle hSrcAllocation, NvLength length, NvU64 *gpuOffset);

void nvGpuOpsChannelDestroy(struct gpuChannel *channel);

void nvGpuOpsMemoryFree(gpuAddressSpaceHandle vaSpace,
     NvU64 pointer);

NV_STATUS  nvGpuOpsMemoryCpuMap(gpuAddressSpaceHandle vaSpace,
                                NvU64 memory, NvLength length,
                                void **cpuPtr, NvU32 pageSize);

void nvGpuOpsMemoryCpuUnMap(gpuAddressSpaceHandle vaSpace,
     void* cpuPtr);

NV_STATUS nvGpuOpsCopyEngineAllocate(gpuChannelHandle channel,
          unsigned ceIndex, unsigned *_class, gpuObjectHandle *copyEngineHandle);

NV_STATUS nvGpuOpsCopyEngineAlloc(gpuChannelHandle channel,
                                  unsigned ceIndex,
                                  gpuObjectHandle *copyEngineHandle,
                                  gpuChannelInfo *channelInfo);

NV_STATUS nvGpuOpsQueryCaps(struct gpuAddressSpace *vaSpace,
                            gpuCaps *caps);

NV_STATUS nvGpuOpsDupAllocation(NvHandle hPhysHandle, 
                                struct gpuAddressSpace *sourceVaspace, 
                                NvU64 sourceAddress,
                                struct gpuAddressSpace *destVaspace,
                                NvU64 *destAddress,
                                NvBool bPhysHandleValid);

NV_STATUS nvGpuOpsDupMemory(struct gpuAddressSpace *vaSpace,
                            NvHandle hClient,
                            NvHandle hPhysMemory,
                            NvHandle *hDupMemory,
                            gpuMemoryInfo *pGpuMemoryInfo);

NV_STATUS nvGpuOpsGetGuid(NvHandle hClient, NvHandle hDevice, 
                          NvHandle hSubDevice, NvU8 *gpuGuid, 
                          unsigned guidLength);
                          
NV_STATUS nvGpuOpsGetClientInfoFromPid(unsigned pid,
                                       const NvU8 *gpuUuid,
                                       NvHandle *hClient,
                                       NvHandle *hDevice,
                                       NvHandle *hSubDevice);

NV_STATUS nvGpuOpsFreeDupedHandle(struct gpuAddressSpace *sourceVaspace,
                                  NvHandle hPhysHandle);

NV_STATUS nvGpuOpsGetAttachedGpus(NvU8 *guidList, unsigned *numGpus);

NV_STATUS nvGpuOpsGetGpuInfo(const NvProcessorUuid *gpuUuid, gpuInfo *pGpuInfo);

NV_STATUS nvGpuOpsGetGpuIds(const NvU8 *pUuid, unsigned uuidLength, NvU32 *pDeviceId,
                            NvU32 *pSubdeviceId);

NV_STATUS nvGpuOpsOwnPageFaultIntr(const NvProcessorUuid *gpuUuid, NvBool bOwnInterrupts);

NV_STATUS nvGpuOpsServiceDeviceInterruptsRM(struct gpuAddressSpace *vaSpace);

NV_STATUS nvGpuOpsCheckEccErrorSlowpath(struct gpuChannel * channel, NvBool *bEccDbeSet);

NV_STATUS nvGpuOpsSetPageDirectory(struct gpuAddressSpace * vaSpace,
                                   NvU64 physAddress, unsigned numEntries,
                                   NvBool bVidMemAperture);

NV_STATUS nvGpuOpsUnsetPageDirectory(struct gpuAddressSpace * vaSpace);

NV_STATUS nvGpuOpsGetGmmuFmt(struct gpuAddressSpace * vaSpace, void ** pFmt);

NV_STATUS nvGpuOpsInvalidateTlb(struct gpuAddressSpace * vaSpace);

NV_STATUS nvGpuOpsGetFbInfo(struct gpuAddressSpace * vaSpace, gpuFbInfo * fbInfo);

NV_STATUS nvGpuOpsInitFaultInfo(struct gpuAddressSpace *vaSpace, gpuFaultInfo *pFaultInfo);

NV_STATUS nvGpuOpsDestroyFaultInfo(struct gpuAddressSpace *vaSpace, gpuFaultInfo *pFaultInfo);

NV_STATUS nvGpuOpsHasPendingNonReplayableFaults(gpuFaultInfo *pFaultInfo, NvBool *hasPendingFaults);

NV_STATUS nvGpuOpsGetNonReplayableFaults(gpuFaultInfo *pFaultInfo, void *faultBuffer, NvU32 *numFaults);

NV_STATUS nvGpuOpsDupAddressSpace(struct gpuSession *session,
                                  const NvProcessorUuid *gpuUuid,
                                  NvHandle hUserClient,
                                  NvHandle hUserVASpace,
                                  struct gpuAddressSpace **vaSpace,
                                  UvmGpuAddressSpaceInfo *vaSpaceInfo);

NV_STATUS nvGpuOpsGetPmaObject(const NvProcessorUuid *gpuUUID,
                               void **pPma);

NV_STATUS nvGpuOpsInitAccessCntrInfo(struct gpuAddressSpace *vaSpace, gpuAccessCntrInfo *pAccessCntrInfo);

NV_STATUS nvGpuOpsDestroyAccessCntrInfo(struct gpuAddressSpace *vaSpace, gpuAccessCntrInfo *pAccessCntrInfo);

NV_STATUS nvGpuOpsOwnAccessCntrIntr(struct gpuSession *session, gpuAccessCntrInfo *pAccessCntrInfo,
                                    NvBool bOwnInterrupts);

NV_STATUS nvGpuOpsEnableAccessCntr(struct gpuAddressSpace *vaSpace, gpuAccessCntrInfo *pAccessCntrInfo,
                                   gpuAccessCntrConfig *pAccessCntrConfig);

NV_STATUS nvGpuOpsDisableAccessCntr(struct gpuAddressSpace *vaSpace, gpuAccessCntrInfo *pAccessCntrInfo);

NV_STATUS nvGpuOpsP2pObjectCreate(gpuAddressSpaceHandle vaSpace1,
                                  gpuAddressSpaceHandle vaSpace2,
                                  NvHandle *hP2pObject);

void nvGpuOpsP2pObjectDestroy(struct gpuSession *session,
                              NvHandle hP2pObject);

NV_STATUS nvGpuOpsGetExternalAllocPtes(struct gpuAddressSpace *vaSpace,
                                       NvHandle hDupedMemory,
                                       NvU64 offset,
                                       NvU64 size,
                                       gpuExternalMappingInfo *pGpuExternalMappingInfo);

NV_STATUS nvGpuOpsRetainChannel(struct gpuAddressSpace *vaSpace,
                                NvHandle hClient,
                                NvHandle hChannel,
                                gpuRetainedChannel **retainedChannel,
                                gpuChannelInstanceInfo *channelInstanceInfo);

void nvGpuOpsReleaseChannel(gpuRetainedChannel *retainedChannel);

NV_STATUS nvGpuOpsRetainChannelResources(gpuRetainedChannel *retainedChannel,
                                         gpuChannelResourceInfo *channelResourceInfo);

NV_STATUS nvGpuOpsBindChannelResources(gpuRetainedChannel *retainedChannel,
                                       gpuChannelResourceBindParams *channelResourceBindParams);

void nvGpuOpsReleaseChannelResources(NvP64 *resourceDescriptors, NvU32 descriptorCount);

void nvGpuOpsStopChannel(gpuRetainedChannel *retainedChannel, NvBool bImmediate);

NV_STATUS nvGpuOpsGetChannelResourcePtes(struct gpuAddressSpace *vaSpace,
                                         NvP64 resourceDescriptor,
                                         NvU64 offset,
                                         NvU64 size,
                                         gpuExternalMappingInfo *pGpuExternalMappingInfo);

#endif /* _NV_GPU_OPS_H_*/
