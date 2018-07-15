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
 * This file sets up the communication between the UVM driver and RM. RM will
 * call the UVM driver providing to it the set of OPS it supports.  UVM will
 * then return by filling out the structure with the callbacks it supports.
 */

#define  __NO_VERSION__

#include "nv-misc.h"
#include "os-interface.h"
#include "nv-linux.h"

#if defined(NV_UVM_ENABLE)

#include "nv_uvm_interface.h"
#include "nv_gpu_ops.h"

// This is really a struct UvmOpsUvmEvents *. It needs to be an atomic because
// it can be read outside of the g_pNvUvmEventsLock. Use getUvmEvents and
// setUvmEvents to access it.
static nv_atomic_long_t g_pNvUvmEvents;
static struct semaphore g_pNvUvmEventsLock;

static struct UvmOpsUvmEvents *getUvmEvents(void)
{
    return (struct UvmOpsUvmEvents *)NV_ATOMIC_LONG_READ(g_pNvUvmEvents);
}

static void setUvmEvents(struct UvmOpsUvmEvents *newEvents)
{
    NV_ATOMIC_LONG_SET(g_pNvUvmEvents, (long)newEvents);
}

static nvidia_stack_t *g_sp;
static struct semaphore g_spLock;

// Use these to test g_sp usage. When DEBUG_GLOBAL_STACK, one out of every
// DEBUG_GLOBAL_STACK_THRESHOLD calls to nvUvmGetSafeStack will use g_sp.
#define DEBUG_GLOBAL_STACK 0
#define DEBUG_GLOBAL_STACK_THRESHOLD 2

static atomic_t g_debugGlobalStackCount = ATOMIC_INIT(0);

// Called at module load, not by an external client
int nv_uvm_init(void)
{
    int rc = nv_kmem_cache_alloc_stack(&g_sp);
    if (rc != 0)
        return rc;

    NV_INIT_MUTEX(&g_spLock);
    NV_INIT_MUTEX(&g_pNvUvmEventsLock);
    return 0;
}

void nv_uvm_exit(void)
{
    // If this fires, the dependent driver never unregistered its callbacks with
    // us before going away, leaving us potentially making callbacks to garbage
    // memory.
    WARN_ON(getUvmEvents() != NULL);

    nv_kmem_cache_free_stack(g_sp);
}


// Testing code to force use of the global stack every now and then
static NvBool forceGlobalStack(void)
{
    // Make sure that we do not try to allocate memory in interrupt or atomic
    // context
    if (DEBUG_GLOBAL_STACK || !NV_MAY_SLEEP())
    {
        if ((atomic_inc_return(&g_debugGlobalStackCount) %
             DEBUG_GLOBAL_STACK_THRESHOLD) == 0)
            return NV_TRUE;
    }
    return NV_FALSE;
}

// Guaranteed to always return a valid stack. It first attempts to allocate one
// from the pool. If that fails, it falls back to the global pre-allocated
// stack. This fallback will serialize.
//
// This is required so paths that free resources do not themselves require
// allocation of resources.
static nvidia_stack_t *nvUvmGetSafeStack(void)
{
    nvidia_stack_t *sp;
    if (forceGlobalStack() || nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        sp = g_sp;
        down(&g_spLock);
    }
    return sp;
}

static void nvUvmFreeSafeStack(nvidia_stack_t *sp)
{
    if (sp == g_sp)
        up(&g_spLock);
    else
        nv_kmem_cache_free_stack(sp);
}

NV_STATUS nvUvmInterfaceRegisterGpu(const NvProcessorUuid *gpuUuid, UvmGpuPlatformInfo *gpuInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;
    int rc;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
        return NV_ERR_NO_MEMORY;

    rc = nvidia_dev_get_uuid(gpuUuid->uuid, sp);
    if (rc == 0)
    {
        rc = nvidia_dev_get_pci_info(gpuUuid->uuid,
                                     &gpuInfo->pci_dev,
                                     &gpuInfo->dma_addressable_start,
                                     &gpuInfo->dma_addressable_limit);
    }

    switch (rc)
    {
        case 0:
            status = NV_OK;
            break;
        case -ENOMEM:
            status = NV_ERR_NO_MEMORY;
            break;
        case -ENODEV:
            status = NV_ERR_GPU_UUID_NOT_FOUND;
            break;
        default:
            status = NV_ERR_GENERIC;
            break;
    }

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceRegisterGpu);

void nvUvmInterfaceUnregisterGpu(const NvProcessorUuid *gpuUuid)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    nvidia_dev_put_uuid(gpuUuid->uuid, sp);
    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceUnregisterGpu);

NV_STATUS nvUvmInterfaceSessionCreate(uvmGpuSessionHandle *session)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_create_session(sp, (gpuSessionHandle *)session);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceSessionCreate);

NV_STATUS nvUvmInterfaceSessionDestroy(uvmGpuSessionHandle session)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    status = rm_gpu_ops_destroy_session(sp, (gpuSessionHandle)session);

    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceSessionDestroy);

NV_STATUS nvUvmInterfaceDupAddressSpace(uvmGpuSessionHandle session,
                                        const NvProcessorUuid *gpuUuid,
                                        NvHandle hUserClient,
                                        NvHandle hUserVASpace,
                                        uvmGpuAddressSpaceHandle *vaSpace,
                                        UvmGpuAddressSpaceInfo *vaSpaceInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_dup_address_space(sp,
                                          (gpuSessionHandle)session,
                                          gpuUuid,
                                          hUserClient,
                                          hUserVASpace,
                                          (gpuAddressSpaceHandle *)vaSpace,
                                          vaSpaceInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDupAddressSpace);

NV_STATUS nvUvmInterfaceAddressSpaceCreate(uvmGpuSessionHandle session,
                                           const NvProcessorUuid *gpuUuid,
                                           unsigned long long vaBase,
                                           unsigned long long vaSize,
                                           uvmGpuAddressSpaceHandle *vaSpace,
                                           UvmGpuAddressSpaceInfo *vaSpaceInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_address_space_create(sp,
                                             (gpuSessionHandle)session,
                                             gpuUuid,
                                             vaBase,
                                             vaSize,
                                             (gpuAddressSpaceHandle *)vaSpace,
                                             vaSpaceInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceAddressSpaceCreate);

void nvUvmInterfaceAddressSpaceDestroy(uvmGpuAddressSpaceHandle vaSpace)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_address_space_destroy(
        sp, (gpuAddressSpaceHandle)vaSpace);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceAddressSpaceDestroy);

NV_STATUS nvUvmInterfaceMemoryAllocFB(uvmGpuAddressSpaceHandle vaSpace,
                    NvLength length, UvmGpuPointer * gpuPointer,
                    UvmGpuAllocInfo * allocInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_memory_alloc_fb(
             sp, (gpuAddressSpaceHandle)vaSpace,
             length, (NvU64 *) gpuPointer,
             allocInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceMemoryAllocFB);

NV_STATUS nvUvmInterfaceMemoryAllocSys(uvmGpuAddressSpaceHandle vaSpace,
                    NvLength length, UvmGpuPointer * gpuPointer,
                    UvmGpuAllocInfo * allocInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_memory_alloc_sys(
             sp, (gpuAddressSpaceHandle)vaSpace,
             length, (NvU64 *) gpuPointer,
             allocInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}

EXPORT_SYMBOL(nvUvmInterfaceMemoryAllocSys);

NV_STATUS nvUvmInterfaceGetP2PCaps(uvmGpuAddressSpaceHandle vaSpace1,
                                   uvmGpuAddressSpaceHandle vaSpace2,
                                   UvmGpuP2PCapsParams * p2pCapsParams)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_p2p_caps(sp,
                                     (gpuAddressSpaceHandle)vaSpace1,
                                     (gpuAddressSpaceHandle)vaSpace2,
                                     p2pCapsParams);
    nv_kmem_cache_free_stack(sp);
    return status;
}

EXPORT_SYMBOL(nvUvmInterfaceGetP2PCaps);

NV_STATUS nvUvmInterfaceGetPmaObject(const NvProcessorUuid *gpuUUID, void **pPma)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_pma_object(sp, gpuUUID, pPma);

    nv_kmem_cache_free_stack(sp);
    return status;
}

EXPORT_SYMBOL(nvUvmInterfaceGetPmaObject);

NV_STATUS nvUvmInterfacePmaRegisterEvictionCallbacks(void *pPma,
                                                     uvmPmaEvictPagesCallback evictPages,
                                                     uvmPmaEvictRangeCallback evictRange,
                                                     void *callbackData)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_pma_register_callbacks(sp, pPma, evictPages, evictRange, callbackData);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfacePmaRegisterEvictionCallbacks);

void nvUvmInterfacePmaUnregisterEvictionCallbacks(void *pPma)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_pma_unregister_callbacks(sp, pPma);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfacePmaUnregisterEvictionCallbacks);

NV_STATUS nvUvmInterfacePmaAllocPages(void *pPma,
                                      NvLength pageCount,
                                      NvU32 pageSize,
                                      UvmPmaAllocationOptions *pPmaAllocOptions,
                                      NvU64 *pPages)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_pma_alloc_pages(
             sp, pPma,
             pageCount,
             pageSize,
             (nvgpuPmaAllocationOptions_t)pPmaAllocOptions,
             pPages);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfacePmaAllocPages);

NV_STATUS nvUvmInterfacePmaPinPages(void *pPma,
                                    NvU64 *pPages,
                                    NvLength pageCount,
                                    NvU32 pageSize,
                                    NvU32 flags)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_pma_pin_pages(sp, pPma, pPages, pageCount, pageSize, flags);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfacePmaPinPages);

NV_STATUS nvUvmInterfacePmaUnpinPages(void *pPma,
                                      NvU64 *pPages,
                                      NvLength pageCount,
                                      NvU32 pageSize)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_pma_unpin_pages(sp, pPma, pPages, pageCount, pageSize);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfacePmaUnpinPages);

void nvUvmInterfaceMemoryFree(uvmGpuAddressSpaceHandle vaSpace,
                    UvmGpuPointer gpuPointer)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_memory_free(
    sp, (gpuAddressSpaceHandle)vaSpace,
    (NvU64) gpuPointer);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceMemoryFree);

void nvUvmInterfacePmaFreePages(void *pPma,
                                NvU64 *pPages,
                                NvLength pageCount,
                                NvU32 pageSize,
                                NvU32 flags)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_pma_free_pages(sp, pPma, pPages, pageCount, pageSize, flags);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfacePmaFreePages);

NV_STATUS nvUvmInterfaceMemoryCpuMap(uvmGpuAddressSpaceHandle vaSpace,
           UvmGpuPointer gpuPointer, NvLength length, void **cpuPtr,
           NvU32 pageSize)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_memory_cpu_map(
             sp, (gpuAddressSpaceHandle)vaSpace,
             (NvU64) gpuPointer, length, cpuPtr, pageSize);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceMemoryCpuMap);

void nvUvmInterfaceMemoryCpuUnMap(uvmGpuAddressSpaceHandle vaSpace,
                                  void *cpuPtr)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    rm_gpu_ops_memory_cpu_ummap(sp, (gpuAddressSpaceHandle)vaSpace, cpuPtr);
    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceMemoryCpuUnMap);

NV_STATUS nvUvmInterfaceChannelAllocate(uvmGpuAddressSpaceHandle  vaSpace,
                                        uvmGpuChannelHandle *channel,
                                        const UvmGpuChannelAllocParams *allocParams,
                                        UvmGpuChannelPointers * pointers)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_channel_allocate(sp,
                                         (gpuAddressSpaceHandle)vaSpace,
                                         (gpuChannelHandle *)channel,
                                         allocParams,
                                         pointers);

    nv_kmem_cache_free_stack(sp);

    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceChannelAllocate);

void nvUvmInterfaceChannelDestroy(uvmGpuChannelHandle channel)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    rm_gpu_ops_channel_destroy(sp, (gpuChannelHandle)channel);
    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceChannelDestroy);

NV_STATUS nvUvmInterfaceCopyEngineAlloc(uvmGpuChannelHandle channel,
                                        unsigned copyEngineIndex,
                                        uvmGpuCopyEngineHandle *copyEngine,
                                        UvmGpuChannelPointers *pointers)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_copy_engine_alloc(sp,
                                          (gpuChannelHandle) channel,
                                          copyEngineIndex,
                                          (gpuObjectHandle *) copyEngine,
                                          pointers);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceCopyEngineAlloc);

NV_STATUS nvUvmInterfaceQueryCaps(uvmGpuAddressSpaceHandle vaSpace,
                                  UvmGpuCaps * caps)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_query_caps(sp, (gpuAddressSpaceHandle)vaSpace, caps);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceQueryCaps);

NV_STATUS nvUvmInterfaceGetGpuInfo(const NvProcessorUuid *gpuUuid, UvmGpuInfo *pGpuInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_gpu_info(sp, gpuUuid, pGpuInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceGetGpuInfo);

NV_STATUS nvUvmInterfaceServiceDeviceInterruptsRM(uvmGpuAddressSpaceHandle vaSpace)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_service_device_interrupts_rm(sp,
                                                    (gpuAddressSpaceHandle) vaSpace);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceServiceDeviceInterruptsRM);

NV_STATUS nvUvmInterfaceSetPageDirectory(uvmGpuAddressSpaceHandle vaSpace,
                                         NvU64 physAddress, unsigned numEntries,
                                         NvBool bVidMemAperture)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_set_page_directory(sp, (gpuAddressSpaceHandle)vaSpace,
                                    physAddress, numEntries, bVidMemAperture);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceSetPageDirectory);

NV_STATUS nvUvmInterfaceUnsetPageDirectory(uvmGpuAddressSpaceHandle vaSpace)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    status =
           rm_gpu_ops_unset_page_directory(sp, (gpuAddressSpaceHandle)vaSpace);
    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceUnsetPageDirectory);

NV_STATUS nvUvmInterfaceDupAllocation(NvHandle hPhysHandle,
                                      uvmGpuAddressSpaceHandle srcVaspace,
                                      NvU64 srcAddress,
                                      uvmGpuAddressSpaceHandle dstVaspace,
                                      NvU64 *dstAddress,
                                      NvBool bPhysHandleValid)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_dup_allocation(sp,
                                      hPhysHandle,
                                      (gpuAddressSpaceHandle)srcVaspace,
                                      srcAddress,
                                      (gpuAddressSpaceHandle)dstVaspace,
                                      dstAddress,
                                      bPhysHandleValid);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDupAllocation);

NV_STATUS nvUvmInterfaceDupMemory(uvmGpuAddressSpaceHandle vaSpace,
                                  NvHandle hClient,
                                  NvHandle hPhysMemory,
                                  NvHandle *hDupMemory,
                                  UvmGpuMemoryInfo *pGpuMemoryInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_dup_memory(sp,
                                   (gpuAddressSpaceHandle)vaSpace,
                                   hClient,
                                   hPhysMemory,
                                   hDupMemory,
                                   pGpuMemoryInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDupMemory);


NV_STATUS nvUvmInterfaceFreeDupedHandle(uvmGpuAddressSpaceHandle vaspace,
                                        NvHandle hPhysHandle)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    status = rm_gpu_ops_free_duped_handle(sp,
                                         (gpuAddressSpaceHandle)vaspace,
                                         hPhysHandle);

    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceFreeDupedHandle);

NV_STATUS nvUvmInterfaceGetFbInfo(uvmGpuAddressSpaceHandle vaSpace,
                                  UvmGpuFbInfo * fbInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_fb_info(sp, (gpuAddressSpaceHandle)vaSpace, fbInfo);

    nv_kmem_cache_free_stack(sp);

    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceGetFbInfo);

NV_STATUS nvUvmInterfaceOwnPageFaultIntr(const NvProcessorUuid *gpuUuid, NvBool bOwnInterrupts)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_own_page_fault_intr(sp, gpuUuid, bOwnInterrupts);
    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceOwnPageFaultIntr);


NV_STATUS nvUvmInterfaceInitFaultInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuFaultInfo *pFaultInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_init_fault_info(sp,
                                       (gpuAddressSpaceHandle)vaSpace,
                                       pFaultInfo);

    // Preallocate a stack for functions called from ISR top half
    pFaultInfo->nonReplayable.isr_sp = NULL;
    pFaultInfo->nonReplayable.isr_bh_sp = NULL;
    if (status == NV_OK)
    {
        // NOTE: nv_kmem_cache_alloc_stack does not allocate a stack on PPC.
        // Therefore, the pointer can be NULL on success. Always use the
        // returned error code to determine if the operation was successful.
        int err = nv_kmem_cache_alloc_stack((nvidia_stack_t **)&pFaultInfo->nonReplayable.isr_sp);
        if (!err)
        {
            err = nv_kmem_cache_alloc_stack((nvidia_stack_t **)&pFaultInfo->nonReplayable.isr_bh_sp);
            if (err)
            {
                nv_kmem_cache_free_stack(pFaultInfo->nonReplayable.isr_sp);
                pFaultInfo->nonReplayable.isr_sp = NULL;
            }
        }

        if (err)
        {
            rm_gpu_ops_destroy_fault_info(sp,
                                          (gpuAddressSpaceHandle)vaSpace,
                                          pFaultInfo);

            status = NV_ERR_NO_MEMORY;
        }
    }

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceInitFaultInfo);

NV_STATUS nvUvmInterfaceInitAccessCntrInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_init_access_cntr_info(sp,
                                       (gpuAddressSpaceHandle)vaSpace,
                                       pAccessCntrInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceInitAccessCntrInfo);

NV_STATUS nvUvmInterfaceOwnAccessCntrIntr(uvmGpuSessionHandle session,
    UvmGpuAccessCntrInfo *pAccessCntrInfo,
    NvBool bOwnInterrupts)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_own_access_cntr_intr(sp, (gpuSessionHandle)session,
                                             pAccessCntrInfo,
                                             bOwnInterrupts);
    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceOwnAccessCntrIntr);

NV_STATUS nvUvmInterfaceEnableAccessCntr(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo,
    UvmGpuAccessCntrConfig *pAccessCntrConfig)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_enable_access_cntr (sp,
                                       (gpuAddressSpaceHandle)vaSpace,
                                       pAccessCntrInfo,
                                       pAccessCntrConfig);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceEnableAccessCntr);

NV_STATUS nvUvmInterfaceDestroyFaultInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuFaultInfo *pFaultInfo)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    // Free the preallocated stack for functions called from ISR
    if (pFaultInfo->nonReplayable.isr_sp != NULL)
    {
        nv_kmem_cache_free_stack((nvidia_stack_t *)pFaultInfo->nonReplayable.isr_sp);
        pFaultInfo->nonReplayable.isr_sp = NULL;
    }

    if (pFaultInfo->nonReplayable.isr_bh_sp != NULL)
    {
        nv_kmem_cache_free_stack((nvidia_stack_t *)pFaultInfo->nonReplayable.isr_bh_sp);
        pFaultInfo->nonReplayable.isr_bh_sp = NULL;
    }

    status = rm_gpu_ops_destroy_fault_info(sp,
                                          (gpuAddressSpaceHandle)vaSpace,
                                          pFaultInfo);

    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDestroyFaultInfo);

NV_STATUS nvUvmInterfaceHasPendingNonReplayableFaults(UvmGpuFaultInfo *pFaultInfo,
                                                      NvBool *hasPendingFaults)
{
    return rm_gpu_ops_has_pending_non_replayable_faults(pFaultInfo->nonReplayable.isr_sp,
                                                        pFaultInfo,
                                                        hasPendingFaults);
}
EXPORT_SYMBOL(nvUvmInterfaceHasPendingNonReplayableFaults);

NV_STATUS nvUvmInterfaceGetNonReplayableFaults(UvmGpuFaultInfo *pFaultInfo,
                                               void *pFaultBuffer,
                                               NvU32 *numFaults)
{
    return rm_gpu_ops_get_non_replayable_faults(pFaultInfo->nonReplayable.isr_bh_sp,
                                                pFaultInfo,
                                                pFaultBuffer,
                                                numFaults);
}
EXPORT_SYMBOL(nvUvmInterfaceGetNonReplayableFaults);

NV_STATUS nvUvmInterfaceDestroyAccessCntrInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    status = rm_gpu_ops_destroy_access_cntr_info(sp,
                                          (gpuAddressSpaceHandle)vaSpace,
                                          pAccessCntrInfo);

    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDestroyAccessCntrInfo);

NV_STATUS nvUvmInterfaceDisableAccessCntr(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();
    NV_STATUS status;

    status = rm_gpu_ops_disable_access_cntr(sp,
                                          (gpuAddressSpaceHandle)vaSpace,
                                          pAccessCntrInfo);

    nvUvmFreeSafeStack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceDisableAccessCntr);

// this function is called by the UVM driver to register the ops
NV_STATUS nvUvmInterfaceRegisterUvmCallbacks(struct UvmOpsUvmEvents *importedUvmOps)
{
    NV_STATUS status = NV_OK;

    if (!importedUvmOps)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    down(&g_pNvUvmEventsLock);
    if (getUvmEvents() != NULL)
    {
        status = NV_ERR_IN_USE;
    }
    else
    {
        // Be careful: as soon as the pointer is assigned, top half ISRs can
        // start reading it to make callbacks, even before we drop the lock.
        setUvmEvents(importedUvmOps);
    }
    up(&g_pNvUvmEventsLock);

    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceRegisterUvmCallbacks);

static void flush_top_half(void *info)
{
    // Prior top halves on this core must have completed for this callback to
    // run at all, so we're done.
    return;
}

void nvUvmInterfaceDeRegisterUvmOps(void)
{
    // Taking the lock forces us to wait for non-interrupt callbacks to finish
    // up.
    down(&g_pNvUvmEventsLock);
    setUvmEvents(NULL);
    up(&g_pNvUvmEventsLock);

    // We cleared the pointer so nv_uvm_event_interrupt can't invoke any new
    // top half callbacks, but prior ones could still be executing on other
    // cores. We can wait for them to finish by waiting for a context switch to
    // happen on every core.
    //
    // This is slow, but since nvUvmInterfaceDeRegisterUvmOps is very rare
    // (module unload) it beats having the top half synchronize with a spin lock
    // every time.
    //
    // Note that since we dropped the lock, another set of callbacks could have
    // already been registered. That's ok, since we just need to wait for old
    // ones to finish.
    NV_ON_EACH_CPU(flush_top_half, NULL);
}
EXPORT_SYMBOL(nvUvmInterfaceDeRegisterUvmOps);

void nv_uvm_notify_start_device(const NvU8 *pUuid)
{
    NvProcessorUuid uvmUuid;
    struct UvmOpsUvmEvents *events;

    memcpy(uvmUuid.uuid, pUuid, UVM_UUID_LEN);

    // Synchronize callbacks with unregistration
    down(&g_pNvUvmEventsLock);

    // It's not strictly necessary to use a cached local copy of the events
    // pointer here since it can't change under the lock, but we'll do it for
    // consistency.
    events = getUvmEvents();
    if(events && events->startDevice)
    {
        events->startDevice(&uvmUuid);
    }
    up(&g_pNvUvmEventsLock);
}

void nv_uvm_notify_stop_device(const NvU8 *pUuid)
{
    NvProcessorUuid uvmUuid;
    struct UvmOpsUvmEvents *events;

    memcpy(uvmUuid.uuid, pUuid, UVM_UUID_LEN);

    // Synchronize callbacks with unregistration
    down(&g_pNvUvmEventsLock);

    // It's not strictly necessary to use a cached local copy of the events
    // pointer here since it can't change under the lock, but we'll do it for
    // consistency.
    events = getUvmEvents();
    if(events && events->stopDevice)
    {
        events->stopDevice(&uvmUuid);
    }
    up(&g_pNvUvmEventsLock);
}

NV_STATUS nv_uvm_event_interrupt(const NvU8 *pUuid)
{
    //
    // This is called from interrupt context, so we can't take
    // g_pNvUvmEventsLock to prevent the callbacks from being unregistered. Even
    // if we could take the lock, we don't want to slow down the ISR more than
    // absolutely necessary.
    //
    // Instead, we allow this function to be called concurrently with
    // nvUvmInterfaceDeRegisterUvmOps. That function will clear the events
    // pointer, then wait for all top halves to finish out. This means the
    // pointer may change out from under us, but the callbacks are still safe to
    // invoke while we're in this function.
    //
    // This requires that we read the pointer exactly once here so neither we
    // nor the compiler make assumptions about the pointer remaining valid while
    // in this function.
    //
    struct UvmOpsUvmEvents *events = getUvmEvents();

    if (events && events->isrTopHalf)
        return events->isrTopHalf((const NvProcessorUuid *)pUuid);

    //
    // NV_OK means that the interrupt was for the UVM driver, so use
    // NV_ERR_NO_INTR_PENDING to tell the caller that we didn't do anything.
    //
    return NV_ERR_NO_INTR_PENDING;
}

NV_STATUS nvUvmInterfaceP2pObjectCreate(uvmGpuAddressSpaceHandle vaSpace1,
                                        uvmGpuAddressSpaceHandle vaSpace2,
                                        NvHandle *hP2pObject)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;
    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_p2p_object_create(sp,
                                          (gpuAddressSpaceHandle)vaSpace1,
                                          (gpuAddressSpaceHandle)vaSpace2,
                                          hP2pObject);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceP2pObjectCreate);

void nvUvmInterfaceP2pObjectDestroy(uvmGpuSessionHandle session,
                                         NvHandle hP2pObject)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_p2p_object_destroy(sp, (gpuSessionHandle)session, hP2pObject);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceP2pObjectDestroy);

NV_STATUS nvUvmInterfaceGetExternalAllocPtes(uvmGpuAddressSpaceHandle vaSpace,
                                             NvHandle hDupedMemory,
                                             NvU64 offset,
                                             NvU64 size,
                                             UvmGpuExternalMappingInfo *gpuExternalMappingInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_external_alloc_ptes(sp,
                                                (gpuAddressSpaceHandle)vaSpace,
                                                hDupedMemory,
                                                offset,
                                                size,
                                                gpuExternalMappingInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceGetExternalAllocPtes);

NV_STATUS nvUvmInterfaceRetainChannel(uvmGpuAddressSpaceHandle vaSpace,
                                      NvHandle hClient,
                                      NvHandle hChannel,
                                      void **retainedChannel,
                                      UvmGpuChannelInstanceInfo *channelInstanceInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_retain_channel(sp,
                                       (gpuAddressSpaceHandle)vaSpace,
                                       hClient,
                                       hChannel,
                                       retainedChannel,
                                       channelInstanceInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceRetainChannel);

NV_STATUS nvUvmInterfaceRetainChannelResources(void *retainedChannel,
                                               UvmGpuChannelResourceInfo *channelResourceInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_retain_channel_resources(sp,
                                                 retainedChannel,
                                                 channelResourceInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceRetainChannelResources);

NV_STATUS nvUvmInterfaceBindChannelResources(void *retainedChannel,
                                             UvmGpuChannelResourceBindParams *channelResourceBindParams)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_bind_channel_resources(sp,
                                               retainedChannel,
                                               channelResourceBindParams);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceBindChannelResources);

void nvUvmInterfaceReleaseChannel(void *retainedChannel)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_release_channel(sp, retainedChannel);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceReleaseChannel);

void nvUvmInterfaceReleaseChannelResources(NvP64 *resourceDescriptors, NvU32 descriptorCount)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_release_channel_resources(sp, resourceDescriptors, descriptorCount);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceReleaseChannelResources);

void nvUvmInterfaceStopChannel(void *retainedChannel, NvBool bImmediate)
{
    nvidia_stack_t *sp = nvUvmGetSafeStack();

    rm_gpu_ops_stop_channel(sp, retainedChannel, bImmediate);

    nvUvmFreeSafeStack(sp);
}
EXPORT_SYMBOL(nvUvmInterfaceStopChannel);

NV_STATUS nvUvmInterfaceGetChannelResourcePtes(uvmGpuAddressSpaceHandle vaSpace,
                                               NvP64 resourceDescriptor,
                                               NvU64 offset,
                                               NvU64 size,
                                               UvmGpuExternalMappingInfo *externalMappingInfo)
{
    nvidia_stack_t *sp = NULL;
    NV_STATUS status;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return NV_ERR_NO_MEMORY;
    }

    status = rm_gpu_ops_get_channel_resource_ptes(sp,
                                                  (gpuAddressSpaceHandle)vaSpace,
                                                  resourceDescriptor,
                                                  offset,
                                                  size,
                                                  externalMappingInfo);

    nv_kmem_cache_free_stack(sp);
    return status;
}
EXPORT_SYMBOL(nvUvmInterfaceGetChannelResourcePtes);

#endif // NV_UVM_ENABLE
