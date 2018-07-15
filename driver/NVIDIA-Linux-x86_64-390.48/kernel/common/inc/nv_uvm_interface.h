/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

//
// This file provides the interface that RM exposes to UVM.
//

#ifndef _NV_UVM_INTERFACE_H_
#define _NV_UVM_INTERFACE_H_

// Forward references, to break circular header file dependencies:
struct UvmOpsUvmEvents;

//
// TODO (bug 1359805): This should all be greatly simplified. It is still
// carrying along a lot of baggage from when RM depended on UVM. Now that
// direction is reversed: RM is independent, and UVM depends on RM.
//
#if defined(NVIDIA_UVM_ENABLED)

// We are in the UVM build system, for a Linux target.
#include "uvmtypes.h"
#include "uvm_linux.h"

#else

// We are in the RM build system, for a Linux target:
#include "nv-linux.h"

#endif // NVIDIA_UVM_ENABLED

#include "nvgputypes.h"
#include "nvstatus.h"
#include "nv_uvm_types.h"

// Define the type here as it's Linux specific, used only by the Linux specific
// nvUvmInterfaceRegisterGpu() API.
typedef struct
{
    struct pci_dev *pci_dev;

    // DMA addressable range of the device, mirrors fields in nv_state_t.
    NvU64 dma_addressable_start;
    NvU64 dma_addressable_limit;
} UvmGpuPlatformInfo;

/*******************************************************************************
    nvUvmInterfaceRegisterGpu

    Registers the GPU with the provided UUID for use. A GPU must be registered
    before its UUID can be used with any other API. This call is ref-counted so
    every nvUvmInterfaceRegisterGpu must be paired with a corresponding
    nvUvmInterfaceUnregisterGpu.

    You don't need to call nvUvmInterfaceSessionCreate before calling this.

    Error codes:
        NV_ERR_GPU_UUID_NOT_FOUND
        NV_ERR_NO_MEMORY
        NV_ERR_GENERIC
*/
NV_STATUS nvUvmInterfaceRegisterGpu(const NvProcessorUuid *gpuUuid, UvmGpuPlatformInfo *gpuInfo);

/*******************************************************************************
    nvUvmInterfaceUnregisterGpu

    Unregisters the GPU with the provided UUID. This drops the ref count from
    nvUvmInterfaceRegisterGpu. Once the reference count goes to 0 the device may
    no longer be accessible until the next nvUvmInterfaceRegisterGpu call. No
    automatic resource freeing is performed, so only make the last unregister
    call after destroying all your allocations associated with that UUID (such
    as those from nvUvmInterfaceAddressSpaceCreate).

    If the UUID is not found, no operation is performed.
*/
void nvUvmInterfaceUnregisterGpu(const NvProcessorUuid *gpuUuid);

/*******************************************************************************
    nvUvmInterfaceSessionCreate

    TODO: Creates session object.  All allocations are tied to the session.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceSessionCreate(uvmGpuSessionHandle *session);

/*******************************************************************************
    nvUvmInterfaceSessionDestroy

    Destroys a session object.  All allocations are tied to the session will
    be destroyed.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceSessionDestroy(uvmGpuSessionHandle session);

/*******************************************************************************
    nvUvmInterfaceAddressSpaceCreate

    This function creates an address space.
    This virtual address space is created on the GPU specified
    by the gpuUuid.


    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceAddressSpaceCreate(uvmGpuSessionHandle session,
                                           const NvProcessorUuid *gpuUuid,
                                           unsigned long long vaBase,
                                           unsigned long long vaSize,
                                           uvmGpuAddressSpaceHandle *vaSpace,
                                           UvmGpuAddressSpaceInfo *vaSpaceInfo);

/*******************************************************************************
    nvUvmInterfaceDupAddressSpace

    This function will dup the given vaspace from the users client to the 
    kernel client was created as an ops session.
    
    By duping the vaspace it is guaranteed that RM will refcount the vaspace object.

    Error codes:
      NV_ERR_GENERIC
*/
NV_STATUS nvUvmInterfaceDupAddressSpace(uvmGpuSessionHandle session,
                                        const NvProcessorUuid *gpuUuid,
                                        NvHandle hUserClient,
                                        NvHandle hUserVASpace,
                                        uvmGpuAddressSpaceHandle *vaSpace,
                                        UvmGpuAddressSpaceInfo *vaSpaceInfo);

/*******************************************************************************
    nvUvmInterfaceAddressSpaceDestroy

    Destroys an address space that was previously created via
    nvUvmInterfaceAddressSpaceCreate.
*/

void nvUvmInterfaceAddressSpaceDestroy(uvmGpuAddressSpaceHandle vaSpace);

/*******************************************************************************
    nvUvmInterfaceMemoryAllocFB

    This function will allocate video memory and provide a mapped Gpu
    virtual address to this allocation. It also returns the Gpu physical
    offset if contiguous allocations are requested.

    This function will allocate a minimum page size if the length provided is 0
    and will return a unique GPU virtual address.

    The default page size will be the small page size (as returned by query
    caps). The Alignment will also be enforced to small page size(64K/128K).
 
    Arguments:
        vaSpace[IN]          - Pointer to vaSpace object
        length [IN]          - Length of the allocation
        gpuPointer[OUT]      - GPU VA mapping
        allocInfo[IN/OUT]    - Pointer to allocation info structure which
                               contains below given fields
 
        allocInfo Members: 
        rangeBegin[IN]             - Allocation will be made between rangeBegin
        rangeEnd[IN]                 and rangeEnd(both inclusive). Default will be
                                     no-range limitation.
        gpuPhysOffset[OUT]         - Physical offset of allocation returned only
                                     if contiguous allocation is requested. 
        bContiguousPhysAlloc[IN]   - Flag to request contiguous allocation. Default
                                     will follow the vidHeapControl default policy.
        bHandleProvided [IN]       - Flag to signify that the client has provided 
                                     the handle for phys allocation.
        hPhysHandle[IN/OUT]        - The handle will be used in allocation if provided.
                                     If not provided; allocator will return the handle
                                     it used eventually.
    Error codes:
        NV_ERR_INVALID_ARGUMENT  
        NV_ERR_NO_MEMORY              - Not enough physical memory to service
                                        allocation request with provided constraints
        NV_ERR_INSUFFICIENT_RESOURCES - Not enough available resources to satisfy allocation request
        NV_ERR_INVALID_OWNER          - Target memory not accessible by specified owner
        NV_ERR_NOT_SUPPORTED          - Operation not supported on broken FB
 
*/
NV_STATUS nvUvmInterfaceMemoryAllocFB(uvmGpuAddressSpaceHandle vaSpace,
                                      NvLength length,
                                      UvmGpuPointer * gpuPointer,
                                      UvmGpuAllocInfo * allocInfo);

/*******************************************************************************
    nvUvmInterfaceMemoryAllocSys

    This function will allocate system memory and provide a mapped Gpu
    virtual address to this allocation.

    This function will allocate a minimum page size if the length provided is 0
    and will return a unique GPU virtual address.

    The default page size will be the small page size (as returned by query caps)
    The Alignment will also be enforced to small page size.

    Arguments:
        vaSpace[IN]          - Pointer to vaSpace object
        length [IN]          - Length of the allocation
        gpuPointer[OUT]      - GPU VA mapping
        allocInfo[IN/OUT]    - Pointer to allocation info structure which
                               contains below given fields
 
        allocInfo Members: 
        rangeBegin[IN]             - Allocation will be made between rangeBegin
        rangeEnd[IN]                 and rangeEnd(both inclusive). Default will be
                                     no-range limitation.
        gpuPhysOffset[OUT]         - Physical offset of allocation returned only
                                     if contiguous allocation is requested. 
        bContiguousPhysAlloc[IN]   - Flag to request contiguous allocation. Default
                                     will follow the vidHeapControl default policy.
        bHandleProvided [IN]       - Flag to signify that the client has provided 
                                     the handle for phys allocation.
        hPhysHandle[IN/OUT]        - The handle will be used in allocation if provided.
                                     If not provided; allocator will return the handle
                                     it used eventually.
    Error codes:
        NV_ERR_INVALID_ARGUMENT  
        NV_ERR_NO_MEMORY              - Not enough physical memory to service
                                        allocation request with provided constraints
        NV_ERR_INSUFFICIENT_RESOURCES - Not enough available resources to satisfy allocation request
        NV_ERR_INVALID_OWNER          - Target memory not accessible by specified owner
        NV_ERR_NOT_SUPPORTED          - Operation not supported
*/
NV_STATUS nvUvmInterfaceMemoryAllocSys(uvmGpuAddressSpaceHandle vaSpace,
                                       NvLength length,
                                       UvmGpuPointer * gpuPointer,
                                       UvmGpuAllocInfo * allocInfo);

/*******************************************************************************
    nvUvmInterfaceGetP2PCaps

    Obtain the P2P capabilities between the devices with the given gpu address
    space handles.

    This function accepts gpu address space handles or NULL to indicate cpu.
    At least one gpu address space handle is required. Only cpus connected
    to a gpu using an npu + nvlink2 or greater are considered to be peers.

    If a cpu is included in the query or indirect access is reported, the
    contents of peerIds are undefined.

    Arguments:
        vaSpace1[IN]              - first gpu address space handle
        vaSpace1[IN]              - second gpu address space handle
        p2pCapsParams members:
        peerLink[OUT]             - type of the link enabled between the peers
        peerIds[OUT]              - peer ids between given pair of peers
        writeSupported[OUT]       - p2p writes between peers are supported
        readSupported[OUT]        - p2p reads between peers are supported 
        atomicSupported[OUT]      - p2p atomics between peers are supported
        directAccess[OUT]         - if a valid peer link is returned, this
                                    field tells whether there is a direct link
                                    between the devices or they communicate
                                    through an intermediate device such as
                                    a npu. If indirect access is reported link
                                    must be NVLink2 or greater.

    Error codes:
        NV_ERR_INVALID_ARGUMENT
        NV_ERR_GENERIC:
          Unexpected error. We try hard to avoid returning this error
          code,because it is not very informative.

*/
NV_STATUS nvUvmInterfaceGetP2PCaps(uvmGpuAddressSpaceHandle vaSpace1,
                                   uvmGpuAddressSpaceHandle vaSpace2,
                                   UvmGpuP2PCapsParams * p2pCapsParams);

/*******************************************************************************
    nvUvmInterfaceGetPmaObject

    This function will returns pointer to PMA object for the GPU whose UUID is 
    passed as an argument. This PMA object handle is required for invoking PMA 
    for allocate and free pages.

    Arguments:
        uuidMsb [IN]        - MSB part of the GPU UUID.
        uuidLsb [IN]        - LSB part of the GPU UUID.
        pPma [OUT]          - Pointer to PMA object

    Error codes:
        NV_ERR_NOT_SUPPORTED          - Operation not supported on broken FB
        NV_ERR_GENERIC:
          Unexpected error. We try hard to avoid returning this error
          code,because it is not very informative.
*/
NV_STATUS nvUvmInterfaceGetPmaObject(const NvProcessorUuid * gpuUUID, void **pPma);

// Mirrors pmaEvictPagesCb_t, see its documentation in pma.h.
typedef NV_STATUS (*uvmPmaEvictPagesCallback)(void *callbackData,
                                              NvU32 pageSize,
                                              NvU64 *pPages,
                                              NvU32 count,
                                              NvU64 physBegin,
                                              NvU64 physEnd);

// Mirrors pmaEvictRangeCb_t, see its documentation in pma.h.
typedef NV_STATUS (*uvmPmaEvictRangeCallback)(void *callbackData, NvU64 physBegin, NvU64 physEnd);

/*******************************************************************************
    nvUvmInterfacePmaRegisterEvictionCallbacks

    Simple wrapper for pmaRegisterEvictionCb(), see its documentation in pma.h.
*/
NV_STATUS nvUvmInterfacePmaRegisterEvictionCallbacks(void *pPma,
                                                     uvmPmaEvictPagesCallback evictPages,
                                                     uvmPmaEvictRangeCallback evictRange,
                                                     void *callbackData);

/******************************************************************************
    nvUvmInterfacePmaUnregisterEvictionCallbacks

    Simple wrapper for pmaUnregisterEvictionCb(), see its documentation in pma.h.
*/
void nvUvmInterfacePmaUnregisterEvictionCallbacks(void *pPma);

/*******************************************************************************
    nvUvmInterfacePmaAllocPages

    @brief Synchronous API for allocating pages from the PMA.
    PMA will decide which pma regions to allocate from based on the provided
    flags.  PMA will also initiate UVM evictions to make room for this
    allocation unless prohibited by PMA_FLAGS_DONT_EVICT.  UVM callers must pass
    this flag to avoid deadlock.  Only UVM may allocated unpinned memory from
    this API.

    For broadcast methods, PMA will guarantee the same physical frames are
    allocated on multiple GPUs, specified by the PMA objects passed in.

    If allocation is contiguous, only one page in pPages will be filled.
    Also, contiguous flag must be passed later to nvUvmInterfacePmaFreePages.

    Arguments:
        pPma[IN]             - Pointer to PMA object
        pageCount [IN]       - Number of pages required to be allocated.
        pageSize [IN]        - 64kb, 128kb or 2mb.  No other values are permissible.
        pPmaAllocOptions[IN] - Pointer to PMA allocation info structure.
        pPages[OUT]          - Array of pointers, containing the PA base 
                               address of each page.

    Error codes:
        NV_ERR_NO_MEMORY:
          Internal memory allocation failed.
        NV_ERR_GENERIC:
          Unexpected error. We try hard to avoid returning this error
          code,because it is not very informative.
*/
NV_STATUS nvUvmInterfacePmaAllocPages(void *pPma,
                                      NvLength pageCount,
                                      NvU32 pageSize,
                                      UvmPmaAllocationOptions *pPmaAllocOptions,
                                      NvU64 *pPages);

/*******************************************************************************
    nvUvmInterfacePmaPinPages

    This function will pin the physical memory allocated using PMA. The pages 
    passed as input must be unpinned else this function will return an error and
    rollback any change if any page is not previously marked "unpinned".

    Arguments:
        pPma[IN]             - Pointer to PMA object.
        pPages[IN]           - Array of pointers, containing the PA base 
                               address of each page to be pinned.
        pageCount [IN]       - Number of pages required to be pinned.
        pageSize [IN]        - Page size of each page to be pinned.
        flags [IN]           - UVM_PMA_CALLED_FROM_PMA_EVICTION if called from
                               PMA eviction, 0 otherwise.
    Error codes:
        NV_ERR_INVALID_ARGUMENT       - Invalid input arguments.
        NV_ERR_GENERIC                - Unexpected error. We try hard to avoid 
                                        returning this error code as is not very
                                        informative.
        NV_ERR_NOT_SUPPORTED          - Operation not supported on broken FB
*/
NV_STATUS nvUvmInterfacePmaPinPages(void *pPma,
                                    NvU64 *pPages,
                                    NvLength pageCount,
                                    NvU32 pageSize,
                                    NvU32 flags);

/*******************************************************************************
    nvUvmInterfacePmaUnpinPages

    This function will unpin the physical memory allocated using PMA. The pages 
    passed as input must be already pinned, else this function will return an 
    error and rollback any change if any page is not previously marked "pinned".
    Behaviour is undefined if any blacklisted pages are unpinned.

    Arguments:
        pPma[IN]             - Pointer to PMA object.
        pPages[IN]           - Array of pointers, containing the PA base 
                               address of each page to be unpinned.
        pageCount [IN]       - Number of pages required to be unpinned.
        pageSize [IN]        - Page size of each page to be unpinned.

    Error codes:
        NV_ERR_INVALID_ARGUMENT       - Invalid input arguments.
        NV_ERR_GENERIC                - Unexpected error. We try hard to avoid 
                                        returning this error code as is not very
                                        informative.
        NV_ERR_NOT_SUPPORTED          - Operation not supported on broken FB
*/
NV_STATUS nvUvmInterfacePmaUnpinPages(void *pPma,
                                      NvU64 *pPages,
                                      NvLength pageCount,
                                      NvU32 pageSize);

/*******************************************************************************
    nvUvmInterfaceMemoryFree

    Free up a GPU allocation
*/
void nvUvmInterfaceMemoryFree(uvmGpuAddressSpaceHandle vaSpace,
                              UvmGpuPointer gpuPointer);

/*******************************************************************************
    nvUvmInterfacePmaFreePages

    This function will free physical memory allocated using PMA.  It marks a list
    of pages as free. This operation is also used by RM to mark pages as "scrubbed"
    for the initial ECC sweep. This function does not fail.

    When allocation was contiguous, an appropriate flag needs to be passed.

    Arguments:
        pPma[IN]             - Pointer to PMA object
        pPages[IN]           - Array of pointers, containing the PA base 
                               address of each page.
        pageCount [IN]       - Number of pages required to be allocated.
        pageSize [IN]        - Page size of each page
        flags [IN]           - Flags with information about allocation type
                               with the same meaning as flags in options for
                               nvUvmInterfacePmaAllocPages. When called from PMA
                               eviction, UVM_PMA_CALLED_FROM_PMA_EVICTION needs
                               to be added to flags.
    Error codes:
        NV_ERR_INVALID_ARGUMENT  
        NV_ERR_NO_MEMORY              - Not enough physical memory to service
                                        allocation request with provided constraints
        NV_ERR_INSUFFICIENT_RESOURCES - Not enough available resources to satisfy allocation request
        NV_ERR_INVALID_OWNER          - Target memory not accessible by specified owner
        NV_ERR_NOT_SUPPORTED          - Operation not supported on broken FB
*/
void nvUvmInterfacePmaFreePages(void *pPma,
                                NvU64 *pPages,
                                NvLength pageCount,
                                NvU32 pageSize,
                                NvU32 flags);

/*******************************************************************************
    nvUvmInterfaceMemoryCpuMap

    This function creates a CPU mapping to the provided GPU address.
    If the address is not the same as what is returned by the Alloc
    function, then the function will map it from the address provided.
    This offset will be relative to the gpu offset obtained from the
    memory alloc functions.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceMemoryCpuMap(uvmGpuAddressSpaceHandle vaSpace,
                                     UvmGpuPointer gpuPointer,
                                     NvLength length, void **cpuPtr,
                                     NvU32 pageSize);

/*******************************************************************************
    uvmGpuMemoryCpuUnmap

    Unmaps the cpuPtr provided from the process virtual address space.
*/
void nvUvmInterfaceMemoryCpuUnMap(uvmGpuAddressSpaceHandle vaSpace,
                                  void *cpuPtr);

/*******************************************************************************
    nvUvmInterfaceChannelAllocate

    This function will allocate a channel

    UvmGpuChannelPointers: this structure will be filled out with channel
    get/put. The errorNotifier is filled out when the channel hits an RC error.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/

NV_STATUS nvUvmInterfaceChannelAllocate(uvmGpuAddressSpaceHandle  vaSpace,
                                        uvmGpuChannelHandle *channel,
                                        const UvmGpuChannelAllocParams *allocParams,
                                        UvmGpuChannelPointers * pointers);

void nvUvmInterfaceChannelDestroy(uvmGpuChannelHandle channel);

/*******************************************************************************
    nvUvmInterfaceCopyEngineAlloc

    copyEngineIndex corresponds to the indexing of the
    UvmGpuCaps::copyEngineCaps array. The possible values are
    [0, UVM_COPY_ENGINE_COUNT_MAX), but notably only the copy engines that have
    UvmGpuCopyEngineCaps::supported set to true can be allocated. On Volta+
    devices, this function computes the work submission token to be used in the
    Host channel submission doorbell.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceCopyEngineAlloc(uvmGpuChannelHandle channel,
                                        unsigned copyEngineIndex,
                                        uvmGpuCopyEngineHandle *copyEngine,
                                        UvmGpuChannelPointers *pointers);

/*******************************************************************************
    nvUvmInterfaceQueryCaps

    Return capabilities for the provided GPU.
    If GPU does not exist, an error will be returned.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
*/
NV_STATUS nvUvmInterfaceQueryCaps(uvmGpuAddressSpaceHandle vaSpace,
                                  UvmGpuCaps * caps);

/*******************************************************************************
    nvUvmInterfaceGetGpuInfo

    Return various gpu info, refer to the UvmGpuInfo struct for details.
    If no gpu matching the uuid is found, an error will be returned.

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INSUFFICIENT_RESOURCES
 */
NV_STATUS nvUvmInterfaceGetGpuInfo(const NvProcessorUuid *gpuUuid, UvmGpuInfo *pGpuInfo);

/*******************************************************************************
    nvUvmInterfaceServiceDeviceInterruptsRM

    Tells RM to service all pending interrupts. This is helpful in ECC error
    conditions when ECC error interrupt is set & error can be determined only
    after ECC notifier will be set or reset.

    Error codes:
      NV_ERR_GENERIC
      UVM_INVALID_ARGUMENTS
*/
NV_STATUS nvUvmInterfaceServiceDeviceInterruptsRM(uvmGpuAddressSpaceHandle vaSpace);

/*******************************************************************************
    nvUvmInterfaceSetPageDirectory
    Sets pageDirectory in the provided location. Also moves the existing PDE to
    the provided pageDirectory.

    RM will propagate the update to all channels using the provided VA space.
    All channels must be idle when this call is made.

    Arguments:
      vaSpace[IN}         - VASpace Object
      physAddress[IN]     - Physical address of new page directory
      numEntries[IN]      - Number of entries including previous PDE which will be copied
      bVidMemAperture[IN] - If set pageDirectory will reside in VidMem aperture else sysmem

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceSetPageDirectory(uvmGpuAddressSpaceHandle vaSpace,
                                         NvU64 physAddress, unsigned numEntries,
                                         NvBool bVidMemAperture);

/*******************************************************************************
    nvUvmInterfaceUnsetPageDirectory
    Unsets/Restores pageDirectory to RM's defined location.

    Arguments:
      vaSpace[IN}         - VASpace Object

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceUnsetPageDirectory(uvmGpuAddressSpaceHandle vaSpace);

/*******************************************************************************
    nvUvmInterfaceDupAllocation

    Duplicate an allocation represented by a physical handle.
    Duplication means: the physical handle will be duplicated from src vaspace 
    to dst vaspace and a new mapping will be created in the dst vaspace.
 
    Arguments:
        hPhysHandle[IN]          - Handle representing the phys allocation.
        srcVaspace[IN]           - Pointer to source vaSpace object
        srcAddress[IN]           - Offset of the gpu mapping in source vaspace.
        dstVaspace[IN]           - Pointer to destination vaSpace object
        dstAddress[OUT]          - Offset of the gpu mapping in destination 
                                   vaspace.
        bPhysHandleValid[IN]     - Whether the client has provided the handle
                                   for source allocation.
                                   If True; hPhysHandle will be used.
                                   Else; ops will find out the handle using
                                   srcVaspace and srcAddress

    Error codes:
      NV_ERROR
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceDupAllocation(NvHandle hPhysHandle,
                                      uvmGpuAddressSpaceHandle srcVaspace,
                                      NvU64 srcAddress,
                                      uvmGpuAddressSpaceHandle dstVaspace,
                                      NvU64 *dstAddress,
                                      NvBool bPhysHandleValid);

/*******************************************************************************
    nvUvmInterfaceDupMemory

    Duplicates a physical memory allocation. If requested, provides information
    about the allocation.
 
    Arguments:
        vaSpace[IN]                     - VA space linked to a client and a device under which
                                          the phys memory needs to be duped.
        hClient[IN]                     - Client owning the memory.
        hPhysMemory[IN]                 - Phys memory which is to be duped.
        hDupedHandle[OUT]               - Handle of the duped memory object.
        pGpuMemoryInfo[OUT]             - see nv_uvm_types.h for more information.
                                          This parameter can be NULL. (optional)
    Error codes:
      NV_ERR_INVALID_ARGUMENT   - If the parameter/s is invalid.
      NV_ERR_NOT_SUPPORTED      - If the allocation is not a physical allocation.
      NV_ERR_OBJECT_NOT_FOUND   - If the allocation is not found in under the provided client.
*/
NV_STATUS nvUvmInterfaceDupMemory(uvmGpuAddressSpaceHandle vaSpace,
                                  NvHandle hClient,
                                  NvHandle hPhysMemory,
                                  NvHandle *hDupMemory,
                                  UvmGpuMemoryInfo *pGpuMemoryInfo);

/*******************************************************************************
    nvUvmInterfaceFreeDupedAllocation

    Free the lallocation represented by the physical handle used to create the
    duped allocation.
 
    Arguments:
        vaspace[IN]              - Pointer to source vaSpace object
        hPhysHandle[IN]          - Handle representing the phys allocation.
        
    Error codes:
      NV_ERROR
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceFreeDupedHandle(uvmGpuAddressSpaceHandle vaspace,
                                        NvHandle hPhysHandle);

/*******************************************************************************
    nvUvmInterfaceGetFbInfo

    Gets FB information from RM.
 
    Arguments:
        vaspace[IN]       - Pointer to source vaSpace object
        fbInfo [OUT]      - Pointer to FbInfo structure which contains
                            reservedHeapSize & heapSize
    Error codes:
      NV_ERROR
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceGetFbInfo(uvmGpuAddressSpaceHandle vaSpace,
                                  UvmGpuFbInfo * fbInfo);

/*******************************************************************************
    nvUvmInterfaceOwnPageFaultIntr

    This function transfers ownership of the replayable page fault interrupt,
    between RM and UVM, for a particular GPU.

    bOwnInterrupts == NV_TRUE: UVM is taking ownership from the RM. This causes
    the following: RM will not service, enable or disable this interrupt and it
    is up to the UVM driver to handle this interrupt. In this case, replayable
    page fault interrupts are disabled by this function, before it returns.

    bOwnInterrupts == NV_FALSE: UVM is returning ownership to the RM: in this
    case, replayable page fault interrupts MUST BE DISABLED BEFORE CALLING this
    function.

    The cases above both result in transferring ownership of a GPU that has its
    replayable page fault interrupts disabled. Doing otherwise would make it
    very difficult to control which driver handles any interrupts that build up
    during the hand-off.

    The calling pattern should look like this:

    UVM setting up a new GPU for operation:
        UVM GPU LOCK
           nvUvmInterfaceOwnPageFaultIntr(..., NV_TRUE)
        UVM GPU UNLOCK

        Enable replayable page faults for that GPU

    UVM tearing down a GPU:

        Disable replayable page faults for that GPU

        UVM GPU GPU LOCK
           nvUvmInterfaceOwnPageFaultIntr(..., NV_FALSE)
        UVM GPU UNLOCK

    Arguments:
        gpuUuid[IN]          - UUID of the GPU to operate on
        bOwnInterrupts       - Set to NV_TRUE for UVM to take ownership of the
                               replayable page fault interrupts. Set to NV_FALSE
                               to return ownership of the page fault interrupts
                               to RM.
    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceOwnPageFaultIntr(const NvProcessorUuid *gpuUuid, NvBool bOwnInterrupts);
/*******************************************************************************
    nvUvmInterfaceInitFaultInfo

    This function obtains fault buffer address, size and a few register mappings
    for replayable faults, and creates a shadow buffer to store non-replayable
    faults if the GPU supports it.

    Arguments:
        vaspace[IN]       - Pointer to vaSpace object associated with the gpu
        pFaultInfo[OUT]   - information provided by RM for fault handling

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_NO_MEMORY
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceInitFaultInfo(uvmGpuAddressSpaceHandle vaSpace,
                                      UvmGpuFaultInfo *pFaultInfo);

/*******************************************************************************
    nvUvmInterfaceDestroyFaultInfo

    This function obtains destroys unmaps the fault buffer and clears faultInfo
    for replayable faults, and frees the shadow buffer for non-replayable faults.

    Arguments:
        vaspace[IN]       - Pointer to vaSpace object associated with the gpu
        pFaultInfo[OUT]   - information provided by RM for fault handling

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceDestroyFaultInfo(uvmGpuAddressSpaceHandle vaSpace,
                                         UvmGpuFaultInfo *pFaultInfo);

/*******************************************************************************
    nvUvmInterfaceHasPendingNonReplayableFaults

    This function tells whether there are pending non-replayable faults in the
    client shadow fault buffer ready to be consumed.

    NOTES:
    - This function uses a pre-allocated stack per GPU (stored in the
    UvmGpuFaultInfo object) for calls related to non-replayable faults from the
    top half.
    - Concurrent calls to this function using the same pFaultInfo are not
    thread-safe due to pre-allocated stack. Therefore, locking is the caller's
    responsibility.
    - This function DOES NOT acquire the RM API or GPU locks. That is because
    it is called during fault servicing, which could produce deadlocks.

    Arguments:
        pFaultInfo[IN]        - information provided by RM for fault handling.
                                Contains a pointer to the shadow fault buffer
        hasPendingFaults[OUT] - return value that tells if there are
                                non-replayable faults ready to be consumed by
                                the client

    Error codes:
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceHasPendingNonReplayableFaults(UvmGpuFaultInfo *pFaultInfo,
                                                      NvBool *hasPendingFaults);

/*******************************************************************************
    nvUvmInterfaceGetNonReplayableFaults

    This function consumes all the non-replayable fault packets in the client
    shadow fault buffer and copies them to the given buffer. It also returns the
    number of faults that have been copied

    NOTES:
    - This function uses a pre-allocated stack per GPU (stored in the
    UvmGpuFaultInfo object) for calls from the bottom half that handles
    non-replayable faults.
    - See nvUvmInterfaceHasPendingNonReplayableFaults for the implications of
    using a shared stack.
    - This function DOES NOT acquire the RM API or GPU locks. That is because
    it is called during fault servicing, which could produce deadlocks.

    Arguments:
        pFaultInfo[IN]    - information provided by RM for fault handling.
                            Contains a pointer to the shadow fault buffer
        pFaultBuffer[OUT] - buffer provided by the client where fault buffers
                            are copied when they are popped out of the shadow
                            fault buffer (which is a circular queue).
        numFaults[OUT]    - return value that tells the number of faults copied
                            to the client's buffer

    Error codes:
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceGetNonReplayableFaults(UvmGpuFaultInfo *pFaultInfo,
                                               void *pFaultBuffer, NvU32 *numFaults);

/*******************************************************************************
    nvUvmInterfaceInitAccessCntrInfo

    This function obtains access counter buffer address, size and a few register mappings
 
    Arguments:
        vaspace[IN]          - Pointer to vaSpace object associated with the gpu
        pAccessCntrInfo[OUT] - Information provided by RM for access counter handling

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceInitAccessCntrInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo);

/*******************************************************************************
    nvUvmInterfaceDestroyAccessCntrInfo

    This function obtains, destroys, unmaps the access counter buffer and clears accessCntrInfo
 
    Arguments:
        vaspace[IN]         - Pointer to vaSpace object associated with the gpu
        pAccessCntrInfo[IN] - Information provided by RM for access counter handling

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceDestroyAccessCntrInfo(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo);

/*******************************************************************************
    nvUvmInterfaceEnableAccessCntr

    This function enables access counters using the given configuration

    Arguments:
        vaspace[IN]           - Pointer to vaSpace object associated with the gpu
        pAccessCntrInfo[IN]   - Pointer to structure filled out by nvUvmInterfaceInitAccessCntrInfo
        pAccessCntrConfig[IN] - Configuration for access counters

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceEnableAccessCntr(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo,
    UvmGpuAccessCntrConfig *pAccessCntrConfig);

/*******************************************************************************
    nvUvmInterfaceDisableAccessCntr

    This function disables acccess counters

    Arguments:
        vaspace[IN]          - Pointer to vaSpace object associated with the gpu
        pAccessCntrInfo[IN]  - Pointer to structure filled out by nvUvmInterfaceInitAccessCntrInfo

    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceDisableAccessCntr(uvmGpuAddressSpaceHandle vaSpace,
    UvmGpuAccessCntrInfo *pAccessCntrInfo);

/*******************************************************************************
    nvUvmInterfaceOwnAccessCntrIntr

    This function transfers ownership of the access counter notification
    interrupt, between RM and UVM, for a particular GPU.

    bOwnInterrupts == NV_TRUE: UVM is taking ownership from the RM. This causes
    the following: RM will not service, enable or disable this interrupt and it
    is up to the UVM driver to handle this interrupt. In this case, access
    counter notificaion interrupts are disabled by this function, before it
    returns.

    bOwnInterrupts == NV_FALSE: UVM is returning ownership to the RM: in this
    case, access counter notification interrupts MUST BE DISABLED BEFORE CALLING
    this function.

    The cases above both result in transferring ownership of a GPU that has its
    access counter notification interrupts disabled. Doing otherwise would make
    it very difficult to control which driver handles any interrupts that build
    up during the hand-off.

    The calling pattern should look like this:

    UVM setting up a new GPU for operation:
        UVM GPU LOCK
           nvUvmInterfaceOwnAccessCntrIntr(..., NV_TRUE)
        UVM GPU UNLOCK

        Enable access counter notifications for that GPU

    UVM tearing down a GPU:

        Disable access counter notifications for that GPU

        UVM GPU GPU LOCK
           nvUvmInterfaceOwnAccessCntrIntr(..., NV_FALSE)
        UVM GPU UNLOCK

    Arguments:
        session[IN]         - Session handle created for this gpu
        pAccessCntrInfo[IN] - Pointer to structure filled out by nvUvmInterfaceInitAccessCntrInfo
        bOwnInterrupts      - Set to NV_TRUE for UVM to take ownership of the
                              access counter notification interrupts. Set to
                              NV_FALSE to return ownership of the access
                              counter notification interrupts to RM.
    Error codes:
      NV_ERR_GENERIC
      NV_ERR_INVALID_ARGUMENT
*/
NV_STATUS nvUvmInterfaceOwnAccessCntrIntr(uvmGpuSessionHandle session,
                                          UvmGpuAccessCntrInfo *pAccessCntrInfo,
                                          NvBool bOwnInterrupts);

//
// Called by the UVM driver to register operations with RM. Only one set of
// callbacks can be registered by any driver at a time. If another set of
// callbacks was already registered, NV_ERR_IN_USE is returned.
//
NV_STATUS nvUvmInterfaceRegisterUvmCallbacks(struct UvmOpsUvmEvents *importedUvmOps);

//
// Counterpart to nvUvmInterfaceRegisterUvmCallbacks. This must only be called
// if nvUvmInterfaceRegisterUvmCallbacks returned NV_OK.
//
// Upon return, the caller is guaranteed that any outstanding callbacks are done
// and no new ones will be invoked.
//
void nvUvmInterfaceDeRegisterUvmOps(void);

/*******************************************************************************
    nvUvmInterfaceP2pObjectCreate

    This API creates an NV50_P2P object for the gpus with the given address
    space handles and returns the handle to the object.

    Arguments:
        vaSpace1[IN]        - first gpu address space handle
        vaSpace1[IN]        - second gpu address space handle
        hP2pObject[OUT]     - handle to the created P2p object.

    Error codes:
      NV_ERR_INVALID_ARGUMENT
      NV_ERR_OBJECT_NOT_FOUND : If device object associated with the uuids aren't found.
*/
NV_STATUS nvUvmInterfaceP2pObjectCreate(uvmGpuAddressSpaceHandle vaSpace1,
                                        uvmGpuAddressSpaceHandle vaSpace2,
                                        NvHandle *hP2pObject);

/*******************************************************************************
    nvUvmInterfaceP2pObjectDestroy

    This API destroys the NV50_P2P associated with the passed handle.

    Arguments:
        session[IN]        - Session handle.
        hP2pObject[IN]     - handle to an P2p object.

    Error codes: NONE
*/
void nvUvmInterfaceP2pObjectDestroy(uvmGpuSessionHandle session,
                                    NvHandle hP2pObject);

/*******************************************************************************
    nvUvmInterfaceGetExternalAllocPtes

    The interface builds the RM PTEs using the provided input parameters.

    Arguments:
        vaSpace[IN]                     -  vaSpace handle.
        hMemory[IN]                     -  Memory handle.
        offset [IN]                     -  Offset from the beginning of the allocation
                                           where PTE mappings should begin.
                                           Should be aligned with pagesize associated
                                           with the allocation.
        size [IN]                       -  Length of the allocation for which PTEs
                                           should be built.
                                           Should be aligned with pagesize associated
                                           with the allocation.
                                           size = 0 will be interpreted as the total size
                                           of the allocation.
        gpuExternalMappingInfo[IN/OUT]  -  See nv_uvm_types.h for more information.

   Error codes:
        NV_ERR_INVALID_ARGUMENT         - Invalid parameter/s is passed.
        NV_ERR_INVALID_OBJECT_HANDLE    - Invalid memory handle is passed.
        NV_ERR_NOT_SUPPORTED            - Functionality is not supported (see comments in nv_gpu_ops.c)
        NV_ERR_INVALID_BASE             - offset is beyond the allocation size
        NV_ERR_INVALID_LIMIT            - (offset + size) is beyond the allocation size.
        NV_ERR_BUFFER_TOO_SMALL         - gpuExternalMappingInfo.pteBufferSize is insufficient to
                                          store single PTE.
*/
NV_STATUS nvUvmInterfaceGetExternalAllocPtes(uvmGpuAddressSpaceHandle vaSpace,
                                             NvHandle hMemory,
                                             NvU64 offset,
                                             NvU64 size,
                                             UvmGpuExternalMappingInfo *gpuExternalMappingInfo);

/*******************************************************************************
    nvUvmInterfaceRetainChannel

    Validates and returns information about the user's channel. The state is
    refcounted and must be released by calling nvUvmInterfaceReleaseChannel.

    Arguments:
        vaSpace[IN]               - vaSpace handle.
        hClient[IN]               - Client handle
        hChannel[IN]              - Channel handle
        retainedChannel[OUT]      - Opaque pointer to use to refer to this
                                    channel in other nvUvmInterface APIs.
        channelInstanceInfo[OUT]  - Channel instance information to be filled out.
                                    See nv_uvm_types.h for details.

    Error codes:
        NV_ERR_INVALID_ARGUMENT : If the parameter/s are invalid.
        NV_ERR_OBJECT_NOT_FOUND : If the object associated with the handle isn't found.
        NV_ERR_INVALID_CHANNEL : If the channel verification fails.
 */
NV_STATUS nvUvmInterfaceRetainChannel(uvmGpuAddressSpaceHandle vaSpace,
                                      NvHandle hClient,
                                      NvHandle hChannel,
                                      void **retainedChannel,
                                      UvmGpuChannelInstanceInfo *channelInstanceInfo);

/*******************************************************************************
    nvUvmInterfaceRetainChannelResources

    Returns information about channel resources (local CTX buffers + global CTX buffers).
    Also, it refcounts the memory descriptors associated with the resources.

    Arguments:
        retainedChannel[IN]       - Channel pointer returned by nvUvmInterfaceRetainChannel
        channelResourceInfo[OUT]  - This should be a buffer which can fit at least
                                    UvmGpuChannelInstanceInfo::resourceCount UvmGpuChannelResourceInfo
                                    entries. The channel resource information will be written in this
                                    buffer. See nv_uvm_types.h for details.

    Error codes:
        NV_ERR_INVALID_ARGUMENT : If the parameter/s are invalid.
        NV_ERR_OBJECT_NOT_FOUND : If the object associated with the handle isn't found.
        NV_ERR_INSUFFICIENT_RESOURCES : If no memory available to store the resource information.
 */
NV_STATUS nvUvmInterfaceRetainChannelResources(void *retainedChannel,
                                               UvmGpuChannelResourceInfo *channelResourceInfo);

/*******************************************************************************
    nvUvmInterfaceBindChannelResources

    Associates the mapping address of the channel resources (VAs) provided by the
    caller with the channel.

    Arguments:
        retainedChannel[IN]           - Channel pointer returned by nvUvmInterfaceRetainChannel
        channelResourceBindParams[IN] - Buffer of initialized UvmGpuChannelInstanceInfo::resourceCount
                                        entries. See nv_uvm_types.h for details.

    Error codes:
        NV_ERR_INVALID_ARGUMENT : If the parameter/s are invalid.
        NV_ERR_OBJECT_NOT_FOUND : If the object associated with the handle aren't found.
        NV_ERR_INSUFFICIENT_RESOURCES : If no memory available to store the resource information.
 */
NV_STATUS nvUvmInterfaceBindChannelResources(void *retainedChannel,
                                             UvmGpuChannelResourceBindParams *channelResourceBindParams);

/*******************************************************************************
    nvUvmInterfaceReleaseChannel

    Releases state retained by nvUvmInterfaceRetainChannel. If
    nvUvmInterfaceBindChannelResources has been called, you must release the
    resources by calling nvUvmInterfaceReleaseChannelResources before calling
    this function.
 */
void nvUvmInterfaceReleaseChannel(void *retainedChannel);

/*******************************************************************************
    nvUvmInterfaceReleaseChannelResources

    Release refcounts on the memory descriptors associated with the resources.
    Also, frees the memory descriptors if refcount reaches zero.

    Arguments:
        descriptors[IN]         - The call expects the input buffer of size(NvP64) * descriptorCount initialized
                                  with the descriptors returned by nvUvmInterfaceRetainChannelResources as input.
        descriptorCount[IN]     - The count of descriptors to be released.
 */
void nvUvmInterfaceReleaseChannelResources(NvP64 *resourceDescriptors, NvU32 descriptorCount);

/*******************************************************************************
    nvUvmInterfaceStopChannel

    Idles the channel and takes it off the runlist.

    Arguments:
        retainedChannel[IN]           - Channel pointer returned by nvUvmInterfaceRetainChannel
        bImmediate[IN]                - If true, kill the channel without attempting to wait for it to go idle.
*/
void nvUvmInterfaceStopChannel(void *retainedChannel, NvBool bImmediate);

/*******************************************************************************
    nvUvmInterfaceGetChannelResourcePtes

    The interface builds the RM PTEs using the provided input parameters.

    Arguments:
        vaSpace[IN]                     -  vaSpace handle.
        resourceDescriptor[IN]          -  The channel resource descriptor returned by returned by
                                           nvUvmInterfaceRetainChannelResources.
        offset[IN]                      -  Offset from the beginning of the allocation
                                           where PTE mappings should begin.
                                           Should be aligned with pagesize associated
                                           with the allocation.
        size[IN]                        -  Length of the allocation for which PTEs
                                           should be built.
                                           Should be aligned with pagesize associated
                                           with the allocation.
                                           size = 0 will be interpreted as the total size
                                           of the allocation.
        gpuExternalMappingInfo[IN/OUT]  -  See nv_uvm_types.h for more information.

   Error codes:
        NV_ERR_INVALID_ARGUMENT         - Invalid parameter/s is passed.
        NV_ERR_INVALID_OBJECT_HANDLE    - Invalid memory handle is passed.
        NV_ERR_NOT_SUPPORTED            - Functionality is not supported.
        NV_ERR_INVALID_BASE             - offset is beyond the allocation size
        NV_ERR_INVALID_LIMIT            - (offset + size) is beyond the allocation size.
        NV_ERR_BUFFER_TOO_SMALL         - gpuExternalMappingInfo.pteBufferSize is insufficient to
                                          store single PTE.
*/
NV_STATUS nvUvmInterfaceGetChannelResourcePtes(uvmGpuAddressSpaceHandle vaSpace,
                                               NvP64 resourceDescriptor,
                                               NvU64 offset,
                                               NvU64 size,
                                               UvmGpuExternalMappingInfo *externalMappingInfo);

#endif // _NV_UVM_INTERFACE_H_
