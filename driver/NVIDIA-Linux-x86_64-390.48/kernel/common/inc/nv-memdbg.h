/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NVMEMDBG_H_
#define _NVMEMDBG_H_

#include <nvtypes.h>

void nv_memdbg_init(void);
void nv_memdbg_add(void *addr, NvU64 size, const char *file, int line);
void nv_memdbg_remove(void *addr, NvU64 size, const char *file, int line);
void nv_memdbg_exit(void);

#if defined(NV_MEM_LOGGER)

#define NV_MEMDBG_ADD(ptr, size) \
    nv_memdbg_add(ptr, size, __FILE__, __LINE__)

#define NV_MEMDBG_REMOVE(ptr, size) \
    nv_memdbg_remove(ptr, size, __FILE__, __LINE__)

#else

#define NV_MEMDBG_ADD(ptr, size)
#define NV_MEMDBG_REMOVE(ptr, size)

#endif /* NV_MEM_LOGGER */

#endif /* _NVMEMDBG_H_ */
