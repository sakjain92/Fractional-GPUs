/*******************************************************************************
    Copyright (c) 2015-2016 NVidia Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/


#ifndef _NVLINK_COMMON_H_
#define _NVLINK_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvtypes.h"
#include "nvCpuUuid.h"

#ifndef NULL
#define NULL ((void *)0)
#endif

#define NVLINK_FREE(x) nvlink_free((void *)x)

#if !defined(NV_WINDOWS)
    #if defined(NVCPU_X86)
        #define NVLINK_API_CALL __attribute__((regparm(0)))
    #else
        #define NVLINK_API_CALL
    #endif
#else
    #define NVLINK_API_CALL
#endif

typedef int                 NvlStatus;

// nvlink pci bar information
struct nvlink_pci_bar_info
{
    NvU64  busAddress;
    NvU64  baseAddr;
    NvU64  barSize;
    NvU32  offset;
    void  *pBar;
};

#define MAX_NVLINK_BARS    2

// nvlink pci information
struct nvlink_pci_info
{
    NvU16   domain;
    NvU8    bus;
    NvU8    device;
    NvU8    function;
    NvU32   pciDeviceId;
    NvU32   irq;
    NvBool  intHooked;
    struct  nvlink_pci_bar_info bars[MAX_NVLINK_BARS];
};

struct nvlink_conn_info
{
    NvU16  domain;
    NvU16  bus;
    NvU16  device;
    NvU16  function;
    NvU32  pciDeviceId;
    NvU8   devUuid[NV_UUID_LEN];
    NvU64  deviceType;
    NvU32  linkNumber;
    NvBool bConnected;
};

struct nvlink_ioctrl_params
{
    NvU32       cmd;
    void        *buf;
    NvU32       size;
};

// Typedefs
typedef struct nvlink_pci_bar_info nvlink_pci_bar_info;
typedef struct nvlink_pci_info nvlink_pci_info;
typedef struct nvlink_conn_info nvlink_conn_info;
typedef struct nvlink_ioctrl_params nvlink_ioctrl_params;


// Memory management functions
void *      NVLINK_API_CALL nvlink_malloc(NvLength);
void        NVLINK_API_CALL nvlink_free(void *);
void *      NVLINK_API_CALL nvlink_memset(void *, int, NvLength);
void *      NVLINK_API_CALL nvlink_memcpy(void *, void *, NvLength);
int         NVLINK_API_CALL nvlink_memRd32(const volatile void *);
void        NVLINK_API_CALL nvlink_memWr32(volatile void *, unsigned int);
int         NVLINK_API_CALL nvlink_memRd64(const volatile void *);
void        NVLINK_API_CALL nvlink_memWr64(volatile void *, unsigned long long);

// String management functions
char *      NVLINK_API_CALL nvlink_strcpy(char *, const char *);
NvLength    NVLINK_API_CALL nvlink_strlen(const char *);
int         NVLINK_API_CALL nvlink_strcmp(const char *, const char *);
int         NVLINK_API_CALL nvlink_snprintf(char *, NvLength, const char *, ...);

// Locking support functions
void *      NVLINK_API_CALL nvlink_allocLock(void);
void        NVLINK_API_CALL nvlink_acquireLock(void *);
NvBool      NVLINK_API_CALL nvlink_isLockOwner(void *);
void        NVLINK_API_CALL nvlink_releaseLock(void *);
void        NVLINK_API_CALL nvlink_freeLock(void *);

// Miscellaneous functions
void        NVLINK_API_CALL nvlink_assert(int expression);
void        NVLINK_API_CALL nvlink_sleep(unsigned int ms);
void        NVLINK_API_CALL nvlink_print(const char *, int, const char *, int, const char *, ...);

#define NVLINK_DBG_LEVEL_INFO       0x0
#define NVLINK_DBG_LEVEL_SETUP      0x1
#define NVLINK_DBG_LEVEL_USERERRORS 0x2
#define NVLINK_DBG_LEVEL_WARNINGS   0x3
#define NVLINK_DBG_LEVEL_ERRORS     0x4

#define NVLINK_DBG_WHERE       __FILE__, __LINE__, __FUNCTION__
#define NVLINK_DBG_INFO        NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_INFO
#define NVLINK_DBG_SETUP       NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_SETUP
#define NVLINK_DBG_USERERRORS  NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_USERERRORS
#define NVLINK_DBG_WARNINGS    NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_WARNINGS
#define NVLINK_DBG_ERRORS      NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_ERRORS

#ifdef __cplusplus
}
#endif

#endif //_NVLINK_COMMON_H_
