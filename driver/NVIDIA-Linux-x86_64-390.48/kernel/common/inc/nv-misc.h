/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1993-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_MISC_H_
#define _NV_MISC_H_

#include "nvtypes.h"
#include "nvstatus.h"

#ifndef BOOL
#define BOOL            NvS32
#endif
#ifndef TRUE
#define TRUE            1L
#endif
#ifndef FALSE
#define FALSE           0L
#endif
#ifndef NULL
#define NULL            0L
#endif

/*
 * Device state and configuration information
 */

typedef void *PHWINFO;

#endif /* _NV_MISC_H_ */
