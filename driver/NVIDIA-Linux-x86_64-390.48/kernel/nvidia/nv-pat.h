/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _NV_PAT_H_
#define _NV_PAT_H_

#include "nv-linux.h"


#if defined(NV_ENABLE_PAT_SUPPORT)
extern int nv_init_pat_support(nvidia_stack_t *sp);
extern void nv_teardown_pat_support(void);
extern int nv_enable_pat_support(void);
extern void nv_disable_pat_support(void);
#else
static inline int nv_init_pat_support(nvidia_stack_t *sp)
{
    (void)sp;
    return 0;
}

static inline void nv_teardown_pat_support(void)
{
    return;
}

static inline int nv_enable_pat_support(void)
{
    return 1;
}

static inline void nv_disable_pat_support(void)
{
    return;
}
#endif

#endif /* _NV_PAT_H_ */
