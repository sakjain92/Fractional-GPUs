/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2013 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _NV_FRONTEND_H_
#define _NV_FRONTEND_H_

#include "nvtypes.h"
#include "nv-misc.h"
#include "nv-linux.h"
#include "nv-register-module.h"

#define NV_MAX_MODULE_INSTANCES                 8

#define NV_FRONTEND_MINOR_NUMBER(x)             minor((x)->i_rdev)

#define NV_FRONTEND_CONTROL_DEVICE_MINOR_MAX    255
#define NV_FRONTEND_CONTROL_DEVICE_MINOR_MIN    (NV_FRONTEND_CONTROL_DEVICE_MINOR_MAX - \
                                                 NV_MAX_MODULE_INSTANCES)

#define NV_FRONTEND_IS_CONTROL_DEVICE(x)        ((x <= NV_FRONTEND_CONTROL_DEVICE_MINOR_MAX) && \
                                                 (x > NV_FRONTEND_CONTROL_DEVICE_MINOR_MIN))

int nvidia_frontend_add_device(nvidia_module_t *, nv_linux_state_t *);
int nvidia_frontend_remove_device(nvidia_module_t *, nv_linux_state_t *);

#endif
