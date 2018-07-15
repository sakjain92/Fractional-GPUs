/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2011 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#define  __NO_VERSION__
#include "nv-misc.h"

#include "os-interface.h"
#include "nv-linux.h"
#include "nv-frontend.h"
#include "nv-instance.h"

int nv_register_chrdev(void *param)
{
    nvidia_module_t *module = (nvidia_module_t *)param;

    module->instance = nv_module_instance;

    return (nvidia_register_module(module));
}

void nv_unregister_chrdev(void *param)
{
    nvidia_module_t *module = (nvidia_module_t *)param;

    nvidia_unregister_module(module);
}
