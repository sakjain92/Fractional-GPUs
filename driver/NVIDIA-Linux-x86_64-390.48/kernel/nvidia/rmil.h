/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _RMIL_H_
#define _RMIL_H_

int        NV_API_CALL  rm_gvi_isr                  (nvidia_stack_t *, nv_state_t *, NvU32 *);
void       NV_API_CALL  rm_gvi_bh                   (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_gvi_attach_device        (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_gvi_detach_device        (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_gvi_init_private_state   (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_init_gvi_device          (nvidia_stack_t *, nv_state_t *);
NvU32      NV_API_CALL  rm_shutdown_gvi_device      (nvidia_stack_t *, nv_state_t *);
NvU32      NV_API_CALL  rm_gvi_suspend              (nvidia_stack_t *, nv_state_t *);
NvU32      NV_API_CALL  rm_gvi_resume               (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_gvi_free_private_state   (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_gvi_get_device_name      (nvidia_stack_t *, nv_state_t *, NvU32, NvU32, NvU8 *);
NV_STATUS  NV_API_CALL  rm_gvi_get_firmware_version (nvidia_stack_t *, nv_state_t *, NvU32 *, NvU32 *, NvU32 *);

#endif // _RMIL_H_
