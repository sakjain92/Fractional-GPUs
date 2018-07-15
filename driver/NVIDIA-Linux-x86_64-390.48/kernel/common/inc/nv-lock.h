/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_LOCK_H_
#define _NV_LOCK_H_

#include "conftest.h"

#include <linux/spinlock.h>

#if defined(NV_LINUX_SEMAPHORE_H_PRESENT)
#include <linux/semaphore.h>
#else
#include <asm/semaphore.h>
#endif

typedef spinlock_t                nv_spinlock_t;
#define NV_SPIN_LOCK_INIT(lock)   spin_lock_init(lock)
#define NV_SPIN_LOCK_IRQ(lock)    spin_lock_irq(lock)
#define NV_SPIN_UNLOCK_IRQ(lock)  spin_unlock_irq(lock)
#define NV_SPIN_LOCK_IRQSAVE(lock,flags) spin_lock_irqsave(lock,flags)
#define NV_SPIN_UNLOCK_IRQRESTORE(lock,flags) spin_unlock_irqrestore(lock,flags)
#define NV_SPIN_LOCK(lock)        spin_lock(lock)
#define NV_SPIN_UNLOCK(lock)      spin_unlock(lock)
#define NV_SPIN_UNLOCK_WAIT(lock) spin_unlock_wait(lock)

#if defined(NV_CONFIG_PREEMPT_RT)
#define NV_INIT_SEMA(sema, val) sema_init(sema,val)
#else
#if !defined(__SEMAPHORE_INITIALIZER) && defined(__COMPAT_SEMAPHORE_INITIALIZER)
#define __SEMAPHORE_INITIALIZER __COMPAT_SEMAPHORE_INITIALIZER
#endif
#define NV_INIT_SEMA(sema, val)                    \
    {                                              \
        struct semaphore __sema =                  \
            __SEMAPHORE_INITIALIZER(*(sema), val); \
        *(sema) = __sema;                          \
    }
#endif
#define NV_INIT_MUTEX(mutex) NV_INIT_SEMA(mutex, 1)

#endif  /* _NV_LOCK_H_ */
