/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

/*!
 * Define the entry points which the NVKMS kernel interface layer
 * provides to core NVKMS.
 */

#if !defined(_NVIDIA_MODESET_OS_INTERFACE_H_)
#define _NVIDIA_MODESET_OS_INTERFACE_H_

#include <stddef.h>  /* size_t */
#include <stdarg.h>  /* va_list */

#include "nvtypes.h" /* NvU8 */

#include "nvkms.h"   /* NVKMS_API_CALL */

void   NVKMS_API_CALL nvkms_call_rm    (void *ops);
void*  NVKMS_API_CALL nvkms_alloc      (size_t size,
                                        NvBool zero);
void   NVKMS_API_CALL nvkms_free       (void *ptr,
                                        size_t size);
void*  NVKMS_API_CALL nvkms_memset     (void *ptr,
                                        NvU8 c,
                                        size_t size);
void*  NVKMS_API_CALL nvkms_memcpy     (void *dest,
                                        const void *src,
                                        size_t n);
void*  NVKMS_API_CALL nvkms_memmove    (void *dest,
                                        const void *src,
                                        size_t n);
int    NVKMS_API_CALL nvkms_memcmp     (const void *s1,
                                        const void *s2,
                                        size_t n);
size_t NVKMS_API_CALL nvkms_strlen     (const char *s);
int    NVKMS_API_CALL nvkms_strcmp     (const char *s1,
                                        const char *s2);
char*  NVKMS_API_CALL nvkms_strncpy    (char *dest,
                                        const char *src,
                                        size_t n);
void   NVKMS_API_CALL nvkms_usleep     (NvU64 usec);
NvU64  NVKMS_API_CALL nvkms_get_usec   (void);
int    NVKMS_API_CALL nvkms_copyin     (void *kptr,
                                        NvU64 uaddr,
                                        size_t n);
int    NVKMS_API_CALL nvkms_copyout    (NvU64 uaddr,
                                        const void *kptr,
                                        size_t n);
void   NVKMS_API_CALL nvkms_yield      (void);
int    NVKMS_API_CALL nvkms_snprintf   (char *str,
                                        size_t size,
                                        const char *format, ...)
    __attribute__((format (printf, 3, 4)));

int    NVKMS_API_CALL nvkms_vsnprintf  (char *str,
                                        size_t size,
                                        const char *format,
                                        va_list ap);

#define NVKMS_LOG_LEVEL_INFO  0
#define NVKMS_LOG_LEVEL_WARN  1
#define NVKMS_LOG_LEVEL_ERROR 2

void   NVKMS_API_CALL nvkms_log        (const int level,
                                        const char *gpuPrefix,
                                        const char *msg);

/*!
 * Refcounted pointer to an object that may be freed while references still
 * exist.
 *
 * This structure is intended to be used for nvkms timers to refer to objects
 * that may be freed while timers with references to the object are still
 * pending.
 *
 * When the owner of an nvkms_ref_ptr is freed, the teardown code should call
 * nvkms_free_ref_ptr().  That marks the pointer as invalid so that later calls
 * to nvkms_dec_ref() (i.e. from a workqueue callback) return NULL rather than
 * the pointer originally passed to nvkms_alloc_ref_ptr().
 */
struct nvkms_ref_ptr;

/*!
 * Allocate and initialize a ref_ptr.
 *
 * The pointer stored in the ref_ptr is initialized to ptr, and its refcount is
 * initialized to 1.
 */
struct nvkms_ref_ptr* NVKMS_API_CALL nvkms_alloc_ref_ptr(void *ptr);

/*!
 * Clear a ref_ptr.
 *
 * This function sets the pointer stored in the ref_ptr to NULL and drops the
 * reference created by nvkms_alloc_ref_ptr().  This function should be called
 * when the object pointed to by the ref_ptr is freed.
 *
 * A caller should make sure that no code that can call nvkms_inc_ref() can
 * execute after nvkms_free_ref_ptr() is called.
 */
void NVKMS_API_CALL nvkms_free_ref_ptr(struct nvkms_ref_ptr *ref_ptr);

/*!
 * Increment the refcount of a ref_ptr.
 *
 * This function should be used when a pointer to the ref_ptr is stored
 * somewhere.  For example, when the ref_ptr is used as the argument to
 * nvkms_alloc_timer.
 *
 * This may be called outside of the nvkms_lock, for example by an RM callback.
 */
void NVKMS_API_CALL nvkms_inc_ref(struct nvkms_ref_ptr *ref_ptr);

/*!
 * Decrement the refcount of a ref_ptr and extract the embedded pointer.
 *
 * This should be used by code that needs to atomically determine whether the
 * object pointed to by the ref_ptr still exists.  To prevent the object from
 * being destroyed while the current thread is executing, this should be called
 * from inside the nvkms_lock.
 */
void* NVKMS_API_CALL nvkms_dec_ref(struct nvkms_ref_ptr *ref_ptr);

typedef void NVKMS_API_CALL nvkms_timer_proc_t(void *dataPtr, NvU32 dataU32);
typedef struct nvkms_timer_t nvkms_timer_handle_t;

/*!
 * Schedule a callback function to be called in the future.
 *
 * The callback function 'proc' will be called with the arguments
 * 'dataPtr' and 'dataU32' at 'usec' (or later) microseconds from now.
 * If usec==0, the callback will be scheduled to be called as soon as
 * possible.
 *
 * The callback function is guaranteed to be called back with the
 * nvkms_lock held, and in process context.
 *
 * Returns an opaque handle, nvkms_timer_handle_t*, or NULL on
 * failure.  If non-NULL, the caller is responsible for caching the
 * handle and eventually calling nvkms_free_timer() to free the
 * memory.
 *
 * The nvkms_lock may be held when nvkms_alloc_timer() is called, but
 * the nvkms_lock is not required.
 */
nvkms_timer_handle_t*
      NVKMS_API_CALL nvkms_alloc_timer (nvkms_timer_proc_t *proc,
                                        void *dataPtr, NvU32 dataU32,
                                        NvU64 usec);

/*!
 * Schedule a callback function to be called in the future.
 *
 * This function is like nvkms_alloc_timer() except that instead of returning a
 * pointer to a structure that the caller should free later, the timer will free
 * itself after executing the callback function.  This is only intended for
 * cases where the caller cannot cache the nvkms_alloc_timer() return value.
 */
NvBool NVKMS_API_CALL
nvkms_alloc_timer_with_ref_ptr(nvkms_timer_proc_t *proc,
                               struct nvkms_ref_ptr *ref_ptr,
                               NvU32 dataU32, NvU64 usec);

/*!
 * Free the nvkms_timer_t object.  If the callback function has not
 * yet been called, freeing the nvkms_timer_handle_t will guarantee
 * that it is not called.
 *
 * The nvkms_lock must be held when calling nvkms_free_timer().
 */
void  NVKMS_API_CALL nvkms_free_timer  (nvkms_timer_handle_t *handle);



/*!
 * Notify the NVKMS kernel interface that the event queue has changed.
 *
 * \param[in]  pOpenKernel      This indicates the file descriptor
 *                              ("per-open") of the client whose event queue
 *                              has been updated.  This is the pointer
 *                              passed by the kernel interface to nvKmsOpen().
 * \param[in]  eventsAvailable  If TRUE, a new event has been added to the
 *                              event queue.  If FALSE, the last event has
 *                              been removed from the event queue.
 */
void  NVKMS_API_CALL
nvkms_event_queue_changed(nvkms_per_open_handle_t *pOpenKernel,
                          NvBool eventsAvailable);

/*!
 * Get some random data.
 */
void NVKMS_API_CALL nvkms_get_random(void *ptr, size_t size);


/*!
 * Get the "per-open" data (the pointer returned by nvKmsOpen())
 * associated with this fd.
 */
void* NVKMS_API_CALL nvkms_get_per_open_data(int fd);


/*!
 * Raise and lower the reference count of the specified GPU.
 */
NvBool NVKMS_API_CALL nvkms_open_gpu(NvU32 gpuId);
void NVKMS_API_CALL nvkms_close_gpu(NvU32 gpuId);


/*!
 * Enumerate nvidia gpus.
 */

NvU32 NVKMS_API_CALL nvkms_enumerate_gpus(nv_gpu_info_t *gpu_info);

/*!
 * Availability of write combining support for video memory.
 */

NvBool NVKMS_API_CALL nvkms_allow_write_combining(void);

/*!
 * NVKMS interface for kernel space NVKMS clients like KAPI
 */

struct nvkms_per_open;

struct nvkms_per_open* NVKMS_API_CALL nvkms_open_from_kapi
(
    struct NvKmsKapiDevice *device
);

void NVKMS_API_CALL nvkms_close_from_kapi(struct nvkms_per_open *popen);

NvBool NVKMS_API_CALL nvkms_ioctl_from_kapi
(
    struct nvkms_per_open *popen,
    NvU32 cmd, void *params_address, const size_t params_size
);

/*!
 * APIs for locking.
 */

typedef struct nvkms_sema_t nvkms_sema_handle_t;

nvkms_sema_handle_t*
     NVKMS_API_CALL nvkms_sema_alloc    (void);
void NVKMS_API_CALL nvkms_sema_free     (nvkms_sema_handle_t *sema);
void NVKMS_API_CALL nvkms_sema_down     (nvkms_sema_handle_t *sema);
void NVKMS_API_CALL nvkms_sema_up       (nvkms_sema_handle_t *sema);

#endif /* _NVIDIA_MODESET_OS_INTERFACE_H_ */

