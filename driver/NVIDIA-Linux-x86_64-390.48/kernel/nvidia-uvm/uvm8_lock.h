/*******************************************************************************
    Copyright (c) 2015 NVIDIA Corporation

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

#ifndef __UVM8_LOCK_H__
#define __UVM8_LOCK_H__

#include "uvm8_forward_decl.h"
#include "uvm_linux.h"
#include "uvm_common.h"

// --------------------------- UVM Locking Order ---------------------------- //
//
// Any locks described here should have their locking order added to
// uvm_lock_order_t below.
//
// - Global driver state lock (g_uvm_global.global_lock)
//      Order: UVM_LOCK_ORDER_GLOBAL
//      Exclusive lock (mutex)
//
//      This protects state associated with GPUs, such as the P2P table
//      and instance pointer mappings.
//
//      This should be taken whenever global GPU state might need to be modified.
//
// - GPU ISR lock
//      Order: UVM_LOCK_ORDER_ISR
//      Exclusive lock (mutex) per gpu
//
//      Protects:
//      - gpu->isr.replayable_faults.service_lock:
//        Changes to the state of a GPU as it transitions from top-half to bottom-half
//        interrupt handler for replayable faults. This lock is acquired for that GPU,
//        in the ISR top-half. Then a bottom-half is scheduled (to run in a workqueue).
//        Then the bottom-half releases the lock when that GPU's processing appears to
//        be done.
//      - gpu->isr.non_replayable_faults.service_lock:
//        Changes to the state of a GPU in the bottom-half for non-replayable faults.
//        Non-replayable faults are handed-off from RM instead of directly from the GPU
//        hardware. This means that we do not keep receiving interrupts after RM pops
//        out the faults from the HW buffer. In order not to miss fault notifications,
//        we will always schedule a bottom-half for non-replayable faults if there are
//        faults ready to be consumed in the buffer, even if there already is some
//        bottom-half running or scheduled. This lock serializes all scheduled bottom
//        halves per GPU which service non-replayable faults.
//      - gpu->isr.access_counters.service_lock:
//        Changes to the state of a GPU as it transitions from top-half to bottom-half
//        interrupt handler for access counter notifications. This lock is acquired for
//        that GPU, in the ISR top-half. Then a bottom-half is scheduled (to run in a
//        workqueue). Then the bottom-half releases the lock when that GPU's processing
//        appears to be done.
//
// - mmap_sem
//      Order: UVM_LOCK_ORDER_MMAP_SEM
//      Reader/writer lock (rw_semaphore)
//
//      We're often called with the kernel already holding mmap_sem: mmap,
//      munmap, fault, etc. These operations may have to take any number of UVM
//      locks, so mmap_sem requires special consideration in the lock order,
//      since it's sometimes out of our control.
//
//      We need to hold mmap_sem when calling vm_insert_page, which means that
//      any time an operation (such as an ioctl) might need to install a CPU
//      mapping, it must take current->mm->mmap_sem in read mode very early on.
//
//      However, current->mm is not necessarily the owning mm of the UVM vma.
//      fork or fd passing via a UNIX doman socket can cause that. Notably, this
//      is also the case when handling GPU faults from a kernel thread. This
//      means we must lock current->mm->mmap_sem, then look up the UVM vma and
//      compare its mm before operating on that vma.
//
//      With HMM and ATS, the GPU fault handler takes mmap_sem. GPU faults may
//      block forward progress of threads holding the RM GPUs lock until those
//      faults are serviced, which means that mmap_sem cannot be held when the
//      UVM driver calls into RM. In other words, mmap_sem and the RM GPUs lock
//      are mutually exclusive.
//
// - VA space writer serialization lock (va_space->serialize_writers_lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS
//      Exclusive lock (mutex) per uvm_va_space (UVM struct file)
//
//      This lock prevents a deadlock between RM and UVM by only allowing one
//      writer to queue up on the VA space lock at a time.
//
//      GPU faults are serviced by the UVM bottom half with the VA space lock
//      held in read mode. Until they're serviced, these faults may block
//      forward progress of RM threads.
//
//      This constraint means that the UVM driver cannot call into RM while
//      GPU fault servicing is blocked. We may block GPU fault servicing by:
//      - Taking the VA space lock in write mode
//      - Holding the VA space lock in read mode with a writer pending, since
//        Linux rw_semaphores are fair.
//
//      Example of the second condition:
//      Thread A        Thread B        UVM BH          Thread C
//      UVM API call    UVM API call    GPU fault       RM API call
//      ------------    ------------    ------------    ------------
//      down_read
//                      down_write
//                      // Blocked on A
//                                      down_read
//                                      // Blocked on B
//                                                      RM GPU lock
//                                                      // Blocked on GPU fault
//      RM GPU lock
//      // Deadlock
//
//      The writer serialization lock works around this by biasing the VA space
//      lock towards readers, without causing starvation of writers. Writers and
//      readers which will make RM calls take this lock, which prevents them
//      from queueing up on the VA space rw_semaphore and blocking the UVM
//      bottom half.
//
//      TODO: Bug 1799173: A better long-term approach might be to never allow
//            RM calls under the VA space lock at all, but that will take a
//            larger restructuring.
//
// - VA space serialization of down_read with up_write of the VA space lock
//   (va_space->read_acquire_write_release_lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK
//      Exclusive lock (mutex) per uvm_va_space (UVM struct file)
//
//      This lock prevents a deadlock between RM and UVM by preventing any
//      interleaving of down_reads on the VA space lock with concurrent
//      up_writes/downgrade_writes. The Linux rw_semaphore implementation does
//      not guarantee that two readers will always run concurrently, as shown by
//      the following interleaving:
//
//      Thread A                Thread B
//      UVM API call            UVM BH
//      ------------            ------------
//      down_write
//                              down_read
//                                  // Fails, calls handler
//      up_write
//      down_read
//          // Success
//                                  // Handler sees the lock still active
//                                  // Handler waits for lock to be released
//                                  // Blocked on A
//      RM GPU lock
//      // Blocked on GPU fault
//
//      Given the above interleaving, the kernel's implementation of the
//      down_read failure handler running in thread B does not distinguish
//      between a reader vs writer holding the lock. From the perspective of all
//      other threads, even those which attempt to take the lock for read while
//      thread A's reader holds it, a writer is active. Therefore no other
//      readers can take the lock, and we result in the same deadlock described
//      in the above comments on the VA space writer serialization lock.
//
//      This lock prevents any such interleaving:
//      - Writers take this lock for the duration of the write lock.
//
//      - Readers which do not call into RM only take this lock across the
//        down_read call. If a writer holds the lock, the reader would be
//        blocked on the VA space lock anyway. Concurrent readers will serialize
//        the taking of the VA space lock, but they will not be serialized
//        across their read sections.
//
//      - Readers which call into RM do not need to take this lock. Their
//        down_read is already serialized with a writer's up_write by the
//        serialize_writers_lock.
//
// - VA space lock (va_space->lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE
//      Reader/writer lock (rw_semaphore) per uvm_va_space (UVM struct file)
//
//      This is the UVM equivalent of mmap_sem. It protects all state under that
//      va_space, such as the VA range tree.
//
//      Read mode: Faults (CPU and GPU), mapping creation, prefetches. These
//      will be serialized at the VA block level if necessary. RM calls are
//      allowed only if the VA space serialize_writers_lock is also taken.
//
//      Write mode: Modification of the range state such as mmap and changes to
//      logical permissions or location preferences. RM calls are never allowed.
//
// - GPU semaphore pool lock (semaphore_pool->mutex)
//      Order: UVM_LOCK_ORDER_GPU_SEMAPHORE_POOL
//      Exclusive lock (mutex) per uvm_gpu_semaphore_pool
//
//      Protects the state of the semaphore pool.
//
// - RM API lock
//      Order: UVM_LOCK_ORDER_RM_API
//      Exclusive lock
//
//      This is an internal RM lock that's acquired by most if not all UVM-RM
//      APIs.
//      Notably this lock is also held on PMA eviction.
//
// - RM GPUs lock
//      Order: UVM_LOCK_ORDER_RM_GPUS
//      Exclusive lock
//
//      This is an internal RM lock that's acquired by most if not all UVM-RM
//      APIs and disables interrupts for the GPUs.
//      Notably this lock is *not* held on PMA eviction.
//
// - VA block lock (va_block->lock)
//      Order: UVM_LOCK_ORDER_VA_BLOCK
//      Exclusive lock (mutex)
//
//      Protects:
//      - CPU and GPU page table mappings for all VAs under the block
//      - Updates to the GPU work tracker for that block (migrations)
//
//      Operations allowed while holding the lock:
//      - CPU allocation (we don't evict CPU memory)
//      - GPU memory allocation which cannot evict
//      - CPU page table mapping/unmapping
//      - Pushing work (GPU page table mapping/unmapping)
//
//      Operations not allowed while holding the lock:
//      - GPU memory allocation which can evict memory (would require nesting
//        block locks)
//
// - Page tree lock
//      Order: UVM_LOCK_ORDER_PAGE_TREE
//      Exclusive lock per GPU page tree
//
//      This protects a page tree.  All modifications to the device's page tree
//      and the host-side cache of that tree must be done under this lock.
//      The host-side cache and device state must be consistent when this lock is released
//
//      Operations allowed while holding this lock
//      - Pushing work
//
//      Operations not allowed while holding this lock
//      - GPU memory allocation which can evict
//
// - GPU big page staging lock
//      Order: UVM_LOCK_ORDER_SWIZZLE_STAGING
//      Exclusive lock (mutex)
//
//      Protects operations which use the per-GPU big page chunk and tracker.
//
// - Concurrent push semaphore
//      Order: UVM_LOCK_ORDER_PUSH
//      Semaphore (uvm_semaphore_t)
//
//      This is a semaphore limiting the amount of concurrent pushes that is
//      held for the duration of a push (between uvm_push_begin*() and
//      uvm_push_end()).
//
// - PMM GPU lock (pmm->lock)
//      Order: UVM_LOCK_ORDER_PMM
//      Exclusive lock (mutex) per uvm_pmm_gpu_t
//
//      Protects the state of PMM - internal to PMM.
//
// - PMM GPU PMA lock (pmm->pma_lock)
//      Order: UVM_LOCK_ORDER_PMM_PMA
//      Reader/writer lock (rw_semaphore) per per uvm_pmm_gpu_t
//
//      Lock internal to PMM for synchronizing allocations from PMA with
//      PMA eviction.
//
// - PMM root chunk lock (pmm->root_chunks_bitlocks)
//      Order: UVM_LOCK_ORDER_PMM_ROOT_CHUNK
//      Exclusive bitlock (mutex) per each root chunk internal to PMM.
//
// - Channel lock
//      Order: UVM_LOCK_ORDER_CHANNEL
//      Spinlock (uvm_spinlock_t)
//
// - Tools global VA space list lock (g_tools_va_space_list_lock)
//      Order: UVM_LOCK_ORDER_TOOLS_VA_SPACE_LIST
//      Reader/writer lock (rw_sempahore)
//
//      This lock protects the list of VA spaces used when broadcasting
//      UVM profiling events.
//
// - VA space events
//      Order: UVM_LOCK_ORDER_VA_SPACE_EVENTS
//      Reader/writer lock (rw_semaphore) per uvm_perf_va_space_events_t.
//      serializes perf callbacks with event register/unregister. It's separate
//      from the VA space lock so it can be taken on the eviction path.
//
// - VA space tools
//      Order: UVM_LOCK_ORDER_VA_SPACE_TOOLS
//      Reader/writer lock (rw_semaphore) per uvm_va_space_t. Serializes tools
//      reporting with tools register/unregister. Since some of the tools
//      events come from perf events, both VA_SPACE_EVENTS and VA_SPACE_TOOLS
//      must be taken to register/report some tools events.
//
// - Leaf locks
//      Order: UVM_LOCK_ORDER_LEAF
//
//      All leaf locks.
//
// -------------------------------------------------------------------------- //

// Remember to add any new lock orders to uvm_lock_order_to_string() in uvm8_lock.c
typedef enum
{
    UVM_LOCK_ORDER_INVALID = 0,
    UVM_LOCK_ORDER_GLOBAL,
    UVM_LOCK_ORDER_ISR,
    UVM_LOCK_ORDER_MMAP_SEM,
    UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS,
    UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK,
    UVM_LOCK_ORDER_VA_SPACE,
    UVM_LOCK_ORDER_GPU_SEMAPHORE_POOL,
    UVM_LOCK_ORDER_RM_API,
    UVM_LOCK_ORDER_RM_GPUS,
    UVM_LOCK_ORDER_VA_BLOCK,
    UVM_LOCK_ORDER_PAGE_TREE,
    UVM_LOCK_ORDER_SWIZZLE_STAGING,
    UVM_LOCK_ORDER_PUSH,
    UVM_LOCK_ORDER_PMM,
    UVM_LOCK_ORDER_PMM_PMA,
    UVM_LOCK_ORDER_PMM_ROOT_CHUNK,
    UVM_LOCK_ORDER_CHANNEL,
    UVM_LOCK_ORDER_TOOLS_VA_SPACE_LIST,
    UVM_LOCK_ORDER_VA_SPACE_EVENTS,
    UVM_LOCK_ORDER_VA_SPACE_TOOLS,
    UVM_LOCK_ORDER_SEMA_POOL_TRACKER,
    UVM_LOCK_ORDER_LEAF,
    UVM_LOCK_ORDER_COUNT,
} uvm_lock_order_t;

const char *uvm_lock_order_to_string(uvm_lock_order_t lock_order);

typedef enum
{
    UVM_LOCK_MODE_EXCLUSIVE,
    UVM_LOCK_MODE_SHARED,

    // Special value so uvm_check_locked can check for either mode
    UVM_LOCK_MODE_ANY,
} uvm_lock_mode_t;

// Record locking a lock of given lock_order in exclusive or shared mode
// Returns true if the recorded lock follows all the locking rules and false otherwise.
bool __uvm_record_lock(void *lock, uvm_lock_order_t lock_order, uvm_lock_mode_t mode);

// Record unlocking a lock of given lock_order in exclusive or shared mode and
// possibly out of order.
// Returns true if the unlock follows all the locking rules and false otherwise.
bool __uvm_record_unlock(void *lock, uvm_lock_order_t lock_order, uvm_lock_mode_t mode, bool out_of_order);

bool __uvm_record_downgrade(void *lock, uvm_lock_order_t lock_order);

// Check whether a lock of given lock_order is held in exclusive, shared, or
// either mode by the current thread.
bool __uvm_check_locked(void *lock, uvm_lock_order_t lock_order, uvm_lock_mode_t mode);

// Check that no locks are held with the given lock order
bool __uvm_check_unlocked_order(uvm_lock_order_t lock_order);

// Check that a lock of the given order can be locked, i.e. that no locks are
// held with the given or deeper lock order.
bool __uvm_check_lockable_order(uvm_lock_order_t lock_order);

// Check that all locks have been released in a uvm thread context
bool __uvm_check_all_unlocked(uvm_thread_context_t *uvm_context);

// Check that all locks have been released in the current uvm thread context
bool __uvm_thread_check_all_unlocked(void);

#if UVM_IS_DEBUG()
  // These macros are intended to be expanded on the call site directly and will print
  // the precise location of the violation while the __uvm_record* functions will error print the details.
  #define uvm_record_lock_raw(lock, lock_order, mode) \
      UVM_ASSERT_MSG(__uvm_record_lock((lock), (lock_order), (mode)), "Locking violation\n")
  #define uvm_record_unlock_raw(lock, lock_order, mode, out_of_order) \
      UVM_ASSERT_MSG(__uvm_record_unlock((lock), (lock_order), (mode), (out_of_order)), "Locking violation\n")
  #define uvm_record_downgrade_raw(lock, lock_order) \
      UVM_ASSERT_MSG(__uvm_record_downgrade((lock), (lock_order)), "Locking violation\n")

  // Record UVM lock (a lock that has a lock_order member) operation and assert that it's correct
  #define uvm_record_lock(lock, mode) uvm_record_lock_raw((lock), (lock)->lock_order, (mode))
  #define uvm_record_unlock(lock, mode) uvm_record_unlock_raw((lock), (lock)->lock_order, (mode), false)
  #define uvm_record_unlock_out_of_order(lock, mode) uvm_record_unlock_raw((lock), (lock)->lock_order, (mode), true)
  #define uvm_record_downgrade(lock) uvm_record_downgrade_raw((lock), (lock)->lock_order)

  // Check whether a UVM lock (a lock that has a lock_order member) is held in
  // the given mode.
  #define uvm_check_locked(lock, mode) __uvm_check_locked((lock), (lock)->lock_order, (mode))

  // Helpers for recording and asserting mmap_sem state
  #define uvm_record_lock_mmap_sem_read(mmap_sem) \
          uvm_record_lock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_SHARED)

  #define uvm_record_unlock_mmap_sem_read(mmap_sem) \
          uvm_record_unlock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_SHARED, false)

  #define uvm_record_unlock_mmap_sem_read_out_of_order(mmap_sem) \
          uvm_record_unlock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_SHARED, true)

  #define uvm_record_lock_mmap_sem_write(mmap_sem) \
          uvm_record_lock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_EXCLUSIVE)

  #define uvm_record_unlock_mmap_sem_write(mmap_sem) \
          uvm_record_unlock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_EXCLUSIVE, false)

  #define uvm_record_unlock_mmap_sem_write_out_of_order(mmap_sem) \
          uvm_record_unlock_raw((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, UVM_LOCK_MODE_EXCLUSIVE, true)

  #define uvm_check_locked_mmap_sem(mmap_sem, mode) \
           __uvm_check_locked((mmap_sem), UVM_LOCK_ORDER_MMAP_SEM, (mode))

  // Helpers for recording RM API lock usage around UVM-RM interfaces
  #define uvm_record_lock_rm_api() \
          uvm_record_lock_raw((void*)UVM_LOCK_ORDER_RM_API, UVM_LOCK_ORDER_RM_API, UVM_LOCK_MODE_EXCLUSIVE)
  #define uvm_record_unlock_rm_api() \
          uvm_record_unlock_raw((void*)UVM_LOCK_ORDER_RM_API, UVM_LOCK_ORDER_RM_API, UVM_LOCK_MODE_EXCLUSIVE, false)

  // Helpers for recording RM GPUS lock usage around UVM-RM interfaces
  #define uvm_record_lock_rm_gpus() \
          uvm_record_lock_raw((void*)UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_MODE_EXCLUSIVE)
  #define uvm_record_unlock_rm_gpus() \
          uvm_record_unlock_raw((void*)UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_MODE_EXCLUSIVE, false)

  // Helpers for recording both RM locks usage around UVM-RM interfaces
  #define uvm_record_lock_rm_all() ({ uvm_record_lock_rm_api(); uvm_record_lock_rm_gpus(); })
  #define uvm_record_unlock_rm_all() ({ uvm_record_unlock_rm_gpus(); uvm_record_unlock_rm_api(); })

#else
  #define uvm_record_lock                               UVM_IGNORE_EXPR2
  #define uvm_record_unlock                             UVM_IGNORE_EXPR2
  #define uvm_record_unlock_out_of_order                UVM_IGNORE_EXPR2
  #define uvm_record_downgrade                          UVM_IGNORE_EXPR

  static bool uvm_check_locked(void *lock, uvm_lock_mode_t mode)
  {
      return false;
  }

  #define uvm_record_lock_mmap_sem_read                 UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_sem_read               UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_sem_read_out_of_order  UVM_IGNORE_EXPR
  #define uvm_record_lock_mmap_sem_write                UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_sem_write              UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_sem_write_out_of_order UVM_IGNORE_EXPR

  #define uvm_check_locked_mmap_sem                     uvm_check_locked

  #define uvm_record_lock_rm_api()
  #define uvm_record_unlock_rm_api()

  #define uvm_record_lock_rm_gpus()
  #define uvm_record_unlock_rm_gpus()

  #define uvm_record_lock_rm_all()
  #define uvm_record_unlock_rm_all()
#endif

#define uvm_thread_assert_all_unlocked() UVM_ASSERT(__uvm_thread_check_all_unlocked())
#define uvm_assert_lockable_order(order) UVM_ASSERT(__uvm_check_lockable_order(order))
#define uvm_assert_unlocked_order(order) UVM_ASSERT(__uvm_check_unlocked_order(order))

// Helpers for locking mmap_sem and recording its usage
#define uvm_assert_mmap_sem_locked_mode(mmap_sem, mode) ({                          \
      typeof(mmap_sem) _sem = (mmap_sem);                                           \
      UVM_ASSERT(rwsem_is_locked(_sem) && uvm_check_locked_mmap_sem(_sem, (mode))); \
  })

#define uvm_assert_mmap_sem_locked(mmap_sem)        uvm_assert_mmap_sem_locked_mode((mmap_sem), UVM_LOCK_MODE_ANY)
#define uvm_assert_mmap_sem_locked_read(mmap_sem)   uvm_assert_mmap_sem_locked_mode((mmap_sem), UVM_LOCK_MODE_SHARED)
#define uvm_assert_mmap_sem_locked_write(mmap_sem)  uvm_assert_mmap_sem_locked_mode((mmap_sem), UVM_LOCK_MODE_EXCLUSIVE)

#define uvm_down_read_mmap_sem(mmap_sem) ({             \
        typeof(mmap_sem) _sem = (mmap_sem);             \
        uvm_record_lock_mmap_sem_read(_sem);            \
        down_read(_sem);                                \
    })

#define uvm_up_read_mmap_sem(mmap_sem) ({               \
        typeof(mmap_sem) _sem = (mmap_sem);             \
        up_read(_sem);                                  \
        uvm_record_unlock_mmap_sem_read(_sem);          \
    })

#define uvm_up_read_mmap_sem_out_of_order(mmap_sem) ({      \
        typeof(mmap_sem) _sem = (mmap_sem);                 \
        up_read(_sem);                                      \
        uvm_record_unlock_mmap_sem_read_out_of_order(_sem); \
    })

#define uvm_down_write_mmap_sem(mmap_sem) ({            \
        typeof(mmap_sem) _sem = (mmap_sem);             \
        uvm_record_lock_mmap_sem_write(_sem);           \
        down_write(_sem);                               \
    })

#define uvm_up_write_mmap_sem(mmap_sem) ({              \
        typeof(mmap_sem) _sem = (mmap_sem);             \
        up_write(_sem);                                 \
        uvm_record_unlock_mmap_sem_write(_sem);         \
    })

// Helper for calling a UVM-RM interface function with lock recording
#define uvm_rm_locked_call(call) ({                     \
        typeof(call) ret;                               \
        uvm_record_lock_rm_all();                       \
        ret = call;                                     \
        uvm_record_unlock_rm_all();                     \
        ret;                                            \
    })

// Helper for calling a UVM-RM interface function that returns void with lock recording
#define uvm_rm_locked_call_void(call) ({                \
        uvm_record_lock_rm_all();                       \
        call;                                           \
        uvm_record_unlock_rm_all();                     \
    })

typedef struct
{
    struct rw_semaphore sem;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_rw_semaphore_t;

#define uvm_assert_rwsem_locked_mode(uvm_sem, mode) ({                              \
        typeof(uvm_sem) _sem = (uvm_sem);                                           \
        UVM_ASSERT(rwsem_is_locked(&_sem->sem) && uvm_check_locked(_sem, (mode)));  \
    })

#define uvm_assert_rwsem_locked(uvm_sem)        uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_MODE_ANY)
#define uvm_assert_rwsem_locked_read(uvm_sem)   uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_MODE_SHARED)
#define uvm_assert_rwsem_locked_write(uvm_sem)  uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_MODE_EXCLUSIVE)

#define uvm_assert_rwsem_unlocked(uvm_sem) UVM_ASSERT(!rwsem_is_locked(&(uvm_sem)->sem))

static void uvm_init_rwsem(uvm_rw_semaphore_t *uvm_sem, uvm_lock_order_t lock_order)
{
    init_rwsem(&uvm_sem->sem);
#if UVM_IS_DEBUG()
    uvm_sem->lock_order = lock_order;
#endif
    uvm_assert_rwsem_unlocked(uvm_sem);
}

static void __uvm_down_read(uvm_rw_semaphore_t *uvm_sem)
{
    down_read(&uvm_sem->sem);
    uvm_assert_rwsem_locked_read(uvm_sem);
}

#define uvm_down_read(sem) ({                           \
        typeof(sem) _sem = (sem);                       \
        uvm_record_lock(_sem, UVM_LOCK_MODE_SHARED);    \
        __uvm_down_read(_sem);                          \
    })

static void __uvm_up_read(uvm_rw_semaphore_t *uvm_sem)
{
    uvm_assert_rwsem_locked_read(uvm_sem);
    up_read(&uvm_sem->sem);
}
#define uvm_up_read(sem) ({                             \
        typeof(sem) _sem = (sem);                       \
        __uvm_up_read(_sem);                            \
        uvm_record_unlock(_sem, UVM_LOCK_MODE_SHARED);  \
    })

static void __uvm_down_write(uvm_rw_semaphore_t *uvm_sem)
{
    down_write(&uvm_sem->sem);
    uvm_assert_rwsem_locked_write(uvm_sem);
}
#define uvm_down_write(sem) ({                          \
        typeof (sem) _sem = (sem);                      \
        uvm_record_lock(_sem, UVM_LOCK_MODE_EXCLUSIVE); \
        __uvm_down_write(_sem);                         \
    })

// trylock for writing: returns 1 if successful, 0 if not:
static int __uvm_down_write_trylock(uvm_rw_semaphore_t *uvm_sem)
{
    int ret = down_write_trylock(&uvm_sem->sem);
    if (ret == 0)
        return ret;

    uvm_assert_rwsem_locked_write(uvm_sem);

    return ret;
}
#define uvm_down_write_trylock(sem) ({                          \
        typeof(sem) _sem = (sem);                               \
        int locked;                                             \
        uvm_record_lock(_sem, UVM_LOCK_MODE_EXCLUSIVE);         \
        locked = __uvm_down_write_trylock(_sem);                \
        if (locked == 0)                                        \
            uvm_record_unlock(_sem, UVM_LOCK_MODE_EXCLUSIVE);   \
        locked;                                                 \
    })

static void __uvm_up_write(uvm_rw_semaphore_t *uvm_sem)
{
    uvm_assert_rwsem_locked_write(uvm_sem);
    up_write(&uvm_sem->sem);
}
#define uvm_up_write(sem) ({                                \
        typeof(sem) _sem = (sem);                           \
        __uvm_up_write(_sem);                               \
        uvm_record_unlock(_sem, UVM_LOCK_MODE_EXCLUSIVE);   \
    })

static void __uvm_downgrade_write(uvm_rw_semaphore_t *uvm_sem)
{
    uvm_assert_rwsem_locked_write(uvm_sem);
    downgrade_write(&uvm_sem->sem);
}
#define uvm_downgrade_write(sem) ({                     \
        typeof(sem) _sem = (sem);                       \
        __uvm_downgrade_write(_sem);                    \
        uvm_record_downgrade(_sem);                     \
    })

typedef struct
{
    struct mutex m;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_mutex_t;

#define uvm_assert_mutex_locked(uvm_mutex) ({                                                           \
        typeof(uvm_mutex) _mutex = (uvm_mutex);                                                         \
        UVM_ASSERT(mutex_is_locked(&_mutex->m) && uvm_check_locked(_mutex, UVM_LOCK_MODE_EXCLUSIVE));   \
    })

#define uvm_assert_mutex_unlocked(uvm_mutex) UVM_ASSERT(!mutex_is_locked(&(uvm_mutex)->m))

static void uvm_mutex_init(uvm_mutex_t *mutex, uvm_lock_order_t lock_order)
{
    mutex_init(&mutex->m);
#if UVM_IS_DEBUG()
    mutex->lock_order = lock_order;
#endif
    uvm_assert_mutex_unlocked(mutex);
}

static void __uvm_mutex_lock(uvm_mutex_t *mutex)
{
    mutex_lock(&mutex->m);
    uvm_assert_mutex_locked(mutex);
}
#define uvm_mutex_lock(mutex) ({                            \
        typeof(mutex) _mutex = (mutex);                     \
        uvm_record_lock(_mutex, UVM_LOCK_MODE_EXCLUSIVE);   \
        __uvm_mutex_lock(_mutex);                           \
    })

// Lock w/o any tracking. This should be extremely rare and *_no_tracking
// helpers will be added only as needed.
#define uvm_mutex_lock_no_tracking(mutex) mutex_lock(&(mutex)->m)

static void __uvm_mutex_unlock(uvm_mutex_t *mutex)
{
    uvm_assert_mutex_locked(mutex);
    mutex_unlock(&mutex->m);
}
#define uvm_mutex_unlock(mutex) ({                          \
        typeof(mutex) _mutex = (mutex);                     \
        __uvm_mutex_unlock(_mutex);                         \
        uvm_record_unlock(_mutex, UVM_LOCK_MODE_EXCLUSIVE); \
    })
#define uvm_mutex_unlock_out_of_order(mutex) ({                          \
        typeof(mutex) _mutex = (mutex);                                  \
        __uvm_mutex_unlock(_mutex);                                      \
        uvm_record_unlock_out_of_order(_mutex, UVM_LOCK_MODE_EXCLUSIVE); \
    })

// Unlock w/o any tracking. This should be extremely rare and *_no_tracking
// helpers will be added only as needed.
#define uvm_mutex_unlock_no_tracking(mutex) mutex_unlock(&(mutex)->m)

typedef struct
{
    struct semaphore sem;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_semaphore_t;

static void uvm_sema_init(uvm_semaphore_t *semaphore, int val, uvm_lock_order_t lock_order)
{
    sema_init(&semaphore->sem, val);
#if UVM_IS_DEBUG()
    semaphore->lock_order = lock_order;
#endif
}

static void __uvm_down(uvm_semaphore_t *semaphore)
{
    down(&semaphore->sem);
}
#define uvm_down(sem) ({                                \
        typeof(sem) _sem = (sem);                       \
        uvm_record_lock(_sem, UVM_LOCK_MODE_SHARED);    \
        __uvm_down(_sem);                               \
    })

static void __uvm_up(uvm_semaphore_t *semaphore)
{
    up(&semaphore->sem);
}
#define uvm_up(sem) ({                                  \
        typeof(sem) _sem = (sem);                       \
        __uvm_up(_sem);                                 \
        uvm_record_unlock(_sem, UVM_LOCK_MODE_SHARED);  \
    })
#define uvm_up_out_of_order(sem) ({                                      \
        typeof(sem) _sem = (sem);                                        \
        __uvm_up(_sem);                                                  \
        uvm_record_unlock_out_of_order(_sem, UVM_LOCK_MODE_SHARED);      \
    })

// A regular spinlock
// Locked/unlocked with uvm_spin_lock()/uvm_spin_unlock()
typedef struct
{
    spinlock_t lock;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_spinlock_t;

// A separate spinlock type for spinlocks that need to disable interrupts. For
// guaranteed correctness and convenience embed the saved and restored irq state
// in the lock itself.
// Locked/unlocked with uvm_spin_lock_irqsave()/uvm_spin_unlock_irqrestore()
typedef struct
{
    spinlock_t lock;
    unsigned long irq_flags;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_spinlock_irqsave_t;

// Asserts that the spinlock is held. Notably the macros below support both
// types of spinlocks.
#define uvm_assert_spinlock_locked(spinlock) ({                                                         \
        typeof(spinlock) _lock = (spinlock);                                                            \
        UVM_ASSERT(spin_is_locked(&_lock->lock) && uvm_check_locked(_lock, UVM_LOCK_MODE_EXCLUSIVE));   \
    })

#define uvm_assert_spinlock_unlocked(spinlock) UVM_ASSERT(!spin_is_locked(&(spinlock)->lock))

static void uvm_spin_lock_init(uvm_spinlock_t *spinlock, uvm_lock_order_t lock_order)
{
    spin_lock_init(&spinlock->lock);
#if UVM_IS_DEBUG()
    spinlock->lock_order = lock_order;
#endif
    uvm_assert_spinlock_unlocked(spinlock);
}

static void __uvm_spin_lock(uvm_spinlock_t *spinlock)
{
    spin_lock(&spinlock->lock);
    uvm_assert_spinlock_locked(spinlock);
}
#define uvm_spin_lock(lock) ({                              \
        typeof(lock) _lock = (lock);                        \
        uvm_record_lock(_lock, UVM_LOCK_MODE_EXCLUSIVE);    \
        __uvm_spin_lock(_lock);                             \
    })

static void __uvm_spin_unlock(uvm_spinlock_t *spinlock)
{
    uvm_assert_spinlock_locked(spinlock);
    spin_unlock(&spinlock->lock);
}
#define uvm_spin_unlock(lock) ({                            \
        typeof(lock) _lock = (lock);                        \
        __uvm_spin_unlock(_lock);                           \
        uvm_record_unlock(_lock, UVM_LOCK_MODE_EXCLUSIVE);  \
    })

static void uvm_spin_lock_irqsave_init(uvm_spinlock_irqsave_t *spinlock, uvm_lock_order_t lock_order)
{
    spin_lock_init(&spinlock->lock);
#if UVM_IS_DEBUG()
    spinlock->lock_order = lock_order;
#endif
    uvm_assert_spinlock_unlocked(spinlock);
}

static void __uvm_spin_lock_irqsave(uvm_spinlock_irqsave_t *spinlock)
{
    // Use a temp to not rely on flags being written after acquiring the lock.
    unsigned long irq_flags;
    spin_lock_irqsave(&(spinlock)->lock, irq_flags);
    spinlock->irq_flags = irq_flags;
    uvm_assert_spinlock_locked(spinlock);
}
#define uvm_spin_lock_irqsave(lock) ({                      \
        typeof(lock) _lock = (lock);                        \
        uvm_record_lock(_lock, UVM_LOCK_MODE_EXCLUSIVE);    \
        __uvm_spin_lock_irqsave(_lock);                     \
    })

static void __uvm_spin_unlock_irqrestore(uvm_spinlock_irqsave_t *spinlock)
{
    // Use a temp to not rely on flags being read before releasing the lock.
    unsigned long irq_flags = spinlock->irq_flags;
    uvm_assert_spinlock_locked(spinlock);
    spin_unlock_irqrestore(&(spinlock)->lock, irq_flags);
}
#define uvm_spin_unlock_irqrestore(lock) ({                 \
        typeof(lock) _lock = (lock);                        \
        __uvm_spin_unlock_irqrestore(_lock);                \
        uvm_record_unlock(_lock, UVM_LOCK_MODE_EXCLUSIVE);  \
    })

// Bit locks are 'compressed' mutexes which take only 1 bit per lock by virtue
// of using shared waitqueues.
typedef struct
{
    unsigned long *bits;

#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_bit_locks_t;

NV_STATUS uvm_bit_locks_init(uvm_bit_locks_t *bit_locks, size_t count, uvm_lock_order_t lock_order);
void uvm_bit_locks_deinit(uvm_bit_locks_t *bit_locks);

// Asserts that the bit lock is held.
//
// TODO: Bug 1766601:
//  - assert for the right ownership (defining the owner might be tricky in
//    the kernel).
#define uvm_assert_bit_locked(bit_locks, bit) ({                        \
    typeof(bit_locks) _bit_locks = (bit_locks);                         \
    typeof(bit) _bit = (bit);                                           \
    UVM_ASSERT(test_bit(_bit, _bit_locks->bits));                       \
    UVM_ASSERT(uvm_check_locked(_bit_locks, UVM_LOCK_MODE_EXCLUSIVE));  \
})

#define uvm_assert_bit_unlocked(bit_locks, bit) ({                      \
    typeof(bit_locks) _bit_locks = (bit_locks);                         \
    typeof(bit) _bit = (bit);                                           \
    UVM_ASSERT(!test_bit(_bit, _bit_locks->bits));                      \
})

static void __uvm_bit_lock(uvm_bit_locks_t *bit_locks, unsigned long bit)
{
    int res;

    res = UVM_WAIT_ON_BIT_LOCK(bit_locks->bits, bit, TASK_UNINTERRUPTIBLE);
    UVM_ASSERT_MSG(res == 0, "Uninterruptible task interrupted: %d\n", res);
    uvm_assert_bit_locked(bit_locks, bit);
}
#define uvm_bit_lock(bit_locks, bit) ({                     \
    typeof(bit_locks) _bit_locks = (bit_locks);             \
    typeof(bit) _bit = (bit);                               \
    uvm_record_lock(_bit_locks, UVM_LOCK_MODE_EXCLUSIVE);   \
    __uvm_bit_lock(_bit_locks, _bit);                       \
})

static void __uvm_bit_unlock(uvm_bit_locks_t *bit_locks, unsigned long bit)
{
    uvm_assert_bit_locked(bit_locks, bit);

    clear_bit_unlock(bit, bit_locks->bits);
    // Make sure we don't reorder release with wakeup as it would cause
    // deadlocks (other thread checking lock and adding itself to queue
    // in reversed order). clear_bit_unlock has only release semantics.
    smp_mb__after_atomic();
    wake_up_bit(bit_locks->bits, bit);
}
#define uvm_bit_unlock(bit_locks, bit) ({                   \
    typeof(bit_locks) _bit_locks = (bit_locks);             \
    typeof(bit) _bit = (bit);                               \
    __uvm_bit_unlock(_bit_locks, _bit);                     \
    uvm_record_unlock(_bit_locks, UVM_LOCK_MODE_EXCLUSIVE); \
})

#endif // __UVM8_LOCK_H__
