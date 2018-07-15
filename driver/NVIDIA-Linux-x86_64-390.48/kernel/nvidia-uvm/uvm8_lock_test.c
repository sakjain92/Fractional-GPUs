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

#include "uvm8_test.h"
#include "uvm8_lock.h"
#include "uvm8_global.h"
#include "uvm8_thread_context.h"

static bool fake_lock(uvm_lock_order_t lock_order, uvm_lock_mode_t mode)
{
    // Just use the lock_order as the void * handle for the lock
    return __uvm_record_lock((void*)(long)lock_order, lock_order, mode);
}

static bool fake_unlock_common(uvm_lock_order_t lock_order, uvm_lock_mode_t mode, bool out_of_order)
{
    // Just use the lock_order as the void * handle for the lock
    return __uvm_record_unlock((void*)(long)lock_order, lock_order, mode, out_of_order);
}

static bool fake_unlock(uvm_lock_order_t lock_order, uvm_lock_mode_t mode)
{
    return fake_unlock_common(lock_order, mode, false);
}

static bool fake_unlock_out_of_order(uvm_lock_order_t lock_order, uvm_lock_mode_t mode)
{
    return fake_unlock_common(lock_order, mode, true);
}

static bool fake_downgrade(uvm_lock_order_t lock_order)
{
    // Just use the lock_order as the void * handle for the lock
    return __uvm_record_downgrade((void*)(long)lock_order, lock_order);
}

static bool fake_check_locked(uvm_lock_order_t lock_order, uvm_lock_mode_t mode)
{
    return __uvm_check_locked((void*)(long)lock_order, lock_order, mode);
}

// TODO: Bug 1799173: The lock asserts verify that the RM GPU lock isn't taken
//       with the VA space lock in exclusive mode, and that the RM GPU lock
//       isn't taken with mmap_sem held in any mode. Hack around this in the
//       test to enable the checks until we figure out something better.
static bool skip_lock(uvm_lock_order_t lock_order, uvm_lock_mode_t mode)
{
    if (lock_order == UVM_LOCK_ORDER_RM_GPUS)
        return mode == UVM_LOCK_MODE_EXCLUSIVE;

    return lock_order == UVM_LOCK_ORDER_MMAP_SEM;
}

static NV_STATUS test_all_locks_from(uvm_lock_order_t from_lock_order)
{
    NvU32 exclusive;
    uvm_lock_mode_t mode;
    NvU32 out_of_order;
    NvU32 lock_order;

    TEST_CHECK_RET(from_lock_order != UVM_LOCK_ORDER_INVALID);

    for (out_of_order = 0; out_of_order < 2; ++out_of_order) {
        for (exclusive = 0; exclusive < 2; ++exclusive) {
            mode = exclusive ? UVM_LOCK_MODE_EXCLUSIVE : UVM_LOCK_MODE_SHARED;

            for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                TEST_CHECK_RET(__uvm_check_unlocked_order(lock_order));
                TEST_CHECK_RET(__uvm_check_lockable_order(lock_order));
            }

            for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                if (skip_lock(lock_order, mode))
                    continue;
                TEST_CHECK_RET(fake_lock(lock_order, mode));
            }

            if (!skip_lock(from_lock_order, mode)) {
                TEST_CHECK_RET(!__uvm_check_unlocked_order(from_lock_order));
                TEST_CHECK_RET(!__uvm_check_lockable_order(from_lock_order));
            }

            for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                if (skip_lock(lock_order, mode))
                    continue;
                TEST_CHECK_RET(fake_check_locked(lock_order, mode));
            }

            for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                if (skip_lock(lock_order, mode))
                    continue;
                TEST_CHECK_RET(fake_check_locked(lock_order, UVM_LOCK_MODE_ANY));
            }

            if (out_of_order == 0) {
                for (lock_order = UVM_LOCK_ORDER_COUNT - 1; lock_order != from_lock_order - 1; --lock_order) {
                    if (skip_lock(lock_order, mode))
                        continue;
                    TEST_CHECK_RET(fake_unlock(lock_order, mode));
                }
            }
            else {
                for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                    if (skip_lock(lock_order, mode))
                        continue;
                    TEST_CHECK_RET(fake_unlock_out_of_order(lock_order, mode));
                }
            }

            for (lock_order = from_lock_order; lock_order < UVM_LOCK_ORDER_COUNT; ++lock_order) {
                if (skip_lock(lock_order, mode))
                    continue;
                TEST_CHECK_RET(__uvm_check_unlocked_order(lock_order));
                TEST_CHECK_RET(__uvm_check_lockable_order(lock_order));
            }
        }
    }

    return NV_OK;
}

NV_STATUS uvm8_test_lock_sanity(UVM_TEST_LOCK_SANITY_PARAMS *params, struct file *filp)
{
    int first_lock  = UVM_LOCK_ORDER_INVALID + 1;
    int second_lock = UVM_LOCK_ORDER_INVALID + 2;

    // The test needs all locks to be released initially
    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    TEST_CHECK_RET(test_all_locks_from(UVM_LOCK_ORDER_INVALID + 1) == NV_OK);

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(test_all_locks_from(first_lock + 1) == NV_OK);
    TEST_CHECK_RET(fake_unlock(first_lock, UVM_LOCK_MODE_SHARED));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(test_all_locks_from(second_lock + 1) == NV_OK);
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking a lock w/o locking any lock at all
    TEST_CHECK_RET(!fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking a different lock than locked
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking a different instance of a lock than locked
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!__uvm_record_unlock(NULL, first_lock, UVM_LOCK_MODE_EXCLUSIVE, false));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking with different mode
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_unlock(first_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(!fake_unlock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking in different order than locked
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock_out_of_order(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Unlocking in different order than locked (not necessarily incorrect, but
    // commonly pointing to issues)
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_unlock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Locking in wrong order
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Locking the same order twice (lock tracking doesn't support this case although
    // it's not necessarily incorrect)
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Nothing locked
    TEST_CHECK_RET(!fake_check_locked(second_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(!fake_check_locked(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_check_locked(second_lock, UVM_LOCK_MODE_ANY));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Expecting exclusive while locked as shared
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(!fake_check_locked(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_SHARED));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Expecting shared while locked as exclusive
    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_check_locked(second_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_EXCLUSIVE));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Wrong instance of a lock held
    TEST_CHECK_RET(__uvm_record_lock(NULL, first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_check_locked(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(__uvm_record_unlock(NULL, first_lock, UVM_LOCK_MODE_EXCLUSIVE, false));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    TEST_CHECK_RET(fake_lock(second_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(!__uvm_thread_check_all_unlocked());
    TEST_CHECK_RET(fake_unlock(second_lock, UVM_LOCK_MODE_SHARED));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Lock downgrade
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_check_locked(first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(fake_check_locked(first_lock, UVM_LOCK_MODE_ANY));
    TEST_CHECK_RET(fake_downgrade(first_lock));
    TEST_CHECK_RET(fake_check_locked(first_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(fake_check_locked(first_lock, UVM_LOCK_MODE_ANY));

    // Can't downgrade twice
    TEST_CHECK_RET(!fake_downgrade(first_lock));
    TEST_CHECK_RET(fake_check_locked(first_lock, UVM_LOCK_MODE_ANY));
    TEST_CHECK_RET(fake_unlock(first_lock, UVM_LOCK_MODE_SHARED));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Downgrading a lock w/o locking any lock at all
    TEST_CHECK_RET(!fake_downgrade(first_lock));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Wrong instance of lock to downgrade
    TEST_CHECK_RET(__uvm_record_lock(NULL, first_lock, UVM_LOCK_MODE_EXCLUSIVE));
    TEST_CHECK_RET(!fake_downgrade(first_lock));
    TEST_CHECK_RET(__uvm_record_unlock(NULL, first_lock, UVM_LOCK_MODE_EXCLUSIVE, false));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    // Downgrading a lock that was acquired as shared
    TEST_CHECK_RET(fake_lock(first_lock, UVM_LOCK_MODE_SHARED));
    TEST_CHECK_RET(!fake_downgrade(first_lock));
    TEST_CHECK_RET(fake_unlock(first_lock, UVM_LOCK_MODE_SHARED));

    TEST_CHECK_RET(__uvm_thread_check_all_unlocked());

    return NV_OK;
}
