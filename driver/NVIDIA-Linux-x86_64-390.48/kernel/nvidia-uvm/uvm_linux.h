/*******************************************************************************
    Copyright (c) 2013-2016 NVIDIA Corporation

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

//
// uvm_linux.h
//
// This file, along with conftest.h and umv_linux.c, helps to insulate
// the (out-of-tree) UVM driver from changes to the upstream Linux kernel.
//
//

#ifndef _UVM_LINUX_H
#define _UVM_LINUX_H

#include "nv-misc.h"
#include "nvtypes.h"

#define NV_BUILD_MODULE_INSTANCES 0
#include "nv-linux.h"

#if defined(NV_LINUX_LOG2_H_PRESENT)
#include <linux/log2.h>
#endif
#if defined(NV_PRIO_TREE_PRESENT)
#include <linux/prio_tree.h>
#endif

#include <linux/rwsem.h>
#include <linux/rbtree.h>
#include <asm/current.h>

#include <linux/random.h>           /* get_random_bytes()               */
#include <linux/radix-tree.h>       /* Linux kernel radix tree          */

#include <linux/file.h>             /* fget()                           */

#include <linux/percpu.h>

#if defined(NV_LINUX_PRINTK_H_PRESENT)
#include <linux/printk.h>
#endif

#if defined(NV_LINUX_RATELIMIT_H_PRESENT)
#include <linux/ratelimit.h>
#endif

#if defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT)
#include <asm/powernv.h>
#endif

#if defined(NV_LINUX_SCHED_TASK_STACK_H_PRESENT)
#include <linux/sched/task_stack.h>
#endif

// TODO: Bug 1772628: remove the "defined(NV_BUILD_SUPPORTS_HMM)" part,
// once the HMM (Heterogeneous Memory Management Linux kernel feature) patch
// gets submitted to the Linux kernel.
//
// Until HMM is part of the upstream kernel, NV_BUILD_SUPPORTS_HMM will not be
// defined in the source code. However, our kernel module build allows you to
// specify this, via:
//     "make modules NV_BUILD_SUPPORTS_HMM=1"
//
#if defined(CONFIG_HMM) && defined(NV_BUILD_SUPPORTS_HMM)
    #include <linux/hmm.h>
    #define UVM_IS_CONFIG_HMM() 1
#else
    #define UVM_IS_CONFIG_HMM() 0
#endif

// See bug 1707453 for further details about setting the minimum kernel version.
#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 32)
#  error This driver does not support kernels older than 2.6.32!
#endif

#if !defined(VM_RESERVED)
#define VM_RESERVED    0x00000000
#endif
#if !defined(VM_DONTEXPAND)
#define VM_DONTEXPAND  0x00000000
#endif
#if !defined(VM_DONTDUMP)
#define VM_DONTDUMP    0x00000000
#endif
#if !defined(VM_MIXEDMAP)
#define VM_MIXEDMAP    0x00000000
#endif

// USHORT_MAX was renamed USHRT_MAX in 2.6.35 via 4be929be34f9bdeffa40d815d32d7d60d2c7f03b
#if !defined(USHRT_MAX)
    #define USHRT_MAX USHORT_MAX
#endif

#define NV_UVM_FENCE()   mb()

//
// printk.h already defined pr_fmt, so we have to redefine it so the pr_*
// routines pick up our version
//
#undef pr_fmt
#define NVIDIA_UVM_PRETTY_PRINTING_PREFIX "nvidia-uvm: "
#define pr_fmt(fmt) NVIDIA_UVM_PRETTY_PRINTING_PREFIX fmt

// Dummy printing function that maintains syntax and format specifier checking
// but doesn't print anything and doesn't evaluate the print parameters. This is
// roughly equivalent to the kernel's no_printk function. We use this instead
// because:
// 1) no_printk was not available until 2.6.36
// 2) Until 4.5 no_printk was implemented as a static function, meaning its
//    parameters were always evaluated
#define UVM_NO_PRINT(fmt, ...)          \
    do {                                \
        if (0)                          \
            printk(fmt, ##__VA_ARGS__); \
    } while(0)

// printk_ratelimited was added in 2.6.33 via commit
// 8a64f336bc1d4aa203b138d29d5a9c414a9fbb47. If not available, we prefer not
// printing anything since it's supposed to be rate-limited.
#if !defined(printk_ratelimited)
    #define printk_ratelimited UVM_NO_PRINT
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(3,8,0)
    // Just too much compilation trouble with the rate-limiting printk feature
    // until about k3.8. Because the non-rate-limited printing will cause
    // surprises and problems, just turn it off entirely in this situation.
    //
    #undef pr_debug_ratelimited
    #define pr_debug_ratelimited UVM_NO_PRINT
#endif

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
#if !defined(pmd_large)
#define pmd_large(_pmd) \
    ((pmd_val(_pmd) & (_PAGE_PSE|_PAGE_PRESENT)) == (_PAGE_PSE|_PAGE_PRESENT))
#endif
#endif /* defined(NVCPU_X86) || defined(NVCPU_X86_64) */

#if !defined(NV_VMWARE)
#define NV_GET_PAGE_COUNT(page_ptr) \
  ((unsigned int)page_count(NV_GET_PAGE_STRUCT(page_ptr->phys_addr)))
#define NV_GET_PAGE_FLAGS(page_ptr) \
  (NV_GET_PAGE_STRUCT(page_ptr->phys_addr)->flags)
#define NV_LOCK_PAGE(ptr_ptr) \
  SetPageReserved(NV_GET_PAGE_STRUCT(page_ptr->phys_addr))
#define NV_UNLOCK_PAGE(page_ptr) \
  ClearPageReserved(NV_GET_PAGE_STRUCT(page_ptr->phys_addr))
#endif

#if !defined(GFP_DMA32)
/*
 * GFP_DMA32 is similar to GFP_DMA, but instructs the Linux zone
 * allocator to allocate memory from the first 4GB on platforms
 * such as Linux/x86-64; the alternative is to use an IOMMU such
 * as the one implemented with the K8 GART, if available.
 */
#define GFP_DMA32 0
#endif

#if !defined(__GFP_NOWARN)
#define __GFP_NOWARN 0
#endif

#if !defined(__GFP_NORETRY)
#define __GFP_NORETRY 0
#endif

#define NV_UVM_GFP_FLAGS (GFP_KERNEL | __GFP_NORETRY)

#if defined(NV_VM_INSERT_PAGE_PRESENT)
#define NV_VM_INSERT_PAGE(vma, addr, page) \
    vm_insert_page(vma, addr, page)
#endif

#if defined(NV_REMAP_PFN_RANGE_PRESENT)
#define NV_REMAP_PAGE_RANGE(from, offset, x...) \
    remap_pfn_range(vma, from, ((offset) >> PAGE_SHIFT), x)
#elif defined(NV_REMAP_PAGE_RANGE_PRESENT)
#if (NV_REMAP_PAGE_RANGE_ARGUMENT_COUNT == 5)
#define NV_REMAP_PAGE_RANGE(x...) remap_page_range(vma, x)
#elif (NV_REMAP_PAGE_RANGE_ARGUMENT_COUNT == 4)
#define NV_REMAP_PAGE_RANGE(x...) remap_page_range(x)
#else
#error "NV_REMAP_PAGE_RANGE_ARGUMENT_COUNT value unrecognized!"
#endif
#else
#error "NV_REMAP_PAGE_RANGE() undefined!"
#endif

#if !defined(NV_ADDRESS_SPACE_INIT_ONCE_PRESENT)
    void address_space_init_once(struct address_space *mapping);
#endif

#if !defined(NV_FATAL_SIGNAL_PENDING_PRESENT)
    static inline int __fatal_signal_pending(struct task_struct *p)
    {
        return unlikely(sigismember(&p->pending.signal, SIGKILL));
    }

    static inline int fatal_signal_pending(struct task_struct *p)
    {
        return signal_pending(p) && __fatal_signal_pending(p);
    }
#endif

// Develop builds define DEBUG but enable optimization
#if defined(DEBUG) && !defined(NVIDIA_UVM_DEVELOP)
  // Wrappers for functions not building correctly without optimizations on,
  // implemented in uvm_debug_optimized.c. Notably the file is only built for
  // debug builds, not develop or release builds.

  // Unoptimized builds of atomic_xchg() hit a BUILD_BUG() on arm64 as it relies
  // on __xchg being completely inlined:
  //   /usr/src/linux-3.12.19/arch/arm64/include/asm/cmpxchg.h:67:3: note: in expansion of macro 'BUILD_BUG'
  //
  // Powerppc hits a similar issue, but ends up with an undefined symbol:
  //   WARNING: "__xchg_called_with_bad_pointer" [...] undefined!
  int nv_atomic_xchg(atomic_t *val, int new);

  // Same problem as atomic_xchg() on powerppc:
  //   WARNING: "__cmpxchg_called_with_bad_pointer" [...] undefined!
  int nv_atomic_cmpxchg(atomic_t *val, int old, int new);

  // Same problem as atomic_xchg() on powerppc:
  //   WARNING: "__cmpxchg_called_with_bad_pointer" [...] undefined!
  long nv_atomic_long_cmpxchg(atomic_long_t *val, long old, long new);

  // This Linux kernel commit:
  // 2016-08-30  0d025d271e55f3de21f0aaaf54b42d20404d2b23
  // leads to build failures on x86_64, when compiling without optimization. Avoid
  // that problem, by providing our own builds of copy_from_user / copy_to_user,
  // for debug (non-optimized) UVM builds. Those are accessed via these
  // nv_copy_to/from_user wrapper functions.
  //
  // Bug 1849583 has further details.
  unsigned long nv_copy_from_user(void *to, const void __user *from, unsigned long n);
  unsigned long nv_copy_to_user(void __user *to, const void *from, unsigned long n);

#else
  #define nv_atomic_xchg            atomic_xchg
  #define nv_atomic_cmpxchg         atomic_cmpxchg
  #define nv_atomic_long_cmpxchg    atomic_long_cmpxchg
  #define nv_copy_to_user           copy_to_user
  #define nv_copy_from_user         copy_from_user
#endif

#if defined(NV_ATOMIC64_PRESENT)
typedef atomic64_t NV_ATOMIC64;
#define NV_ATOMIC64_INC(data)           atomic64_inc(&(data))
#define NV_ATOMIC64_SET(data,val)       atomic64_set(&(data), (val))
#define NV_ATOMIC64_READ(data)          atomic64_read(&(data))
#else
#warning "atomic64_t unavailable, demoting to atomic_t!"
typedef atomic_t NV_ATOMIC64;
#define NV_ATOMIC64_INC(data)           atomic_inc(&(data))
#define NV_ATOMIC64_SET(data,val)       atomic_set(&(data), (val))
#define NV_ATOMIC64_READ(data)          atomic_read(&(data))
#endif

#ifndef NV_ALIGN_DOWN
#define NV_ALIGN_DOWN(v,g) ((v) & ~((g) - 1))
#endif

//
// This provides a value that can be used where vmf->flags would normally
// be used, but on older kernels that do not have a vmf, nor FAULT_FLAG_*
// definitions:
//
#define FAULT_FLAG_FROM_OLD_KERNEL     0x80000000

#if defined(NV_FAULT_FLAG_PRESENT)
#define NV_FAULT_FLAG_WRITE            FAULT_FLAG_WRITE
#else
#define NV_FAULT_FLAG_WRITE            0x01
#endif

#if !defined(NV_KBASENAME_PRESENT)
static inline const char *kbasename(const char *str)
{
    const char *p = strrchr(str, '/');
    if (!p)
        return str;
    return p + 1;
}
#endif

#if defined(NVCPU_X86)
/* Some old IA32 kernels don't have 64/64 division routines,
 * they only support 64/32 division with do_div(). */
static inline uint64_t NV_DIV64(uint64_t dividend, uint64_t divisor, uint64_t *remainder)
{
    /* do_div() only accepts a 32-bit divisor */
    *remainder = do_div(dividend, (uint32_t)divisor);

    /* do_div() modifies the dividend in-place */
    return dividend;
}
#else
/* All other 32/64-bit kernels we support (including non-x86 kernels) support
 * 64/64 division. */
static inline uint64_t NV_DIV64(uint64_t dividend, uint64_t divisor, uint64_t *remainder)
{
    *remainder = dividend % divisor;

    return dividend / divisor;
}
#endif

#if defined(CLOCK_MONOTONIC_RAW)
/* Return a nanosecond-precise value */
static inline NvU64 NV_GETTIME(void)
{
    struct timespec ts = {0};

    getrawmonotonic(&ts);

    /* Wraps around every 583 years */
    return (ts.tv_sec * 1000000000ULL + ts.tv_nsec);
}
#else
/* We can only return a microsecond-precise value with the
 * available non-GPL symbols. */
static inline NvU64 NV_GETTIME(void)
{
    struct timeval tv = {0};

    do_gettimeofday(&tv);

    return (tv.tv_sec * 1000000000ULL + tv.tv_usec * 1000ULL);
}
#endif

#if !defined(ilog2)
    static inline int NV_ILOG2_U32(u32 n)
    {
        return fls(n) - 1;
    }
    static inline int NV_ILOG2_U64(u64 n)
    {
        return fls64(n) - 1;
    }
    #define ilog2(n) (sizeof(n) <= 4 ? NV_ILOG2_U32(n) : NV_ILOG2_U64(n))
#endif

// for_each_bit added in 2.6.24 via commit 3e037454bcfa4b187e8293d2121bd8c0f5a5c31c
// later renamed in 2.6.34 via commit 984b3f5746ed2cde3d184651dabf26980f2b66e5
#if !defined(for_each_set_bit)
    #define for_each_set_bit(bit, addr, size) for_each_bit((bit), (addr), (size))
#endif

// for_each_set_bit_cont was added in 3.2 via 1e2ad28f80b4e155678259238f51edebc19e4014
// It was renamed to for_each_set_bit_from in 3.3 via 307b1cd7ecd7f3dc5ce3d3860957f034f0abe4df
#if !defined(for_each_set_bit_from)
    #define for_each_set_bit_from(bit, addr, size)              \
        for ((bit) = find_next_bit((addr), (size), (bit));      \
             (bit) < (size);                                    \
             (bit) = find_next_bit((addr), (size), (bit) + 1))
#endif

// for_each_clear_bit and for_each_clear_bit_from were added in 3.10 via
// 03f4a8226c2f9c14361f75848d1e93139bab90c4
#if !defined(for_each_clear_bit)
    #define for_each_clear_bit(bit, addr, size)                     \
        for ((bit) = find_first_zero_bit((addr), (size));           \
             (bit) < (size);                                        \
             (bit) = find_next_zero_bit((addr), (size), (bit) + 1))
#endif

#if !defined(for_each_clear_bit_from)
    #define for_each_clear_bit_from(bit, addr, size)                \
        for ((bit) = find_next_zero_bit((addr), (size), (bit));     \
             (bit) < (size);                                        \
             (bit) = find_next_zero_bit((addr), (size), (bit) + 1))
#endif

// bitmap_clear was added in 2.6.33 via commit c1a2a962a2ad103846e7950b4591471fabecece7
#if !defined(NV_BITMAP_CLEAR_PRESENT)
    static inline void bitmap_clear(unsigned long *map, unsigned int start, int len)
    {
        unsigned int index = start;
        for_each_set_bit_from(index, map, start + len)
            __clear_bit(index, map);
    }

    static inline void bitmap_set(unsigned long *map, unsigned int start, int len)
    {
        unsigned int index = start;
        for_each_clear_bit_from(index, map, start + len)
            __set_bit(index, map);
    }
#endif

// smp_mb__before_atomic was added in 3.16, provide a fallback
#ifndef smp_mb__before_atomic
  #if NVCPU_IS_X86 || NVCPU_IS_X86_64
    // That's what the kernel does for x86
    #define smp_mb__before_atomic() barrier()
  #else
    // That's what the kernel does for at least arm32, arm64 and powerpc as of 4.3
    #define smp_mb__before_atomic() smp_mb()
  #endif
#endif

// smp_mb__after_atomic was added in 3.16, provide a fallback
#ifndef smp_mb__after_atomic
  #if NVCPU_IS_X86 || NVCPU_IS_X86_64
    // That's what the kernel does for x86
    #define smp_mb__after_atomic() barrier()
  #else
    // That's what the kernel does for at least arm32, arm64 and powerpc as of 4.3
    #define smp_mb__after_atomic() smp_mb()
  #endif
#endif

// Added in 2.6.24
#ifndef ACCESS_ONCE
  #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

// WRITE_ONCE/READ_ONCE have incompatible definitions across versions, which produces warnings.
// Therefore, we define our own macros
#define UVM_WRITE_ONCE(x, val) (ACCESS_ONCE(x) = (val))
#define UVM_READ_ONCE(x) ACCESS_ONCE(x)

// Added in 3.11
#ifndef PAGE_ALIGNED
    #define PAGE_ALIGNED(addr) (((addr) & (PAGE_SIZE - 1)) == 0)
#endif

// Added in 2.6.37 via commit e1ca7788dec6773b1a2bce51b7141948f2b8bccf
#if !defined(NV_VZALLOC_PRESENT)
    static inline void *vzalloc(unsigned long size)
    {
        void *p = vmalloc(size);
        if (p)
            memset(p, 0, size);
        return p;
    }
#endif

// Changed in 3.17 via commit 743162013d40ca612b4cb53d3a200dff2d9ab26e
#if (NV_WAIT_ON_BIT_LOCK_ARGUMENT_COUNT == 3)
    #define UVM_WAIT_ON_BIT_LOCK(word, bit, mode) \
        wait_on_bit_lock(word, bit, mode)
#elif (NV_WAIT_ON_BIT_LOCK_ARGUMENT_COUNT == 4)
    static __sched int uvm_bit_wait(void *word)
    {
        if (signal_pending_state(current->state, current))
            return 1;
        schedule();
        return 0;
    }
    #define UVM_WAIT_ON_BIT_LOCK(word, bit, mode) \
        wait_on_bit_lock(word, bit, uvm_bit_wait, mode)
#else
#error "Unknown number of arguments"
#endif

static void uvm_init_radix_tree_preloadable(struct radix_tree_root *tree)
{
    // GFP_NOWAIT, or some combination of flags that avoids setting
    // __GFP_DIRECT_RECLAIM (__GFP_WAIT prior to commit
    // d0164adc89f6bb374d304ffcc375c6d2652fe67d from Nov 2015), is required for
    // using radix_tree_preload() for the tree.
    INIT_RADIX_TREE(tree, GFP_NOWAIT);
}

#if !defined(NV_RADIX_TREE_EMPTY_PRESENT)
static bool radix_tree_empty(struct radix_tree_root *tree)
{
    void *dummy;
    return radix_tree_gang_lookup(tree, &dummy, 0, 1) == 0;
}
#endif

// The radix tree root parameter was added to radix_tree_replace_slot in 4.10.
// That same change moved radix_tree_replace_slot from a header-only
// implementation to a .c file, but the symbol wasn't exported until later so
// we cannot use the function on 4.10. UVM uses this macro to ensure that
// radix_tree_replace_slot is not called when using that kernel.
#ifndef NV_RADIX_TREE_REPLACE_SLOT_PRESENT
    #define NV_RADIX_TREE_REPLACE_SLOT(...) \
        UVM_ASSERT_MSG(false, "radix_tree_replace_slot cannot be used in 4.10\n");
#else
#if (NV_RADIX_TREE_REPLACE_SLOT_ARGUMENT_COUNT == 2)
    #define NV_RADIX_TREE_REPLACE_SLOT(root, slot, entry) \
        radix_tree_replace_slot((slot), (entry))
#elif  (NV_RADIX_TREE_REPLACE_SLOT_ARGUMENT_COUNT == 3)
    #define NV_RADIX_TREE_REPLACE_SLOT(root, slot, entry) \
        radix_tree_replace_slot((root), (slot), (entry))
#else
#error "Unknown number of arguments"
#endif
#endif

#if !defined(NV_USLEEP_RANGE_PRESENT)
static void __sched usleep_range(unsigned long min, unsigned long max)
{
    unsigned min_msec = min / 1000;
    unsigned max_msec = max / 1000;

    if (min_msec != 0)
        msleep(min_msec);
    else if (max_msec != 0)
        msleep(max_msec);
    else
        msleep(1);
}
#endif

#endif // _UVM_LINUX_H

