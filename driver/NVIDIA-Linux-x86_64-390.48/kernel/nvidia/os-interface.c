/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2017 by NVIDIA Corporation.  All rights reserved.  All
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

#include "nv-gpu-numa.h"

#define MAX_ERROR_STRING 512
static char nv_error_string[MAX_ERROR_STRING];
nv_spinlock_t nv_error_string_lock;

NV_STATUS NV_API_CALL os_disable_console_access(void)
{
    NV_ACQUIRE_CONSOLE_SEM();
    return NV_OK;
}

NV_STATUS NV_API_CALL os_enable_console_access(void)
{
    NV_RELEASE_CONSOLE_SEM();
    return NV_OK;
}

typedef struct semaphore os_mutex_t;

//
// os_alloc_mutex - Allocate the RM mutex
//
//  ppMutex - filled in with pointer to opaque structure to mutex data type
//
NV_STATUS NV_API_CALL os_alloc_mutex
(
    void **ppMutex
)
{
    NV_STATUS rmStatus;
    os_mutex_t *os_mutex;

    rmStatus = os_alloc_mem(ppMutex, sizeof(os_mutex_t));
    if (rmStatus != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate mutex!\n");
        return rmStatus;
    }
    os_mutex = (os_mutex_t *)*ppMutex;
    NV_INIT_MUTEX(os_mutex);

    return NV_OK;
}

//
// os_free_mutex - Free resources associated with mutex allocated
//                via os_alloc_mutex above. 
//
//  pMutex - Pointer to opaque structure to mutex data type
//
void NV_API_CALL os_free_mutex
(
    void  *pMutex
)
{
    os_mutex_t *os_mutex = (os_mutex_t *)pMutex;

    if (os_mutex != NULL)
    {
        os_free_mem(pMutex);
    }
}

//
//  pMutex - Pointer to opaque structure to mutex data type
//

NV_STATUS NV_API_CALL os_acquire_mutex
(
    void  *pMutex
)
{
    os_mutex_t *os_mutex = (os_mutex_t *)pMutex;

    if (!NV_MAY_SLEEP())
    {
        return NV_ERR_INVALID_REQUEST;
    }
    down(os_mutex);

    return NV_OK;
}

NV_STATUS NV_API_CALL os_cond_acquire_mutex
(
    void * pMutex
)
{
    os_mutex_t *os_mutex = (os_mutex_t *)pMutex;
    if (!NV_MAY_SLEEP())
    {
        return NV_ERR_INVALID_REQUEST;
    }

    if (down_trylock(os_mutex))
    {
        return NV_ERR_TIMEOUT_RETRY;
    }

    return NV_OK;
}


void NV_API_CALL os_release_mutex
(
    void *pMutex
)
{
    os_mutex_t *os_mutex = (os_mutex_t *)pMutex;
    up(os_mutex);
}

typedef struct semaphore os_semaphore_t;


void* NV_API_CALL os_alloc_semaphore
(
    NvU32 initialValue
)
{
    NV_STATUS rmStatus;
    os_semaphore_t *os_sema;

    rmStatus = os_alloc_mem((void *)&os_sema, sizeof(os_semaphore_t));
    if (rmStatus != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate semaphore!\n");
        return NULL;
    }

    NV_INIT_SEMA(os_sema, initialValue);

    return (void *)os_sema;
}

void NV_API_CALL os_free_semaphore
(
    void *pSema
)
{
    os_semaphore_t *os_sema = (os_semaphore_t *)pSema;

    os_free_mem(os_sema);
}

NV_STATUS NV_API_CALL os_acquire_semaphore
(
    void *pSema
)
{
    os_semaphore_t *os_sema = (os_semaphore_t *)pSema;

    if (!NV_MAY_SLEEP())
    {
        return NV_ERR_INVALID_REQUEST;
    }
    down(os_sema);
    return NV_OK;
}

NV_STATUS NV_API_CALL os_release_semaphore
(
    void *pSema
)
{
    os_semaphore_t *os_sema = (os_semaphore_t *)pSema;
    up(os_sema);
    return NV_OK;
}

BOOL NV_API_CALL os_semaphore_may_sleep(void)
{
    return NV_MAY_SLEEP();
}

BOOL NV_API_CALL os_is_isr(void)
{
    return (in_irq());
}

// return TRUE if the caller is the super-user
BOOL NV_API_CALL os_is_administrator(
    PHWINFO pDev
)
{
    return NV_IS_SUSER();
}

NvU32 NV_API_CALL os_get_page_size(void)
{
    return PAGE_SIZE;
}

NvU64 NV_API_CALL os_get_page_mask(void)
{
    return NV_PAGE_MASK;
}

NvU8 NV_API_CALL os_get_page_shift(void)
{
    return PAGE_SHIFT;
}

NvU64 NV_API_CALL os_get_num_phys_pages(void)
{
    return (NvU64)NV_NUM_PHYSPAGES;
}

char* NV_API_CALL os_string_copy(
    char *dst,
    const char *src
)
{
    return strcpy(dst, src);
}

NvU32 NV_API_CALL os_string_length(
    const char* str
)
{
    return strlen(str);
}

NvU32 NV_API_CALL os_strtoul(const char *str, char **endp, NvU32 base)
{
    return (NvU32)simple_strtoul(str, endp, base);
}

NvS32 NV_API_CALL os_string_compare(const char *str1, const char *str2)
{
    return strcmp(str1, str2);
}

NvU8* NV_API_CALL os_mem_copy(
    NvU8       *dst,
    const NvU8 *src,
    NvU32       length
)
{
#if defined(CONFIG_CC_OPTIMIZE_FOR_SIZE)
    /*
     * When the kernel is configured with CC_OPTIMIZE_FOR_SIZE=y, Kbuild uses
     * -Os universally. With -Os, GCC will aggressively inline builtins, even
     * if -fno-builtin is specified, including memcpy with a tiny byte-copy
     * loop on x86 (rep movsb). This is horrible for performance - a strict
     * dword copy is much faster - so when we detect this case, just provide
     * our own implementation.
     */
    NvU8 *ret = dst;
    NvU32 dwords, bytes = length;

    if ((length >= 128) &&
        (((NvUPtr)dst & 3) == 0) & (((NvUPtr)src & 3) == 0))
    {
        dwords = (length / sizeof(NvU32));
        bytes = (length % sizeof(NvU32));

        while (dwords != 0)
        {
            *(NvU32 *)dst = *(const NvU32 *)src;
            dst += sizeof(NvU32);
            src += sizeof(NvU32);
            dwords--;
        }
    }

    while (bytes != 0)
    {
        *dst = *src;
        dst++;
        src++;
        bytes--;
    }

    return ret;
#else
    /*
     * Generally speaking, the kernel-provided memcpy will be the fastest,
     * (optimized much better for the target architecture than the above
     * loop), so we want to use that whenever we can get to it.
     */
    return memcpy(dst, src, length);
#endif
}

NV_STATUS NV_API_CALL os_memcpy_from_user(
    void       *to,
    const void *from,
    NvU32       n
)
{
    return (NV_COPY_FROM_USER(to, from, n) ? NV_ERR_INVALID_ADDRESS : NV_OK);
}

NV_STATUS NV_API_CALL os_memcpy_to_user(
    void       *to,
    const void *from,
    NvU32       n
)
{
    return (NV_COPY_TO_USER(to, from, n) ? NV_ERR_INVALID_ADDRESS : NV_OK);
}

void* NV_API_CALL os_mem_set(
    void  *dst,
    NvU8   c,
    NvU32  length
)
{
    return memset(dst, (int)c, length);
}

NvS32 NV_API_CALL os_mem_cmp(
    const NvU8 *buf0,
    const NvU8* buf1,
    NvU32 length
)
{
    return memcmp(buf0, buf1, length);
}


/*
 * Operating System Memory Functions
 *
 * There are 2 interesting aspects of resource manager memory allocations
 * that need special consideration on Linux:
 *
 * 1. They are typically very large, (e.g. single allocations of 164KB)
 *
 * 2. The resource manager assumes that it can safely allocate memory in
 *    interrupt handlers.
 *
 * The first requires that we call vmalloc, the second kmalloc. We decide
 * which one to use at run time, based on the size of the request and the
 * context. Allocations larger than 128KB require vmalloc, in the context
 * of an ISR they fail.
 */

#define KMALLOC_LIMIT 131072
#define VMALLOC_ALLOCATION_SIZE_FLAG (1 << 0)

NV_STATUS NV_API_CALL os_alloc_mem(
    void **address,
    NvU64 size
)
{
    unsigned long alloc_size;

    if (address == NULL)
        return NV_ERR_INVALID_ARGUMENT;

    *address = NULL;
    NV_MEM_TRACKING_PAD_SIZE(size);

    //
    // NV_KMALLOC, nv_vmalloc take an input of 4 bytes in x86. To avoid 
    // truncation and wrong allocation, below check is required.
    //
    alloc_size = size;

    if (alloc_size != size)
        return NV_ERR_INVALID_PARAMETER;

    if (!NV_MAY_SLEEP())
    {
        if (alloc_size <= KMALLOC_LIMIT)
            NV_KMALLOC_ATOMIC(*address, alloc_size);
    }
    else
    {
        if (alloc_size <= KMALLOC_LIMIT)
        {
            NV_KMALLOC(*address, alloc_size);
        }
        if (*address == NULL)
        {
            *address = nv_vmalloc(alloc_size);
            alloc_size |= VMALLOC_ALLOCATION_SIZE_FLAG;
        }
    }

    NV_MEM_TRACKING_HIDE_SIZE(address, alloc_size);

    return ((*address != NULL) ? NV_OK : NV_ERR_NO_MEMORY);
}

void NV_API_CALL os_free_mem(void *address)
{
    NvU32 size;

    NV_MEM_TRACKING_RETRIEVE_SIZE(address, size);

    if (size & VMALLOC_ALLOCATION_SIZE_FLAG)
    {
        size &= ~VMALLOC_ALLOCATION_SIZE_FLAG;
        nv_vfree(address, size);
    }
    else
        NV_KFREE(address, size);
}


/*****************************************************************************
*
*   Name: osGetCurrentTime
*
*****************************************************************************/

NV_STATUS NV_API_CALL os_get_current_time(
    NvU32 *seconds,
    NvU32 *useconds
)
{
    struct timeval tm;

    do_gettimeofday(&tm);

    *seconds = tm.tv_sec;
    *useconds = tm.tv_usec;

    return NV_OK;
}

#if BITS_PER_LONG >= 64

void NV_API_CALL os_get_current_tick(NvU64 *nseconds)
{
    struct timespec ts;

    jiffies_to_timespec(jiffies, &ts);

    *nseconds = ((NvU64)ts.tv_sec * NSEC_PER_SEC + (NvU64)ts.tv_nsec);
}

#else

void NV_API_CALL os_get_current_tick(NvU64 *nseconds)
{
    /*
     * 'jiffies' overflows regularly on 32-bit builds (unsigned long is 4 bytes
     * instead of 8 bytes), so it's unwise to build a tick counter on it, since
     * the rest of the Resman assumes the 'tick' returned from this function is
     * monotonically increasing and never overflows.
     *
     * Instead, use the previous implementation that we've lived with since the
     * beginning, which uses system clock time to calculate the tick. This is
     * subject to problems if the system clock time changes dramatically
     * (more than a second or so) while the Resman is actively tracking a
     * timeout.
     */
    NvU32 seconds, useconds;

    (void) os_get_current_time(&seconds, &useconds);

    *nseconds = ((NvU64)seconds * NSEC_PER_SEC +
                 (NvU64)useconds * NSEC_PER_USEC);
}

#endif

//---------------------------------------------------------------------------
//
//  Misc services.
//
//---------------------------------------------------------------------------

#define NV_MSECS_PER_JIFFIE         (1000 / HZ)
#define NV_MSECS_TO_JIFFIES(msec)   ((msec) * HZ / 1000)
#define NV_USECS_PER_JIFFIE         (1000000 / HZ)
#define NV_USECS_TO_JIFFIES(usec)   ((usec) * HZ / 1000000)

// #define NV_CHECK_DELAY_ACCURACY 1

/*
 * It is generally a bad idea to use udelay() to wait for more than
 * a few milliseconds. Since the caller is most likely not aware of
 * this, we use mdelay() for any full millisecond to be safe.
 */

NV_STATUS NV_API_CALL os_delay_us(NvU32 MicroSeconds)
{
    unsigned long mdelay_safe_msec;
    unsigned long usec;

#ifdef NV_CHECK_DELAY_ACCURACY
    struct timeval tm1, tm2;

    do_gettimeofday(&tm1);
#endif

    if (in_irq() && (MicroSeconds > NV_MAX_ISR_DELAY_US))
        return NV_ERR_GENERIC;
    
    mdelay_safe_msec = MicroSeconds / 1000;
    if (mdelay_safe_msec)
        mdelay(mdelay_safe_msec);

    usec = MicroSeconds % 1000;
    if (usec)
        udelay(usec);

#ifdef NV_CHECK_DELAY_ACCURACY
    do_gettimeofday(&tm2);
    nv_printf(NV_DBG_ERRORS, "NVRM: osDelayUs %d: 0x%x 0x%x\n",
        MicroSeconds, tm2.tv_sec - tm1.tv_sec, tm2.tv_usec - tm1.tv_usec);
#endif

    return NV_OK;
}

/* 
 * On Linux, a jiffie represents the time passed in between two timer
 * interrupts. The number of jiffies per second (HZ) varies across the
 * supported platforms. On i386, where HZ is 100, a timer interrupt is
 * generated every 10ms; the resolution is a lot better on ia64, where
 * HZ is 1024. NV_MSECS_TO_JIFFIES should be accurate independent of
 * the actual value of HZ; any partial jiffies will be 'floor'ed, the
 * remainder will be accounted for with mdelay().
 */

NV_STATUS NV_API_CALL os_delay(NvU32 MilliSeconds)
{
    unsigned long MicroSeconds;
    unsigned long jiffies;
    unsigned long mdelay_safe_msec;
    struct timeval tm_end, tm_aux;
#ifdef NV_CHECK_DELAY_ACCURACY
    struct timeval tm_start;
#endif

    do_gettimeofday(&tm_aux);
#ifdef NV_CHECK_DELAY_ACCURACY
    tm_start = tm_aux;
#endif

    if (in_irq() && (MilliSeconds > NV_MAX_ISR_DELAY_MS))
        return NV_ERR_GENERIC;

    if (!NV_MAY_SLEEP()) 
    {
        mdelay(MilliSeconds);
        return NV_OK;
    }

    MicroSeconds = MilliSeconds * 1000;
    tm_end.tv_usec = MicroSeconds;
    tm_end.tv_sec = 0;
    NV_TIMERADD(&tm_aux, &tm_end, &tm_end);

    /* do we have a full jiffie to wait? */
    jiffies = NV_USECS_TO_JIFFIES(MicroSeconds);

    if (jiffies)
    {
        //
        // If we have at least one full jiffy to wait, give up
        // up the CPU; since we may be rescheduled before
        // the requested timeout has expired, loop until less
        // than a jiffie of the desired delay remains.
        //
        current->state = TASK_INTERRUPTIBLE;
        do
        {
            schedule_timeout(jiffies);
            do_gettimeofday(&tm_aux);
            if (NV_TIMERCMP(&tm_aux, &tm_end, <))
            {
                NV_TIMERSUB(&tm_end, &tm_aux, &tm_aux);
                MicroSeconds = tm_aux.tv_usec + tm_aux.tv_sec * 1000000;
            }
            else
                MicroSeconds = 0;
        } while ((jiffies = NV_USECS_TO_JIFFIES(MicroSeconds)) != 0);
    }

    if (MicroSeconds > 1000)
    {
        mdelay_safe_msec = MicroSeconds / 1000;
        mdelay(mdelay_safe_msec);
        MicroSeconds %= 1000;
    }
    if (MicroSeconds)
    {
        udelay(MicroSeconds);
    }
#ifdef NV_CHECK_DELAY_ACCURACY
    do_gettimeofday(&tm_aux);
    timersub(&tm_aux, &tm_start, &tm_aux);
    nv_printf(NV_DBG_ERRORS, "NVRM: osDelay %dmsec: %d.%06dsec\n",
        MilliSeconds, tm_aux.tv_sec, tm_aux.tv_usec);
#endif

    return NV_OK;
}

NvU64 NV_API_CALL os_get_cpu_frequency(void)
{
    NvU64 cpu_hz = 0;
#if defined(CONFIG_CPU_FREQ)
    cpu_hz = (cpufreq_get(0) * 1000);
#elif (defined(NVCPU_X86) || defined(NVCPU_X86_64))
    NvU64 tsc[2];

    tsc[0] = nv_rdtsc();
    mdelay(250);
    tsc[1] = nv_rdtsc();

    cpu_hz = ((tsc[1] - tsc[0]) * 4);
#endif
    return cpu_hz;
}

NvU32 NV_API_CALL os_get_current_process(void)
{
    return NV_GET_CURRENT_PROCESS();
}

void NV_API_CALL os_get_current_process_name(char *buf, NvU32 len)
{
    task_lock(current);
    strncpy(buf, current->comm, len - 1);
    buf[len - 1] = '\0';
    task_unlock(current);
}

NvU32 NV_API_CALL os_get_current_pasid(void)
{
#if defined(NV_MM_CONTEXT_T_HAS_ID)
    if (current && current->mm)
        return current->mm->context.id;
    else
        return 0;
#else
    // Unsupported. Return "0" as it's invalid PASID
    return 0;
#endif
}

NV_STATUS NV_API_CALL os_get_current_thread(NvU64 *threadId)
{
    if (in_interrupt())
        *threadId = 0;
    else
        *threadId = (NvU64) current->pid;

    return NV_OK;
}

/*******************************************************************************/
/*                                                                             */
/* Debug and logging utilities follow                                          */
/*                                                                             */
/*******************************************************************************/

// The current debug display level (default to maximum debug level)
NvU32 cur_debuglevel = 0xffffffff;

/*
 * The binary core of RM (nv-kernel.o) calls both out_string, and nv_printf.
 */
inline void NV_API_CALL out_string(const char *str)
{
    printk("%s", str);
}

/*
 * nv_printf() prints to the kernel log for the driver.
 * Returns the number of characters written.
 */
int NV_API_CALL nv_printf(NvU32 debuglevel, const char *printf_format, ...)
{
    va_list arglist;
    int chars_written = 0;

    if (debuglevel >= ((cur_debuglevel >> 4) & 0x3))
    {
        va_start(arglist, printf_format);
        chars_written = vprintk(printf_format, arglist);
        va_end(arglist);
    }

    return chars_written;
}

NvS32 NV_API_CALL os_snprintf(char *buf, NvU32 size, const char *fmt, ...)
{
    va_list arglist;
    int chars_written;

    va_start(arglist, fmt);
    chars_written = vsnprintf(buf, size, fmt, arglist);
    va_end(arglist);

    return chars_written;
}

void NV_API_CALL os_log_error(const char *fmt, va_list ap)
{
    unsigned long flags;

    NV_SPIN_LOCK_IRQSAVE(&nv_error_string_lock, flags);

    vsnprintf(nv_error_string, MAX_ERROR_STRING, fmt, ap);
    nv_error_string[MAX_ERROR_STRING - 1] = 0;
    printk(KERN_ERR "%s", nv_error_string);

    NV_SPIN_UNLOCK_IRQRESTORE(&nv_error_string_lock, flags);
}

void NV_API_CALL os_io_write_byte(
    NvU32 address,
    NvU8 value
)
{
    outb(value, address);
}

void NV_API_CALL os_io_write_word(
    NvU32 address,
    NvU16 value
)
{
    outw(value, address);
}

void NV_API_CALL os_io_write_dword(
    NvU32 address,
    NvU32 value
)
{
    outl(value, address);
}

NvU8 NV_API_CALL os_io_read_byte(
    NvU32 address
)
{
    return inb(address);
}

NvU16 NV_API_CALL os_io_read_word(
    NvU32 address
)
{
    return inw(address);
}

NvU32 NV_API_CALL os_io_read_dword(
    NvU32 address
)
{
    return inl(address);
}


static NvBool NV_API_CALL xen_support_fully_virtualized_kernel(void)
{
#if defined(NV_XEN_SUPPORT_FULLY_VIRTUALIZED_KERNEL)
    return (os_is_vgx_hyper());
#endif
    return FALSE;
}

void* NV_API_CALL os_map_kernel_space(
    NvU64 start,
    NvU64 size_bytes,
    NvU32 mode,
    NvU32 memType
)
{
    void *vaddr;
    NvU64 offset_in_page = 0;

    if (!xen_support_fully_virtualized_kernel() && start == 0)
    {
        if (mode != NV_MEMORY_CACHED)
        {
            nv_printf(NV_DBG_ERRORS,
                "NVRM: os_map_kernel_space: won't map address 0x%0llx UC!\n", start);
            return NULL;
        }
        else
            return (void *)PAGE_OFFSET;
    }

    if (!NV_MAY_SLEEP())
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: os_map_kernel_space: can't map 0x%0llx, invalid context!\n", start);
        os_dbg_breakpoint();
        return NULL;
    }

#if defined(NVCPU_X86)
    if (start > 0xffffffff)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: os_map_kernel_space: can't map > 32-bit address 0x%0llx!\n", start);
        os_dbg_breakpoint();
        return NULL;
    }
#endif

    switch (mode)
    {
        case NV_MEMORY_CACHED:
            vaddr = nv_ioremap_cache(start, size_bytes);
            break;
        case NV_MEMORY_WRITECOMBINED:
            vaddr = nv_ioremap_wc(start, size_bytes);
            break;
        case NV_MEMORY_NUMA: 
            vaddr = kmap(NV_GET_PAGE_STRUCT(start));

            // 
            // kmap aligns the busAddress to OS page size before mapping, 
            // so we need to add back the offset after mapping is done. 
            // This fix assumes no more than one compound page is being mapped 
            // in one call. For final fix please refer to bug #2024344
            //
            offset_in_page = start & (PAGE_SIZE - 1);
            vaddr = (void *)(((NvU64)vaddr) + offset_in_page);
            break;
        case NV_MEMORY_UNCACHED:
        case NV_MEMORY_DEFAULT:
            vaddr = nv_ioremap_nocache(start, size_bytes);
            break;
        default:
            nv_printf(NV_DBG_ERRORS,
                "NVRM: os_map_kernel_space: unsupported mode!\n");
            return NULL;
    }

    return vaddr;
}

void NV_API_CALL os_unmap_kernel_space(
    void *addr,
    NvU64 size_bytes
)
{
    if (addr == (void *)PAGE_OFFSET)
        return;

    nv_iounmap(addr, size_bytes);
}

void NV_API_CALL os_unmap_kernel_numa(
    void *addr,
    NvU64 size_bytes
)
{
    kunmap(virt_to_page(addr));
}

// flush the cpu's cache, uni-processor version
NV_STATUS NV_API_CALL os_flush_cpu_cache()
{
    CACHE_FLUSH();
    return NV_OK;
}

// flush the cache of all cpus
NV_STATUS NV_API_CALL os_flush_cpu_cache_all()
{
#if defined(NVCPU_FAMILY_ARM)
    CACHE_FLUSH_ALL();
#if defined(NVCPU_ARM) && defined(NV_OUTER_FLUSH_ALL_PRESENT)
    /* flush the external L2 cache in cortex-A9 and cortex-a15 */
    OUTER_FLUSH_ALL();
#endif
    return NV_OK;
#endif
    return NV_ERR_NOT_SUPPORTED;
}

// Flush and/or invalidate a range of memory in user space.
// start, end are the user virtual addresses
// physStart, physEnd are the corresponding physical addresses
// Start addresses are inclusive, end addresses exclusive
// The flags argument states whether to flush, invalidate, or do both
NV_STATUS NV_API_CALL os_flush_user_cache(NvU64 start, NvU64 end, 
                                          NvU64 physStart, NvU64 physEnd, 
                                          NvU32 flags)
{
#if defined(NVCPU_FAMILY_ARM)
    if (!NV_MAY_SLEEP())
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    if (flags & OS_UNIX_FLUSH_USER_CACHE)
    {
#if defined(NVCPU_AARCH64)
        //
        // The Linux kernel does not export an interface for flushing a range,
        // although it is possible. For now, just flush the entire cache to be
        // safe.
        //
        CACHE_FLUSH_ALL();
#else
        // Flush L1 cache
        __cpuc_flush_dcache_area((void *)(NvU32)start, (NvU32)(end-start));
#if defined(OUTER_FLUSH_RANGE)
        // Now flush L2 cache.
        OUTER_FLUSH_RANGE((NvU32)physStart, (NvU32)physEnd);
#endif
#endif
    }

    if (flags & OS_UNIX_INVALIDATE_USER_CACHE)
    {
        // Invalidate L1/L2 cache
        dma_sync_single_for_device(NULL, (dma_addr_t) physStart, (NvU32)(physEnd - physStart), DMA_FROM_DEVICE);
    }
    return NV_OK;
#else
    return NV_ERR_NOT_SUPPORTED;
#endif
}

void NV_API_CALL os_flush_cpu_write_combine_buffer()
{
    WRITE_COMBINE_FLUSH();
}

// override initial debug level from registry
void NV_API_CALL os_dbg_init(void)
{
    NvU32 new_debuglevel;
    nvidia_stack_t *sp = NULL;

    NV_SPIN_LOCK_INIT(&nv_error_string_lock);

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return;
    }

    if (NV_OK == rm_read_registry_dword(sp, NULL,
                                        "NVreg",
                                        "ResmanDebugLevel",
                                        &new_debuglevel))
    {
        if (new_debuglevel != (NvU32)~0)
            cur_debuglevel = new_debuglevel;
    }

    nv_kmem_cache_free_stack(sp);
}

void NV_API_CALL os_dbg_set_level(NvU32 new_debuglevel)
{
    nv_printf(NV_DBG_SETUP, "NVRM: Changing debuglevel from 0x%x to 0x%x\n",
        cur_debuglevel, new_debuglevel);
    cur_debuglevel = new_debuglevel;
}

NV_STATUS NV_API_CALL os_schedule(void)
{
    if (NV_MAY_SLEEP())
    {
        set_current_state(TASK_INTERRUPTIBLE);
        schedule_timeout(1);
        return NV_OK;
    }
    else
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: os_schedule: Attempted to yield"
                                 " the CPU while in atomic or interrupt"
                                 " context\n");
        return NV_ERR_ILLEGAL_ACTION;
    }
}

static void os_execute_work_item(
    NV_WORKQUEUE_DATA_T *data
)
{
    nv_work_t *work = NV_WORKQUEUE_UNPACK_DATA(data);
    nvidia_stack_t *sp = NULL;

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return;
    }

    rm_execute_work_item(sp, work->data);

    os_free_mem((void *)work);

    nv_kmem_cache_free_stack(sp);
}

NV_STATUS NV_API_CALL os_queue_work_item(
    void *nv_work
)
{
    NV_STATUS status;
    nv_work_t *work;

    status = os_alloc_mem((void **)&work, sizeof(nv_work_t));

    if (NV_OK != status)
        return status;

    work->data = nv_work;

    NV_WORKQUEUE_INIT(&work->task, os_execute_work_item,
                      (void *)work);
    NV_WORKQUEUE_SCHEDULE(&work->task);

    return NV_OK;
}

NV_STATUS NV_API_CALL os_flush_work_queue(void)
{
    if (NV_MAY_SLEEP())
    {
        NV_WORKQUEUE_FLUSH();
        return NV_OK;
    }
    else
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: os_flush_work_queue: attempted to execute passive"
                  "work from an atomic or interrupt context.\n");
        return NV_ERR_ILLEGAL_ACTION;
    }
}

void NV_API_CALL os_dbg_breakpoint(void)
{
#if defined(DEBUG)
  #if defined(CONFIG_X86_REMOTE_DEBUG) || defined(CONFIG_KGDB) || defined(CONFIG_XMON)
    #if defined(NVCPU_X86) || defined(NVCPU_X86_64)
        __asm__ __volatile__ ("int $3");
    #elif defined(NVCPU_ARM)
        __asm__ __volatile__ (".word %c0" :: "i" (KGDB_COMPILED_BREAK));
    #elif defined(NVCPU_AARCH64)
        # warning "Need to implement os_dbg_breakpoint() for aarch64"
    #elif defined(NVCPU_PPC64LE)
        __asm__ __volatile__ ("trap");
    #endif // NVCPU_X86 || NVCPU_X86_64
  #elif defined(CONFIG_KDB)
      KDB_ENTER();
  #endif // CONFIG_X86_REMOTE_DEBUG || CONFIG_KGDB || CONFIG_XMON
#endif // DEBUG
}

NvU32 NV_API_CALL os_get_cpu_number()
{
    NvU32 cpu_id = get_cpu();
    put_cpu();
    return cpu_id;
}

NvU32 NV_API_CALL os_get_cpu_count()
{
    return NV_NUM_CPUS();
}

void NV_API_CALL os_register_compatible_ioctl(NvU32 cmd, NvU32 size)
{
#if defined(NVCPU_X86_64) && defined(CONFIG_IA32_EMULATION) && \
  !defined(NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL)
    unsigned int request = _IOWR(NV_IOCTL_MAGIC, cmd, char[size]);
    register_ioctl32_conversion(request, (void *)sys_ioctl);
#endif
}

void NV_API_CALL os_unregister_compatible_ioctl(NvU32 cmd, NvU32 size)
{
#if defined(NVCPU_X86_64) && defined(CONFIG_IA32_EMULATION) && \
  !defined(NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL)
    unsigned int request = _IOWR(NV_IOCTL_MAGIC, cmd, char[size]);
    unregister_ioctl32_conversion(request);
#endif
}

BOOL NV_API_CALL os_pat_supported(void)
{
    return (nv_pat_mode != NV_PAT_MODE_DISABLED);
}

BOOL NV_API_CALL os_is_efi_enabled(void)
{
    return NV_EFI_ENABLED();
}

BOOL NV_API_CALL os_iommu_is_snooping_enabled(void)
{
    return TRUE;
}

void NV_API_CALL os_get_screen_info(
    NvU64 *pPhysicalAddress,
    NvU16 *pFbWidth,
    NvU16 *pFbHeight,
    NvU16 *pFbDepth,
    NvU16 *pFbPitch
)
{
#if (defined(NVCPU_X86) || defined(NVCPU_X86_64))
    //
    // If there is not a framebuffer console, return 0 size.
    //
    // orig_video_isVGA is set to 1 during early Linux kernel
    // initialization, and then will be set to a value, such as
    // VIDEO_TYPE_VLFB or VIDEO_TYPE_EFI if an fbdev console is used.
    //
    if (screen_info.orig_video_isVGA <= 1)
    {
        *pPhysicalAddress = 0;
        *pFbWidth = *pFbHeight = *pFbDepth = *pFbPitch = 0;
        return;
    }

    *pPhysicalAddress = screen_info.lfb_base;
#if defined(VIDEO_CAPABILITY_64BIT_BASE)
    *pPhysicalAddress |= (NvU64)screen_info.ext_lfb_base << 32;
#endif
    *pFbWidth = screen_info.lfb_width;
    *pFbHeight = screen_info.lfb_height;
    *pFbDepth = screen_info.lfb_depth;
    *pFbPitch = screen_info.lfb_linelength;
#else
    *pPhysicalAddress = 0;
    *pFbWidth = *pFbHeight = *pFbDepth = *pFbPitch = 0;
#endif
}

void NV_API_CALL os_dump_stack()
{
#if defined(DEBUG)
    dump_stack();
#endif
}

typedef struct os_spinlock_s
{
    nv_spinlock_t      lock;
    unsigned long      eflags;
} os_spinlock_t;

NV_STATUS NV_API_CALL os_alloc_spinlock(void **ppSpinlock)
{
    NV_STATUS rmStatus;
    os_spinlock_t *os_spinlock;

    rmStatus = os_alloc_mem(ppSpinlock, sizeof(os_spinlock_t));
    if (rmStatus != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate spinlock!\n");
        return rmStatus;
    }

    os_spinlock = (os_spinlock_t *)*ppSpinlock;
    NV_SPIN_LOCK_INIT(&os_spinlock->lock);
    os_spinlock->eflags = 0;
    return NV_OK;
}

void NV_API_CALL os_free_spinlock(void *pSpinlock)
{
    os_free_mem(pSpinlock);
}

NvU64 NV_API_CALL os_acquire_spinlock(void *pSpinlock)
{
    os_spinlock_t *os_spinlock = (os_spinlock_t *)pSpinlock;
    unsigned long eflags;

    NV_SPIN_LOCK_IRQSAVE(&os_spinlock->lock, eflags);
    os_spinlock->eflags = eflags;

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
    eflags &= X86_EFLAGS_IF;
#elif defined(NVCPU_FAMILY_ARM)
    eflags &= PSR_I_BIT;
#endif
    return eflags;
}

void NV_API_CALL os_release_spinlock(void *pSpinlock, NvU64 oldIrql)
{
    os_spinlock_t *os_spinlock = (os_spinlock_t *)pSpinlock;
    unsigned long eflags;

    eflags = os_spinlock->eflags;
    os_spinlock->eflags = 0;
    NV_SPIN_UNLOCK_IRQRESTORE(&os_spinlock->lock, eflags);
}

NV_STATUS NV_API_CALL os_get_address_space_info(
    NvU64 *userStartAddress,
    NvU64 *userEndAddress,
    NvU64 *kernelStartAddress,
    NvU64 *kernelEndAddress
)
{
    NV_STATUS status = NV_OK;

#if !defined(CONFIG_X86_4G)
    *kernelStartAddress = PAGE_OFFSET;
    *kernelEndAddress = (NvUPtr)0xffffffffffffffffULL;
    *userStartAddress = 0;
    *userEndAddress = TASK_SIZE;
#else
    *kernelStartAddress = 0;
    *kernelEndAddress = 0;   /* invalid */
    *userStartAddress = 0;
    *userEndAddress = 0;     /* invalid */
    status = NV_ERR_NOT_SUPPORTED;
#endif
    return status;
}

#define NV_KERNEL_RELEASE    ((LINUX_VERSION_CODE >> 16) & 0x0ff)
#define NV_KERNEL_VERSION    ((LINUX_VERSION_CODE >> 8)  & 0x0ff)
#define NV_KERNEL_SUBVERSION ((LINUX_VERSION_CODE)       & 0x0ff)

NV_STATUS NV_API_CALL os_get_version_info(os_version_info * pOsVersionInfo)
{
    NV_STATUS status      = NV_OK;

    pOsVersionInfo->os_major_version = NV_KERNEL_RELEASE;
    pOsVersionInfo->os_minor_version = NV_KERNEL_VERSION;
    pOsVersionInfo->os_build_number  = NV_KERNEL_SUBVERSION;

#if defined(UTS_RELEASE)
    do
    {
        char * version_string = NULL;
        status = os_alloc_mem((void **)&version_string,
                              (strlen(UTS_RELEASE) + 1));
        if (status != NV_OK)
        {
            return status;
        }
        strcpy(version_string, UTS_RELEASE);
        pOsVersionInfo->os_build_version_str = version_string;
    }while(0);
#endif

#if defined(UTS_VERSION)
    do
    {
        char * date_string    = NULL;
        status = os_alloc_mem((void **)&date_string, (strlen(UTS_VERSION) + 1));
        if (status != NV_OK)
        {
            return status;
        }
        strcpy(date_string, UTS_VERSION);
        pOsVersionInfo->os_build_date_plus_str = date_string;
    }while(0);
#endif

    return status;
}

NvBool NV_API_CALL os_is_xen_dom0(void)
{
#if defined(NV_DOM0_KERNEL_PRESENT)
    return TRUE;
#else
    return FALSE;
#endif
}

NvBool NV_API_CALL os_is_vgx_hyper(void)
{
#if defined(NV_VGX_HYPER)
    return TRUE;
#else
    return FALSE;
#endif
}

NV_STATUS NV_API_CALL os_inject_vgx_msi(NvU16 guestID, NvU64 msiAddr, NvU32 msiData)
{
#if defined(NV_VGX_HYPER) && defined(NV_DOM0_KERNEL_PRESENT) && \
    defined(NV_XEN_IOEMU_INJECT_MSI)
    int rc = 0;
    rc = xen_ioemu_inject_msi(guestID, msiAddr, msiData);
    if (rc)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: %s: can't inject MSI to guest:%d, addr:0x%x, data:0x%x, err:%d\n",
            __FUNCTION__, guestID, msiAddr, msiData, rc);
        return NV_ERR_OPERATING_SYSTEM; 
    }
    return NV_OK;
#else
    return NV_ERR_NOT_SUPPORTED;  
#endif
}

NvBool NV_API_CALL os_is_grid_supported(void)
{
#if defined(NV_GRID_BUILD)
    return TRUE;
#else
    return FALSE;
#endif
}

void NV_API_CALL os_bug_check(NvU32 bugCode, const char *bugCodeStr)
{
    panic(bugCodeStr);
}

NV_STATUS NV_API_CALL os_get_euid(NvU32 *pSecToken)
{
    *pSecToken = NV_CURRENT_EUID();
    return NV_OK;
}

// These functions are needed only on x86_64 platforms.
#if (defined(NVCPU_X86_64))

static NvBool os_verify_checksum(const NvU8 *pMappedAddr, NvU32 length)
{
    NvU8 sum = 0;
    NvU32 iter = 0;

    for (iter = 0; iter < length; iter++)
        sum += pMappedAddr[iter];

    return sum == 0;
}

#define _VERIFY_SMBIOS3(_pMappedAddr)                        \
        _pMappedAddr &&                                      \
        (os_mem_cmp(_pMappedAddr, "_SM3_", 5) == 0  &&       \
        _pMappedAddr[6] < 32 &&                              \
        _pMappedAddr[6] > 0 &&                               \
        os_verify_checksum(_pMappedAddr, _pMappedAddr[6]))

#define OS_VERIFY_SMBIOS3(pMappedAddr) _VERIFY_SMBIOS3((pMappedAddr))

#define _VERIFY_SMBIOS(_pMappedAddr)                           \
        _pMappedAddr &&                                        \
        (os_mem_cmp(_pMappedAddr, "_SM_", 4) == 0  &&          \
        _pMappedAddr[5] < 32 &&                                \
        _pMappedAddr[5] > 0 &&                                 \
        os_verify_checksum(_pMappedAddr, _pMappedAddr[5]) &&   \
        os_mem_cmp((_pMappedAddr + 16), "_DMI_", 5) == 0  &&   \
        os_verify_checksum((_pMappedAddr + 16), 15))

#define OS_VERIFY_SMBIOS(pMappedAddr) _VERIFY_SMBIOS((pMappedAddr))

#define SMBIOS_LEGACY_BASE 0xF0000
#define SMBIOS_LEGACY_SIZE 0x10000

static NV_STATUS os_get_smbios_header_legacy(NvU64 *pSmbsAddr)
{
    NV_STATUS status = NV_ERR_OPERATING_SYSTEM;
    NvU8 *pMappedAddr = NULL;
    NvU8 *pIterAddr = NULL;

    pMappedAddr = (NvU8*)os_map_kernel_space(SMBIOS_LEGACY_BASE,
                                             SMBIOS_LEGACY_SIZE,
                                             NV_MEMORY_CACHED,
                                             NV_MEMORY_TYPE_SYSTEM);
    if (pMappedAddr == NULL)
    {
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    pIterAddr = pMappedAddr;

    for (; pIterAddr < (pMappedAddr + SMBIOS_LEGACY_SIZE); pIterAddr += 16)
    {
        if (OS_VERIFY_SMBIOS3(pIterAddr))
        {
            *pSmbsAddr = SMBIOS_LEGACY_BASE + (pIterAddr - pMappedAddr);
            status = NV_OK;
            break;
        }

        if (OS_VERIFY_SMBIOS(pIterAddr))
        {
            *pSmbsAddr = SMBIOS_LEGACY_BASE + (pIterAddr - pMappedAddr);
            status = NV_OK;
            break;
        }
    }

    os_unmap_kernel_space(pMappedAddr, SMBIOS_LEGACY_SIZE);

    return status;
}

// This function is needed only if "efi" is enabled.
#if (defined(NV_LINUX_EFI_H_PRESENT) && defined(CONFIG_EFI))
static NV_STATUS os_verify_smbios_header_uefi(NvU64 smbsAddr)
{
    NV_STATUS status = NV_ERR_OBJECT_NOT_FOUND;
    NvU64 start= 0, offset =0 , size = 32;
    NvU8 *pMappedAddr = NULL, *pBufAddr = NULL;

    start = smbsAddr;
    offset = (start & ~OS_PAGE_MASK);
    start &= OS_PAGE_MASK;
    size = ((size + offset + ~OS_PAGE_MASK) & OS_PAGE_MASK);

    pBufAddr = (NvU8*)os_map_kernel_space(start,
                                          size,
                                          NV_MEMORY_CACHED,
                                          NV_MEMORY_TYPE_SYSTEM);
    if (pBufAddr == NULL)
    {
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    pMappedAddr = pBufAddr + offset;

    if (OS_VERIFY_SMBIOS3(pMappedAddr))
    {
        status = NV_OK;
        goto done;
    }

    if (OS_VERIFY_SMBIOS(pMappedAddr))
    {
        status = NV_OK;
    }

done:
    os_unmap_kernel_space(pBufAddr, size);
    return status;
}
#endif

static NV_STATUS os_get_smbios_header_uefi(NvU64 *pSmbsAddr)
{
    NV_STATUS status = NV_ERR_OPERATING_SYSTEM;

// Make sure that efi.h is present before using "struct efi".
#if (defined(NV_LINUX_EFI_H_PRESENT) && defined(CONFIG_EFI))

// Make sure that efi.h has SMBIOS3_TABLE_GUID present.
#if defined(SMBIOS3_TABLE_GUID)
    if (efi.smbios3 != EFI_INVALID_TABLE_ADDR)
    {
        status = os_verify_smbios_header_uefi(efi.smbios3);
        if (status == NV_OK)
        {
            *pSmbsAddr = efi.smbios3;
            return NV_OK;
        }
    }
#endif

    if (efi.smbios != EFI_INVALID_TABLE_ADDR)
    {
        status = os_verify_smbios_header_uefi(efi.smbios);
        if (status == NV_OK)
        {
            *pSmbsAddr = efi.smbios;
            return NV_OK;
        }
    }
#endif

    return status;
}

#endif // #if (defined(NVCPU_X86_64))

// The function locates the SMBIOS entry point.
NV_STATUS NV_API_CALL os_get_smbios_header(NvU64 *pSmbsAddr)
{

#if (!defined(NVCPU_X86_64))
    return NV_ERR_NOT_SUPPORTED;
#else
    NV_STATUS status = NV_OK;

    if (os_is_efi_enabled())
    {
        status = os_get_smbios_header_uefi(pSmbsAddr);
    }
    else
    {
        status = os_get_smbios_header_legacy(pSmbsAddr);
    }

    return status;
#endif
}

NV_STATUS NV_API_CALL os_get_acpi_rsdp_from_uefi
(
    NvU32  *pRsdpAddr
)
{
    NV_STATUS status = NV_ERR_NOT_SUPPORTED;

    if (pRsdpAddr == NULL)
    {
        return NV_ERR_INVALID_STATE;
    }

    *pRsdpAddr = 0;

// Make sure that efi.h is present before using "struct efi".
#if (defined(NV_LINUX_EFI_H_PRESENT) && defined(CONFIG_EFI))

    if (efi.acpi20 != EFI_INVALID_TABLE_ADDR)
    {
        *pRsdpAddr = efi.acpi20;
        status = NV_OK;
    }
    else if (efi.acpi != EFI_INVALID_TABLE_ADDR)
    {
        *pRsdpAddr = efi.acpi;
        status = NV_OK;
    }
    else
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: RSDP Not found!\n");
        status = NV_ERR_OPERATING_SYSTEM;
    }
#endif

    return status;
}

void NV_API_CALL os_add_record_for_crashLog(void *pbuffer, NvU32 size)
{
}

void NV_API_CALL os_delete_record_for_crashLog(void *pbuffer)
{
}

#if !defined(NV_VGPU_KVM_BUILD)
NV_STATUS NV_API_CALL os_call_vgpu_vfio(void *pvgpu_vfio_info, NvU32 cmd_type)
{
    return NV_ERR_NOT_SUPPORTED;
}
#endif

NV_STATUS NV_API_CALL os_numa_memblock_size(NvU64 *memblock_size)
{
    return nv_numa_memblock_size(memblock_size); 
}

NV_STATUS NV_API_CALL os_numa_online_memory
(
    NvS32  node_id, 
    NvU64  region_gpu_addr, 
    NvU64  region_gpu_size, 
    NvU64  ats_base_addr, 
    NvU64  memblock_size,
    NvBool should_probe
)
{
    if (NVreg_EnableUserNUMAManagement)
    {
        return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    return nv_numa_online_memory(node_id, region_gpu_addr, region_gpu_size, 
                                 ats_base_addr, memblock_size, should_probe);
}

void NV_API_CALL os_numa_offline_memory(NvS32 node_id)
{
    nv_numa_offline_memory(node_id);
}

NV_STATUS NV_API_CALL os_alloc_pages_node
(
    NvS32  nid, 
    NvU32  size, 
    NvU32  flag, 
    NvU64 *pAddress
)
{
    NV_STATUS status = NV_ERR_NOT_SUPPORTED;

#if defined(__GFP_THISNODE) && defined(GFP_HIGHUSER_MOVABLE) && \
    defined(__GFP_COMP) && defined(__GFP_NORETRY) && defined(__GFP_NOWARN) && \
    defined(__GFP_ZERO)
    gfp_t gfp_mask; 
    struct page *alloc_addr;
    unsigned int order = get_order(size);

    /*
     * Explanation of flags used: 
     *
     * 1. __GFP_THISNODE:           This will make sure the allocation happens
     *                              on the node specified by nid.
     *
     * 2. GFP_HIGHUSER_MOVABLE:     This makes allocations from ZONE_MOVABLE.
     *
     * 3. __GFP_COMP:               This will make allocations with compound
     *                              pages, which is needed in order to use
     *                              vm_insert_page API.
     *
     * 4. __GFP_NORETRY:            Used to avoid the Linux kernel OOM killer.
     *
     * 5. __GFP_NOWARN:             Used to avoid a WARN_ON in the slowpath if
     *                              the requested order is too large (just fail
     *                              instead).
     *
     * 6. (Optional) __GFP_ZERO:    Used to make kernel scrub-on-allocate.
     *
     * Some of these flags are relatively more recent, with the last of them
     * (GFP_HIGHUSER_MOVABLE) having been added with this Linux kernel commit:
     *
     * 2007-07-17 769848c03895b63e5662eb7e4ec8c4866f7d0183
     *
     * Assume that this feature will only be used on kernels that support all
     * of the needed GFP flags.
     */

    gfp_mask = __GFP_THISNODE | GFP_HIGHUSER_MOVABLE | __GFP_COMP |
               __GFP_NORETRY | __GFP_NOWARN;

    if (flag & NV_ALLOC_PAGES_NODE_SCRUB_ON_ALLOC)
    {
        gfp_mask |= __GFP_ZERO; 
    }

    alloc_addr = alloc_pages_node(nid, gfp_mask, order);
    if (alloc_addr == NULL)
    {
        nv_printf(NV_DBG_INFO,
            "NVRM: alloc_pages_node(node = %d, order = %u) failed\n",
            nid, order);
        status = NV_ERR_NO_MEMORY;
    }
    else
    {
        *pAddress = (NvU64)page_to_phys(alloc_addr);
        status = NV_OK;
    }
#endif // GFP flags

    return status;
}

NV_STATUS NV_API_CALL os_get_page
(
    NvU64 address
)
{
    get_page(NV_GET_PAGE_STRUCT(address));
    return NV_OK;
}

NV_STATUS NV_API_CALL  os_put_page
(
    NvU64 address
)
{
    put_page(NV_GET_PAGE_STRUCT(address));
    return NV_OK;
}

void NV_API_CALL os_free_pages_phys
(
    NvU64 address, 
    NvU32 size
)
{
    __free_pages(NV_GET_PAGE_STRUCT(address), get_order(size));
}

#if (defined(CONFIG_IPMI_HANDLER) || defined(CONFIG_IPMI_HANDLER_MODULE))

#define NV_IPMI_READ_TIMEOUT_US         1000000 // 1 sec
#define NV_IPMI_SLEEP_MS                1       // 1 ms

struct nv_ipmi_softc
{
    ipmi_user_t         p_user;     // ptr to ipmi_msghandler user structure
    spinlock_t          msg_lock;
    struct list_head    msgs;
    NvU32               seqNum;     //request sequence number
};

static inline int
nv_ipmi_set_my_address
(
    ipmi_user_t     user,
    unsigned char   address
)
{
#if defined(IPMICTL_SET_MY_CHANNEL_ADDRESS_CMD)
    //
    // Commit c14979b993021377228958498937bcdd9539cbce
    // has introduced new IPMI ioctl commands, that allow to specify
    // per-channel IPMB addresses. Consequently ipmi_set_my_address()
    // has received the additional channel argument, and started
    // returning a status (-EINVAL for invalid channel).
    //
    return ipmi_set_my_address(user, 0, address);
#else
    ipmi_set_my_address(user, address);
    return 0;
#endif
}

static void
nv_ipmi_receive_handler
(
    struct ipmi_recv_msg    *rx_msg,
    void                    *priv
)
{
    struct nv_ipmi_softc    *p_priv = priv;
    unsigned long           flags;

    spin_lock_irqsave(&p_priv->msg_lock, flags);

    list_add_tail(&rx_msg->link, &p_priv->msgs);

    spin_unlock_irqrestore(&p_priv->msg_lock, flags);
}

static struct ipmi_user_hndl nv_ipmi_hndlrs =
{
    .ipmi_recv_hndl = nv_ipmi_receive_handler,
};

static NV_STATUS NV_API_CALL _os_ipmi_send_cmd(struct nv_ipmi_softc *p_priv, nvipmi_req_t *p_req);
static NV_STATUS NV_API_CALL _os_ipmi_receive_resp(struct nv_ipmi_softc *p_priv, nvipmi_resp_t *p_resp);

NV_STATUS NV_API_CALL os_ipmi_connect
(
    NvU32   devIndex,
    NvU8    myAddr,
    void    **ppOsPriv
)
{
    NV_STATUS               status;
    struct nv_ipmi_softc    *p_priv;
    int                     err_no;

    if (ppOsPriv == 0)
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto quit;
    }

    status = os_alloc_mem((void **)&p_priv, sizeof(*p_priv));
    if (status != NV_OK)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: failed to allocate private data\n",
                __FUNCTION__);
        goto quit;
    }

    os_mem_set(p_priv, 0, sizeof(*p_priv));

    // Register a client with ipmi_msghandler
    err_no = ipmi_create_user(devIndex, &nv_ipmi_hndlrs, p_priv, &p_priv->p_user);
    if (err_no < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: Error connecting to IPMI handler: %d\n",
                __FUNCTION__, -err_no);
        status = NV_ERR_OPERATING_SYSTEM;
        goto quit;
    }

    spin_lock_init(&p_priv->msg_lock);
    INIT_LIST_HEAD(&p_priv->msgs);

    err_no = ipmi_set_gets_events(p_priv->p_user, 0);
    if (err_no < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: Error disabling IPMI event receiver: %d\n",
                __FUNCTION__, -err_no);
        status = NV_ERR_OPERATING_SYSTEM;
        goto quit;
    }

    err_no = nv_ipmi_set_my_address(p_priv->p_user, myAddr);

    if (err_no < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: Error setting own address: %d\n",
                __FUNCTION__, -err_no);
        status = NV_ERR_OPERATING_SYSTEM;
        goto quit;
    }

    *ppOsPriv = p_priv;
    status = NV_OK;
quit:
    if (status != NV_OK)
    {
        if (p_priv != NULL)
        {
            if (p_priv->p_user != NULL)
            {
                ipmi_destroy_user(p_priv->p_user);
            }
            os_free_mem(p_priv);
        }
    }
    return status;
}

void NV_API_CALL os_ipmi_disconnect
(
    void    *pOsPriv
)
{
    struct nv_ipmi_softc    *p_priv = (struct nv_ipmi_softc *)pOsPriv;
    struct  ipmi_recv_msg *msg, *next;

    if (p_priv == NULL)
    {
        return;
    }

    ipmi_destroy_user(p_priv->p_user);

    list_for_each_entry_safe(msg, next, &p_priv->msgs, link)
    {
        ipmi_free_recv_msg(msg);
    }

    os_free_mem(p_priv);
}

static NV_STATUS NV_API_CALL _os_ipmi_send_cmd
(
    struct nv_ipmi_softc    *p_priv,
    nvipmi_req_t            *p_req
)
{
    struct kernel_ipmi_msg              msg;
    union {
        struct ipmi_addr                    i_addr;
        struct ipmi_system_interface_addr   bmc_addr;
    }                                   tx_addr;
    int                                 err_no;

    os_mem_set(&msg, 0, sizeof(msg));

    tx_addr.bmc_addr.addr_type = IPMI_SYSTEM_INTERFACE_ADDR_TYPE;
    tx_addr.bmc_addr.channel = IPMI_BMC_CHANNEL;
    tx_addr.bmc_addr.lun = p_req->lun;

    err_no = ipmi_validate_addr(&tx_addr.i_addr, sizeof(tx_addr.bmc_addr));
    if (err_no < 0)
    {
        goto tx_exit;
    }

    if (os_alloc_mem((void **)&msg.data, IPMI_MAX_MSG_LENGTH) != NV_OK)
    {
        err_no = -ENOMEM;
        goto tx_exit;
    }

    msg.netfn = p_req->netfn;
    msg.cmd = p_req->cmd;

    if (p_req->data != NULL)
    {
        if (p_req->data_len > IPMI_MAX_MSG_LENGTH)
        {
            err_no = -EMSGSIZE;
            goto tx_exit;
        }
        os_mem_copy(msg.data, p_req->data, p_req->data_len);
        msg.data_len = p_req->data_len;
    }
    else
    {
        msg.data_len = 0;
    }

    err_no = ipmi_request_settime(p_priv->p_user, &tx_addr.i_addr,
                            p_priv->seqNum++, &msg, NULL, 0, -1, 0);

tx_exit:
    if (msg.data != NULL)
    {
        os_free_mem(msg.data);
    }

    if (err_no < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: Error sending command %x/%x: %d\n",
                __FUNCTION__, p_req->netfn, p_req->cmd, -err_no);
        return NV_ERR_OPERATING_SYSTEM;
    }

    return NV_OK;
}

static NV_STATUS NV_API_CALL _os_ipmi_receive_resp
(
    struct nv_ipmi_softc    *p_priv,
    nvipmi_resp_t           *p_resp
)
{
    struct ipmi_recv_msg    *rx_msg;
    int                     err_no;
    struct timeval          tv;
    NvU64                   start_time;

    do_gettimeofday(&tv);
    start_time = NV_TIMEVAL_TO_US(tv);

    err_no = -EAGAIN;
    do
    {
        unsigned long       flags;
        struct list_head    *ent;

        rx_msg = NULL;

        spin_lock_irqsave(&p_priv->msg_lock, flags);
        if (!list_empty(&p_priv->msgs))
        {
            ent = p_priv->msgs.next;
            rx_msg = list_entry(ent, struct ipmi_recv_msg, link);
            list_del(ent);
            spin_unlock_irqrestore(&p_priv->msg_lock, flags);

            err_no = 0;
            break;
        }

        spin_unlock_irqrestore(&p_priv->msg_lock, flags);
        os_delay(NV_IPMI_SLEEP_MS);
        do_gettimeofday(&tv);
    } while (NV_TIMEVAL_TO_US(tv) < (start_time + NV_IPMI_READ_TIMEOUT_US));

    if (rx_msg != NULL)
    {
        if (rx_msg->msg.data_len > 0)
        {
            if ((rx_msg->msg.data_len - 1) > NVIPMI_DATA_BUF_SIZE)
            {
                err_no = -EMSGSIZE;
                goto rx_exit;
            }
            p_resp->ccode = rx_msg->msg.data[0];
            p_resp->data_len = rx_msg->msg.data_len - 1;
            if (p_resp->ccode == 0)
            {
                os_mem_copy(p_resp->data, rx_msg->msg.data + 1,
                            p_resp->data_len);
            }
        }
        else
        {
            p_resp->ccode = NVIPMI_CCODE_INVALID;
            p_resp->data_len = 0;
        }
rx_exit:
        ipmi_free_recv_msg(rx_msg);
    }

    if (err_no < 0)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM:%s: Error reading response: %d\n",
                __FUNCTION__, -err_no);
        return err_no == -EAGAIN ? NV_ERR_TIMEOUT : NV_ERR_OPERATING_SYSTEM;
    }

    return NV_OK;
}

NV_STATUS NV_API_CALL os_ipmi_send_receive_cmd
(
    void                *pOsPriv,
    nvipmi_req_resp_t   *pReq
)
{
    struct nv_ipmi_softc    *p_priv = (struct nv_ipmi_softc *)pOsPriv;
    NV_STATUS               status;

    if ((p_priv == NULL) || (pReq == NULL))
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    status = _os_ipmi_send_cmd(p_priv, &pReq->req);

    if (status == NV_OK)
    {
        status = _os_ipmi_receive_resp(p_priv, &pReq->resp);
    }

    return status;
}

#else   // !(defined(CONFIG_IPMI_HANDLER) || defined(CONFIG_IPMI_HANDLER_MODULE))

NV_STATUS NV_API_CALL os_ipmi_connect
(
    NvU32   devIndex,
    NvU8    myAddr,
    void    **ppOsPriv
)
{
    return NV_ERR_NOT_SUPPORTED;
}

void NV_API_CALL os_ipmi_disconnect
(
    void    *pOsPriv
)
{
}

NV_STATUS NV_API_CALL os_ipmi_send_receive_cmd
(
    void                *pOsPriv,
    nvipmi_req_resp_t   *pReq
)
{
    return NV_ERR_NOT_SUPPORTED;
}
#endif  // (defined(CONFIG_IPMI_HANDLER) || defined(CONFIG_IPMI_HANDLER_MODULE))
