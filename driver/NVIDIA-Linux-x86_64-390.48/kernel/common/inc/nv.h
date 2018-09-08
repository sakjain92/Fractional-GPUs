/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _NV_H_
#define _NV_H_

#include <nvlimits.h>
#include <nvtypes.h>
#include <nvCpuUuid.h>
#include <stdarg.h>
#ifndef __KERNEL__
#include "g_nvconfig.h"
#endif

#include <uvm_config.h>

#if !defined(NV_MIN)
#define NV_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#endif
#if !defined(NV_MAX)
#define NV_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#endif

/* NVIDIA's reserved major character device number (Linux). */
#define NV_MAJOR_DEVICE_NUMBER 195

#define GPU_UUID_LEN    (16)

typedef struct {
    NvU32    domain;        /* PCI domain number   */
    NvU8     bus;           /* PCI bus number      */
    NvU8     slot;          /* PCI slot number     */
    NvU8     function;      /* PCI function number */
    NvU16    vendor_id;     /* PCI vendor ID       */
    NvU16    device_id;     /* PCI device ID       */
    NvBool   valid;         /* validation flag     */
} nv_pci_info_t;

/* NOTE: using an ioctl() number > 55 will overflow! */
#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_CARD_INFO         (NV_IOCTL_BASE + 0)
#define NV_ESC_ENV_INFO          (NV_IOCTL_BASE + 2)
#define NV_ESC_ALLOC_OS_EVENT    (NV_IOCTL_BASE + 6)
#define NV_ESC_FREE_OS_EVENT     (NV_IOCTL_BASE + 7)
#define NV_ESC_STATUS_CODE       (NV_IOCTL_BASE + 9)
#define NV_ESC_CHECK_VERSION_STR (NV_IOCTL_BASE + 10)
#define NV_ESC_IOCTL_XFER_CMD    (NV_IOCTL_BASE + 11)
#define NV_ESC_ATTACH_GPUS_TO_FD (NV_IOCTL_BASE + 12)
#define NV_ESC_QUERY_DEVICE_INTR (NV_IOCTL_BASE + 13)
#define NV_ESC_SYS_PARAMS        (NV_IOCTL_BASE + 14)
#define NV_ESC_NUMA_INFO         (NV_IOCTL_BASE + 15)
#define NV_ESC_SET_NUMA_STATUS   (NV_IOCTL_BASE + 16)

/*
 * #define an absolute maximum used as a sanity check for the
 * NV_ESC_IOCTL_XFER_CMD ioctl() size argument.
 */
#define NV_ABSOLUTE_MAX_IOCTL_SIZE  16384

/*
 * Solaris provides no more than 8 bits for the argument size in
 * the ioctl() command encoding; make sure we don't exceed this
 * limit.
 */
#define __NV_IOWR_ASSERT(type) ((sizeof(type) <= NV_PLATFORM_MAX_IOCTL_SIZE) ? 1 : -1)
#define __NV_IOWR(nr, type) ({                                        \
    typedef char __NV_IOWR_TYPE_SIZE_ASSERT[__NV_IOWR_ASSERT(type)];  \
    _IOWR(NV_IOCTL_MAGIC, (nr), type);                                \
})

/*
 * ioctl()'s with parameter structures too large for the
 * _IOC cmd layout use the nv_ioctl_xfer_t structure
 * and the NV_ESC_IOCTL_XFER_CMD ioctl() to pass the actual
 * size and user argument pointer into the RM, which
 * will then copy it to/from kernel space in separate steps.
 */
typedef struct nv_ioctl_xfer
{
    NvU32   cmd;
    NvU32   size;
    NvP64   ptr  NV_ALIGN_BYTES(8);
} nv_ioctl_xfer_t;

typedef struct nv_ioctl_card_info
{
    NvU16         flags;               /* see below                   */
    nv_pci_info_t pci_info;            /* PCI config information      */
    NvU32         gpu_id;
    NvU16         interrupt_line;
    NvU64         reg_address    NV_ALIGN_BYTES(8);
    NvU64         reg_size       NV_ALIGN_BYTES(8);
    NvU64         fb_address     NV_ALIGN_BYTES(8);
    NvU64         fb_size        NV_ALIGN_BYTES(8);
    NvU32         minor_number;
    NvU8          dev_name[10];  /* device names such as vmgfx[0-32] for vmkernel */
} nv_ioctl_card_info_t;

#define NV_IOCTL_CARD_INFO_BUS_TYPE_PCI            0x0001
#define NV_IOCTL_CARD_INFO_BUS_TYPE_AGP            0x0002
#define NV_IOCTL_CARD_INFO_BUS_TYPE_PCI_EXPRESS    0x0003

#define NV_IOCTL_CARD_INFO_FLAG_PRESENT       0x0001

#define SIM_ENV_GPU       0
#define SIM_ENV_IKOS      1
#define SIM_ENV_CSIM      2

#define NV_SLI_DISABLED   0
#define NV_SLI_ENABLED    1

typedef struct nv_ioctl_env_info
{
    NvU32 pat_supported;
} nv_ioctl_env_info_t;

/* old rm api check
 *
 * this used to be used to verify client/rm interaction both ways by
 * overloading the structure passed into the NV_IOCTL_CARD_INFO ioctl.
 * This interface is deprecated and NV_IOCTL_CHECK_VERSION_STR should
 * be used instead.  We keep the structure and defines here so that RM
 * can recognize and handle old clients.
 */
typedef struct nv_ioctl_rm_api_old_version
{
    NvU32 magic;
    NvU32 major;
    NvU32 minor;
    NvU32 patch;
} nv_ioctl_rm_api_old_version_t;

#define NV_RM_API_OLD_VERSION_MAGIC_REQ              0x0197fade
#define NV_RM_API_OLD_VERSION_MAGIC_REP              0xbead2929
#define NV_RM_API_OLD_VERSION_MAGIC_LAX_REQ         (NV_RM_API_OLD_VERSION_MAGIC_REQ ^ '1')
#define NV_RM_API_OLD_VERSION_MAGIC_OVERRIDE_REQ    (NV_RM_API_OLD_VERSION_MAGIC_REQ ^ '2')
#define NV_RM_API_OLD_VERSION_MAGIC_IGNORE           0xffffffff

typedef enum {
    NV_CPU_TYPE_UNKNOWN = 0,
    NV_CPU_TYPE_ARM_A9
} nv_cpu_type_t;

/* alloc event */
typedef struct nv_ioctl_alloc_os_event
{
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hOsEvent;
    NvU32    fd;
    NvU32    Status;
} nv_ioctl_alloc_os_event_t;

/* free event */
typedef struct nv_ioctl_free_os_event
{
    NvHandle hClient;
    NvHandle hDevice;
    NvU32    fd;
    NvU32    Status;
} nv_ioctl_free_os_event_t;

#define NV_PCI_DEV_FMT          "%04x:%02x:%02x.%x"
#define NV_PCI_DEV_FMT_ARGS(nv) (nv)->pci_info.domain, (nv)->pci_info.bus, \
                                (nv)->pci_info.slot, (nv)->pci_info.function

/* status code */
typedef struct nv_ioctl_status_code
{
    NvU32 domain;
    NvU8  bus;
    NvU8  slot;
    NvU32 status;
} nv_ioctl_status_code_t;

/* check version string */
#define NV_RM_API_VERSION_STRING_LENGTH 64

typedef struct nv_ioctl_rm_api_version
{
    NvU32 cmd;
    NvU32 reply;
    char versionString[NV_RM_API_VERSION_STRING_LENGTH];
} nv_ioctl_rm_api_version_t;

#define NV_RM_API_VERSION_CMD_STRICT         0
#define NV_RM_API_VERSION_CMD_RELAXED       '1'
#define NV_RM_API_VERSION_CMD_OVERRIDE      '2'

#define NV_RM_API_VERSION_REPLY_UNRECOGNIZED 0
#define NV_RM_API_VERSION_REPLY_RECOGNIZED   1

typedef struct nv_ioctl_query_device_intr
{
    NvU32 intrStatus NV_ALIGN_BYTES(4);
    NvU32 status;
} nv_ioctl_query_device_intr;

#define NV_RM_DEVICE_INTR_ADDRESS 0x100


/* system parameters that the kernel driver may use for configuration */
typedef struct nv_ioctl_sys_params
{
    NvU64 memblock_size NV_ALIGN_BYTES(8);
} nv_ioctl_sys_params_t;

/* per-device NUMA memory info as assigned by the system */
typedef struct nv_ioctl_numa_info
{
    NvS32 nid;
    NvS32 status;
    NvU64 memblock_size NV_ALIGN_BYTES(8);
    NvU64 numa_mem_addr NV_ALIGN_BYTES(8);
    NvU64 numa_mem_size NV_ALIGN_BYTES(8);
} nv_ioctl_numa_info_t;

/* set the status of the device NUMA memory */
typedef struct nv_ioctl_set_numa_status
{
    NvS32 status;
} nv_ioctl_set_numa_status_t;

#define NV_IOCTL_NUMA_STATUS_DISABLED               0
#define NV_IOCTL_NUMA_STATUS_OFFLINE                1
#define NV_IOCTL_NUMA_STATUS_ONLINE_IN_PROGRESS     2
#define NV_IOCTL_NUMA_STATUS_ONLINE                 3
#define NV_IOCTL_NUMA_STATUS_ONLINE_FAILED          4
#define NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS    5
#define NV_IOCTL_NUMA_STATUS_OFFLINE_FAILED         6

#ifdef NVRM

extern const char *pNVRM_ID;

/*
 * ptr arithmetic convenience
 */

typedef union
{
    volatile NvV8 Reg008[1];
    volatile NvV16 Reg016[1];
    volatile NvV32 Reg032[1];
} nv_hwreg_t, * nv_phwreg_t;


#define NVRM_PCICFG_NUM_BARS            6
#define NVRM_PCICFG_BAR_OFFSET(i)       (0x10 + (i) * 4)
#define NVRM_PCICFG_BAR_REQTYPE_MASK    0x00000001
#define NVRM_PCICFG_BAR_REQTYPE_MEMORY  0x00000000
#define NVRM_PCICFG_BAR_MEMTYPE_MASK    0x00000006
#define NVRM_PCICFG_BAR_MEMTYPE_64BIT   0x00000004
#define NVRM_PCICFG_BAR_ADDR_MASK       0xfffffff0

#define NVRM_PCICFG_NUM_DWORDS          16

#define NV_GPU_NUM_BARS                 3
#define NV_GPU_BAR_INDEX_REGS           0
#define NV_GPU_BAR_INDEX_FB             1
#define NV_GPU_BAR_INDEX_IMEM           2

typedef struct
{
    NvU64 cpu_address;
    NvU64 bus_address;
    NvU64 strapped_size;
    NvU64 size;
    NvU32 offset;
    NvU32 *map;
    nv_phwreg_t map_u;
} nv_aperture_t;

typedef struct
{
    char *node;
    char *name;
    NvU32 *data;
} nv_parm_t;

#define NV_RM_PAGE_SHIFT    12
#define NV_RM_PAGE_SIZE     (1 << NV_RM_PAGE_SHIFT)
#define NV_RM_PAGE_MASK     (NV_RM_PAGE_SIZE - 1)

#define NV_RM_TO_OS_PAGE_SHIFT      (OS_PAGE_SHIFT - NV_RM_PAGE_SHIFT)
#define NV_RM_PAGES_PER_OS_PAGE     (1U << NV_RM_TO_OS_PAGE_SHIFT)
#define NV_RM_PAGES_TO_OS_PAGES(count) \
    ((((NvUPtr)(count)) >> NV_RM_TO_OS_PAGE_SHIFT) + \
     ((((count) & ((1 << NV_RM_TO_OS_PAGE_SHIFT) - 1)) != 0) ? 1 : 0))

#if defined(NVCPU_X86_64)
#define NV_STACK_SIZE (NV_RM_PAGE_SIZE * 3)
#else
#define NV_STACK_SIZE (NV_RM_PAGE_SIZE * 2)
#endif

typedef struct nvidia_stack_s
{
    NvU32 size;
    void *top;
    NvU8  stack[NV_STACK_SIZE-16] __attribute__ ((aligned(16)));
} nvidia_stack_t;

/*
 * TODO: Remove once all UNIX layers have been converted to use nvidia_stack_t
 */
typedef nvidia_stack_t nv_stack_t;

/*
 * this is a wrapper for unix events
 * unlike the events that will be returned to clients, this includes
 * kernel-specific data, such as file pointer, etc..
 */
typedef struct nv_event_s
{
    NvHandle            hParent;
    NvHandle            hObject;
    NvU32               index;
    void               *file;  /* per file-descriptor data pointer */
    NvHandle            handle;
    NvU32               fd;
    struct nv_event_s  *next;
} nv_event_t;

/*
 * this is to be able to associate a fd memdesc to a file descriptor
 */
typedef struct nv_fd_memdesc_s
{
    NvBool              bValid;
    NvHandle            hParent;
    NvHandle            hMemory;
} nv_fd_memdesc_t;

typedef struct nv_kern_mapping_s
{
    void  *addr;
    NvU64 size;
    NvU32 modeFlag;
    struct nv_kern_mapping_s *next;
} nv_kern_mapping_t;

typedef struct nv_usermap_access_params_s
{
    NvU64 addr;
    NvU64 size;
    NvU64 mmap_start;
    NvU64 mmap_size;
    NvU64 access_start;
    NvU64 access_size;
    NvU64 remap_prot_extra;
} nv_usermap_access_params_t;

/*
 * It stores mapping context per mapping
 */
typedef struct nv_alloc_mapping_context_s {
    void  *alloc;
    NvU64  page_index;
    NvU64 mmap_start;
    NvU64 mmap_size;
    NvU64 access_start;
    NvU64 access_size;
    NvU64 remap_prot_extra;
    NvU32  prot;
    NvBool bValid;
} nv_alloc_mapping_context_t;

/*
 * per device state
 */

typedef struct
{
    void  *priv;                    /* private data */
    void  *os_state;                /* os-specific device state */

    int    flags;

    /* PCI config info */
    nv_pci_info_t pci_info;
    NvU16 subsystem_id;
    NvU16 subsystem_vendor;
    NvU32 gpu_id;
    NvU32 iovaspace_id;
    struct
    {
        NvBool         bValid;
        NvU8           uuid[GPU_UUID_LEN];
    } nv_gpu_uuid_cache;
    void *handle;

    NvU32 pci_cfg_space[NVRM_PCICFG_NUM_DWORDS];

    /* physical characteristics */
    nv_aperture_t bars[NV_GPU_NUM_BARS];
    nv_aperture_t *regs;
    nv_aperture_t *fb, ud;

    NvU32  interrupt_line;

    NvBool primary_vga;

    NvU32 sim_env;

    NvU32 rc_timer_enabled;

    /* list of events allocated for this device */
    nv_event_t *event_list;

    /* lock to protect event_list */
    void *event_spinlock;

    nv_kern_mapping_t *kern_mappings;

    /* DMA addressable range of the device */
    NvU64 dma_addressable_start;
    NvU64 dma_addressable_limit;

    NvBool nvlink_sysmem_links_enabled;
} nv_state_t;

// Forward define the gpu ops structures
typedef struct gpuSession                           *nvgpuSessionHandle_t;
typedef struct gpuAddressSpace                      *nvgpuAddressSpaceHandle_t;
typedef struct gpuChannel                           *nvgpuChannelHandle_t;
typedef struct gpuObject                            *nvgpuObjectHandle_t;
typedef struct UvmGpuChannelPointers_tag            *nvgpuChannelInfo_t;
typedef struct UvmGpuChannelAllocParams_tag          nvgpuChannelAllocParams_t;
typedef struct UvmGpuCaps_tag                       *nvgpuCaps_t;
typedef struct UvmGpuAddressSpaceInfo_tag           *nvgpuAddressSpaceInfo_t;
typedef struct UvmGpuAllocInfo_tag                  *nvgpuAllocInfo_t;
typedef struct UvmGpuP2PCapsParams_tag              *nvgpuP2PCapsParams_t;
typedef struct gpuVaAllocInfo                       *nvgpuVaAllocInfo_t;
typedef struct gpuMapInfo                           *nvgpuMapInfo_t;
typedef struct UvmGpuFbInfo_tag                     *nvgpuFbInfo_t;
typedef struct UvmGpuFaultInfo_tag                  *nvgpuFaultInfo_t;
typedef struct UvmGpuAccessCntrInfo_tag             *nvgpuAccessCntrInfo_t;
typedef struct UvmGpuAccessCntrConfig_tag           *nvgpuAccessCntrConfig_t;
typedef struct UvmGpuInfo_tag                       *nvgpuInfo_t;
typedef struct gpuPmaAllocationOptions              *nvgpuPmaAllocationOptions_t;
typedef struct UvmGpuMemoryInfo_tag                 *nvgpuMemoryInfo_t;
typedef struct UvmGpuExternalMappingInfo_tag        *nvgpuExternalMappingInfo_t;
typedef struct UvmGpuChannelResourceInfo_tag        *nvgpuChannelResourceInfo_t;
typedef struct UvmGpuChannelInstanceInfo_tag        *nvgpuChannelInstanceInfo_t;
typedef struct UvmGpuChannelResourceBindParams_tag  *nvgpuChannelResourceBindParams_t;
typedef NV_STATUS (*nvPmaEvictPagesCallback)(void *, NvU32, NvU64 *, NvU32, NvU64, NvU64);
typedef NV_STATUS (*nvPmaEvictRangeCallback)(void *, NvU64, NvU64);

/*
 * flags
 */

#define NV_FLAG_OPEN                   0x0001
// EMPTY SLOT                          0x0002
#define NV_FLAG_CONTROL                0x0004
#define NV_FLAG_MAP_REGS_EARLY         0x0008
#define NV_FLAG_USE_BAR0_CFG           0x0010
#define NV_FLAG_USES_MSI               0x0020
// EMPTY SLOT                          0x0040
#define NV_FLAG_PASSTHRU               0x0080
#define NV_FLAG_GVI_IN_SUSPEND         0x0100
#define NV_FLAG_GVI                    0x0200
#define NV_FLAG_GVI_INTR_EN            0x0400
#define NV_FLAG_PERSISTENT_SW_STATE    0x0800
#define NV_FLAG_IN_RECOVERY            0x1000
#define NV_FLAG_SKIP_CFG_CHECK         0x2000
#define NV_FLAG_UNBIND_LOCK            0x4000

#define NV_PM_ACPI_HIBERNATE    0x0001
#define NV_PM_ACPI_STANDBY      0x0002
#define NV_PM_ACPI_RESUME       0x0003

#define NV_PRIMARY_VGA(nv)      ((nv)->primary_vga)

#define NV_IS_GVI_DEVICE(nv)    ((nv)->flags & NV_FLAG_GVI)
#define NV_IS_CTL_DEVICE(nv)    ((nv)->flags & NV_FLAG_CONTROL)

/*
 * The ACPI specification defines IDs for various ACPI video
 * extension events like display switch events, AC/battery
 * events, docking events, etc..
 * Whenever an ACPI event is received by the corresponding
 * event handler installed within the core NVIDIA driver, the
 * code can verify the event ID before processing it.
 */
#define ACPI_DISPLAY_DEVICE_CHANGE_EVENT      0x80
#define NVIF_NOTIFY_DISPLAY_DETECT           0xCB
#define NVIF_DISPLAY_DEVICE_CHANGE_EVENT     NVIF_NOTIFY_DISPLAY_DETECT
/*
 * NVIDIA ACPI event IDs to be passed into the core NVIDIA
 * driver for various events like display switch events,
 * AC/battery events, docking events, etc..
 */
#define NV_SYSTEM_ACPI_DISPLAY_SWITCH_EVENT  0x8001
#define NV_SYSTEM_ACPI_BATTERY_POWER_EVENT   0x8002
#define NV_SYSTEM_ACPI_DOCK_EVENT            0x8003

/*
 * Status bit definitions for display switch hotkey events.
 */
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_LCD 0x01
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_CRT 0x02
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_TV  0x04
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_DFP 0x08

/*
 * NVIDIA ACPI sub-event IDs (event types) to be passed into
 * to core NVIDIA driver for ACPI events.
 */
#define NV_SYSTEM_ACPI_EVENT_VALUE_DISPLAY_SWITCH_DEFAULT    0
#define NV_SYSTEM_ACPI_EVENT_VALUE_POWER_EVENT_AC            0
#define NV_SYSTEM_ACPI_EVENT_VALUE_POWER_EVENT_BATTERY       1
#define NV_SYSTEM_ACPI_EVENT_VALUE_DOCK_EVENT_UNDOCKED       0
#define NV_SYSTEM_ACPI_EVENT_VALUE_DOCK_EVENT_DOCKED         1

#define NV_ACPI_NVIF_HANDLE_PRESENT 0x01
#define NV_ACPI_DSM_HANDLE_PRESENT  0x02
#define NV_ACPI_WMMX_HANDLE_PRESENT 0x04
#define NV_ACPI_MXMI_HANDLE_PRESENT 0x08
#define NV_ACPI_MXMS_HANDLE_PRESENT 0x10

#define NV_EVAL_ACPI_METHOD_NVIF     0x01
#define NV_EVAL_ACPI_METHOD_WMMX     0x02
#define NV_EVAL_ACPI_METHOD_MXMI     0x03
#define NV_EVAL_ACPI_METHOD_MXMS     0x04

#define NV_I2C_CMD_READ              1
#define NV_I2C_CMD_WRITE             2
#define NV_I2C_CMD_SMBUS_READ        3
#define NV_I2C_CMD_SMBUS_WRITE       4
#define NV_I2C_CMD_SMBUS_QUICK_WRITE 5
#define NV_I2C_CMD_SMBUS_QUICK_READ  6
#define NV_I2C_CMD_SMBUS_BLOCK_READ  7
#define NV_I2C_CMD_SMBUS_BLOCK_WRITE 8

// Flags needed by OSAllocPagesNode
#define NV_ALLOC_PAGES_NODE_NONE                0x0
#define NV_ALLOC_PAGES_NODE_SCRUB_ON_ALLOC      0x1
#define NV_ALLOC_PAGES_NODE_FORCE_ALLOC         0x2

/*
** where we hide our nv_state_t * ...
*/
#define NV_SET_NV_STATE(pgpu,p) ((pgpu)->pOsHwInfo = (p))
#define NV_GET_NV_STATE(pGpu) \
    (nv_state_t *)((pGpu) ? (pGpu)->pOsHwInfo : NULL)

#define IS_DMA_ADDRESSABLE(nv, offset)                                          \
    (((offset) >= (nv)->dma_addressable_start) &&                               \
     ((offset) <= (nv)->dma_addressable_limit))

#define IS_REG_OFFSET(nv, offset, length)                                       \
    (((offset) >= (nv)->regs->cpu_address) &&                                   \
    (((offset) + ((length)-1)) <=                                               \
        (nv)->regs->cpu_address + ((nv)->regs->size-1)))

#define IS_FB_OFFSET(nv, offset, length)                                        \
    (((offset) >= (nv)->fb->cpu_address) &&                                     \
    (((offset) + ((length)-1)) <= (nv)->fb->cpu_address + ((nv)->fb->size-1)))

#define IS_UD_OFFSET(nv, offset, length)                                        \
    (((nv)->ud.cpu_address != 0) && ((nv)->ud.size != 0) &&                     \
    ((offset) >= (nv)->ud.cpu_address) &&                                       \
    (((offset) + ((length)-1)) <= (nv)->ud.cpu_address + ((nv)->ud.size-1)))

#define IS_IMEM_OFFSET(nv, offset, length)                                      \
    (((nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address != 0) &&                    \
     ((nv)->bars[NV_GPU_BAR_INDEX_IMEM].size != 0) &&                           \
     ((offset) >= (nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address) &&             \
     (((offset) + ((length) - 1)) <=                                            \
        (nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address +                         \
            ((nv)->bars[NV_GPU_BAR_INDEX_IMEM].size - 1)))

/* device name length; must be atleast 8 */

#define NV_DEVICE_NAME_LENGTH 40

#define NV_MAX_ISR_DELAY_US           20000
#define NV_MAX_ISR_DELAY_MS           (NV_MAX_ISR_DELAY_US / 1000)

#define NV_TIMERCMP(a, b, CMP)                                              \
    (((a)->tv_sec == (b)->tv_sec) ?                                         \
        ((a)->tv_usec CMP (b)->tv_usec) : ((a)->tv_sec CMP (b)->tv_sec))

#define NV_TIMERADD(a, b, result)                                           \
    {                                                                       \
        (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;                       \
        (result)->tv_usec = (a)->tv_usec + (b)->tv_usec;                    \
        if ((result)->tv_usec >= 1000000)                                   \
        {                                                                   \
            ++(result)->tv_sec;                                             \
            (result)->tv_usec -= 1000000;                                   \
        }                                                                   \
    }

#define NV_TIMERSUB(a, b, result)                                           \
    {                                                                       \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                       \
        (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                    \
        if ((result)->tv_usec < 0)                                          \
        {                                                                   \
          --(result)->tv_sec;                                               \
          (result)->tv_usec += 1000000;                                     \
        }                                                                   \
    }

#define NV_TIMEVAL_TO_US(tv)    ((NvU64)(tv).tv_sec * 1000000 + (tv).tv_usec)

#ifndef NV_ALIGN_UP
#define NV_ALIGN_UP(v,g) (((v) + ((g) - 1)) & ~((g) - 1))
#endif
#ifndef NV_ALIGN_DOWN
#define NV_ALIGN_DOWN(v,g) ((v) & ~((g) - 1))
#endif

/*
 * driver internal interfaces
 */

#ifndef NVWATCH

/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for UNIX specific OS interface.
 *
 * ---------------------------------------------------------------------------
 */

/*
 * Make sure that arguments to and from the core resource manager
 * are passed and expected on the stack. define duplicated in os-interface.h
 */
#if !defined(NV_API_CALL)
#if defined(NVCPU_X86)
#if defined(__use_altstack__)
#define NV_API_CALL __attribute__((regparm(0),altstack(false)))
#else
#define NV_API_CALL __attribute__((regparm(0)))
#endif
#elif defined(NVCPU_X86_64) && defined(__use_altstack__)
#define NV_API_CALL __attribute__((altstack(false)))
#else
#define NV_API_CALL
#endif
#endif /* !defined(NV_API_CALL) */

void*      NV_API_CALL  nv_alloc_kernel_mapping  (nv_state_t *, void *, NvU64, NvU32, NvU64, void **);
NV_STATUS  NV_API_CALL  nv_free_kernel_mapping   (nv_state_t *, void *, void *, void *);
NV_STATUS  NV_API_CALL  nv_alloc_user_mapping    (nv_state_t *, void *, NvU64, NvU32, NvU64, NvU32, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_free_user_mapping     (nv_state_t *, void *, NvU64, void *);
NV_STATUS  NV_API_CALL  nv_add_mapping_context_to_file (nv_state_t *, nv_usermap_access_params_t*, NvU32, void *, NvU64, NvU32);

NvU64  NV_API_CALL  nv_get_kern_phys_address     (NvU64);
NvU64  NV_API_CALL  nv_get_user_phys_address     (NvU64);
nv_state_t*  NV_API_CALL  nv_get_adapter_state   (NvU32, NvU8, NvU8);
nv_state_t*  NV_API_CALL  nv_get_ctl_state       (void);

void   NV_API_CALL  nv_set_dma_address_size      (nv_state_t *, NvU32 );

NV_STATUS  NV_API_CALL  nv_alias_pages           (nv_state_t *, NvU32, NvU32, NvU32, NvU64, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_alloc_pages           (nv_state_t *, NvU32, NvBool, NvU32, NvBool, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_free_pages            (nv_state_t *, NvU32, NvBool, NvU32, void *);

NV_STATUS  NV_API_CALL  nv_register_user_pages   (nv_state_t *, NvU64, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_unregister_user_pages (nv_state_t *, NvU64, void **);

NV_STATUS NV_API_CALL   nv_register_peer_io_mem   (nv_state_t *, NvU64 *, NvU64, void **);
void      NV_API_CALL   nv_unregister_peer_io_mem (nv_state_t *, void *);

NV_STATUS  NV_API_CALL  nv_dma_map_pages         (nv_state_t *, NvU64, NvU64 *, NvBool, void **);
NV_STATUS  NV_API_CALL  nv_dma_unmap_pages       (nv_state_t *, NvU64, NvU64 *, void **);

NV_STATUS  NV_API_CALL  nv_dma_map_alloc         (nv_state_t *, NvU64, NvU64 *, NvBool, void **);
NV_STATUS  NV_API_CALL  nv_dma_unmap_alloc       (nv_state_t *, NvU64, NvU64 *, void **);

NV_STATUS  NV_API_CALL  nv_dma_map_peer          (nv_state_t *, nv_state_t *, NvU8, NvU64, NvU64 *);
void       NV_API_CALL  nv_dma_unmap_peer        (nv_state_t *, NvU64, NvU64);

NV_STATUS  NV_API_CALL  nv_dma_map_mmio          (nv_state_t *, NvU64, NvU64 *);
void       NV_API_CALL  nv_dma_unmap_mmio        (nv_state_t *, NvU64, NvU64);

NvS32  NV_API_CALL  nv_start_rc_timer            (nv_state_t *);
NvS32  NV_API_CALL  nv_stop_rc_timer             (nv_state_t *);

void   NV_API_CALL  nv_post_event                (nv_state_t *, nv_event_t *, NvHandle, NvU32, NvBool);
NvS32  NV_API_CALL  nv_get_event                 (nv_state_t *, void *, nv_event_t *, NvU32 *);

void   NV_API_CALL  nv_verify_pci_config         (nv_state_t *, BOOL);

void*  NV_API_CALL  nv_i2c_add_adapter           (nv_state_t *, NvU32);
BOOL   NV_API_CALL  nv_i2c_del_adapter           (nv_state_t *, void *);

void   NV_API_CALL  nv_acpi_methods_init         (NvU32 *);
void   NV_API_CALL  nv_acpi_methods_uninit       (void);

NV_STATUS  NV_API_CALL  nv_acpi_method           (NvU32, NvU32, NvU32, void *, NvU16, NvU32 *, void *, NvU16 *);
NV_STATUS  NV_API_CALL  nv_acpi_dsm_method       (nv_state_t *, NvU8 *, NvU32, NvU32, void *, NvU16, NvU32 *, void *, NvU16 *);
NV_STATUS  NV_API_CALL  nv_acpi_ddc_method       (nv_state_t *, void *, NvU32 *);
NV_STATUS  NV_API_CALL  nv_acpi_dod_method       (nv_state_t *, NvU32 *, NvU32 *);
NV_STATUS  NV_API_CALL  nv_acpi_rom_method       (nv_state_t *, NvU32 *, NvU32 *);
NV_STATUS  NV_API_CALL  nv_log_error             (nv_state_t *, NvU32, const char *, va_list);

NvU64      NV_API_CALL  nv_get_dma_start_address (nv_state_t *);
NV_STATUS  NV_API_CALL  nv_set_primary_vga_status(nv_state_t *);
NV_STATUS  NV_API_CALL  nv_pci_trigger_recovery  (nv_state_t *);
NvBool     NV_API_CALL  nv_requires_dma_remap    (nv_state_t *);

#if defined(NVCPU_X86) || defined(NVCPU_X86_64)
NvBool     NV_API_CALL  nv_is_virtualized_system  (nv_stack_t *);
#endif

nv_fd_memdesc_t* NV_API_CALL nv_get_fd_memdesc   (void *);
NV_STATUS  NV_API_CALL  nv_add_fd_memdesc_to_fd  (NvU32, const nv_fd_memdesc_t*);

NV_STATUS NV_API_CALL nv_export_rm_object_to_fd    (NvHandle, NvS32);
NV_STATUS NV_API_CALL nv_import_rm_object_from_fd  (NvS32, NvHandle *);

NV_STATUS NV_API_CALL nv_get_device_memory_config(nv_state_t *, NvU32 *, NvU32 *, NvU32 *, NvU32 *, NvS32 *);

NV_STATUS NV_API_CALL nv_get_ibmnpu_genreg_info(nv_state_t *, NvU64 *, NvU64 *);
NV_STATUS NV_API_CALL nv_get_ibmnpu_relaxed_ordering_mode(nv_state_t *nv, NvBool *mode);

#if defined(NVCPU_PPC64LE)
NV_STATUS NV_API_CALL nv_get_nvlink_line_rate    (nv_state_t *, NvU32 *);
#endif

void      NV_API_CALL nv_warn_about_vesafb       (void);
void      NV_API_CALL nv_register_backlight      (nv_state_t *, NvU32, NvU32);
void      NV_API_CALL nv_unregister_backlight    (nv_state_t *);
/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for Resource Manager interface.
 *
 * ---------------------------------------------------------------------------
 */

BOOL       NV_API_CALL  rm_init_rm               (nvidia_stack_t *);
BOOL       NV_API_CALL  rm_shutdown_rm           (nvidia_stack_t *);
BOOL       NV_API_CALL  rm_init_private_state    (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_free_private_state    (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_init_adapter          (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_disable_adapter       (nvidia_stack_t *, nv_state_t *);
BOOL       NV_API_CALL  rm_shutdown_adapter      (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_validate_mmap_request (nvidia_stack_t *, nv_state_t *, void *, NvU64, NvU64, NvU32 *, void **, NvU64 *);
NV_STATUS  NV_API_CALL  rm_acquire_api_lock      (nvidia_stack_t *);
NV_STATUS  NV_API_CALL  rm_release_api_lock      (nvidia_stack_t *);
NV_STATUS  NV_API_CALL  rm_ioctl                 (nvidia_stack_t *, nv_state_t *, void *, NvU32, void *, NvU32);
BOOL       NV_API_CALL  rm_isr                   (nvidia_stack_t *, nv_state_t *, NvU32 *);
void       NV_API_CALL  rm_isr_bh                (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_isr_bh_unlocked       (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_power_management      (nvidia_stack_t *, nv_state_t *, NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_save_low_res_mode     (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_get_vbios_version     (nvidia_stack_t *, nv_state_t *, NvU32 *, NvU32 *, NvU32 *, NvU32 *, NvU32 *);
NV_STATUS  NV_API_CALL  rm_get_gpu_uuid          (nvidia_stack_t *, nv_state_t *, NvU8 **, NvU32 *);
NV_STATUS  NV_API_CALL  rm_get_gpu_uuid_raw      (nvidia_stack_t *, nv_state_t *, NvU8 **, NvU32 *);
void       NV_API_CALL  rm_free_unused_clients   (nvidia_stack_t *, nv_state_t *, void *);
void       NV_API_CALL  rm_unbind_lock           (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_free_exported_object  (nvidia_stack_t *, NvHandle);
NV_STATUS  NV_API_CALL  rm_read_registry_dword   (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU8 *, NvU32 *);
NV_STATUS  NV_API_CALL  rm_write_registry_dword  (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU8 *, NvU32);
NV_STATUS  NV_API_CALL  rm_read_registry_binary  (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU8 *, NvU8 *, NvU32 *);
NV_STATUS  NV_API_CALL  rm_write_registry_binary (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU8 *, NvU8 *, NvU32);
NV_STATUS  NV_API_CALL  rm_write_registry_string (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU8 *, const char *, NvU32);
void       NV_API_CALL  rm_parse_option_string   (nvidia_stack_t *, const char *);
char*      NV_API_CALL  rm_remove_spaces         (const char *);
char*      NV_API_CALL  rm_string_token          (char **, const char);

NV_STATUS  NV_API_CALL  rm_run_rc_callback       (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_execute_work_item     (nvidia_stack_t *, void *);
NV_STATUS  NV_API_CALL  rm_get_device_name       (nvidia_stack_t *, nv_state_t *, NvU16, NvU16, NvU16, NvU32, NvU8 *);

NvU64      NV_API_CALL  nv_rdtsc                 (void);

void       NV_API_CALL  rm_register_compatible_ioctls   (nvidia_stack_t *);
void       NV_API_CALL  rm_unregister_compatible_ioctls (nvidia_stack_t *);

BOOL       NV_API_CALL  rm_is_legacy_device      (nvidia_stack_t *, NvU16, NvU16, NvU16, BOOL);
NV_STATUS  NV_API_CALL  rm_is_supported_device   (nvidia_stack_t *, nv_state_t *);

void       NV_API_CALL  rm_check_pci_config_space (nvidia_stack_t *, nv_state_t *nv, BOOL, BOOL, BOOL);

NV_STATUS  NV_API_CALL  rm_i2c_remove_adapters    (nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL  rm_i2c_is_smbus_capable   (nvidia_stack_t *, nv_state_t *, void *);
NV_STATUS  NV_API_CALL  rm_i2c_transfer           (nvidia_stack_t *, nv_state_t *, void *, NvU8, NvU8, NvU8, NvU32, NvU8 *);

NV_STATUS  NV_API_CALL  rm_perform_version_check  (nvidia_stack_t *, void *, NvU32);

NV_STATUS  NV_API_CALL  rm_system_event           (nvidia_stack_t *, NvU32, NvU32);

void       NV_API_CALL  rm_disable_gpu_state_persistence    (nvidia_stack_t *sp, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_p2p_init_mapping       (nvidia_stack_t *, NvU64, NvU64 *, NvU64 *, NvU64 *, NvU64 *, NvU64, NvU64, NvU64, NvU64, void (*)(void *), void *);
NV_STATUS  NV_API_CALL  rm_p2p_destroy_mapping    (nvidia_stack_t *, NvU64);
NV_STATUS  NV_API_CALL  rm_p2p_get_pages          (nvidia_stack_t *, NvU64, NvU32, NvU64, NvU64, NvU64 *, NvU32 *, NvU32 *, NvU32 *, NvU8 **, void *, void (*)(void *), void *);
NV_STATUS  NV_API_CALL  rm_p2p_put_pages          (nvidia_stack_t *, NvU64, NvU32, NvU64, void *);
NV_STATUS  NV_API_CALL  rm_p2p_dma_map_pages      (nvidia_stack_t *, nv_state_t *, NvU8 *, NvU32, NvU32, NvU64 *, void **);
NV_STATUS  NV_API_CALL  rm_p2p_dma_unmap_pages    (nvidia_stack_t *, nv_state_t *, NvU32, NvU32, NvU64 *, void *);
NV_STATUS  NV_API_CALL  rm_get_cpu_type           (nvidia_stack_t *, nv_cpu_type_t *);
NV_STATUS  NV_API_CALL  rm_log_gpu_crash          (nv_stack_t *, nv_state_t *);

NV_STATUS  NV_API_CALL  rm_gpu_ops_create_session (nvidia_stack_t *, nvgpuSessionHandle_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_destroy_session (nvidia_stack_t *, nvgpuSessionHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_create(nvidia_stack_t *, nvgpuSessionHandle_t, const NvProcessorUuid *, unsigned long long, unsigned long long, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_address_space(nvidia_stack_t *, nvgpuSessionHandle_t, const NvProcessorUuid *, NvHandle, NvHandle, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_destroy(nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_fb(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_alloc_pages(nvidia_stack_t *, void *, NvLength, NvU32 , nvgpuPmaAllocationOptions_t, NvU64 *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_free_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_pin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_unpin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_get_pma_object(nvidia_stack_t *, const NvProcessorUuid *, void **);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_register_callbacks(nvidia_stack_t *sp, void *, nvPmaEvictPagesCallback, nvPmaEvictRangeCallback, void *);
void       NV_API_CALL  rm_gpu_ops_pma_unregister_callbacks(nvidia_stack_t *sp, void *);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_sys(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_get_p2p_caps(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, nvgpuP2PCapsParams_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_map(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, NvLength, void **, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_ummap(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, void*);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_allocate(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuChannelHandle_t *, const nvgpuChannelAllocParams_t *, nvgpuChannelInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_destroy(nvidia_stack_t *, nvgpuChannelHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_copy_engine_alloc(nvidia_stack_t *, nvgpuChannelHandle_t, NvU32, nvgpuObjectHandle_t *, nvgpuChannelInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_memory_free(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64);
NV_STATUS  NV_API_CALL rm_gpu_ops_query_caps(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuCaps_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_gpu_info(nvidia_stack_t *, const NvProcessorUuid *pUuid, nvgpuInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_service_device_interrupts_rm(nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_dup_allocation(nvidia_stack_t *, NvHandle, nvgpuAddressSpaceHandle_t, NvU64, nvgpuAddressSpaceHandle_t, NvU64*, NvBool);

NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_memory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, NvHandle *, nvgpuMemoryInfo_t);

NV_STATUS  NV_API_CALL rm_gpu_ops_free_duped_handle(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_fb_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuFbInfo_t);
NV_STATUS NV_API_CALL rm_gpu_ops_own_page_fault_intr(nvidia_stack_t *, const NvProcessorUuid *, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_fault_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_fault_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, void *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_ops_has_pending_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, NvBool *);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_access_cntr_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_access_cntr_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_own_access_cntr_intr(nvidia_stack_t *, nvgpuSessionHandle_t, nvgpuAccessCntrInfo_t, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_enable_access_cntr(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAccessCntrInfo_t, nvgpuAccessCntrConfig_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_disable_access_cntr(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_set_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, unsigned, NvBool);
NV_STATUS  NV_API_CALL  rm_gpu_ops_unset_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_p2p_object_create(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, NvHandle *);
void       NV_API_CALL rm_gpu_ops_p2p_object_destroy(nvidia_stack_t *, nvgpuSessionHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_external_alloc_ptes(nvidia_stack_t*, nvgpuAddressSpaceHandle_t, NvHandle, NvU64, NvU64, nvgpuExternalMappingInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_retain_channel(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, void **, nvgpuChannelInstanceInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_retain_channel_resources(nvidia_stack_t *, void *, nvgpuChannelResourceInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_bind_channel_resources(nvidia_stack_t *, void *, nvgpuChannelResourceBindParams_t);
void       NV_API_CALL rm_gpu_ops_release_channel(nvidia_stack_t *, void *);
void       NV_API_CALL rm_gpu_ops_release_channel_resources(nvidia_stack_t *, NvP64*, NvU32);
void       NV_API_CALL rm_gpu_ops_stop_channel(nvidia_stack_t *, void *, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_channel_resource_ptes(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvP64, NvU64, NvU64, nvgpuExternalMappingInfo_t);
void       NV_API_CALL rm_kernel_rmapi_op(nvidia_stack_t *sp, void *ops_cmd);
NvBool     NV_API_CALL rm_get_device_remove_flag(nvidia_stack_t *sp, NvU32 gpu_id);
NV_STATUS  NV_API_CALL rm_gpu_copy_mmu_faults(nvidia_stack_t *, nv_state_t *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_copy_mmu_faults_unlocked(nvidia_stack_t *, nv_state_t *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_need_4k_page_isolation(nv_state_t *, NvBool *);
NvBool     NV_API_CALL rm_is_chipset_io_coherent(nv_stack_t *);
NvBool     NV_API_CALL rm_init_event_locks(nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL rm_destroy_event_locks(nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL rm_get_gpu_numa_info(nvidia_stack_t *, nv_state_t *, NvS32 *, NvU64 *, NvU64 *);
NV_STATUS  NV_API_CALL rm_set_backlight(nvidia_stack_t *, nv_state_t *, NvU32, NvU32);
NV_STATUS  NV_API_CALL rm_get_backlight(nvidia_stack_t *, nv_state_t *, NvU32, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_numa_online(nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL rm_gpu_numa_offline(nvidia_stack_t *, nv_state_t *);

/* vGPU VFIO specific functions */
NV_STATUS  NV_API_CALL  nv_vgpu_create_request(nvidia_stack_t *, nv_state_t *, NvU8 *, NvU32, NvU16 *);
NV_STATUS  NV_API_CALL  nv_vgpu_delete(nvidia_stack_t *, NvU8 *, NvU16);
NV_STATUS  NV_API_CALL  nv_vgpu_get_type_ids(nvidia_stack_t *, nv_state_t *, NvU32 *, NvU32 **);
NV_STATUS  NV_API_CALL  nv_vgpu_get_type_info(nvidia_stack_t *, nv_state_t *, NvU32, NvU8 *, int);
NV_STATUS  NV_API_CALL  nv_vgpu_get_bar_info(nvidia_stack_t *, nv_state_t *, NvU8 *, NvU64 *, NvU32, void *, void **);
NV_STATUS  NV_API_CALL  nv_vgpu_start(nvidia_stack_t *, NvU8 *, void *, NvS32 *, NvU8 *, NvU32, NvU8 *);
NV_STATUS  NV_API_CALL  nv_vgpu_get_sparse_mmap(nvidia_stack_t *, nv_state_t *, NvU8 *, NvU64 **, NvU64 **, NvU32 *);
NV_STATUS  NV_API_CALL  nv_vgpu_update_request(nvidia_stack_t *, NvU8 *);
NV_STATUS  NV_API_CALL  nv_gpu_bind_event(nvidia_stack_t *);

NV_STATUS NV_API_CALL nv_get_usermap_access_params(nv_state_t*, nv_usermap_access_params_t*);
#endif /* NVWATCH */

#endif /* NVRM */

static inline int nv_count_bits(NvU64 word)
{
    NvU64 bits;

    bits = (word & 0x5555555555555555ULL) + ((word >>  1) & 0x5555555555555555ULL);
    bits = (bits & 0x3333333333333333ULL) + ((bits >>  2) & 0x3333333333333333ULL);
    bits = (bits & 0x0f0f0f0f0f0f0f0fULL) + ((bits >>  4) & 0x0f0f0f0f0f0f0f0fULL);
    bits = (bits & 0x00ff00ff00ff00ffULL) + ((bits >>  8) & 0x00ff00ff00ff00ffULL);
    bits = (bits & 0x0000ffff0000ffffULL) + ((bits >> 16) & 0x0000ffff0000ffffULL);
    bits = (bits & 0x00000000ffffffffULL) + ((bits >> 32) & 0x00000000ffffffffULL);

    return (int)(bits);
}

#endif
