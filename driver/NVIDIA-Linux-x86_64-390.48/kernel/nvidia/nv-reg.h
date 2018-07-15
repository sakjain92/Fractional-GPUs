/*
 * _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2013 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _RM_REG_H_
#define _RM_REG_H_

#include "nvtypes.h"

/*
 * use NV_REG_STRING to stringify a registry key when using that registry key 
 */

#define __NV_REG_STRING(regkey)  #regkey
#define NV_REG_STRING(regkey)  __NV_REG_STRING(regkey)

/* 
 * use NV_DEFINE_REG_ENTRY and NV_DEFINE_PARAMS_TABLE_ENTRY to simplify definition
 * of registry keys in the kernel module source code.
 */

#define __NV_REG_VAR(regkey)  NVreg_##regkey

#if defined(NV_MODULE_PARAMETER)
#define NV_DEFINE_REG_ENTRY(regkey, default_value)        \
    static NvU32 __NV_REG_VAR(regkey) = (default_value);  \
    NV_MODULE_PARAMETER(__NV_REG_VAR(regkey))
#define NV_DEFINE_REG_ENTRY_GLOBAL(regkey, default_value) \
    NvU32 __NV_REG_VAR(regkey) = (default_value);         \
    NV_MODULE_PARAMETER(__NV_REG_VAR(regkey))
#else
#define NV_DEFINE_REG_ENTRY(regkey, default_value)        \
    static NvU32 __NV_REG_VAR(regkey) = (default_value)
#define NV_DEFINE_REG_ENTRY_GLOBAL(regkey, default_value) \
    NvU32 __NV_REG_VAR(regkey) = (default_value)
#endif

#if defined(NV_MODULE_STRING_PARAMETER)
#define NV_DEFINE_REG_STRING_ENTRY(regkey, default_value) \
    char *__NV_REG_VAR(regkey) = (default_value);         \
    NV_MODULE_STRING_PARAMETER(__NV_REG_VAR(regkey))
#else
#define NV_DEFINE_REG_STRING_ENTRY(regkey, default_value) \
    char *__NV_REG_VAR(regkey) = (default_value)
#endif

#define NV_DEFINE_PARAMS_TABLE_ENTRY(regkey) \
    { "NVreg", NV_REG_STRING(regkey), &__NV_REG_VAR(regkey) }


/*
 *----------------- registry key definitions--------------------------
 */

/*
 * Option: Mobile
 *
 * Description:
 *
 * The Mobile registry key should only be needed on mobile systems if
 * SoftEDIDs is disabled (see above), in which case the mobile value
 * will be used to lookup the correct EDID for the mobile LCD.
 *
 * Possible Values:
 *
 *  ~0 = auto detect the correct value (default)
 *   1 = Dell notebooks
 *   2 = non-Compal Toshiba
 *   3 = all other notebooks
 *   4 = Compal/Toshiba
 *   5 = Gateway
 *
 * Make sure to specify the correct value for your notebook.
 */

#define __NV_MOBILE Mobile
#define NV_REG_MOBILE NV_REG_STRING(__NV_MOBILE)

/*
 * Option: ModifyDeviceFiles
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA driver will verify the validity
 * of the NVIDIA device files in /dev and attempt to dynamically modify
 * and/or (re-)create them, if necessary. If you don't wish for the NVIDIA
 * driver to touch the device files, you can use this registry key.
 *
 * Possible Values:
 *  0 = disable dynamic device file management
 *  1 = enable  dynamic device file management (default)
 */

#define __NV_MODIFY_DEVICE_FILES ModifyDeviceFiles
#define NV_REG_MODIFY_DEVICE_FILES NV_REG_STRING(__NV_MODIFY_DEVICE_FILES)

/*
 * Option: DeviceFileUID
 *
 * Description:
 *
 * This registry key specifies the UID assigned to the NVIDIA device files
 * created and/or modified by the NVIDIA driver when dynamic device file
 * management is enabled.
 *
 * The default UID is 0 ('root').
 */

#define __NV_DEVICE_FILE_UID DeviceFileUID
#define NV_REG_DEVICE_FILE_UID NV_REG_STRING(__NV_DEVICE_FILE_UID)

/*
 * Option: DeviceFileGID
 *
 * Description:
 *
 * This registry key specifies the GID assigned to the NVIDIA device files
 * created and/or modified by the NVIDIA driver when dynamic device file
 * management is enabled.
 *
 * The default GID is 0 ('root').
 */

#define __NV_DEVICE_FILE_GID DeviceFileGID
#define NV_REG_DEVICE_FILE_GID NV_REG_STRING(__NV_DEVICE_FILE_GID)

/*
 * Option: DeviceFileMode
 *
 * Description:
 *
 * This registry key specifies the device file mode assigned to the NVIDIA
 * device files created and/or modified by the NVIDIA driver when dynamic
 * device file management is enabled.
 *
 * The default mode is 0666 (octal, rw-rw-rw-).
 */

#define __NV_DEVICE_FILE_MODE DeviceFileMode
#define NV_REG_DEVICE_FILE_MODE NV_REG_STRING(__NV_DEVICE_FILE_MODE)

/*
 * Option: ResmanDebugLevel
 *
 * Default value: ~0
 */

#define __NV_RESMAN_DEBUG_LEVEL ResmanDebugLevel
#define NV_REG_RESMAN_DEBUG_LEVEL NV_REG_STRING(__NV_RESMAN_DEBUG_LEVEL)

/*
 * Option: RmLogonRC
 *
 * Default value: 1
 */

#define __NV_RM_LOGON_RC RmLogonRC
#define NV_REG_RM_LOGON_RC NV_REG_STRING(__NV_RM_LOGON_RC)

/*
 * Option: UpdateMemoryTypes
 *
 * Description:
 *
 * Many kernels have broken implementations of the change_page_attr()
 * kernel interface that may cause cache aliasing problems. Linux/x86-64
 * kernels between 2.6.0 and 2.6.10 may prompt kernel BUG()s due to
 * improper accounting in the interface's large page management code, for
 * example. For this reason, the NVIDIA Linux driver is very careful about
 * not using the change_page_attr() kernel interface on these kernels.
 *
 * Due to the serious nature of the problems that can arise from bugs in
 * the change_page_attr(), set_pages_{uc,wb}() and other kernel interfaces
 * used to modify memory types, the NVIDIA driver implements a manual
 * registry key override to allow forcibly enabling or disabling use of
 * these APIs.
 *
 * Possible values:
 *
 * ~0 = use the NVIDIA driver's default logic (default)
 *  1 = enable use of change_page_attr(), etc.
 *  0 = disable use of change_page_attr(), etc.
 *
 * By default, the NVIDIA driver will attempt to auto-detect if it can
 * safely use the change_page_attr() and other kernel interfaces to modify
 * the memory types of kernel mappings.
 */

#define __NV_UPDATE_MEMORY_TYPES UpdateMemoryTypes
#define NV_REG_UPDATE_MEMORY_TYPES NV_REG_STRING(__NV_UPDATE_MEMORY_TYPES)

/*
 * Option: InitializeSystemMemoryAllocations
 *
 * Description:
 *
 * The NVIDIA Linux driver normally clears system memory it allocates
 * for use with GPUs or within the driver stack. This is to ensure
 * that potentially sensitive data is not rendered accessible by
 * arbitrary user applications.
 *
 * Owners of single-user systems or similar trusted configurations may
 * choose to disable the aforementioned clears using this option and
 * potentially improve performance.
 *
 * Possible values:
 *
 *  1 = zero out system memory allocations (default)
 *  0 = do not perform memory clears
 */

#define __NV_INITIALIZE_SYSTEM_MEMORY_ALLOCATIONS \
    InitializeSystemMemoryAllocations
#define NV_REG_INITIALIZE_SYSTEM_MEMORY_ALLOCATIONS \
    NV_REG_STRING(__NV_INITIALIZE_SYSTEM_MEMORY_ALLOCATIONS)

/*
 * Option: RegistryDwords
 *
 * Description:
 *
 * This option accepts a semicolon-separated list of key=value pairs. Each
 * key name is checked agains the table of static options; if a match is
 * found, the static option value is overridden, but invalid options remain
 * invalid. Pairs that do not match an entry in the static option table
 * are passed on to the RM directly.
 *
 * Format:
 *
 *  NVreg_RegistryDwords="<key=value>;<key=value>;..."
 */

#define __NV_REGISTRY_DWORDS RegistryDwords
#define NV_REG_REGISTRY_DWORDS NV_REG_STRING(__NV_REGISTRY_DWORDS)

/*
 * Option: RegistryDwordsPerDevice
 *
 * Description:
 *
 * This option allows to specify registry keys per GPU device. It helps to
 * control registry at GPU level of granularity. It accepts a semicolon
 * separated list of key=value pairs. The first key value pair MUST be
 * "pci=DDDD:BB:DD.F;" where DDDD is Domain, BB is Bus Id, DD is device slot
 * number and F is the Function. This PCI BDF is used to identify which GPU to
 * assign the registry keys that follows next.
 * If a GPU corresponding to the value specified in "pci=DDDD:BB:DD.F;" is NOT
 * found, then all the registry keys that follows are skipped, until we find next
 * valid pci identified "pci=DDDD:BB:DD.F;". Following are the valid formats for
 * the value of the "pci" string:
 * 1)  bus:slot                 : Domain and function defaults to 0.
 * 2)  domain:bus:slot          : Function defaults to 0.
 * 3)  domain:bus:slot.func     : Complete PCI dev id string.
 *
 * For each of the registry keys that follows, key name is checked against the
 * table of static options; if a match is found, the static option value is
 * overridden, but invalid options remain invalid. Pairs that do not match an
 * entry in the static option table are passed on to the RM directly.
 *
 * Format:
 *
 *  NVreg_RegistryDwordsPerDevice="pci=DDDD:BB:DD.F;<key=value>;<key=value>;..; \
 *               pci=DDDD:BB:DD.F;<key=value>;..;"
 */

#define __NV_REGISTRY_DWORDS_PER_DEVICE RegistryDwordsPerDevice
#define NV_REG_REGISTRY_DWORDS_PER_DEVICE NV_REG_STRING(__NV_REGISTRY_DWORDS_PER_DEVICE)

#define __NV_RM_MSG RmMsg
#define NV_RM_MSG NV_REG_STRING(__NV_RM_MSG)

/*
 * Option: UsePageAttributeTable
 *
 * Description:
 *
 * Enable/disable use of the page attribute table (PAT) available in
 * modern x86/x86-64 processors to set the effective memory type of memory
 * mappings to write-combining (WC).
 *
 * If enabled, an x86 processor with PAT support is present and the host
 * system's Linux kernel did not configure one of the PAT entries to
 * indicate the WC memory type, the driver will change the second entry in
 * the PAT from its default (write-through (WT)) to WC at module load
 * time. If the kernel did update one of the PAT entries, the driver will
 * not modify the PAT.
 *
 * In both cases, the driver will honor attempts to map memory with the WC
 * memory type by selecting the appropriate PAT entry using the correct
 * set of PTE flags.
 *
 * Possible values:
 *
 * ~0 = use the NVIDIA driver's default logic (default)
 *  1 = enable use of the PAT for WC mappings.
 *  0 = disable use of the PAT for WC mappings.
 */

#define __NV_USE_PAGE_ATTRIBUTE_TABLE UsePageAttributeTable
#define NV_USE_PAGE_ATTRIBUTE_TABLE NV_REG_STRING(__NV_USE_PAGE_ATTRIBUTE_TABLE)

/*  
 * Option: EnableMSI
 *
 * Description: 
 *
 * When this option is enabled and the host kernel supports the MSI feature,
 * the NVIDIA driver will enable the PCI-E MSI capability of GPUs with the 
 * support for this feature instead of using PCI-E wired interrupt.
 *
 * Possible Values:
 *
 *  0 = disable MSI interrupt
 *  1 = enable MSI interrupt (default)
 *
 */

#define __NV_ENABLE_MSI EnableMSI
#define NV_REG_ENABLE_MSI NV_REG_STRING(__NV_ENABLE_MSI)

/*
 * Option: MapRegistersEarly
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA kernel module will attempt to
 * map the device registers of NVIDIA GPUs at probe(), rather than at
 * open() time. This is useful for debugging purposes, only.
 *
 * Possible Values:
 *
 *   0 = do not map GPU registers early (default)
 *   1 = map GPU registers early
 */

#define __NV_MAP_REGISTERS_EARLY MapRegistersEarly
#define NV_REG_MAP_REGISTERS_EARLY NV_REG_STRING(__NV_MAP_REGISTERS_EARLY)

/*
 * Option: RegisterForACPIEvents
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA driver will register with the
 * ACPI subsystem to receive notification of ACPI events.
 *
 * Possible values:
 *
 *  1 - register for ACPI events (default)
 *  0 - do not register for ACPI events
 */

#define __NV_REGISTER_FOR_ACPI_EVENTS  RegisterForACPIEvents
#define NV_REG_REGISTER_FOR_ACPI_EVENTS NV_REG_STRING(__NV_REGISTER_FOR_ACPI_EVENTS)

/*
 * Option: CheckPCIConfigSpace
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA driver will perform checks upon
 * entry to system calls, interrupts and various callbacks to determine
 * if key registers in a given GPU's PCI configuration space, such as
 * the PCI command register and BARs, were altered by third-party
 * software.
 *
 * Possible values:
 *
 *  1 - perform PCI configuration space checks (default on non-virtualized system)
 *  0 - do not perform PCI configuration space checks (default on virtual machine)
 *
 *  Dummy value:
 *  0xFFFFFFFF: It is only used to detect if user has over-ridden the value.
 *              We should never see this value after registry init.
 */

#define __NV_CHECK_PCI_CONFIG_SPACE  CheckPCIConfigSpace
#define NV_REG_CHECK_PCI_CONFIG_SPACE NV_REG_STRING(__NV_CHECK_PCI_CONFIG_SPACE)

#define NV_CHECK_PCI_CONFIG_SPACE_DISABLED          0
#define NV_CHECK_PCI_CONFIG_SPACE_ENABLED           1
#define NV_CHECK_PCI_CONFIG_SPACE_INIT     0xFFFFFFFF

/*
 * Option: AssignGpus
 *
 * Description:
 *
 * This option accepts string of pci bus locations of GPUs, seperated by comma,
 * that NVIDIA module should probe. GPU PCI bus location string should be in any
 * of the below formats, with domain, bus and slot numbers in hex:
 *   bus:slot
 *   domain:bus:slot
 *   domain:bus:slot.func
 * By default, this option is not set. When this option is not set, all
 * supported NVIDIA GPUs are assigned to the NVIDIA kernel module.
 */

#define __NV_ASSIGN_GPUS AssignGpus
#define NV_REG_ASSIGN_GPUS NV_REG_STRING(__NV_ASSIGN_GPUS)

/*
 * Option: EnablePCIeGen3
 *
 * Description:
 *
 * Due to interoperability problems seen with Kepler PCIe Gen3 capable GPUs
 * when configured on SandyBridge E desktop platforms, NVIDIA feels that
 * delivering a reliable, high-quality experience is not currently possible in
 * PCIe Gen3 mode on all PCIe Gen3 platforms. Therefore, Quadro, Tesla and
 * NVS Kepler products operate in PCIe Gen2 mode by default. You may use this
 * option to enable PCIe Gen3 support.
 *
 * This is completely unsupported!
 *
 * Possible Values:
 *
 *  0: disable PCIe Gen3 support (default)
 *  1: enable PCIe Gen3 support
 */

#define __NV_ENABLE_PCIE_GEN3 EnablePCIeGen3
#define NV_REG_ENABLE_PCIE_GEN3 NV_REG_STRING(__NV_ENABLE_PCIE_GEN3)

/*
 * Option: MemoryPoolSize
 *
 * Description:
 *
 * When set to a non-zero value, this option specifies the size of the
 * memory pool, given as a multiple of 1 GB, created on VMware ESXi to
 * satisfy any system memory allocations requested by the NVIDIA kernel
 * module.
 */

#define __NV_MEMORY_POOL_SIZE MemoryPoolSize
#define NV_REG_MEMORY_POOL_SIZE NV_REG_STRING(__NV_MEMORY_POOL_SIZE)

/*
 * Option: IgnoreMMIOCheck
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA kernel module will ignore
 * MMIO limit check during device probe on VMWare ESXi kernel. This is
 * typically necessary when VMware ESXi MMIO limit differs between any
 * base version and its updates. Customer using updates can set regkey
 * to avoid probe failure.
 */

#define __NV_IGNORE_MMIO_CHECK IgnoreMMIOCheck
#define NV_REG_IGNORE_MMIO_CHECK NV_REG_STRING(__NV_IGNORE_MMIO_CHECK)

/*
 * Option: TCEBypassMode
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA kernel module will attempt to setup
 * all GPUs in "TCE bypass mode", in which DMA mappings of system memory bypass
 * the IOMMU/TCE remapping hardware on IBM POWER systems. This is typically
 * necessary for CUDA applications in which large system memory mappings may
 * exceed the default TCE remapping capacity when operated in non-bypass mode.
 *
 * This option has no effect on non-POWER platforms.
 *
 * Possible Values:
 * 
 *  0: system default TCE mode on all GPUs
 *  1: enable TCE bypass mode on all GPUs
 *  2: disable TCE bypass mode on all GPUs
 */
#define __NV_TCE_BYPASS_MODE TCEBypassMode
#define NV_REG_TCE_BYPASS_MODE NV_REG_STRING(__NV_TCE_BYPASS_MODE)

#define NV_TCE_BYPASS_MODE_DEFAULT  0
#define NV_TCE_BYPASS_MODE_ENABLE   1
#define NV_TCE_BYPASS_MODE_DISABLE  2

/*
 * Option: UseThreadedInterrupts
 *
 * Description:
 *
 * When this option is enabled and the host kernel supports the
 * request_threaded_irq feature, the NVIDIA driver will call
 * request_threaded_irq and will therefore NOT use tasklets for the bottom half
 * ISR handler. This is instead of: calling request_irq and using tasklets.
 *
 * So far, this only affects GPU interrupts. Other device interrupts, such as
 * the GVI device, could eventually honor this mechanism, but they do not right
 * now.
 *
 * Possible Values:
 *
 *  0 = use request_irq to set up the interrupt handler.
 *  1 = use request_threaded_irq to set up the interrupt handler (default)
 */
#define __NV_USE_THREADED_INTERRUPTS UseThreadedInterrupts
#define NV_REG_USE_THREADED_INTERRUPTS NV_REG_STRING(__NV_USE_THREADED_INTERRUPTS)

/*
 * Option: pci
 *
 * Description:
 *
 * On Unix platforms, per GPU based registry key can be specified as: 
 * NVreg_RegistryDwordsPerDevice="pci=DDDD:BB:DD.F,<per-gpu registry keys>".
 * where DDDD:BB:DD.F refers to Domain:Bus:Device.Function.
 * We need this key "pci" to identify what follows next is a PCI BDF identifier, 
 * for which the registry keys are to be applied.
 *
 * This define is not used on non-UNIX platforms.
 *
 * Possible Formats for value:
 *
 * 1)  bus:slot                 : Domain and function defaults to 0.
 * 2)  domain:bus:slot          : Function defaults to 0.
 * 3)  domain:bus:slot.func     : Complete PCI BDF identifier string.
 */
#define __NV_PCI_DEVICE_BDF pci
#define NV_REG_PCI_DEVICE_BDF NV_REG_STRING(__NV_PCI_DEVICE_BDF)

/*
 * Option: EnableStreamMemOPs
 *
 * Description:
 *
 * When this option is enabled, the CUDA driver will enable support for
 * CUDA Stream Memory Operations in user-mode applications, which are so
 * far required to be disabled by default due to limited support in
 * devtools.
 *
 * Note: this is treated as a hint. MemOPs may still be left disabled by CUDA
 * driver for other reasons.
 *
 * Possible Values:
 *
 *  0 = disable feature (default)
 *  1 = enable feature
 */
#define __NV_ENABLE_STREAM_MEMOPS EnableStreamMemOPs
#define NV_REG_ENABLE_STREAM_MEMOPS NV_REG_STRING(__NV_ENABLE_STREAM_MEMOPS)

/*
 * Option: EnableBacklightHandler
 *
 * Description:
 *
 * When this option is enabled and the platform supports it, the NVIDIA driver
 * will register a backlight handler with the operating system.
 *
 * The backlight handler will be registered when the corresponding device is
 * initialized, and will be removed when the device is shut down.
 *
 * Possible Values:
 *
 *  0 = disable feature (default)
 *  1 = enable feature
 */
#define __NV_ENABLE_BACKLIGHT_HANDLER EnableBacklightHandler
#define NV_REG_ENABLE_BACKLIGHT_HANDLER NV_REG_STRING(__NV_ENABLE_BACKLIGHT_HANDLER)

/*
 * Option: EnableUserNUMAManagement
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA kernel module will require the
 * user-mode NVIDIA Persistence daemon to manage the onlining and offlining
 * of its NUMA device memory.
 *
 * This option has no effect on platforms that do not support onlining
 * device memory to a NUMA node (this feature is only supported on certain
 * POWER9 systems).
 *
 * Possible Values:
 * 
 *  0: disable user-mode NUMA management
 *  1: enable user-mode NUMA management (default)
 */
#define __NV_ENABLE_USER_NUMA_MANAGEMENT EnableUserNUMAManagement
#define NV_REG_ENABLE_USER_NUMA_MANAGEMENT NV_REG_STRING(__NV_ENABLE_USER_NUMA_MANAGEMENT)

/*
 * Option: EnableIBMNPURelaxedOrderingMode
 *
 * Description:
 *
 * When this option is enabled, the NVIDIA kernel module will start issuing
 * necessary barriers/synchronizations to ensure I/O consistency on all the
 * registered and compatible NPU devices.
 *
 * Note: this option is intended for testing purposes only and will be removed
 * in a future release.
 *
 * Possible Values:
 * 
 *  0: disable relaxed ordering (default)
 *  1: enable relaxed ordering mode
 */
#define __NV_ENABLE_IBMNPU_RELAXED_ORDERING_MODE EnableIBMNPURelaxedOrderingMode
#define NV_REG_ENABLE_IBMNPU_RELAXED_ORDERING_MODE NV_REG_STRING(__NV_ENABLE_IBMNPU_RELAXED_ORDERING_MODE)

#if defined(NV_DEFINE_REGISTRY_KEY_TABLE)

/*
 *---------registry key parameter declarations--------------
 */

NV_DEFINE_REG_ENTRY(__NV_MOBILE, ~0);
NV_DEFINE_REG_ENTRY(__NV_RESMAN_DEBUG_LEVEL, ~0);
NV_DEFINE_REG_ENTRY(__NV_RM_LOGON_RC, 1);
NV_DEFINE_REG_ENTRY(__NV_MODIFY_DEVICE_FILES, 1);
NV_DEFINE_REG_ENTRY(__NV_DEVICE_FILE_UID, 0);
NV_DEFINE_REG_ENTRY(__NV_DEVICE_FILE_GID, 0);
NV_DEFINE_REG_ENTRY(__NV_DEVICE_FILE_MODE, 0666);
NV_DEFINE_REG_ENTRY(__NV_UPDATE_MEMORY_TYPES, ~0);
NV_DEFINE_REG_ENTRY(__NV_INITIALIZE_SYSTEM_MEMORY_ALLOCATIONS, 1);
NV_DEFINE_REG_ENTRY(__NV_USE_PAGE_ATTRIBUTE_TABLE, ~0);
NV_DEFINE_REG_ENTRY(__NV_MAP_REGISTERS_EARLY, 0);
NV_DEFINE_REG_ENTRY(__NV_REGISTER_FOR_ACPI_EVENTS, 1);
NV_DEFINE_REG_ENTRY(__NV_CHECK_PCI_CONFIG_SPACE, NV_CHECK_PCI_CONFIG_SPACE_INIT);
NV_DEFINE_REG_ENTRY(__NV_ENABLE_PCIE_GEN3, 0);
NV_DEFINE_REG_ENTRY(__NV_ENABLE_MSI, 1);
NV_DEFINE_REG_ENTRY(__NV_TCE_BYPASS_MODE, NV_TCE_BYPASS_MODE_DEFAULT);
NV_DEFINE_REG_ENTRY(__NV_USE_THREADED_INTERRUPTS, 1);
NV_DEFINE_REG_ENTRY(__NV_ENABLE_STREAM_MEMOPS, 0);
NV_DEFINE_REG_ENTRY(__NV_ENABLE_BACKLIGHT_HANDLER, 0);
NV_DEFINE_REG_ENTRY_GLOBAL(__NV_ENABLE_USER_NUMA_MANAGEMENT, 1);
NV_DEFINE_REG_ENTRY_GLOBAL(__NV_ENABLE_IBMNPU_RELAXED_ORDERING_MODE, 0);
NV_DEFINE_REG_ENTRY_GLOBAL(__NV_MEMORY_POOL_SIZE, 0);
NV_DEFINE_REG_ENTRY_GLOBAL(__NV_IGNORE_MMIO_CHECK, 0);

NV_DEFINE_REG_STRING_ENTRY(__NV_REGISTRY_DWORDS, NULL);
NV_DEFINE_REG_STRING_ENTRY(__NV_REGISTRY_DWORDS_PER_DEVICE, NULL);
NV_DEFINE_REG_STRING_ENTRY(__NV_RM_MSG, NULL);

#if !defined(NV_VMWARE)
NV_DEFINE_REG_STRING_ENTRY(__NV_ASSIGN_GPUS, NULL);
#endif

/*
 *----------------registry database definition----------------------
 */

/*
 * You can enable any of the registry options disabled by default by
 * editing their respective entries in the table below. The last field
 * determines if the option is considered valid - in order for the
 * changes to take effect, you need to recompile and reload the NVIDIA
 * kernel module.
 */
nv_parm_t nv_parms[] = {
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_MOBILE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_RESMAN_DEBUG_LEVEL),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_RM_LOGON_RC),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_MODIFY_DEVICE_FILES),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_DEVICE_FILE_UID),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_DEVICE_FILE_GID),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_DEVICE_FILE_MODE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_UPDATE_MEMORY_TYPES),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_INITIALIZE_SYSTEM_MEMORY_ALLOCATIONS),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_USE_PAGE_ATTRIBUTE_TABLE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_MSI),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_MAP_REGISTERS_EARLY),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_REGISTER_FOR_ACPI_EVENTS),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_CHECK_PCI_CONFIG_SPACE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_PCIE_GEN3),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_MEMORY_POOL_SIZE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_IGNORE_MMIO_CHECK),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_TCE_BYPASS_MODE),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_USE_THREADED_INTERRUPTS),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_STREAM_MEMOPS),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_BACKLIGHT_HANDLER),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_USER_NUMA_MANAGEMENT),
    NV_DEFINE_PARAMS_TABLE_ENTRY(__NV_ENABLE_IBMNPU_RELAXED_ORDERING_MODE),
    {NULL, NULL, NULL}
};

#elif defined(NVRM)

extern nv_parm_t nv_parms[];

#endif /* NV_DEFINE_REGISTRY_KEY_TABLE */

#endif /* _RM_REG_H_ */
