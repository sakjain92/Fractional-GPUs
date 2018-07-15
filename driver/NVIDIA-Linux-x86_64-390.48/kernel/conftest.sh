#!/bin/sh

PATH="${PATH}:/bin:/sbin:/usr/bin"

# make sure we are in the directory containing this script
SCRIPTDIR=`dirname $0`
cd $SCRIPTDIR

#
# HOSTCC vs. CC - if a conftest needs to build and execute a test
# binary, like get_uname, then $HOSTCC needs to be used for this
# conftest in order for the host/build system to be able to execute
# it in X-compile environments.
# In all other cases, $CC should be used to minimize the risk of
# false failures due to conflicts with architecture specific header
# files.
#
CC="$1"
HOSTCC="$2"
ARCH=$3
ISYSTEM=`$CC -print-file-name=include 2> /dev/null`
SOURCES=$4
HEADERS=$SOURCES/include
OUTPUT=$5
XEN_PRESENT=1
PREEMPT_RT_PRESENT=0
KERNEL_ARCH="$ARCH"

if [ "$ARCH" = "i386" -o "$ARCH" = "x86_64" ]; then
    if [ -d "$SOURCES/arch/x86" ]; then
        KERNEL_ARCH="x86"
    fi
fi

HEADERS_ARCH="$SOURCES/arch/$KERNEL_ARCH/include"

# VGX_BUILD parameter defined only for VGX builds (vGPU Host driver)
# VGX_KVM_BUILD parameter defined only vGPU builds on KVM hypervisor
# GRID_BUILD parameter defined only for GRID builds (GRID Guest driver)

test_xen() {
    #
    # Determine if the target kernel is a Xen kernel. It used to be
    # sufficient to check for CONFIG_XEN, but the introduction of
    # modular para-virtualization (CONFIG_PARAVIRT, etc.) and
    # Xen guest support, it is no longer possible to determine the
    # target environment at build time. Therefore, if both
    # CONFIG_XEN and CONFIG_PARAVIRT are present, text_xen() treats
    # the kernel as a stand-alone kernel.
    #
    if ! test_configuration_option CONFIG_XEN ||
         test_configuration_option CONFIG_PARAVIRT; then
        XEN_PRESENT=0
    fi
}

append_conftest() {
    #
    # Echo data from stdin: this is a transitional function to make it easier
    # to port conftests from drivers with parallel conftest generation to
    # older driver versions
    #

    while read LINE; do
        echo ${LINE}
    done
}

translate_and_find_header_files() {
    # Inputs:
    #   $1: a parent directory (full path), in which to search
    #   $2: a list of relative file paths
    #
    # This routine creates an upper case, underscore version of each of the
    # relative file paths, and uses that as the token to either define or
    # undefine in a C header file. For example, linux/fence.h becomes
    # NV_LINUX_FENCE_H_PRESENT, and that is either defined or undefined, in the
    # output (which goes to stdout, just like the rest of this file).

    local parent_dir=$1
    shift

    for file in $@; do
        local file_define=NV_`echo $file | tr '/.' '_' | tr '-' '_' | tr 'a-z' 'A-Z'`_PRESENT
        if [ -f $parent_dir/$file -o -f $OUTPUT/include/$file ]; then
            echo "#define $file_define"
        else
            echo "#undef $file_define"
        fi
    done
}

test_headers() {
    #
    # Determine which header files (of a set that may or may not be
    # present) are provided by the target kernel.
    #
    FILES="asm/system.h"
    FILES="$FILES drm/drmP.h"
    FILES="$FILES drm/drm_auth.h"
    FILES="$FILES drm/drm_gem.h"
    FILES="$FILES drm/drm_crtc.h"
    FILES="$FILES drm/drm_atomic.h"
    FILES="$FILES drm/drm_atomic_helper.h"
    FILES="$FILES drm/drm_encoder.h"
    FILES="$FILES generated/autoconf.h"
    FILES="$FILES generated/compile.h"
    FILES="$FILES generated/utsrelease.h"
    FILES="$FILES linux/efi.h"
    FILES="$FILES linux/kconfig.h"
    FILES="$FILES linux/screen_info.h"
    FILES="$FILES linux/semaphore.h"
    FILES="$FILES linux/printk.h"
    FILES="$FILES linux/ratelimit.h"
    FILES="$FILES linux/prio_tree.h"
    FILES="$FILES linux/log2.h"
    FILES="$FILES linux/of.h"
    FILES="$FILES linux/bug.h"
    FILES="$FILES linux/sched/signal.h"
    FILES="$FILES linux/sched/task.h"
    FILES="$FILES linux/sched/task_stack.h"
    FILES="$FILES xen/ioemu.h"
    FILES="$FILES linux/fence.h"

    # Arch specific headers which need testing
    FILES_ARCH="asm/book3s/64/hash-64k.h"
    FILES_ARCH="$FILES_ARCH asm/set_memory.h"
    FILES_ARCH="$FILES_ARCH asm/powernv.h"
    FILES_ARCH="$FILES_ARCH asm/tlbflush.h"

    translate_and_find_header_files $HEADERS      $FILES
    translate_and_find_header_files $HEADERS_ARCH $FILES_ARCH
}

build_cflags() {
    BASE_CFLAGS="-O2 -D__KERNEL__ \
-DKBUILD_BASENAME=\"#conftest$$\" -DKBUILD_MODNAME=\"#conftest$$\" \
-nostdinc -isystem $ISYSTEM"

    if [ "$OUTPUT" != "$SOURCES" ]; then
        OUTPUT_CFLAGS="-I$OUTPUT/include2 -I$OUTPUT/include"
        if [ -f "$OUTPUT/include/generated/autoconf.h" ]; then
            AUTOCONF_FILE="$OUTPUT/include/generated/autoconf.h"
        else
            AUTOCONF_FILE="$OUTPUT/include/linux/autoconf.h"
        fi
    else
        if [ -f "$HEADERS/generated/autoconf.h" ]; then
            AUTOCONF_FILE="$HEADERS/generated/autoconf.h"
        else
            AUTOCONF_FILE="$HEADERS/linux/autoconf.h"
        fi
    fi

    test_xen

    if [ "$XEN_PRESENT" != "0" ]; then
        MACH_CFLAGS="-I$HEADERS/asm/mach-xen"
    fi

    SOURCE_HEADERS="$HEADERS"
    SOURCE_ARCH_HEADERS="$SOURCES/arch/$KERNEL_ARCH/include"
    OUTPUT_HEADERS="$OUTPUT/include"
    OUTPUT_ARCH_HEADERS="$OUTPUT/arch/$KERNEL_ARCH/include"

    # Look for mach- directories on this arch, and add it to the list of
    # includes if that platform is enabled in the configuration file, which
    # may have a definition like this:
    #   #define CONFIG_ARCH_<MACHUPPERCASE> 1
    for _mach_dir in `ls -1d $SOURCES/arch/$KERNEL_ARCH/mach-* 2>/dev/null`; do
        _mach=`echo $_mach_dir | \
            sed -e "s,$SOURCES/arch/$KERNEL_ARCH/mach-,," | \
            tr 'a-z' 'A-Z'`
        grep "CONFIG_ARCH_$_mach \+1" $AUTOCONF_FILE > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            MACH_CFLAGS="$MACH_CFLAGS -I$_mach_dir/include"
        fi
    done

    if [ "$ARCH" = "arm" ]; then
        MACH_CFLAGS="$MACH_CFLAGS -D__LINUX_ARM_ARCH__=7"
    fi

    # Add the mach-default includes (only found on x86/older kernels)
    MACH_CFLAGS="$MACH_CFLAGS -I$SOURCE_HEADERS/asm-$KERNEL_ARCH/mach-default"
    MACH_CFLAGS="$MACH_CFLAGS -I$SOURCE_ARCH_HEADERS/asm/mach-default"

    CFLAGS="$BASE_CFLAGS $MACH_CFLAGS $OUTPUT_CFLAGS -include $AUTOCONF_FILE"
    CFLAGS="$CFLAGS -I$SOURCE_HEADERS"
    CFLAGS="$CFLAGS -I$SOURCE_HEADERS/uapi"
    CFLAGS="$CFLAGS -I$SOURCE_HEADERS/xen"
    CFLAGS="$CFLAGS -I$OUTPUT_HEADERS/generated/uapi"
    CFLAGS="$CFLAGS -I$SOURCE_ARCH_HEADERS"
    CFLAGS="$CFLAGS -I$SOURCE_ARCH_HEADERS/uapi"
    CFLAGS="$CFLAGS -I$OUTPUT_ARCH_HEADERS/generated"
    CFLAGS="$CFLAGS -I$OUTPUT_ARCH_HEADERS/generated/uapi"

    if [ -n "$BUILD_PARAMS" ]; then
        CFLAGS="$CFLAGS -D$BUILD_PARAMS"
    fi
}

CONFTEST_PREAMBLE="#include \"conftest/headers.h\"
    #if defined(NV_LINUX_KCONFIG_H_PRESENT)
    #include <linux/kconfig.h>
    #endif
    #if defined(NV_GENERATED_AUTOCONF_H_PRESENT)
    #include <generated/autoconf.h>
    #else
    #include <linux/autoconf.h>
    #endif
    #if defined(CONFIG_XEN) && \
        defined(CONFIG_XEN_INTERFACE_VERSION) &&  !defined(__XEN_INTERFACE_VERSION__)
    #define __XEN_INTERFACE_VERSION__ CONFIG_XEN_INTERFACE_VERSION
    #endif"

test_configuration_option() {
    #
    # Check to see if the given configuration option is defined
    #

    get_configuration_option $1 >/dev/null 2>&1

    return $?

}

compile_check_conftest() {
    #
    # Compile the current conftest C file and check+output the result
    #
    CODE="$1"
    DEF="$2"
    VAL="$3"
    CAT="$4"

    echo "$CONFTEST_PREAMBLE
    $CODE" > conftest$$.c

    $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
    rm -f conftest$$.c

    if [ -f conftest$$.o ]; then
        rm -f conftest$$.o
        if [ "${CAT}" = "functions" ]; then
            #
            # The logic for "functions" compilation tests is inverted compared to
            # other compilation steps: if the function is present, the code
            # snippet will fail to compile because the function call won't match
            # the prototype. If the function is not present, the code snippet
            # will produce an object file with the function as an unresolved
            # symbol.
            #
            echo "#undef ${DEF}" | append_conftest "${CAT}"
        else
            echo "#define ${DEF} ${VAL}" | append_conftest "${CAT}"
        fi
        return
    else
        if [ "${CAT}" = "functions" ]; then
            echo "#define ${DEF} ${VAL}" | append_conftest "${CAT}"
        else
            echo "#undef ${DEF}" | append_conftest "${CAT}"
        fi
        return
    fi
}

export_symbol_gpl_conftest() {
    #
    # Check Module.symvers to see whether the given symbol is present and its
    # export type is GPL-only (including deprecated GPL-only symbols).
    #

    SYMBOL="$1"
    TAB='	'

    if grep -e "${TAB}${SYMBOL}${TAB}.*${TAB}EXPORT_\(UNUSED_\)*SYMBOL_GPL\$" \
               "$OUTPUT/Module.symvers" >/dev/null 2>&1; then
        echo "#define NV_IS_EXPORT_SYMBOL_GPL_$SYMBOL 1" |
            append_conftest "symbols"
    else
        # May be a false negative if Module.symvers is absent or incomplete,
        # or if the Module.symvers format changes.
        echo "#define NV_IS_EXPORT_SYMBOL_GPL_$SYMBOL 0" |
            append_conftest "symbols"
    fi
}

get_configuration_option() {
    #
    # Print the value of given configuration option, if defined
    #
    RET=1
    OPTION=$1

    OLD_FILE="linux/autoconf.h"
    NEW_FILE="generated/autoconf.h"
    FILE=""

    if [ -f $HEADERS/$NEW_FILE -o -f $OUTPUT/include/$NEW_FILE ]; then
        FILE=$NEW_FILE
    elif [ -f $HEADERS/$OLD_FILE -o -f $OUTPUT/include/$OLD_FILE ]; then
        FILE=$OLD_FILE
    fi

    if [ -n "$FILE" ]; then
        #
        # We are looking at a configured source tree; verify
        # that its configuration includes the given option
        # via a compile check, and print the option's value.
        #

        if [ -f $HEADERS/$FILE ]; then
            INCLUDE_DIRECTORY=$HEADERS
        elif [ -f $OUTPUT/include/$FILE ]; then
            INCLUDE_DIRECTORY=$OUTPUT/include
        else
            return 1
        fi

        echo "#include <$FILE>
        #ifndef $OPTION
        #error $OPTION not defined!
        #endif

        $OPTION
        " > conftest$$.c

        $CC -E -P -I$INCLUDE_DIRECTORY -o conftest$$ conftest$$.c > /dev/null 2>&1

        if [ -e conftest$$ ]; then
            tr -d '\r\n\t ' < conftest$$
            RET=$?
        fi

        rm -f conftest$$.c conftest$$
    else
        CONFIG=$OUTPUT/.config
        if [ -f $CONFIG ] && grep "^$OPTION=" $CONFIG; then
            grep "^$OPTION=" $CONFIG | cut -f 2- -d "="
            RET=$?
        fi
    fi

    return $RET

}

compile_test() {
    case "$1" in
        set_memory_uc)
            #
            # Determine if the set_memory_uc() function is present.
            #
            CODE="
            #if defined(NV_ASM_SET_MEMORY_H_PRESENT)
            #include <asm/set_memory.h>
            #else
            #include <asm/cacheflush.h>
            #endif
            void conftest_set_memory_uc(void) {
                set_memory_uc();
            }"

            compile_check_conftest "$CODE" "NV_SET_MEMORY_UC_PRESENT" "" "functions"
        ;;

        set_memory_array_uc)
            #
            # Determine if the set_memory_array_uc() function is present.
            #
            CODE="
            #if defined(NV_ASM_SET_MEMORY_H_PRESENT)
            #include <asm/set_memory.h>
            #else
            #include <asm/cacheflush.h>
            #endif
            void conftest_set_memory_array_uc(void) {
                set_memory_array_uc();
            }"

            compile_check_conftest "$CODE" "NV_SET_MEMORY_ARRAY_UC_PRESENT" "" "functions"
        ;;

        set_pages_uc)
            #
            # Determine if the set_pages_uc() function is present.
            #
            CODE="
            #if defined(NV_ASM_SET_MEMORY_H_PRESENT)
            #include <asm/set_memory.h>
            #else
            #include <asm/cacheflush.h>
            #endif
            void conftest_set_pages_uc(void) {
                set_pages_uc();
            }"

            compile_check_conftest "$CODE" "NV_SET_PAGES_UC_PRESENT" "" "functions"
        ;;

        outer_flush_all)
            #
            # Determine if the outer_cache_fns struct has flush_all member.
            #
            CODE="
            #include <asm/outercache.h>
            int conftest_outer_flush_all(void) {
                return offsetof(struct outer_cache_fns, flush_all);
            }"

            compile_check_conftest "$CODE" "NV_OUTER_FLUSH_ALL_PRESENT" "" "types"
        ;;

        change_page_attr)
            #
            # Determine if the change_page_attr() function is
            # present.
            #
            CODE="
            #include <linux/version.h>
            #include <linux/utsname.h>
            #include <linux/mm.h>
            #include <asm/cacheflush.h>
            void conftest_change_page_attr(void) {
                change_page_attr();
            }"

            compile_check_conftest "$CODE" "NV_CHANGE_PAGE_ATTR_PRESENT" "" "functions"
        ;;

        pci_get_class)
            #
            # Determine if the pci_get_class() function is
            # present.
            #
            CODE="
            #include <linux/pci.h>
            void conftest_pci_get_class(void) {
                pci_get_class();
            }"

            compile_check_conftest "$CODE" "NV_PCI_GET_CLASS_PRESENT" "" "functions"
        ;;

        pci_get_domain_bus_and_slot)
            #
            # Determine if the pci_get_domain_bus_and_slot() function
            # is present.
            #
            CODE="
            #include <linux/pci.h>
            void conftest_pci_get_domain_bus_and_slot(void) {
                pci_get_domain_bus_and_slot();
            }"

            compile_check_conftest "$CODE" "NV_PCI_GET_DOMAIN_BUS_AND_SLOT_PRESENT" "" "functions"
        ;;

        pci_save_state)
            #
            # Determine the number of arguments of pci_(save|restore)_state().
            # The explicit buffer argument is only present on 2.6.9. Assume the
            # interface is always present.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/pci.h>
            void conftest_pci_save_state(void) {
                pci_save_state(NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_PCI_SAVE_STATE_ARGUMENT_COUNT 1" | append_conftest "functions"
                rm -f conftest$$.o
                return
            else
                echo "#define NV_PCI_SAVE_STATE_ARGUMENT_COUNT 2" | append_conftest "functions"
                return
            fi
        ;;

        pci_bus_address)
            #
            # Determine if the pci_bus_address() function is
            # present.
            #
            CODE="
            #include <linux/pci.h>
            void conftest_pci_bus_address(void) {
                pci_bus_address();
            }"

            compile_check_conftest "$CODE" "NV_PCI_BUS_ADDRESS_PRESENT" "" "functions"
        ;;

        remap_pfn_range)
            #
            # Determine if the remap_pfn_range() function is
            # present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_remap_pfn_range(void) {
                remap_pfn_range();
            }"

            compile_check_conftest "$CODE" "NV_REMAP_PFN_RANGE_PRESENT" "" "functions"
        ;;

        hash__remap_4k_pfn)
            #
            # Determine if the hash__remap_4k_pfn() function is
            # present.
            # hash__remap_4k_pfn was added by this commit:
            # 2016-04-29  6cc1a0ee4ce29ad1cbdc622db6f9bc16d3056067
            #
            CODE="
            #if defined(NV_ASM_BOOK3S_64_HASH_64K_H_PRESENT)
            #include <linux/mm.h>
            #include <asm/book3s/64/hash-64k.h>
            #endif
            void conftest_hash__remap_4k_pfn(void) {
                hash__remap_4k_pfn();
            }"

            compile_check_conftest "$CODE" "NV_HASH__REMAP_4K_PFN_PRESENT" "" "functions"
        ;;

        follow_pfn)
            #
            # Determine if the follow_pfn() function is
            # present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_follow_pfn(void) {
                follow_pfn();
            }"

            compile_check_conftest "$CODE" "NV_FOLLOW_PFN_PRESENT" "" "functions"
        ;;

        i2c_adapter)
            #
            # Determine if the 'i2c_adapter' structure has the
            # client_register() field.
            #
            CODE="
            #include <linux/i2c.h>
            int conftest_i2c_adapter(void) {
                return offsetof(struct i2c_adapter, client_register);
            }"

            compile_check_conftest "$CODE" "NV_I2C_ADAPTER_HAS_CLIENT_REGISTER" "" "types"
        ;;

        pm_message_t)
            #
            # Determine if the 'pm_message_t' data type is present
            # and if it as an 'event' member.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/pm.h>
            void conftest_pm_message_t(pm_message_t state) {
                pm_message_t *p = &state;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_PM_MESSAGE_T_PRESENT" | append_conftest "types"
                rm -f conftest$$.o
            else
                echo "#undef NV_PM_MESSAGE_T_PRESENT" | append_conftest "types"
                echo "#undef NV_PM_MESSAGE_T_HAS_EVENT" | append_conftest "types"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/pm.h>  
            int conftest_pm_message_t(void) {
                return offsetof(pm_message_t, event);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_PM_MESSAGE_T_HAS_EVENT" | append_conftest "types"
                rm -f conftest$$.o
                return
            else
                echo "#undef NV_PM_MESSAGE_T_HAS_EVENT" | append_conftest "types"
                return
            fi
        ;;

        pci_choose_state)
            #
            # Determine if the pci_choose_state() function is
            # present.
            #
            CODE="
            #include <linux/pci.h>
            void conftest_pci_choose_state(void) {
                pci_choose_state();
            }"

            compile_check_conftest "$CODE" "NV_PCI_CHOOSE_STATE_PRESENT" "" "functions"
        ;;

        vm_insert_page)
            #
            # Determine if the vm_insert_page() function is
            # present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_vm_insert_page(void) {
                vm_insert_page();
            }"

            compile_check_conftest "$CODE" "NV_VM_INSERT_PAGE_PRESENT" "" "functions"
        ;;

        irq_handler_t)
            #
            # Determine if the 'irq_handler_t' type is present and
            # if it takes a 'struct ptregs *' argument.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/interrupt.h>
            irq_handler_t conftest_isr;
            " > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ ! -f conftest$$.o ]; then
                echo "#undef NV_IRQ_HANDLER_T_PRESENT" | append_conftest "types"
                rm -f conftest$$.o
                return
            fi

            rm -f conftest$$.o

            echo "$CONFTEST_PREAMBLE
            #include <linux/interrupt.h>
            irq_handler_t conftest_isr;
            int conftest_irq_handler_t(int irq, void *arg) {
                return conftest_isr(irq, arg);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_IRQ_HANDLER_T_PRESENT" | append_conftest "types"
                echo "#define NV_IRQ_HANDLER_T_ARGUMENT_COUNT 2" | append_conftest "types"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/interrupt.h>
            irq_handler_t conftest_isr;
            int conftest_irq_handler_t(int irq, void *arg, struct pt_regs *regs) {
                return conftest_isr(irq, arg, regs);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_IRQ_HANDLER_T_PRESENT" | append_conftest "types"
                echo "#define NV_IRQ_HANDLER_T_ARGUMENT_COUNT 3" | append_conftest "types"
                rm -f conftest$$.o
                return
            else
                echo "#error irq_handler_t() conftest failed!" | append_conftest "types"
                return
            fi
        ;;

        request_threaded_irq)
            #
            # Determine if the request_threaded_irq() function is present.
            #
            # added:   2009-03-23  3aa551c9b4c40018f0e261a178e3d25478dc04a9
            #
            CODE="
            #include <linux/interrupt.h>
            int conftest_request_threaded_irq(void) {
                return request_threaded_irq();
            }"
            compile_check_conftest "$CODE" "NV_REQUEST_THREADED_IRQ_PRESENT" "" "functions"
        ;;

        acpi_device_ops)
            #
            # Determine if the 'acpi_device_ops' structure has
            # a match() member.
            #
            CODE="
            #include <linux/acpi.h>
            int conftest_acpi_device_ops(void) {
                return offsetof(struct acpi_device_ops, match);
            }"

            compile_check_conftest "$CODE" "NV_ACPI_DEVICE_OPS_HAS_MATCH" "" "types"
        ;;

        acpi_op_remove)
            #
            # Determine the number of arguments to pass to the
            # 'acpi_op_remove' routine.
            #

            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>

            acpi_op_remove conftest_op_remove_routine;

            int conftest_acpi_device_ops_remove(struct acpi_device *device) {
                return conftest_op_remove_routine(device);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ACPI_DEVICE_OPS_REMOVE_ARGUMENT_COUNT 1" | append_conftest "types"
                return
            fi

            CODE="
            #include <linux/acpi.h>

            acpi_op_remove conftest_op_remove_routine;

            int conftest_acpi_device_ops_remove(struct acpi_device *device, int type) {
                return conftest_op_remove_routine(device, type);
            }"

            compile_check_conftest "$CODE" "NV_ACPI_DEVICE_OPS_REMOVE_ARGUMENT_COUNT" "2" "types"
        ;;

        acpi_device_id)
            #
            # Determine if the 'acpi_device_id' structure has 
            # a 'driver_data' member.
            #
            CODE="
            #include <linux/acpi.h>
            int conftest_acpi_device_id(void) {
                return offsetof(struct acpi_device_id, driver_data);
            }"

            compile_check_conftest "$CODE" "NV_ACPI_DEVICE_ID_HAS_DRIVER_DATA" "" "types"
        ;;

        acquire_console_sem)
            #
            # Determine if the acquire_console_sem() function
            # is present.
            #
            CODE="
            #include <linux/console.h>
            void conftest_acquire_console_sem(void) {
                acquire_console_sem(NULL);
            }"

            compile_check_conftest "$CODE" "NV_ACQUIRE_CONSOLE_SEM_PRESENT" "" "functions"
        ;;

        console_lock)
            #
            # Determine if the console_lock() function is present.
            #
            CODE="
            #include <linux/console.h>
            void conftest_console_lock(void) {
                console_lock(NULL);
            }"

            compile_check_conftest "$CODE" "NV_CONSOLE_LOCK_PRESENT" "" "functions"
        ;;

        kmem_cache_create)
            #
            # Determine if the kmem_cache_create() function is
            # present and how many arguments it takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/slab.h>
            void conftest_kmem_cache_create(void) {
                kmem_cache_create();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#undef NV_KMEM_CACHE_CREATE_PRESENT" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/slab.h>
            void conftest_kmem_cache_create(void) {
                kmem_cache_create(NULL, 0, 0, 0L, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_KMEM_CACHE_CREATE_PRESENT" | append_conftest "functions"
                echo "#define NV_KMEM_CACHE_CREATE_ARGUMENT_COUNT 6" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/slab.h>
            void conftest_kmem_cache_create(void) {
                kmem_cache_create(NULL, 0, 0, 0L, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_KMEM_CACHE_CREATE_PRESENT" | append_conftest "functions"
                echo "#define NV_KMEM_CACHE_CREATE_ARGUMENT_COUNT 5" | append_conftest "functions"
                return
            else
                echo "#error kmem_cache_create() conftest failed!" | append_conftest "functions"
            fi
        ;;

        smp_call_function)
            #
            # Determine if the smp_call_function() function is
            # present and how many arguments it takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_smp_call_function(void) {
            #ifdef CONFIG_SMP
                smp_call_function();
            #endif
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#undef NV_SMP_CALL_FUNCTION_PRESENT" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_smp_call_function(void) {
                smp_call_function(NULL, NULL, 0, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_SMP_CALL_FUNCTION_PRESENT" | append_conftest "functions"
                echo "#define NV_SMP_CALL_FUNCTION_ARGUMENT_COUNT 4" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_smp_call_function(void) {
                smp_call_function(NULL, NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_SMP_CALL_FUNCTION_PRESENT" | append_conftest "functions"
                echo "#define NV_SMP_CALL_FUNCTION_ARGUMENT_COUNT 3" | append_conftest "functions"
                return
            else
                echo "#error smp_call_function() conftest failed!" | append_conftest "functions"
            fi
        ;;

        on_each_cpu)
            #
            # Determine if the on_each_cpu() function is present
            # and how many arguments it takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_on_each_cpu(void) {
            #ifdef CONFIG_SMP
                on_each_cpu();
            #endif
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#undef NV_ON_EACH_CPU_PRESENT" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_on_each_cpu(void) {
                on_each_cpu(NULL, NULL, 0, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ON_EACH_CPU_PRESENT" | append_conftest "functions"
                echo "#define NV_ON_EACH_CPU_ARGUMENT_COUNT 4" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/smp.h>
            void conftest_on_each_cpu(void) {
                on_each_cpu(NULL, NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ON_EACH_CPU_PRESENT" | append_conftest "functions"
                echo "#define NV_ON_EACH_CPU_ARGUMENT_COUNT 3" | append_conftest "functions"
                return
            else
                echo "#error on_each_cpu() conftest failed!" | append_conftest "functions"
            fi
        ;;

        register_cpu_notifier)
            #
            # Determine if register_cpu_notifier() is present
            # 
            # register_cpu_notifier() was removed by the following commit
            #   2016 Dec 25: b272f732f888d4cf43c943a40c9aaa836f9b7431
            #
            CODE="
            #include <linux/cpu.h>
            void conftest_register_cpu_notifier(void) {
                register_cpu_notifier();
            }" > conftest$$.c
            compile_check_conftest "$CODE" "NV_REGISTER_CPU_NOTIFIER_PRESENT" "" "functions"
        ;;

        cpuhp_setup_state)
            #
            # Determine if cpuhp_setup_state() is present
            # 
            # cpuhp_setup_state() was added by the following commit
            #   2016 Feb 26: 5b7aa87e0482be768486e0c2277aa4122487eb9d 
            # 
            # It is used as a replacement for register_cpu_notifier
            CODE="
            #include <linux/cpu.h>
            void conftest_cpuhp_setup_state(void) {
                cpuhp_setup_state();
            }" > conftest$$.c
            compile_check_conftest "$CODE" "NV_CPUHP_SETUP_STATE_PRESENT" "" "functions"
        ;;

        acpi_evaluate_integer)
            #
            # Determine if the acpi_evaluate_integer() function is
            # present and the type of its 'data' argument.
            #

            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>
            acpi_status acpi_evaluate_integer(acpi_handle h, acpi_string s,
                struct acpi_object_list *l, unsigned long long *d) {
                return AE_OK;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ACPI_EVALUATE_INTEGER_PRESENT" | append_conftest "functions"
                echo "typedef unsigned long long nv_acpi_integer_t;" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>
            acpi_status acpi_evaluate_integer(acpi_handle h, acpi_string s,
                struct acpi_object_list *l, unsigned long *d) {
                return AE_OK;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ACPI_EVALUATE_INTEGER_PRESENT" | append_conftest "functions"
                echo "typedef unsigned long nv_acpi_integer_t;" | append_conftest "functions"
                return
            else
                #
                # We can't report a compile test failure here because
                # this is a catch-all for both kernels that don't
                # have acpi_evaluate_integer() and kernels that have
                # broken header files that make it impossible to
                # tell if the function is present.
                #
                echo "#undef NV_ACPI_EVALUATE_INTEGER_PRESENT" | append_conftest "functions"
                echo "typedef unsigned long nv_acpi_integer_t;" | append_conftest "functions"
            fi
        ;;

        acpi_walk_namespace)
            #
            # Determine if the acpi_walk_namespace() function is present
            # and how many arguments it takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>
            void conftest_acpi_walk_namespace(void) {
                acpi_walk_namespace();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#undef NV_ACPI_WALK_NAMESPACE_PRESENT" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>
            void conftest_acpi_walk_namespace(void) {
                acpi_walk_namespace(0, NULL, 0, NULL, NULL, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ACPI_WALK_NAMESPACE_PRESENT" | append_conftest "functions"
                echo "#define NV_ACPI_WALK_NAMESPACE_ARGUMENT_COUNT 7" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/acpi.h>
            void conftest_acpi_walk_namespace(void) {
                acpi_walk_namespace(0, NULL, 0, NULL, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_ACPI_WALK_NAMESPACE_PRESENT" | append_conftest "functions"
                echo "#define NV_ACPI_WALK_NAMESPACE_ARGUMENT_COUNT 6" | append_conftest "functions"
                return
            else
                echo "#error acpi_walk_namespace() conftest failed!" | append_conftest "functions"
            fi
        ;;

        ioremap_cache)
            #
            # Determine if the ioremap_cache() function is present.
            #
            CODE="
            #include <asm/io.h>
            void conftest_ioremap_cache(void) {
                ioremap_cache();
            }"

            compile_check_conftest "$CODE" "NV_IOREMAP_CACHE_PRESENT" "" "functions"
        ;;

        ioremap_wc)
            #
            # Determine if the ioremap_wc() function is present.
            #
            CODE="
            #include <asm/io.h>
            void conftest_ioremap_wc(void) {
                ioremap_wc();
            }"

            compile_check_conftest "$CODE" "NV_IOREMAP_WC_PRESENT" "" "functions"
        ;;

        proc_dir_entry)
            #
            # Determine if the 'proc_dir_entry' structure has 
            # an 'owner' member.
            #
            CODE="
            #include <linux/proc_fs.h>
            int conftest_proc_dir_entry(void) {
                return offsetof(struct proc_dir_entry, owner);
            }"

            compile_check_conftest "$CODE" "NV_PROC_DIR_ENTRY_HAS_OWNER" "" "types"
        ;;

      INIT_WORK)
            #
            # Determine how many arguments the INIT_WORK() macro
            # takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/workqueue.h>
            void conftest_INIT_WORK(void) {
                INIT_WORK();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#undef NV_INIT_WORK_PRESENT" | append_conftest "macros"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/workqueue.h>
            void conftest_INIT_WORK(void) {
                INIT_WORK((struct work_struct *)NULL, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_INIT_WORK_PRESENT" | append_conftest "macros"
                echo "#define NV_INIT_WORK_ARGUMENT_COUNT 3" | append_conftest "macros"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/workqueue.h>
            void conftest_INIT_WORK(void) {
                INIT_WORK((struct work_struct *)NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_INIT_WORK_PRESENT" | append_conftest "macros"
                echo "#define NV_INIT_WORK_ARGUMENT_COUNT 2" | append_conftest "macros"
                rm -f conftest$$.o
                return
            else
                echo "#error INIT_WORK() conftest failed!" | append_conftest "macros"
                return
            fi
        ;;

      pci_dma_mapping_error)
            #
            # Determine how many arguments pci_dma_mapping_error()
            # takes.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/pci.h>
            int conftest_pci_dma_mapping_error(void) {
                return pci_dma_mapping_error(NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_PCI_DMA_MAPPING_ERROR_PRESENT" | append_conftest "functions"
                echo "#define NV_PCI_DMA_MAPPING_ERROR_ARGUMENT_COUNT 2" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/pci.h>
            int conftest_pci_dma_mapping_error(void) {
                return pci_dma_mapping_error(0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_PCI_DMA_MAPPING_ERROR_PRESENT" | append_conftest "functions"
                echo "#define NV_PCI_DMA_MAPPING_ERROR_ARGUMENT_COUNT 1" | append_conftest "functions"
                rm -f conftest$$.o
                return
            else
                echo "#error pci_dma_mapping_error() conftest failed!" | append_conftest "functions"
                return
            fi
        ;;

        scatterlist)
            #
            # Determine if the 'scatterlist' structure has
            # a 'page_link' member.
            #
            CODE="
            #include <linux/types.h>
            #include <linux/scatterlist.h>
            int conftest_scatterlist(void) {
                return offsetof(struct scatterlist, page_link);
            }"

            compile_check_conftest "$CODE" "NV_SCATTERLIST_HAS_PAGE_LINK" "" "types"
        ;;

        pci_domain_nr)
            #
            # Determine if the pci_domain_nr() function is present.
            #
            CODE="
            #include <linux/types.h>
            #include <linux/pci.h>
            int conftest_pci_domain_nr(struct pci_dev *dev) {
                return pci_domain_nr();
            }"

            compile_check_conftest "$CODE" "NV_PCI_DOMAIN_NR_PRESENT" "" "functions"
        ;;

        file_operations)
            #
            # Determine if the 'file_operations' structure has
            # 'ioctl', 'unlocked_ioctl' and 'compat_ioctl' fields.
            #
            CODE="
            #include <linux/fs.h>
            int conftest_file_operations(void) {
                return offsetof(struct file_operations, ioctl);
            }"

            compile_check_conftest "$CODE" "NV_FILE_OPERATIONS_HAS_IOCTL" "" "types"

            CODE="
            #include <linux/fs.h>
            int conftest_file_operations(void) {
                return offsetof(struct file_operations, unlocked_ioctl);
            }"

            compile_check_conftest "$CODE" "NV_FILE_OPERATIONS_HAS_UNLOCKED_IOCTL" "" "types"

            CODE="
            #include <linux/fs.h>
            int conftest_file_operations(void) {
                return offsetof(struct file_operations, compat_ioctl);
            }"

            compile_check_conftest "$CODE" "NV_FILE_OPERATIONS_HAS_COMPAT_IOCTL" "" "types"
        ;;

        sg_table)
            #
            # Determine if the struct sg_table type is present.
            #
            CODE="
            #include <linux/scatterlist.h>
            struct sg_table conftest_sg_table;
            "

            compile_check_conftest "$CODE" "NV_SG_TABLE_PRESENT" "" "types"
        ;;

        sg_alloc_table)
            #
            # Determine if include/linux/scatterlist.h exists and which table
            # allocation functions are present if so.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/scatterlist.h>
            void conftest_sg_alloc_table(void) {
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ ! -f conftest$$.o ]; then
                echo "#undef NV_SG_ALLOC_TABLE_PRESENT" | append_conftest "functions"
                echo "#undef NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT" | append_conftest "functions"
                return
            fi
            
            rm -f conftest$$.o

            CODE="
            #include <linux/scatterlist.h>
            void conftest_sg_alloc_table(void) {
                sg_alloc_table();
            }"

            compile_check_conftest "$CODE" "NV_SG_ALLOC_TABLE_PRESENT" "" "functions"

            CODE="
            #include <linux/scatterlist.h>
            void conftest_sg_alloc_table_from_pages(void) {
                sg_alloc_table_from_pages();
            }"

            compile_check_conftest "$CODE" "NV_SG_ALLOC_TABLE_FROM_PAGES_PRESENT" "" "functions"
        ;;

        efi_enabled)
            #
            # Determine if the efi_enabled symbol is present, or if
            # the efi_enabled() function is present and how many
            # arguments it takes.
            #
            echo "$CONFTEST_PREAMBLE
            #if defined(NV_LINUX_EFI_H_PRESENT)
            #include <linux/efi.h> 
            #endif
            int conftest_efi_enabled(void) {
                return efi_enabled();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#undef NV_EFI_ENABLED_PRESENT" | append_conftest "symbols"
                echo "#undef NV_EFI_ENABLED_PRESENT" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #if defined(NV_LINUX_EFI_H_PRESENT)
            #include <linux/efi.h> 
            #endif
            int conftest_efi_enabled(void) {
                return efi_enabled(0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_EFI_ENABLED_PRESENT" | append_conftest "functions"
                echo "#define NV_EFI_ENABLED_ARGUMENT_COUNT 1" | append_conftest "functions"
                rm -f conftest$$.o
                return
            else
                echo "#define NV_EFI_ENABLED_PRESENT" | append_conftest "symbols"
                return
            fi
        ;;

        dom0_kernel_present)
            #
            # Add config parameter if running on DOM0.
            #
            if [ -n "$VGX_BUILD" ]; then
                echo "#define NV_DOM0_KERNEL_PRESENT" | append_conftest "generic"
            else
                echo "#undef NV_DOM0_KERNEL_PRESENT" | append_conftest "generic"
            fi
            return
        ;;

        nvidia_vgpu_kvm_build)
           #
           # Add config parameter if running on KVM host.
           #
           if [ -n "$VGX_KVM_BUILD" ]; then
                echo "#define NV_VGPU_KVM_BUILD" | append_conftest "generic"
            else
                echo "#undef NV_VGPU_KVM_BUILD" | append_conftest "generic"
            fi
            return
        ;;

        vfio_register_notifier)
            #
            # Check number of arguments required.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/vfio.h>
            int conftest_vfio_register_notifier(void) {
                return vfio_register_notifier((struct device *) NULL, (struct notifier_block *) NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_VFIO_NOTIFIER_ARGUMENT_COUNT 2" | append_conftest "functions"
                rm -f conftest$$.o
                return
            else
                echo "#define NV_VFIO_NOTIFIER_ARGUMENT_COUNT 4" | append_conftest "functions"
                return
            fi
        ;;

        vfio_info_add_capability_has_cap_type_id_arg)
            #
            # Check if vfio_info_add_capability() has cap_type_id field.
            # cap_type_id field was removed in commit:
            # 2017-12-12 dda01f787df9f9e46f1c0bf8aa11f246e300750d
            #
            CODE="
            #include <linux/vfio.h>
            int vfio_info_add_capability(struct vfio_info_cap *caps,
                                         int cap_type_id,
                                         void *cap_type) {
                return 0;
            }"

            compile_check_conftest "$CODE" "NV_VFIO_INFO_ADD_CAPABILITY_HAS_CAP_TYPE_ID_ARGS" "" "types"
        ;;

        nvidia_grid_build)
            if [ -n "$GRID_BUILD" ]; then
                echo "#define NV_GRID_BUILD" | append_conftest "generic"
            else
                echo "#undef NV_GRID_BUILD" | append_conftest "generic"
            fi
            return
        ;;

        vm_fault_present)
            #
            # Determine if the 'vm_fault' structure is present. The earlier
            # name for this struct was fault_data, and it was renamed to
            # vm_fault by:
            #
            #  2007-07-19  d0217ac04ca6591841e5665f518e38064f4e65bd
            #
            CODE="
            #include <linux/mm.h>
            int conftest_vm_fault_present(void) {
                return offsetof(struct vm_fault, flags);
            }"

            compile_check_conftest "$CODE" "NV_VM_FAULT_PRESENT" "" "types"
        ;;

        vm_fault_has_address)
            #
            # Determine if the 'vm_fault' structure has an 'address', or a
            # 'virtual_address' field. The .virtual_address field was
            # effectively renamed to .address, by these two commits:
            #
            # struct vm_fault: .address was added by:
            #  2016-12-14  82b0f8c39a3869b6fd2a10e180a862248736ec6f
            #
            # struct vm_fault: .virtual_address was removed by:
            #  2016-12-14  1a29d85eb0f19b7d8271923d8917d7b4f5540b3e
            #
            CODE="
            #include <linux/mm.h>
            int conftest_vm_fault_has_address(void) {
                return offsetof(struct vm_fault, address);
            }"

            compile_check_conftest "$CODE" "NV_VM_FAULT_HAS_ADDRESS" "" "types"
        ;;

        mdev_uuid)
            #
            # Determine if mdev_uuid() function is present or not
            #
            CODE="
            #include <linux/pci.h>
            #include <linux/mdev.h>
            void conftest_mdev_uuid() {
                mdev_uuid();
            }"

            compile_check_conftest "$CODE" "NV_MDEV_UUID_PRESENT" "" "functions"
        ;;

        mdev_dev)
            #
            # Determine if mdev_dev() function is present or not
            #
            CODE="
            #include <linux/pci.h>
            #include <linux/mdev.h>
            void conftest_mdev_dev() {
                mdev_dev();
            }"

            compile_check_conftest "$CODE" "NV_MDEV_DEV_PRESENT" "" "functions"
        ;;

        mdev_parent)
            #
            # Determine if the struct mdev_parent type is present.
            #
            CODE="
            #include <linux/pci.h>
            #include <linux/mdev.h>
            struct mdev_parent_ops conftest_mdev_parent;
            "

            compile_check_conftest "$CODE" "NV_MDEV_PARENT_OPS_STRUCT_PRESENT" "" "types"
        ;; 

        mdev_parent_dev)
            #
            # Determine if mdev_parent_dev() function is present or not
            #
            CODE="
            #include <linux/pci.h>
            #include <linux/mdev.h>
            void conftest_mdev_parent_dev() {
                mdev_parent_dev();
            }"

            compile_check_conftest "$CODE" "NV_MDEV_PARENT_DEV_PRESENT" "" "functions"
        ;;

        mdev_from_dev)
            #
            # Determine if mdev_from_dev() function is present or not.
            #
            # Added: 2016-12-30  99e3123e3d72616a829dad6d25aa005ef1ef9b13
            #
            CODE="
            #include <linux/pci.h>
            #include <linux/mdev.h>
            void conftest_mdev_from_dev() {
                mdev_from_dev();
            }"

            compile_check_conftest "$CODE" "NV_MDEV_FROM_DEV_PRESENT" "" "functions"
        ;;

        drm_available)
            #
            # Determine if the DRM subsystem is usable
            #
            CODE="
            #if defined(NV_DRM_DRMP_H_PRESENT)
            #include <drm/drmP.h>
            #endif
            #if !defined(CONFIG_DRM) && !defined(CONFIG_DRM_MODULE)
            #error DRM not enabled
            #endif
            void conftest_drm_available(void) {
                struct drm_driver drv;

                /* 2013-01-15 89177644a7b6306e6084a89eab7e290f4bfef397 */
                drv.gem_prime_pin = 0;
                drv.gem_prime_get_sg_table = 0;
                drv.gem_prime_vmap = 0;
                drv.gem_prime_vunmap = 0;
                (void)drm_gem_prime_import;
                (void)drm_gem_prime_export;

                /* 2013-10-02 1bb72532ac260a2d3982b40bdd4c936d779d0d16 */
                (void)drm_dev_alloc;

                /* 2013-10-02 c22f0ace1926da399d9a16dfaf09174c1b03594c */
                (void)drm_dev_register;

                /* 2013-10-02 c3a49737ef7db0bdd4fcf6cf0b7140a883e32b2a */
                (void)drm_dev_unregister;
            }"

            compile_check_conftest "$CODE" "NV_DRM_AVAILABLE" "" "generic"
        ;;

        drm_dev_unref)
            #
            # Determine if drm_dev_unref() is present.
            #
            CODE="
            #if defined(NV_DRM_DRMP_H_PRESENT)
            #include <drm/drmP.h>
            #endif
            void conftest_drm_dev_unref(void) {
                /*
                 * drm_dev_free() was added in:
                 *  2013-10-02 0dc8fe5985e01f238e7dc64ff1733cc0291811e8
                 * drm_dev_free() was renamed to drm_dev_unref() in:
                 *  2014-01-29 099d1c290e2ebc3b798961a6c177c3aef5f0b789
                 */
                drm_dev_unref();
            }"

            compile_check_conftest "$CODE" "NV_DRM_DEV_UNREF_PRESENT" "" "functions"
        ;;

        proc_create_data)
            #
            # Determine if the proc_create_data() function is present.
            #
            CODE="
            #include <linux/proc_fs.h>
            void conftest_proc_create_data(void) {
                proc_create_data();
            }"

            compile_check_conftest "$CODE" "NV_PROC_CREATE_DATA_PRESENT" "" "functions"
        ;;


        pde_data)
            #
            # Determine if the PDE_DATA() function is present.
            #
            CODE="
            #include <linux/proc_fs.h>
            void conftest_PDE_DATA(void) {
                PDE_DATA();
            }"

            compile_check_conftest "$CODE" "NV_PDE_DATA_PRESENT" "" "functions"
        ;;

        get_num_physpages)
            #
            # Determine if the get_num_physpages() function is
            # present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_get_num_physpages(void) {
                get_num_physpages(NULL);
            }"

            compile_check_conftest "$CODE" "NV_GET_NUM_PHYSPAGES_PRESENT" "" "functions"
        ;;

        proc_remove)
            #
            # Determine if the proc_remove() function is present.
            #
            CODE="
            #include <linux/proc_fs.h>
            void conftest_proc_remove(void) {
                proc_remove();
            }"

            compile_check_conftest "$CODE" "NV_PROC_REMOVE_PRESENT" "" "functions"
        ;;

        vm_operations_struct)
            #
            # Determine if the 'vm_operations_struct' structure has
            # 'fault' and 'access' fields.
            #
            CODE="
            #include <linux/mm.h>
            int conftest_vm_operations_struct(void) {
                return offsetof(struct vm_operations_struct, fault);
            }"

            compile_check_conftest "$CODE" "NV_VM_OPERATIONS_STRUCT_HAS_FAULT" "" "types"

            CODE="
            #include <linux/mm.h>
            int conftest_vm_operations_struct(void) {
                return offsetof(struct vm_operations_struct, access);
            }"

            compile_check_conftest "$CODE" "NV_VM_OPERATIONS_STRUCT_HAS_ACCESS" "" "types"
        ;;

        fault_flags)
            # Determine if the FAULT_FLAG_WRITE is defined
            CODE="
            #include <linux/mm.h>
            void conftest_fault_flags(void) {
                int flag = FAULT_FLAG_WRITE;
            }"

            compile_check_conftest "$CODE" "NV_FAULT_FLAG_PRESENT" "" "types"
        ;;

        atomic_long_type)
            # Determine if atomic_long_t and associated functions are defined
            # Added in 2.6.16 2006-01-06 d3cb487149bd706aa6aeb02042332a450978dc1c
            CODE="
            #include <asm/atomic.h>
            void conftest_atomic_long(void) {
                atomic_long_t data;
                atomic_long_read(&data);
                atomic_long_set(&data, 0);
                atomic_long_inc(&data);
            }"

            compile_check_conftest "$CODE" "NV_ATOMIC_LONG_PRESENT" "" "types"
        ;;

        atomic64_type)
            # Determine if atomic64_t and associated functions are defined
            CODE="
            #include <asm/atomic.h>
            void conftest_atomic64(void) {
                atomic64_t data;
                atomic64_read(&data);
                atomic64_set(&data, 0);
                atomic64_inc(&data);
            }"

            compile_check_conftest "$CODE" "NV_ATOMIC64_PRESENT" "" "types"
        ;;

        task_struct)
            #
            # Determine if the 'task_struct' structure has
            # a 'cred' field.
            #
            CODE="
            #include <linux/sched.h>
            int conftest_task_struct(void) {
                return offsetof(struct task_struct, cred);
            }"

            compile_check_conftest "$CODE" "NV_TASK_STRUCT_HAS_CRED" "" "types"
        ;;

        backing_dev_info)
            #
            # Determine if the 'address_space' structure has
            # a 'backing_dev_info' field.
            #
            CODE="
            #include <linux/fs.h>
            int conftest_backing_dev_info(void) {
                return offsetof(struct address_space, backing_dev_info);
            }"

            compile_check_conftest "$CODE" "NV_ADDRESS_SPACE_HAS_BACKING_DEV_INFO" "" "types"
        ;;

        address_space)
            #
            # Determine if the 'address_space' structure has
            # a 'tree_lock' field of type rwlock_t.
            #
            CODE="
            #include <linux/fs.h>
            int conftest_address_space(void) {
                struct address_space as;
                rwlock_init(&as.tree_lock);
                return offsetof(struct address_space, tree_lock);
            }"

            compile_check_conftest "$CODE" "NV_ADDRESS_SPACE_HAS_RWLOCK_TREE_LOCK" "" "types"
        ;;

        address_space_init_once)
            #
            # Determine if address_space_init_once is present.
            #
            CODE="
            #include <linux/fs.h>
            void conftest_address_space_init_once(void) {
                address_space_init_once();
            }"

            compile_check_conftest "$CODE" "NV_ADDRESS_SPACE_INIT_ONCE_PRESENT" "" "functions"
        ;;

        kbasename)
            #
            # Determine if the kbasename() function is present.
            #
            CODE="
            #include <linux/string.h>
            void conftest_kbasename(void) {
                kbasename();
            }"

            compile_check_conftest "$CODE" "NV_KBASENAME_PRESENT" "" "functions"
        ;;

        fatal_signal_pending)
            #
            # Determine if fatal_signal_pending is present.
            #
            CODE="
            #if defined(NV_LINUX_SCHED_SIGNAL_H_PRESENT)
            #include <linux/sched/signal.h>
            #else
            #include <linux/sched.h>
            #endif
            void conftest_fatal_signal_pending(void) {
                fatal_signal_pending();
            }"

            compile_check_conftest "$CODE" "NV_FATAL_SIGNAL_PENDING_PRESENT" "" "functions"
        ;;

        kuid_t)
            #
            # Determine if the 'kuid_t' type is present.
            #
            CODE="
            #include <linux/sched.h>
            kuid_t conftest_kuid_t;
            "

            compile_check_conftest "$CODE" "NV_KUID_T_PRESENT" "" "types"
        ;;

        pm_vt_switch_required)
            #
            # Determine if the pm_vt_switch_required() function is present.
            #
            CODE="
            #include <linux/pm.h>
            void conftest_pm_vt_switch_required(void) {
                pm_vt_switch_required();
            }"

            compile_check_conftest "$CODE" "NV_PM_VT_SWITCH_REQUIRED_PRESENT" "" "functions"
        ;;
        
        list_cut_position)
            #
            # Determine if the list_cut_position() function is present.
            #
            CODE="
            #include <linux/list.h>
            void conftest_list_cut_position(void) {
                list_cut_position();
            }"

            compile_check_conftest "$CODE" "NV_LIST_CUT_POSITION_PRESENT" "" "functions"
        ;;

        file_inode)
            #
            # Determine if the 'file' structure has
            # a 'f_inode' field.
            #
            CODE="
            #include <linux/fs.h>
            int conftest_file_inode(void) {
                return offsetof(struct file, f_inode);
            }"

            compile_check_conftest "$CODE" "NV_FILE_HAS_INODE" "" "types"
        ;;

        xen_ioemu_inject_msi)
            #
            # Determine if the xen_ioemu_inject_msi() function is present.
            #
            CODE="
            #if defined(NV_XEN_IOEMU_H_PRESENT)
            #include <linux/kernel.h>
            #include <xen/interface/xen.h>
            #include <xen/hvm.h>
            #include <xen/ioemu.h>
            #endif
            void conftest_xen_ioemu_inject_msi(void) {
                xen_ioemu_inject_msi();
            }"

            compile_check_conftest "$CODE" "NV_XEN_IOEMU_INJECT_MSI" "" "functions"
        ;; 

        phys_to_dma)
            #
            # Determine if the phys_to_dma function is present.
            #
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_phys_to_dma(void) {
                phys_to_dma();
            }"

            compile_check_conftest "$CODE" "NV_PHYS_TO_DMA_PRESENT" "" "functions"
        ;;

        dma_ops)
            #
            # Determine if the 'dma_ops' structure is present.
            #
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_dma_ops(void) {
                (void)dma_ops;
            }"

            compile_check_conftest "$CODE" "NV_DMA_OPS_PRESENT" "" "symbols"
        ;;

        dma_map_ops)
            #
            # Determine if the 'struct dma_map_ops' type is present.
            # 
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_dma_map_ops(void) {
                struct dma_map_ops ops;
            }"

            compile_check_conftest "$CODE" "NV_DMA_MAP_OPS_PRESENT" "" "types"
        ;;
 
        get_dma_ops)
            #
            # Determine if the get_dma_ops() function is present.
            #
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_get_dma_ops(void) {
                get_dma_ops();
            }"

            compile_check_conftest "$CODE" "NV_GET_DMA_OPS_PRESENT" "" "functions"
        ;;

        noncoherent_swiotlb_dma_ops)
            #
            # Determine if the 'noncoherent_swiotlb_dma_ops' symbol is present.
            #
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_noncoherent_swiotlb_dma_ops(void) {
                (void)noncoherent_swiotlb_dma_ops;
            }"

            compile_check_conftest "$CODE" "NV_NONCOHERENT_SWIOTLB_DMA_OPS_PRESENT" "" "symbols"
        ;;

        dma_map_resource)
            #
            # Determine if the dma_map_resource() function is present.
            #
            # dma_map_resource() was added by:
            #   2016-08-10  6f3d87968f9c8b529bc81eff5a1f45e92553493d
            #
            CODE="
            #include <linux/dma-mapping.h>
            void conftest_dma_map_resource(void) {
                dma_map_resource();
            }"

            compile_check_conftest "$CODE" "NV_DMA_MAP_RESOURCE_PRESENT" "" "functions"
        ;;

        write_cr4)
            #
            # Determine if the write_cr4() function is present.
            #
            CODE="
            #include <asm/processor.h>
            void conftest_write_cr4(void) {
                write_cr4();
            }"

            compile_check_conftest "$CODE" "NV_WRITE_CR4_PRESENT" "" "functions"
        ;;

        of_get_property)
            #
            # Determine if the of_get_property function is present.
            #
            CODE="
            #if defined(NV_LINUX_OF_H_PRESENT)
            #include <linux/of.h>
            #endif
            void conftest_of_get_property() {
                of_get_property();
            }"

            compile_check_conftest "$CODE" "NV_OF_GET_PROPERTY_PRESENT" "" "functions"
        ;;

        of_find_node_by_phandle)
            #
            # Determine if the of_find_node_by_phandle function is present.
            #
            CODE="
            #if defined(NV_LINUX_OF_H_PRESENT)
            #include <linux/of.h>
            #endif
            void conftest_of_find_node_by_phandle() {
                of_find_node_by_phandle();
            }"

            compile_check_conftest "$CODE" "NV_OF_FIND_NODE_BY_PHANDLE_PRESENT" "" "functions"
        ;;

        of_node_to_nid)
            #
            # Determine if of_node_to_nid is present
            #
            CODE="
            #include <linux/version.h>
            #include <linux/utsname.h>
            #if defined(NV_LINUX_OF_H_PRESENT)
              #include <linux/of.h>
            #endif
            void conftest_of_node_to_nid() {
              of_node_to_nid();
            }"

            compile_check_conftest "$CODE" "NV_OF_NODE_TO_NID_PRESENT" "" "functions"
        ;;

        pnv_pci_get_npu_dev)
            #
            # Determine if the pnv_pci_get_npu_dev function is present.
            #
            CODE="
            #include <linux/pci.h>
            void conftest_pnv_pci_get_npu_dev() {
                pnv_pci_get_npu_dev();
            }"

            compile_check_conftest "$CODE" "NV_PNV_PCI_GET_NPU_DEV_PRESENT" "" "functions"
        ;;

        for_each_online_node)
            #
            # Determine if the for_each_online_node() function is present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_for_each_online_node() {
                for_each_online_node();
            }"

            compile_check_conftest "$CODE" "NV_FOR_EACH_ONLINE_NODE_PRESENT" "" "functions"
        ;;

        node_end_pfn)
            #
            # Determine if the node_end_pfn() function is present.
            #
            CODE="
            #include <linux/mm.h>
            void conftest_node_end_pfn() {
                node_end_pfn();
            }"

            compile_check_conftest "$CODE" "NV_NODE_END_PFN_PRESENT" "" "functions"
        ;;

        kernel_write)
            #
            # Determine if kernel_write function is present
            #
            CODE="
            #include <linux/fs.h>
            void conftest_kernel_write() {
                kernel_write();
            }"

            compile_check_conftest "$CODE" "NV_KERNEL_WRITE_PRESENT" "" "functions"
        ;;

        strnstr)
            #
            # Determine if strnstr function is present
            #
            CODE="
            #include <linux/string.h>
            void conftest_strnstr() {
                strnstr();
            }"

            compile_check_conftest "$CODE" "NV_STRNSTR_PRESENT" "" "functions"
        ;;

        iterate_dir)
            #
            # Determine if iterate_dir function is present
            #
            CODE="
            #include <linux/fs.h>
            void conftest_iterate_dir() {
                iterate_dir();
            }"

            compile_check_conftest "$CODE" "NV_ITERATE_DIR_PRESENT" "" "functions"
        ;;
        kstrtoull)
            #
            # Determine if kstrtoull function is present
            #
            CODE="
            #include <linux/kernel.h>
            void conftest_kstrtoull() {
                kstrtoull();
            }"

            compile_check_conftest "$CODE" "NV_KSTRTOULL_PRESENT" "" "functions"
        ;;

        drm_atomic_available)
            #
            # Determine if the DRM atomic modesetting subsystem is usable
            #
            CODE="
            #if defined(NV_DRM_DRMP_H_PRESENT)
            #include <drm/drmP.h>
            #endif
            #include <drm/drm_atomic.h>
            #if !defined(CONFIG_DRM) && !defined(CONFIG_DRM_MODULE)
            #error DRM not enabled
            #endif
            void conftest_drm_atomic_modeset_available(void) {
                size_t a;

                /* 2015-05-18 036ef5733ba433760a3512bb5f7a155946e2df05 */
                a = offsetof(struct drm_mode_config_funcs, atomic_state_alloc);
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_MODESET_AVAILABLE" "" "generic"
        ;;

        drm_bus_present)
            #
            # Determine if the 'struct drm_bus' type is present.
            #
            # added:   2010-12-15  8410ea3b95d105a5be5db501656f44bbb91197c1
            # removed: 2014-08-29  c5786fe5f1c50941dbe27fc8b4aa1afee46ae893
            #
            CODE="
            #include <drm/drmP.h>
            void conftest_drm_bus_present(void) {
                struct drm_bus bus;
            }"

            compile_check_conftest "$CODE" "NV_DRM_BUS_PRESENT" "" "types"
        ;;

        drm_bus_has_bus_type)
            #
            # Determine if the 'drm_bus' structure has a 'bus_type' field.
            #
            # added:   2010-12-15  8410ea3b95d105a5be5db501656f44bbb91197c1
            # removed: 2013-11-03  42b21049fc26513ca8e732f47559b1525b04a992
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_bus_has_bus_type(void) {
                return offsetof(struct drm_bus, bus_type);
            }"

            compile_check_conftest "$CODE" "NV_DRM_BUS_HAS_BUS_TYPE" "" "types"
        ;;

        drm_bus_has_get_irq)
            #
            # Determine if the 'drm_bus' structure has a 'get_irq' field.
            #
            # added:   2010-12-15  8410ea3b95d105a5be5db501656f44bbb91197c1
            # removed: 2013-11-03  b2a21aa25a39837d06eb24a7f0fef1733f9843eb
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_bus_has_get_irq(void) {
                return offsetof(struct drm_bus, get_irq);
            }"

            compile_check_conftest "$CODE" "NV_DRM_BUS_HAS_GET_IRQ" "" "types"
        ;;

        drm_bus_has_get_name)
            #
            # Determine if the 'drm_bus' structure has a 'get_name' field.
            #
            # added:   2010-12-15  8410ea3b95d105a5be5db501656f44bbb91197c1
            # removed: 2013-11-03  9de1b51f1fae6476155350a0670dc637c762e718
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_bus_has_get_name(void) {
                return offsetof(struct drm_bus, get_name);
            }"

            compile_check_conftest "$CODE" "NV_DRM_BUS_HAS_GET_NAME" "" "types"
        ;;

        drm_driver_has_legacy_dev_list)
            #
            # Determine if the 'drm_driver' structure has a 'legacy_dev_list' field.
            #
            # drm_driver::device_list was added by:
            #   2008-11-28  e7f7ab45ebcb54fd5f814ea15ea079e079662f67
            # and then renamed to drm_driver::legacy_device_list by:
            #   2013-12-11  b3f2333de8e81b089262b26d52272911523e605f
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_driver_has_legacy_dev_list(void) {
                return offsetof(struct drm_driver, legacy_dev_list);
            }"

            compile_check_conftest "$CODE" "NV_DRM_DRIVER_HAS_LEGACY_DEV_LIST" "" "types"
        ;;

        drm_init_function_args)
            #
            # Determine if these functions:
            #   drm_universal_plane_init()
            #   drm_crtc_init_with_planes()
            #   drm_encoder_init()
            # have a 'name' argument, which was added by these commits:
            #   drm_universal_plane_init:   2015-12-09  b0b3b7951114315d65398c27648705ca1c322faa
            #   drm_crtc_init_with_planes:  2015-12-09  f98828769c8838f526703ef180b3088a714af2f9
            #   drm_encoder_init:           2015-12-09  13a3d91f17a5f7ed2acd275d18b6acfdb131fb15
            #
            # Additionally determine whether drm_universal_plane_init() has a
            # 'format_modifiers' argument, which was added by:
            #   2017-07-23  e6fc3b68558e4c6d8d160b5daf2511b99afa8814
            #
            CODE="
            #include <drm/drmP.h>

            int conftest_drm_crtc_init_with_planes_has_name_arg(void) {
                return
                    drm_crtc_init_with_planes(
                            NULL,  /* struct drm_device *dev */
                            NULL,  /* struct drm_crtc *crtc */
                            NULL,  /* struct drm_plane *primary */
                            NULL,  /* struct drm_plane *cursor */
                            NULL,  /* const struct drm_crtc_funcs *funcs */
                            NULL);  /* const char *name */
            }"

            compile_check_conftest "$CODE" "NV_DRM_CRTC_INIT_WITH_PLANES_HAS_NAME_ARG" "" "types"

            CODE="
            #include <drm/drmP.h>

            int conftest_drm_encoder_init_has_name_arg(void) {
                return
                    drm_encoder_init(
                            NULL,  /* struct drm_device *dev */
                            NULL,  /* struct drm_encoder *encoder */
                            NULL,  /* const struct drm_encoder_funcs *funcs */
                            DRM_MODE_ENCODER_NONE, /* int encoder_type */
                            NULL); /* const char *name */
            }"

            compile_check_conftest "$CODE" "NV_DRM_ENCODER_INIT_HAS_NAME_ARG" "" "types"

            echo "$CONFTEST_PREAMBLE
            #include <drm/drmP.h>

            int conftest_drm_universal_plane_init_has_format_modifiers_arg(void) {
                return
                    drm_universal_plane_init(
                            NULL,  /* struct drm_device *dev */
                            NULL,  /* struct drm_plane *plane */
                            0,     /* unsigned long possible_crtcs */
                            NULL,  /* const struct drm_plane_funcs *funcs */
                            NULL,  /* const uint32_t *formats */
                            0,     /* unsigned int format_count */
                            NULL,  /* const uint64_t *format_modifiers */
                            DRM_PLANE_TYPE_PRIMARY,
                            NULL);  /* const char *name */
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o

                echo "#define NV_DRM_UNIVERSAL_PLANE_INIT_HAS_FORMAT_MODIFIERS_ARG" | append_conftest "types"
                echo "#define NV_DRM_UNIVERSAL_PLANE_INIT_HAS_NAME_ARG" | append_conftest "types"
            else
                echo "#undef NV_DRM_UNIVERSAL_PLANE_INIT_HAS_FORMAT_MODIFIERS_ARG" | append_conftest "types"

                echo "$CONFTEST_PREAMBLE
                #include <drm/drmP.h>

                int conftest_drm_universal_plane_init_has_name_arg(void) {
                    return
                        drm_universal_plane_init(
                                NULL,  /* struct drm_device *dev */
                                NULL,  /* struct drm_plane *plane */
                                0,     /* unsigned long possible_crtcs */
                                NULL,  /* const struct drm_plane_funcs *funcs */
                                NULL,  /* const uint32_t *formats */
                                0,     /* unsigned int format_count */
                                DRM_PLANE_TYPE_PRIMARY,
                                NULL);  /* const char *name */
                }" > conftest$$.c

                $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1

                if [ -f conftest$$.o ]; then
                    rm -f conftest$$.o

                    echo "#define NV_DRM_UNIVERSAL_PLANE_INIT_HAS_NAME_ARG" | append_conftest "types"
                else
                    echo "#undef NV_DRM_UNIVERSAL_PLANE_INIT_HAS_NAME_ARG" | append_conftest "types"
                fi
            fi

        ;;

        drm_mode_connector_list_update_has_merge_type_bits_arg)
            #
            # Detect if drm_mode_connector_list_update() has a
            # 'merge_type_bits' second argument.  This argument was
            # remove by:
            #   2015-12-03  6af3e6561243f167dabc03f732d27ff5365cd4a4
            #
            CODE="
            #include <drm/drmP.h>
            void conftest_drm_mode_connector_list_update_has_merge_type_bits_arg(void) {
                drm_mode_connector_list_update(
                    NULL,  /* struct drm_connector *connector */
                    true); /* bool merge_type_bits */
            }"

            compile_check_conftest "$CODE" "NV_DRM_MODE_CONNECTOR_LIST_UPDATE_HAS_MERGE_TYPE_BITS_ARG" "" "types"
        ;;

        vzalloc)
            #
            # Determine if the vzalloc function is present
            # Added in 2.6.37 2010-10-26 e1ca7788dec6773b1a2bce51b7141948f2b8bccf
            #
            CODE="
            #include <linux/vmalloc.h>
            void conftest_vzalloc() {
                vzalloc();
            }"

            compile_check_conftest "$CODE" "NV_VZALLOC_PRESENT" "" "functions"
        ;;

        drm_driver_has_set_busid)
            #
            # Determine if the drm_driver structure has a 'set_busid' callback
            # field.
            #
            # drm_driver::set_busid field were added by:
            #   2014-08-29  915b4d11b8b9e7b84ba4a4645b6cc7fbc0c071cf
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_driver_has_set_busid(void) {
                return offsetof(struct drm_driver, set_busid);
            }"

            compile_check_conftest "$CODE" "NV_DRM_DRIVER_HAS_SET_BUSID" "" "types"
        ;;

        drm_driver_has_gem_prime_res_obj)
            #
            # Determine if the drm_driver structure has a 'gem_prime_res_obj'
            # callback field.
            #
            # drm_driver::gem_prime_res_obj field was added by:
            #   2014-07-01  3aac4502fd3f80dcf7e65dbf6edd8676893c1f46
            #
            CODE="
            #include <drm/drmP.h>
            int conftest_drm_driver_has_gem_prime_res_obj(void) {
                return offsetof(struct drm_driver, gem_prime_res_obj);
            }"

            compile_check_conftest "$CODE" "NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ" "" "types"
        ;;

        drm_crtc_state_has_connectors_changed)
            #
            # Determine if the crtc_state has a 'connectors_changed' field.
            #
            # drm_crtc_state::connectors_changed was added by:
            #   2015-07-21  fc596660dd4e83f7f84e3cd7b25dc5e8e83000ef
            #
            CODE="
            #include <drm/drm_crtc.h>
            void conftest_drm_crtc_state_has_connectors_changed(void) {
                struct drm_crtc_state foo;
                (void)foo.connectors_changed;
            }"

            compile_check_conftest "$CODE" "NV_DRM_CRTC_STATE_HAS_CONNECTORS_CHANGED" "" "types"
        ;;

        drm_reinit_primary_mode_group)
            #
            # Determine if the function drm_reinit_primary_mode_group() is
            # present.
            #
            # drm_reinit_primary_mode_group was added by:
            #   2014-06-05  2390cd11bfbe8d2b1b28c4e0f01fe7e122f7196d
            # removed by commit:
            #   2015-07-09  3fdefa399e4644399ce3e74e65a75122d52dba6a
            #
            CODE="
            #if defined(NV_DRM_DRM_CRTC_H_PRESENT)
            #include <drm/drm_crtc.h>
            #endif
            void conftest_drm_reinit_primary_mode_group(void) {
                drm_reinit_primary_mode_group();
            }"

            compile_check_conftest "$CODE" "NV_DRM_REINIT_PRIMARY_MODE_GROUP_PRESENT" "" "functions"
        ;;

        wait_on_bit_lock_argument_count)
            #
            # Determine how many arguments wait_on_bit_lock takes.
            #
            #  wait_on_bit_lock changed by
            #    2014-07-07  743162013d40ca612b4cb53d3a200dff2d9ab26e
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/wait.h>
            void conftest_wait_on_bit_lock(void) {
                wait_on_bit_lock(NULL, 0, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_WAIT_ON_BIT_LOCK_ARGUMENT_COUNT 3" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/wait.h>
            void conftest_wait_on_bit_lock(void) {
                wait_on_bit_lock(NULL, 0, NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_WAIT_ON_BIT_LOCK_ARGUMENT_COUNT 4" | append_conftest "functions"
                return
            fi
            echo "#error wait_on_bit_lock() conftest failed!" | append_conftest "functions"
        ;;

        bitmap_clear)
            #
            # Determine if the bitmap_clear function is present
            # Added in 2.6.33 2009-12-15 c1a2a962a2ad103846e7950b4591471fabecece7
            #
            CODE="
            #include <linux/bitmap.h>
            void conftest_bitmap_clear() {
                bitmap_clear();
            }"

            compile_check_conftest "$CODE" "NV_BITMAP_CLEAR_PRESENT" "" "functions"
        ;;

        pci_stop_and_remove_bus_device)
            #
            # Determine if the pci_stop_and_remove_bus_device() function is present.
            # Added in 3.4-rc1 2012-02-25 210647af897af8ef2d00828aa2a6b1b42206aae6
            #
            CODE="
            #include <linux/types.h>
            #include <linux/pci.h>
            void conftest_pci_stop_and_remove_bus_device() {
                pci_stop_and_remove_bus_device();
            }"

            compile_check_conftest "$CODE" "NV_PCI_STOP_AND_REMOVE_BUS_DEVICE_PRESENT" "" "functions"
        ;;

        pci_remove_bus_device)
            #
            # Determine if the pci_remove_bus_device() function is present.
            # Added before Linux-2.6.12-rc2 2005-04-16
            #
            CODE="
            #include <linux/types.h>
            #include <linux/pci.h>
            void conftest_pci_remove_bus_device() {
                pci_remove_bus_device();
            }"

            compile_check_conftest "$CODE" "NV_PCI_REMOVE_BUS_DEVICE_PRESENT" "" "functions"
        ;;

        drm_atomic_set_mode_for_crtc)
            #
            # Determine if the function drm_atomic_set_mode_for_crtc() is
            # present.
            #
            # drm_atomic_set_mode_for_crtc() was added by:
            #   2015-05-26  819364da20fd914aba2fd03e95ee0467286752f5
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_H_PRESENT)
            #include <drm/drm_atomic.h>
            #endif
            void conftest_drm_atomic_clean_old_fb(void) {
                drm_atomic_set_mode_for_crtc();
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_SET_MODE_FOR_CRTC" "" "functions"
        ;;

        drm_atomic_clean_old_fb)
            #
            # Determine if the function drm_atomic_clean_old_fb() is
            # present.
            #
            # drm_atomic_clean_old_fb() was added by:
            #   2015-11-11  0f45c26fc302c02b0576db37d4849baa53a2bb41
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_H_PRESENT)
            #include <drm/drm_atomic.h>
            #endif
            void conftest_drm_atomic_clean_old_fb(void) {
                drm_atomic_clean_old_fb();
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_CLEAN_OLD_FB" "" "functions"
        ;;

        drm_helper_mode_fill_fb_struct | drm_helper_mode_fill_fb_struct_has_const_mode_cmd_arg)
            #
            # Determine if the drm_helper_mode_fill_fb_struct function takes
            # 'dev' argument.
            #
            # The drm_helper_mode_fill_fb_struct() has been updated to
            # take 'dev' parameter by:
            #   2016-12-14  a3f913ca98925d7e5bae725e9b2b38408215a695
            #
            echo "$CONFTEST_PREAMBLE
            #include <drm/drm_crtc_helper.h>
            void drm_helper_mode_fill_fb_struct(struct drm_device *dev,
                                                struct drm_framebuffer *fb,
                                                const struct drm_mode_fb_cmd2 *mode_cmd)
            {
                return;
            }" > conftest$$.c;

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_DEV_ARG" | append_conftest "function"
                echo "#define NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_CONST_MODE_CMD_ARG" | append_conftest "function"
                rm -f conftest$$.o
            else
                echo "#undef NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_DEV_ARG" | append_conftest "function"

                #
                # Determine if the drm_mode_fb_cmd2 pointer argument is const in
                # drm_mode_config_funcs::fb_create and drm_helper_mode_fill_fb_struct().
                #
                # The drm_mode_fb_cmd2 pointer through this call chain was made const by:
                #   2015-11-11  1eb83451ba55d7a8c82b76b1591894ff2d4a95f2
                #
                echo "$CONFTEST_PREAMBLE
                #include <drm/drm_crtc_helper.h>
                void drm_helper_mode_fill_fb_struct(struct drm_framebuffer *fb,
                                                    const struct drm_mode_fb_cmd2 *mode_cmd)
                {
                    return;
                }" > conftest$$.c;

                $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
                rm -f conftest$$.c

                if [ -f conftest$$.o ]; then
                    echo "#define NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_CONST_MODE_CMD_ARG" | append_conftest "function"
                    rm -f conftest$$.o
                else
                    echo "#undef NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_CONST_MODE_CMD_ARG" | append_conftest "function"
                fi
            fi
        ;;

        mm_context_t)
            #
            # Determine if the 'mm_context_t' data type is present
            # and if it has an 'id' member.
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            int conftest_mm_context_t(void) {
                return offsetof(mm_context_t, id);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_MM_CONTEXT_T_HAS_ID" | append_conftest "types"
                rm -f conftest$$.o
                return
            else
                echo "#undef NV_MM_CONTEXT_T_HAS_ID" | append_conftest "types"
                return
            fi
        ;;
        get_user_pages)     
            #
            # Conftest for get_user_pages()
            #
            # Use long type for get_user_pages and unsigned long for nr_pages
            # 2013 Feb 22: 28a35716d317980ae9bc2ff2f84c33a3cda9e884
            #
            # Removed struct task_struct *tsk & struct mm_struct *mm from get_user_pages.
            # 2016 Feb 12: cde70140fed8429acf7a14e2e2cbd3e329036653
            #
            # Replaced get_user_pages6 with get_user_pages.
            # 2016 April 4: c12d2da56d0e07d230968ee2305aaa86b93a6832
            #
            # Replaced write and force parameters with gup_flags.
            # 2016 Oct 12: 768ae309a96103ed02eb1e111e838c87854d8b51
            #
            # Conftest #1: Check if get_user_pages accepts 6 arguments.
            # Return if true.
            # Fall through to conftest #2 on failure.

            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            long get_user_pages(unsigned long start,
                                unsigned long nr_pages,
                                int write,
                                int force,
                                struct page **pages,
                                struct vm_area_struct **vmas) {
                return 0;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c    
            if [ -f conftest$$.o ]; then
                echo "#define NV_GET_USER_PAGES_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"
                echo "#undef NV_GET_USER_PAGES_HAS_TASK_STRUCT" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi
            
            # Conftest #2: Check if get_user_pages has gup_flags instead of write and force parameters.
            # Return if available.
            # Fall through to default case if absent.

            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            long get_user_pages(unsigned long start,
                                unsigned long nr_pages,
                                unsigned int gup_flags,
                                struct page **pages,
                                struct vm_area_struct **vmas) {
                return 0;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#undef NV_GET_USER_PAGES_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"
                echo "#undef NV_GET_USER_PAGES_HAS_TASK_STRUCT" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi
            
            echo "#define NV_GET_USER_PAGES_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"
            echo "#define NV_GET_USER_PAGES_HAS_TASK_STRUCT" | append_conftest "functions"

            return
        ;;

        get_user_pages_remote)
            #
            # Determine if the function get_user_pages_remote() is
            # present and has write/force parameters.
            #
            # get_user_pages_remote() was added by:
            #   2016 Feb 12: 1e9877902dc7e11d2be038371c6fbf2dfcd469d7
            #
            # get_user_pages[_remote]() write/force parameters
            # replaced with gup_flags:
            #   2016 Oct 12: 768ae309a96103ed02eb1e111e838c87854d8b51
            #   2016 Oct 12: 9beae1ea89305a9667ceaab6d0bf46a045ad71e7
            #
            # get_user_pages_remote() added 'locked' parameter
            #   2016 Dec 14:5b56d49fc31dbb0487e14ead790fc81ca9fb2c99
            #
            # conftest #1: check if get_user_pages_remote() is available
            # return if not available.
            # Fall through to conftest #2 if it is present

            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            void conftest_get_user_pages_remote(void) {
                get_user_pages_remote();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#undef NV_GET_USER_PAGES_REMOTE_PRESENT" | append_conftest "functions"
                echo "#undef NV_GET_USER_PAGES_REMOTE_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"
                echo "#undef NV_GET_USER_PAGES_REMOTE_HAS_LOCKED_ARG" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi

            # conftest #2: check if get_user_pages_remote() has write and
            # force arguments. Return if these arguments are present
            # Fall through to conftest #3 if these args are absent.
            echo "#define NV_GET_USER_PAGES_REMOTE_PRESENT" | append_conftest "functions"
            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            long get_user_pages_remote(struct task_struct *tsk,
                                       struct mm_struct *mm,
                                       unsigned long start,
                                       unsigned long nr_pages,
                                       int write,
                                       int force,
                                       struct page **pages,
                                       struct vm_area_struct **vmas) {
                return 0;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_GET_USER_PAGES_REMOTE_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"
                echo "#undef NV_GET_USER_PAGES_REMOTE_HAS_LOCKED_ARG" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi

            # conftest #3: check if get_user_pages_remote() has locked argument
            
            echo "$CONFTEST_PREAMBLE
            #include <linux/mm.h>
            long get_user_pages_remote(struct task_struct *tsk,
                                       struct mm_struct *mm,
                                       unsigned long start,
                                       unsigned long nr_pages,
                                       unsigned int gup_flags,
                                       struct page **pages,
                                       struct vm_area_struct **vmas,
                                       int *locked) {
                return 0;
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_GET_USER_PAGES_REMOTE_HAS_LOCKED_ARG" | append_conftest "functions"
                rm -f conftest$$.o
            else
                echo "#undef NV_GET_USER_PAGES_REMOTE_HAS_LOCKED_ARG" | append_conftest "functions"
            fi
            echo "#undef NV_GET_USER_PAGES_REMOTE_HAS_WRITE_AND_FORCE_ARGS" | append_conftest "functions"

        ;;

        usleep_range)
            #
            # Determine if the function usleep_range() is present.
            #
            # usleep_range() was added by:
            #  2010 Aug 4 : 5e7f5a178bba45c5aca3448fddecabd4e28f1f6b
            #
            CODE="
            #include <linux/delay.h>
            void conftest_usleep_range(void) {
                usleep_range();
            }"

            compile_check_conftest "$CODE" "NV_USLEEP_RANGE_PRESENT" "" "functions"
        ;;

         radix_tree_empty)
            #
            # Determine if the function  radix_tree_empty() is present.
            #
            #  radix_tree_empty() was added by:
            #  2016 May 21 : e9256efcc8e390fa4fcf796a0c0b47d642d77d32
            #
            CODE="
            #include <linux/radix-tree.h>
            int conftest_radix_tree_empty(void) {
                radix_tree_empty();
            }"

            compile_check_conftest "$CODE" "NV_RADIX_TREE_EMPTY_PRESENT" "" "functions"
        ;;

        drm_gem_object_lookup)
            #
            # Determine the number of arguments of drm_gem_object_lookup().
            #
            # drm_gem_object_lookup() was originally added to the kernel by:
            #  2008-07-30 : 673a394b1e3b69be886ff24abfd6df97c52e8d08
            #
            # First argument of type drm_device has been removed by:
            #  2016-05-09 : a8ad0bd84f986072314595d05444719fdf29e412
            #
            echo "$CONFTEST_PREAMBLE
            #if defined(NV_DRM_DRMP_H_PRESENT)
            #include <drm/drmP.h>
            #endif
            #if defined(NV_DRM_DRM_GEM_H_PRESENT)
            #include <drm/drm_gem.h>
            #endif
            void conftest_drm_gem_object_lookup(void) {
                drm_gem_object_lookup(NULL, NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_DRM_GEM_OBJECT_LOOKUP_PRESENT" | append_conftest "functions"
                echo "#define NV_DRM_GEM_OBJECT_LOOKUP_ARGUMENT_COUNT 3" | append_conftest "functions"
                rm -f conftest$$.o
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #if defined(NV_DRM_DRMP_H_PRESENT)
            #include <drm/drmP.h>
            #endif
            #if defined(NV_DRM_DRM_GEM_H_PRESENT)
            #include <drm/drm_gem.h>
            #endif
            void conftest_drm_gem_object_lookup(void) {
                drm_gem_object_lookup(NULL, 0);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#define NV_DRM_GEM_OBJECT_LOOKUP_PRESENT" | append_conftest "functions"
                echo "#define NV_DRM_GEM_OBJECT_LOOKUP_ARGUMENT_COUNT 2" | append_conftest "functions"
                rm -f conftest$$.o
            else
                echo "#undef NV_DRM_GEM_OBJECT_LOOKUP_PRESENT" | append_conftest "functions"
                echo "#undef NV_DRM_GEM_OBJECT_LOOKUP_ARGUMENT_COUNT" | append_conftest "functions"
            fi
        ;;

        drm_master_drop_has_from_release_arg)
            #
            # Determine if drm_driver::master_drop() has 'from_release' argument.
            #
            # Last argument 'bool from_release' has been removed by:
            #  2016-06-21 : d6ed682eba54915ea56315bc2e5a33fca5922997
            #
            CODE="
            #include <drm/drmP.h>
            void conftest_drm_master_drop_has_from_release_arg(struct drm_driver *drv) {
                drv->master_drop(NULL, NULL, false);
            }"

            compile_check_conftest "$CODE" "NV_DRM_MASTER_DROP_HAS_FROM_RELEASE_ARG" "" "types"
        ;;

        drm_mode_config_funcs_has_atomic_state_alloc)
            #
            # Determine if the 'drm_mode_config_funcs' structure has
            # an 'atomic_state_alloc' field.
            #
            # added:   2015-05-18  036ef5733ba433760a3512bb5f7a155946e2df05
            #
            CODE="
            #include <drm/drm_crtc.h>
            int conftest_drm_mode_config_funcs_has_atomic_state_alloc(void) {
                return offsetof(struct drm_mode_config_funcs, atomic_state_alloc);
            }"

            compile_check_conftest "$CODE" "NV_DRM_MODE_CONFIG_FUNCS_HAS_ATOMIC_STATE_ALLOC" "" "types"
        ;;

        drm_atomic_modeset_nonblocking_commit_available)
            #
            # Determine if nonblocking commit support avaiable in the DRM atomic
            # modesetting subsystem.
            #
            # added:   2016-05-08  9f2a7950e77abf00a2a87f3b4cbefa36e9b6009b
            #
            CODE="
            #include <drm/drm_crtc.h>
            int conftest_drm_atomic_modeset_nonblocking_commit_available(void) {
                return offsetof(struct drm_mode_config, helper_private);
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_MODESET_NONBLOCKING_COMMIT_AVAILABLE" "" "generic"
        ;;

        drm_atomic_state_free)
            #
            # Determine if the function drm_atomic_state_free() is
            # present.
            #
            # drm_atomic_state_free() has been removed by:
            #   2016-10-14  0853695c3ba46f97dfc0b5885f7b7e640ca212dd
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_H_PRESENT)
            #include <drm/drm_atomic.h>
            #endif
            void conftest_drm_atomic_state_free(void) {
                drm_atomic_state_free();
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_STATE_FREE" "" "functions"
        ;;

        vm_ops_fault_removed_vma_arg)
            #
            # Determine if vma.vm_ops.fault takes (vma, vmf), or just (vmf)
            # args. Acronym key:
            #   vma: struct vm_area_struct
            #   vm_ops: struct vm_operations_struct
            #   vmf: struct vm_fault
            #
            # The redundant vma arg was removed from BOTH vma.vm_ops.fault and
            # vma.vm_ops.page_mkwrite, with the following commit:
            #
            #   2017-02-24  11bac80004499ea59f361ef2a5516c84b6eab675
            #
            CODE="
            #include <linux/mm.h>
            void conftest_vm_ops_fault_removed_vma_arg(void) {
                struct vm_operations_struct vm_ops;
                struct vm_fault *vmf;
                (void)vm_ops.fault(vmf);
            }"

            compile_check_conftest "$CODE" "NV_VM_OPS_FAULT_REMOVED_VMA_ARG" "" "types"
        ;;

        pnv_npu2_init_context)
            #
            # Determine if the pnv_npu2_init_context() function is
            # present.
            #
            CODE="
            #if defined(NV_ASM_POWERNV_H_PRESENT)
            #include <linux/pci.h>
            #include <asm/powernv.h>
            #endif
            void conftest_pnv_npu2_init_context(void) {
                pnv_npu2_init_context();
            }"

            compile_check_conftest "$CODE" "NV_PNV_NPU2_INIT_CONTEXT_PRESENT" "" "functions"
        ;;

        drm_driver_unload_has_int_return_type)
            #
            # Determine if drm_driver::unload() returns integer value, which has
            # been changed to void by commit -
            #
            #   2017-01-06  11b3c20bdd15d17382068be569740de1dccb173d
            #
            CODE="
            #include <drm/drmP.h>

            int conftest_drm_driver_unload_has_int_return_type(struct drm_driver *drv) {
                return drv->unload(NULL /* dev */);
            }"

            compile_check_conftest "$CODE" "NV_DRM_DRIVER_UNLOAD_HAS_INT_RETURN_TYPE" "" "types"
        ;;

        kref_has_refcount_of_type_refcount_t)
            CODE="
            #include <linux/kref.h>

            refcount_t conftest_kref_has_refcount_of_type_refcount_t(struct kref *ref) {
                return ref->refcount;
            }"

            compile_check_conftest "$CODE" "NV_KREF_HAS_REFCOUNT_OF_TYPE_REFCOUNT_T" "" "types"
        ;;

        is_export_symbol_gpl_*)
            export_symbol_gpl_conftest $(echo $1 | cut -f5- -d_)
        ;;

        drm_atomic_helper_disable_all)
            #
            # Determine if the function drm_atomic_helper_disable_all() is
            # present.
            #
            # drm_atomic_helper_disable_all() has been added by:
            #   2015-12-02  1494276000db789c6d2acd85747be4707051c801
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_HELPER_H_PRESENT)
            #include <drm/drm_atomic_helper.h>
            #endif
            void conftest_drm_atomic_helper_disable_all(void) {
                drm_atomic_helper_disable_all();
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_HELPER_DISABLE_ALL_PRESENT" "" "functions"
        ;;

        drm_atomic_helper_set_config)
            #
            # Determine if drm_atomic_helper_set_config() has 'ctx' argument.
            #
            # drm_atomic_helper_set_config() was added by:
            #   2014-06-27  042652ed95996a9ef6dcddddc53b5d8bc7fa887e
            # and it has been updated to take ctx parameter by:
            #   2017-03-22  a4eff9aa6db8eb3d1864118f3558214b26f630b4
            #
            echo "$CONFTEST_PREAMBLE
            #if defined(NV_DRM_DRM_ATOMIC_HELPER_H_PRESENT)
            #include <drm/drm_atomic_helper.h>
            #endif
            void conftest_drm_atomic_helper_set_config(void) {
                drm_atomic_helper_set_config();
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                echo "#undef NV_DRM_ATOMIC_HELPER_SET_CONFIG_PRESENT" | append_conftest "functions"
            else
                echo "#define NV_DRM_ATOMIC_HELPER_SET_CONFIG_PRESENT" | append_conftest "functions"

                echo "$CONFTEST_PREAMBLE
                #include <drm/drm_atomic_helper.h>
                int drm_atomic_helper_set_config(struct drm_mode_set *set,
                                                 struct drm_modeset_acquire_ctx *ctx) {
                    return 0;
                }" > conftest$$.c

                $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
                rm -f conftest$$.c

                if [ -f conftest$$.o ]; then
                    echo "#define NV_DRM_ATOMIC_HELPER_SET_CONFIG_HAS_CTX_ARG" | append_conftest "types"
                    rm -f conftest$$.o
                else
                    echo "#undef NV_DRM_ATOMIC_HELPER_SET_CONFIG_HAS_CTX_ARG" | append_conftest "types"
                fi
            fi
        ;;

        drm_atomic_helper_crtc_destroy_state_has_crtc_arg)
            #
            # Determine if __drm_atomic_helper_crtc_destroy_state() has 'crtc'
            # argument.
            #
            # __drm_atomic_helper_crtc_destroy_state() is updated to drop
            # crtc argument by:
            #   2016-05-09  ec2dc6a0fe38de8d73a7b7638a16e7d33a19a5eb
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_HELPER_H_PRESENT)
            #include <drm/drm_atomic_helper.h>
            #endif
            void conftest_drm_atomic_helper_crtc_destroy_state_has_crtc_arg(void) {
                __drm_atomic_helper_crtc_destroy_state(NULL, NULL);
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_HELPER_CRTC_DESTROY_STATE_HAS_CRTC_ARG" "" "types"
        ;;

        drm_crtc_helper_funcs_has_atomic_enable)
            #
            # Determine if struct drm_crtc_helper_funcs has an 'atomic_enable'
            # member.
            #
            # The 'enable' callback was renamed to 'atomic_enable' by:
            #   2017-06-30  0b20a0f8c3cb6f74fe326101b62eeb5e2c56a53c
            #
            CODE="
            #include <drm/drm_modeset_helper_vtables.h>
            void conftest_drm_crtc_helper_funcs_has_atomic_enable(void) {
                struct drm_crtc_helper_funcs funcs;
                funcs.atomic_enable = NULL;
            }"

            compile_check_conftest "$CODE" "NV_DRM_CRTC_HELPER_FUNCS_HAS_ATOMIC_ENABLE" "" "types"
        ;;

        drm_atomic_helper_connector_dpms)
            #
            # Determine if the function drm_atomic_helper_connector_dpms() is present.
            #
            # drm_atomic_helper_connector_dpms() was removed by:
            #   2017-07-25 7d902c05b480cc44033dcb56e12e51b082656b42
            #
            CODE="
            #if defined(NV_DRM_DRM_ATOMIC_HELPER_H_PRESENT)
            #include <drm/drm_atomic_helper.h>
            #endif
            void conftest_drm_atomic_helper_connector_dpms(void) {
                drm_atomic_helper_connector_dpms();
            }"

            compile_check_conftest "$CODE" "NV_DRM_ATOMIC_HELPER_CONNECTOR_DPMS_PRESENT" "" "functions"
        ;;

        backlight_device_register)
            #
            # Determine if the backlight_device_register() function is present
            # and how many arguments it takes.
            #
            # Don't try to support the 4-argument form of backlight_device_register().
            #
            echo "$CONFTEST_PREAMBLE
            #include <linux/backlight.h>
            #if !defined(CONFIG_BACKLIGHT_CLASS_DEVICE)
            #error Backlight class device not enabled
            #endif
            void conftest_backlight_device_register(void) {
                backlight_device_register(NULL, NULL, NULL, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_BACKLIGHT_DEVICE_REGISTER_PRESENT" | append_conftest "functions"
                return
            else
                echo "#undef NV_BACKLIGHT_DEVICE_REGISTER_PRESENT" | append_conftest "functions"
                return
            fi
        ;;

        backlight_properties_type)
            #
            # Determine if the backlight_properties structure has a 'type' field
            # and whether BACKLIGHT_RAW is defined.
            #
            CODE="
            #include <linux/backlight.h>
            void conftest_backlight_props_type(void) {
                struct backlight_properties tmp;
                tmp.type = BACKLIGHT_RAW;
            }"

            compile_check_conftest "$CODE" "NV_BACKLIGHT_PROPERTIES_TYPE_PRESENT" "" "types"
        ;;

        register_acpi_notifier)
            #
            # Determine if the register_acpi_notifier() and unregister_acpi_notifier()
            # functions are present.
            #
            # register_acpi_notifier() and unregister_acpi_notifier() are
            # added by:
            #     2008-01-25  9ee85241fdaab358dff1d8647f20a478cfa512a1
            #
            CODE="
            #include <acpi/acpi_bus.h>
            int conftest_register_acpi_notifier(void) {
                return register_acpi_notifier();
            }"
            compile_check_conftest "$CODE" "NV_REGISTER_ACPI_NOTIFER_PRESENT" "" "functions"
        ;;

        timer_setup)
            #
            # Determine if the function timer_setup() is present.
            #
            # timer_setup() was added by:
            #     2017-09-28  686fef928bba6be13cabe639f154af7d72b63120
            #
            CODE="
            #include <linux/timer.h>
            int conftest_timer_setup(void) {
                return timer_setup();
            }"
            compile_check_conftest "$CODE" "NV_TIMER_SETUP_PRESENT" "" "functions"
        ;;

        radix_tree_replace_slot)
            #
            # Determine if the radix_tree_replace_slot() function is
            # present and how many arguments it takes.
            #
            # radix_tree_replace_slot added
            #   2006-12-06 7cf9c2c76c1a17b32f2da85b50cd4fe468ed44b5
            # root parameter added to radix_tree_replace_slot in 4.10 (but the symbol was not exported)
            #   2016-12-12 6d75f366b9242f9b17ed7d0b0604d7460f818f21
            # radix_tree_replace_slot symbol export was introduced in 4.11
            #   2017-10-11 10257d719686706aa669b348309cfd9fd9783ad9
            #
            CODE="
            #include <linux/bug.h>
            #include <linux/version.h>
            void conftest_radix_tree_replace_slot(void) {
                BUILD_BUG_ON(LINUX_VERSION_CODE < KERNEL_VERSION(4, 10, 0) || LINUX_VERSION_CODE >= KERNEL_VERSION(4, 11, 0));
            }"
            compile_check_conftest "$CODE" "NV_RADIX_TREE_REPLACE_SLOT_PRESENT" "" "functions"

            echo "$CONFTEST_PREAMBLE
            #include <linux/radix-tree.h>
            void conftest_radix_tree_replace_slot(void) {
                radix_tree_replace_slot(NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_RADIX_TREE_REPLACE_SLOT_ARGUMENT_COUNT 2" | append_conftest "functions"
                return
            fi

            echo "$CONFTEST_PREAMBLE
            #include <linux/radix-tree.h>
            void conftest_radix_tree_replace_slot(void) {
                radix_tree_replace_slot(NULL, NULL, NULL);
            }" > conftest$$.c

            $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
            rm -f conftest$$.c

            if [ -f conftest$$.o ]; then
                rm -f conftest$$.o
                echo "#define NV_RADIX_TREE_REPLACE_SLOT_ARGUMENT_COUNT 3" | append_conftest "functions"
                return
            else
                echo "#error radix_tree_replace_slot() conftest failed!" | append_conftest "functions"
            fi
        ;;

        drm_old_atomic_state_iterators_present)
            #
            # Determine if the old atomic state iterators
            # for_each_crtc_in_state(), for_each_connector_in_state() and
            # for_each_plane_in_state() are present.
            #
            # These old atomic state iterators were added by:
            #   2015-04-10  df63b9994eaf942afcdb946d27a28661d7dfbf2a
            # and they are removed by:
            #   2017-07-19  77ac3b00b13185741effd0d5e2f1f05e4bfef7dc
            #
            CODE="
            #include <drm/drmP.h>
            #include <drm/drm_atomic.h>
            void conftest_drm_old_atomic_state_iterators_present(void) {
                struct drm_crtc_state *crtc_state;
                struct drm_atomic_state *state;
                struct drm_crtc *crtc;
                int i;

                for_each_crtc_in_state(state, crtc, crtc_state, i) {
                }
            }"

            compile_check_conftest "$CODE" "NV_DRM_OLD_ATOMIC_STATE_ITERATORS_PRESENT" | append_conftest "types"
        ;;

        drm_mode_object_find_has_file_priv_arg)
            #
            # Determine if drm_mode_object_find() has 'file_priv' arguments.
            #
            # The function drm_mode_object_find() was added by:
            #   2008-11-07  f453ba0460742ad027ae0c4c7d61e62817b3e7ef
            # and it is updated to take 'file_priv' argument by:
            #   2017-03-14  418da17214aca5ef5f0b6f7588905ee7df92f98f
            #
            CODE="
            #include <drm/drm_mode_object.h>
            void conftest_drm_mode_object_find_has_file_priv_arg(
                    struct drm_device *dev,
                    struct drm_file *file_priv,
                    uint32_t id,
                    uint32_t type) {
                (void)drm_mode_object_find(dev, file_priv, id, type);
            }"

            compile_check_conftest "$CODE" "NV_DRM_MODE_OBJECT_FIND_HAS_FILE_PRIV_ARG" | append_conftest "types"
        ;;
    esac
}

case "$6" in
    cc_sanity_check)
        #
        # Check if the selected compiler can create object files
        # in the current environment.
        #
        VERBOSE=$7

        echo "int cc_sanity_check(void) {
            return 0;
        }" > conftest$$.c

        $CC -c conftest$$.c > /dev/null 2>&1
        rm -f conftest$$.c

        if [ ! -f conftest$$.o ]; then
            if [ "$VERBOSE" = "full_output" ]; then
                echo "";
            fi
            if [ "$CC" != "cc" ]; then
                echo "The C compiler '$CC' does not appear to be able to"
                echo "create object files.  Please make sure you have "
                echo "your Linux distribution's libc development package"
                echo "installed and that '$CC' is a valid C compiler";
                echo "name."
            else
                echo "The C compiler '$CC' does not appear to be able to"
                echo "create executables.  Please make sure you have "
                echo "your Linux distribution's gcc and libc development"
                echo "packages installed."
            fi
            if [ "$VERBOSE" = "full_output" ]; then
                echo "";
                echo "*** Failed CC sanity check. Bailing out! ***";
                echo "";
            fi
            exit 1
        else
            rm -f conftest$$.o
            exit 0
        fi
    ;;

    cc_version_check)
        #
        # Verify that the same compiler major and minor version is
        # used for the kernel and kernel module.
        #
        # Some gcc version strings that have proven problematic for parsing
        # in the past:
        #
        #  gcc.real (GCC) 3.3 (Debian)
        #  gcc-Version 3.3 (Debian)
        #  gcc (GCC) 3.1.1 20020606 (Debian prerelease)
        #  version gcc 3.2.3
        #
        VERBOSE=$7

        kernel_compile_h=$OUTPUT/include/generated/compile.h

        if [ ! -f ${kernel_compile_h} ]; then
            # The kernel's compile.h file is not present, so there
            # isn't a convenient way to identify the compiler version
            # used to build the kernel.
            IGNORE_CC_MISMATCH=1
        fi

        if [ -n "$IGNORE_CC_MISMATCH" ]; then
            exit 0
        fi

        kernel_cc_string=`cat ${kernel_compile_h} | \
            grep LINUX_COMPILER | cut -f 2 -d '"'`

        kernel_cc_version=`echo ${kernel_cc_string} | grep -o '[0-9]\+\.[0-9]\+' | head -n 1`
        kernel_cc_major=`echo ${kernel_cc_version} | cut -d '.' -f 1`
        kernel_cc_minor=`echo ${kernel_cc_version} | cut -d '.' -f 2`

        echo "
        #if (__GNUC__ != ${kernel_cc_major}) || (__GNUC_MINOR__ != ${kernel_cc_minor})
        #error \"cc version mismatch\"
        #endif
        " > conftest$$.c

        $CC $CFLAGS -c conftest$$.c > /dev/null 2>&1
        rm -f conftest$$.c

        if [ -f conftest$$.o ]; then
            rm -f conftest$$.o
            exit 0;
        else
            #
            # The gcc version check failed
            #

            if [ "$VERBOSE" = "full_output" ]; then
                echo "";
                echo "Compiler version check failed:";
                echo "";
                echo "The major and minor number of the compiler used to";
                echo "compile the kernel:";
                echo "";
                echo "${kernel_cc_string}";
                echo "";
                echo "does not match the compiler used here:";
                echo "";
                $CC --version
                echo "";
                echo "It is recommended to set the CC environment variable";
                echo "to the compiler that was used to compile the kernel.";
                echo ""
                echo "The compiler version check can be disabled by setting";
                echo "the IGNORE_CC_MISMATCH environment variable to \"1\".";
                echo "However, mixing compiler versions between the kernel";
                echo "and kernel modules can result in subtle bugs that are";
                echo "difficult to diagnose.";
                echo "";
                echo "*** Failed CC version check. Bailing out! ***";
                echo "";
            elif [ "$VERBOSE" = "just_msg" ]; then
                echo "The kernel was built with ${kernel_cc_string}, but the" \
                     "current compiler version is `$CC --version | head -n 1`.";
            fi
            exit 1;
        fi
    ;;

    get_uname)
        #
        # Print UTS_RELEASE from the kernel sources, if the kernel header
        # file ../linux/version.h or ../linux/utsrelease.h exists. If
        # neither header file is found, but a Makefile is found, extract
        # PATCHLEVEL and SUBLEVEL, and use them to build the kernel
        # release name.
        #
        # If no source file is found, or if an error occurred, return the
        # output of `uname -r`.
        #
        RET=1
        DIRS="generated linux"
        FILE=""
        
        for DIR in $DIRS; do
            if [ -f $HEADERS/$DIR/utsrelease.h ]; then
                FILE="$HEADERS/$DIR/utsrelease.h"
                break
            elif [ -f $OUTPUT/include/$DIR/utsrelease.h ]; then
                FILE="$OUTPUT/include/$DIR/utsrelease.h"
                break
            fi
        done

        if [ -z "$FILE" ]; then
            if [ -f $HEADERS/linux/version.h ]; then
                FILE="$HEADERS/linux/version.h"
            elif [ -f $OUTPUT/include/linux/version.h ]; then
                FILE="$OUTPUT/include/linux/version.h"
            fi
        fi

        if [ -n "$FILE" ]; then
            #
            # We are either looking at a configured kernel source tree
            # or at headers shipped for a specific kernel.  Determine
            # the kernel version using a CPP check.
            #
            VERSION=`echo "UTS_RELEASE" | $CC - -E -P -include $FILE 2>&1`

            if [ "$?" = "0" -a "VERSION" != "UTS_RELEASE" ]; then
                echo "$VERSION"
                RET=0
            fi
        else
            #
            # If none of the kernel headers ar found, but a Makefile is,
            # extract PATCHLEVEL and SUBLEVEL and use them to find
            # the kernel version.
            #
            MAKEFILE=$HEADERS/../Makefile

            if [ -f $MAKEFILE ]; then
                #
                # This source tree is not configured, but includes
                # the top-level Makefile.
                #
                PATCHLEVEL=$(grep "^PATCHLEVEL =" $MAKEFILE | cut -d " " -f 3)
                SUBLEVEL=$(grep "^SUBLEVEL =" $MAKEFILE | cut -d " " -f 3)

                if [ -n "$PATCHLEVEL" -a -n "$SUBLEVEL" ]; then
                    echo 2.$PATCHLEVEL.$SUBLEVEL
                    RET=0
                fi
            fi
        fi

        if [ "$RET" != "0" ]; then
            uname -r
            exit 1
        else
            exit 0
        fi
    ;;

    xen_sanity_check)
        #
        # Check if the target kernel is a Xen kernel. If so, exit, since
        # the RM doesn't currently support Xen.
        #
        VERBOSE=$7

        if [ -n "$IGNORE_XEN_PRESENCE" -o -n "$VGX_BUILD" ]; then
            exit 0
        fi

        test_xen

        if [ "$XEN_PRESENT" != "0" ]; then
            echo "The kernel you are installing for is a Xen kernel!";
            echo "";
            echo "The NVIDIA driver does not currently support Xen kernels. If ";
            echo "you are using a stock distribution kernel, please install ";
            echo "a variant of this kernel without Xen support; if this is a ";
            echo "custom kernel, please install a standard Linux kernel.  Then ";
            echo "try installing the NVIDIA kernel module again.";
            echo "";
            if [ "$VERBOSE" = "full_output" ]; then
                echo "*** Failed Xen sanity check. Bailing out! ***";
                echo "";
            fi
            exit 1
        else
            exit 0
        fi
    ;;

    preempt_rt_sanity_check)
        #
        # Check if the target kernel has the PREEMPT_RT patch set applied. If
        # so, exit, since the RM doesn't support this configuration.
        #
        VERBOSE=$7

        if [ -n "$IGNORE_PREEMPT_RT_PRESENCE" ]; then
            exit 0
        fi

        if test_configuration_option CONFIG_PREEMPT_RT; then
            PREEMPT_RT_PRESENT=1
        elif test_configuration_option CONFIG_PREEMPT_RT_FULL; then
            PREEMPT_RT_PRESENT=1
        fi

        if [ "$PREEMPT_RT_PRESENT" != "0" ]; then
            echo "The kernel you are installing for is a PREEMPT_RT kernel!";
            echo "";
            echo "The NVIDIA driver does not support real-time kernels. If you ";
            echo "are using a stock distribution kernel, please install ";
            echo "a variant of this kernel that does not have the PREEMPT_RT ";
            echo "patch set applied; if this is a custom kernel, please ";
            echo "install a standard Linux kernel.  Then try installing the ";
            echo "NVIDIA kernel module again.";
            echo "";
            if [ "$VERBOSE" = "full_output" ]; then
                echo "*** Failed PREEMPT_RT sanity check. Bailing out! ***";
                echo "";
            fi
            exit 1
        else
            exit 0
        fi
    ;;

    patch_check)
        #
        # Check for any "official" patches that may have been applied and
        # construct a description table for reporting purposes.
        #
        PATCHES=""

        for PATCH in patch-*.h; do
            if [ -f $PATCH ]; then
                echo "#include \"$PATCH\""
                PATCHES="$PATCHES "`echo $PATCH | sed -s 's/patch-\(.*\)\.h/\1/'`
            fi
        done

        echo "static struct {
                const char *short_description;
                const char *description;
              } __nv_patches[] = {"
            for i in $PATCHES; do
                echo "{ \"$i\", NV_PATCH_${i}_DESCRIPTION },"
            done
        echo "{ NULL, NULL } };"

        exit 0
    ;;

    compile_tests)
        #
        # Run a series of compile tests to determine the set of interfaces
        # and features available in the target kernel.
        #
        shift 6

        CFLAGS=$1
        shift

        for i in $*; do compile_test $i; done

        exit 0
    ;;

    dom0_sanity_check)
        #
        # Determine whether running in DOM0.
        #
        VERBOSE=$7

        if [ -n "$VGX_BUILD" ]; then
            if [ -f /proc/xen/capabilities ]; then
                if [ "`cat /proc/xen/capabilities`" == "control_d" ]; then
                    exit 0
                fi
            else
                echo "The kernel is not running in DOM0.";
                echo "";
                if [ "$VERBOSE" = "full_output" ]; then
                    echo "*** Failed DOM0 sanity check. Bailing out! ***";
                    echo "";
                fi
            fi
            exit 1
        fi
    ;;
    vgpu_kvm_sanity_check)
        #
        # Determine whether we are running a vGPU on KVM host.
        #
        VERBOSE=$7
        iommu=CONFIG_VFIO_IOMMU_TYPE1
        mdev=CONFIG_VFIO_MDEV_DEVICE
        kvm=CONFIG_KVM_VFIO

        if [ -n "$VGX_KVM_BUILD" ]; then
            if (test_configuration_option ${iommu} || test_configuration_option ${iommu}_MODULE) &&
               (test_configuration_option ${mdev} || test_configuration_option ${mdev}_MODULE) &&
               (test_configuration_option ${kvm} || test_configuration_option ${kvm}_MODULE); then
                    exit 0
            else
                echo "The kernel is not running a vGPU on KVM host.";
                echo "";
                if [ "$VERBOSE" = "full_output" ]; then
                    echo "*** Failed vGPU on KVM sanity check. Bailing out! ***";
                    echo "";
                fi
            fi
            exit 1
        else
            exit 0
        fi
    ;;
    test_configuration_option)
        #
        # Check to see if the given config option is set.
        #
        OPTION=$7

        test_configuration_option $OPTION
        exit $?
    ;;

    get_configuration_option)
        #
        # Get the value of the given config option.
        #
        OPTION=$7

        get_configuration_option $OPTION
        exit $?
    ;;


    guess_module_signing_hash)
        #
        # Determine the best cryptographic hash to use for module signing,
        # to the extent that is possible.
        #

        HASH=$(get_configuration_option CONFIG_MODULE_SIG_HASH)

        if [ $? -eq 0 ] && [ -n $HASH ]; then
            echo $HASH
            exit 0
        else
            for SHA in 512 384 256 224 1; do
                if test_configuration_option CONFIG_MODULE_SIG_SHA$SHA; then
                    echo sha$SHA
                    exit 0
                fi
            done
        fi
        exit 1
    ;;


    test_kernel_headers)
        #
        # Check for the availability of certain kernel headers
        #

        test_headers
        exit $?
    ;;


    build_cflags)
        #
        # Generate CFLAGS for use in the compile tests
        #

        build_cflags
        echo $CFLAGS
        exit 0
    ;;

esac
