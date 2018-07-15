###########################################################################
# Kbuild fragment for nvidia-uvm.ko
###########################################################################

UVM_BUILD_TYPE = release

MIN_VERSION    := 2
MIN_PATCHLEVEL := 6
MIN_SUBLEVEL   := 32

KERNEL_VERSION_NUMERIC := $(shell echo $$(( $(VERSION) * 65536 + $(PATCHLEVEL) * 256 + $(SUBLEVEL) )))
MIN_VERSION_NUMERIC    := $(shell echo $$(( $(MIN_VERSION) * 65536 + $(MIN_PATCHLEVEL) * 256 + $(MIN_SUBLEVEL) )))

KERNEL_NEW_ENOUGH_FOR_UVM := $(shell [ $(KERNEL_VERSION_NUMERIC) -ge $(MIN_VERSION_NUMERIC) ] && echo 1)

#
# Define NVIDIA_UVM_{SOURCES,OBJECTS}
#

NVIDIA_UVM_OBJECTS =
NVIDIA_UVM_UNSUPPORTED_SOURCE := nvidia-uvm/uvm_unsupported.c

ifeq ($(KERNEL_NEW_ENOUGH_FOR_UVM),1)
  include $(src)/nvidia-uvm/nvidia-uvm-sources.Kbuild
  NVIDIA_UVM_OBJECTS += $(patsubst %.c,%.o,\
      $(filter-out $(NVIDIA_UVM_UNSUPPORTED_SOURCE),$(NVIDIA_UVM_SOURCES)))
else
  NVIDIA_UVM_SOURCES = $(NVIDIA_UVM_UNSUPPORTED_SOURCE)
  NVIDIA_UVM_OBJECTS += $(patsubst %.c,%.o,$(NVIDIA_UVM_SOURCES))
endif

# Some linux kernel functions rely on being built with optimizations on and
# to work around this we put wrappers for them in a separate file that's built
# with optimizations on in debug builds and skipped in other builds.
# Notably gcc 4.4 supports per function optimization attributes that would be
# easier to use, but is too recent to rely on for now.
NVIDIA_UVM_DEBUG_OPTIMIZED_SOURCE := nvidia-uvm/uvm_debug_optimized.c
NVIDIA_UVM_DEBUG_OPTIMIZED_OBJECT := $(patsubst %.c,%.o,$(NVIDIA_UVM_DEBUG_OPTIMIZED_SOURCE))

ifneq ($(UVM_BUILD_TYPE),debug)
  # Only build the wrappers on debug builds
  NVIDIA_UVM_OBJECTS := $(filter-out $(NVIDIA_UVM_DEBUG_OPTIMIZED_OBJECT), $(NVIDIA_UVM_OBJECTS))
endif

obj-m += nvidia-uvm.o
nvidia-uvm-y := $(NVIDIA_UVM_OBJECTS)

NVIDIA_UVM_KO = nvidia-uvm/nvidia-uvm.ko

#
# Define nvidia-uvm.ko-specific CFLAGS.
#

ifeq ($(UVM_BUILD_TYPE),debug)
  NVIDIA_UVM_CFLAGS += -DDEBUG $(call cc-option,-Og,-O0) -g
else
  ifeq ($(UVM_BUILD_TYPE),develop)
    # -DDEBUG is required, in order to allow pr_devel() print statements to
    # work:
    NVIDIA_UVM_CFLAGS += -DDEBUG
    NVIDIA_UVM_CFLAGS += -DNVIDIA_UVM_DEVELOP
  endif
  NVIDIA_UVM_CFLAGS += -O2
endif

NVIDIA_UVM_CFLAGS += -DNVIDIA_UVM_ENABLED
NVIDIA_UVM_CFLAGS += -DNVIDIA_UNDEF_LEGACY_BIT_MACROS

NVIDIA_UVM_CFLAGS += -DLinux
NVIDIA_UVM_CFLAGS += -D__linux__
NVIDIA_UVM_CFLAGS += -I$(src)/nvidia-uvm

# Avoid even building HMM until the HMM patch is in the upstream kernel.
# Bug 1772628 has details.
NV_BUILD_SUPPORTS_HMM ?= 0

ifeq ($(NV_BUILD_SUPPORTS_HMM),1)
  NVIDIA_UVM_CFLAGS += -DNV_BUILD_SUPPORTS_HMM
endif

$(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_UVM_OBJECTS), $(NVIDIA_UVM_CFLAGS))

ifeq ($(UVM_BUILD_TYPE),debug)
  # Force optimizations on for the wrappers
  $(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_UVM_DEBUG_OPTIMIZED_OBJECT), $(NVIDIA_UVM_CFLAGS) -O2)
endif

#
# Register the conftests needed by nvidia-uvm.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(NVIDIA_UVM_OBJECTS)

NV_CONFTEST_FUNCTION_COMPILE_TESTS += remap_page_range
NV_CONFTEST_FUNCTION_COMPILE_TESTS += remap_pfn_range
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vm_insert_page
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kmem_cache_create
NV_CONFTEST_FUNCTION_COMPILE_TESTS += address_space_init_once
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kbasename
NV_CONFTEST_FUNCTION_COMPILE_TESTS += fatal_signal_pending
NV_CONFTEST_FUNCTION_COMPILE_TESTS += list_cut_position
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vzalloc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += wait_on_bit_lock_argument_count
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_create_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pde_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_remove
NV_CONFTEST_FUNCTION_COMPILE_TESTS += bitmap_clear
NV_CONFTEST_FUNCTION_COMPILE_TESTS += usleep_range
NV_CONFTEST_FUNCTION_COMPILE_TESTS += radix_tree_empty
NV_CONFTEST_FUNCTION_COMPILE_TESTS += radix_tree_replace_slot

NV_CONFTEST_TYPE_COMPILE_TESTS += proc_dir_entry
NV_CONFTEST_TYPE_COMPILE_TESTS += irq_handler_t
NV_CONFTEST_TYPE_COMPILE_TESTS += outer_flush_all
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_operations_struct
NV_CONFTEST_TYPE_COMPILE_TESTS += file_operations
NV_CONFTEST_TYPE_COMPILE_TESTS += task_struct
NV_CONFTEST_TYPE_COMPILE_TESTS += kuid_t
NV_CONFTEST_TYPE_COMPILE_TESTS += fault_flags
NV_CONFTEST_TYPE_COMPILE_TESTS += atomic64_type
NV_CONFTEST_TYPE_COMPILE_TESTS += address_space
NV_CONFTEST_TYPE_COMPILE_TESTS += backing_dev_info
NV_CONFTEST_TYPE_COMPILE_TESTS += mm_context_t
NV_CONFTEST_TYPE_COMPILE_TESTS += get_user_pages_remote
NV_CONFTEST_TYPE_COMPILE_TESTS += get_user_pages
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_has_address
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_present
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_ops_fault_removed_vma_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += pnv_npu2_init_context
