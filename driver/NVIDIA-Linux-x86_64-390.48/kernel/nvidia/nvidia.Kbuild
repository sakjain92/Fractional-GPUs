###########################################################################
# Kbuild fragment for nvidia.ko
###########################################################################

#
# Define NVIDIA_{SOURCES,OBJECTS}
#

include $(src)/nvidia/nvidia-sources.Kbuild
NVIDIA_OBJECTS = $(patsubst %.c,%.o,$(NVIDIA_SOURCES))

obj-m += nvidia.o
nvidia-y := $(NVIDIA_OBJECTS)

NVIDIA_KO = nvidia/nvidia.ko


#
# nv-kernel.o_binary is the core binary component of nvidia.ko, shared
# across all UNIX platforms. Create a symlink, "nv-kernel.o" that
# points to nv-kernel.o_binary, and add nv-kernel.o to the list of
# objects to link into nvidia.ko.
#
# Note that:
# - The kbuild "clean" rule will delete all objects in nvidia-y (which
# is why we use a symlink instead of just adding nv-kernel.o_binary
# to nvidia-y).
# - kbuild normally uses the naming convention of ".o_shipped" for
# binary files. That is not used here, because the kbuild rule to
# create the "normal" object file from ".o_shipped" does a copy, not
# a symlink. This file is quite large, so a symlink is preferred.
# - The file added to nvidia-y should be relative to gmake's cwd.
# But, the target for the symlink rule should be prepended with $(obj).
# - The "symlink" command is called using kbuild's if_changed macro to
# generate an .nv-kernel.o.cmd file which can be used on subsequent
# runs to determine if the command line to create the symlink changed
# and needs to be re-executed.
#

NVIDIA_BINARY_OBJECT := $(src)/nvidia/nv-kernel.o_binary
NVIDIA_BINARY_OBJECT_O := nvidia/nv-kernel.o

quiet_cmd_symlink = SYMLINK $@
 cmd_symlink = ln -sf $< $@

targets += $(NVIDIA_BINARY_OBJECT_O)

$(obj)/$(NVIDIA_BINARY_OBJECT_O): $(NVIDIA_BINARY_OBJECT) FORCE
	$(call if_changed,symlink)

nvidia-y += $(NVIDIA_BINARY_OBJECT_O)


#
# Define nvidia.ko-specific CFLAGS.
#

NVIDIA_CFLAGS += -I$(src)/nvidia
NVIDIA_CFLAGS += -DNV_BUILD_MODULE_INSTANCES=0
NVIDIA_CFLAGS += -DNVIDIA_UNDEF_LEGACY_BIT_MACROS
NVIDIA_CFLAGS += -UDEBUG -U_DEBUG -DNDEBUG

$(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_OBJECTS), $(NVIDIA_CFLAGS))


#
# nv-procfs.c requires nv-compiler.h
#

NV_COMPILER_VERSION_HEADER = $(obj)/nv_compiler.h

$(NV_COMPILER_VERSION_HEADER):
	@echo \#define NV_COMPILER \"`$(CC) -v 2>&1 | tail -n 1`\" > $@

$(obj)/nvidia/nv-procfs.o: $(NV_COMPILER_VERSION_HEADER)

clean-files += $(NV_COMPILER_VERSION_HEADER)


#
# Build nv-interface.o from the kernel interface layer objects, suitable
# for further processing by the top-level makefile to produce a precompiled
# kernel interface file.
#

NVIDIA_INTERFACE := nvidia/nv-interface.o

always += $(NVIDIA_INTERFACE)

$(obj)/$(NVIDIA_INTERFACE): $(addprefix $(obj)/,$(NVIDIA_OBJECTS))
	$(LD) -r -o $@ $^


#
# Register the conftests needed by nvidia.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(NVIDIA_OBJECTS)

NV_CONFTEST_FUNCTION_COMPILE_TESTS += remap_pfn_range
NV_CONFTEST_FUNCTION_COMPILE_TESTS += hash__remap_4k_pfn
NV_CONFTEST_FUNCTION_COMPILE_TESTS += follow_pfn
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vmap
NV_CONFTEST_FUNCTION_COMPILE_TESTS += set_pages_uc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += set_memory_uc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += set_memory_array_uc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += change_page_attr
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_get_class
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_choose_state
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vm_insert_page
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acpi_device_id
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acquire_console_sem
NV_CONFTEST_FUNCTION_COMPILE_TESTS += console_lock
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kmem_cache_create
NV_CONFTEST_FUNCTION_COMPILE_TESTS += on_each_cpu
NV_CONFTEST_FUNCTION_COMPILE_TESTS += smp_call_function
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acpi_evaluate_integer
NV_CONFTEST_FUNCTION_COMPILE_TESTS += ioremap_cache
NV_CONFTEST_FUNCTION_COMPILE_TESTS += ioremap_wc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acpi_walk_namespace
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_domain_nr
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_dma_mapping_error
NV_CONFTEST_FUNCTION_COMPILE_TESTS += sg_alloc_table
NV_CONFTEST_FUNCTION_COMPILE_TESTS += sg_init_table
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_get_domain_bus_and_slot
NV_CONFTEST_FUNCTION_COMPILE_TESTS += get_num_physpages
NV_CONFTEST_FUNCTION_COMPILE_TESTS += efi_enabled
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_create_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pde_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_remove
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pm_vt_switch_required
NV_CONFTEST_FUNCTION_COMPILE_TESTS += xen_ioemu_inject_msi
NV_CONFTEST_FUNCTION_COMPILE_TESTS += phys_to_dma
NV_CONFTEST_FUNCTION_COMPILE_TESTS += get_dma_ops
NV_CONFTEST_FUNCTION_COMPILE_TESTS += write_cr4
NV_CONFTEST_FUNCTION_COMPILE_TESTS += of_get_property
NV_CONFTEST_FUNCTION_COMPILE_TESTS += of_find_node_by_phandle
NV_CONFTEST_FUNCTION_COMPILE_TESTS += of_node_to_nid
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pnv_pci_get_npu_dev
NV_CONFTEST_FUNCTION_COMPILE_TESTS += for_each_online_node
NV_CONFTEST_FUNCTION_COMPILE_TESTS += node_end_pfn
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_bus_address
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_stop_and_remove_bus_device
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_remove_bus_device
NV_CONFTEST_FUNCTION_COMPILE_TESTS += request_threaded_irq
NV_CONFTEST_FUNCTION_COMPILE_TESTS += register_cpu_notifier
NV_CONFTEST_FUNCTION_COMPILE_TESTS += cpuhp_setup_state
NV_CONFTEST_FUNCTION_COMPILE_TESTS += dma_map_resource
NV_CONFTEST_FUNCTION_COMPILE_TESTS += backlight_device_register
NV_CONFTEST_FUNCTION_COMPILE_TESTS += register_acpi_notifier
NV_CONFTEST_FUNCTION_COMPILE_TESTS += timer_setup

NV_CONFTEST_SYMBOL_COMPILE_TESTS += is_export_symbol_gpl_of_node_to_nid

NV_CONFTEST_TYPE_COMPILE_TESTS += i2c_adapter
NV_CONFTEST_TYPE_COMPILE_TESTS += pm_message_t
NV_CONFTEST_TYPE_COMPILE_TESTS += irq_handler_t
NV_CONFTEST_TYPE_COMPILE_TESTS += acpi_device_ops
NV_CONFTEST_TYPE_COMPILE_TESTS += acpi_op_remove
NV_CONFTEST_TYPE_COMPILE_TESTS += acpi_device_id
NV_CONFTEST_TYPE_COMPILE_TESTS += outer_flush_all
NV_CONFTEST_TYPE_COMPILE_TESTS += proc_dir_entry
NV_CONFTEST_TYPE_COMPILE_TESTS += scatterlist
NV_CONFTEST_TYPE_COMPILE_TESTS += sg_table
NV_CONFTEST_TYPE_COMPILE_TESTS += file_operations
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_operations_struct
NV_CONFTEST_TYPE_COMPILE_TESTS += atomic_long_type
NV_CONFTEST_TYPE_COMPILE_TESTS += pci_save_state
NV_CONFTEST_TYPE_COMPILE_TESTS += file_inode
NV_CONFTEST_TYPE_COMPILE_TESTS += task_struct
NV_CONFTEST_TYPE_COMPILE_TESTS += kuid_t
NV_CONFTEST_TYPE_COMPILE_TESTS += dma_ops
NV_CONFTEST_TYPE_COMPILE_TESTS += dma_map_ops
NV_CONFTEST_TYPE_COMPILE_TESTS += noncoherent_swiotlb_dma_ops
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_present
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_has_address
NV_CONFTEST_TYPE_COMPILE_TESTS += kernel_write
NV_CONFTEST_TYPE_COMPILE_TESTS += strnstr
NV_CONFTEST_TYPE_COMPILE_TESTS += iterate_dir
NV_CONFTEST_TYPE_COMPILE_TESTS += kstrtoull
NV_CONFTEST_TYPE_COMPILE_TESTS += backlight_properties_type

NV_CONFTEST_GENERIC_COMPILE_TESTS += dom0_kernel_present
NV_CONFTEST_GENERIC_COMPILE_TESTS += nvidia_vgpu_kvm_build
NV_CONFTEST_GENERIC_COMPILE_TESTS += nvidia_grid_build
NV_CONFTEST_GENERIC_COMPILE_TESTS += get_user_pages
NV_CONFTEST_GENERIC_COMPILE_TESTS += get_user_pages_remote
NV_CONFTEST_GENERIC_COMPILE_TESTS += list_cut_position

NV_CONFTEST_MACRO_COMPILE_TESTS += INIT_WORK
