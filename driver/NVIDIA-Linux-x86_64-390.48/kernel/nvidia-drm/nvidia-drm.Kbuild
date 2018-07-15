###########################################################################
# Kbuild fragment for nvidia-drm.ko
###########################################################################

#
# Define NVIDIA_DRM_{SOURCES,OBJECTS}
#

NVIDIA_DRM_SOURCES =
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-drv.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-utils.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-crtc.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-encoder.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-connector.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-gem.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-fb.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-modeset.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-prime-fence.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-linux.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-helper.c
NVIDIA_DRM_SOURCES += nvidia-drm/nv-pci-table.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-gem-nvkms-memory.c
NVIDIA_DRM_SOURCES += nvidia-drm/nvidia-drm-gem-user-memory.c

NVIDIA_DRM_OBJECTS = $(patsubst %.c,%.o,$(NVIDIA_DRM_SOURCES))

obj-m += nvidia-drm.o
nvidia-drm-y := $(NVIDIA_DRM_OBJECTS)

NVIDIA_DRM_KO = nvidia-drm/nvidia-drm.ko

NV_KERNEL_MODULE_TARGETS += $(NVIDIA_DRM_KO)

#
# Define nvidia-drm.ko-specific CFLAGS.
#

NVIDIA_DRM_CFLAGS += -I$(src)/nvidia-drm
NVIDIA_DRM_CFLAGS += -UDEBUG -U_DEBUG -DNDEBUG -DNV_BUILD_MODULE_INSTANCES=0

$(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_DRM_OBJECTS), $(NVIDIA_DRM_CFLAGS))

#
# Register the conftests needed by nvidia-drm.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(NVIDIA_DRM_OBJECTS)

NV_CONFTEST_GENERIC_COMPILE_TESTS += drm_available
NV_CONFTEST_GENERIC_COMPILE_TESTS += drm_atomic_available
NV_CONFTEST_GENERIC_COMPILE_TESTS += drm_atomic_modeset_nonblocking_commit_available
NV_CONFTEST_GENERIC_COMPILE_TESTS += is_export_symbol_gpl_refcount_inc
NV_CONFTEST_GENERIC_COMPILE_TESTS += is_export_symbol_gpl_refcount_dec_and_test

NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_dev_unref
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_reinit_primary_mode_group
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_set_mode_for_crtc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_clean_old_fb
NV_CONFTEST_FUNCTION_COMPILE_TESTS += get_user_pages_remote
NV_CONFTEST_FUNCTION_COMPILE_TESTS += get_user_pages
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_gem_object_lookup
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_state_free
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_driver_has_gem_prime_res_obj
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_helper_disable_all
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_helper_set_config
NV_CONFTEST_FUNCTION_COMPILE_TESTS += drm_atomic_helper_connector_dpms

NV_CONFTEST_TYPE_COMPILE_TESTS += drm_bus_present
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_bus_has_bus_type
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_bus_has_get_irq
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_bus_has_get_name
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_driver_has_legacy_dev_list
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_driver_has_set_busid
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_crtc_state_has_connectors_changed
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_init_function_args
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_mode_connector_list_update_has_merge_type_bits_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_helper_mode_fill_fb_struct
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_master_drop_has_from_release_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_driver_unload_has_int_return_type
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_has_address
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_present
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_ops_fault_removed_vma_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += kref_has_refcount_of_type_refcount_t
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_atomic_helper_crtc_destroy_state_has_crtc_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_crtc_helper_funcs_has_atomic_enable
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_old_atomic_state_iterators_present
NV_CONFTEST_TYPE_COMPILE_TESTS += drm_mode_object_find_has_file_priv_arg
