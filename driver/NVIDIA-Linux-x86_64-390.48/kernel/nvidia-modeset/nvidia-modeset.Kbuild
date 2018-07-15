###########################################################################
# Kbuild fragment for nvidia-modeset.ko
###########################################################################

#
# Define NVIDIA_MODESET_{SOURCES,OBJECTS}
#

NVIDIA_MODESET_SOURCES = nvidia-modeset/nvidia-modeset-linux.c
NVIDIA_MODESET_OBJECTS = $(patsubst %.c,%.o,$(NVIDIA_MODESET_SOURCES))

obj-m += nvidia-modeset.o
nvidia-modeset-y := $(NVIDIA_MODESET_OBJECTS)

NVIDIA_MODESET_KO = nvidia-modeset/nvidia-modeset.ko

NV_KERNEL_MODULE_TARGETS += $(NVIDIA_MODESET_KO)


#
# nv-modeset-kernel.o_binary is the core binary component of nvidia-modeset.ko,
# shared across all UNIX platforms. Create a symlink, "nv-modeset-kernel.o"
# that points to nv-modeset-kernel.o_binary, and add nv-modeset-kernel.o to the
# list of objects to link into nvidia-modeset.ko.
#
# Note that:
# - The kbuild "clean" rule will delete all objects in nvidia-modeset-y (which
# is why we use a symlink instead of just adding nv-modeset-kernel.o_binary
# to nvidia-modeset-y).
# - kbuild normally uses the naming convention of ".o_shipped" for
# binary files. That is not used here, because the kbuild rule to
# create the "normal" object file from ".o_shipped" does a copy, not
# a symlink. This file is quite large, so a symlink is preferred.
# - The file added to nvidia-modeset-y should be relative to gmake's cwd.
# But, the target for the symlink rule should be prepended with $(obj).
#

NVIDIA_MODESET_BINARY_OBJECT := $(src)/nvidia-modeset/nv-modeset-kernel.o_binary
NVIDIA_MODESET_BINARY_OBJECT_O := nvidia-modeset/nv-modeset-kernel.o

quiet_cmd_symlink = SYMLINK $@
cmd_symlink = ln -sf $< $@

targets += $(NVIDIA_MODESET_BINARY_OBJECT_O)

$(obj)/$(NVIDIA_MODESET_BINARY_OBJECT_O): $(NVIDIA_MODESET_BINARY_OBJECT) FORCE
	$(call if_changed,symlink)

nvidia-modeset-y += $(NVIDIA_MODESET_BINARY_OBJECT_O)


#
# Define nvidia-modeset.ko-specific CFLAGS.
#

NVIDIA_MODESET_CFLAGS += -I$(src)/nvidia-modeset
NVIDIA_MODESET_CFLAGS += -UDEBUG -U_DEBUG -DNDEBUG -DNV_BUILD_MODULE_INSTANCES=0

$(call ASSIGN_PER_OBJ_CFLAGS, $(NVIDIA_MODESET_OBJECTS), $(NVIDIA_MODESET_CFLAGS))


#
# Build nv-modeset-interface.o from the kernel interface layer
# objects, suitable for further processing by the installer and
# inclusion as a precompiled kernel interface file.
#

NVIDIA_MODESET_INTERFACE := nvidia-modeset/nv-modeset-interface.o

always += $(NVIDIA_MODESET_INTERFACE)

$(obj)/$(NVIDIA_MODESET_INTERFACE): $(addprefix $(obj)/,$(NVIDIA_MODESET_OBJECTS))
	$(LD) -r -o $@ $^

#
# Register the conftests needed by nvidia-modeset.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(NVIDIA_MODESET_OBJECTS)

NV_CONFTEST_MACRO_COMPILE_TESTS += INIT_WORK
NV_CONFTEST_TYPE_COMPILE_TESTS += file_inode
NV_CONFTEST_TYPE_COMPILE_TESTS += file_operations
NV_CONFTEST_TYPE_COMPILE_TESTS += proc_dir_entry
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_create_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pde_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_remove
NV_CONFTEST_FUNCTION_COMPILE_TESTS += timer_setup
