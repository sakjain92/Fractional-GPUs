# Pre-requisites/Setup/Build/Installation

This file contains details regarding pre-requisites, setup, build and installation
of FGPU on your machine.

## Pre-requisites
* Linux
    * We have only tested on x86 machines. Specifically a x86_64 machine using Ubuntu 16.04.
* cmake (Minimum version 3.8)
* make
* gcc

## Configuration

FGPU can be configured upto certain degree.

### Global configuration
File **config.cmake.in** contains the following options that can be configured:

* **FGPU_COMP_COLORING_ENABLE**
    * Default - Enabled
    * Disabling this disables compute partitioning. In this case, each application utilizes whole GPU.
    * Enabling this enables compute partitioning. In this case, each application utilizes only subsets of total SMs in a GPU.

* **FGPU_MEM_COLORING_ENABLED**
    * Default - Enabled
    * Disabling this disabled memory bandwidth partitioning. In this case, each application utilizes whole GPU memory bandwidth.
    * Enabling this enabled memory bandwidth partitioning. In this case, each application utilizes only a fraction of whole GPU memory bandwidth.

* **FGPU_TEST_MEM_COLORING_ENABLED**
    * Default - Disabled.
    * Enabling this enables contiguous memory allocation when using fgpu_memory_allocate() API.
    * This feature is useful only when reverse engineering a new GPU.
    * To be kept disabled during production mode.

* **FGPU_USER_MEM_COLORING_ENABLED**
    * Default - Disabled.
    * Deprecated - Keep default value.
    * Its purpose was to do memory coloring while using large pages.

* **FGPU_PARANOID_CHECK_ENABLED**
    * Default - Disabled.
    * Deprecated - Keep default value.

* **FGPU_COMPUTE_CHECK_ENABLED**
    * Default - Disabled.
    * Deprecated - Keep default value.

* **FGPU_SERIALIZED_LAUNCH**
    * Default - Disabled.
    * Deprecated - Keep default value.

Hence an application using FGPU can run in these modes:
* **No partitioning**
    * *FGPU_COMP_COLORING_ENABLE* is disabled.
    * *FGPU_MEM_COLORING_ENABLED* is disabled.
    * One application runs at a time using the whole GPU.
* **Compute partitioning**
    * *FGPU_COMP_COLORING_ENABLE* is enabled.
    * *FGPU_MEM_COLORING_ENABLED* is disabled.
    * Multiple applications can run together.
* **Compute and Memory bandwidth partitioning**
    * *FGPU_COMP_COLORING_ENABLE* is enabled.
    * *FGPU_MEM_COLORING_ENABLED* is enabled.
    * Multiple applications can run together.
* **Reverse engineering** (Not to be used to run external application)
    * *FGPU_COMP_COLORING_ENABLE* is enabled.
    * *FGPU_MEM_COLORING_ENABLED* is disabled.
    * *FGPU_TEST_MEM_COLORING_ENABLED* is enabled.
    * Only reverse engineering code is intended to run (one at a time) in this scenario.


### Fine tuning configuration

For fine tuning, parameters in **include/fgpu_internal_config.hpp** can be modified. Specifically,
the following paramters might be of interest:

* **FGPU_PREFERRED_NUM_COLORS**
    * Default - 2
    * This parameter is used to provide hints to FGPU regarding the number of partitions wanted.
    * This only specifices an upper bound on the total number of partitions. If it is -1, effectively no hint is passed to FGPU.
    * If only compute partitioning is enabled, number of partitions = MIN(number of SM, FGPU_PREFERRED_NUM_COLORS)
    * If both compute and memory bandwidth partitioning is enabled, number of partitions = MIN(number of SM, number of memory colors, FGPU_PREFERRED_NUM_COLORS)
    * Currently each partition has equal number of SMs and equal memory bandwidth.

## Setup
FGPU requires following setup prior to build/installation:
* CUDA SDK 9.1 is required.
    * CUDA SDK version can be probed
        ```
        nvcc --version
        ```
    * If nvcc command does not exists, then CUDA SDK is not installed.
    * If CUDA SDK is not 9.1, remove it and install correct verions
        * To remove CUDA SDK
            ```
            sudo apt-get --purge remove 'cuda*'
            ```
        * To install CUDA SDK 9.1, download it from https://developer.nvidia.com/cuda-downloads
        * After installing add the following to ~/.bashrc file
            ```
            export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
            export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
            ```
* Nvidia driver needs to be uninstalled
    * To remove old Nvidia drivers
        ```
        sudo apt-get purge nvidia*
        ```

## Build

To build the FGPU code, follow these steps
```
cd $PROJ_DIR
mkdir build
cd build
cmake ..
make
```

In the build directory, following files should be present:

* *libfractional_gpu.so* - Link external applications with this library
* *fgpu_server* - Server that is required by FGPU applications.

## Installation

Nvidia device driver is also configured by the FGPU configuration parameters.
Hence only after the build step, Nvidia device driver should to be installed. And the Nvidia driver
needs to be installed before any application can run. Following are the steps to install device driver:

* Disable nouveau driver (This step needs to be only done once)
    * Create a new file */etc/modprobe.d/blacklist-nouveau.conf*
    * Add following contents
        ```
        blacklist nouveau
        options nouveau modeset=0
        ```
    * Regenerate the kernel initramfs
        ```
        sudo update-initramfs -u
        ```
    * Reboot
        ```
        sudo reboot
        ```

* Stop X server
    * X server is responsible for graphics. It need to be stopped for driver to be installed.
        ```
        sudo service lightdm stop
        ```
    * Keep it stopped if running benchmarks as you do not want graphics applications to mess with your evaluations.

* Stop all CUDA programs
    * Before install Nvidia driver, all existing applications using Nvidia driver needs to be stopped.
    * Nvidia MPS also needs to be stopped
        ```
        sudo $PROJ_DIR/scripts/mps_stop.sh
        ```

* Compile and install nvidia drivers
    * Nvidia provides an installation script
        ```
        cd $PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48
        sudo ./nvidia-installer
        ```
    * Follow all default options that come on screen.

## Note
Each time any of the options in **config.cmake.in** are modified, all the build and installation
steps needs to be repeated. Your application code might also need to be compiled again
(as these options modify the header files). Hence all the following steps needs to be redone 
(in the same order as described below):

* Rebuild FGPU.
    * Delete all the build files and CMakeCache files.
        ```
        cd $PROJ_DIR/build
        rm -rf *
        ```
    * Recompile.
        ```
        cmake ..
        make
        ```
* Compile and Install device driver
* Rebuild all external applications that are using FGPU API.
