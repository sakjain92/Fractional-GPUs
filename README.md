# Fractional GPUs
Fractional GPUs framework allows system architects to split a single GPU into
smaller independent partitions, each completely isolated from the other. 

Currently we support only Nvidia's GPU and CUDA. Due to hardware limitations,
applications require modifications to use this functionality. Only one GPU is
supported currently.

## Lanuguages Supported
* C++
* CUDA


## Pre-requisites
* CUDA runtime library
* CUDA driver library
* nvcc
* Nvidia driver
* Linux
* Nvidia MPS
* cmake
* make

## Devices Supported
Currently testing has been done on limited GPUs. We only support the following 
devices:
* Nvidia's GTX 1070

## Directory Hierarchy
* include/
    * Contains files to be included
    * External headers files:
        * fractional_gpu_cuda.cuh
        * fractional_gpu.h
* persistent/
    * Code for persistent API
* programs/
    * Contains example codes for testing/benchmarking
    * Folders ending with 'persistent' have application that use compute partitioning features
    * Other folders are native code (not using paritioning feature)
* scripts/
    * Contains basic scripts to start/stop/kill MPS server

## Build Steps
The project uses cmake, so compilation is simple.

```
cd $PROJECT_DIR
mkdir build
cd build/
cmake ..
make
```

## Running Application
The paritioning feature requires the following:
* Partition server
* Nvidia MPS

Steps to run an application are:
```
sudo nvidia-smi --persistence-mode=1    // Optional : Useful for benchmarking. Tries to limit dynamic scaling of GPU
cd $PROJECT_DIR
mkdir build
cd build
make
./server                        // Run the server. Server does some initialization and then goes to sleep.
sudo ../scripts/mps_init.sh     // Starts the MPS server
./<application>
```

In case the server is not properly closed, it might complain the next time it is 
ran. Running the server twice should make it work.

Steps to reset the system state:
```
cd $PROJECT_DIR
sudo scripts/mps_stop.sh
```

## Add new applications
Following steps need to be taken to add support for partitioning in new
applications. (All applications should be added under programs/ directory.
Additionally, application which are part of cuda samples (distributed by Nvidia)
should be kept under programs/cuda_samples/)

### Creation
* Make directory under programs/ or programs/cuda_samples/
* Add all code

### Header Files
* fractional_gpu.h needs to be included in any file calling host side API
    * E.g. fgpu_init()/fgpu_deinit()/FGPU_LAUNCH_KERNEL()
* fractional_gpu_cuda.cuh needs to be included in any file contains device code/
uses device side API
    * E.g. FGPU_DEVICE_INIT()/FGPU_FOR_EACH_DEVICE_BLOCK()/FGPU_FOR_EACH_END
* For cuda sample applications, all the common header files are kept at 
programs/cuda_samples/common/inc/. These header files are included during
compilation steps.

### Code Modification
As the hardware doesn't expose all the neccesary functionality, a software
mechanism is used to provide these functionalities instead. To do so, it is required
to modify the applications to replace certain CUDA APIs with Fractional GPU API.
For reference and to compare CUDA API v.s. Fractional GPU API, please see 
programs/cuda_samples/matrixMul/ and programs/cuda_samples/matrixMul_persistent/.
The major API are as follows:

#### Host Side APIs
* int fgpu_init(): This is the first function that needs to be called before any other
API is used. It does some initialization. It returns < 0 on error, else 0 on success.

* FGPU_LAUNCH_KERNEL(): CUDA kernels shouldn't be launched using <<<>>> operation
provided by CUDA. Instead, this macro should be used (FGPU_LAUNCH_VOID_KERNEL macro 
should be used if the cuda kernel takes no arguments). Currently all the kernel launches 
are blocking. We plan to support non-blocking kernel launches.The arguments of this macro
are as follows:

    * Color : Which color to run on paritioned GPU. Kernels can run in parallel if 
launched in seperated colors. Currently we support only 2 colors, 0 and 1. This 
argument is of type int.

    * gridDim : This is a uint3 type argument which defines the grid dimensions of 
the kernel.

    * blockDim: This is a uint3 type argument which defines the block dimensions of
the kernel.

    * sharedMem: This defines the amount of shared memory to be used by kernel that
is not explicitly allocated within CUDA kernel. For most kernels, this might be
left as 0.

    * func : This is the kernel function to be launched. All the arguments of the
    macro after this argument are the arguments that need to be passed to be kernel.

* int fgpu_color_stream_synchronize(color): This function waits for all the prior
kernel launched by the application within the specified color to complete.
This function returns < 0 on error else 0 on success.

* void fgpu_deinit(): This is the last function to be called. This does cleanups.
#### Device Side APIs

* FGPU_DEFINE_KERNEL() : This macro is to be used while defining or declaring
a global CUDA kernel.

* FGPU_DEVICE_INIT() : This needs to be the first function to be called in the
cuda kernel. It optionally returs the grid dimensions. The CUDA provided gridDim
intrinsics __shouldn't__ be used.

* FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx)/FGPU_FOR_EACH_END : This is a for-loop,
within which the main body of CUDA kernel should reside. This macro returns block
index (of type uint3) to run, one at a time, till no more block index are left.
The CUDA provided blockIdx __shouldn't__ be used.

### Adding application to build steps
There are two methods to build an application

#### Using shared library (currently untested)
After the build steps are completed, a shared library (libfractional_gpu.so) is
created in the build directory. The application can be linked with this library.
Remember to include the 'include/' directory in the include path while building
the application to include the header files.

#### Static Compilation
Applications can be added to CMakeList.txt to be included in build steps.
CMakeList.txt is self explanatory. add_native_target() function should be used
for application that don't use partitioning feature and add_persistent_target()
function should be used for application that do use the feature. Both these
functions expect all the include files of the application to be contained within
one single directory.
