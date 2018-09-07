# Fractional GPUs
Fractional GPUs framework allows system architects to split a single GPU into
smaller independent partitions with each partition having performance isolation
from all the other partitions.

Unlike Nvidia's MPS, we aim to provide strong isolation between partitions
to allow Fractional GPUs to be used in real-time systems such as autonomous vehicles.

Currently, we support only Nvidia's GPU and CUDA. Due to hardware limitations,
applications require modifications to source code to use this functionality
(we hope that this can be mitigated in future via compiler aided source code
transformation).

## How it works
There are two components to providing isolation between partitions:

### Compute Isolation
Given two partitions (P1 and P2) on a single GPU and two pure compute-bound 
applications (A1 and A2) (such as pseudo-random number generations), compute
isolation is said to be achieved if the runtime of A1 remains unchanged in the 
following scenario:
1) A1 running on P1. P2 is idle.
2) A1 running on P1. A2 running on P2.

i.e. A2 has no effect on A1 (A2 doesn't steal A1's compute resources). In our
implementation, compute isolation is achieved by assigning non-overlapping sets
of SMs to P1 and P2 and forcing A1 to use only P1's SM and A2 to use only P2's
SM. For example, if a GPU has 10 SMs, we might assign 0-4 SMs to P1 and 5-10 SMs
to P2. When A1 runs, it will be forced to use only 0-4 SMs (A1 is allowed to 
under-utilize these SMs). Hence the maximum number of compute partitions is equal
to the total number of SMs (as SM is the smallest compute unit allocated to a
partition in this implementation).

### Memory Bandwith Isolation
Given two partitions (P1 and P2) on a single GPU, each having compute isolation,
and two memory-bound applications (A1 and A2) (such as vector addition), memory
bandwidth isolation is said to be achieved if the runtime of A1 remains unchanged
in the following scenario
1) A1 running on P1. P2 is idle.
2) A1 running on P1. A2 running on P2.

i.e. reads/writes issued by A2 have no slowdown on A1's read/writes (Without
memory bandwidth isolation, A2 reads/writes can slowdown A1's reads/writes
due to cache evictions in L2 cache, queuing delays in the memory controller and/or 
DRAM bandwidth saturation). In our implementation, we use page coloring for this.
To use page coloring, memory hierarchy of GPU needs to be known. Since this is
not publicly released information for Nvidia's GPU, we have written code to reverse
engineer this information. We then used this information (along with modifying the
application's code and Nvidia's kernel driver) to exploit page coloring. 
We find the GPU's memory hierarchy is very well suited for page coloring and gives
near perfect memory bandwidth isolation. The maximum number of memory bandwidth 
partition is equal to the number of "colors" provided by page coloring. This is
dependent on the GPU's hardware.

### Combining Compute and Memory Bandwith Isolation
For the most part, the two ideas are orthogonal to each other.

### Disclaimer
As the software stack by Nvidia is mostly closed source (CUDA driver/runtime 
userspace libraries are completely closed source. Nvidia's kernel driver is 
partly open source for some platforms), we had to employ various hacks/tricks to
get around this issue which might make the code difficult to understand.

## Languages Supported
* C++
* CUDA

## Frameworks supported
* Caffe (Work in progress)

## Pre-requisites
* CUDA runtime library
* CUDA driver library
* nvcc
* Linux
* Nvidia MPS
* cmake
* make

## Devices Supported
Currently, testing has been done on limited GPUs. We only support the following 
devices:
* Nvidia's GTX 1070

## Status
This work is currently in progress and version 1.0 is yet to be released.

(Note: The rest of the document below is out-of-date due to recent changes in the code
base)

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
    * Folders ending with 'persistent' have applications that use compute partitioning features
    * Other folders are native code (not using partitioning feature)
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
The partitioning feature requires the following:
* Partition server
* Nvidia MPS

Steps to run an application are:
```
sudo nvidia-smi --persistence-mode=1    // Optional : Useful for benchmarking. Tries to limit dynamic scaling of GPU
cd $PROJECT_DIR
mkdir build
cd build
make
./server                        // Run the server. The server does some initialization and then goes to sleep.
sudo ../scripts/mps_init.sh     // Starts the MPS server
<run application>
```

In case the server is not properly closed, it might complain the next time it
runs. Running the server twice should make it work.

Steps to reset the system state:
```
cd $PROJECT_DIR
sudo scripts/mps_stop.sh
```

## Port new applications
Following steps need to be taken to add support for partitioning in new
applications.

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
As the hardware doesn't expose all the necessary functionality, a software
mechanism is used to provide these functionalities instead. To do so, it is required
to modify the applications to replace certain CUDA APIs with Fractional GPU API.
For reference and to compare CUDA API v.s. Fractional GPU API, please see 
programs/cuda_samples/matrixMul/ and programs/cuda_samples/matrixMul_persistent/.
The major API are as follows:

#### Host Side APIs
* int fgpu_init(): This is the first function that needs to be called before any other
API is used. It does some initialization. It returns < 0 on error, else 0 on success.

* FGPU_LAUNCH_KERNEL(): CUDA kernels shouldn't be launched using <<<>>> operation
provided by CUDA. Instead, this macro should be used (FGPU_LAUNCH_VOID_KERNEL macro should be used if the cuda kernel takes no arguments). Currently, all the kernel launches are blocking. We plan to support non-blocking kernel launches. The arguments of this macro
are as follows:

    * Color: Which color to run on partitioned GPU. Kernels can run in parallel if launched in separate colors. Currently, we support only 2 colors, 0 and 1. This argument is of type int.

    * gridDim: This is a uint3 type argument which defines the grid dimensions of 
the kernel.

    * blockDim: This is a uint3 type argument which defines the block dimensions of
the kernel.

    * sharedMem: This defines the amount of shared memory to be used by the kernel that
is not explicitly allocated within CUDA kernel. For most kernels, this might be
left as 0.

    * func : This is the kernel function to be launched. All the arguments of the
    macro after this argument are the arguments that need to be passed to be the kernel.

* int fgpu_color_stream_synchronize(color): This function waits for all the prior
kernel launched by the application within the specified color to complete.
This function returns < 0 on error else 0 on success.

* void fgpu_deinit(): This is the last function to be called. This does cleanups.
#### Device Side APIs

* FGPU_DEFINE_KERNEL(): This macro is to be used while defining or declaring
a global CUDA kernel.

* FGPU_DEVICE_INIT() : This needs to be the first function to be called in the
cuda kernel. It optionally returns the grid dimensions. The CUDA provided gridDim
intrinsics __shouldn't__ be used.

* FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) /FGPU_FOR_EACH_END: This is a for-loop,
within which the main body of CUDA kernel should reside. This macro returns block
index (of type uint3) to run, one at a time, till no more block index are left.
The CUDA provided blockIdx __shouldn't__ be used.

### Adding application to build steps
There are two methods to build an application (you can choose any one of them)

#### Using shared library (currently untested)
After the build steps are completed, a shared library (libfractional_gpu.so) is
created in the build directory. The application can be linked with this library.
Remember to include the 'include/' directory in the include path while building
the application to include the header files.

#### Static Compilation
Stati compilation can be used for small applications that don't have their own
Makefiles. Applications can be added to CMakeList.txt to be included in build steps.
CMakeList.txt is self-explanatory. add_native_target() function should be used
for an application that doesn't use partitioning feature and add_persistent_target()
function should be used for an application that does use the feature. Both these
functions expect all the include files of the application to be contained within
one single directory.

