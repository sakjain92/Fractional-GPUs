# Porting application to use FGPU API

This file broadly explains the API of FGPU and showcase how to run an application.
Refer to *$PROJ_DIR/benchmarks/cudaSDK/vectorAdd/vectorAdd.cu* to see an example application.

## FGPU API

The following header files contains various FGPU API:
* *$PROJ_DIR/include/fractional_gpu.hpp*
* *$PROJ_DIR/include/fractional_gpu_cuda.cuh* <br/>
This file can only be included by .cu or .cuh files (i.e. files that will be compiled by nvcc and not gcc).
* *$PROJ_DIR/include/fractional_gpu_testing.hpp*

FGPU API can be divided into two parts:
* Host (CPU) side API <br/>
These APIs are called by CPU code. These API mainly initializes the FGPU, configure the parameter,
handles the memory coloring and allows parallel execution of application on GPU. 
All these API return an *int* type value. A value of '0' indicates success, otherwise an error occured.

* Device (GPU) side API <br/>
These APIs mostly macros that are used to modify CUDA kernels to add support for compute partitioning.

Note: FGPU currently only allows support for one GPU. Hence the first gpu (as reported by *nvidia-smi*) is
selected by FGPU.

Following are the list of major FGPU API that can be called by host (i.e. CPU) code:
* **fgpu_init** - This initializes the FGPU. This is the first API that needs to be called.
* **fgpu_deinit** - This function deinitializes the FGPU. This function is intended to be called when application is ending.
* **fgpu_set_color_prop** - This function sets up the partitioning property of application. It takes a 'color' and
'memory size' as input arguments. The color defines which partition should the application run in. The memory size
is only significant when memory coloring is enabled. It defines the amount of GPU memory that needs to be 
reserved for the application. Application cannot allocate more than this limit.
* **fgpu_memory_allocate** - This function should be used in lieu of *cudaMalloc()* or *cudaMallocManaged()* for allocating
memory on GPU. It allocated 'colored' GPU memory.
* **fgpu_memory_free** - This function is the counter-part of *fgpu_memory_allocate()*.
* **fgpu_memory_copy_async** - This function should be used for transfering data between CPU and GPU instead of 
*cudaMemcpy()* when dealing with 'colored memory'.
* **fgpu_memory_memset_async** - This function should be used for initializing GPU memory instead of 
*cudaMemset()* when dealing with 'colored memory'.
* **FGPU_LAUNCH_KERNEL** - This function should be used for launching CUDA kernels instead of CUDA provided primitives *<<<>>>*.
This macro takes care of launching CUDA kernels in a manner to facilitate compute partitioning.
* **fgpu_color_stream_synchronize** - The functions *fgpu_memory_copy_async(), fgpu_memory_memset_async() and FGPU_LAUNCH_KERNEL()* are all
asynchronous. To block till these functions are completed, *fgpu_color_stream_synchronize()* can be used. Each FGPU operation within an
application is carried out on same stream.

Instead of having to directly call the above functions, there are some wrappers for these function present in *$PROJ_DIR/include/fractional_gpu_testing.hpp*.

Following are the list of major FGPU API that can be called by device (i.e. GPU) code (specifically, these APIs are used to modify CUDA kernels):
* *FGPU_DEFINE_KERNEL* - All kernel definitions and declarations need to be replaced with this macro.
* *FGPU_DEVICE_INIT* - This macro needs to be the first this called in a CUDA kernel. This initialized metadata used by FGPU for compute partitioning.
* *FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx)* - All the CUDA kernel code needs to be placed within this loop macro. Also, instead of using CUDA provided *blockIdx* primitive, the variable that is given as input to this macro should be used for deriving index of block within the loop.
* *FGPU_GET_GRIDDIM* - Similar to *blockIdx*, CUDA provided *gridDim* primitive should not be used and instead the value returned by this macro should be used.

Note: As a TODO item, we wish to remove the need to modify the applications using compiler assisted code transformations.

## Building an application

To build an application, link it with *libfractional_gpu.so* library that is created in the build directory when FGPU is compiled. Also,
include the appropriate FGPU header files.

## Running an application

Prior to running an application that uses FGPU, Nvidia MPS and FGPU server needs to be running.
```
sudo $PROJ_DIR/scripts/mps_init.sh
cd $PROJ_DIR
./fgpu_server
```

To run an external application that is dynamically linked with *libfractional_gpu.so*, run the following command:
```
LD_PRELOAD=$PROJ_DIR/build/libfractional_gpu.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROJ_DIR/build ./\<app\>
```
assuming that *$PROJ_DIR* variable is appropriately set.

To end, stop the server and all the applications. After that, disable MPS.
```
sudo $PROJ_DIR/scripts/mps_stop.sh
```

## Caffe

We have ported Caffe to use FGPU API. We have setup an example image classification applicatication for Caffe.
To see how to run that application, please refer to *$PROJ_DIR/framework/caffe/NOTES.txt*. Some extra setup
has been done to ensure real-time performance. It is possible to add more application in Caffe and exploit the
benefits of FGPU.
