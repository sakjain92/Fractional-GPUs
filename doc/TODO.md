# TODO List

This file lists all the broad high-level TODO items. They are broken into sub-categories
* **Critical** - These features are neccesary and of high importance. 
* **Important** - These features will make it easier to adopt FGPU or increase its utility.
* **Nice to have** - These features should be tackled if sufficient time is available.
* **Minor** - These features should not take much time to implement but are low on priority.

## Critical
1) Find reasons for high overheads on small kernels (such as SortingNetworks/NN).
2) Find out cache-line size of Tesla V100.
3) Add compiler support for compiler assisted code transformation to avoid manual kernel modifications. This will allow for virtualization of GPUs. Figure out how a single GPU is currently virtualized.
Allows porting Caffe without the need to modify the source code (Caffe can use the closed source cuBLAS library).
4) Confirm that compute partitioning is functionally correct (as there is no therotical guarantee that it will always work). If true, remove optional checks that exists currently in the code to ensure code runs functionally correct.
5) Page color the metadata used by device driver also (such as page tables).
6) Add proper licence for code distribution.
7) Conduct case study to showcase FGPU can be used to improve throughput as compared to running back-to-back by exploiting the under-uilization of GPU by most applications.
8) Understand the reason for high tail latency for kernel exeuction on FGPU.
9) FGPU uses cudaMallocManaged() (as this is the API for which we have open source code in device driver). This API has a limitation that it blocks kernel execution and memory transfers from happening in parallel (just by invoking a single instance of this API changes the whole behaviour of cuda library). Need to figure out how to fool CUDA library into thinking this function was never called but still be able to call it.
10) Promote FGPU on sites such as kubernetes in which users have explicitly opened up issues regarding need for fractional GPUs.
11) Caffe test failing with FGPU: GPUStochasticPoolingLayerTest 

## Important
1) Eliminate requirement for MPS - Allows use of FGPU on embedded systems that lack MPS support from NVIDIA.
Also, MPS in pre-Volta architecute causes breakdown of functional isolation between different application wrt to GPU memory.
2) Reverse engineer memory hierarchy for embedded systems.
3) Add support for multiple GPUs.
4) Confirm that use of 4 KB pages have limited impact on performance as compared to use of 2 MB pages. If true, remove upport for userspace coloring.
5) Reverse engineer memory hierarchy of other GPUs.
6) Give more control to user on how to partition the SMs (currently all the partitions receive equal amount of SMs).
7) Fix the bugs in device driver. Currently, some of the bugs are lying in the modified device driver. Hacks have been placed to bypass them currently. Need to fix these issues.
8) Add support for a single application to use multiple colors - Helps in case hardware supports multiple colors.
9) Add support for having more compute partitions that memory partitions.
10) Find all the reasons for NVIDIA driver not being real-time (such as spawining background kernel threads).
11) Add testing framework to test functionality (apart from benchmark framework)

## Nice to have
1) Find out how to disable power mangement on discrete GPUs to avoid variablility in timing during benchmarks.
2) Make it easier to port changes in device driver from one version to another (possibly in form of a patch).
Currently we are forced to use a single device driver with the modified code hence limiting FGPU to CUDA 9.1.
3) Add C API (instead of just C++).
4) Explore virtualization of graphics pipeline also. 
5) Detailed commented code.
6) Add suport for multiple CUDA streams.

## Minor
1) Compile programs based on the generation of GPU (i.e. compute_arch or sm_arch compiler options passed to nvcc).
2) Add better error code handling.
3) Replace FGPU API with API with better names for functions/macros.
