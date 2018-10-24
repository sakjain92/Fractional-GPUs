# Fractional GPUs
Fractional GPUs (FGPU) framework allows system architects to split a single GP-GPU into
smaller independent partitions with each partition having performance isolation
from all the other partitions. This allows multiple applications to run in parallel with
strong isolation between each other.

Unlike Nvidia's MPS, we aim to provide strong isolation between partitions
to allow Fractional GPUs to be used in real-time systems such as autonomous vehicles
or to allow virtualization of GPU with strong run-time guarantees.

FGPU also have code to reverse engineer details about GPU memory hierarchy, specifically
details about structure of GPU L2 cache and DRAM. Our reverse engineering code can find
details such as L2 cache line size, number of cache sets, mapping from a physical address
to L2 cache set index, number of DRAM banks, number of DRAM chips/memory controllers, 
width of bus for a single DRAM chip and mapping from a physical address to DRAM bank index.
We find that the GPU memory hierarchy is sufficiently different from CPU memory hierarchy.

Currently, we support only Nvidia's GPU (discrete GPUs) and CUDA. Due to hardware limitations,
applications require modifications to source code to use this functionality
(we will mitigate this limitation in future via compiler aided source code transformation).

Interested readers can read our academic paper which is present at doc/FGPU-RTAS-2019.pdf
which gives high level details about functionality of FGPU.

## How it works
There are two components to providing isolation between partitions:

### Compute Isolation
Given two partitions (P1 and P2) on a single GPU and two
applications (A1 and A2), compute isolation is said to be achieved if the 
runtime of A1 remains unchanged in the following scenario:
1) A1 running on P1. P2 is idle.
2) A1 running on P1. A2 running on P2.

i.e. A2 has no effect on A1 (A2 doesn't steal A1's compute resources). While 
considering compute isolation, we do not consider the impact of A2’s
memory transactions (reads/writes) on A1’s run-time through any conflicts in the 
memory hierarchy (such as L2 cache conflicts). 

In our implementation, compute isolation is achieved by assigning non-overlapping sets
of SMs to P1 and P2 and forcing A1 to use only P1's SM and A2 to use only P2's
SM. For example, if a GPU has 10 SMs, we might assign 0-4 SMs to P1 and 5-10 SMs
to P2. When A1 runs, it will be forced to use only 0-4 SMs (A1 is allowed to 
under-utilize these SMs). Hence the maximum number of compute partitions is equal
to the total number of SMs (as SM is the smallest compute unit allocated to a
partition in this implementation).

### Memory Bandwith Isolation
Given two partitions (P1 and P2) on a single GPU, each having compute isolation,
and two applications (A1 and A2), memory bandwidth isolation is said to be 
achieved if the runtime of A1 remains unchanged
in the following scenario
1) A1 running on P1. P2 is idle.
2) A1 running on P1. A2 running on P2.

i.e. reads/writes issued by A2 have no slowdown on A1's read/writes (Without
memory bandwidth isolation, A2 reads/writes can slowdown A1's reads/writes
due to cache evictions in L2 cache, queuing delays in the memory controller and/or 
DRAM bandwidth saturation).

In our implementation, we use page coloring for this.
To use page coloring, memory hierarchy of GPU needs to be known. Since this is
not publicly released information for Nvidia's GPU, we have written code to reverse
engineer this information. We then used this information (along with modifying the
application's code and Nvidia's kernel driver) to exploit page coloring. 
We find the GPU's memory hierarchy is very well suited for page coloring and gives
near perfect memory bandwidth isolation. The maximum number of memory bandwidth 
partition is equal to the number of "colors" provided by page coloring. This is
dependent on the GPU's hardware.

### Combining Compute and Memory Bandwith Isolation
For the most part, the two ideas are orthogonal to each other. The number of 
compute partitions and memory bandwidth partitions do not need to be equal.

## Disclaimer
As the software stack by Nvidia is mostly closed source (CUDA driver/runtime 
userspace libraries are completely closed source. Nvidia's kernel driver is 
partly open source for some platforms), we had to employ various hacks/tricks to
get around this issue. This might make some part of the code tricky to understand.

## Languages Supported
* C++
* CUDA


## Frameworks supported
* Caffe


## Devices Supported
Currently, testing has been done on limited GPUs. We only support the following 
devices:
* Nvidia's GTX 1070
* Nvidia's GTX 1080
* Nvidia's Tesla V100

Though we are confident that it should be easy to add support for other Nvidia's
GPU. Please open an issue if you would like support for a specific Nvidia GPU.
We might require you to provide us with access to this GPU for testing.


## Status
Work is currently in progress. 

* **v0.1** has been released as a tag and is stable and working.
* **v0.2** has proper documentation. This tag was used for RTAS 2019 paper evaluations.


## Directory Hierarchy
* **benchmarks/**
    * Contains benchmarks used to evaluate FGPU performance.
* **doc/**
    * Contains various documents that describe various aspects of FGPU.
* **driver/**
    * Contains the modified nvidia driver to support memory coloring.
* **framework/**
    * Contains frameworks (such as Caffe) which have to ported to use FGPU API.
* **include/**
    * Contains files to be included by applications using FPGU.
* **persistent/**
    * Code for FGPU API.
* **programs/**
    * Contains example codes for testing/benchmarking.
    * Folders ending with 'persistent' have applications that use compute partitioning features.
    * Other folders are native code (not using partitioning feature).
* **reverse_engineering/**
    * Contains code used to reverse engineer the memory hierarchy of GPUs.
* **scripts/**
    * Contains basic utility scripts.


## Documentation

Apart from this README, the following text files exist to help new developers:

* **doc/BUILD.md**          
    * This file contains steps to setup/build/install FGPU.
* **doc/FAQ.md**
    * This file contains solutions to frequently encountered issues.
* **doc/FGPU-RTAS-2019.pdf** 
    * This file contains overall details about FGPU functionality and evaluation results.
* **doc/PORT.md**           
    * This file contains details about how to port applications to use FGPU and how to
    run applications that use FGPU. Including how to use Caffe that is ported to FGPU API.
* **doc/REVERSE.md**        
    * This file contains details about how to reverse engineer GPU memory hierarchy.
    * It also contains details about GPUs that have been reversed engineered.
* **doc/TEST.md**           
    * This file contains details about how to add new tests and how to run benchmarks.
* **doc/TODO.md**           
    * This file contains a wish-list of features/issues to be added/solved.
