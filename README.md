# Fractional GPUs
Fractional GPUs (FGPU) framework allows system architects to split a single GP-GPU into
smaller independent partitions with each partition having performance isolation
from all the other partitions. This allows multiple applications to run in parallel with
strong isolation between each other.

Unlike Nvidia's MPS, we aim to provide strong isolation between partitions
to allow Fractional GPUs to be used in real-time systems such as autonomous vehicles
or to allow virtualization of GPU with strong run-time guarantees.

FGPU also have code to reverse engineer details about GPU memory hierarchy, specifically
details about the structure of GPU L2 cache and DRAM. Our reverse engineering code can find
details such as L2 cache line size, number of cache sets, mapping from a physical address
to L2 cache set index, number of DRAM banks, number of DRAM chips/memory controllers, 
the width of the bus for a single DRAM chip and mapping from a physical address to DRAM bank index.
We find that the GPU memory hierarchy is sufficiently different from the CPU memory hierarchy.

Currently, we support only Nvidia's GPU (discrete GPUs) and CUDA. Due to hardware limitations,
applications require modifications to source code to use this functionality
(we will mitigate this limitation in future via compiler aided source code transformation).

Interested readers can read our academic paper which is present at doc/FGPU-RTAS-2019.pdf
which gives high-level details about the functionality of FGPU.

## How it works
There are two components to providing isolation between partitions:

### Compute Isolation
Given two partitions (P1 and P2) on a single GPU and two
applications (A1 and A2), compute isolation is said to be achieved if the 
runtime of A1 remains unchanged in the following scenario:
1) A1 running on P1. P2 is idle.
2) A1 running on P1. A2 running on P2.

i.e. A2 has no effect on A1 (A2 doesn't steal A1's compute resources). While considering 
compute isolation, we do not consider the impact of A2’s
memory transactions (reads/writes) on A1’s run-time through any conflicts in the 
memory hierarchy (such as L2 cache conflicts). 

In our implementation, compute isolation is achieved by assigning non-overlapping sets
of SMs to P1 and P2 and forcing A1 to use only P1's SM and A2 to use only P2's
SM. For example, if a GPU has 10 SMs, we might assign 0-4 SMs to P1 and 5-9 SMs
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
We find that the GPU's memory hierarchy is very well suited for page coloring and gives
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
* **[benchmarks/](benchmarks/)**
    * Contains benchmarks used to evaluate FGPU performance.
* **[doc/](doc/)**
    * Contains various documents that describe various aspects of FGPU.
* **[driver/](driver/)**
    * Contains the modified nvidia driver to support memory coloring.
* **[framework/](framework/)**
    * Contains frameworks (such as Caffe) which have to ported to use FGPU API.
* **[include/](include/)**
    * Contains files to be included by applications using FPGU.
* **[persistent/](persistent/)**
    * Code for FGPU API.
* **[programs/](programs/)**
    * *Update:* Do not refer to examples in this folder as they use some of the deprecated
      APIs (which makes it harder to understand the code even though the code is fully functional). 
      Please instead refer to benchmark folder.
    * Contains example codes for testing/benchmarking.
    * Folders ending with 'persistent' have applications that use partitioning features.
    * Other folders are native code (not using the partitioning feature).
* **[reverse_engineering/](reverse_engineering/)**
    * Contains code used to reverse engineer the memory hierarchy of GPUs.
* **[scripts/](scripts/)**
    * Contains basic utility scripts.


## Documentation

Apart from this README, the following text files exist to help new developers:

* **[doc/BUILD.md](doc/BUILD.md)**  
    * This file contains steps to setup/build/install FGPU.
* **[doc/FAQ.md](doc/FAQ.md)**
    * This file contains solutions to frequently encountered issues.
* **[doc/FGPU-RTAS-2019.pdf](doc/FGPU-RTAS-2019.pdf)** 
    * This file contains overall details about FGPU functionality and evaluation results.
* **[doc/PORT.md](doc/PORT.md)**           
    * This file contains details about how to port applications to use FGPU and how to
    run applications that use FGPU. Including how to use Caffe that is ported to FGPU API.
* **[doc/REVERSE.md](doc/REVERSE.md)**        
    * This file contains details about how to reverse engineer GPU memory hierarchy.
    * It also contains details about GPUs that have been reversed engineered.
* **[doc/TEST.md](doc/TEST.md)**           
    * This file contains details about how to add new tests and how to run benchmarks.
* **[doc/TODO.md](doc/TODO.md)**           
    * This file contains a wish-list of features/issues to be added/solved.

## Help! Too many document and I am impatient!

For a quick demo, first setup your system according to *SETUP* section of 
[doc/BUILD.md](doc/BUILD.md#setup) (also ensure that you have one of the supported
GPUs in your system. The first GPU in the system is the GPU used by FGPU by default)

Then, follow these steps:
```
cd $PROJ_DIR/scripts
./evaluations # Follow the steps shown on screen. 
```

[scripts/evaluation.sh](scripts/evlaution.sh) is a demo script that allows to
quickly reverse engineer GPUs and evaluate existing benchmarks without needing
to understand the details of compilation and launch steps associated with FGPU
(the script handles these steps). This script re-runs all experiments that were
done for the FGPU academic paper. Following is a brief explaination of what this 
script does (Read  [doc/FGPU-RTAS-2019.pdf](doc/FGPU-RTAS-2019.pdf) before proceeding)

FGPU paper can be broken down into 3 chiefs parts: 
1) Reverse engineering of GPU
2) Micro-benchmark evaluation on CUDA/Rodinia applications
3) Macro-benchmark evaluation on Caffe application

This script allows user to re-run all of the above experiments (it has been 
tested on GTX 1080 and Tesla V100 GPUs). Apart from these, it also allows
functional testing of FGPU. The script initially asks user to chose one of these
options.

### Reverse Engineering of GPU

If user chooses this option, the script executes reverse engineering code
([reverse_engineering/](reverse_engineering/)). 
1) First, access time of multiple pairs 
of physical address are collected (Algorithm 1 in FGPU paper). Using this data, the
script generates trendline of access times and histogram of same
(Fig 3 and 4 in FGPU paper). 
2) Then, with this information, hash function for DRAM banks is reversed engineered. 
3) Next, similar approach is taken to reverse engineer hash function for cacheline. 
4) Finally, using data about these hash functions, experiments are conducted to see 
which page coloring scheme gives least interference (Fig. 8 of FGPU paper)

The scripts ends up generating plots for Fig. 3/4/8 of FGPU paper.

### Micro-benchmark evalution on CUDA/Rodinia applications
If user chooses this option, the script executes benchmarks available in 
[benchmarks/](benchmarks). There are 3 types of modes in which FGPU can operate
1) Compute Partitioning Only (CP)
2) Compute and Mammory Partitioning (CMP)
3) Volta MPS based Compute Partitioning only (MPS)

(The last option is based on Nvidia's MPS and is only for comparision with isolation
provided by FGPU. FGPU plays no role in this mode).

User is prompted by the script to chose one of these modes. Using this mode, the GPU 
is split into multiple partitions. The user is also asked for how many partitions
does the user want the GPU to be split into (the number of available partitions is 
hardware dependent).

To see amount of isolation FGPU provides, each of these benchmark is ran in parallel alongside with 
interfering applications (these interfering applications are themself subsets of the 
benchmark applications). Hence, on one partition of the GPU, one of the benchmark runs and on other
partitions other interfering applications run. The runtime of benchmark application
is measured across different interfering applications. For perfect isolation, this runtime
should be independent of the interfering application. This is repeated for all different benchmarks.

To normalize these runtimes, as a baseline, each benchmark also runs on the whole GPU without any interfering application.
These normalized runtimes are then ploted in a graph for the specific FGPU mode by the script. Fig 9a/9b/11
in the FGPU paper are basically merge of two or more of these plots (E.g Fig 9a is combination of 
CP and CMP)

### Macro-benchmark evaluation on Caffe applications
This option is similar as the above option. The only difference is that instead of using CUDA/Rodinia
applications, we use a ML (image classification) application that uses FGPU ported Caffe.

Some of the data file required by the [example](http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html)
image classification application needs to be downloaded before running in this mode.

```
cd $PROJ_DIR/framework/caffe/
./scripts/download_model_binary.py models/bvlc_reference_caffenet # Download the model
./data/ilsvrc12/get_ilsvrc_aux.s # Download example data
```

### Testing
In this mode, the script uses selected CUDA benchmark applications
([benchmarks/cudaSDK](benchmarks/cudaSDK)) to test FGPU functionally.
To do these, we compare the output of these applications
(when running with/without interfering applications) with known expected output.
For example [benchmarks/cudaSDK/vectorAdd/](benchmarks/cudaSDK/vectorAdd/) adds two
vectors together on GPU. Hence we know the epected output based on inputs which we compare
with the runtime output of GPU implementation of vectorAdd.
