# Reverse Engineering GPU L2 cache/DRAM structure

## Running reverse engineering code

We suspect that the memory hierarchy is same for all the GPUs of the same
architecture family. We base this suspicion on the fact that both GTX 1070 and GTX 1080
have the exact same memory hierarchy (based on our reverse engineering) and both belong to
Pascal architecture. We have already reversed engineering Pascal and Volta architectures,
details of which are provided below.

To reverse engineer details about other architectures, it is required to run reverse engineering code.
For this, FGPU code needs to be compiled in reverse engineering mode (See *$PROJ_DIR/doc/BUILD.md*).

When reverse engineering mode is enable and FGPU is compiled, reverse engineering 
code is builded in form of a binary *gpu_reverse_engineering*, present in the build
directory. Running it provides with the details of L2 cache and DRAM structure of the current GPU. 
Refer to *$PROJ_DIR/doc/PORT.md* on how to run FGPU applications.

Also, some support needs to be added for that specific architecture in the device driver. Please refer to the following
files to see what changes are required:
* *$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_pascal.c*
* *$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_volta.c*

Similar changes are required to be made in the file 
*$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_\<arch\>.c*

After reverse engineering details about a GPU architecture, support for memory coloring can be added for that
architecture. We have alreeady added support for memory coloring on Pascal and Volta architecture. Please
refer to the following files:
* *$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_pascal_mmu.c*
* *$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_volta_mmu.c*

To add memory coloring in any other architecture, similar changes need to be made in the file
*$PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/kernel/nvidia-uvm/uvm8_\<arch\>_mmu.c*

Note: Currently even though Volta has capability to have 8 memory colors, we have only added support for 2 memory
colors. This can be easily changed.

For more details about GPU memory hierarchy, please refer to *$PROJ_DIR/doc/FGPU-RTAS-2019.pdf*.

## Reversed Engineered GPUs

For the following sections, we define two functions:
* **X**
    * <a><img src="https://latex.codecogs.com/gif.latex?X(addr, x_1, x_2 .., x_n) = \bigoplus_{i=1}^n{addr[x_i]}" /></a>
    * where <a><img src="https://latex.codecogs.com/gif.latex?$addr[x_i]$"/></a> refers to <a><img src="https://latex.codecogs.com/gif.latex?x_{i}^{th}"/></a> bit of <a><img src="https://latex.codecogs.com/gif.latex?addr"/></a>.
    * i.e. X takes a physical address and n bit indices and returns XOR sum over all those bits of the address.
    * E.g. X(0x13, 0, 4) = 0 and X(0x13, 0, 1, 4) = 1.

* **C**
    * <a><img src="https://latex.codecogs.com/gif.latex?C(v_1, v_2, ..., v_n) = v_n\| v_{n-1}\|\dotsb\| v_1" /></a>
    * i.e. C concatenates bits together.
    * E.g. C(1, 0, 0, 0, 1) = 0x11 and C(1, 1, 0, 0, 1) = 0x13.

The following GPUs have been reversed engineered. 

### GeForce GTX 1070

#### Physical Address to DRAM Bank Index Mapping function
bbit0 = X(addr, 10, 12, 16, 20, 23, 26, 29, 30)

bbit1 = X(addr, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30)

bbit2 = X(addr, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31)

bbit3 = X(addr, 13, 15, 20, 24, 26, 29, 32)

bbit4 = X(addr, 15, 16, 21, 22, 23, 25, 26, 28, 29)

bbit5 = X(addr, 16, 19, 23, 27, 30)

bbit6 = X(addr, 17, 20, 22, 23, 24, 27, 28, 29, 31)

BankIndex = C(bbit0, bbit1, bbit2, bbit3, bbit4, bbit5, bbit6)

#### Physical Address to Cache Set Index Mapping function

cbit0 = X(addr, 10, 12, 16, 20, 23, 26, 29, 30)

cbit1 = X(addr, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30)

cbit2 = X(addr, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31)

cbit3 = X(addr, 7, 8, 16, 17, 23, 26, 31)

cbit4 = X(addr, 8, 10, 12, 16, 17, 21, 24, 25, 26, 27)

cbit5 = X(addr, 9, 10, 18, 25, 29, 30, 31)

cbit6 = X(addr, 13, 14, 20, 23, 28, 29, 30)

cbit7 = X(addr, 14, 15, 17, 20, 21, 23, 24, 28, 31)

cbit8 = X(addr, 15, 16, 19, 20, 23, 24, 25, 26, 28, 29, 30, 32)

cbit9 = X(addr, 16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 30)

CacheSetIndex = C(cbit0, cbit1, cbit2, cbit3, cbit4, cbit5, cbit6, cbit7, cbit8, cbit9)

#### Physical Address to Memory Module Index Mapping function
mbit0 = X(addr, 10, 12, 16, 20, 23, 26, 29, 30)

mbit1 = X(addr, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30)

mbit2 = X(addr, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31)

MemoryModuleIndex = C(mbit0, mbit1, mbit2)

#### Miscellaneous Details
* Architecture - Pascal
* Number of SMs - 15
* Max number of Compute Partitions - 15
* Number of DRAM banks - 128
* Number of Cache sets - 1024
* Number of Memory Modules - 8
* Max number of Memory Bandwidth Partitions - 2
* Cache-line size - 128 bytes
* Cache set associativity - 16
* Page Sizes - 4 KB/64 KB/2 MB. Default is 2 MB.

### GeForce GTX 1080

Exactly same as GeForce GTX 1070 (as both have same architecture, Pascal) except
the number of SMs is 20 and correspondingly the maximum number of compute partitions
is 20.

### Tesla V100-SXM2-16GB
#### Physical Address to DRAM Bank Index Mapping function
bbit0 = X(addr, 10, 11, 20, 23, 25, 28, 30, 33)

bbit1 = X(addr, 11, 12, 16, 20, 25, 26, 29, 30, 32, 33)

bbit2 = X(addr, 12, 16, 17, 19, 23, 25, 26, 27, 31)

bbit3 = X(addr, 13, 24, 26, 27, 28, 30, 31, 33)

bbit4 = X(addr, 15, 17, 19, 20, 27, 28, 30, 31, 32)

bbit5 = X(addr, 16, 19, 20, 23, 27, 29, 31, 33)

bbit6 = X(addr, 17, 18, 23, 24, 25, 27, 28, 30, 31)

bbit7 = X(addr, 18, 21, 25, 29, 32)

bbit8 = X(addr, 19, 20, 22, 24, 25, 26, 29, 30, 31, 33)

BankIndex = C(bbit0, bbit1, bbit2, bbit3, bbit4, bbit5, bbit6, bbit7, bbit8)

#### Physical Address to Cache Set Index Mapping function

cbit0 = X(addr, 10, 17, 20, 22, 24, 26, 27, 28, 30, 32, 33)

cbit1 = X(addr, 11, 12, 18, 24, 26, 28, 29, 30, 31, 32)

cbit2 = X(addr, 12, 13, 22, 26, 27, 28, 29, 30, 31, 33)

cbit3 = X(addr, 13, 14, 22, 24, 30, 31, 32, 33)

cbit4 = X(addr, 15, 18, 22, 23, 26, 29, 30, 31, 32, 33)

cbit5 = X(addr, 7, 15, 21, 23, 24, 25, 28, 32)

cbit6 = X(addr, 8, 9, 15, 19, 21, 23, 24, 25, 28, 29, 31)

cbit7 = X(addr, 9, 15, 19, 21, 22, 24, 25, 26, 27, 30, 31, 32, 33)

cbit8 = X(addr, 14, 19, 20, 24, 25, 27, 28, 29, 30, 31, 32, 33)

cbit9 = X(addr, 16, 19, 21, 22, 25, 26, 28, 30, 32)

CacheSetIndex = C(cbit0, cbit1, cbit2, cbit3, cbit4, cbit5, cbit6, cbit7, cbit8, cbit9)

#### Physical Address to Memory Module Index Mapping function

mbit0 = X(addr, 10, 13, 17, 19, 24, 25, 26, 29, 30, 32, 33)

mbit1 = X(addr, 11, 13, 15, 23, 24, 26, 27, 29, 30, 31)

mbit2 = X(addr, 12, 15, 16, 18, 20, 21, 23, 26, 28, 29, 30)

mbit3 = X(addr, 13, 19, 20, 22, 25, 27, 28, 29)

mbit4 = X(addr, 15, 18, 22, 23, 26, 29, 30, 31, 32, 33)

MemoryModuleIndex = C(mbit0, mbit1, mbit2. mbit3, mbit4)

#### Miscellaneous Details
* Architecture - Volta
* Number of SMs - 80
* Max number of Compute Partitions - 80
* Number of DRAM banks - 512
* Number of Cache sets - 1024
* Number of Memory Modules - 32
* Max number of Memory Bandwidth Partitions - 8
* Cache-line size - 128 bytes (Yet to be confirmed)
* Cache Set Associativity - 3 (Yet to be confirmed. There is a factor of 16 missing somewhere).
* Page Sizes - 4 KB/64 KB/2 MB. Default is 2 MB.
