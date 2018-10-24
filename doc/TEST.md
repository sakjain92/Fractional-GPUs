# Testing and Benchmarks

## Testing

Various applications have been ported to use FPGU API. These are present in
*$PROJ_DIR/programs*. Some of the application within *$PROJ_DIR/programs* 
use user-space memory coloring, which is an obselete feature FGPU provided, 
and hence should not be taken as reference. For an example, refer to 
*$PROJ_DIR/benchmarks/cudaSDK/vectorAdd/vectorAdd.cu*. File *$PROJ_DIR/doc/PORT.md* 
explains how to port an application to use FGPU API.

The header file *$PROJ_DIR/include/fractional_gpu_testing.hpp* contains some helper functions that
are particularly useful for test/benchmark application.

Currently, there is no proper framework for testing. This is a TODO item.

### Adding test applications

To add a new test application to build system, refer to *$PROJ_DIR/CMakelists.txt*.

## Benchmarks

In micro-benchmark evaluation, we have taken codes from CUDA SDK and Rodinia suite and 
modified them to use FGPU API. These benchmarks can be found in *$PROJ_DIR/benchmarks*.
The methodology for benchmark is as follows:
1) Configure FGPU into a specific mode (Refer to *$PROJ_DIR/doc/BUILD.md* to see all modes that FGPU supports)
2) Run a single application on the first partition of GPU. All other partitions are empty. Measure time taken by
kernels of the application.
3) Run an application on the first partition of GPU. On all other partitions, run an interfering application (copies
of same application is ran across all the partitions). Measure
time taken by kernels of the application of interest.

Amount of variation between 2) and 3) cases highlight the amount of isolation between partitions. For more detailed
explanation of the methodology and results of micro-benchmarks, please refer to *$PROJ_DIR/doc/FGPU-RTAS-2019.pdf*.

### Configuring benchmarks

The file *$PROJ_DIR/benchmarks/config_benchmark.sh* allows to modify some paramters of benchmarks. Below we list the
paramters of importance:
* **benchmarks** - This array lists all the applications that we are interested in.
* **interference** - This array lists all the interfering application to be ran in parallel along the application of interest.
* **NUM_COLORS** - This variable defines how many partitions has the FGPU been split into. For now, it should be set to 2.
* **ENABLE_VOLTA_MPS_PARTITION** - On Volta architectures, MPS can be used to restrict amount of computation resources allocated to
an application. Enable this option when you wish to evaluate effectiveness of MPS on Volta GPU. The script uses MPS to split the 
computation resources equally between all partitions of GPU. Note: FGPU should be set in *No Partitioning*
mode when using this option as we rely on MPS to do the partition and not FGPU.

### Running benchmarks
After selecting the mode for FGPU and building the code (Refer to *$PROJ_DIR/doc/BUILD.md*),
run the following commands:

```
cd $PROJ_DIR/benchmarks
sudo ./run_benchmark.sh
```


Note: The benchmark script requires *schedtool* and *taskset* to be installed. They can be installed
on popular Linux distributions.
