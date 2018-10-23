# This file contains various configurable parameters for running benchmarks

# This script is used to collect statistics on benchmarks
PROJ_PATH=$PWD/../                  # Path to FGPU project directory
BIN_PATH=$PROJ_PATH/build           # Path to binaries (edit if different)
SCRIPTS_PATH=$PROJ_PATH/scripts     # Folder containing scripts
LOG_FILE=$PWD/log.txt               # File to log results

ENABLE_VOLTA_MPS_PARTITION="0"      # If 1, then we use Volta MPS for QoS

SERVER=fgpu_server                  # Server name

NUM_ITERATION=1000                  # Number of iterations of the benchmarks
MEMORY=1000000000                   # Amount of memory for each benchmark (About 1GB)
INTERFERENCE_COLOR=1                # Color on which to run interference app (starting color)
BENCHMARK_COLOR=0                   # Color on which to run benchmarks
INTERFERENCE_PRIO=1                 # Real time priority of interference app
BENCHMARK_PRIO=99                   # Real time priority of benchmark app
NUM_COLORS=2		    	    # Number of interferenceing applications + 1

# List of benchmarks to run
benchmarks=(
"cudaSDK_mms"       # Matrix Multiplication
"cudaSDK_sn"        # Sorting Networks
"cudaSDK_va"        # Vector Addition
"cudaSDK_sp"        # Scalar Product
"cudaSDK_fwt"       # Fast Walsh Transform
"rodinia_cfd"       # Computational Fluid Dynamics
"rodinia_nn"        # Nearest neighbours
"rodinia_gaussian"  # Gaussian Elimination
)

# List of interference applications that run in background
interference=(
"cudaSDK_mms"       # Computational Intensive
"cudaSDK_fwt"       # Shared memory and memory intensive
"cudaSDK_va"        # Memory Intensive
)
