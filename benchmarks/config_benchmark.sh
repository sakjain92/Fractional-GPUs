#!/bin/bash

# This file contains various configurable parameters for running benchmarks

#Below are default parameters that can either be changed via changing this script
# or suuplying appropriate arguments

COMMON_SCRIPT=../scripts/common.sh

if [ ! -f $COMMON_SCRIPT ]; then
    echo "Run this script from \$PROJ_DIR/benchmark folder"
fi

source $COMMON_SCRIPT

# This script is used to collect statistics on benchmarks
LOG_FILE=$PWD/log.txt               # File to log results

ENABLE_VOLTA_MPS_PARTITION="0"      # If 1, then we use Volta MPS for QoS

NUM_ITERATION=1000                  # Number of iterations of the benchmarks
MEMORY=2000000000                   # Amount of memory for each benchmark (About 2GB)
INTERFERENCE_COLOR=1                # Color on which to run interference app (starting color)
BENCHMARK_COLOR=0                   # Color on which to run benchmarks
INTERFERENCE_PRIO=1                 # Real time priority of interference app
BENCHMARK_PRIO=99                   # Real time priority of benchmark app
NUM_COLORS=2		    	        # Number of interferenceing applications + 1

# List of benchmarks to run
default_benchmarks=(
"cudaSDK_mms"       # Matrix Multiplication
"cudaSDK_sn"        # Sorting Networks
"cudaSDK_va"        # Vector Addition
"cudaSDK_sp"        # Scalar Product
"cudaSDK_fwt"       # Fast Walsh Transform
"rodinia_cfd"       # Computational Fluid Dynamics
#"rodinia_nn"        # Nearest neighbours
#"rodinia_gaussian"  # Gaussian Elimination
)

# Current selected benchmarks is same as default benchmarks
benchmarks=("${default_benchmarks[@]}")

NO_INTERFERENCE_DUMMY_APP="__none__"

# List of interference applications that run in background
# For now, they are subset of the list of benchmarks are these are the most
# interesting intereference applications
default_interferences=(
$NO_INTERFERENCE_DUMMY_APP
"cudaSDK_mms"       # Computational Intensive
"cudaSDK_fwt"       # Shared memory and memory intensive
"cudaSDK_va"        # Memory Intensive
)

# Current selected interefence is same as default interefence
interferences=("${default_interferences[@]}")


# Prefix for statement that shows that average runtime
# Statement is only printed if there is only one benchmark and one interefence application
# Otherwise a matrix is printed (without having the prefix)
RESULT_PREFIX="AVG_RUNTIME:"

print_usage() {
    echo "Usage:"
    echo "./run_benchmark.sh"
    echo "Optional Options:"
    echo "-b=<name>, --benchmark=<name> Select one benchmark from all available benchmarks applications"
    echo "-e=<command>, --external=<command> Supply a command to run as a benchmark"
    echo "-E=<command>, --External=<command> Supply a command to run as a interference"
    echo "-B, --benchmarks Prints default benchmark applications"
    echo "-c=<val>, --colors=<val> Sets the number of colors that hardware is configured for."
    echo "-h, --help Shows this usage."
    echo "-i=<name>, --interference=<name> Select one interference from all available interfence applications"
    echo "-I, --interferences Prints default interfence applications"
    echo "-n=<val>, --numIterations=<val> Set the number of iterations. Time show in average of runtimes of all iterations."
    echo "-v, --voltaMPS Enables MPS compute partitioning on Volta GPU. FGPU compute/memory coloring should be disabled in this configuration."
}

# Prints all benchmark applications
print_benchmarks() {
    echo "List of benchmark applications:"
    for i in ${benchmarks[@]}; do
        echo $i
    done
}

# Prints all interefence applications
print_interference() {
    echo "List of interfence applications:"
    for i in ${interferences[@]}; do
        echo $i
    done
}

# The values above are the default values of the variable. Some of the variables
# can be configured at runtime.
parse_args() {

    unset b
    unset it
    c=$NUM_COLORS
    n=$NUM_ITERATION
    v=$ENABLE_VOLTA_MPS_PARTITION

    for i in "$@"
    do
        case "$i" in
        -h*|--help*)
            print_usage
            exit 1
            ;;

        -I*|--interferences*)
            # List all interfence
            print_interference
            exit 1
            ;;

        -B*|--benchmarks*)
            # List all benchmarks
            print_benchmarks
            exit 1;
            ;;

        -b*|--benchmark=*)
            b="${i#*=}"
            
            # Check if benchmark exists
            unset found
            for j in "${benchmarks[@]}"
            do
                if [ "$j" = "$b" ] ; then
                    found=1
                fi
            done

            if [ -z "$found" ]; then
                echo "Invalid benchmark application"
                print_benchmarks
                exit 1
            fi

            shift # past argument=value
            ;;

        -e*|--external=*)
            b="${i#*=}"

            shift # past argument=value
            ;;

        -E*|--External=*)
            it="${i#*=}"

            shift # past argument=value
            ;;

        -i*|--interference=*)
            it="${i#*=}"
            
            # Check if benchmark exists
            unset found
            for j in "${interferences[@]}"
            do
                if [ "$j" = "$it" ] ; then
                    found=1
                fi
            done

            if [ -z "$found" ]; then
                echo "Invalid interfence application"
                print_interference
                exit 1
            fi

            shift # past argument=value
            ;;

        -c=*|--colors=*)
            c="${i#*=}"
            check_arg_is_number $c
            if [ $? -ne 0 ]; then
                print_usage
                exit 1
            fi

            shift # past argument=value
            ;;
        -n=*|--numIterations=*)
            it="${i#*=}"
            check_arg_is_number $it
            if [ $? -ne 0 ]; then
                print_usage
                exit 1
            fi

            shift # past argument=value
            ;;
        -v*|--voltaMPS*)
            v="1"
            shift # past argument=value
            ;;
        *)
            # unknown option
            print_usage
            exit 1
            ;;
        esac
    done

    NUM_COLORS=$c
    NUM_ITERATION=$n
    ENABLE_VOLTA_MPS_PARTITION=$v
    
    if [ ! -z "$b" ]; then
        benchmarks=("$b")
    fi

    if [ ! -z "$it" ]; then
        interferences=("$it")
    fi

    echo "Number of colors:$NUM_COLORS"
    echo "Running benchmarks with number of iterations:$NUM_ITERATION"
    if [ "$ENABLE_VOLTA_MPS_PARTITION" -eq "0" ]; then
        echo "Compute Partition based on Volta MPS is stated to be OFF"
    else
        echo "Compute Partition based on Volta MPS is stated to be ON"
    fi
}
