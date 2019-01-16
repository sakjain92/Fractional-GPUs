# Get this script's path
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Change the display manager depending on the system
DISPLAY_MANAGER="lightdm"

PROJ_PATH=$SCRIPTPATH/../           # Path to FGPU project directory
BIN_PATH=$PROJ_PATH/build           # Path to binaries (edit if different)
SCRIPTS_PATH=$PROJ_PATH/scripts     # Folder containing scripts
DRIVER_PATH="$PROJ_PATH/driver/NVIDIA-Linux-x86_64-390.48/"
DRIVER_INSTALL_PATH="$DRIVER_PATH/nvidia-installer"
REVERSE_ENGINEERING_PATH="$PROJ_PATH/reverse_engineering"
BENCHMARK_PATH="$PROJ_PATH/benchmarks"
FGPU_LIBRARY="libfractional_gpu.so"
CAFFE_PATH="$PROJ_PATH/framework/caffe/"

SERVER="fgpu_server"                # Server name
REVERSE_ENGINEERING_BINARY="gpu_reverse_engineering"
REVERSE_ENGINEERING_PLOT="plot_reverse_engineering.sh"
BENCHMARK_SCRIPT="run_benchmark.sh"

# Following GPUS are supported by FGPU currently
supported_gpus=(
"GeForce GTX 1070"
"GeForce GTX 1080"
"Tesla V100-SXM2-16GB"
)

# Aliases used in academic paper
declare -A default_aliases=(
["cudaSDK_mms"]="MM" 
["cudaSDK_sn"]="SN"
["cudaSDK_va"]="VA"
["cudaSDK_sp"]="SP"
["cudaSDK_fwt"]="FWT"
["rodinia_cfd"]="CFD"
["__none__"]="<NONE>"
)

# Get any input from user to continue
pause_for_user_input() {
    read -p "Press enter to continue"
}

# Function exits with printing error
# First option is the error message
do_error_exit() {
    echo "Error: $1"
    exit -1
}

# Function tests if input is a number or not. It should also be greater than 0.
check_arg_is_number() {
    if [ -n "$1" ] && [ "$1" -eq "$1" ] && [ "$1" -gt "0" ] 2>/dev/null; then
        return 0
    else
        echo "Invalid arguments"
        return 1
    fi
}

# Checks if input is an input and lies between two ranges [a, b]
# First argument is the input
# Second argument is the lower bound
# Third argument is the upper bound
check_arg_between() {

    if [ $# -ne 3 ];
    then
        do_error_exit "Invalid argument"
    fi

    check_arg_is_number $1
    if [ $? -ne 0 ]; then
        do_error_exit "Invalid argument"
    fi

    if [ $1 -lt $2 ] || [ $1 -gt $3 ]; then
        do_error_exit "Invalid argument"
    fi
    return 0
}

# Checks if script has sudo permissions
check_if_not_sudo() {
    if [[ $EUID -eq 0 ]]; then
        echo "This script should not be run using sudo or as the root user"
        exit 1
    fi
    return 0
}

# Compiles FGPU and install driver 
# Inputs are cmake options
build_and_install_fgpu() {

    cmake_args=""
    while [ $# -gt 0 ]
    do
        cmake_args="$cmake_args -D$1"
        shift
    done

    # Compile FGPU
    echo "*************************************"
    echo "Compiling FGPU (Might take some time)"
    echo "*************************************"

    mkdir -p $BIN_PATH
    cd $BIN_PATH
    rm -f "CMakeCache.txt"
    log=`mktemp`
    cmake $cmake_args $PROJ_PATH &> $log
    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't configure FGPU. See log file $log"
    fi
    
    numcores=`nproc`
    make -j$numcores &>> $log
   
    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't compile FGPU. See log file $log"
    fi

    cur_dir=$PWD

    echo "***********************************************************************"
    echo "Installing Kernel NVIDIA driver (might take some time, ignore warnings)"
    echo "***********************************************************************"

    # Install driver
    # Need to first kill display manager that might be using the GPU
    # Basically the driver installation requires Xorg to not be running
    sudo service $DISPLAY_MANAGER stop &> /dev/null
    sudo pkill -9 Xorg

    # In normal scenarios, the default options should be fine so no need to
    # prompt user for inputs
    cd $DRIVER_PATH
    sudo $DRIVER_INSTALL_PATH --silent

    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't install driver"
    fi

    cd $cur_dir
}

# Compiles Caffe
compile_caffe() {
    
    echo "*************************************"
    echo "Compiling Caffe (Might take some time)"
    echo "*************************************"

    cur_dir=$PWD

    cd $CAFFE_PATH
    # Copy the config file if it doesn't exist
    if [ ! -f Makefile.config ]; then
        cp Makefile.config.example Makefile.config
        file="0"
    fi

    numcores=`nproc`
    log=`mktemp`
    make -j$numcores &>> $log
   
    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't compile Caffe. See log file $log"
    fi

    if [ "$file" = "0" ]; then
        rm Makefile.config
    fi

    cur_dir=$PWD
}

# Prints all supported GPUs
print_supported_gpus() {
    echo "List of FGPU supported GPUs:"
    for i in "${supported_gpus[@]}"
    do
        echo "$i"
    done
}

# Checks if gpu is volta gpu
# First argument is the gpu name
check_is_volta() {
    if [[ $1 ==  *"V100"* ]]; then
        return 1
    fi

    return 0
}

# Finds the first GPU in the system and checks if it is supported and if it is
# of volta architecture
check_is_volta_gpu() {
    # Get the first GPU
    status=`nvidia-smi --list-gpus | head -n 1`
    if [ $? -ne 0 ]; then
        do_error_exit "nvidia-smi command gave error. Check CUDA/Nvidia driver is properly installed."
    fi

    # The format of nvidia-status is:
    # GPU 0: <GPU NAME> (UUID:...)
    gpu_name=`echo $status | sed -e 's/GPU 0: \(.*\) (.*)/\1/'`

    # Check if GPU is supported
    found=0
    for i in "${supported_gpus[@]}"
    do
        if [ "$i" = "$gpu_name" ]; then
            found=1
        fi
    done

    echo "*******************************************************************************"
    echo "First GPU (the GPU that will be used by FGPU) is detected to be '$gpu_name'"
    echo "*******************************************************************************"
    if [ $found -eq 0 ]; then
        echo "Non-supported GPU ($gpu_name)"
        print_supported_gpus
        do_error_exit ""
    fi

    check_is_volta "$gpu_name"
    return $?
}

# Kills a process 
# First argument is the name of the process
kill_process() {

    # Get the process name from the cmd
    cmd="$1"
    name=`echo "$cmd" | awk '{ print $1 }'`
    pid=`pgrep $name`
    if [ ! -z "$pid" ]
    then
        sudo kill $pid &> /dev/null
        wait $pid &> /dev/null
    fi
}

# Cleanly clears the enviroment after FGPU is finished
deinit_fgpu() {
    kill_process $SERVER

    sleep 2

    # Kill all possibly running benchmarks
    sudo pkill -9 $SERVER
    sudo pkill -9 cudaSDK
    sudo pkill -9 rodinia

    # End MPS
    sudo $SCRIPTS_PATH/mps_kill.sh  &> /dev/null

    return 0
}

# Sets up environment before launching fgpu applications
init_fgpu() {

    # Start afresh
    deinit_fgpu

    # Start MPS
    sudo $SCRIPTS_PATH/mps_init.sh &> /dev/null
    if [ $? -ne 0 ]; then 
        do_error_exit "Couldn't start MPS"
    fi

    # Remove server file
    sudo rm -f /dev/shm/fgpu_shmem > /dev/null
    sudo rm -f /dev/shm/fgpu_host_shmem > /dev/null

    sleep 5

    # Start server in background
    $BIN_PATH/$SERVER &> /dev/null &
    sleep 5
 
    return 0
}

# Checks if file exists
# First argument is the file path
check_file_exists() {
    if [ ! -f $1 ]; then
        do_error_exit "File $1 not found"
    fi
    return 0
}

# Checks if an application is installed
# First argument is the name of the application
check_if_installed() {

    which $1 &> /dev/null
    if [ $? -ne 0 ]; then
        do_error_exit "$1 is not installed. Please install it."
    fi
    return 0
}
