#!/bin/bash

# Include file containing configurable options
source ./config_benchmark.sh

# Script needs root permissions
if [[ $EUID -ne 0 ]]; then
    echo "This script should not be run using sudo or as the root user"
    exit 1
fi

#Change pwd to Proj directory 
cd $PROJ_PATH

# Remove log file 
rm $LOG_FILE &> /dev/null

touch $LOG_FILE
TEMP_LOG_FILE=`mktemp`
TEMP_TIME_FILE=`mktemp`

# Get number of processes
proc=`nproc`

# Simuntaneously running application - Benchmark and interference app
# Divide cpus into two part
app_proc=$((proc/NUM_COLORS))
num_it=$((NUM_COLORS - 1))

int_proc_ranges=()
# Give first partitions to interference and last to benchmark
for i in `seq 1 $NUM_COLORS`;
do
	index=$((i-1))
	int_proc_range=$((index * app_proc))'-'$((index * app_proc + app_proc - 1))
	int_proc_ranges+=($int_proc_range)
done
bench_proc_range=$((proc - app_proc))'-'$((proc-1))

AVG="0"

# Kills a process 
# First argument is the name of the process
kill_process() {
    pid=`pgrep $1`
    if [ ! -z "$pid" ]
    then
         kill $pid &> /dev/null
         wait $pid &> /dev/null
    fi
}

# Kill all known processes that this script might have ran
kill_all_processes () {
    for bench in ${benchmarks[@]}; do
        kill_process $bench
    done
   
    for int in ${interference[@]}; do
        kill_process $int
    done

    kill_process $SERVER
}

# Incase of sigint, kill all processes (fresh state) and stop mps
function do_for_sigint() {
    kill_all_processes
    $SCRIPTS_PATH/mps_kill.sh  &> /dev/null
    exit 1
}

trap 'do_for_sigint' 2

# Function to run a benchmark with an interfering application
# First Arg - Application to run
# Second Arg - Interference Application (can be empty string)
# For return value, it modifies AVG variable
run_benchmark () { 

    bench_app=$1
    int_app=$2

    echo "Starting New benchmark: [$bench_app, $int_app]"

    # Kill all processes
    kill_all_processes

    # Start MPS
     $SCRIPTS_PATH/mps_init.sh &> /dev/null

    # Remove server file
     rm -f /dev/shm/fgpu_shmem > /dev/null
     rm -f /dev/shm/fgpu_host_shmem > /dev/null

    sleep 5

    # Start server
    $BIN_PATH/$SERVER &> /dev/null &
    sleep 5

    percentage=`expr 100 / $NUM_COLORS`

    if [ ! -z $int_app ]
    then
        for i in `seq 1 $num_it`;
        do  
		color=`expr $INTERFERENCE_COLOR + $i  - 1`
		index=`expr $i - 1`
		range=${int_proc_ranges[$index]}
	        if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
        	then
            		CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e  $BIN_PATH/$int_app -c $color -m $MEMORY  -k &> /dev/null &
			echo "Interference:$int_app, Color:$color, Range:$range, MPS Percentage:$percentage"
        	else
            		taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e  $BIN_PATH/$int_app -c $color -m $MEMORY  -k &> /dev/null &
                        echo "Interference:$int_app, Color:$color, Range:$range"
        	fi
    	done
        sleep 5
    fi

    echo "New Benchmark: [$bench_app, $int_app]" > $TEMP_LOG_FILE
    if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
    then
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e  $BIN_PATH/$bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE
        echo "Benchmark: $bench_app, Color:$BENCHMARK_COLOR, Range:$bench_proc_range, MPS Percentage:$percentage"
    else
        taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e  $BIN_PATH/$bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE
    	echo "Benchmark:$bench_app, Color:$BENCHMARK_COLOR, Range:$bench_proc_range"
    fi

    ret_code=$?

    # Kill all processes
    kill_all_processes
    sleep 2

    # End MPS
     $SCRIPTS_PATH/mps_kill.sh  &> /dev/null

    # Append to log file
    cat $TEMP_LOG_FILE >> $LOG_FILE
    echo "################################################" >> $LOG_FILE

    # Check if program successful?
    if [ $ret_code -ne 0 ]
    then
        echo "Benchmark failed. See file $TEMP_LOG_FILE"
        #exit -1
    fi

    # Get the stats
    stats=`grep -A 1  "Kernel Stats" $TEMP_LOG_FILE`
    if [ -z "$stats" ]
    then
        echo "Couldn't find kernel stats. See file $TEMP_LOG_FILE"
        #exit -1
    fi

    # Extract the averge runtime
    avg=`echo $stats |  grep -Po 'Avg:([a-z0-9.]+)' | sed -n "s/Avg://p"`

    echo "Avg Runtime: $avg"
    AVG=$avg
}

runtimes=()

echo "Running benchmarks with number of iterations:$NUM_ITERATION"
echo "ENABLE_VOLTA_MPS_PARTITION is $ENABLE_VOLTA_MPS_PARTITION"

# Kill all process first 
kill_all_processes

# Pretty print the columns - Inteference Applications
echo -n -e "App\tNo Interference\t" > $TEMP_TIME_FILE
for i in ${interference[@]}; do
    echo -n -e "$i\t" >> $TEMP_TIME_FILE
done
echo "" >> $TEMP_TIME_FILE

# Get and print the averge runtime of benchmarks (w and w/o interference) 
for b in ${benchmarks[@]}; do
    
    echo -n -e "$b\t" >> $TEMP_TIME_FILE

    # First run with no interference
    run_benchmark $b ""
    echo -n -e "$AVG\t" >> $TEMP_TIME_FILE
 
    for i in ${interference[@]}; do
        run_benchmark $b $i 
        echo -n -e "$AVG\t" >> $TEMP_TIME_FILE
    done

    echo "" >> $TEMP_TIME_FILE

done


cat $TEMP_TIME_FILE

echo "See log file at $LOG_FILE"

# Delete temp file
rm $TEMP_LOG_FILE
rm $TEMP_TIME_FILE
