#!/bin/bash

# Include file containing configurable options
source ./config_benchmark.sh

# Shouldn't be running as root
check_if_not_sudo

# Parse command line arguments
parse_args "$@"

#Change pwd to Proj directory 
cd $PROJ_PATH

touch $LOG_FILE
TEMP_LOG_FILE=`mktemp`
TEMP_TIME_FILE=`mktemp`

# Get number of processes
proc=`nproc`

# Simuntaneously running application - Benchmark and interference app
# Divide cpus into two part
# Each interference/benchmark app needs atleast two CPU cores since CUDA spawns
# background threads. Hence for no inteference on CPU side, it is better to assign
# non-overlapping CPU cores.
app_proc=$((proc/NUM_COLORS))
if [[ "$app_proc"  -eq "0" ]] ||  [[ "$app_proc" -eq "1" ]]; then
    do_error_exit "Not enough CPU cores in the system to run benchmark (Cores: $proc, Colors: $NUM_COLORS)"
fi

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

# Kill all known processes that this script might have ran
kill_all_processes () {
    
    for ((i = 0; i < "${#benchmarks[@]}"; i++))
    do
        kill_process "${benchmarks[$i]}"
    done
   
    for ((i = 0; i < "${#interferences[@]}"; i++))
    do
        kill_process "${interferences[$i]}"
    done
}

# Incase of sigint, kill all processes (fresh state) and stop mps
function do_for_sigint() {
    kill_all_processes

    # Caffe application (incase it was running)
    pkill -9 -f "classification.bin"
    
    deinit_fgpu
    exit 1
}

trap 'do_for_sigint' SIGINT

# Function to run a benchmark with an interfering application
# First Arg - Application to run
# Second Arg - Interference Application (can be empty string)
# For return value, it modifies AVG variable
run_benchmark () { 

    bench_app="$1"
    int_app="$2"
    is_external_bench="0"
    is_external_int="0"

    # Check if external benchmark
    unset found
    for ((j = 0; j < ${#default_benchmarks[@]}; j++))
    do
        if [ "${default_benchmarks[$j]}" = "$bench_app" ]; then
            found=1
       fi
    done

    if [ -z "$found" ]; then
        is_external_bench="1"
    fi

    if [ "$is_external_bench" = "1" ]; then
        # Remove start and end quotes around the cmd if any 
        bench_app="${bench_app%\"}"
        bench_app="${bench_app#\"}"

        # Get the process name
        app_path=`echo $bench_app | awk '{print $1;}'`
        bench_name=`basename $app_path`
    else
        bench_name=$bench_app
    fi

    # Check if external interference
    unset found
    for ((j = 0; j < ${#default_interferences[@]}; j++))
    do
        if [ "${default_interferences[$j]}" = "$int_app" ]; then
            found=1
       fi
    done

    if [ -z "$found" ]; then
        is_external_int="1"
    fi

    if [ "$is_external_int" = "1" ]; then
        # Remove start and end quotes around the cmd if any 
        int_app="${int_app%\"}"
        int_app="${int_app#\"}"

        # Get the process name
        app_path=`echo $int_app | awk '{print $1;}'`
        int_name=`basename $app_path`
    else
        int_name=$int_app
    fi

    echo "Starting New Benchmark/Interference combination: [$bench_name, $int_name]"

    # Kill all processes
    kill_all_processes
    
    init_fgpu

    percentage=`expr 100 / $NUM_COLORS`

    # Run inteference applications (in a infinte loop). Wait for sometime for them to start running.
    # Then start running the benchmark. Wait for benchmark to complete. Kill all
    # inteference applications then (since they will not exit on their own as running in infinte loop).

    # If no interference, skip launching inteference application
    # We need to run schedtool which requires root permission but we don't want to
    # run the benchmark as root
    
    # Some benchmarks/Inteference take arguments via command line arguments and some take arguments from environment variables
    # Hence supply both
    # Also for external benchmarks/Inteference, they might not be properly linked with fgpu library
    # Hence supply LD_PRELOAD and LD_LIBRARY_PATH

    declare -a background_pid
    pidfile=`mktemp`
    user=`whoami`
    if ! [[ $int_name = $NO_INTERFERENCE_DUMMY_APP ]]
    then
        for i in `seq 1 $num_it`;
        do  
            color=`expr $INTERFERENCE_COLOR + $i  - 1`
            index=`expr $i - 1`
            range=${int_proc_ranges[$index]}
            # Max number of iterations
            ((num_iterations=2**31-1))
            if [ "$is_external_int" = "1" ]; then

                if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
                then
                    sudo taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e    \
                        su -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage        \
                                FGPU_NUM_ITER_ENV=$num_iterations                   \
                                FGPU_COLOR_ENV=$color                               \
                                FGPU_COLOR_MEM_SIZE_ENV=$MEMORY                     \
                                LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BIN_PATH          \
                                LD_PRELOAD=$BIN_PATH/$FGPU_LIBRARY                  \
                                $int_app -c $color -m $MEMORY -k &> /dev/null & echo \$! >> $pidfile" $user
                    echo "Interference:$int_name, Color:$color, CPU Affinity:$range, MPS Percentage:$percentage"
                else
                    sudo taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e    \
                        su -c "FGPU_NUM_ITER_ENV=$num_iterations                    \
                                FGPU_COLOR_ENV=$color                               \
                                FGPU_COLOR_MEM_SIZE_ENV=$MEMORY                     \
                                LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BIN_PATH          \
                                LD_PRELOAD=$BIN_PATH/$FGPU_LIBRARY                  \
                                $int_app -c $color -m $MEMORY -k &> /dev/null & echo \$! >> $pidfile" $user
                    echo "Interference:$int_name, Color:$color, CPU Affinity:$range"
                fi
            else
                if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
                then
                    sudo taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e    \
                        su -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage        \
                        $BIN_PATH/$int_app -c $color -m $MEMORY  -k &> /dev/null & echo \$! >> $pidfile" $user
                    echo "Interference:$int_name, Color:$color, CPU Affinity:$range, MPS Percentage:$percentage"
                else
                    sudo taskset -c $range schedtool -R -p $INTERFERENCE_PRIO -e    \
                        su -c "$BIN_PATH/$int_app -c $color -m $MEMORY  -k &> /dev/null & echo \$! >> $pidfile" $user
                    echo "Interference:$int_name, Color:$color, CPU Affinity:$range"
                fi
            fi
    	done
        readarray background_pid < $pidfile
        sleep 5
    fi
    
    echo "New Benchmark: [$bench_name, $int_name]" > $TEMP_LOG_FILE
    if [ "$is_external_bench" = "1" ]; then

        if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
        then
            echo "Benchmark: $bench_name, Color:$BENCHMARK_COLOR, CPU Affinity:$bench_proc_range, MPS Percentage:$percentage"
            sudo taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e    \
                su -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage                \
                        FGPU_NUM_ITER_ENV=$NUM_ITERATION                            \
                        FGPU_COLOR_ENV=$BENCHMARK_COLOR                             \
                        FGPU_COLOR_MEM_SIZE_ENV=$MEMORY                             \
                        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BIN_PATH                  \
                        LD_PRELOAD=$BIN_PATH/$FGPU_LIBRARY                          \
                        $bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE" $user
        else
            echo "Benchmark:$bench_name, Color:$BENCHMARK_COLOR, CPU Affinity:$bench_proc_range"
            sudo taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e    \
                su -c "FGPU_NUM_ITER_ENV=$NUM_ITERATION                             \
                        FGPU_COLOR_ENV=$BENCHMARK_COLOR                             \
                        FGPU_COLOR_MEM_SIZE_ENV=$MEMORY                             \
                        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BIN_PATH                  \
                        LD_PRELOAD=$BIN_PATH/$FGPU_LIBRARY                          \
                        $bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE" $user
        fi
    else
        if [ $ENABLE_VOLTA_MPS_PARTITION -ne "0" ];
        then
            echo "Benchmark: $bench_name, Color:$BENCHMARK_COLOR, CPU Affinity:$bench_proc_range, MPS Percentage:$percentage"
            sudo taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e    \
                su -c "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage                \
                $BIN_PATH/$bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE" $user
        else
            echo "Benchmark:$bench_name, Color:$BENCHMARK_COLOR, CPU Affinity:$bench_proc_range"
            sudo taskset -c $bench_proc_range schedtool -R -p $BENCHMARK_PRIO -e    \
                su -c "$BIN_PATH/$bench_app -c $BENCHMARK_COLOR -m $MEMORY -i $NUM_ITERATION >> $TEMP_LOG_FILE" $user
        fi

    fi
    ret_code=$?

    # Kill all background applications
    sync
    kill_all_processes
    for ((i = 0; i < "${#background_pid[@]}"; i++))
    do
        kill -9 ${background_pid[$i]} &> /dev/null
    done

    sleep 2
    deinit_fgpu

    # Append to log file
    cat $TEMP_LOG_FILE >> $LOG_FILE
    echo "################################################" >> $LOG_FILE

    # Check if program successful?
    if [ $ret_code -ne 0 ]
    then
        do_error_exit "Benchmark failed. See file $TEMP_LOG_FILE"
    fi

    # Get the stats
    stats=`grep -A 1  "Kernel Stats" $TEMP_LOG_FILE`
    if [ -z "$stats" ]
    then
        stats=`grep -A 1  "Overall Stats" $TEMP_LOG_FILE`
        if [ -z "$stats" ]; then
            do_error_exit "Couldn't find kernel/overall stats. See file $TEMP_LOG_FILE"
        fi
        echo "NOTE: Taking overall (GPU Kernel + CPU) time into consideration for benchmark"
    fi

    # Extract the averge runtime
    avg=`echo $stats |  grep -Po 'Avg:([a-z0-9.]+)' | sed -n "s/Avg://p"`

    echo "Avg Runtime: $avg"
    AVG=$avg

    # Kill all processes
    if ! [[ $int_name = $NO_INTERFERENCE_DUMMY_APP ]]; then
        echo "NOTE: Terminating background interference applications that where running in forever loop"
    fi
}

runtimes=()

# Kill all process first 
kill_all_processes

AVG_RUNTIMES=()

# Pretty print the columns - Inteference Applications
echo -n -e "App\t" > $TEMP_TIME_FILE
for ((i = 0; i < ${#interferences[@]}; i++))
do
    echo -n -e "${interferences[$i]}\t" >> $TEMP_TIME_FILE
done

# Get and print the averge runtime of benchmarks (w and w/o interference) 
for ((i = 0; i < ${#benchmarks[@]}; i++))
do
    b="${benchmarks[$i]}"
    
    echo -n -e "$b\t" >> $TEMP_TIME_FILE

    for ((j = 0; j < ${#interferences[@]}; j++))
    do
        it=${interferences[$j]}

        run_benchmark "$b" "$it"
        echo -n -e "$AVG\t" >> $TEMP_TIME_FILE
        AVG_RUNTIMES+=("$AVG")
    done

    echo "" >> $TEMP_TIME_FILE

done

# If one benchmark and one inteference application, print the result statement also
if [ ${#AVG_RUNTIMES[@]} = "1" ]; then
    echo "$RESULT_PREFIX ${AVG_RUNTIMES[0]} us"
else
    cat $TEMP_TIME_FILE
fi

echo "See log file at $LOG_FILE"

# Delete temp file
rm $TEMP_LOG_FILE
rm $TEMP_TIME_FILE

exit 0
