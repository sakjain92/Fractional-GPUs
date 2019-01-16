#!/bin/bash

# Run this script without any arguments. It takes arguments from user during
# runtime.

COMMON_SCRIPT=../scripts/common.sh

if [ ! -f $COMMON_SCRIPT ]; then
    echo "Run this script from \$PROJ_DIR/scripts folder"
fi

source $COMMON_SCRIPT

# Shouldn't be running as root
check_if_not_sudo

# Different modes of FGPU
FGPU_DISABLED=1                  # No compute/memory partitioning
FGPU_COMPUTE_ONLY=2              # Compute partitioning only
FGPU_COMPUTE_AND_MEMORY=3        # Compute and memory partitioning
FGPU_VOLTA_COMPUTE_ONLY=4        # For only volta GPU, we have Volta MPS based compute partitoning
FGPU_REVERSE_ENGINEERING=5       # Reverse engineering

# Tracks the current FGPU mode
FGPU_MODE=''
FGPU_MODE_NAME=''

# Different modes of evaluation
EVAL_REVERSE=1                   # Reverse engineering
EVAL_BENCHMARK=2                 # Benchmark using CUDA/Rodinia
EVAL_CAFFE=3                     # Benchmark using caffe

# Tracks the current evaluation mode
EVAL_MODE=''

# Refresh state of machine
echo "INFO:Cleaning up any previous FGPU related state"
deinit_fgpu

# Step 1 - Find the GPU
check_is_volta_gpu
IS_VOLTA=$?

# Prints out current FGPU mode
print_fgpu_mode() {
    echo    "*****************************************"
    echo    "Running in FGPU MODE: $FGPU_MODE_NAME"
    echo    "*****************************************"
    
    return 0
}

# Configures FGPU in a specific mode
# First argument is the FGPU mode number
configure_fgpu() {
    # If FGPU is already set to the desired mode, skip configuring
    if [ "$FGPU_MODE" = "$1" ]; then
        return 0
    fi

    case $1 in
    1)
        echo "*********************************"
        echo "Configuring FGPU in disabled mode"
        echo "*********************************"
        FGPU_MODE=$FGPU_DISABLED
        FGPU_MODE_NAME="FGPU DISABLED"
        build_and_install_fgpu "FGPU_COMP_COLORING_ENABLE=OFF" "FGPU_MEM_COLORING_ENABLED=OFF" "FGPU_TEST_MEM_COLORING_ENABLED=OFF"
        ;;

    2)
        echo "**************************************************"
        echo "Configuring FGPU in compute partitioning only mode"
        echo "**************************************************"
        FGPU_MODE=$FGPU_COMPUTE_ONLY
        FGPU_MODE_NAME="FGPU COMPUTE ONLY PARTITIONING MODE (CP)"
        build_and_install_fgpu "FGPU_COMP_COLORING_ENABLE=ON" "FGPU_MEM_COLORING_ENABLED=OFF" "FGPU_TEST_MEM_COLORING_ENABLED=OFF"
        ;;
    3)
        echo "********************************************************"
        echo "Configuring FGPU in compute and memory partitioning mode"
        echo "********************************************************"
        FGPU_MODE=$FGPU_COMPUTE_AND_MEMORY 
        FGPU_MODE_NAME="FGPU COMPUTE AND MEMORY PARTITIONING MODE (CMP)"
        build_and_install_fgpu "FGPU_COMP_COLORING_ENABLE=ON" "FGPU_MEM_COLORING_ENABLED=ON" "FGPU_TEST_MEM_COLORING_ENABLED=OFF"
        ;;
    4)
        echo "************************************************************************************************"
        echo "Configuring FGPU in disabled mode (Volta MPS will do the compute partitioning, FGPU is bypassed)"
        echo "************************************************************************************************"
        FGPU_MODE=$FGPU_VOLTA_COMPUTE_ONLY
        FGPU_MODE_NAME="FGPU VOLTA COMPUTE ONLY PARTITIONING MODE (MPS)"
        build_and_install_fgpu "FGPU_COMP_COLORING_ENABLE=OFF" "FGPU_MEM_COLORING_ENABLED=OFF" "FGPU_TEST_MEM_COLORING_ENABLED=OFF"
        ;;

    5)
        echo "********************************************"
        echo "Configuring FGPU in reverse engineering mode"
        echo "********************************************"
        FGPU_MODE=$FGPU_REVERSE_ENGINEERING 
        FGPU_MODE_NAME="FGPU REVERSE ENGINEERING MODE"
        build_and_install_fgpu "FGPU_COMP_COLORING_ENABLE=ON" "FGPU_MEM_COLORING_ENABLED=ON" "FGPU_TEST_MEM_COLORING_ENABLED=ON"
        ;;
    esac

    return 0
}

# Ask user's input and correspondingly configure FGPU
ask_and_configure_fgpu() {
    echo "Choose one of the following FGPU modes of configuration (available for current GPU)"
    echo "1) Computational Partitioning Only"
    echo "2) Computational and Memory Partitioning"
    if [ $IS_VOLTA -ne 0 ]; then
        echo "3) Volta MPS based Compute partitioning (FGPU bypassed)"
        echo "Enter option(1 or 2 or 3): "
    else
        echo "Enter option(1 or 2): "
    fi
    
    read fgpu_mode_number
    echo ""

    if [ $IS_VOLTA -ne 0 ]; then
        check_arg_between $fgpu_mode_number 1 3
    else
        check_arg_between $fgpu_mode_number 1 2
    fi

    # We do not allow user to disable fgpu so we skipped that option
    # Internal scripts function can disable fgpu
    fgpu_mode_number=$((fgpu_mode_number+1))

    configure_fgpu $fgpu_mode_number

    return 0
}

# If can of error, cleanup
function do_for_sigint() {
    kill_process $REVERSE_ENGINEERING_BINARY
    deinit_fgpu
    exit 1
}

trap 'do_for_sigint' EXIT

# Runs a benchmark with an interference
# Arg 1 is the command to launch the benchmark
# Sets the variable BENCHMARK_RUNTIME to the average runtime
unset BENCHMARK_RUNTIME
run_benchmark_cmd() {

    cmd=$1
    # Enable volta mps based compute partitioning
    if [ "$FGPU_MODE" = "$FGPU_VOLTA_COMPUTE_ONLY" ]; then
        cmd="$cmd -v"
    fi

    cur_dir=`pwd`

    cd $BENCHMARK_PATH
    log=`mktemp`

    # Print to stdout and store in variable
    echo ""

    bash -c "$cmd | tee $log"
    echo ""

    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't run benchmark"
    fi

    # Get the runtime
    BENCHMARK_RUNTIME=`cat $log | grep -oP '(?<=AVG_RUNTIME: )[0-9.]+'`
    if [ $? -ne 0 ]; then
        do_error_exit "Couldn't find the runtime of benchmark. See $log"
    fi

    cd $cur_dir
}

# Given a pair of benchmark and interference, run them
# First argument is the benchmark command to run
# Second argument is the interference command to run
# Third argument is the benchmark name
# Fourth argument is the intereference name
# Fifth argument is the benchmark aliases
# Sixth argument is the interference aliases
# Seventh argument is the number of colors
# Eigth argument is the number of iterations
run_benchmark() {
    bcmd="$1"
    icmd="$2"
    bname="$3"
    iname="$4"
    balias="$5"
    ialias="$6"
    num_colors=$7
    num_iterations=$8

    # Different commands based on whether an external command or one of the default benchmark/interference
    if [[ -v "default_aliases[$bname]" ]]; then
        if [[ -v "default_aliases[$iname]" ]]; then
            cmd="$BENCHMARK_PATH/$BENCHMARK_SCRIPT -c=$num_colors -n=$num_iterations -i=$icmd -b=$bcmd"
            run_benchmark_cmd "$cmd"
        else
            cmd="$BENCHMARK_PATH/$BENCHMARK_SCRIPT -c=$num_colors -n=$num_iterations -E=\"$icmd\" -b=$bcmd"
            run_benchmark_cmd "$cmd"
        fi
    else
        if [[ -v "default_aliases[$iname]" ]]; then
            cmd="$BENCHMARK_PATH/$BENCHMARK_SCRIPT -c=$num_colors -n=$num_iterations -i=$icmd -e=\"$bcmd\""
            run_benchmark_cmd "$cmd"
        else
            cmd="$BENCHMARK_PATH/$BENCHMARK_SCRIPT -c=$num_colors -n=$num_iterations -E=\"$icmd\" -e=\"$bcmd\""
            run_benchmark_cmd "$cmd"
        fi
    fi
}

# Given a list of benchmarks and interferences, run them
# FGPU must be properly configured before calling this
# First argument is the array of benchmark command to run
# Second argument is the array of interference command to run
# Third argument is the array of benchmark name
# Fourth argument is the array of intereference name
# Fifth argument is the array of benchmark aliases
# Sixth argument is the array of interference aliases
# Seventh argument is the number of colors
# Eigth argument is the number of iterations
# Nineth argument is if caffe needs to be compiled
run_all_benchmark() {

    local -n benchcmds=$1
    local -n intcmds=$2
    local -n benchnames=$3
    local -n intnames=$4
    local -n benchaliases=$5
    local -n intaliases=$6
    num_colors=$7
    num_iterations=$8
    is_caffe=$9

    declare -a runtimes
    declare -a baseline
    declare -a normalized

    # Current FGPU mode
    chosen_fgpu_mode="$FGPU_MODE_NAME"

    for ((i = 0; i < "${#benchcmds[@]}"; i++))
    do
        for ((j = 0; j < "${#intcmds[@]}"; j++))
        do
            bcmd=${benchcmds[$i]}
            icmd=${intcmds[$j]}
            bname=${benchnames[$i]}
            iname=${intnames[$j]}
            balias=${benchaliases[$i]}
            ialias=${intaliases[$j]}

            echo "*************************************************************************************"
            echo "Running Benchmark:$bname(Alias:$balias) with Interference:$iname(Alias:$ialias)"
            echo "*************************************************************************************"
            run_benchmark "$bcmd" "$icmd" "$bname" "$iname" "$balias" "$ialias" $num_colors $num_iterations
            runtimes+=($BENCHMARK_RUNTIME)
        done
    done

    echo "INFO: Running with FGPU disabled (no partitioning) mode to gather baseline runtimes (to normalize)"
    echo "INFO: For measuring baseline, we disable FGPU and for each benchmark, we run it alone without any interference"
    configure_fgpu $FGPU_DISABLED
    print_fgpu_mode
    
    if [ "$is_caffe" = "1" ]; then
        compile_caffe
    fi

    baseline_interference="__none__"
   
    for ((i = 0; i < "${#benchcmds[@]}"; i++))
    do
        # For normalization, base case is when FGPU is disabled,
        # and benchmark application runs alone fully utilizing the whole
        # GPU
        bcmd=${benchcmds[$i]}
        icmd=$baseline_interference
        bname=${benchnames[$i]}
        iname=$baseline_interference
        balias=${benchaliases[$i]}
        ialias=${default_aliases[$baseline_interference]}

        echo "**************************************************************************************"
        echo "Running Benchmark:$bname(Alias:$balias) with Interference:$iname(Alias:$ialias)"
        echo "**************************************************************************************"

        run_benchmark "$bcmd" "$icmd" "$bname" "$iname" "$balias" "$ialias" $num_colors $num_iterations
        base=($BENCHMARK_RUNTIME)

        for ((j = 0; j < "${#intcmds[@]}"; j++))
        do
            baseline+=($base)
        done
    done

    for i in "${!runtimes[@]}"; 
    do
        run=${runtimes[$i]}
        base=${baseline[$i]}
        norm=`bc -l <<< $run/$base/$num_colors`
        normalized+=("$norm")

        # Trim
        runtimes[i]=`printf "%.2f" $run`
        baseline[i]=`printf "%.2f" $base`
        normalized[i]=`printf "%.2f" $norm`
    done

    result_file=`mktemp`
    printf "Index\tBenchmark(Alias)-Interference(Alias)\tNumIterations\tNumColors\tAvgRunTime(usec)\tBaselineBenchmark(Alias)-Inteference(Alias)\tBaselineAvgRunTime(usec)\tNormalizedRunTime\n" > $result_file

    index=0
    for ((i = 0; i < "${#benchcmds[@]}"; i++))
    do
        for ((j = 0; j < "${#intcmds[@]}"; j++))
        do
            run=${runtimes[$index]}
            norm=${normalized[$index]}
            base=${baseline[$index]}
            index=$((index+1))
            bname=${benchnames[$i]}
            iname=${intnames[$j]}
            balias=${benchaliases[$i]}
            ialias=${intaliases[$j]}
            bialias=${default_aliases[$baseline_interference]}
            printf "$index\t%-40s\t$num_iterations\t$num_colors\t%-10s\t%-40s\t%-10s\t$norm\n" "$bname($balias)-$iname($ialias)" "$run" "$bname($balias)-$baseline_interference($bialias)" "$base" >> $result_file
        done
    done

    echo ""
    echo "Printing Results for $chosen_fgpu_mode"

    cat $result_file

    # Refer to academic paper for definition of variation
    echo ""
    echo "Printing Variation"
    overall_max_variation=0
    sum_variation=0
    count_variation=0
    index=0
    for ((i = 0; i < "${#benchcmds[@]}"; i++))
    do
        max_run=0
        divisor=0
        for ((j = 0; j < "${#intcmds[@]}"; j++))
        do
            run=${runtimes[$index]}
            iname=${intnames[$j]}
            if [ "$iname" = "$baseline_interference" ]; then
                divisor=$run
            fi
            if [ `echo $run'>'$max_run | bc -l` -eq 1 ]; then
                max_run=$run
            fi
            index=$((index+1))
        done
        max_variation=`echo $max_run'/'$divisor'*100 -100'  | bc -l`
        if [ `echo $max_variation'>'$overall_max_variation | bc -l` -eq 1 ]; then
            overall_max_variation=$max_variation
        fi
        sum_variation=`echo $sum_variation'+'$max_variation | bc -l`
        count_variation=$((count_variation+1))
    done

    avg_variation=`echo $sum_variation'/'$count_variation | bc -l`
    echo "*************************************************************************"
    printf "Average Variation: %.2f%% Max variation: %.2f%%\n" "$avg_variation" "$overall_max_variation"
    echo "*************************************************************************"
    echo ""
    echo "****************************************************"
    echo "Raw benchmark results saved in file $result_file"
    echo "****************************************************"
    
    output_plot=`mktemp`
    output_plot+="_benchmarks.png"
    plot_benchmark "$result_file" "$output_plot" "$chosen_fgpu_mode" $num_colors $num_iterations "${#benchcmds[*]}" "${#intcmds[*]}"

    echo ""
    echo "****************************************************"
    echo "Benchmark results plot is saved in file $output_plot"
    echo "****************************************************"
    echo "Open the file to see the plot"
    pause_for_user_input 
}


# Using GNU plot, plots and save benchmark results
# First argument is the input data file
# Second argument is the output png file
# Third argument is the FGPU mode
# Fourth is number of colors
# Fifth is number of iterations
# Sixth is number of benchmark applications
# Seventh is number of interference
plot_benchmark() {
    
    # Check if gnuplot exists?
    check_if_installed "gnuplot"

    gnu_command=""

    # Create color pallete
    # Varying shaded of red
    num_it=$7
    num_b=$6
    index=0
    for i in $(seq 1 $num_b); 
    do
        shade="0xFF"
        for j in $(seq 1 $num_it)
        do
            index=$((index+1))
            color_shade=`printf "%.2X\n" $((shade - 0XFF/2/num_it))`
            shade="0x$color_shade"
            color="#FF$color_shade$color_shade"
            gnu_command="$gnu_command;set linetype $index lc rgb  '$color'"
        done
    done

    gnu_command="$gnu_command;
        set term 'pngcairo' size 2560,1920;
        set output '$2';
        set title 'Normalized Average Runtime of Benchmarks ($3, Number of Colors:$4, Number of Iterations:$5)';
        set ylabel 'Normalized Runtime';
        set xlabel 'Benchmark(Alias)-Interference(Alias)';
        set key autotitle columnhead;
        set boxwidth 0.5;
        set style fill solid;
        set grid;
        set xtics rotate;
        set key noenhanced;
        set xtics noenhanced;
        set yrange [0:*];
        plot '$1' using 1:8:1:xticlabels(stringcolumn(2)) with boxes lc var, '' using 1:8:8 with labels offset char 0,1;
    "

    gnuplot -e "$gnu_command"

    if [ $? -ne 0 ]; then
        do_error_exit "Failed to plot histogram"
    fi

    return 0

}

# Step 2 - Chose the evaluation
echo "Choose one of the following evaluation modes"
echo "1) Reverse engineering of GPU (Get hidden details about the GPU)"
echo "2) Show results of micro benchmarks (CUDA/Rodinia applications)"
echo "3) Show results of macro benchmarks (Caffe application)"
echo "Enter option(1 or 2 or 3): " 
read evaluation_mode_number
echo ""

check_arg_between $evaluation_mode_number 1 3

case $evaluation_mode_number in
    1)
        EVAL_MODE=$EVAL_REVERSE
        configure_fgpu $FGPU_REVERSE_ENGINEERING
        
        init_fgpu

        echo "********************************"
        echo "Running reverse engineering code"
        echo "********************************"

        # Start the reverse engineering code
        # Print the histogram and treadline of DRAM bank access time
        # Histogram - 10K samples and bin size of 5 clock cycles
        # Also show the results of inteference on cachelines and DRAM banks
        hist_file=`mktemp`
        treadline_file=`mktemp`
        inteference_file=`mktemp`
        hist_outfile=`mktemp`
        treadline_outfile=`mktemp`
        inteference_outfile=`mktemp`
        hist_outfile+="_dram_bank_access_histogram.png"
        treadline_outfile+="_dram_bank_access_trendline.png"
        inteference_outfile+="_cache_and_dram_interference_experiments.png"

        # Volta GPU takes more samples to get meaningful plots as bigger memory size
        # This is only for plotting. For reverse engineering these inputs are not needed
        if [ "$IS_VOLTA" = "1" ]; then
            $BIN_PATH/$REVERSE_ENGINEERING_BINARY -n 20000 -s 5 -H $hist_file -T $treadline_file -I $inteference_file
        else
            $BIN_PATH/$REVERSE_ENGINEERING_BINARY -n 10000 -s 5 -H $hist_file -T $treadline_file -I $inteference_file
        fi

	if [ $? -ne 0 ]; then
            do_error_exit "Reverse engineering code failed"
        fi

        echo "***********************************************************************"
        echo "Saving plot of Treadline of DRAM Bank access time to $treadline_outfile"
        echo "***********************************************************************"
        $REVERSE_ENGINEERING_PATH/$REVERSE_ENGINEERING_PLOT -T=$treadline_file -t=$treadline_outfile
        if [ $? -ne 0 ]; then
            do_error_exit "Failed to plot trendline"
        fi
        echo "Open the file to see the plot"
        pause_for_user_input

        echo "***************************************************************"
        echo "Saving plot of Histogram of DRAM Bank access time $hist_outfile"
        echo "***************************************************************"
        $REVERSE_ENGINEERING_PATH/$REVERSE_ENGINEERING_PLOT -G=$hist_file -g=$hist_outfile
        if [ $? -ne 0 ]; then
            do_error_exit "Failed to plot histogram"
        fi
        echo "Open the file to see the plot"
        pause_for_user_input

        echo "************************************************************"
        echo "Saving plot of interference expermients $inteference_outfile"
        echo "************************************************************"
        $REVERSE_ENGINEERING_PATH/$REVERSE_ENGINEERING_PLOT -I=$inteference_file -i=$inteference_outfile
        if [ $? -ne 0 ]; then
            do_error_exit "Failed to plot inteference expermients result"
        fi
        echo "Open the file to see the plot"
        pause_for_user_input

        deinit_fgpu
        ;;

    2)
        EVAL_MODE=$EVAL_BENCHMARK

        echo "INFO: Benchmarks can run in different modes. The runtimes of benchmarks are normalized wrt to FGPU disabled mode."
        ask_and_configure_fgpu
        chosen_fgpu_mode=$FGPU_MODE_NAME

        num_iterations=1000
        echo "Enter the number of iterations of each benchmark. Average results are reported (Default $num_iterations):"
        read num_iterations
        echo ""

        if [ -z $num_iterations ]; then
            num_iterations=1000
        else
            check_arg_is_number $num_iterations
        fi

        num_colors=2

        if [ $IS_VOLTA -ne 0 ]; then
            echo "Enter the number of total partitions. Default is $num_colors (Valid options are 2,4,8):"
            read num_colors
            echo ""
            check_arg_is_number $num_colors

            if [ $num_colors -ne 2 ] && [ $num_colors -ne 4 ] && [ $num_colors -ne 8 ]; then
                do_error_exit "Invalid argument"
            fi

        else
            echo "GPU only supports 2 partitions. Using this value."
        fi

        cur_dir=`pwd`
        cd $BENCHMARK_PATH
        # Get list of benchmark applications
        list_benchmark=`$BENCHMARK_PATH/$BENCHMARK_SCRIPT -B`
        # Get list of inteference applications
        list_interference=`$BENCHMARK_PATH/$BENCHMARK_SCRIPT -I`
        cd $cur_dir

        # Print out
        echo ""
        echo "$list_benchmark"
        echo ""
        echo "$list_interference"
        echo ""

        declare -a benchmarks
        declare -a inteferences

        while read b; do
            benchmarks+=("$b")
        done < <(echo "$list_benchmark" | tail -n +2) # Omit the first line which is a explaining statement

        while read i; do
            inteferences+=("$i")
        done < <(echo "$list_interference" | tail -n +2)

        print_fgpu_mode

        declare -a baliases
        declare -a ialiases

        for ((i = 0; i < "${#benchmarks[@]}"; i++))
        do
            bname=${benchmarks[$i]}
            baliases+=("${default_aliases[$bname]}")
        done

        for ((i = 0; i < "${#inteferences[@]}"; i++))
        do
            iname=${inteferences[$i]}
            ialiases+=("${default_aliases[$iname]}")
        done

        run_all_benchmark benchmarks inteferences benchmarks inteferences baliases ialiases $num_colors $num_iterations "0"

        ;;
    3)
        EVAL_MODE=$EVAL_CAFFE

        echo "INFO: Benchmarks can run in different modes. The runtimes of benchmarks are normalized wrt to FGPU disabled mode."
        ask_and_configure_fgpu
        chosen_fgpu_mode=$FGPU_MODE_NAME

        compile_caffe

        num_iterations=1000
        echo "Enter the number of iterations of each benchmark. Average results are reported (Default $num_iterations):"
        read num_iterations
        echo ""

        if [ -z $num_iterations ]; then
            num_iterations=1000
        else
            check_arg_is_number $num_iterations
        fi

        num_colors=2

        if [ $IS_VOLTA -ne 0 ]; then
            echo "Enter the number of total partitions. Default is $num_colors (Valid options are 2,4,8):"
            read num_colors
            echo ""
            check_arg_is_number $num_colors

            if [ $num_colors -ne 2 ] && [ $num_colors -ne 4 ] && [ $num_colors -ne 8 ]; then
                do_error_exit "Invalid argument"
            fi

        else
            echo "GPU only supports 2 partitions. Using this value."
        fi

        # Use a caffe example application that has been ported to use FGPU
        declare -a bcmds
        declare -a icmds
        declare -a bnames
        declare -a inames
        declare -a baliases
        declare -a ialiases

        bcmd="$CAFFE_PATH/build/examples/cpp_classification/classification.bin                  \
                $CAFFE_PATH/models/bvlc_reference_caffenet/deploy.prototxt                      \
                $CAFFE_PATH/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel   \
                $CAFFE_PATH/data/ilsvrc12/imagenet_mean.binaryproto                             \
                $CAFFE_PATH/data/ilsvrc12/synset_words.txt                                      \
                $CAFFE_PATH/examples/images/cat.jpg"
        bname="Caffe_Image_Classification"
        balias="IC"

        bcmds+=("$bcmd")
        bnames+=("$bnames")
        baliases+=("$balias")

        # Add Caffe to list of intereference applications also
        icmds+=("$bcmd")
        inames+=("$bnames")
        ialiases+=("$balias")

        cur_dir=`pwd`
        cd $BENCHMARK_PATH
        # Get list of default inteference applications
        list_interference=`$BENCHMARK_PATH/$BENCHMARK_SCRIPT -I`
        cd $cur_dir

        # Print out
        echo ""
        echo "List of bencharks:"
        echo "$bname"
        echo ""
        echo "$list_interference"
        echo "$bname"
        echo ""

        while read i; do
            icmds+=($i)
            inames+=("$i")
            ialiases+=("${default_aliases[$i]}")
        done < <(echo "$list_interference" | tail -n +2)

        print_fgpu_mode

        run_all_benchmark bcmds icmds bnames inames baliases ialiases $num_colors $num_iterations "1"

        ;;
esac

echo "Reached end. Restart script to explore more options."
