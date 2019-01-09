#!/bin/bash

COMMON_SCRIPT=../scripts/common.sh

if [ ! -f $COMMON_SCRIPT ]; then
    echo "Run this script from \$PROJ_DIR/reverse_engineering folder"
fi

source $COMMON_SCRIPT

print_usage() {
    echo "This file uses gnuplot to plot different results of reverse engineering"
    echo "Usage:"
    echo "./plot_reverse_engineering.sh"
    echo "Optional arguments:"
    echo "-h, --help Shows this usage."
    echo "-G=<filename>, --HistogramInput=<filename> Input file for histogram of DRAM bank access"
    echo "-I=<filename>, --InterferenceInput=<filename> Input file for interference on DRM Bank/cacheline"
    echo "-T=<filename>, --TrendlineInput=<filename> Input file for trendline of DRAM Bank access "
    echo "-g=<filename>, --HistogramOutput=<filename> Output file for plot of histogram of DRAM bank access"
    echo "-i=<filename>, --InterferenceOutput=<filename> Output file for plot of interference on DRM Bank/cacheline"
    echo "-t=<filename>, --TrendlineOutput=<filename> Output file for plot of trendline of DRAM Bank access "

}

# Check if gnuplot exists?
check_if_installed "gnuplot"

unset histogram_input
unset trendline_input
unset interfence_input
unset histogram_output
unset trendline_output
unset interfence_output

parse_input_args() {

    for i in "$@"
    do
        case $i in
        -h*|--help*)
            print_usage
            exit 1
            ;;

        -G=*|--HistogramInput=*)
            histogram_input="${i#*=}"
            check_file_exists $histogram
            shift # past argument=value
            ;;
        
        -I=*|--InterferenceInput=*)
            interfence_input="${i#*=}"
            check_file_exists $interfence
            shift # past argument=value
            ;;

        -T=*|--TrendlineInput=*)
            trendline_input="${i#*=}"
            check_file_exists $trendline
            shift # past argument=value
            ;;

        -g=*|--HistogramOutput=*)
            histogram_output="${i#*=}"
            check_file_exists $histogram
            shift # past argument=value
            ;;
        
        -i=*|--InterferenceOutput=*)
            interfence_output="${i#*=}"
            check_file_exists $interfence
            shift # past argument=value
            ;;

        -t=*|--TrendlineOutput=*)
            trendline_output="${i#*=}"
            check_file_exists $trendline
            shift # past argument=value
            ;;

        *)
            # unknown option
            print_usage
            exit 1
            ;;
        esac
    done
}

# Saves gnuplot to a file
# First argument is gnu command
# Second argument is output filename
save_plot() {
    gnu_command="
    set term 'pngcairo' size 1280,960;
    set output '$2';
    $1
    "

    gnuplot -e "$gnu_command"

    return 0
}

# Plots histogram.
# First argument is the filename of input file
# Second argument optionally is the output file
plot_histogram() {

    # Sum of count in histogram is sample size
    sample_size=`cat $1 | tail -n +2 | cut -f2 | paste -sd+ | bc`

    gnu_command="
        set title 'Histogram of DRAM Bank Access (Sample Size:$sample_size)';
        set ylabel 'Count of Histogram';
        set xlabel 'StartTime(GPU cycles) - EndTime(GPU cycles) Range';
        set key autotitle columnhead;
        set boxwidth 1;
        set style fill solid;
        set grid;
        set xtics font ',8';
        plot '$1' using 1:2:xticlabels(stringcolumn(1)) with boxes, '' using 1:2:2 with labels offset char 0,1;
    "

    if [ "$#" -ne 1 ]; then
        save_plot "$gnu_command" $2
    else
        gnuplot -p -e "$gnu_command"
    fi

    return 0
}

# Plots trendline.
# First argument is the filename of input file
# Second argument optionally is the output file
plot_trendline() {
    
    # Get the primary address - First word of second line
    primary_addr=`cat $1 | head -n 2 | tail -1 | awk '{print $1;}'`
    # First line is the legend
    sample_size=`wc -l $1 | awk '{print $1;}'`
    sample_size=$((sample_size-1))


    gnu_command="
        set title 'Trendline of Access Time to DRAM Bank (Primary Address:$primary_addr, Sample Size:$sample_size)';
        set ylabel 'Access Time (GPU cycles)';
        set xlabel 'Secondary Address';
        set format x '0x%x';
        set key autotitle columnhead;
        set grid;
        plot '$1' using 2:3 with lines, '$1' using 2:4 with line;
    "
    
    if [ "$#" -ne 1 ]; then
        save_plot "$gnu_command" $2
    else
        gnuplot -p -e "$gnu_command"
    fi

    return 0

}

# Plots interfence.
# First argument is the filename of input file
# Second argument optionally is the output file
plot_interference() {

    gnu_command="
        set title 'Interference Experiments Result';
        set ylabel 'Access Time (GPU cycles)';
        set xlabel 'Number of Total Threads';
        set key autotitle columnhead;
        set grid;
        plot for [col=2:6]'$1' using 1:col with linespoints;
    "

    if [ "$#" -ne 1 ]; then
        save_plot "$gnu_command" $2
    else
        gnuplot -p -e "$gnu_command"
    fi

    return 0
}

parse_input_args $@

if [ ! -z  $histogram_input ]; then
    plot_histogram $histogram_input $histogram_output
fi

if [ ! -z $trendline_input ]; then
    plot_trendline $trendline_input $trendline_output
fi

if [ ! -z $interfence_input ]; then
    plot_interference $interfence_input $interfence_output
fi

exit 0
