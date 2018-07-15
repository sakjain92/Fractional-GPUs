#!/bin/sh

PATH="/sbin:/usr/sbin:$PATH"

BASE_LOG_FILENAME="nvidia-bug-report.log"

# check if gzip is present
GZIP_CMD=`which gzip 2> /dev/null | head -n 1`
if [ $? -eq 0 -a "$GZIP_CMD" ]; then
    GZIP_CMD="gzip -c"
else
    GZIP_CMD="cat"
fi

DPY="$DISPLAY"
[ "$DPY" ] || DPY=":0"

set_filename() {
    if [ "$GZIP_CMD" = "gzip -c" ]; then
        LOG_FILENAME="$BASE_LOG_FILENAME.gz"
        OLD_LOG_FILENAME="$BASE_LOG_FILENAME.old.gz"
    else
        LOG_FILENAME=$BASE_LOG_FILENAME
        OLD_LOG_FILENAME="$BASE_LOG_FILENAME.old"
    fi
}


if [ -d /proc/driver/nvidia -a ! -f /proc/driver/nvidia/version ]; then
    proc_module_dirs=`ls /proc/driver/nvidia/`
    module_names="nvidia-frontend"
    for instance in $proc_module_dirs; do
        module_names="$module_names nvidia$instance"
    done
else
    proc_module_dirs="."
    module_names="nvidia"
fi

usage_bug_report_message() {
    echo "Please include the '$LOG_FILENAME' log file when reporting"
    echo "your bug via the NVIDIA Linux forum (see devtalk.nvidia.com)"
    echo "or by sending email to 'linux-bugs@nvidia.com'."
}

usage() {
    echo ""
    echo "$(basename $0): NVIDIA Linux Graphics Driver bug reporting shell script."
    echo ""
    usage_bug_report_message
    echo ""
    echo "$(basename $0) [OPTION]..."
    echo "    -h / --help"
    echo "        Print this help output and exit."
    echo "    --output-file <file>"
    echo "        Write output to <file>. If gzip is available, the output file"
    echo "        will be automatically compressed, and \".gz\" will be appended"
    echo "        to the filename. Default: write to nvidia-bug-report.log(.gz)."
    echo "    --safe-mode"
    echo "        Disable some parts of the script that may hang the system."
    echo "    --extra-system-data"
    echo "        Enable additional data collection that may aid in the analysis"
    echo "        of certain classes of bugs. If running the script without the"
    echo "        --safe-mode option hangs the system, consider using this"
    echo "        option to help identify stuck kernel software."
    echo ""
}

NVIDIA_BUG_REPORT_CHANGE='$Change: 23400452 $'
NVIDIA_BUG_REPORT_VERSION=`echo "$NVIDIA_BUG_REPORT_CHANGE" | tr -c -d "[:digit:]"`

# Set the default filename so that it won't be empty in the usage message
set_filename

# Parse arguments: Optionally set output file, run in safe mode, include extra
# system data, or print help
BUG_REPORT_SAFE_MODE=0
BUG_REPORT_EXTRA_SYSTEM_DATA=0
SAVED_FLAGS=$@
while [ "$1" != "" ]; do
    case $1 in
        -o | --output-file )    if [ -z $2 ]; then
                                    usage
                                    exit 1
                                elif [ "$(echo "$2" | cut -c 1)" = "-" ]; then
                                    echo "Warning: Questionable filename"\
                                         "\"$2\": possible missing argument?"
                                fi
                                BASE_LOG_FILENAME="$2"
                                # override the default filename
                                set_filename
                                shift
                                ;;
        --safe-mode )           BUG_REPORT_SAFE_MODE=1
                                ;;
        --extra-system-data )   BUG_REPORT_EXTRA_SYSTEM_DATA=1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

#
# echo_metadata() - echo metadata of specified file
#

echo_metadata() {
    printf "*** ls: "
    /bin/ls -l --full-time "$1" 2> /dev/null

    if [ $? -ne 0 ]; then
        # Run dumb ls -l. We might not get one-second mtime granularity, but
        # that is probably okay.
        ls -l "$1" 2>&1
    fi
}


#
# append() - append the contents of the specified file to the log
#

append() {
    (
        echo "____________________________________________"
        echo ""

        if [ ! -f "$1" ]; then
            echo "*** $1 does not exist"
        elif [ ! -r "$1" ]; then
            echo "*** $1 is not readable"
        else
            echo "*** $1"
            echo_metadata "$1"
            cat  "$1"
        fi
        echo ""
    ) | $GZIP_CMD >> $LOG_FILENAME
}

#
# append_silent() - same as append(), but don't print anything
# if the file does not exist
#

append_silent() {
    (
        if [ -f "$1" -a -r "$1" ]; then
            echo "____________________________________________"
            echo ""
            echo "*** $1"
            echo_metadata "$1"
            cat  "$1"
            echo ""
        fi
    ) | $GZIP_CMD >> $LOG_FILENAME
}

#
# append_glob() - use the shell to expand a list of files, and invoke
# append() for each of them
#

append_glob() {
    for i in `ls $1 2> /dev/null;`; do
        append "$i"
    done
}

#
# append_file_or_dir_silent() - if $1 is a regular file, append it; otherwise,
# if $1 is a directory, append all files under it.  Don't print anything if the
# file does not exist.
#

append_file_or_dir_silent() {
    if [ -f "$1" ]; then
        append "$1"
    elif [ -d "$1" ]; then
        append_glob "$1/*"
    fi
}

#
# append_binary_file() - Encode a binary file into a ascii string format
# using 'base64' and append the contents output to the log file
#

append_binary_file() {
    (
        base64=`which base64 2> /dev/null | head -n 1`

        if [ $? -eq 0 -a -x "$base64" ]; then
                if [ -f "$1" -a -r "$1" ]; then
                    echo "____________________________________________"
                    echo ""
                    echo "base64 \"$1\""
                    echo ""
                    base64 "$1" 2> /dev/null
                    echo ""
                fi
        else
            echo "Skipping $1 output (base64 not found)"
            echo ""
        fi

    ) | $GZIP_CMD >> $LOG_FILENAME
}

#
# append_command() - append the output of the specified command to the log
#

append_command() {
    if [ -n "$1" ]; then
        echo "$1"
        echo ""
        $1
        echo ""
    fi
}

#
# search_string_in_logs() - search for string $2 in log file $1
#

search_string_in_logs() {
    if [ -f "$1" ]; then
        echo ""
        if [ -r "$1" ]; then
            echo "  $1:"
            grep $2 "$1" 2> /dev/null
            return 0
        else
            echo "$1 is not readable"
        fi
    fi
    return 1
}

#
# Start of script
#


# check that we are root (needed for `lspci -vxxx` and potentially for
# accessing kernel log files)

if [ `id -u` -ne 0 ]; then
    echo "ERROR: Please run $(basename $0) as root."
    exit 1
fi


# move any old log file (zipped) out of the way

if [ -f $LOG_FILENAME ]; then
    mv $LOG_FILENAME $OLD_LOG_FILENAME
fi


# make sure what we can write to the log file

touch $LOG_FILENAME 2> /dev/null

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Working directory is not writable; please cd to a directory"
    echo "       where you have write permission so that the $LOG_FILENAME"
    echo "       file can be written."
    echo
    exit 1
fi


# print a start message to stdout

echo ""
echo "nvidia-bug-report.sh will now collect information about your"
echo "system and create the file '$LOG_FILENAME' in the current"
echo "directory.  It may take several seconds to run.  In some"
echo "cases, it may hang trying to capture data generated dynamically"
echo "by the Linux kernel and/or the NVIDIA kernel module.  While"
echo "the bug report log file will be incomplete if this happens, it"
echo "may still contain enough data to diagnose your problem." 
echo ""
usage_bug_report_message
echo ""
echo -n "Running $(basename $0)...";


# print prologue to the log file

(
    echo "____________________________________________"
    echo ""
    echo "Start of NVIDIA bug report log file.  Please include this file, along"
    echo "with a detailed description of your problem, when reporting a graphics"
    echo "driver bug via the NVIDIA Linux forum (see devtalk.nvidia.com)"
    echo "or by sending email to 'linux-bugs@nvidia.com'."
    echo ""
    echo "nvidia-bug-report.sh Version: $NVIDIA_BUG_REPORT_VERSION"
    echo ""
    echo "Date: `date`"
    echo "uname: `uname -a`"
    echo "command line flags: $SAVED_FLAGS"
    echo ""
) | $GZIP_CMD >> $LOG_FILENAME

# append OPAL (IBM POWER system firmware) messages

append_silent "/sys/firmware/opal/msglog"

# append useful files

append "/etc/issue"

append_silent "/etc/redhat-release"
append_silent "/etc/redhat_version"
append_silent "/etc/fedora-release"
append_silent "/etc/slackware-release"
append_silent "/etc/slackware-version"
append_silent "/etc/debian_release"
append_silent "/etc/debian_version"
append_silent "/etc/mandrake-release"
append_silent "/etc/yellowdog-release"
append_silent "/etc/sun-release"
append_silent "/etc/release"
append_silent "/etc/gentoo-release"


append "/var/log/nvidia-installer.log"
append_silent "/var/log/nvidia-uninstall.log"

# find and append all make.log files in /var/lib/dkms for module nvidia
if [ -d "/var/lib/dkms/nvidia" ]; then
    for log in `find "/var/lib/dkms/nvidia" -name "make.log"`; do
        append $log
    done
fi

# use systemd's journalctl to capture X logs where applicable

journalctl=`which journalctl 2> /dev/null | head -n 1`

if [ $? -eq 0 -a -x "$journalctl" ]; then
    for comm in Xorg Xorg.bin X; do
        for boot in -0 -1 -2; do
            if journalctl -b $boot -n 1 _COMM=$comm >/dev/null 2>&1; then
                (
                    echo "____________________________________________"
                    echo ""
                    echo "journalctl -b $boot _COMM=$comm"
                    echo ""
                    journalctl -b $boot _COMM=$comm
                    echo ""
                ) 2> /dev/null | $GZIP_CMD >> $LOG_FILENAME
            fi
        done
    done
fi

# append the X log; also, extract the config file named in the X log
# and append it; look for X log files with names of the form:
# /var/log/{XFree86,Xorg}.{0,1,2,3,4,5,6,7}.{log,log.old}

xconfig_file_list=
svp_config_file_list=
NEW_LINE="
"

for log_basename in /var/log/XFree86 /var/log/Xorg; do
    for i in 0 1 2 3 4 5 6 7; do
        for log_suffix in log log.old; do
            log_filename="${log_basename}.${i}.${log_suffix}"
            append_silent "${log_filename}"

            # look for the X configuration files/directories referenced by this X log
            if [ -f ${log_filename} -a -r ${log_filename} ]; then
                config_file=`grep "Using config file" ${log_filename} | cut -f 2 -d \"`
                config_dir=`grep "Using config directory" ${log_filename} | cut -f 2 -d \"`
                sys_config_dir=`grep "Using system config directory" ${log_filename} | cut -f 2 -d \"`
                for j in "$config_file" "$config_dir" "$sys_config_dir"; do
                    if [ "$j" ]; then
                        # multiple of the logs we find above might reference the
                        # same X configuration file; keep a list of which X
                        # configuration files we find, and only append X
                        # configuration files we have not already appended
                        echo "${xconfig_file_list}" | grep ":${j}:" > /dev/null
                        if [ "$?" != "0" ]; then
                            xconfig_file_list="${xconfig_file_list}:${j}:"
                            if [ -d "$j" ]; then
                                append_glob "$j/*.conf"
                            else
                                append "$j"
                            fi
                        fi
                    fi
                done

                # append NVIDIA 3D Vision Pro configuration settings
                svp_conf_files=`grep "Option \"3DVisionProConfigFile\"" ${log_filename} | cut -f 4 -d \"`
                if [ "${svp_conf_files}" ]; then
                    OLD_IFS="$IFS"
                    IFS=$NEW_LINE
                    for svp_file in ${svp_conf_files}; do
                        IFS="$OLD_IFS"
                        echo "${svp_config_file_list}" | grep ":${svp_file}:" > /dev/null
                        if [ "$?" != "0" ]; then
                            svp_config_file_list="${svp_config_file_list}:${svp_file}:"
                            append_binary_file "${svp_file}"
                        fi
                        IFS=$NEW_LINE
                    done
                    IFS="$OLD_IFS"
                fi
            fi

        done
    done
done

# Append any config files found in home directories
cat /etc/passwd \
    | cut -d : -f 6 \
    | sort | uniq \
    | while read DIR; do
        append_file_or_dir_silent "$DIR/.nv/nvidia-application-profiles-rc"
        append_file_or_dir_silent "$DIR/.nv/nvidia-application-profiles-rc.backup"
        append_file_or_dir_silent "$DIR/.nv/nvidia-application-profiles-rc.d"
        append_file_or_dir_silent "$DIR/.nv/nvidia-application-profiles-rc.d.backup"
        append_silent "$DIR/.nv/nvidia-application-profile-globals-rc"
        append_silent "$DIR/.nv/nvidia-application-profile-globals-rc.backup"
        append_silent "$DIR/.nvidia-settings-rc"
    done

# Capture global app profile configs
append_file_or_dir_silent "/etc/nvidia/nvidia-application-profiles-rc"
append_file_or_dir_silent "/etc/nvidia/nvidia-application-profiles-rc.d"
append_file_or_dir_silent /usr/share/nvidia/nvidia-application-profiles-*-rc


# append ldd info

(
    echo "____________________________________________"
    echo ""

    glxinfo=`which glxinfo 2> /dev/null | head -n 1`

    if [ $? -eq 0 -a -x "$glxinfo" ]; then
        echo "ldd $glxinfo"
        echo ""
        ldd $glxinfo 2> /dev/null
        echo ""
    else
        echo "Skipping ldd output (glxinfo not found)"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# append Vulkan info

(
    echo "____________________________________________"
    echo ""

    vkinfo=`ldconfig -N -v -p 2> /dev/null | /bin/grep libvulkan.so.1 | awk 'NF>1{print $NF}'`

    if [ $? -eq 0 -a -n "$vkinfo" ]; then
        echo "Found Vulkan loader(s):"
        readlink -f ${vkinfo} 2> /dev/null
        echo ""
        # See https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/loader/LoaderAndLayerInterface.md
        echo "Listing common ICD paths:"
        ls -d /usr/local/etc/vulkan/icd.d/* 2> /dev/null
        ls -d /usr/local/share/vulkan/icd.d/* 2> /dev/null
        ls -d /etc/vulkan/icd.d/* 2> /dev/null
        ls -d /usr/share/vulkan/icd.d/* 2> /dev/null
        echo ""
    else
        echo "Vulkan loader not found"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# lspci information

(
    echo "____________________________________________"
    echo ""

    lspci=`which lspci 2> /dev/null | head -n 1`

    if [ $? -eq 0 -a -x "$lspci" ]; then
        # Capture NVIDIA devices
        echo "$lspci -d \"10de:*\" -v -xxx"
        echo ""
        $lspci -d "10de:*" -v -xxx 2> /dev/null
        echo ""
        echo "____________________________________________"
        echo ""
        # Capture PLX devices
        echo "$lspci -d \"10b5:*\" -v -xxx"
        echo ""
        $lspci -d "10b5:*" -v -xxx 2> /dev/null
        echo ""
        echo "____________________________________________"
        echo ""
        echo "$lspci -t"
        echo ""
        $lspci -t 2> /dev/null
        echo ""
        echo "____________________________________________"
        echo ""
        echo "$lspci -nn"
        echo ""
        $lspci -nn 2> /dev/null
    else
        echo "Skipping lspci output (lspci not found)"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# lsusb information

(
    echo "____________________________________________"
    echo ""

    lsusb=`which lsusb 2> /dev/null | head -n 1`

    if [ $? -eq 0 -a -x "$lsusb" ]; then
        echo "$lsusb"
        echo ""
        $lsusb 2> /dev/null
        echo ""
    else
        echo "Skipping lsusb output (lsusb not found)"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# dmidecode

(
    echo "____________________________________________"
    echo ""

    dmidecode=`which dmidecode 2> /dev/null | head -n 1`

    if [ $? -eq 0 -a -x "$dmidecode" ]; then
        echo "$dmidecode"
        echo ""
        $dmidecode 2> /dev/null
        echo ""
    else
        echo "Skipping dmidecode output (dmidecode not found)"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# module version magic

(
    echo "____________________________________________"
    echo ""

    modinfo=`which modinfo 2> /dev/null | head -n 1`

    if [ $? -eq 0 -a -x "$modinfo" ]; then
        for name in $module_names; do
            echo "$modinfo $name | grep vermagic"
            echo ""
            ( $modinfo $name | grep vermagic ) 2> /dev/null
            echo ""
        done
    else
        echo "Skipping modinfo output (modinfo not found)"
        echo ""
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# get any relevant kernel messages

(
    echo "____________________________________________"
    echo ""
    echo "Scanning kernel log files for NVIDIA kernel messages:"

    grep_args="-e NVRM -e nvidia-"
    logfound=0
    search_string_in_logs /var/log/messages "$grep_args" && logfound=1
    search_string_in_logs /var/log/kernel.log "$grep_args" && logfound=1
    search_string_in_logs /var/log/dmesg "$grep_args" && logfound=1

    journalctl=`which journalctl 2> /dev/null | head -n 1`
    if [ $? -eq 0 -a -x "$journalctl" ]; then
        logfound=1
        nvrmfound=0

        for boot in -0 -1 -2; do
            if (journalctl -b $boot | grep ${grep_args}) > /dev/null 2>&1; then
                echo ""
                echo "  journalctl -b $boot:"
                (journalctl -b $boot | grep ${grep_args}) 2> /dev/null
                nvrmfound=1
            fi
        done

        if [ $nvrmfound -eq 0 ]; then
            echo ""
            echo "No NVIDIA kernel messages found in recent systemd journal entries."
        fi
    fi

    if [ $logfound -eq 0 ]; then
        echo ""
        echo "No suitable log found."
    fi

    echo ""
) | $GZIP_CMD >> $LOG_FILENAME


# If extra data collection is enabled, dump all active CPU backtraces to be
# picked up in dmesg
if [ $BUG_REPORT_EXTRA_SYSTEM_DATA -ne 0 ]; then
    (
        echo "____________________________________________"
        echo ""
        echo "Triggering SysRq backtrace on active CPUs (see dmesg output)"
        sysrq_enabled=`cat /proc/sys/kernel/sysrq`
        if [ "$sysrq_enabled" -ne "1" ]; then
            echo 1 > /proc/sys/kernel/sysrq
        fi
    
        echo l > /proc/sysrq-trigger
    
        if [ "$sysrq_enabled" -ne "1" ]; then
            echo $sysrq_enabled > /proc/sys/kernel/sysrq
        fi
    ) | $GZIP_CMD >> $LOG_FILENAME
fi

# append dmesg output

(
    echo "____________________________________________"
    echo ""
    echo "dmesg:"
    echo ""
    dmesg 2> /dev/null
) | $GZIP_CMD >> $LOG_FILENAME

# print gcc & g++ version info

(
    which gcc >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "____________________________________________"
        echo ""
        gcc -v 2>&1
    fi

    which g++ >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "____________________________________________"
        echo ""
        g++ -v 2>&1
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# In case of failure, if xset returns with delay, we print the
# message from check "$?" & if it returns error immediately before kill,
# we directly write the error to the log file.

(
    echo "____________________________________________"
    echo ""
    echo "xset -q:"
    echo ""

    xset -q 2>&1 & sleep 1 ; kill -9 $! > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        # The xset process is still there.
        echo "xset could not connect to an X server"
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# In case of failure, if nvidia-settings returns with delay, we print the
# message from check "$?" & if it returns error immediately before kill,
# we directly write the error to the log file.

(
    echo "____________________________________________"
    echo ""
    echo "nvidia-settings -q all:"
    echo ""

    DISPLAY= nvidia-settings -c "$DPY" -q all 2>&1 & sleep 1 ; kill -9 $! > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        # The nvidia-settings process is still there.
        echo "nvidia-settings could not connect to an X server"
    fi
) | $GZIP_CMD >> $LOG_FILENAME

# In case of failure, if xrandr returns with delay, we print the
# message from check "$?" & if it returns error immediately before kill,
# we directly write the error to the log file.

(
    if [ -x "`which xrandr 2>/dev/null`" ] ; then
         echo "____________________________________________"
         echo ""
         echo "xrandr --verbose:"
         echo ""

         xrandr -display $DPY --verbose 2>&1 & sleep 1 ; kill -9 $! > /dev/null 2>&1
         if [ $? -eq 0 ]; then
             # The xrandr process is still there.
             echo "xrandr could not connect to an X server"
         fi
    else
        echo "Skipping xrandr output (xrandr not found)"
    fi
) | $GZIP_CMD >> $LOG_FILENAME

(
    if [ -x "`which xprop 2>/dev/null`" ] ; then
        echo "____________________________________________"
        echo ""
        echo "Running window manager properties:"
        echo ""

        TMP=`xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null & sleep 1 ; kill -9 $! > /dev/null 2>&1`
        WINDOW=`echo $TMP | grep -o '0x[0-9a-z]\+'`
        if [ "$WINDOW" ]; then
            xprop -id "$WINDOW" 2>&1 & sleep 1 ; kill -9 $! > /dev/null 2>&1
        else
            echo "Unable to detect window manager properties"
        fi
    fi
) | $GZIP_CMD >> $LOG_FILENAME

sync > /dev/null 2>&1
sync > /dev/null 2>&1

# append useful /proc files

append "/proc/cmdline"
append "/proc/cpuinfo"
append "/proc/interrupts"
append "/proc/meminfo"
append "/proc/modules"
append "/proc/version"
append "/proc/pci"
append "/proc/iomem"
append "/proc/mtrr"

for subdir in $proc_module_dirs; do
    append "/proc/driver/nvidia/$subdir/version"
    for GPU in `ls /proc/driver/nvidia/$subdir/gpus/`; do
        append "/proc/driver/nvidia/$subdir/gpus/$GPU/information"
        append "/proc/driver/nvidia/$subdir/gpus/$GPU/registry"
    done
    append_glob "/proc/driver/nvidia/$subdir/warnings/*"
    append "/proc/driver/nvidia/$subdir/params"
    append "/proc/driver/nvidia/$subdir/registry"
done

append_glob "/proc/acpi/video/*/*/info"

append "/proc/asound/cards"
append "/proc/asound/pcm"
append "/proc/asound/modules"
append "/proc/asound/devices"
append "/proc/asound/version"
append "/proc/asound/timers"
append "/proc/asound/hwdep"

for CARD in /proc/asound/card[0-9]*; do
    for CODEC in $CARD/codec*; do
        [ -d $CODEC ] && append_glob "$CODEC/*"
        [ -f $CODEC ] && append "$CODEC"
    done
    for ELD in $CARD/eld*; do
        [ -f $ELD ] && append "$ELD"
    done
done

# List the mapping of DRM drivers to DRM device files
(
    echo "____________________________________________"
    echo ""

    for CARD in /sys/class/drm/*/device/driver ; do
        echo_metadata $CARD
    done

    echo ""

) | $GZIP_CMD >> $LOG_FILENAME

# disable these when safemode is requested
if [ $BUG_REPORT_SAFE_MODE -eq 0 ]; then

    # Inform user about safemode
    (
        echo ""
        echo ""
        echo "If the bug report script hangs after this point consider running with"
        echo "--safe-mode and --extra-system-data command line arguments."
        echo ""
    )
    
    # nvidia-smi

    (
        echo "____________________________________________"
        echo ""

        nvidia_smi=`which nvidia-smi 2> /dev/null | head -n 1`

        if [ $? -eq 0 -a -x "$nvidia_smi" ]; then
            for instance in $proc_module_dirs; do
                if [ $instance != '.' ]; then
                    export __RM_MODULE_INSTANCE=$instance
                    echo "NVIDIA Kernel module instance $instance"
                fi
                append_command "$nvidia_smi --query"
                append_command "$nvidia_smi --query --unit"
                unset __RM_MODULE_INSTANCE
            done
        else
            echo "Skipping nvidia-smi output (nvidia-smi not found)"
            echo ""
        fi
    ) | $GZIP_CMD >> $LOG_FILENAME

    # nvidia-debugdump

    (
        echo "____________________________________________"
        echo ""

        nvidia_debugdump=`which nvidia-debugdump 2> /dev/null | head -n 1`

        if [ $? -eq 0 -a -x "$nvidia_debugdump" ]; then
        
            base64=`which base64 2> /dev/null | head -n 1`
            
            if [ $? -eq 0 -a -x "$base64" ]; then
                # make sure what we can write to the temp file
                
                NVDD_TEMP_FILENAME="nvidia-debugdump-temp$$.log"
                
                touch $NVDD_TEMP_FILENAME 2> /dev/null

                if [ $? -ne 0 ]; then
                    echo "Skipping nvidia-debugdump output (can't create temp file $NVDD_TEMP_FILENAME)"
                    echo ""
                    # don't fail here, continue
                else
                    for instance in $proc_module_dirs; do
                        if [ $instance != '.' ]; then
                            export __RM_MODULE_INSTANCE=$instance
                            echo "NVIDIA Kernel module instance $instance"
                        fi

                        echo "$nvidia_debugdump -D"
                        echo ""
                        $nvidia_debugdump -D -f $NVDD_TEMP_FILENAME 2> /dev/null
                        $base64 $NVDD_TEMP_FILENAME 2> /dev/null
                        echo ""

                        # remove the temporary file when complete
                        rm $NVDD_TEMP_FILENAME 2> /dev/null
                        unset __RM_MODULE_INSTANCE
                    done
                fi
            else
                echo "Skipping nvidia-debugdump output (base64 not found)"
                echo ""
            fi
        else
            echo "Skipping nvidia-debugdump output (nvidia-debugdump not found)"
            echo ""
        fi
    ) | $GZIP_CMD >> $LOG_FILENAME

else
    (
        echo "Skipping nvidia-smi and nvidia-debugdump due to --safe-mode argument."
        echo ""
    ) | $GZIP_CMD >> $LOG_FILENAME
fi

(
    echo "____________________________________________"

    # print epilogue to log file

    echo ""
    echo "End of NVIDIA bug report log file."
) | $GZIP_CMD >> $LOG_FILENAME

# Done

echo " complete."
echo ""
