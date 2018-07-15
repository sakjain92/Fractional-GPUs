#! /bin/sh
#
# makeself 1.6.0-nv2
#
# $Id: makeself.sh,v 1.22 2002/04/03 08:10:25 megastep Exp $
#
# Utility to create self-extracting tar.gz archives.
# The resulting archive is a file holding the tar.gz archive with
# a small Shell script stub that uncompresses the archive to a temporary
# directory and then executes a given script from withing that directory.
#
# Makeself home page: http://www.megastep.org/makeself/
#
# Version history :
# - 1.0 : Initial public release
# - 1.1 : The archive can be passed parameters that will be passed on to
#         the embedded script, thanks to John C. Quillan
# - 1.2 : Package distribution, bzip2 compression, more command line options,
#         support for non-temporary archives. Ideas thanks to Francois Petitjean
# - 1.3 : More patches from Bjarni R. Einarsson and Francois Petitjean:
#         Support for no compression (--nocomp), script is no longer mandatory,
#         automatic launch in an xterm, optional verbose output, and -target 
#         archive option to indicate where to extract the files.
# - 1.4 : Improved UNIX compatibility (Francois Petitjean)
#         Automatic integrity checking, support of LSM files (Francois Petitjean)
# - 1.5 : Many bugfixes. Optionally disable xterm spawning.
# - 1.5.1 : More bugfixes, added archive options -list and -check.
# - 1.5.2 : Cosmetic changes to inform the user of what's going on with big 
#           archives (Quake III demo)
# - 1.5.3 : Check for validity of the DISPLAY variable before launching an xterm.
#           More verbosity in xterms and check for embedded command's return value.
#           Bugfix for Debian 2.0 systems that have a different "print" command.
# - 1.5.4 : Many bugfixes. Print out a message if the extraction failed.
# - 1.5.5 : More bugfixes. Added support for SETUP_NOCHECK environment variable to
#           bypass checksum verification of archives.
# - 1.6.0 : Compute MD5 checksums with the md5sum command (patch from Ryan Gordon)
# - 1.6.0-nv : Patched for use by NVIDIA.
# - 1.6.0-nv2 : Added support for xz compression and embedding a decompressor in
#               the archive.
#
# (C) 1998-2001 by Stéphane Peter <megastep@megastep.org>
# (C) 2012 NVIDIA Corporation
#
# This software is released under the terms of the GNU GPL
# Please read the license at http://www.gnu.org/copyleft/gpl.html
#
VERSION=1.6.0-nv2
COMPRESS=gzip
COMPRESS_LEVEL=9
COMPRESS_CMD="$COMPRESS -c"
UNCOMPRESS_CMD="gzip -cd"
KEEP=n
ADD_THIS_KERNEL=n
APPLY_PATCH=n
CURRENT=n
TAR_ARGS=cf
PRINT_HELP=n
VERSION_STRING=0
PKG_VERSION=0
TARGET_OS="Unknown"
TARGET_ARCH="Unknown"
HELP_SCRIPT=
SILENT=n
TAR_DEREFERENCE_SYMLINKS=n
THREADS=
EMBED_DECOMPRESS=

# parse the params portion of the commandline; this is done until we
# find an argument that names a directory

while [ ! -d $1 ]; do
    case $1 in
        "--version")
            echo "Makeself version $VERSION"
            exit 0
            ;;
        "--bzip2")
            if which bzip2 2>&1 > /dev/null; then
                COMPRESS=bzip2
                COMPRESS_CMD="$COMPRESS -c"
                UNCOMPRESS_CMD="$COMPRESS -d"
            else
                echo "Unable to locate the bzip2 program in your \$PATH." >&2
                exit 1
            fi
            ;;
        "--xz")
            if which xz 2>&1 > /dev/null; then
                COMPRESS=xz
                # Use the default compression level for xz, since increasing it
                # increases memory usage on the decompress side.
                [ $COMPRESS_LEVEL = 9 ] && COMPRESS_LEVEL=6
                # Disable integrity checks: xzminicomp can't handle the headers
                # they add to the xz archive.
                COMPRESS_CMD="$COMPRESS -c -C none"
                UNCOMPRESS_CMD="$COMPRESS -d"
            else
                echo "Unable to locate the xz program in your \$PATH." >&2
                exit 1
            fi
            ;;
        "--nocomp")
            COMPRESS=none; COMPRESS_CMD=cat; UNCOMPRESS_CMD=cat;
            ;;
        "--fastcomp")
            COMPRESS_LEVEL=1
            ;;
        "--threads")
            shift 1
            THREADS=$1
            ;;
        "--notemp")
            KEEP=y
            ;;
        "--current")
            CURRENT=y
            ;;
        "--follow")
            TAR_ARGS=cvfh
            ;;
        "--lsm")
            shift 1
            lsm_file=$1
            [ -r $lsm_file ] || {
                echo "can't read LSM file " $lsm_file ;
                lsm_file="no_LSM";
            }
            ;;
        "--version-string")
            shift 1
            VERSION_STRING=$1
            ;;
        "--pkg-version")
            shift 1
            PKG_VERSION=$1
            ;;
        "--target-os")
            shift 1
            TARGET_OS=$1
            ;;
        "--target-arch")
            shift 1
            TARGET_ARCH=$1
            ;;
        "--pkg-history")
            shift 1
            pkg_history_file=$1
            if [ ! -r $pkg_history_file ]; then
                echo "can't read pkg-history file $pkg_history_file" ;
                pkg_history_file=
            fi
            ;;
        "--help-script")
            shift 1
            HELP_SCRIPT=$1
            ;;
        "--silent")
            SILENT=y;
            ;;
        "--tar-dereference-symlinks")
            TAR_DEREFERENCE_SYMLINKS=y;
            ;;
        "--embed-decompress")
            shift 1
            EMBED_DECOMPRESS="$1"
            ;;
        *)
            echo "unrecognized option '$1'"
            PRINT_HELP=y;
            break
            ;;
    esac
    shift 1
done

if [ "$SILENT" = "n" ]; then
    TAR_ARGS="${TAR_ARGS}v"
fi

if [ "$TAR_DEREFERENCE_SYMLINKS" = "y" ]; then
    TAR_ARGS="${TAR_ARGS}h"
fi

if [ "$KEEP" = "n" -a $# = 3 ]; then
    echo "Making a temporary archive with no embedded command does not make sense!"
    echo
    shift 1 # To force the command usage
fi

if [ -n "$THREADS" ]; then
    case "$COMPRESS" in
        gzip)
            if which pigz 2>&1 > /dev/null; then
                COMPRESS_CMD="pigz -p$THREADS -c"
            else
                echo "Unable to locate the pigz program in your \$PATH." >&2
                exit 1
            fi
            ;;
        xz)
            COMPRESS_CMD="$COMPRESS_CMD --threads=$THREADS"
            ;;
        *)
            echo "--threads not supported with compression method $COMPRESS" >&2
            exit 1
            ;;
    esac
fi

[ "$COMPRESS" != "none" ] && COMPRESS_CMD="$COMPRESS_CMD -$COMPRESS_LEVEL"

if [ "$COMPRESS" = "xz" -a -z "$EMBED_DECOMPRESS" ]; then
    echo "xz is not installed on all target platforms, so --embed-decompress " >&2
    echo "is required" >&2
    exit 1
fi

if [ -n "$EMBED_DECOMPRESS" -a ! -f "$EMBED_DECOMPRESS" ]; then
    echo "Can't find decompression program '$EMBED_DECOMPRESS' to embed" >&2
    exit 1
fi

if [ $# -lt 3 -o $PRINT_HELP = "y" ]; then
    echo "Usage: $0 [params] archive_dir file_name label [startup_script] [args]"
    echo "params can be any of these :"
    echo ""
    echo "    --version  : Print out Makeself version number and exit"
    echo "    --bzip2    : Compress using bzip2 instead of gzip"
    echo "    --xz       : Compress using xz instead of gzip"
    echo "    --nocomp   : Do not compress the data"
    echo "    --notemp   : The archive will create archive_dir in the"
    echo "                 current directory and uncompress in ./archive_dir"
    echo "    --current  : Used with --notemp, files will be extracted to the"
    echo "                 current directory."
    echo "    --follow   : Follow the symlinks in the archive"
    echo "    --lsm file : LSM file describing the package"
    echo "    --embed-decompress file"
    echo "               : Embed a decompression program in the archive"
    echo ""
    echo "Do not forget to give a fully qualified startup script name"
    echo "(i.e. with a ./ prefix if inside the archive)."
    exit 1
fi

archdir=$1
archname=$2
label=$3
script=$4

# Infer the script to use to get help output, if it was not specified
[ "$HELP_SCRIPT" ] || HELP_SCRIPT=$archdir/$script

# We don't really want to create an absolute directory...
archdirname=`basename "$1"`
USIZE=`du -Lks --apparent-size $archdir | cut -f1`
DATE=`date`

# The following is the shell script stub code
echo '#! /bin/sh' > $archname
echo 'CRCsum=0000000000' >> $archname
echo 'MD5=00000000000000000000000000000000' >> $archname
echo skip=__SKIP__ >> $archname
echo skip_decompress=__SKIP_DECOMPRESS__ >> $archname
echo size_decompress=__SIZE_DECOMPRESS__ >> $archname
# echo lsm=\"$lsm_contents\" >> $archname
echo label=\"$label\" >> $archname
echo version_string=$VERSION_STRING >> $archname
echo pkg_version=$PKG_VERSION >> $archname
echo script=$script >> $archname
[ x"$4" = x ] || shift 1
echo targetdir="$archdirname" >>$archname
shift 3
echo scriptargs=\"$*\" >> $archname
echo "keep=$KEEP" >> $archname
echo "add_this_kernel=$ADD_THIS_KERNEL" >> $archname
echo "apply_patch=$APPLY_PATCH" >> $archname
echo "TMPROOT=\${TMPDIR:=/tmp}" >> $archname
echo "TARGET_OS=\"$TARGET_OS\"" >> $archname
echo "TARGET_ARCH=\"$TARGET_ARCH\"" >> $archname

# output a banner

cat <<- EODF >> $archname

	#
	# $label
	# Generated by Makeself $VERSION
	# Do not edit by hand.

	# NVIDIA Driver Installation .run file
	#
	# If you were trying to download this file through a web browser, and
	# instead are seeing this, please click your browser's back button,
	# left click on the link, and select "Save as..." (or do whatever is
	# appropriate for your web browser to download a file, rather than view
	# it).

	# print usage information

	if [ "\$1" = "-help" -o "\$1" = "--help" -o "\$1" = "-h" ]; then
	    echo ""
	    echo "\$0 [options]"
	    echo ""
EODF

echo "This program will install the $label by unpacking the embedded tarball and executing the $script installation utility." \
    | fmt -w 70 | awk '{print "    echo \"" $0 "\"" }' >> $archname

echo "    echo \"\"" >> $archname

echo "Below are the most common options; for a complete list use '--advanced-options'." \
    | fmt -w 70 | awk '{print "    echo \"" $0 "\"" }' >> $archname

cat <<- EODF  >> $archname
	    echo ""
	    echo "--info"
	    echo "  Print embedded info (title, default target directory) and exit."
	    echo ""
	    echo "--check"
	    echo "  Check integrity of the archive and exit."
	    echo ""
	    echo "-x, --extract-only"
	    echo "  Extract the contents of \$0, but do not"
	    echo "  run 'nvidia-installer'."
	    echo ""
	    echo ""
EODF

echo "The following arguments will be passed on to the $script utility:" \
    | fmt -w 70 | awk '{print "    echo \"" $0 "\"" }' >> $archname

echo "    echo \"\"" >> $archname

$HELP_SCRIPT --help-args-only \
    | awk '{print "    echo \"" $0 "\"" }' \
    | sed -e 's/`/\\`/g' >> $archname

cat <<- EODF  >> $archname
	    echo ""
	    exit 0;
	fi

	if [ "\$1" = "-A" -o "\$1" = "--advanced-options" ]; then
	    echo ""
	    echo "\$0 [options]"
	    echo ""
EODF

echo "This program will install the $label by unpacking the embedded tarball and executing the $script installation utility." \
    | fmt -w 70 | awk '{print "    echo \"" $0 "\"" }' >> $archname

cat <<- EODF  >> $archname
	    echo ""
	    echo "--info"
	    echo "  Print embedded info (title, default target directory) and exit."
	    echo ""
	    echo "--lsm"
	    echo "  Print embedded lsm entry (or no LSM) and exit."
	    echo ""
	    echo "--pkg-history"
	    echo "  Print the package history of this file and exit."
	    echo ""
	    echo "--list"
	    echo "  Print the list of files in the archive and exit."
	    echo ""
	    echo "--check"
	    echo "  Check integrity of the archive and exit."
	    echo ""
	    echo "-x, --extract-only"
	    echo "  Extract the contents of \$0, but do not"
	    echo "  run 'nvidia-installer'."
	    echo ""
	    echo "--add-this-kernel"
	    echo "  Build a precompiled kernel interface for the currently running"
	    echo "  kernel and repackage the .run file to include this newly built"
	    echo "  precompiled kernel interface.  The new .run file will be placed"
	    echo "  in the current directory and the string \"-custom\" appended"
	    echo "  to its name, unless already present, to distinguish it from the"
	    echo "  original .run file."
	    echo ""
	    echo "--apply-patch [Patch]"
	    echo "  Apply the patch 'Patch' to the kernel interface files included"
	    echo "  in the .run file, remove any precompiled kernel interfaces"
	    echo "  and then repackage the .run file.  The new .run file will be"
	    echo "  placed in the current directory and the string \"-custom\""
	    echo "  appended to its name, unless already present, to distinguish it"
	    echo "  from the original .run file."
	    echo ""
	    echo "--keep"
	    echo "  Do not delete target directory when done."
	    echo ""
	    echo "--target [NewDirectory]"
	    echo "  Extract contents in 'NewDirectory'"
	    echo ""
	    echo "--extract-decompress"
	    echo "  Extract the embedded decompression program to stdout"
	    echo ""
	    echo ""
EODF

echo "The following arguments will be passed on to the $script utility:" \
    | fmt -w 70 | awk '{print "    echo \"" $0 "\"" }' >> $archname

echo "    echo \"\"" >> $archname
echo "    echo \"COMMON OPTIONS:\"" >> $archname
echo "    echo \"\"" >> $archname

$HELP_SCRIPT --help-args-only \
    | awk '{print "    echo \"" $0 "\"" }' \
    | sed -e 's/`/\\`/g' >> $archname

echo "    echo \"\"" >> $archname
echo "    echo \"ADVANCED OPTIONS:\"" >> $archname
echo "    echo \"\"" >> $archname

$HELP_SCRIPT --advanced-options-args-only \
    | awk '{print "    echo \"" $0 "\"" }' \
    | sed -e 's/`/\\`/g' >> $archname

cat <<- EODF  >> $archname
	    echo ""
	    exit 0;
	fi

	if [ "\$1" = "-lsm" -o "\$1" = "--lsm" ]; then
	    cat << EOF_LSM
EODF
    
if [ -f "$lsm_file" ]; then
    cat $lsm_file >> $archname
else
    echo "no LSM" >> $archname
fi

cat <<- EOF >> $archname
	EOF_LSM
	    exit 0;
	fi

	if [ "\$1" = "--pkg-history" ]; then
	    cat << EOF_PKG_HISTORY

EOF
    
if [ -f "$pkg_history_file" ]; then
    cat $pkg_history_file >> $archname
fi

cat <<-EOF >> $archname

	EOF_PKG_HISTORY
	    exit 0;
	fi

	if [ "\$1" = "--label" ]; then
	    echo "\$label";
	    exit 0;
	fi

	if [ "\$1" = "--version-string" ]; then
	    echo "\$version_string";
	    exit 0;
	fi

	if [ "\$1" = "--pkg-version" ]; then
	    echo "\$pkg_version";
	    exit 0;
	fi

	if [ "\$1" = "--target-os" ]; then
	    echo "\$TARGET_OS";
	    exit 0;
	fi

	if [ "\$1" = "--target-arch" ]; then
	    echo "\$TARGET_ARCH";
	    exit 0;
	fi

	if [ "\$1" = "--target-directory" ]; then
	    echo "\$targetdir";
	    exit 0;
	fi

	if [ "\$1" = "--script" ]; then
	   echo "\$script \$scriptargs"
	   exit 0
	fi

	if [ "\$1" = "--info" ]; then
	    echo
	    echo "  Identification    : \$label"
	    echo "  Target directory  : \$targetdir"
	    echo "  Uncompressed size : $USIZE KB"
	    echo "  Compression       : $COMPRESS"
	    echo "  Date of packaging : $DATE"
	    echo "  Application run after extraction : \$script \$scriptargs"
	    echo
	    if [ x"\$keep" = xy ]; then
	        echo "  The directory \$targetdir will not be removed after extraction."
	    else
	        echo "  The directory \$targetdir will be removed after extraction."
	    fi
	    echo
	    exit 0;
	fi

	location="\`pwd\`"
EOF

if [ -n "$EMBED_DECOMPRESS" ]; then
    cat <<-EOF >> $archname
	catDecompress() {
	    tail -n +\${skip_decompress} \$0 | head -n \${size_decompress}
	}

	if ! which "$COMPRESS" > /dev/null 2>&1; then
	    decompressDir=\`mktemp -d "\$TMPDIR/makeself.XXXXXXXX" 2> /dev/null\`
	    decompress="\$decompressDir/$COMPRESS"
	    (cd "\$location"; catDecompress) > "\$decompress"
	    chmod +x "\$decompress"
	    PATH="\$decompressDir:\$PATH"
	    trap cleanupDecompress EXIT
	fi
EOF
else
    cat <<-EOF >> $archname
	catDecompress() {
	    echo "No decompressor embedded in this archive" >&2
	    exit 1
	}
EOF
fi

cat <<-EOF >> $archname

	cleanupDecompress() {
	    if [ -d "\$decompressDir" ]; then
	        rm -r "\$decompressDir"
	    fi
	    decompressDir=""
	}

	if [ "\$1" = "--list" ]; then
	    echo "Target directory: \$targetdir"
	    tail -n +\$skip \$0  | $UNCOMPRESS_CMD | tar tvf - 2> /dev/null
	    exit 0;
	fi

	if [ "\$1" = "--extract-decompress" ]; then
	    catDecompress
	    exit 0
	fi

	if [ "\$1" = "--check" ]; then
	    sum1=\`tail -n +4 \$0 | cksum | awk '{print \$1}'\`
	    [ "\$sum1" != "\$CRCsum" ] && {
	        echo "Error in checksums \$sum1 \$CRCsum"
	        exit 2;
	    }
	    if [ \$MD5 != "00000000000000000000000000000000" ]; then
	        # space separated list of directories
	        [ x"\$GUESS_MD5_PATH" = "x" ] && GUESS_MD5_PATH="/usr/local/ssl/bin /usr/local/bin /usr/bin /bin"
	        MD5_PATH=""
	        for a in \$GUESS_MD5_PATH; do
	            #if which \$a/md5 >/dev/null 2>&1 ; then
	            if [ -x "\$a/md5sum" ]; then
	                MD5_PATH=\$a;
	            fi
	        done
	        if [ -x \$MD5_PATH/md5sum ]; then
	            md5sum=\`tail -n +4 \$0 | \$MD5_PATH/md5sum | cut -b-32\`;
	            [ \$md5sum != \$MD5 ] && {
	                echo "Error in md5 sums \$md5sum \$MD5"
	                exit 2
	            } || { echo "check sums and md5 sums are ok"; exit 0; }
	        fi
	        if [ ! -x \$MD5_PATH/md5sum ]; then
	            echo "an embedded md5 sum of the archive exists but no md5 program was found in \$GUESS_MD5_PATH"
	            echo "if you have md5 on your system, you should try :"
	            echo "env GUESS_MD5_PATH=\"FirstDirectory SecondDirectory ...\" \$0 -check"
	        fi
	    else
	        echo "check sums are OK ; echo \$0 does not contain embedded md5 sum" ;
	    fi
	    exit 0;
	fi

	run_script=y
	keep=n
	apply_patch=n

	while [ "\$1" ]; do
	   case "\$1" in
	       "--extract-only"|"-x")
	           run_script=n;
	           keep=y;
	           ;;
	       "--keep")
	           keep=y;
	           ;;
	       "--target")
	           if [ "\$2" ]; then
	               targetdir="\$2";
	               keep=y;
	               shift;
	           else
	               echo "ERROR: --target: no target directory specified."
	               exit 1;
	           fi
	           ;;
	       "--add-this-kernel")
	           add_this_kernel=y;
	           scriptargs="\$scriptargs \$1"
	           ;;
	       "--tmpdir")
	           scriptargs="\$scriptargs \$1 \$2"
	           if [ "\$2" ]; then
	               TMPROOT="\$2";
	               shift;
	           else
	               echo "ERROR: --tmpdir: no temporary directory specified."
	               exit 1;
	           fi
	           ;;
	       "--apply-patch")
	           if [ "\$2" ]; then
	               if [ "\`dirname \$2\`" != "." ]; then 
	                   patchfile="\$2";
	               else
	                   patchfile="\`pwd\`/\$2"
	               fi
	               run_script=n;
	               apply_patch=y;
	               shift;
	           else
	               echo "ERROR: --apply-patch: no patch file specified."
	               exit 1;
	           fi
	           ;;
	       *)
	           scriptargs="\$scriptargs \$1"
	           ;;
	    esac
	    shift
	done

EOF

# output code to check if tmp is executable

cat <<-EOF >> $archname

	# Check that the tmp directory is executable
	# Make path absolute if not already
	if ! echo "\$TMPROOT" | grep -q "^/"; then 
	    TMPROOT=\`pwd\`/"\$TMPROOT";
	fi

	if [ ! -d "\$TMPROOT" ]; then 
	    mkdir -p "\$TMPROOT" || {
	        echo "Unable to create temp directory \$TMPROOT"
	        exit 1
	    }
	fi

	TMPDIR="\$TMPROOT"
	TMPFILE=\`mktemp "\$TMPDIR/makeself.XXXXXXXX" 2> /dev/null\`

	if [ "a\$TMPFILE" = "a" ]; then
	    echo "Unable to create temporary file in \$TMPDIR"
	    exit 1
	fi

	chmod +x "\$TMPFILE"

	# Try to execute, see what happens
	"\$TMPFILE" > /dev/null 2>&1 || {
	    echo "ERROR: Temporary directory \$TMPDIR is not executable - use the " \\
	         "--tmpdir option to specify a different one."
	    rm "\$TMPFILE"
	    exit 1;
	}

	rm "\$TMPFILE"

EOF

# output code to do a platform check

cat <<-EOF >> $archname

	# Perform a platform check

	LOCAL_OS=\`uname 2> /dev/null\`
	LOCAL_ARCH=\`uname -m 2> /dev/null\`

	[ "\$LOCAL_ARCH" = "i386" ] && LOCAL_ARCH="x86"
	[ "\$LOCAL_ARCH" = "i486" ] && LOCAL_ARCH="x86"
	[ "\$LOCAL_ARCH" = "i586" ] && LOCAL_ARCH="x86"
	[ "\$LOCAL_ARCH" = "i686" ] && LOCAL_ARCH="x86"

	if [ -z "\$LOCAL_OS" -o -z "\$LOCAL_ARCH" ]; then
	    echo "ERROR: missing/broken uname.  Cannot perform platform check."
	    exit 1;
	fi

	if [ "\$LOCAL_ARCH" != "\$TARGET_ARCH" -o "\$LOCAL_OS" != "\$TARGET_OS" ]; then
	    if [ "\$run_script" = "y" ]; then
	        echo ""
	        echo "ERROR: this .run file is intended for the"
	        echo "\${TARGET_OS}-\${TARGET_ARCH} platform, but you appear to be"
	        echo "running on \${LOCAL_OS}-\${LOCAL_ARCH}.  Aborting installation."
	        echo ""
	        exit 1;
	    fi
	fi

EOF

if [ $CURRENT = "n" ]; then
cat <<-EOF >> $archname

	if [ "\$keep" = "y" ]; then
	    echo "Creating directory \$targetdir"; tmpdir=\$targetdir;
	else
	    workingdir="\$TMPROOT/selfgz\$\$"
	    tmpdir="\$workingdir/\$targetdir";
	    rm -rf \$tmpdir
	fi

	if [ -d "\$tmpdir" -o -f "\$tmpdir" ]; then
	    echo "The directory '\$tmpdir' already exists.  Please either"
	    echo "move the existing directory out of the way, or specify a"
	    echo "different directory with the '--target' option."
	    exit 1
	fi

	mkdir -p \$tmpdir || {
	    echo "Unable to create the target directory '\$tmpdir'."
	    exit 1
	}
EOF
else
cat <<-EOF >> $archname
	tmpdir=.
EOF
fi
cat <<-EOF >> $archname

	echo=echo; [ -x /usr/ucb/echo ] && echo=/usr/ucb/echo
	if [ x\$SETUP_NOCHECK != x1 ]; then
	    \$echo -n "Verifying archive integrity... "
	    sum1=\`tail -n +4 \$0 | cksum | awk '{print \$1}'\`
	    [ \$sum1 != \$CRCsum ] && {
	        \$echo "Error in check sums \$sum1 \$CRCsum"
	        exit 2;
	    }
	    echo "OK"
	fi
	if [ \$MD5 != \"00000000000000000000000000000000\" ]; then
	    # space separated list of directories
	    [ x"\$GUESS_MD5_PATH" = "x" ] && GUESS_MD5_PATH="/usr/local/ssl/bin /usr/local/bin /usr/bin /bin"
	    MD5_PATH=""
	    for a in \$GUESS_MD5_PATH; do
	        #if which \$a/md5 >/dev/null 2>&1 ; then
	        if [ -x "\$a/md5sum" ]; then
	            MD5_PATH=\$a;
	        fi
	    done
	    if [ -x \$MD5_PATH/md5sum ]; then
	        md5sum=\`tail -n +4 \$0 | \$MD5_PATH/md5sum | cut -b-32\`;
	        [ \$md5sum != \$MD5 ] && {
	            \$echo "Error in md5 sums \$md5sum \$MD5"
	            exit 2;
	        }
	    fi
	fi

	UnTAR() {
	    tar xvf - 2> /dev/null || {
	        echo "Extraction failed." > /dev/tty; kill -15 \$$;
	    };
	}

	\$echo -n "Uncompressing \$label"
	cd \$tmpdir ; res=3

	[ "\$keep" = "y" ] || trap '\$echo "Signal caught, cleaning up" > /dev/tty; cd \$TMPROOT; rm -rf \$tmpdir; exit 15' 1 2 15

	if (cd "\$location"; tail -n +\$skip \$0; ) | $UNCOMPRESS_CMD | UnTAR | (while read a; do \$echo -n "."; done; \$echo; ); then
	    chown -Rf \`id -u\`:\`id -g\` .
	    res=0;
	    if [ "\$script" -a "\$run_script" = "y" ]; then
	        \$script \$scriptargs \$*; res=\$?
	    fi

	    if [ "\$add_this_kernel" = "y" -a "\$res" = "0" ]; then
	        repackage_file=y;
	    fi

	    if [ "\$apply_patch" = "y" ]; then
	        patch=\`which patch 2> /dev/null | head -n 1\`
	        if [ \$? -eq 0 -a "\$patch" ]; then
	            if [ "\$keep" = "y" ]; then
	                cp -pR usr/src/nv usr/src/nv.orig
	            fi
	            \$patch -p0 < "\$patchfile"
	            if [ \$? -ne 0 ]; then
	                \$echo "Failed to apply patch file \"\$patchfile\"."
	                if [ "\$keep" = "y" ]; then
	                    rm -rf usr/src/nv
	                    mv usr/src/nv.orig usr/src/nv
	                fi
	            else
	                if [ "\$keep" = "y" ]; then
	                    rm -rf usr/src/nv.orig
	                fi
	                rm -rf usr/src/nv/*.orig usr/src/nv/precompiled
	                repackage_file=y
	            fi
	        else
	            \$echo "Couldn't locate the 'patch' utility."
	        fi
	    fi

	    if [ "\$repackage_file" = "y" ]; then

	        cd ..

	        new_targetdir="\`basename \$targetdir | sed -e \"s/-custom//\"\`"
	        new_targetdir="\${new_targetdir}-custom"

	        if [ "\$targetdir" != "\$new_targetdir" ]; then
	            mv \$targetdir \$new_targetdir
	        fi

	        # update the pkg-history.txt file
	        chmod 644 ./\$new_targetdir/pkg-history.txt

	        if [ "\$add_this_kernel" = "y" ]; then
	            \$echo "\$new_targetdir: Added precompiled kernel interface for: " >> ./\$new_targetdir/pkg-history.txt
	            \$echo "\`uname -s -r -v -m 2> /dev/null\`" >> ./\$new_targetdir/pkg-history.txt
	        else
	            \$echo "\$new_targetdir: Applied patch file: \$patchfile" >> ./\$new_targetdir/pkg-history.txt
	        fi

	        \$echo "" >> ./\$new_targetdir/pkg-history.txt

	        # retrieve the lsm file
	        tmplsm="\$TMPDIR/nvidia.lsm.\$\$"
	        if [ "\`dirname \$0\`" != "." ]; then
	            sh \$0 --lsm > \$tmplsm
	        else
	            sh \$location/\$0 --lsm > \$tmplsm
	        fi
 
	        sh ./\$new_targetdir/makeself.sh \
	            --lsm \$tmplsm \
	            --version-string \$version_string \
	            --pkg-version \$pkg_version \
	            --pkg-history ./\$new_targetdir/pkg-history.txt \
	            --target-os \$TARGET_OS \
	            --target-arch \$TARGET_ARCH \
	            \$new_targetdir \$new_targetdir.run \
	            "\$label" "\$script"

	        rm -f \$tmplsm

	        [ "\$keep" = "y" ] || mv \$new_targetdir.run \$location
	    fi

	    [ "\$keep" = "y" ] || { cd \$TMPROOT; rm -rf \$workingdir; }
	else
	    \$echo "Cannot decompress \$0"; exit 1
	fi

        cleanupDecompress

	exit \$res

	END_OF_STUB
EOF

tmpfile="${TMPDIR:=/tmp}/mkself$$"

# Replace __SKIP_DECOMPRESS__ and __SIZE_DECOMPRESS__ with the number of lines
# to skip to find the decompressor program and the size in "lines" of that
# program, respectively.  Add a newline to ensure that the decompressor binary
# ends on a "line" boundary.
skip_decompress=`cat $archname | wc -l`
skip_decompress=`expr ${skip_decompress} + 1`
if [ -n "$EMBED_DECOMPRESS" ]; then
    size_decompress=`( cat "$EMBED_DECOMPRESS"; echo ) | wc -l`
    ( cat "$EMBED_DECOMPRESS"; echo ) >> "$archname"
else
    size_decompress=0
fi

# replace __SKIP__ with the number of lines to skip to find the package archive
skip=`expr ${skip_decompress} + ${size_decompress}`

sed -e "s/__SKIP__/$skip/" \
    -e "s/__SKIP_DECOMPRESS__/${skip_decompress}/" \
    -e "s/__SIZE_DECOMPRESS__/${size_decompress}/" \
    "$archname" > "$tmpfile"
mv "$tmpfile" "$archname"

# Append the compressed tar data after the stub
if [ "$SILENT" = "n" ]; then
  echo "Adding files to archive named \"$archname\"..."
fi
# (cd $archdir; tar cvf - *| $COMPRESS_CMD ) >> $archname && chmod +x $archname && ..

(cd "$archdir"; tar $TAR_ARGS - `find . -maxdepth 1 -mindepth 1` | $COMPRESS_CMD ) >> "$archname" || { echo Aborting; exit 1; }
[ "$SILENT" = "n" ] && echo
echo >> "$archname" >&- 2> /dev/null; # try to close the archive
# echo Self-extractible archive \"$archname\" successfully created.
sum1=`tail -n +4 "$archname" | cksum | awk '{print $1}'`
# space separated list of directories
[ x"$GUESS_MD5_PATH" = "x" ] && GUESS_MD5_PATH="/usr/local/ssl/bin /usr/bin /usr/local/bin /bin"
MD5_PATH=""
for a in $GUESS_MD5_PATH; do
  #if which $a/md5sum >/dev/null 2>&1 ; then
  if [ -x "$a/md5sum" ]; then
     MD5_PATH=$a;
  fi
done

tmpfile="${TMPDIR:=/tmp}/mkself$$"
if [ -x $MD5_PATH/md5sum ]; then
  md5sum=`tail -n +4 "$archname" | $MD5_PATH/md5sum | cut -b-32`;
  [ "$SILENT" = "n" ] && echo -e "CRC: $sum1\nMD5: $md5sum\n"
  sed -e "s/^CRCsum=0000000000/CRCsum=$sum1/" -e "s/^MD5=00000000000000000000000000000000/MD5=$md5sum/" "$archname" > "$tmpfile"
else
  [ "$SILENT" = "n" ] && echo -e "CRC: $sum1\nMD5: none, md5sum binary not found\n"
  sed -e "s/^CRCsum=0000000000/CRCsum=$sum1/" "$archname" > "$tmpfile"
fi
mv "$tmpfile" "$archname"
chmod +x "$archname"

if [ "$SILENT" = "n" ]; then
  echo "Self-extractible archive \"$archname\" successfully created."
fi
