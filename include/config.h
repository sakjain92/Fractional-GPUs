/* This file contains configurable macros */
#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <g_nvconfig.h>

/* Cache line size on GPU. Needed for performance improvement. */
#define FGPU_DEVICE_CACHELINE_SIZE      32

/* Maximum number of colors supported - This is also limited by number of streams */
#define FGPU_MAX_NUM_COLORS             8

/* Maximum number of persistent blocks */
#define FGPU_MAX_NUM_PBLOCKS            6400

/* Minimum threads in a block - Minimum unit is a warp on CUDA device */
#define FGPU_MIN_BLOCKDIMS              32

/* Maximum number of pending CUDA tasks */
#define FGPU_MAX_PENDING_TASKS          100

/* Can be set to -1 if no preference. Preference is like a hint */
#define FGPU_PREFERRED_NUM_COLORS	2

/* When userspace coloring is enabled, coloring must be enabled */
#if defined(FGPU_USER_MEM_COLORING_ENABLED) && !defined(FGPU_MEM_COLORING_ENABLED) || \
    defined(FGPU_TEST_MEM_COLORING_ENABLED) && !defined(FGPU_MEM_COLORING_ENABLED) 
#error "FGPU_MEM_COLORING_ENABLED not defined"
#endif

/* Only one can be selected at a time */
#if defined(FGPU_USER_MEM_COLORING_ENABLED) && defined(FGPU_TEST_MEM_COLORING_ENABLED)
#error "Both FGPU_USER_MEM_COLORING_ENABLED and FGPU_TEST_MEM_COLORING_ENABLED defined"
#endif

#endif /* CONFIG_H */
