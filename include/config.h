/* This file contains configurable macros */
#ifndef __CONFIG_H__
#define __CONFIG_H__

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

#endif /* CONFIG_H */
