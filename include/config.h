/* This file contains configurable macros */
#ifndef __CONFIG_H__
#define __CONFIG_H__

/* Cache line size on GPU. Needed for performance improvement. */
#define FGPU_DEVICE_CACHELINE_SIZE      32

/* Maximum number of colors supported - This is also limited by number of streams */
#define FGPU_MAX_NUM_COLORS             8

/* Maximum number of SMs on device supported */
#define FGPU_MAX_NUM_SM                 100

/* Maximum number of blocks per SM supported */
#define FGPU_MAX_NUM_BLOCKS_PER_SM      64

/* TODO: Remove these and query per device */
#define FGPU_NUM_COLORS                 2
#define FGPU_NUM_SM                     15
#define FGPU_NUM_THREADS_PER_SM         2048

#endif /* CONFIG_H */
