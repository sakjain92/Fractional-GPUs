/* This file contains configurable macros */
#ifndef __FGPU_INTERNAL_CONFIG_HPP__
#define __FGPU_INTERNAL_CONFIG_HPP__

#include <g_nvconfig.h>
#include <fgpu_internal_build_config.hpp>

/* Cache line size on GPU. Needed for performance improvement. */
#define FGPU_DEVICE_CACHELINE_SIZE      32

/* Each memory allocation needs to be aliged to a boundary */
#define FGPU_DEVICE_ADDRESS_ALIGNMENT   16

/* Maximum number of colors supported */
#define FGPU_MAX_NUM_COLORS             8

/* Maximum number of persistent blocks */
#define FGPU_MAX_NUM_PBLOCKS            6400

/* Maximum number of pending CUDA tasks */
#define FGPU_MAX_PENDING_TASKS          100

/* Can be set to -1 if no preference. Preference is like a hint */
#define FGPU_PREFERRED_NUM_COLORS	2

#endif /* __FGPU_INTERNAL_CONFIG_HPP__ */
