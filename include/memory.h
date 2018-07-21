/* Header file for memory coloring functionality */
#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <common.h>

#ifdef FGPU_MEM_COLORING_ENABLED

int fgpu_device_get_num_memory_colors(int device, int *num_colors, size_t *max_len);
int fgpu_process_set_colors_info(int device, int color, size_t length);
void fgpu_memory_deinit(void);
int fgpu_get_memory_info(uintptr_t *start_virt_addr, uintptr_t *start_idx);

#endif /* FGPU_MEM_COLORING_ENABLED */

#endif /* __MEMORY_H__ */
