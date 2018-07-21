/* Header file for memory coloring functionality */
#ifndef __MEMORY_H__
#define __MEMORY_H__

#ifdef FGPU_MEM_COLORING_ENABLED

int fgpu_device_get_num_memory_colors(int device, int *num_colors, size_t *max_len);
int fgpu_process_set_colors_info(int device, int color, size_t *length);

#endif /* FGPU_MEM_COLORING_ENABLED */

#endif /* __MEMORY_H__ */
