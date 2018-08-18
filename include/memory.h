/* Header file for memory coloring functionality */
#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <common.h>

#ifdef FGPU_MEM_COLORING_ENABLED

int fgpu_memory_set_colors_info(int device, int color, size_t length, cudaStream_t stream);
void fgpu_memory_deinit(void);
int fgpu_memory_copy_async_internal(void *dst, const void *src, size_t count,
                                    enum fgpu_memory_copy_type type,
                                    cudaStream_t stream);

#if defined(FGPU_USER_MEM_COLORING_ENABLED)
int fgpu_get_memory_info(uintptr_t *start_virt_addr, uintptr_t *start_idx);
#endif /* FGPU_USER_MEM_COLORING_ENABLED */

#endif /* FGPU_MEM_COLORING_ENABLED */

#endif /* __MEMORY_H__ */
