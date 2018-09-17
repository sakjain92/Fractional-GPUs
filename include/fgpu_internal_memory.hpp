/* Header file for memory coloring functionality */
#ifndef __FGPU_INTERNAL_MEMORY_HPP__
#define __FGPU_INTERNAL_MEMORY_HPP__

#include <fractional_gpu.hpp>

#include <fgpu_internal_common.hpp>

#ifdef FGPU_MEM_COLORING_ENABLED

int fgpu_memory_set_colors_info(int device, int color, size_t length, cudaStream_t stream);
void fgpu_memory_deinit(void);

#if defined(FGPU_USER_MEM_COLORING_ENABLED)

int fgpu_get_memory_info(uintptr_t *start_virt_addr, uintptr_t *start_idx);

#endif /* FGPU_USER_MEM_COLORING_ENABLED */

#endif /* FGPU_MEM_COLORING_ENABLED */

int fgpu_memory_copy_async_internal(void *dst, const void *src, size_t count,
                                    enum fgpu_memory_copy_type type,
                                    cudaStream_t stream);
int fgpu_memory_memset_async_internal(void *address, int value, size_t count, cudaStream_t stream);

#endif /* __FGPU_INTERNAL_MEMORY_HPP__ */
