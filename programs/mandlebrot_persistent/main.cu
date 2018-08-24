// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <cuda.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#include "bmp.h"

__global__
FGPU_DEFINE_KERNEL(render, char *out, int width, int height) {
  
  fgpu_dev_ctx_t *ctx;
  ctx = FGPU_DEVICE_INIT();
  uint3 _blockIdx;

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    unsigned int x_dim = _blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y_dim = _blockIdx.y*blockDim.y + threadIdx.y;
    int index = 3*width*y_dim + x_dim*3;
    float x_origin = ((float) x_dim/width)*3.25 - 2;
    float y_origin = ((float) y_dim/width)*2.5 - 1.25;

    float x = 0.0;
    float y = 0.0;

    int iteration = 0;
    int max_iteration = 256;
    while(x*x + y*y <= 4 && iteration < max_iteration) {
      float xtemp = x*x - y*y + x_origin;
      y = 2*x*y + y_origin;
      x = xtemp;
      iteration++;
    }

    if(iteration == max_iteration) {
      FGPU_COLOR_STORE(ctx, &out[index], 0);
      FGPU_COLOR_STORE(ctx, &out[index + 1], 0);
      FGPU_COLOR_STORE(ctx, &out[index + 2], 0);
    } else {
      FGPU_COLOR_STORE(ctx, &out[index], iteration);
      FGPU_COLOR_STORE(ctx, &out[index + 1], iteration);
      FGPU_COLOR_STORE(ctx, &out[index + 2], iteration);
    }
  } FGPU_FOR_EACH_END 
}

int runCUDA(int width, int height)
{
  // Multiply by 3 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(char) * width * height * 3;

  int ret;
  int nIter = 10000;
  double start, total;


  char *host_image, *device_image;

  ret = fgpu_memory_allocate((void **)&host_image, buffer_size);
  if (ret < 0)
    return ret;
  ret = fgpu_memory_get_device_pointer((void **)&device_image, host_image);
  if (ret < 0)
    return ret;

  ret = fgpu_memory_prefetch_to_device_async(host_image, buffer_size);
  if (ret < 0)
    return ret;
  ret = fgpu_color_stream_synchronize();
  if (ret < 0)
    return ret;


  dim3 blockDim(16, 16, 1);
  dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);
  
  start = dtime_usec(0);
  FGPU_LAUNCH_KERNEL(gridDim, blockDim, 0, render, device_image, width, height);
  ret = fgpu_color_stream_synchronize();
  if (ret < 0)
      return ret;
  total = dtime_usec(start);

  printf("Time:%f us\n", total);
  printf("Looping\n");

  start = dtime_usec(0);
  for (int i = 0; i < nIter; i++) {
    start = dtime_usec(0);
    FGPU_LAUNCH_KERNEL(gridDim, blockDim, 0, render, device_image, width, height);
    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;
    total = dtime_usec(start);
    printf("Time:%f us\n", total);
  }
  ret = fgpu_color_stream_synchronize();
  if (ret < 0)
    return ret;

  total = dtime_usec(start);

  printf("Avg Time:%f us\n", total / nIter);

  ret = fgpu_memory_prefetch_from_device_async(host_image, buffer_size);
  if (ret < 0)
    return ret;
  ret = fgpu_color_stream_synchronize();
  if (ret < 0)
    return ret;

  // Now write the file
  write_bmp("output.bmp", width, height, host_image);

  fgpu_memory_free(host_image);
  return 0;
}

int main(int argc, const char * argv[]) {
  int color, ret;

  if (argc != 2) {
    fprintf(stderr, "Insufficient number of arguments\n");
    exit(-1);
  }

  ret = fgpu_init();
  if (ret < 0)
    return ret;

  color = atoi(argv[1]);
  printf("Color selected:%d\n", color);

  ret = fgpu_set_color_prop(color, 128 * 1024 * 1024);
  if (ret < 0)
    return ret;

  ret = runCUDA(4096, 4096);
  if (ret < 0)
      return ret;

  fgpu_deinit();
  
  return 0;
}
