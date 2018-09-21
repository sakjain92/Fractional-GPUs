// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <cuda.h>

#include <fractional_gpu_testing.hpp>

#include "bmp.h"

__global__ void render(char *out, int width, int height) {
  unsigned int x_dim = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y_dim = blockIdx.y*blockDim.y + threadIdx.y;
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
    out[index] = 0;
    out[index + 1] = 0;
    out[index + 2] = 0;
  } else {
    out[index] = iteration;
    out[index + 1] = iteration;
    out[index + 2] = iteration;
  }
}

void runCUDA(int width, int height, int nIter)
{
  // Multiply by 3 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(char) * width * height * 3;

  char *image;
  double start, total;

  cudaMalloc((void **) &image, buffer_size);

  char *host_image = (char *) malloc(buffer_size);

  dim3 blockDim(16, 16, 1);
  dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);
  
  start = dtime_usec(0);
  render<<< gridDim, blockDim, 0 >>>(image, width, height);
  cudaDeviceSynchronize();
  total = dtime_usec(start);

  printf("Time:%f us\n", total);
  printf("Looping\n");

  start = dtime_usec(0);
  for (int i = 0; i < nIter; i++) {
    render<<< gridDim, blockDim, 0 >>>(image, width, height);
  }
  cudaDeviceSynchronize();
  total = dtime_usec(start);

  printf("Avg Time:%f us\n", total / nIter);

  cudaMemcpy(host_image, image, buffer_size, cudaMemcpyDeviceToHost);

  // Now write the file
  write_bmp("output.bmp", width, height, host_image);

  cudaFree(image);
  free(host_image);
}

int main(int argc, char **argv) {

  int num_iterations;

  test_initialize(argc, argv, &num_iterations);

  runCUDA(4096, 4096, num_iterations);

  test_deinitialize();
  return 0;
}
