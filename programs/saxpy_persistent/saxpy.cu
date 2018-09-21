#include <stdio.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

__global__
FGPU_DEFINE_KERNEL(saxpy, int n, float a, float *x, float *y)
{
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx;
  ctx = FGPU_DEVICE_INIT();

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    int i = _blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
      float res = a * FGPU_COLOR_LOAD(ctx, &x[i]) + FGPU_COLOR_LOAD(ctx, &y[i]);
//      printf("Res:%f, X:%f, Y:%f, A:%f\n", res, FGPU_COLOR_LOAD(ctx, &x[i]) , FGPU_COLOR_LOAD(ctx, &y[i]), a);
      FGPU_COLOR_STORE(ctx, &y[i], res);
    }
  } FGPU_FOR_EACH_END;
}

int main(int argc, char **argv)
{
  int N = 1<<20;
  int nIter;
  pstats_t stats;
  int ret;

  test_initialize(argc, argv, &nIter);

  dim3 grid((N+255)/256, 1, 1), threads(256, 1, 1);
  
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  ret = fgpu_memory_allocate((void **) &d_x, N*sizeof(float));
  if (ret < 0)
    return ret;
  ret = fgpu_memory_allocate((void **) &d_y, N*sizeof(float));
  if (ret < 0)
    return ret;

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  ret = fgpu_memory_copy_async(d_x, x, N*sizeof(float), FGPU_COPY_CPU_TO_GPU);
  if (ret < 0)
      return ret;

  ret = fgpu_memory_copy_async(d_y, y, N*sizeof(float), FGPU_COPY_CPU_TO_GPU);
  if (ret < 0)
      return ret;

  // Functional test
  FGPU_LAUNCH_KERNEL(saxpy, grid, threads, 0, N, 2.0f, d_x, d_y);
  
  ret = fgpu_memory_copy_async(y, d_y, N*sizeof(float), FGPU_COPY_GPU_TO_CPU);
  if (ret < 0)
      return ret;

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  if (maxError != 0) {
      fprintf(stderr, "Failed: Error too large\n");
      exit(-1);
  }

  // Warmup
  for (int i = 0; i < nIter; i++) {

    double sub_start = dtime_usec(0);
    
    FGPU_LAUNCH_KERNEL(saxpy, grid, threads, 0, N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();

    dprintf("Time:%f\n", dtime_usec(sub_start));
  }

  // Actual
  pstats_init(&stats);
  for (int j = 0; j < nIter; j++)
  {
    double sub_start = dtime_usec(0);
    FGPU_LAUNCH_KERNEL(saxpy, grid, threads, 0, N, 2.0f, d_x, d_y);    
    cudaDeviceSynchronize();
    pstats_add_observation(&stats, dtime_usec(sub_start));
  }
    
  cudaDeviceSynchronize();

  pstats_print(&stats);

  fgpu_memory_free(d_x);
  fgpu_memory_free(d_y);
  free(x);
  free(y);

  test_deinitialize();
}
