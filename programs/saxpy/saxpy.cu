#include <stdio.h>

#include <common.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  int nIter = 10000;
  double start, total;
  pstats_t stats;

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Functional test
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

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

    start = dtime_usec(0);
        
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();

    total = dtime_usec(start);
    printf("Time:%f\n", total);
  }

  // Actual
  pstats_init(&stats);
  start = dtime_usec(0);
  for (int j = 0; j < nIter; j++)
  {
    double sub_start = dtime_usec(0);
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();
    pstats_add_observation(&stats, dtime_usec(sub_start));
  }
    
  cudaDeviceSynchronize();
  total = dtime_usec(start);

  pstats_print(&stats);

  // Termination - To allow other color's application to overlap in time
  for (int j = 0; j < nIter; j++)
  {
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();
  }
    
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
