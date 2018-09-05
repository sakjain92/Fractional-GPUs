#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = h[i] + gi*gi;
    g[i] = local_rate * gi / (sqrt(hi) + delta);
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(AdaGradUpdate, int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, N, _blockIdx, _gridDim) {
      Dtype *gaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &g[i]);
      Dtype *haddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &h[i]);

      float gi = *gaddr;
      float hi = *haddr = *haddr + gi*gi;
      *gaddr = local_rate * gi / (sqrt(hi) + delta);
    }

  } 
}

#endif

template <typename Dtype>
void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
#ifndef USE_FGPU
  AdaGradUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(AdaGradUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      N, g, h, delta, local_rate));

#endif
}
template void adagrad_update_gpu<float>(int, float*, float*, float, float);
template void adagrad_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
