#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void AdaDeltaUpdate(int N, Dtype* g, Dtype* h, Dtype* h2,
    Dtype momentum, Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = momentum * h[i] + (1-momentum) * gi * gi;
    gi = gi * sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + (1-momentum) * gi * gi;
    g[i] = local_rate * gi;
  }
}

#else // USE_FGPU
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(AdaDeltaUpdate, int N, Dtype* g, Dtype* h, Dtype* h2,
    Dtype momentum, Dtype delta, Dtype local_rate) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, N, _blockIdx, _gridDim) {
      Dtype *gaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &g[i]);
      Dtype *haddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &h[i]);
      Dtype *h2addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &h2[i]);

      float gi = *gaddr;
      float hi = *haddr = momentum * (*haddr) + (1-momentum) * gi * gi;
      gi = gi * sqrt((*h2addr + delta) / (hi + delta));
      *h2addr = momentum * (*h2addr) + (1-momentum) * gi * gi;
      *gaddr = local_rate * gi;
    }

  } 
}
#endif // USE_FGPU

template <typename Dtype>
void adadelta_update_gpu(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum,
    Dtype delta, Dtype local_rate) {
#ifndef USE_FGPU
  AdaDeltaUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, h2, momentum, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(AdaDeltaUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      N, g, h, h2, momentum, delta, local_rate));
#endif
}
template void adadelta_update_gpu<float>(int , float*, float*, float*,
    float, float, float);
template void adadelta_update_gpu<double>(int, double*, double*, double*,
    double, double, double);

}  // namespace caffe
