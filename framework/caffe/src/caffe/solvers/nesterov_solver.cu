#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void NesterovUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float hi = h[i];
    float hi_new = h[i] = momentum * hi + local_rate * g[i];
    g[i] = (1+momentum) * hi_new - momentum * hi;
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(NesterovUpdate, int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, N, _blockIdx, _gridDim) {
      Dtype *gaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &g[i]);
      Dtype *haddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &h[i]);

      float hi = *haddr;
      float hi_new = *haddr = momentum * hi + local_rate * (*gaddr);
      *gaddr = (1+momentum) * hi_new - momentum * hi;
    }

  }
}
#endif

template <typename Dtype>
void nesterov_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
#ifndef USE_FGPU
  NesterovUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(NesterovUpdate<Dtype> , // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      N, g, h, momentum, local_rate));

#endif
}
template void nesterov_update_gpu<float>(int, float*, float*, float, float);
template void nesterov_update_gpu<double>(int, double*, double*, double,
    double);

}  // namespace caffe
