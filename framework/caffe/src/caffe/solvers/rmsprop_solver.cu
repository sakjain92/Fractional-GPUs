#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void RMSPropUpdate(int N, Dtype* g, Dtype* h,
    Dtype rms_decay, Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = rms_decay*h[i] + (1-rms_decay)*gi*gi;
    g[i] = local_rate * g[i] / (sqrt(hi) + delta);
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(RMSPropUpdate, int N, Dtype* g, Dtype* h,
    Dtype rms_decay, Dtype delta, Dtype local_rate) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, N, _blockIdx, _gridDim) {
      Dtype *gaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &g[i]);
      Dtype *haddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &h[i]);

      float gi = *gaddr;
      float hi = *haddr = rms_decay*(*haddr) + (1-rms_decay)*gi*gi;
      *gaddr = local_rate * (*gaddr) / (sqrt(hi) + delta);
    }

  } 
}
#endif

template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate) {
#ifndef USE_FGPU
  RMSPropUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, rms_decay, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(RMSPropUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      N, g, h, rms_decay, delta, local_rate));
#endif
}
template void rmsprop_update_gpu<float>(int, float*, float*, float, float,
    float);
template void rmsprop_update_gpu<double>(int, double*, double*, double, double,
    double);

}  // namespace caffe
