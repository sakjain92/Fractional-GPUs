#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(AdamUpdate, int N, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, N, _blockIdx, _gridDim) {
      Dtype *gaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &g[i]);
      Dtype *maddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &m[i]);
      Dtype *vaddr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &v[i]);

      float gi = *gaddr;
      float mi = *maddr = (*maddr)*beta1 + gi*(1-beta1);
      float vi = *vaddr = (*vaddr)*beta2 + gi*gi*(1-beta2);
      *gaddr = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
    }

  } 
}
#endif

template <typename Dtype>
void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {

#ifndef USE_FGPU
  AdamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(AdamUpdate<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate));
#endif
}
template void adam_update_gpu<float>(int, float*, float*, float*,
    float, float, float, float);
template void adam_update_gpu<double>(int, double*, double*, double*,
    double, double, double, double);

}  // namespace caffe
