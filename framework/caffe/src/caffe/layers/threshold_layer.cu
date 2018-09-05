#include <vector>

#include "caffe/layers/threshold_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void ThresholdForward(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ThresholdForward, const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &out[index],
              FGPU_COLOR_LOAD(ctx, &in[index]) > threshold ? 1 : 0);
    }

  }
}
#endif

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(ThresholdForward<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, threshold_, bottom_data, top_data));

#endif
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);


}  // namespace caffe
