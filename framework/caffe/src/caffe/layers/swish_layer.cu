#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void SwishBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* sigmoid_output_data, Dtype* out_diff,
    const Dtype beta) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype swish_x = out_data[index];
    out_diff[index] = in_diff[index] * (beta * swish_x
        + sigmoid_output_data[index] * (1 - beta * swish_x));
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(SwishBackward, const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* sigmoid_output_data, Dtype* out_diff,
    const Dtype beta) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      const Dtype swish_x = FGPU_COLOR_LOAD(ctx, &out_data[index]);
      FGPU_COLOR_STORE(ctx, &out_diff[index],
              FGPU_COLOR_LOAD(ctx, &in_diff[index]) * (beta * swish_x
          + FGPU_COLOR_LOAD(ctx, &sigmoid_output_data[index]) * (1 - beta * swish_x)));
    }

  }
}
#endif

template <typename Dtype>
void SwishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* sigmoid_input_data = sigmoid_input_->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype beta = this->layer_param_.swish_param().beta();
  caffe_copy(count, bottom_data, sigmoid_input_data);
  caffe_gpu_scal(count, beta, sigmoid_input_data);
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  caffe_gpu_mul(count, bottom_data, sigmoid_output_->gpu_data(), top_data);
}

template <typename Dtype>
void SwishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype beta = this->layer_param_.swish_param().beta();
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    SwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, sigmoid_output_data, bottom_diff, beta);
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(SwishBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, top_diff, top_data, sigmoid_output_data, bottom_diff, beta));

#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwishLayer);

}  // namespace caffe
