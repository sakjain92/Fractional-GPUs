#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        alpha * (exp(in[index]) - 1);
  }
}

template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? in_diff[index] :
        in_diff[index] * (out_data[index] + alpha);
  }
}

#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ELUForward, const int n, const Dtype* in, Dtype* out,
    Dtype alpha) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_val = FGPU_COLOR_LOAD(ctx, &in[index]);
      FGPU_COLOR_STORE(ctx, &out[index], in_val > 0 ? in_val :
          alpha * (exp(in_val) - 1));
    }
 
  } 
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ELUBackward, const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha) {
  
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index]);
      Dtype in_diff_val = FGPU_COLOR_LOAD(ctx, &in_diff[index]);
      Dtype out_data_val = FGPU_COLOR_LOAD(ctx, &out_data[index]);

      FGPU_COLOR_STORE(ctx, &out_diff[index], in_data_val > 0 ? in_diff_val :
          in_diff_val * (out_data_val + alpha));
    }
  
  } 
}

#endif

template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype alpha = this->layer_param_.elu_param().alpha();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  ELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, alpha);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(ELUForward<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, bottom_data, top_data, alpha));

#endif
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.elu_param().alpha();
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    ELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_data, bottom_diff, alpha);
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(ELUBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, top_diff, top_data, bottom_data, bottom_diff, alpha));
#endif
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);


}  // namespace caffe
