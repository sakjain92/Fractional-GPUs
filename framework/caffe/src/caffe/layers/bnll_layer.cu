#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

#ifndef USE_FGPU
template <typename Dtype>
__global__ void BNLLForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ?
        in[index] + log(1. + exp(-in[index])) :
        log(1. + exp(in[index]));
  }
}


template <typename Dtype>
__global__ void BNLLBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(BNLLForward, const int n, const Dtype* in, Dtype* out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_val = FGPU_COLOR_LOAD(ctx, &in[index]);

      FGPU_COLOR_STORE(ctx, &out[index], in_val > 0 ?
          in_val + log(1. + exp(-in_val)) :
          log(1. + exp(in_val)));
    }
    
  }
}


template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(BNLLBackward, const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype expval = exp(min(FGPU_COLOR_LOAD(ctx, &in_data[index]), Dtype(kBNLL_THRESHOLD)));
      FGPU_COLOR_STORE(ctx, &out_diff[index], 
              FGPU_COLOR_LOAD(ctx, &in_diff[index]) * expval / (expval + 1.));
    }
 
  }
}

#endif

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(BNLLForward<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, bottom_data, top_data));

#endif
}


template <typename Dtype>
void BNLLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    BNLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(BNLLBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, top_diff, bottom_data, bottom_diff));

#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLLLayer);


}  // namespace caffe
