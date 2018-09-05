#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
  }
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(SigmoidForward, const int n, const Dtype* in, Dtype* out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &out[index],
              0.5 * tanh(0.5 * FGPU_COLOR_LOAD(ctx, &in[index])) + 0.5);
    }
  
  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(SigmoidBackward, const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      const Dtype sigmoid_x = FGPU_COLOR_LOAD(ctx, &out_data[index]);
      FGPU_COLOR_STORE(ctx, &out_diff[index], 
              FGPU_COLOR_LOAD(ctx, &in_diff[index]) * sigmoid_x * (1 - sigmoid_x));
    }

  }
}
#endif

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(SigmoidForward<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, bottom_data, top_data));
#endif

  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(SigmoidBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, top_diff, top_data, bottom_diff));
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);


}  // namespace caffe
