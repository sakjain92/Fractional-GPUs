#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

#else // USE_FGPU

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ReLUForward ,const int n, const Dtype* in, 
    Dtype* out, Dtype negative_slope) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_val = FGPU_COLOR_LOAD(ctx, &in[index]);
      FGPU_COLOR_STORE(ctx, &out[index], 
          in_val > 0 ? in_val : in_val * negative_slope);
    }
  } 
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ReLUBackward, const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index]);
      Dtype in_diff_val = FGPU_COLOR_LOAD(ctx, &in_diff[index]);

      FGPU_COLOR_STORE(ctx, &out_diff[index], in_diff_val * ((in_data_val > 0)
        + (in_data_val <= 0) * negative_slope));
    }
  } 
}

#endif // USE_FGPU

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
#ifndef USE_FGPU
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(ReLUForward<Dtype>, CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS, 0, count, bottom_data, top_data, negative_slope));
#endif 
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}



template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
#ifndef USE_FGPU
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
   CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(ReLUBackward<Dtype>, CAFFE_GET_BLOCKS(count),
	CAFFE_CUDA_NUM_THREADS, 0, count, top_diff, bottom_data, bottom_diff, 
        negative_slope));

#endif
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
