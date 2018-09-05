#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
// CUDA kernele for forward
template <typename Dtype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void PReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void PReLUParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    for ( int k = 1; k < rows; k++ ) {
        out_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * (in_data[index + k*rowPitch] <= 0);
    }
  }
}

#else // USE_FGPU
// CUDA kernele for forward
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(PReLUForward, const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor) {
    
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      int c = (index / dim) % channels / div_factor;
      Dtype in_val = FGPU_COLOR_LOAD(ctx, &in[index]);
      FGPU_COLOR_STORE(ctx, &out[index],
              in_val > 0 ? in_val : in_val * FGPU_COLOR_LOAD(ctx, &slope_data[c]));
    }

  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(PReLUBackward, const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const int div_factor) {
  
    
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
      
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      int c = (index / dim) % channels / div_factor;
      Dtype in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index]);
      FGPU_COLOR_STORE(ctx, &out_diff[index],
              FGPU_COLOR_LOAD(ctx, &in_diff[index]) * ((in_data_val > 0)
          + (in_data_val <= 0) * FGPU_COLOR_LOAD(ctx, &slope_data[c])));
    }

  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(PReLUParamBackward, const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index]);
      Dtype in_diff_val = FGPU_COLOR_LOAD(ctx, &in_diff[index]);
      Dtype *out_diff_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &out_diff[index]);

      *out_diff_addr = in_diff_val * in_data_val * (in_data_val <= 0);
      for ( int k = 1; k < rows; k++ ) {
          in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index + k*rowPitch]);
          in_diff_val = FGPU_COLOR_LOAD(ctx, &in_diff[index + k*rowPitch]);

          *out_diff_addr += in_diff_val * in_data_val * (in_data_val <= 0);
      }
    }

  }
}

#endif

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(PReLUForward<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, channels, dim, bottom_data, top_data, slope_data, div_factor));
#endif
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;

#ifndef USE_FGPU
    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(PReLUParamBackward<Dtype>,
      CAFFE_GET_BLOCKS(cdim), CAFFE_CUDA_NUM_THREADS, 0,
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_.mutable_gpu_diff()));
#endif
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;

#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(PReLUBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor));
#endif
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe
