#include <vector>

#include "caffe/layers/clip_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU

__global__ void ClipForward(const int n, const float* in, float* out,
    float p_min, float p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmaxf(p_min, fminf(in[index], p_max));
  }
}

__global__ void ClipForward(const int n, const double* in, double* out,
    double p_min, double p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmax(p_min, fmin(in[index], p_max));
  }
}

template <typename Dtype>
__global__ void ClipBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype p_min, Dtype p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (
            in_data[index] >= p_min && in_data[index] <= p_max);
  }
}

#else // USE_FGPU

__global__ FGPU_DEFINE_KERNEL(ClipForward, const int n, const float* in, float* out,
    float p_min, float p_max) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &out[index], 
              fmaxf(p_min, fminf(FGPU_COLOR_LOAD(ctx, &in[index]), p_max)));
    }

  } 
}

__global__ FGPU_DEFINE_KERNEL(ClipForward, const int n, const double* in, double* out,
    double p_min, double p_max) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &out[index], 
              fmax(p_min, fmin(FGPU_COLOR_LOAD(ctx, &in[index]), p_max)));
    }

  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(ClipBackward, const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype p_min, Dtype p_max) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype in_data_val = FGPU_COLOR_LOAD(ctx, &in_data[index]);

      FGPU_COLOR_STORE(ctx, &out_diff[index], FGPU_COLOR_LOAD(ctx, &in_diff[index]) * (
              in_data_val >= p_min && in_data_val <= p_max));
    }

  } 
}


#endif // USE_FGPU
template <typename Dtype>
void ClipLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype p_min = this->layer_param_.clip_param().min();
  Dtype p_max = this->layer_param_.clip_param().max();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  ClipForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, p_min, p_max);
  CUDA_POST_KERNEL_CHECK;
#else

#if defined(FGPU_COMP_COLORING_ENABLE)
  /* TODO: Try to make a cleaner approach here */
  void (*func)(fgpu_dev_ctx_t, const int, const Dtype*, Dtype *,
    Dtype, Dtype) = static_cast<void (*)(fgpu_dev_ctx_t, const int, const Dtype*, Dtype *,
    Dtype, Dtype)>(ClipForward);
#else
  void (*func)(const int, const Dtype*, Dtype *,
    Dtype, Dtype) = static_cast<void (*)(const int, const Dtype*, Dtype *,
    Dtype, Dtype)>(ClipForward);
#endif

  FGPU_CHECK(FGPU_LAUNCH_KERNEL(func,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, bottom_data, top_data, p_min, p_max));
#endif
}

template <typename Dtype>
void ClipLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype p_min = this->layer_param_.clip_param().min();
    Dtype p_max = this->layer_param_.clip_param().max();
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    ClipBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, p_min, p_max);
    CUDA_POST_KERNEL_CHECK;
#else
   FGPU_CHECK(FGPU_LAUNCH_KERNEL(ClipBackward<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, top_diff, bottom_data, bottom_diff, p_min, p_max));

#endif
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ClipLayer);


}  // namespace caffe
