#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(Concat, const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      const int total_concat_size = concat_size * bottom_concat_axis;
      const int concat_num = index / total_concat_size;
      const int concat_index = index % total_concat_size;
      const int top_index = concat_index +
          (concat_num * top_concat_axis + offset_concat_axis) * concat_size;

      if (forward) {
        FGPU_COLOR_STORE(ctx, &out_data[top_index],
                FGPU_COLOR_LOAD(ctx, &in_data[index]));
      } else {
        FGPU_COLOR_STORE(ctx, &out_data[index],
                FGPU_COLOR_LOAD(ctx, &in_data[top_index]));
      }
    }

  } 
}

#endif

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
#ifndef USE_FGPU
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(Concat<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
        CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data));

#endif
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
#ifndef USE_FGPU
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(Concat<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
          CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff));

#endif
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
