#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}
#else
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(Slice, const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
 
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      const int total_slice_size = slice_size * top_slice_axis;
      const int slice_num = index / total_slice_size;
      const int slice_index = index % total_slice_size;
      const int bottom_index = slice_index +
          (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
      if (forward) {
        FGPU_COLOR_STORE(ctx, &out_data[index], FGPU_COLOR_LOAD(ctx, &in_data[bottom_index]));
      } else {
        FGPU_COLOR_STORE(ctx, &out_data[bottom_index], FGPU_COLOR_LOAD(ctx, &in_data[index]));
      }
    }

  } 
}
#endif

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
#ifndef USE_FGPU
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
#else 
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(Slice<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
        CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data));
#endif
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
#ifndef USE_FGPU
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(Slice<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
        CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff));
#endif
    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);

}  // namespace caffe
