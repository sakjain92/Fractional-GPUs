#include <vector>

#include "caffe/layers/crop_layer.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
__device__ int compute_uncropped_index(
    int index,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets) {
  int dest_index = index;
  int src_index = 0;
  for (int i = 0; i < ndims; ++i) {
      int coord = dest_index / dest_strides[i];
      dest_index -= coord * dest_strides[i];
      src_index += src_strides[i] * (coord + offsets[i]);
  }
  return src_index;
}

template <typename Dtype>
__global__ void crop_kernel_forward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    dest[index] = src[src_index];
  }
}

template <typename Dtype>
__global__ void crop_kernel_backward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    Dtype* src, const Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    src[src_index] = dest[index];
  }
}
#else

__device__ int compute_uncropped_index(
    fgpu_dev_ctx_t *ctx,
    int index,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets) {
  int dest_index = index;
  int src_index = 0;
  for (int i = 0; i < ndims; ++i) {
      int dest_strides_val = FGPU_COLOR_LOAD(ctx, &dest_strides[i]);
      int src_strides_val = FGPU_COLOR_LOAD(ctx, &src_strides[i]);
      int offsets_val = FGPU_COLOR_LOAD(ctx, &offsets[i]);

      int coord = dest_index / dest_strides_val;
      dest_index -= coord * dest_strides_val;
      src_index += src_strides_val * (coord + offsets_val);
  }
  return src_index;
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(crop_kernel_forward, const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    const Dtype* src, Dtype* dest) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      int src_index = compute_uncropped_index(ctx,
          index, ndims, src_strides, dest_strides, offsets);
      FGPU_COLOR_STORE(ctx, &dest[index], FGPU_COLOR_LOAD(ctx, &src[src_index]));
    }

  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(crop_kernel_backward, const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    Dtype* src, const Dtype* dest) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      int src_index = compute_uncropped_index(ctx,
          index, ndims, src_strides, dest_strides, offsets);
      FGPU_COLOR_STORE(ctx, &src[src_index], FGPU_COLOR_LOAD(ctx, &dest[index]));
    }

  }
}

#endif

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int n = top[0]->count();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  crop_kernel_forward<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
      bottom[0]->num_axes(),
      src_strides_.gpu_data(),
      dest_strides_.gpu_data(),
      offsets.gpu_data(),
      bottom_data, top_data);
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(crop_kernel_forward<Dtype>,
      CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, n,
      bottom[0]->num_axes(),
      src_strides_.gpu_data(),
      dest_strides_.gpu_data(),
      offsets.gpu_data(),
      bottom_data, top_data));

#endif
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->count();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
#ifndef USE_FGPU
    // NOLINT_NEXT_LINE(whitespace/operators)
    crop_kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
        bottom[0]->num_axes(),
        src_strides_.gpu_data(),
        dest_strides_.gpu_data(),
        offsets.gpu_data(),
        bottom_diff, top_diff);
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(crop_kernel_backward<Dtype>,
        CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, n,
        bottom[0]->num_axes(),
        src_strides_.gpu_data(),
        dest_strides_.gpu_data(),
        offsets.gpu_data(),
        bottom_diff, top_diff));
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
