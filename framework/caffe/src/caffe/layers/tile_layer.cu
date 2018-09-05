#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void Tile(const int nthreads, const Dtype* bottom_data,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int b = (index / tile_size / num_tiles) % bottom_tile_axis;
    const int n = index / tile_size / num_tiles / bottom_tile_axis;
    const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
__global__ void TileBackward(const int nthreads, const Dtype* top_diff,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int b = (index / tile_size) % bottom_tile_axis;
    const int n = index / tile_size / bottom_tile_axis;
    bottom_diff[index] = 0;
    int top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;
    for (int t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += bottom_tile_axis * tile_size;
    }
  }
}

#else  // USE_FGPU
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(Tile, const int nthreads, const Dtype* bottom_data,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* top_data) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      const int d = index % tile_size;
      const int b = (index / tile_size / num_tiles) % bottom_tile_axis;
      const int n = index / tile_size / num_tiles / bottom_tile_axis;
      const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
      FGPU_COLOR_STORE(ctx, &top_data[index],
              FGPU_COLOR_LOAD(ctx, &bottom_data[bottom_index]));
    }

  } 
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(TileBackward, const int nthreads, const Dtype* top_diff,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* bottom_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      const int d = index % tile_size;
      const int b = (index / tile_size) % bottom_tile_axis;
      const int n = index / tile_size / bottom_tile_axis;
      Dtype *bottom_diff_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &bottom_diff[index]);
      *bottom_diff_addr = 0;
      int top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;
      for (int t = 0; t < num_tiles; ++t) {
        *bottom_diff_addr += FGPU_COLOR_LOAD(ctx, &top_diff[top_index]);
        top_index += bottom_tile_axis * tile_size;
      }
    }
 
  } 
}

#endif // USE_FGPU

template <typename Dtype>
void TileLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int bottom_tile_axis = bottom[0]->shape(axis_);
  const int nthreads = top[0]->count();
#ifndef USE_FGPU
  Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom_data, inner_dim_, tiles_, bottom_tile_axis, top_data);
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(Tile<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
      nthreads, bottom_data, inner_dim_, tiles_, bottom_tile_axis, top_data));
#endif
}

template <typename Dtype>
void TileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_tile_axis = bottom[0]->shape(axis_);
  const int tile_size = inner_dim_ / bottom_tile_axis;
  const int nthreads = bottom[0]->count();
#ifndef USE_FGPU
  TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, top_diff, tile_size, tiles_, bottom_tile_axis, bottom_diff);
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(TileBackward<Dtype>,  // NOLINT_NEXT_LINE(whitespace/operators)
      CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
      nthreads, top_diff, tile_size, tiles_, bottom_tile_axis, bottom_diff));
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(TileLayer);

}  // namespace caffe
