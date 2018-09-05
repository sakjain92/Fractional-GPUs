#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU

template <typename Dtype>
__global__ void EmbedForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* weight, const int M, const int N, const int K,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    #ifdef DEBUG
        assert(index >= 0);
        assert(index < K);
        assert(static_cast<Dtype>(index) == bottom_data[n]);
    #endif
    const int weight_index = index * N + d;
    top_data[top_index] = weight[weight_index];
  }
}

template <typename Dtype>
__global__ void EmbedBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int M, const int N, const int K,
    Dtype* weight_diff);

template <typename Dtype>
__global__ void EmbedBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_diff, const int M, const int N, const int K,
    Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    const int weight_index = index * N + d;
    caffe_gpu_atomic_add(top_diff[top_index], weight_diff + weight_index);
  }
}

#else // USE_FGPU

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(EmbedForward, const int nthreads, 
    const Dtype* bottom_data, const Dtype* weight, const int M, const int N, 
    const int K, Dtype* top_data) {
  
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(top_index, nthreads, _blockIdx, _gridDim) {
      const int n = top_index / N;
      const int d = top_index % N;
      Dtype bdata = FGPU_COLOR_LOAD(ctx, &bottom_data[n]);
      const int index = static_cast<int>(bdata);
      #ifdef DEBUG
        assert(index >= 0);
        assert(index < K);
        assert(static_cast<Dtype>(index) == bdata);
      #endif
      const int weight_index = index * N + d;
      FGPU_COLOR_STORE(ctx, &top_data[top_index], 
        FGPU_COLOR_LOAD(ctx, &weight[weight_index]));
    }
  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(EmbedBackward, const int nthreads,
    const Dtype* bottom_data, const Dtype* top_diff, const int M, 
    const int N, const int K, Dtype* weight_diff) {
    
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(top_index, nthreads, _blockIdx, _gridDim) {
      const int n = top_index / N;
      const int d = top_index % N;
      Dtype bdata = FGPU_COLOR_LOAD(ctx, &bottom_data[n]);
      const int index = static_cast<int>(bdata);
      const int weight_index = index * N + d;
      caffe_gpu_atomic_add(FGPU_COLOR_LOAD(ctx, &top_diff[top_index]), 
        FGPU_COLOR_TRANSLATE_ADDR(ctx, &weight_diff[weight_index]));
    }
  }
}

#endif // USE_FGPU

template <typename Dtype>
void EmbedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
#ifndef USE_FGPU
  EmbedForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight, M_, N_, K_, top_data);
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(EmbedForward<Dtype>, CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS, 0, count, bottom_data, weight, M_, N_, K_, top_data));
#endif
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), Dtype(1), top_data);
  }
}

template <typename Dtype>
void EmbedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const int top_count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
#ifndef USE_FGPU
    EmbedBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, bottom_data, top_diff, M_, N_, K_, weight_diff);
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(EmbedBackward<Dtype>, CAFFE_GET_BLOCKS(top_count), 
        CAFFE_CUDA_NUM_THREADS, 0, top_count, bottom_data, top_diff, M_, N_, K_, 
        weight_diff));
#endif
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.gpu_data(), Dtype(1), bias_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EmbedLayer);

}  // namespace caffe
