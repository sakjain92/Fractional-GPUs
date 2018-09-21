#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* const in,
    const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}


template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

#else // USE_FGPU

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(LRNFillScale, const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      // find out the local offset
      const int w = index % width;
      const int h = (index / width) % height;
      const int n = index / width / height;
      const int offset = (n * channels * height + h) * width + w;
      const int step = height * width;
      const Dtype* const in_off = in + offset;
      Dtype* const scale_off = scale + offset;
      int head = 0;
      const int pre_pad = (size - 1) / 2;
      const int post_pad = size - pre_pad - 1;
      Dtype accum_scale = 0;
      // fill the scale at [n, :, h, w]
      // accumulate values
      while (head < post_pad && head < channels) {
        Dtype in_off_val = FGPU_COLOR_LOAD(ctx, &in_off[head * step]);
        accum_scale +=  in_off_val * in_off_val;
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        Dtype in_off_val = FGPU_COLOR_LOAD(ctx, &in_off[head * step]);
        accum_scale += in_off_val * in_off_val;
        if (head - size >= 0) {
          in_off_val = FGPU_COLOR_LOAD(ctx, &in_off[(head - size) * step]);
          accum_scale -= in_off_val * in_off_val;
        }
        FGPU_COLOR_STORE(ctx, &scale_off[(head - post_pad) * step],
                k + accum_scale * alpha_over_size);
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          Dtype in_off_val = FGPU_COLOR_LOAD(ctx, &in_off[(head - size) * step]);
          accum_scale -= in_off_val * in_off_val;
        }
        FGPU_COLOR_STORE(ctx, &scale_off[(head - post_pad) * step],
                k + accum_scale * alpha_over_size);
        ++head;
      }
    }

  } 
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(LRNComputeOutput, const int nthreads, const Dtype* const in,
    const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    
    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &out[index],
        FGPU_COLOR_LOAD(ctx, &in[index]) * 
        pow(FGPU_COLOR_LOAD(ctx, &scale[index]), negative_beta));
    }

  } 
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(LRNComputeDiff, const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      // find out the local offset
      const int w = index % width;
      const int h = (index / width) % height;
      const int n = index / width / height;
      const int offset = (n * channels * height + h) * width + w;
      const int step = height * width;
      const Dtype* const bottom_off = bottom_data + offset;
      const Dtype* const top_off = top_data + offset;
      const Dtype* const scale_off = scale + offset;
      const Dtype* const top_diff_off = top_diff + offset;
      Dtype* const bottom_diff_off = bottom_diff + offset;
      int head = 0;
      const int pre_pad = size - (size + 1) / 2;
      const int post_pad = size - pre_pad - 1;
      Dtype accum_ratio = 0;
      // accumulate values
      while (head < post_pad && head < channels) {
        accum_ratio += FGPU_COLOR_LOAD(ctx, &top_diff_off[head * step]) * 
            FGPU_COLOR_LOAD(ctx, &top_off[head * step]) /
            FGPU_COLOR_LOAD(ctx, &scale_off[head * step]);
        ++head;
      }
      // both add and subtract
      while (head < channels) {
        accum_ratio += FGPU_COLOR_LOAD(ctx, &top_diff_off[head * step]) * 
            FGPU_COLOR_LOAD(ctx, &top_off[head * step]) /
            FGPU_COLOR_LOAD(ctx, &scale_off[head * step]);
        if (head - size >= 0) {
          accum_ratio -= FGPU_COLOR_LOAD(ctx, &top_diff_off[(head - size) * step]) *
              FGPU_COLOR_LOAD(ctx, &top_off[(head - size) * step]) / 
              FGPU_COLOR_LOAD(ctx, &scale_off[(head - size) * step]);
        }
        FGPU_COLOR_STORE(ctx, &bottom_diff_off[(head - post_pad) * step],
            FGPU_COLOR_LOAD(ctx, &top_diff_off[(head - post_pad) * step]) *
              pow(FGPU_COLOR_LOAD(ctx, &scale_off[(head - post_pad) * step]), negative_beta)
            - cache_ratio * 
            FGPU_COLOR_LOAD(ctx, &bottom_off[(head - post_pad) * step]) * 
            accum_ratio);
        ++head;
      }
      // subtract only
      while (head < channels + post_pad) {
        if (head - size >= 0) {
          accum_ratio -= FGPU_COLOR_LOAD(ctx, &top_diff_off[(head - size) * step]) *
              FGPU_COLOR_LOAD(ctx, &top_off[(head - size) * step]) / 
              FGPU_COLOR_LOAD(ctx, &scale_off[(head - size) * step]);
        }
        FGPU_COLOR_STORE(ctx, &bottom_diff_off[(head - post_pad) * step],
            FGPU_COLOR_LOAD(ctx, &top_diff_off[(head - post_pad) * step]) *
            pow(FGPU_COLOR_LOAD(ctx, &scale_off[(head - post_pad) * step]), negative_beta) -
            cache_ratio * 
            FGPU_COLOR_LOAD(ctx, &bottom_off[(head - post_pad) * step]) * 
            accum_ratio);
        ++head;
      }
    }

  }
}

#endif

template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(LRNFillScale<Dtype>,
      CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS, 0,
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data));
#endif
  n_threads = bottom[0]->count();
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(LRNComputeOutput<Dtype>,
      CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS, 0,
      n_threads, bottom_data, scale_data, -beta_, top_data));
#endif
}
template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int n_threads = num_ * height_ * width_;
#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff());
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(LRNComputeDiff<Dtype>,
      CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS, 0,
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff()));

#endif
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);



INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
