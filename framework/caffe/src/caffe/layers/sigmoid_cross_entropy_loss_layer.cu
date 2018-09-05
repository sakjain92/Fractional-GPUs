#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}

#else

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(SigmoidCrossEntropyLossForwardGPU, const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, nthreads, _blockIdx, _gridDim) {
      Dtype target_val_orig = FGPU_COLOR_LOAD(ctx, &target[i]);
      const int target_value = static_cast<int>(target_val_orig);
      if (has_ignore_label_ && target_value == ignore_label_) {
       loss[i] = 0;
        FGPU_COLOR_STORE(ctx, &counts[i], 0);
      } else {
        Dtype input_data_val = FGPU_COLOR_LOAD(ctx, &input_data[i]);

        FGPU_COLOR_STORE(ctx, &loss[i], input_data_val * (target_val_orig - (input_data_val >= 0)) -
            log(1 + exp(input_data_val - 2 * input_data_val *
            (input_data_val >= 0))));
        FGPU_COLOR_STORE(ctx, &counts[i], 1);
      }
    }

  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(SigmoidCrossEntropyLossIgnoreDiffGPU, const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(i, count, _blockIdx, _gridDim) {
      const int target_value = static_cast<int>(FGPU_COLOR_LOAD(ctx, &target[i]));
      if (target_value == ignore_label) {
        FGPU_COLOR_STORE(ctx, &diff[i], 0);
      }
    }

  }
}
#endif

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;

#ifndef USE_FGPU
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data,
      has_ignore_label_, ignore_label_, count_data);
#else
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(SigmoidCrossEntropyLossForwardGPU<Dtype>,
      CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
      count, input_data, target, loss_data,
      has_ignore_label_, ignore_label_, count_data));
#endif

  // Only launch another CUDA kernel if we actually need the valid count.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(count, count_data, &valid_count);
  } else {
    valid_count = count;
  }
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;

  // Clear scratch memory to prevent interfering with backward (see #6202).
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
#ifndef USE_FGPU
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
#else
      FGPU_CHECK(FGPU_LAUNCH_KERNEL(SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype>,
        CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0,
        count, ignore_label_, target, bottom_diff));
#endif
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
