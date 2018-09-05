#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU
template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

#else // USE_FGPU

template <typename Dtype>
__global__
FGPU_DEFINE_KERNEL(MaxForward ,const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      Dtype maxval = -FLT_MAX;
      Dtype bdata_a = FGPU_COLOR_LOAD(ctx, &bottom_data_a[index]);
      Dtype bdata_b = FGPU_COLOR_LOAD(ctx, &bottom_data_b[index]);

      int maxidx = -1;
      if (bdata_a > bdata_b) {
        // only update for very first bottom_data blob (blob_idx == 0)
        if (blob_idx == 0) {
          maxval = bdata_a;
          maxidx = blob_idx;
          FGPU_COLOR_STORE(ctx, &top_data[index], maxval);
          FGPU_COLOR_STORE(ctx, &mask[index], maxidx);
        }
      } else {
        maxval = bdata_b;
        maxidx = blob_idx + 1;
        FGPU_COLOR_STORE(ctx, &top_data[index], maxval);
        FGPU_COLOR_STORE(ctx, &mask[index], maxidx);
      }
    }
  } 
}

template <typename Dtype>
__global__
FGPU_DEFINE_KERNEL(MaxBackward, const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, nthreads, _blockIdx, _gridDim) {
      Dtype gradient = 0;
      if (FGPU_COLOR_LOAD(ctx, &mask[index]) == blob_idx) {
        gradient += FGPU_COLOR_LOAD(ctx, &top_diff[index]);
      }
      FGPU_COLOR_STORE(ctx, &bottom_diff[index], gradient);
    }
  } 
}

#endif // USE_FGPU

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
#ifndef USE_FGPU
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
 
#else
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(MaxForward<Dtype>, CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS, 0, count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
        0, top_data, mask));
#endif
   for (int i = 2; i < bottom.size(); ++i) {
#ifndef USE_FGPU
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
#else
      FGPU_CHECK(FGPU_LAUNCH_KERNEL( MaxForward<Dtype>, CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS, 0, count, top_data, bottom[i]->gpu_data(), 
          i-1, top_data, mask));
#endif    
   }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}


template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.gpu_data();
#ifndef USE_FGPU
        MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, i, mask, bottom_diff);
#else
        FGPU_CHECK(FGPU_LAUNCH_KERNEL(MaxBackward<Dtype>, CAFFE_GET_BLOCKS(count), 
	    CAFFE_CUDA_NUM_THREADS, 0, count, top_diff, i, mask, bottom_diff));
#endif
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
