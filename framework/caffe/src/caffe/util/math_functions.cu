#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

#ifdef USE_FGPU
#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>
#endif

namespace caffe {

#ifndef USE_FGPU

/* With FGPU, can't use closed source GPU libraries (i.e. can't use cublas) */
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

#if 0 /* Currently not in use in the code-base. No equivalent in FGPU currently */
template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}
#endif

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}


template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void sign_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = (Dtype(0) < x[index]) - (x[index] < Dtype(0));
  }
}

template <>
void caffe_gpu_sign<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sign_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

template <>
void caffe_gpu_sign<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sign_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

template <typename Dtype>
__global__ void sgnbit_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = signbit(x[index]);
  }
}

template <>
void caffe_gpu_sgnbit<float>(const int N, const float* x, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sgnbit_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

template <>
void caffe_gpu_sgnbit<double>(const int N, const double* x, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sgnbit_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y);
}

#else  /* USE_FGPU */

#include <cooperative_groups.h>

/* TODO: Use shfl_down to reduce summation time */
/* Takes sum of all values of threads in a block and do one atomic addition */
template <typename Dtype>
__device__ void reduce_sum(cooperative_groups::thread_group g, Dtype val, Dtype *out)
{
  int lane = g.thread_rank();
  __shared__ Dtype temp [CAFFE_CUDA_NUM_THREADS];

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2)
  {
    temp[lane] = val;
    g.sync(); // wait for all threads to store
    if(lane < i) 
      val += temp[lane + i];
    g.sync(); // wait for all threads to load
  }

  if (g.thread_rank() == 0)
    caffe_gpu_atomic_add(val, out);
}

template <typename Dtype, size_t BLOCK_SIZE> 
__global__ FGPU_DEFINE_KERNEL(gemm_kernel, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int hA, const int wA, 
        const int hB, const int wB, const Dtype *A, const Dtype *B, 
        Dtype *C, const Dtype alpha, const Dtype beta)
{
  // TODO: This can be optimized to load A matrix only once
  // Need API to collect a bunch of blockIdx together
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx;
  ctx = FGPU_DEVICE_INIT();

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    // Block index
    int bx = _blockIdx.x;
    int by = _blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aRow = BLOCK_SIZE * by;
    int aColBegin = 0;

    // Index of the last sub-matrix of A processed by the block
    int aColEnd = wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aColStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bRowBegin = 0;
    int bCol = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bRowStep  = BLOCK_SIZE;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    Dtype Csub = 0;

    // Loop over all the sub-matrices of A and B    
    // required to compute the block sub-matrix
    for (int aCol = aColBegin, bRow = bRowBegin, istep=0;
            aCol <= aColEnd;
            aCol += aColStep, bRow += bRowStep, ++istep)
    {
        int aIndex, bIndex;

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

        aIndex = (TransA == CblasNoTrans) ? (wA * (aRow + ty) + aCol + tx) :
        (hA * (aCol + tx) + aRow + ty);

        bIndex = (TransB == CblasNoTrans) ? (wB * (bRow + ty) + bCol + tx) :
        (hB * (bCol + tx) + bRow + ty);

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        if ((aRow + ty < hA) && (aCol + tx < wA))
          As[ty][tx] = FGPU_COLOR_LOAD(ctx, &A[aIndex]);
        else
          As[ty][tx] = 0;

        if ((bRow + ty < hB) && (bCol + tx < wB))
          Bs[ty][tx] = FGPU_COLOR_LOAD(ctx, &B[bIndex]);
        else
          Bs[ty][tx] = 0;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int cRow = BLOCK_SIZE * by + ty;
    int cCol = BLOCK_SIZE * bx + tx;
    int cIndex = wB * cRow + cCol;
    
    if ((cRow < hA) && (cCol < wB)) {
        Dtype *C_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &C[cIndex]);
        *C_addr = alpha * Csub + beta * (*C_addr);
    } 
  } 
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float *A, const float *B, const float beta,
    float *C) {

  const int block_size = 32;
  dim3 threads(block_size, block_size);
  dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL((gemm_kernel<float, block_size>), grid, threads, 0,
    TransA, TransB, M, K, K, N, A, B, C, alpha, beta));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double *A, const double *B, const double beta,
    double *C) {

  const int block_size = 32;
  dim3 threads(block_size, block_size);
  dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL((gemm_kernel<double, block_size>), grid, threads, 0,
    TransA, TransB, M, K, K, N, A, B, C, alpha, beta));
}

template <typename Dtype> 
__global__ FGPU_DEFINE_KERNEL(gemv_kernel, const CBLAS_TRANSPOSE TransA,
    const int hA, const int wA, const Dtype *A, const Dtype *x, 
        Dtype *y, const Dtype alpha, const Dtype beta)
{
  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx;
  ctx = FGPU_DEVICE_INIT();

  cooperative_groups::thread_group  g = cooperative_groups::this_thread_block();
  
  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    Dtype sum = 0;
    int row = _blockIdx.x;
    Dtype *y_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &y[row]);

    for (int col = threadIdx.x; col < wA; col += blockDim.x) {
      Dtype x_val = FGPU_COLOR_LOAD(ctx, &x[col]);
      if (TransA == CblasNoTrans)
        sum += FGPU_COLOR_LOAD(ctx, &A[wA * row + col]) * x_val;
      else
        sum += FGPU_COLOR_LOAD(ctx, &A[hA * col + row]) * x_val;
    }

    sum = alpha * sum;

    if (g.thread_rank() == 0)
      *y_addr *= beta;

    g.sync();

    reduce_sum<Dtype>(g, sum, y_addr);
  }   

}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float *A, const float *x,
    const float beta, float *y) {
  if (TransA == CblasNoTrans) {
    dim3 grid(M), threads(CAFFE_CUDA_NUM_THREADS);
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(gemv_kernel<float>, grid, threads, 0,
      TransA, M, N, A, x, y, alpha, beta));
  } else {
    dim3 grid(N), threads(CAFFE_CUDA_NUM_THREADS);
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(gemv_kernel<float>, grid, threads, 0,
      TransA, N, M, A, x, y, alpha, beta));
  }
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double *A, const double *x,
    const double beta, double *y) {
  if (TransA == CblasNoTrans) {
    dim3 grid(M), threads(CAFFE_CUDA_NUM_THREADS);
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(gemv_kernel<double>, grid, threads, 0,
      TransA, M, N, A, x, y, alpha, beta));
  } else {
    dim3 grid(N), threads(CAFFE_CUDA_NUM_THREADS);
    FGPU_CHECK(FGPU_LAUNCH_KERNEL(gemv_kernel<double>, grid, threads, 0,
      TransA, N, M, A, x, y, alpha, beta));
  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(axpy_kernel, const int n, const Dtype alpha,
        const Dtype *x, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype *y_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &y[index]);
      const Dtype *x_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &x[index]);
      *y_addr += alpha * (*x_addr);
    }
  }

}

template <>
void caffe_gpu_axpy(int n, float alpha, float const *x, float *y)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(axpy_kernel<float>, grid, threads, 0,
      n, alpha, x, y));
}

template <>
void caffe_gpu_axpy<double>(int n, double alpha, double const *x, double *y)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(axpy_kernel<double>, grid, threads, 0,
      n, alpha, x, y));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    FGPU_CHECK(fgpu_memory_copy_async(Y, X, N, FGPU_COPY_DEFAULT));
    FGPU_CHECK(fgpu_color_stream_synchronize());
  }
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(scal_kernel, const int n, const Dtype alpha,
        Dtype *x) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype *x_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &x[index]);
      *x_addr *= alpha;
    }
  }

}

template <>
void caffe_gpu_scal<float>(const int n, float alpha, float *x)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(scal_kernel<float>, grid, threads, 0,
      n, alpha, x));
}


template <>
void caffe_gpu_scal<double>(const int n, double alpha, double *x)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(scal_kernel<double>, grid, threads, 0,
      n, alpha, x));
}

static void *initialize_temp_variable(void)
{
  void *temp;

  CUDA_CHECK(cudaMalloc(&temp, sizeof(double)));
  CUDA_CHECK(cudaMemset(temp, 0, sizeof(double)));

  return temp;
}


static void deinitialize_temp_variable(void *temp)
{
  CUDA_CHECK(cudaFree(temp));
}


template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(dot_kernel, const int n, const Dtype *x,
        const Dtype *y, Dtype *out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  Dtype sum = 0;
  cooperative_groups::thread_group g = cooperative_groups::this_thread_block();

  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      sum += FGPU_COLOR_LOAD(ctx, &x[index]) * FGPU_COLOR_LOAD(ctx, &y[index]);
    }
  }

  reduce_sum<Dtype>(g, sum, FGPU_COLOR_TRANSLATE_ADDR(ctx, out));
}

template <>
void caffe_gpu_dot<float>(const int n, const float *x, const float *y, float *out)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  void *tempResult = initialize_temp_variable();
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(dot_kernel<float>, grid, threads, 0,
      n, x, y, (float *)tempResult));
  CUDA_CHECK(cudaMemcpy(out, tempResult, sizeof(float), cudaMemcpyDefault));
  deinitialize_temp_variable(tempResult);
}

template <>
void caffe_gpu_dot<double>(const int n, const double *x, const double *y, double *out)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  void *tempResult = initialize_temp_variable();
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(dot_kernel<double>, grid, threads, 0,
      n, x, y, (double *)tempResult));
  CUDA_CHECK(cudaMemcpy(out, tempResult, sizeof(double), cudaMemcpyDefault));
  deinitialize_temp_variable(tempResult);
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(asum_kernel, const int n, const Dtype *x,
        Dtype *out) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  Dtype sum = 0;
  cooperative_groups::thread_group g = cooperative_groups::this_thread_block();

  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      sum += fabs(FGPU_COLOR_LOAD(ctx, &x[index]));
    }
  }

  reduce_sum<Dtype>(g, sum, FGPU_COLOR_TRANSLATE_ADDR(ctx, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float *x, float *out)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  void *tempResult = initialize_temp_variable();
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(asum_kernel<float>, grid, threads, 0,
      n, x, (float *)tempResult));
  CUDA_CHECK(cudaMemcpy(out, tempResult, sizeof(float), cudaMemcpyDefault));
  deinitialize_temp_variable(tempResult);
}

template <>
void caffe_gpu_asum<double>(const int n, const double *x, double *out)
{
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  void *tempResult = initialize_temp_variable();
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(asum_kernel<double>, grid, threads, 0,
      n, x, (double *)tempResult));
  CUDA_CHECK(cudaMemcpy(out, tempResult, sizeof(double), cudaMemcpyDefault));
  deinitialize_temp_variable(tempResult);
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(scale_kernel, const int n, const Dtype alpha,
        const Dtype *x, Dtype *y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], alpha * FGPU_COLOR_LOAD(ctx, &x[index]));
    }
  }

}
template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(scale_kernel<float>, grid, threads, 0,
      n, alpha, x, y));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(n)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(scale_kernel<double>, grid, threads, 0,
      n, alpha, x, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(set_kernel, const int n, const Dtype alpha,
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], alpha);
    }
  }

}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);

  if (alpha == 0) {
    FGPU_CHECK(fgpu_memory_memset_async(Y, 0, sizeof(Dtype) * N));
    FGPU_CHECK(fgpu_color_stream_synchronize());
    return;
  }
    
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(set_kernel<Dtype>, grid, threads, 0,
      N, alpha, Y));
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}


template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(add_scalar_kernel, const int n, const Dtype alpha,
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype *y_addr = FGPU_COLOR_TRANSLATE_ADDR(ctx, &y[index]);
      *y_addr += alpha;
    }
  }

}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(add_scalar_kernel<float>, grid, threads, 0,
      N, alpha, Y));
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(add_scalar_kernel<double>, grid, threads, 0,
      N, alpha, Y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(add_kernel, const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype a_val = FGPU_COLOR_LOAD(ctx, &a[index]);
      Dtype b_val = FGPU_COLOR_LOAD(ctx, &b[index]);
      FGPU_COLOR_STORE(ctx, &y[index], a_val + b_val);
    }
  }

}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(add_kernel<float>, grid, threads, 0,
      N, a, b, y));
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(add_kernel<double>, grid, threads, 0,
      N, a, b, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(sub_kernel, const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype a_val = FGPU_COLOR_LOAD(ctx, &a[index]);
      Dtype b_val = FGPU_COLOR_LOAD(ctx, &b[index]);
      FGPU_COLOR_STORE(ctx, &y[index], a_val - b_val);

    }
  }

}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sub_kernel<float>, grid, threads, 0,
      N, a, b, y));
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sub_kernel<double>, grid, threads, 0,
      N, a, b, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(mul_kernel, const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype a_val = FGPU_COLOR_LOAD(ctx, &a[index]);
      Dtype b_val = FGPU_COLOR_LOAD(ctx, &b[index]);
      FGPU_COLOR_STORE(ctx, &y[index], a_val * b_val);
    }
  }

}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(mul_kernel<float>, grid, threads, 0,
      N, a, b, y));
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(mul_kernel<double>, grid, threads, 0,
      N, a, b, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(div_kernel, const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype a_val = FGPU_COLOR_LOAD(ctx, &a[index]);
      Dtype b_val = FGPU_COLOR_LOAD(ctx, &b[index]);
      FGPU_COLOR_STORE(ctx, &y[index], a_val / b_val);
    }
  }

}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(div_kernel<float>, grid, threads, 0,
      N, a, b, y));
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(div_kernel<double>, grid, threads, 0,
      N, a, b, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(abs_kernel, const int n, const Dtype* a,
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], abs(FGPU_COLOR_LOAD(ctx, &a[index])));
    }
  }

}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(abs_kernel<float>, grid, threads, 0,
      N, a, y));
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(abs_kernel<double>, grid, threads, 0,
      N, a, y));
}


template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(exp_kernel, const int n, const Dtype* a,
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], exp(FGPU_COLOR_LOAD(ctx, &a[index])));
    }
  }

}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(exp_kernel<float>, grid, threads, 0,
      N, a, y));
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(exp_kernel<double>, grid, threads, 0,
      N, a, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(log_kernel, const int n, const Dtype* a,
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], log(FGPU_COLOR_LOAD(ctx, &a[index])));
    }
  }

}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(log_kernel<float>, grid, threads, 0,
      N, a, y));
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(log_kernel<double>, grid, threads, 0,
      N, a, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(powx_kernel, const int n,
        const Dtype* a,
    const Dtype alpha, Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
     FGPU_COLOR_STORE(ctx, &y[index], pow(FGPU_COLOR_LOAD(ctx, &a[index]), alpha));
    }
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(powx_kernel<float>, grid, threads, 0,
      N, a, alpha, y));
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(powx_kernel<double>, grid, threads, 0,
      N, a, alpha, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(sqrt_kernel, const int n, const Dtype* a, 
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      FGPU_COLOR_STORE(ctx, &y[index], sqrt(FGPU_COLOR_LOAD(ctx, &a[index])));
    }
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sqrt_kernel<float>, grid, threads, 0,
      N, a, y));
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sqrt_kernel<double>, grid, threads, 0,
      N, a, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(sign_kernel, const int n, const Dtype* x, 
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
      Dtype val = FGPU_COLOR_LOAD(ctx, &x[index]);
      FGPU_COLOR_STORE(ctx, &y[index], (Dtype(0) < val) - (val < Dtype(0)));
    }
  }

}

template <>
void caffe_gpu_sign<float>(const int N, const float* x, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sign_kernel<float>, grid, threads, 0,
      N, x, y));
}

template <>
void caffe_gpu_sign<double>(const int N, const double* x, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sign_kernel<double>, grid, threads, 0,
      N, x, y));
}

template <typename Dtype>
__global__ FGPU_DEFINE_KERNEL(sgnbit_kernel, const int n, const Dtype* x, 
        Dtype* y) {

  fgpu_dev_ctx_t *ctx;
  dim3 _blockIdx, _gridDim;
  ctx = FGPU_DEVICE_INIT();
  _gridDim = FGPU_GET_GRIDDIM(ctx);

  FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {
    CUDA_KERNEL_LOOP(index, n, _blockIdx, _gridDim) {
        FGPU_COLOR_STORE(ctx, &y[index], signbit(FGPU_COLOR_LOAD(ctx, &x[index])));
    }
  }
}

template <>
void caffe_gpu_sgnbit<float>(const int N, const float* x, float* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sgnbit_kernel<float>, grid, threads, 0,
      N, x, y));
}

template <>
void caffe_gpu_sgnbit<double>(const int N, const double* x, double* y) {
  dim3 grid(CAFFE_GET_BLOCKS(N)), threads(CAFFE_CUDA_NUM_THREADS);
  FGPU_CHECK(FGPU_LAUNCH_KERNEL(sgnbit_kernel<double>, grid, threads, 0,
      N, x, y));
}

#endif /* USE_FGPU */

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
