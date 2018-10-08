/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh

#include <cooperative_groups.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

namespace cg = cooperative_groups;


///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ 
FGPU_DEFINE_KERNEL(fwtBatch1Kernel, float *d_Output, float *d_Input, int log2N)
{
    dim3 _blockIdx;
    FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        const int    N = 1 << log2N;
        const int base = _blockIdx.x << log2N;

        //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
        extern __shared__ float s_data[];
        float *d_Src = d_Input  + base;
        float *d_Dst = d_Output + base;

        for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
        {
            s_data[pos] = d_Src[pos];
        }

        //Main radix-4 stages
        const int pos = threadIdx.x;

        for (int stride = N >> 2; stride > 0; stride >>= 2)
        {
            int lo = pos & (stride - 1);
            int i0 = ((pos - lo) << 2) + lo;
            int i1 = i0 + stride;
            int i2 = i1 + stride;
            int i3 = i2 + stride;

            cg::sync(cta);
            float D0 = s_data[i0];
            float D1 = s_data[i1];
            float D2 = s_data[i2];
            float D3 = s_data[i3];

            float T;
            T = D0;
            D0         = D0 + D2;
            D2         = T - D2;
            T = D1;
            D1         = D1 + D3;
            D3         = T - D3;
            T = D0;
            s_data[i0] = D0 + D1;
            s_data[i1] = T - D1;
            T = D2;
            s_data[i2] = D2 + D3;
            s_data[i3] = T - D3;
        }

        //Do single radix-2 stage for odd power of two
        if (log2N & 1)
        {
            cg::sync(cta);

            for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x)
            {
                int i0 = pos << 1;
                int i1 = i0 + 1;

                float D0 = s_data[i0];
                float D1 = s_data[i1];
                s_data[i0] = D0 + D1;
                s_data[i1] = D0 - D1;
            }
        }

        cg::sync(cta);

        for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
        {
            d_Dst[pos] = s_data[pos];
        }
    }   
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ 
FGPU_DEFINE_KERNEL(fwtBatch2Kernel,
    float *d_Output,
    float *d_Input,
    int stride
)
{
    dim3 _blockIdx;
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        const int pos = _blockIdx.x * blockDim.x + threadIdx.x;
        const int   N = blockDim.x *  FGPU_GET_GRIDDIM(ctx).x * 4;

        float *d_Src = d_Input  + _blockIdx.y * N;
        float *d_Dst = d_Output + _blockIdx.y * N;

        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        float D0 = d_Src[i0];
        float D1 = d_Src[i1];
        float D2 = d_Src[i2];
        float D3 = d_Src[i3];

        float T;
        T = D0;
        D0        = D0 + D2;
        D2        = T - D2;
        T = D1;
        D1        = D1 + D3;
        D3        = T - D3;
        T = D0;
        d_Dst[i0] = D0 + D1;
        d_Dst[i1] = T - D1;
        T = D2;
        d_Dst[i2] = D2 + D3;
        d_Dst[i3] = T - D3;
    }   
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(float *d_Data, int M, int log2N)
{
    const int THREAD_N = 256;
    int ret;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
        ret = FGPU_LAUNCH_KERNEL(fwtBatch2Kernel, grid, THREAD_N, 0, d_Data, d_Data, N / 4);
        if (ret < 0)
            exit(EXIT_FAILURE);
    }

    ret = FGPU_LAUNCH_KERNEL(fwtBatch1Kernel, M, N / 4, N *sizeof(float), 
        d_Data,
        d_Data,
        log2N
    );
    if (ret < 0)
        exit(EXIT_FAILURE);
}



////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ 
FGPU_DEFINE_KERNEL(modulateKernel, float *d_A, float *d_B, int N)
{
    dim3 _blockIdx;
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        int        tid = _blockIdx.x * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * FGPU_GET_GRIDDIM(ctx).x;
        float     rcpN = 1.0f / (float)N;

        for (int pos = tid; pos < N; pos += numThreads)
        {
            d_A[pos] *= d_B[pos] * rcpN;
        }
    }   
}

//Interface to modulateKernel()
void modulateGPU(float *d_A, float *d_B, int N)
{
    int ret = FGPU_LAUNCH_KERNEL(modulateKernel, 128, 256, 0, d_A, d_B, N);
    if (ret < 0)
        exit(EXIT_FAILURE);
}



#endif
#endif
