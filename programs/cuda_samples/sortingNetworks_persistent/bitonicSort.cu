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



//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm



#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__
FGPU_DEFINE_KERNEL(bitonicSortShared,
    uint *c_d_DstKey,	
    uint *c_d_DstVal,
    uint *c_d_SrcKey,
    uint *c_d_SrcVal,
    uint arrayLength,
    uint dir
)
{
    fgpu_dev_ctx_t *ctx;
   	dim3 _blockIdx;
    
    ctx = FGPU_DEVICE_INIT();

   	FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        uint *d_DstKey = c_d_DstKey;
        uint *d_DstVal = c_d_DstVal;
        uint *d_SrcKey = c_d_SrcKey;
        uint *d_SrcVal = c_d_SrcVal;

        // Handle to thread block group
		//cg::thread_block cta = cg::this_thread_block();
		//Shared memory storage for one or more short vectors
		__shared__ uint s_key[SHARED_SIZE_LIMIT];
		__shared__ uint s_val[SHARED_SIZE_LIMIT];

		//Offset to the beginning of subbatch and load data
		d_SrcKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_SrcVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		s_key[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[                      0]);
		s_val[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[                      0]);
		s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[(SHARED_SIZE_LIMIT / 2)]);
		s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[(SHARED_SIZE_LIMIT / 2)]);

		for (uint size = 2; size < arrayLength; size <<= 1)
		{
            //Bitonic merge
            uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

            for (uint stride = size / 2; stride > 0; stride >>= 1)
            {

                __syncthreads();
                //cg::sync(cta);
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
                );
            }
		}

		//ddd == dir for the last bitonic merge step
		{
            for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
            {
                __syncthreads();
                //cg::sync(cta);
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
                );
            }
		}

		//cg::sync(cta);
        __syncthreads();
		FGPU_COLOR_STORE(ctx, &d_DstKey[                      0], s_key[threadIdx.x +                       0]);
		FGPU_COLOR_STORE(ctx, &d_DstVal[                      0], s_val[threadIdx.x +                       0]);
		FGPU_COLOR_STORE(ctx, &d_DstKey[(SHARED_SIZE_LIMIT / 2)], s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
		FGPU_COLOR_STORE(ctx, &d_DstVal[(SHARED_SIZE_LIMIT / 2)], s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
    } FGPU_FOR_EACH_END
}



////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__
FGPU_DEFINE_KERNEL(bitonicSortShared1,
    uint *c_d_DstKey,
    uint *c_d_DstVal,
    uint *c_d_SrcKey,
    uint *c_d_SrcVal
)
{
    fgpu_dev_ctx_t *ctx;
   	dim3 _blockIdx;
	ctx = FGPU_DEVICE_INIT();

   	FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        uint *d_DstKey = c_d_DstKey;
        uint *d_DstVal = c_d_DstVal;
        uint *d_SrcKey = c_d_SrcKey;
        uint *d_SrcVal = c_d_SrcVal;

		// Handle to thread block group
		cg::thread_block cta = cg::this_thread_block();
		//Shared memory storage for current subarray
		__shared__ uint s_key[SHARED_SIZE_LIMIT];
		__shared__ uint s_val[SHARED_SIZE_LIMIT];

		//Offset to the beginning of subarray and load data
		d_SrcKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_SrcVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		s_key[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[                      0]);
		s_val[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[                      0]);
		s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[(SHARED_SIZE_LIMIT / 2)]);
		s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[(SHARED_SIZE_LIMIT / 2)]);

		for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
		{
		//Bitonic merge
		uint ddd = (threadIdx.x & (size / 2)) != 0;

		for (uint stride = size / 2; stride > 0; stride >>= 1)
		{
		    cg::sync(cta);
		    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
		    Comparator(
			s_key[pos +      0], s_val[pos +      0],
			s_key[pos + stride], s_val[pos + stride],
			ddd
		    );
		}
		}

		//Odd / even arrays of SHARED_SIZE_LIMIT elements
		//sorted in opposite directions
		uint ddd = _blockIdx.x & 1;
		{
		for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
		{
		    cg::sync(cta);
		    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
		    Comparator(
			s_key[pos +      0], s_val[pos +      0],
			s_key[pos + stride], s_val[pos + stride],
			ddd
		    );
		}
		}


		cg::sync(cta);
		FGPU_COLOR_STORE(ctx, &d_DstKey[                      0], s_key[threadIdx.x +                       0]);
		FGPU_COLOR_STORE(ctx, &d_DstVal[                      0], s_val[threadIdx.x +                       0]);
		FGPU_COLOR_STORE(ctx, &d_DstKey[(SHARED_SIZE_LIMIT / 2)], s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
		FGPU_COLOR_STORE(ctx, &d_DstVal[(SHARED_SIZE_LIMIT / 2)], s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
	} FGPU_FOR_EACH_END
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__
FGPU_DEFINE_KERNEL(bitonicMergeGlobal,
    uint *c_d_DstKey,
    uint *c_d_DstVal,
    uint *c_d_SrcKey,
    uint *c_d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
)
{
    fgpu_dev_ctx_t *ctx;
   	dim3 _blockIdx;
	ctx = FGPU_DEVICE_INIT();

   	FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        uint *d_DstKey = c_d_DstKey;
        uint *d_DstVal = c_d_DstVal;
        uint *d_SrcKey = c_d_SrcKey;
        uint *d_SrcVal = c_d_SrcVal;

	    uint global_comparatorI = _blockIdx.x * blockDim.x + threadIdx.x;
	    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

	    //Bitonic merge
	    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
	    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

	    uint keyA = FGPU_COLOR_LOAD(ctx, &d_SrcKey[pos +      0]);
	    uint valA = FGPU_COLOR_LOAD(ctx, &d_SrcVal[pos +      0]);
	    uint keyB = FGPU_COLOR_LOAD(ctx, &d_SrcKey[pos + stride]);
	    uint valB = FGPU_COLOR_LOAD(ctx, &d_SrcVal[pos + stride]);

	    Comparator(
		keyA, valA,
		keyB, valB,
		ddd
	    );

	    FGPU_COLOR_STORE(ctx, &d_DstKey[pos +      0], keyA);
	    FGPU_COLOR_STORE(ctx, &d_DstVal[pos +      0], valA);
	    FGPU_COLOR_STORE(ctx, &d_DstKey[pos + stride], keyB);
	    FGPU_COLOR_STORE(ctx, &d_DstVal[pos + stride], valB);
	} FGPU_FOR_EACH_END
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__
FGPU_DEFINE_KERNEL(bitonicMergeShared,
    uint *c_d_DstKey,
    uint *c_d_DstVal,
    uint *c_d_SrcKey,
    uint *c_d_SrcVal,
    uint arrayLength,
    uint size,
    uint dir
)
{
    fgpu_dev_ctx_t *ctx;
   	dim3 _blockIdx;
	ctx = FGPU_DEVICE_INIT();

   	FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        uint *d_DstKey = c_d_DstKey;
        uint *d_DstVal = c_d_DstVal;
        uint *d_SrcKey = c_d_SrcKey;
        uint *d_SrcVal = c_d_SrcVal;

	    // Handle to thread block group
	    cg::thread_block cta = cg::this_thread_block();
	    //Shared memory storage for current subarray
	    __shared__ uint s_key[SHARED_SIZE_LIMIT];
	    __shared__ uint s_val[SHARED_SIZE_LIMIT];

	    d_SrcKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	    d_SrcVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	    d_DstKey += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	    d_DstVal += _blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	    s_key[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[                      0]);
	    s_val[threadIdx.x +                       0] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[                      0]);
	    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcKey[(SHARED_SIZE_LIMIT / 2)]);
	    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = FGPU_COLOR_LOAD(ctx, &d_SrcVal[(SHARED_SIZE_LIMIT / 2)]);

	    //Bitonic merge
	    uint comparatorI = UMAD(_blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
	    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

	    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
	    {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
	    }

	    cg::sync(cta);
	    FGPU_COLOR_STORE(ctx, &d_DstKey[                      0], s_key[threadIdx.x +                       0]);
	    FGPU_COLOR_STORE(ctx, &d_DstVal[                      0], s_val[threadIdx.x +                       0]);
	    FGPU_COLOR_STORE(ctx, &d_DstKey[(SHARED_SIZE_LIMIT / 2)], s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
	    FGPU_COLOR_STORE(ctx, &d_DstVal[(SHARED_SIZE_LIMIT / 2)], s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]);
	} FGPU_FOR_EACH_END
}


////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L)
    {
        *log2L = 0;
        return 0;
    }
    else
    {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

        return L;
    }
}

extern "C" uint bitonicSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    dim3 threads(threadCount, 1, 1);
    dim3 grid(blockCount, 1, 1);

    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        FGPU_LAUNCH_KERNEL(bitonicSortShared, grid, threads, 0, d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
    {
	    FGPU_LAUNCH_KERNEL(bitonicSortShared1, grid, threads, 0, d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT)
                {
		            threads.x = 256;
		            grid.x = (batchSize * arrayLength) / 512;
                    FGPU_LAUNCH_KERNEL(bitonicMergeGlobal, grid, threads, 0, d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
                }
                else
                {
                    threads.x = threadCount;
		            grid.x = blockCount;
		            FGPU_LAUNCH_KERNEL(bitonicMergeShared, grid, threads, 0, d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }

    return threadCount;
}
