/* This file contains common utilities. */
#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>

/* Compile time assertion */
#define COMPILE_ASSERT(val) typedef char assertion_typedef[(val) * 2 - 1];

/* Assertion for CUDA functions */
#define gpuErrAssert(ans) gpuAssert((ans), __FILE__, __LINE__, true)
#define gpuErrCheck(ans) gpuAssert((ans), __FILE__, __LINE__, false)

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUcheck: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          exit(code);
      else
          return -1;
   }
   return 0;
}

/* Assertion for CUDA driver functions */
#define gpuDriverErrAssert(ans) gpuDriverAssert((ans), __FILE__, __LINE__, true)
#define gpuDriverErrCheck(ans) gpuDriverAssert((ans), __FILE__, __LINE__, false)

inline int gpuDriverAssert(CUresult  code, const char *file, int line, bool abort=true)
{
    if (code != CUDA_SUCCESS)
    {
        const char *err;
        cuGetErrorString(code, &err);
        fprintf(stderr,"GPUcheck: %s %s %d\n", err, file, line);
        if (abort)
            exit(code);
        else
            return -1;
    }
    return 0;
}

#define USECPSEC 1000000ULL
inline double dtime_usec(unsigned long long start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return (double)(((tv.tv_sec*USECPSEC)+tv.tv_usec)-start);
}

#define LOG2(x) {(uint32_t)(sizeof(x) * 8  - 1 - __builtin_clz(x))}

/* Assumes size of power of 2 */
#define ROUND_UP(a, size) (((a) + (size) - 1) & ~((size) - 1))

#endif /* COMMON_H */
