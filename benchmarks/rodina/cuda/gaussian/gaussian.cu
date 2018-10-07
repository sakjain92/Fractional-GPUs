/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

#define RD_WG_SIZE_0 128

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 512
#endif

#define RD_WG_SIZE_1 16

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

int Size;
float *a, *b, *finalVec;
float *g_m_cuda = NULL, *g_a_cuda = NULL, *g_b_cuda = NULL;
float *m;
bool is_warmup;
// Execute the kernel
pstats_t stats, kernel_stats;

void InitPerRun();
int ForwardSub();
void BackSub();
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int run_gaussian(void)
{
    int i, ret;
    
	a = (float *) malloc(Size * Size * sizeof(float));
	create_matrix(a, Size);

	b = (float *) malloc(Size * sizeof(float));
	for (i =0; i < Size; i++)
	    b[i]=1.0;

	m = (float *) malloc(Size * Size * sizeof(float));

    InitPerRun();
    
    // run kernels
    ret = ForwardSub();
    if (ret < 0)
        return ret;

    BackSub();
    
    free(m);
    free(a);
    free(b);

    return 0;
}

int main(int argc, char **argv)
{
    int ret;
    int num_iterations;

    test_initialize(argc, argv, &num_iterations);

    Size = 16;

    // allocate memory on GPU - Need not do for each iteration
    ret = fgpu_memory_allocate((void **) &g_m_cuda, Size * Size * sizeof(float));
    if (ret < 0)
        return ret;

	ret = fgpu_memory_allocate((void **) &g_a_cuda, Size * Size * sizeof(float));
	if (ret < 0)
        return ret;

    ret = fgpu_memory_allocate((void **) &g_b_cuda, Size * sizeof(float));	
    if (ret < 0)
        return ret;

    /* Warmup */
    is_warmup = true;
    for (int i = 0; i < num_iterations; i++) {
        ret = run_gaussian();
        if (ret < 0)
            return ret;
    }

    /* Execute with measurements */
    is_warmup = false;
    pstats_init(&stats);
    pstats_init(&kernel_stats);

    for (int i = 0; i < num_iterations; i++) {
        double start = dtime_usec(0);

        ret = run_gaussian();
        if (ret < 0)
            return ret;
        
        pstats_add_observation(&stats, dtime_usec(start)); 
    }

    printf("Overall Stats\n");
    pstats_print(&stats);

    printf("Kernel Stats\n");
    pstats_print(&kernel_stats);

    fgpu_memory_free(g_m_cuda);
	fgpu_memory_free(g_a_cuda);
	fgpu_memory_free(g_b_cuda);

    test_deinitialize();
}

/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties(){
	cudaDeviceProp deviceProp;  
	int nDevCount = 0;  
	
	cudaGetDeviceCount( &nDevCount );  
	printf( "Total Device found: %d", nDevCount );  
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx )  
	{  
	    memset( &deviceProp, 0, sizeof(deviceProp));  
	    if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx))  
	        {
				printf( "\nDevice Name \t\t - %s ", deviceProp.name );  
			    printf( "\n**************************************");  
			    printf( "\nTotal Global Memory\t\t\t - %lu KB", deviceProp.totalGlobalMem/1024 );  
			    printf( "\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock/1024 );  
			    printf( "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock );  
			    printf( "\nWarp size in threads \t\t\t - %d", deviceProp.warpSize );  
			    printf( "\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch );  
			    printf( "\nMaximum threads per block \t\t - %d", deviceProp.maxThreadsPerBlock );  
			    printf( "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );  
			    printf( "\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );  
			    printf( "\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem );  
			    printf( "\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor );  
			    printf( "\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate );  
			    printf( "\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment );  
			    printf( "\nDevice Overlap \t\t\t\t - %s", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );  
			    printf( "\nNumber of Multi processors \t\t - %d\n\n", deviceProp.multiProcessorCount );  
			}  
	    else  
	        printf( "\n%s", cudaGetErrorString(cudaGetLastError()));  
	}  
}
 
/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
		*(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__
FGPU_DEFINE_KERNEL(Fan1, float *m_cuda, float *a_cuda, int Size, int t)
{   
    dim3 _blockIdx;
    FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    	if(threadIdx.x + _blockIdx.x * blockDim.x >= Size-1-t) return;
	    *(m_cuda+Size*(blockDim.x*_blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*_blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
    }
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

__global__
FGPU_DEFINE_KERNEL(Fan2, float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
    dim3 _blockIdx;
    FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

        if(threadIdx.x + _blockIdx.x * blockDim.x >= Size-1-t) return;
        if(threadIdx.y + _blockIdx.y * blockDim.y >= Size-t) return;
        
        int xidx = _blockIdx.x * blockDim.x + threadIdx.x;
        int yidx = _blockIdx.y * blockDim.y + threadIdx.y;
        //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
        
        a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
        //a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
        if(yidx == 0){
            //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
            //printf("xidx:%d,yidx:%d\n",xidx,yidx);
            b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
        }
    }   
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
int ForwardSub()
{
	int t;
    int ret = 0;

	// copy memory to GPU
	ret = fgpu_memory_copy_async(g_m_cuda, m, Size * Size * sizeof(float), FGPU_COPY_CPU_TO_GPU);
	if (ret < 0)
        return ret;

    ret = fgpu_memory_copy_async(g_a_cuda, a, Size * Size * sizeof(float), FGPU_COPY_CPU_TO_GPU);
	if (ret < 0)
        return ret;

    ret = fgpu_memory_copy_async(g_b_cuda, b, Size * sizeof(float), FGPU_COPY_CPU_TO_GPU);
    if (ret < 0)
        return ret;

	ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;


	int block_size,grid_size;
	
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//printf("1d grid size: %d\n",grid_size);


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	
	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);

    double start = dtime_usec(0);

	for (t=0; t < (Size-1); t++) {
		ret = FGPU_LAUNCH_KERNEL(Fan1, dimGrid, dimBlock, 0, g_m_cuda, g_a_cuda, Size, t);
        if (ret < 0)
            return ret;

		ret = fgpu_color_stream_synchronize();
        if (ret < 0)
            return ret;

		ret = FGPU_LAUNCH_KERNEL(Fan2, dimGridXY,dimBlockXY, 0, g_m_cuda, g_a_cuda, g_b_cuda, Size, Size-t, t);
        if (ret < 0)
            return ret;

		ret = fgpu_color_stream_synchronize();
        if (ret < 0)
            return ret;
	}

    if (!is_warmup) {
        pstats_add_observation(&kernel_stats, dtime_usec(start));
    }

    // copy memory back to CPU
	ret = fgpu_memory_copy_async(m, g_m_cuda, Size * Size * sizeof(float), FGPU_COPY_GPU_TO_CPU);
    if (ret < 0)
        return ret;

	ret = fgpu_memory_copy_async(a, g_a_cuda, Size * Size * sizeof(float), FGPU_COPY_GPU_TO_CPU);
    if (ret < 0)
        return ret;

	ret = fgpu_memory_copy_async(b, g_b_cuda, Size * sizeof(float), FGPU_COPY_GPU_TO_CPU);
    if (ret < 0)
        return ret;

    ret = fgpu_color_stream_synchronize();
    if (ret < 0)
        return ret;

    return ret;
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
	// create a new vector to hold the final answer
	finalVec = (float *) malloc(Size * sizeof(float));
	// solve "bottom up"
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
