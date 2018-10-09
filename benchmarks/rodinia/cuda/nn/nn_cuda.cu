/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#include <fractional_gpu.hpp>
#include <fractional_gpu_cuda.cuh>

#define USE_FGPU
#include <fractional_gpu_testing.hpp>

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors

#define FILEPATH        "benchmarks/rodinia/data/nn/"
#define INPUT_FILENAME  "filelist"

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ 
FGPU_DEFINE_KERNEL(euclid, LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
    dim3 _blockIdx;
    fgpu_dev_ctx_t *ctx;
    ctx = FGPU_DEVICE_INIT();

    FGPU_FOR_EACH_DEVICE_BLOCK(_blockIdx) {

    	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    	int globalId = blockDim.x * ( FGPU_GET_GRIDDIM(ctx).x * _blockIdx.y + _blockIdx.x ) + threadIdx.x; // more efficient
        LatLong *latLong = d_locations+globalId;
        if (globalId < numRecords) {
            float *dist=d_distances+globalId;
            *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	    }
    }    
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
	int i, j;
	float lat, lng;

    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount;
    int ret;
    int num_iterations;

    // Default values
    resultsCount = 5;
    lat = 30.0;
    lng = 90.0;

    test_initialize(argc, argv, &num_iterations);

    sprintf(filename, "%s", FILEPATH INPUT_FILENAME);

    int numRecords = loadData(filename, records, locations);
    if (resultsCount > numRecords) 
        resultsCount = numRecords;

    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, 0 );
	cudaThreadSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
	cudaThreadSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	ret = fgpu_memory_allocate((void **) &d_locations,sizeof(LatLong) * numRecords);
    if (ret < 0)
        return ret;

	ret = fgpu_memory_allocate((void **) &d_distances,sizeof(float) * numRecords);
    if (ret < 0)
        return ret;

    pstats_t stats;
    pstats_t kernel_stats;

    pstats_init(&stats);
    pstats_init(&kernel_stats);

    /* Warmup and actual timed execution */
    for (i = 0; i < 2; i++) {
        bool is_warmup = (i == 0); 

        for (j = 0; j < num_iterations; j++) {

            double start = dtime_usec(0);

            /**
            * Transfer data from host to device
            */
            ret = fgpu_memory_copy_async( d_locations, &locations[0], sizeof(LatLong) * numRecords, FGPU_COPY_CPU_TO_GPU);
            if (ret < 0)
                return ret;
            
            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                return ret;

            double kernel_start = dtime_usec(0);

            /**
            * Execute kernel
            */
            ret = FGPU_LAUNCH_KERNEL(euclid, gridDim, threadsPerBlock, 0, d_locations,d_distances,numRecords,lat,lng);
            if (ret < 0)
                return ret;

            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                return ret;

            if (!is_warmup)
                pstats_add_observation(&kernel_stats, dtime_usec(kernel_start));

            //Copy data from device memory to host memory
            fgpu_memory_copy_async( distances, d_distances, sizeof(float)*numRecords, FGPU_COPY_GPU_TO_CPU );
            ret = fgpu_color_stream_synchronize();
            if (ret < 0)
                return ret;

            // find the resultsCount least distances
            findLowest(records,distances,numRecords,resultsCount);

            if (!is_warmup)
                pstats_add_observation(&stats, dtime_usec(start));
        }
    }

    // print out results
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }

    printf("Overall Stats\n");
    pstats_print(&stats);

    printf("Kernel Stats\n");
    pstats_print(&kernel_stats);

    //Free memory
    free(distances);
    fgpu_memory_free(d_locations);
	fgpu_memory_free(d_distances);
}

int loadData(char *filename, std::vector<Record> &records, std::vector<LatLong> &locations)
{
    FILE   *flist,*fp;
	int    i=0;
	char dbname[100];
    char dbpath[100];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {

        /**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }

        sprintf(dbpath, "%s%s", FILEPATH, dbname);

        fp = fopen(dbpath, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            if (fgets(record.recString,49,fp) == NULL && !feof(fp))
                printf("error in reading file\n");

            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}
