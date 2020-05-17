/***********************************************
* GPU Performance Tuning Example with CUDA
* ----------------------------------------------
* Authors: Sandra Wienke, RWTH Aachen University
*          Julian Miller, RWTH Aachen university
************************************************/

#include "realtime.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define N 67107840
#define THREADSPERBLOCK 1024

struct rackSoA_t {
float widthA[N];
float widthB[N];
float doubledWidth[N];
};

static void initGPU(int argc, char **argv);
static void initRacks(rackSoA_t *racks, int n);

#ifdef WIN32
__inline void checkErr(cudaError_t err, const char *file, const int line);
#else
inline void checkErr(cudaError_t err, const char *file, const int line);
#endif

// GPU kernel
__global__ void doubleTheWidth(rackSoA_t *racks, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
	 racks->doubledWidth[tid] = 2* (racks->widthA[tid] + racks->widthB[tid]);
  }
}

int main(int argc, char **argv) {
  initGPU(argc, argv);

  const int n = N;
  cudaError_t err;
  double runtimeAll, runtimeKernel;

  rackSoA_t *h_racks = 0;
  rackSoA_t *d_racks = 0;

  // allocate memory on host

  h_racks = (rackSoA_t *)malloc(sizeof(rackSoA_t));

  // TODO 1: allocate memory on GPU

  cudaMalloc(&d_racks, sizeof(rackSoA_t));

  initRacks(h_racks, n); // init racks struct w/ values
  printf("First rack: w1=%f, w2=%f\n",h_racks->widthA[0], h_racks->widthB[0]);

  runtimeAll = GetRealTime();

  // TODO 2: copy initialized data from CPU to GPU

  cudaMemcpy(d_racks, h_racks, sizeof(rackSoA_t), cudaMemcpyHostToDevice);

  dim3 threads_per_block(THREADSPERBLOCK);
  dim3 blocks_per_grid;

  // TODO 3: Compute the number of blocks_per_grid
  // so that each thread works on one rack element
  blocks_per_grid = dim3((n + THREADSPERBLOCK -1 )/THREADSPERBLOCK);

  printf("blocks: %d\n", blocks_per_grid.x);

  // TODO 4: Call the CUDA kernel
  runtimeKernel = GetRealTime();
  
  doubleTheWidth<<<blocks_per_grid, threads_per_block>>>(d_racks, n);

  cudaDeviceSynchronize();
  runtimeKernel = GetRealTime() - runtimeKernel;

  // TODO 5: Copy results data from GPU to CPU
  cudaMemcpy(h_racks, d_racks, sizeof(rackSoA_t), cudaMemcpyDeviceToHost);

  runtimeAll = GetRealTime() - runtimeAll;

  printf("First rack: doubled width=%f\n", h_racks->doubledWidth[0]);
  printf("Time Elapsed (including data transfer): %f s\n", runtimeAll);
  printf("Time Elapsed (kernel): %f s\n", runtimeKernel);

  // free memory
  free(h_racks);
  cudaFree(d_racks);
  return 0;
}

static void initRacks(rackSoA_t *racks, int n) {
  for (int i = 0; i < n; i++) {
    racks->widthA[i] = i + 2.5;
    racks->widthB[i] = i + 1.5;
  }
}

static void initGPU(int argc, char **argv) {
  // gets the device id (if specified) to run on
  int devId = -1;
  int devCount = 0;
  if (argc > 1) {
    devId = atoi(argv[1]);
    cudaGetDeviceCount(&devCount);
    if (devId < 0 || devId >= devCount) {
      printf("The specified device ID is not supported.\n");
      exit(1);
    }
  }
  if (devId != -1) {
    cudaSetDevice(devId);
  }
  // creates a context on the GPU just to
  // exclude initialization time from computations
  cudaFree(0);

  // print device id
  cudaGetDevice(&devId);
  printf("Running on GPU with ID %d.\n\n", devId);
}

// Checks whether a CUDA error occured
// If so, the error message is printed and the program exits
inline void checkErr(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s: Cuda error in line %d: %s.\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
}
