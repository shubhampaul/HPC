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

struct rack_t {
  float widthA;
  float widthB;
  float doubledWidth;
};

static void initGPU(int argc, char **argv);
static void initRacks(rack_t *racks, int n);

#ifdef WIN32
__inline void checkErr(cudaError_t err, const char *file, const int line);
#else
inline void checkErr(cudaError_t err, const char *file, const int line);
#endif

// GPU kernel
__global__ void doubleTheWidth(rack_t *racks, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    racks[tid].doubledWidth = 2 * (racks[tid].widthA + racks[tid].widthB);
  }
}

int main(int argc, char **argv) {
  initGPU(argc, argv);

  const int n = N;
  cudaError_t err;
  double runtimeAll, runtimeKernel;

  rack_t *h_racks = 0;
  rack_t *d_racks = 0;

  h_racks = (rack_t *)malloc(n * sizeof(rack_t));

  // TODO 1: allocate memory on GPU

  cudaMalloc(&d_racks, n * sizeof(rack_t));

  initRacks(h_racks, n); // init racks struct w/ values
  printf("First rack: w1=%f, w2=%f\n",h_racks[0].widthA, h_racks[0].widthB);

  runtimeAll = GetRealTime();

  // TODO 2: copy initialized data from CPU to GPU

  cudaMemcpy(d_racks, h_racks, n * sizeof(rack_t), cudaMemcpyHostToDevice);

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
  cudaMemcpy(h_racks, d_racks, n * sizeof(rack_t), cudaMemcpyDeviceToHost);

  runtimeAll = GetRealTime() - runtimeAll;

  printf("First rack: doubled width=%f\n", h_racks[0].doubledWidth);
  printf("Time Elapsed (including data transfer): %f s\n", runtimeAll);
  printf("Time Elapsed (kernel): %f s\n", runtimeKernel);

  // free memory
  free(h_racks);
  cudaFree(d_racks);
  return 0;
}

static void initRacks(rack_t *racks, int n) {
  for (int i = 0; i < n; i++) {
    racks[i].widthA = i + 2.5;
    racks[i].widthB = i + 1.5;
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
