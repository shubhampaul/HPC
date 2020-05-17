#Output of nvprof

$ sudo /usr/local/cuda-10.1/bin/nvprof ./a.out

==18288== NVPROF is profiling process 18288, command: ./a.out
Max error: 0
==18288== Profiling application: ./a.out
==18288== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  932.39us         1  932.39us  932.39us  932.39us  add(int, float*, float*)
      API calls:   96.35%  231.65ms         2  115.83ms  680.71us  230.97ms  cudaMallocManaged
                    2.37%  5.6951ms         1  5.6951ms  5.6951ms  5.6951ms  cudaLaunchKernel
                    0.39%  941.04us         1  941.04us  941.04us  941.04us  cudaDeviceSynchronize
                    0.32%  773.10us         2  386.55us  370.95us  402.15us  cudaFree
                    0.30%  725.28us        97  7.4770us     517ns  314.76us  cuDeviceGetAttribute
                    0.23%  554.61us         1  554.61us  554.61us  554.61us  cuDeviceTotalMem
                    0.04%  84.455us         1  84.455us  84.455us  84.455us  cuDeviceGetName
                    0.00%  6.1270us         3  2.0420us     522ns  3.5640us  cuDeviceGetCount
                    0.00%  4.1630us         1  4.1630us  4.1630us  4.1630us  cuDeviceGetPCIBusId
                    0.00%  3.1580us         2  1.5790us     825ns  2.3330us  cuDeviceGet
                    0.00%     913ns         1     913ns     913ns     913ns  cuDeviceGetUuid

==18288== Unified Memory profiling result:
Device "GeForce 840M (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       6  1.3333MB  768.00KB  2.0000MB  8.000000MB  5.445472ms  Host To Device
     102  120.47KB  4.0000KB  0.9961MB  12.00000MB  7.521120ms  Device To Host
Total CPU Page faults: 51

