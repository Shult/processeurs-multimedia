#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Global Memory (bytes): %ld\n",
           prop.totalGlobalMem);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("   Max Thread per block : %d \n",prop.maxThreadsPerBlock);
    printf("   Multiproc count : %d \n",prop.multiProcessorCount);
    printf("   Max Grid size : %d %d %d \n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("   Max thread dim : %d %d %d \n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("   Registres per block : %d \n",prop.regsPerBlock);
    printf("\n");
  }
} 