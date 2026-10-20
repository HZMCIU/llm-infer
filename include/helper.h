#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;

int checkGPUStatus() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found\n";
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // std::cout << "Device: " << prop.name << "\n";
  // std::cout << "Compute capability: " << prop.major << "." << prop.minor
  //           << "\n";
  // std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
  // std::cout << "SM count: " << prop.multiProcessorCount << "\n";
  // std::cout << "Max threads per block: " << prop.maxThreadsPerBlock
  //           << std::endl;
  // size_t free_mem;
  // size_t total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "Free memory: " << free_mem / B_TO_GB
  //           << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
  return 0;
}
