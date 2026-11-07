#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <fstream>
#include <iostream>
#include <library_types.h>
#include <queue>

#include "helper.h"
#include "json.hpp"
#include "kernel.cuh"
#include "nn.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  Weights weights{};
  if (LoadWeight(weights) != 0) {
    return 1;
  }

  InitRopeFrequency(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);

  std::cout << "\nOk bye!" << std::endl;
  cublasDestroy(cublas_handle);
  cudaDeviceSynchronize();
  return 0;
}
