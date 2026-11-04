#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <library_types.h>
#include <queue>

#include "helper.h"
#include "json.hpp"
#include "kernel.cuh"
#include "model.h"

using namespace std;

TEST(KernelTest, embedding) {
  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  EXPECT_EQ(status, CUBLAS_STATUS_SUCCESS) << "cuBLAS init failed, status: ";
  // if (status != CUBLAS_STATUS_SUCCESS) {
  //   std::cout << status << std::endl;
  //   return 1;
  // }

  Weights weights{};
  int ret = loadWeight(weights);
  EXPECT_EQ(0, ret);
}
