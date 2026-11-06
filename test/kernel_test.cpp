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
  Weights weights{};
  int ret = loadWeight(weights);
  EXPECT_EQ(0, ret);

  vector<int> prompt_cpu = {128000, 128006};
  int *prompt_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&prompt_gpu, sizeof(int) * prompt_cpu.size()));
  CUDA_CHECK(cudaMemcpy(prompt_gpu, prompt_cpu.data(),
                        sizeof(int) * prompt_cpu.size(),
                        cudaMemcpyHostToDevice));

  vector<__nv_bfloat16> embedding_cpu(prompt_cpu.size() * EMBEDDING_LENGTH);
  __nv_bfloat16 *embedding_gpu = nullptr;
  CUDA_CHECK(
      cudaMalloc(&embedding_gpu,
                 sizeof(__nv_bfloat16) * prompt_cpu.size() * EMBEDDING_LENGTH));
  CUDA_CHECK(
      cudaMemcpy(embedding_gpu, embedding_cpu.data(),
                 sizeof(__nv_bfloat16) * prompt_cpu.size() * EMBEDDING_LENGTH,
                 cudaMemcpyHostToDevice));

  embeddingGather(prompt_gpu, weights.embed_tokens, embedding_gpu, 2);

  CUDA_CHECK(
      cudaMemcpy(embedding_cpu.data(), embedding_gpu,
                 sizeof(__nv_bfloat16) * prompt_cpu.size() * EMBEDDING_LENGTH,
                 cudaMemcpyDeviceToHost));

  for (int i = 0; i < prompt_cpu.size(); ++i) {
    printf("token %d: %d\n", i, prompt_cpu[i]);
    for (int j = 0; j < EMBEDDING_LENGTH; ++j) {
      printf("%f ", float(embedding_cpu[i * EMBEDDING_LENGTH + j]));
    }
    printf("\n");
  }

  EXPECT_NE(float(embedding_cpu[0]), float(0));

  CUDA_CHECK(cudaFree(prompt_gpu));
  CUDA_CHECK(cudaFree(embedding_gpu));
  // freeWeight(weights);
}

TEST(KernelTest, rmsNorm) {
  vector<__nv_bfloat16> input_cpu(EMBEDDING_LENGTH);
  for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
    input_cpu[i] = (i % 32) + 1;
    // printf("%f ", float(input_cpu[i]));
  }
  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  vector<__nv_bfloat16> output_cpu(EMBEDDING_LENGTH);
  __nv_bfloat16 *output_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&output_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));

  vector<__nv_bfloat16> weight_cpu(EMBEDDING_LENGTH);
  for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
    weight_cpu[i] = (i % 32) + 1;
    // printf("%f ", float(weight_cpu[i]));
  }
  __nv_bfloat16 *weight_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&weight_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));
  CUDA_CHECK(cudaMemcpy(weight_gpu, weight_cpu.data(),
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  rmsNorm(input_gpu, output_gpu, weight_gpu, 1);

  CUDA_CHECK(cudaMemcpy(output_cpu.data(), output_gpu,
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
    printf("%f ", float(output_cpu[i]));
  }

  CUDA_CHECK(cudaFree(input_gpu));
  CUDA_CHECK(cudaFree(output_gpu));
  CUDA_CHECK(cudaFree(weight_gpu));
}

TEST(KernelTest, initRope) {
  init_rope_frequency(64, 10, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);
}

TEST(KernelTest, causalMask) {
  int n = 10;
  vector<__nv_bfloat16> input_cpu(n * n);
  for (int i = 0; i < n * n; ++i) {
    input_cpu[i] = i;
  }

  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * n * n));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyHostToDevice));

  causalMask(input_gpu, n);

  CUDA_CHECK(cudaMemcpy(input_cpu.data(), input_gpu,
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%4.2f ", float(input_cpu[i * n + j]));
    }
    printf("\n");
  }

  CUDA_CHECK(cudaFree(input_gpu));
}

TEST(KernelTest, softmax) {
  int n = 10;
  vector<__nv_bfloat16> input_cpu(n * n);
  for (int i = 0; i < n * n; ++i) {
    input_cpu[i] = i;
  }

  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * n * n));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyHostToDevice));

  causalMask(input_gpu, n);

  CUDA_CHECK(cudaMemcpy(input_cpu.data(), input_gpu,
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%4.2f ", float(input_cpu[i * n + j]));
    }
    printf("\n");
  }

  softmax(input_gpu, n);

  CUDA_CHECK(cudaMemcpy(input_cpu.data(), input_gpu,
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%.10f ", float(input_cpu[i * n + j]));
    }
    printf("\n");
  }

  CUDA_CHECK(cudaFree(input_gpu));
}

TEST(KernelTest, rope) {
  int n = 10;
  init_rope_frequency(64, 10, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);
  vector<__nv_bfloat16> input_cpu(n * EMBEDDING_LENGTH);
  for (int i = 0; i < n * EMBEDDING_LENGTH; ++i) {
    input_cpu[i] = (i % 32) + 1;
  }
  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(
      cudaMalloc(&input_gpu, n * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        n * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  rope(input_gpu, n, EMBEDDING_LENGTH);
  CUDA_CHECK(cudaMemcpy(input_cpu.data(), input_gpu,
                        n * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyDeviceToHost));

  for (int i = EMBEDDING_LENGTH; i < 2 * EMBEDDING_LENGTH; ++i) {
    if (i % 6 == 0) {
      printf("\n");
    }
    printf("%f ", float(input_cpu[i]));
  }
  CUDA_CHECK(cudaFree(input_gpu));
}
