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
#include "nn.h"

using namespace std;

TEST(KernelTest, Embedding) {
  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  EXPECT_EQ(status, CUBLAS_STATUS_SUCCESS) << "cuBLAS init failed, status: ";
  Weights weights{};
  int ret = LoadWeight(weights);
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

  EmbeddingGather(prompt_gpu, weights.embed_tokens, embedding_gpu, 2);

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

TEST(KernelTest, RmsNorm) {
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

  RmsNorm(input_gpu, output_gpu, weight_gpu, 1);

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

TEST(KernelTest, InitRope) {
  InitRopeFrequency(64, 10, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);
}

TEST(KernelTest, CausalMask) {
  int n = 10;
  vector<__nv_bfloat16> input_cpu(n * n);
  for (int i = 0; i < n * n; ++i) {
    input_cpu[i] = i;
  }

  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * n * n));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyHostToDevice));

  CausalMask(input_gpu, n);

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

TEST(KernelTest, Softmax) {
  int n = 10;
  vector<__nv_bfloat16> input_cpu(n * n);
  for (int i = 0; i < n * n; ++i) {
    input_cpu[i] = i;
  }

  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * n * n));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyHostToDevice));

  CausalMask(input_gpu, n);

  CUDA_CHECK(cudaMemcpy(input_cpu.data(), input_gpu,
                        sizeof(__nv_bfloat16) * n * n, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%4.2f ", float(input_cpu[i * n + j]));
    }
    printf("\n");
  }

  Softmax(input_gpu, n);

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

TEST(KernelTest, Rope) {
  int n = 10;
  InitRopeFrequency(64, 10, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);
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

  Rope(input_gpu, n, EMBEDDING_LENGTH);
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

TEST(KernelTest, PagedAttention) {
  int layer = 0;
  __nv_bfloat16 *q_proj_gpu = nullptr;
  __nv_bfloat16 *kv_cache_gpu = nullptr;
  int *block_table_gpu = nullptr;
  __nv_bfloat16 *output_gpu = nullptr;

  // 1. init q_proj
  CUDA_CHECK(cudaMalloc(&q_proj_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));
  vector<__nv_bfloat16> q_proj_cpu(EMBEDDING_LENGTH);
  for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
    q_proj_cpu[i] = (((i + 1) % 17) - 8) * 0.125f;
  }
  CUDA_CHECK(cudaMemcpy(q_proj_gpu, q_proj_cpu.data(),
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  // 2. init block table / kv cache
  vector<int> block_table_cpu(MAX_BLOCKS_PER_SEQ * N_LAYERS, -1);
  CUDA_CHECK(cudaMalloc(&block_table_gpu,
                        sizeof(int) * MAX_BLOCKS_PER_SEQ * N_LAYERS));

  vector<__nv_bfloat16> kv_cache_cpu(N_LAYERS * MAX_PROMPT_LEN * KV_DIM * 2);
  CUDA_CHECK(cudaMalloc(&kv_cache_gpu, KV_CACHE_SIZE_BYTES));
  int block = 0;

  int cnt1 = 1, cnt2 = 1;
  for (int i = 0; i < 32; i += BLOCK_SIZE) {
    block_table_cpu[0 * MAX_BLOCKS_PER_SEQ + i / BLOCK_SIZE] = block;
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      for (int k = 0; k < KV_DIM; ++k) {

        kv_cache_cpu[block * BLOCK_BYTES / 2 + j * KV_DIM + k] =
            ((cnt1++ % 17) - 8) * 0.125f;
      }
      for (int k = 0; k < KV_DIM; ++k) {
        kv_cache_cpu[block * BLOCK_BYTES / 2 + V_OFFSET / 2 + j * KV_DIM + k] =
            ((cnt2++ % 17) - 8) * 0.125f;
      }
    }
    block++;
  }

  CUDA_CHECK(cudaMemcpy(kv_cache_gpu, kv_cache_cpu.data(), KV_CACHE_SIZE_BYTES,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(block_table_gpu, block_table_cpu.data(),
                        sizeof(int) * MAX_BLOCKS_PER_SEQ * N_LAYERS,
                        cudaMemcpyHostToDevice));

  // 3. init output
  vector<__nv_bfloat16> output_cpu(EMBEDDING_LENGTH, 0);
  CUDA_CHECK(cudaMalloc(&output_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));

  // 4. attention
  PagedAttention(0, q_proj_gpu, kv_cache_gpu, block_table_gpu, 32, output_gpu);

  CUDA_CHECK(cudaMemcpy(output_cpu.data(), output_gpu,
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
    printf("%f ", float(output_cpu[i]));
  }
}
