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
#include "sampler.h"

using namespace std;

TEST(NNTest, MatMul) {
  vector<__nv_bfloat16> weight_cpu(EMBEDDING_LENGTH * EMBEDDING_LENGTH);
  for (int i = 0; i < EMBEDDING_LENGTH * EMBEDDING_LENGTH; ++i) {
    weight_cpu[i] = (i % 32) + 1;
  }
  // for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
  //   if (i % EMBEDDING_LENGTH == 0) {
  //     printf("\n");
  //   }
  //   printf("%f ", float(weight_cpu[i]));
  // }

  __nv_bfloat16 *weight_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&weight_gpu, sizeof(__nv_bfloat16) * EMBEDDING_LENGTH *
                                         EMBEDDING_LENGTH));
  CUDA_CHECK(
      cudaMemcpy(weight_gpu, weight_cpu.data(),
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * EMBEDDING_LENGTH,
                 cudaMemcpyHostToDevice));

  int n = 2;
  vector<__nv_bfloat16> input_cpu(n * EMBEDDING_LENGTH);
  for (int i = 0; i < n * EMBEDDING_LENGTH; ++i) {
    input_cpu[i] = (i % 32) + 1;
  }

  // for (int i = 0; i < EMBEDDING_LENGTH; ++i) {
  //   if (i % EMBEDDING_LENGTH == 0) {
  //     printf("\n");
  //   }
  //   printf("%f ", float(input_cpu[i]));
  // }

  __nv_bfloat16 *input_gpu = nullptr;
  CUDA_CHECK(
      cudaMalloc(&input_gpu, sizeof(__nv_bfloat16) * n * EMBEDDING_LENGTH));
  CUDA_CHECK(cudaMemcpy(input_gpu, input_cpu.data(),
                        sizeof(__nv_bfloat16) * n * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  vector<__nv_bfloat16> output_cpu(n * EMBEDDING_LENGTH);
  __nv_bfloat16 *output_gpu = nullptr;
  CUDA_CHECK(
      cudaMalloc(&output_gpu, sizeof(__nv_bfloat16) * n * EMBEDDING_LENGTH));
  CUDA_CHECK(cudaMemcpy(output_gpu, output_cpu.data(),
                        sizeof(__nv_bfloat16) * n * EMBEDDING_LENGTH,
                        cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  float alpha = 1, beta = 0;
  status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

                        EMBEDDING_LENGTH, n, EMBEDDING_LENGTH,

                        &alpha, weight_gpu, CUDA_R_32F, EMBEDDING_LENGTH,

                        input_gpu, CUDA_R_32F, EMBEDDING_LENGTH,

                        &beta, output_gpu, CUDA_R_32F, EMBEDDING_LENGTH,

                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  CUDA_CHECK(cudaMemcpy(output_cpu.data(), output_gpu,
                        sizeof(__nv_bfloat16) * n * EMBEDDING_LENGTH,
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < EMBEDDING_LENGTH * n; ++i) {
    if (i % EMBEDDING_LENGTH == 0) {
      printf("\n");
    }
    printf("%f ", float(output_cpu[i]));
  }

  CUDA_CHECK(cudaFree(weight_gpu));
  CUDA_CHECK(cudaFree(input_gpu));
  CUDA_CHECK(cudaFree(output_gpu));
}

TEST(NNTest, Prefill) {
  CheckGPUStatus();
  Weights weights{};
  int ret = LoadWeight(weights);

  printf("after loads weight\n");
  CheckGPUStatus();
  printf("\n\n");

  KVCacheContext kv_cache_context;
  CUDA_CHECK(cudaMalloc(&kv_cache_context.kv_cache, KV_CACHE_SIZE_BYTES));
  CUDA_CHECK(cudaMalloc(&kv_cache_context.block_table_gpu,
                        N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int)));

  kv_cache_context.free_blocks.resize(NUM_BLOCKS, -1);
  std::iota(kv_cache_context.free_blocks.begin(),
            kv_cache_context.free_blocks.end(), 0);
  kv_cache_context.block_table.resize(N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);

  TokenInfo token_info;
  token_info.queue.push(
      {128000, 3923, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007});
  token_info.is_slot_free.resize(5, true);
  token_info.slot = 0;
  CUDA_CHECK(
      cudaMalloc(&token_info.input_token_gpu,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  // printf("after allocate input token\n");
  // checkGPUStatus();
  // printf("\n\n");

  LayerContext layer_context;
  cublasStatus_t status = cublasCreate(&layer_context.cublas_handle);

  // printf("after allocate layer context input\n");
  // checkGPUStatus();
  // printf("\n\n");

  CUDA_CHECK(
      cudaMalloc(&layer_context.input,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  // printf("after allocate layer context hidden state\n");
  // checkGPUStatus();
  // printf("\n\n");

  CUDA_CHECK(
      cudaMalloc(&layer_context.hidden_state,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  // printf("after allocate layer context attn_scores_v\n");
  // checkGPUStatus();
  // printf("\n\n");

  CUDA_CHECK(
      cudaMalloc(&layer_context.attn_scores_v,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.attn_scores,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.q_proj,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.k_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.v_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.o_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.gate,
                        sizeof(__nv_bfloat16) * HIDDEN_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.up,
                        sizeof(__nv_bfloat16) * HIDDEN_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.down,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.embed_proj,
                        sizeof(__nv_bfloat16) * VOCAB_SIZE * MAX_PROMPT_LEN));

  Sampler sampler;
  sampler.BuildSampler(0.8, 0.92, 30, 1234);

  Prefill(weights, token_info, layer_context, kv_cache_context, sampler);
}

TEST(NNTest, Decode) {
  Weights weights{};
  int ret = LoadWeight(weights);

  printf("after loads weight\n");
  CheckGPUStatus();
  printf("\n\n");

  TokenInfo token_info;
  token_info.queue.push(
      {128000, 128006, 9125,  128007, 271,    38766,  1303,   33025,  2696,
       25,     6790,   220,   508,    1419,   271,    15724,  2696,   25,
       220,    2589,   4723,  220,    508,    1627,   271,    2675,   527,
       264,    11190,  18328, 13,     128009, 128006, 882,    128007, 271,
       791,    1401,   315,   2324,   374,    128009, 128006, 78191,  128007});
  token_info.is_slot_free.resize(5, true);
  token_info.slot = 0;
  CUDA_CHECK(
      cudaMalloc(&token_info.input_token_gpu,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  KVCacheContext kv_cache_context;
  CUDA_CHECK(cudaMalloc(&kv_cache_context.kv_cache, KV_CACHE_SIZE_BYTES));
  CUDA_CHECK(cudaMalloc(&kv_cache_context.block_table_gpu,
                        N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int)));

  kv_cache_context.free_blocks.resize(NUM_BLOCKS, -1);
  std::iota(kv_cache_context.free_blocks.begin(),
            kv_cache_context.free_blocks.end(), 0);
  kv_cache_context.block_table.resize(N_LAYERS * MAX_BLOCKS_PER_SEQ, -1);

  LayerContext layer_context;
  cublasStatus_t status = cublasCreate(&layer_context.cublas_handle);

  CUDA_CHECK(
      cudaMalloc(&layer_context.input,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.hidden_state,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.attn_scores_v,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.attn_scores,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.q_proj,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.k_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.v_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.o_proj,
                        sizeof(__nv_bfloat16) * KV_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.gate,
                        sizeof(__nv_bfloat16) * HIDDEN_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.up,
                        sizeof(__nv_bfloat16) * HIDDEN_DIM * MAX_PROMPT_LEN));

  CUDA_CHECK(
      cudaMalloc(&layer_context.down,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  CUDA_CHECK(cudaMalloc(&layer_context.embed_proj,
                        sizeof(__nv_bfloat16) * VOCAB_SIZE * MAX_PROMPT_LEN));

  Sampler sampler;
  sampler.BuildSampler(0.8, 0.92, 30, 1234);

  Prefill(weights, token_info, layer_context, kv_cache_context, sampler);
  Decode(weights, token_info, layer_context, kv_cache_context, sampler);

  std::vector<int> &prompt = token_info.queue.front();
  printf("[");
  for (int i = 0; i < prompt.size(); ++i) {
    printf("%d, ", prompt[i]);
  }
  printf("]\n");
}

TEST(NNTest, LoadWeight) {
  Weights weights{};
  int ret = LoadWeight(weights);
}
