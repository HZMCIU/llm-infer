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
#include "tokenizer.h"

using json = nlohmann::json;

int InitializeTransformer(Weights &weights, LayerContext &layer_context) {
  if (LoadWeight(weights) != 0) {
    return 1;
  }

  cublasStatus_t status = cublasCreate(&layer_context.cublas_handle);

  InitRopeFrequency(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f, 8192);

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

  return 0;
}

int main(int argc, char *argv[]) {
  Weights weights{};
  LayerContext layer_context;

  InitializeTransformer(weights, layer_context);

  Tokenizer tokenizer;
  std::string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  tokenizer.BuildTokenizer(dir_path);

  std::string input = "the key of life is";
  std::string user_prompt = tokenizer.RenderPrompt(input);

  std::vector<int> tokens;
  tokenizer.Encode(user_prompt, tokens);

  TokenInfo token_info;
  token_info.queue.push(tokens);
  token_info.is_slot_free.resize(5, true);
  token_info.slot = 0;
  CUDA_CHECK(
      cudaMalloc(&token_info.input_token_gpu,
                 sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * MAX_PROMPT_LEN));

  Prefill(weights, token_info, layer_context);
  Decode(weights, token_info, layer_context);

  std::vector<int> prompt = token_info.queue.front();

  for (int token : prompt) {
    printf("%s", tokenizer.Decode(token).c_str());
  }

  std::cout << "\nOk bye!" << std::endl;
  cublasDestroy(layer_context.cublas_handle);
  cudaDeviceSynchronize();
  return 0;
}
