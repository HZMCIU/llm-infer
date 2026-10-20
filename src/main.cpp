#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>

#include "json.hpp"
#include "kernel.cuh"
#include "model.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  Weights weights{};
  loadWeight(weights);

  std::vector<int> prompt{128000, 9906, 4435};
  int prompt_len = 3;
  __nv_bfloat16 *input_token_embedding;
  cudaMalloc(&input_token_embedding,
             MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);

  __nv_bfloat16 *gpu_input_token;
  cudaMalloc(&gpu_input_token, sizeof(int) * prompt_len);
  cudaMemcpy(gpu_input_token, prompt.data(), sizeof(int) * prompt_len,
             cudaMemcpyHostToDevice);
  embeddingGather(prompt.data(), weights.embed_tokens, input_token_embedding,
                  prompt_len);

  std::vector<__nv_bfloat16> host_input_token_embedding(prompt_len *
                                                        EMBEDDING_LENGTH);
  cudaMemcpy(host_input_token_embedding.data(), input_token_embedding,
             sizeof(__nv_bfloat16) * prompt_len * EMBEDDING_LENGTH,
             cudaMemcpyDeviceToHost);

  std::cout << host_input_token_embedding.size() << std::endl;

  __nv_bfloat16 *gpu_rms_norms;
  cudaMalloc(&gpu_input_token,
             sizeof(__nv_bfloat16) * prompt_len * EMBEDDING_LENGTH);
  rmsNorm(input_token_embedding, gpu_rms_norms, weights.input_layernorm[0],
          prompt_len);

  cudaDeviceSynchronize();
  std::vector<__nv_bfloat16> host_rms_norm(prompt_len * EMBEDDING_LENGTH);
  cudaMemcpy(host_rms_norm.data(), gpu_rms_norms,
             sizeof(__nv_bfloat16) * prompt_len * EMBEDDING_LENGTH,
             cudaMemcpyDeviceToHost);

  std::cout << host_rms_norm.size() << std::endl;
  return 0;
}
