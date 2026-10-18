#include "kernel.cuh"
#include <iostream>

__global__ void embeddingGatherKernel(int *input_token,
                                      __nv_bfloat16 *model_token_embedding,
                                      __nv_bfloat16 *input_token_embedding,
                                      int input_token_num) {
  int workIdx = blockIdx.x * 2048 + threadIdx.x;
  if (workIdx >= input_token_num * 2048) {
    return;
  }

  input_token_embedding[workIdx] =
      model_token_embedding[input_token[blockIdx.x] * 2048 + threadIdx.x];
  input_token_embedding[workIdx + 1024] =
      model_token_embedding[input_token[blockIdx.x] * 2048 + threadIdx.x +
                            1024];
}

void embeddingGather(int *input_token, __nv_bfloat16 *model_token_embedding,
                     __nv_bfloat16 *input_token_embedding,
                     int input_token_num) {
  embeddingGatherKernel<<<input_token_num, 1024>>>(
      input_token, model_token_embedding, input_token_embedding,
      input_token_num);

  cudaError error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
  }
}
