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

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output,
                              __nv_bfloat16 *model_norm_weight, int token_num) {
  __shared__ float rms_vector[1024];
  int workIdx = blockIdx.x * 2048 + threadIdx.x;
  if (workIdx >= blockIdx.x * 2048) {
    return;
  }

  rms_vector[threadIdx.x] =
      (float)input[workIdx] * (float)input[workIdx] +
      (float)input[workIdx + 1024] * (float)input[workIdx + 1024];

  __syncthreads();

  for (int i = 1; i < 1024; i = i * 2) {
    if ((threadIdx.x) % (i * 2) == 0) {
      rms_vector[threadIdx.x] =
          rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1e-15);
  }
  __syncthreads();

  output[workIdx] = (__nv_bfloat16)(((float)input[workIdx] / rms_vector[0]) *
                                    (float)model_norm_weight[threadIdx.x]);
  output[workIdx + 1024] =
      (__nv_bfloat16)(((float)input[workIdx + 1024] / rms_vector[0]) *
                      (float)model_norm_weight[threadIdx.x + 1024]);
}

void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output,
             __nv_bfloat16 *norm_weight, int token_num) {
  rmsNormKernel<<<token_num, 1024>>>(input, output, norm_weight, token_num);
}
