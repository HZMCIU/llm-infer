#include "kernel.cuh"
#include <iostream>
#include <vector>

constexpr int HEAD_DIM = 64;

float *gpu_inv_freq = nullptr;
float *gpu_cos_table = nullptr;
float *gpu_sin_table = nullptr;

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

void init_rope_frequency(int head_dim, int max_token_seq_num, float rope_theta,
                         float factor, float low_freq_factor,
                         float high_freq_factor, int max_hidden_dimension) {
  int half_dim = head_dim / 2;
  std::vector<float> inv_freq(half_dim);
  for (int i = 0; i < half_dim; i++) {
    inv_freq[i] = 1.0f / std::pow(rope_theta, 2.0f * i / head_dim);
  }

  float low_freq_wave = (float)max_hidden_dimension / low_freq_factor;
  float high_freq_wave = (float)max_hidden_dimension / high_freq_factor;

  std::vector<float> inv_freq_llama(half_dim);
  for (int i = 0; i < half_dim; ++i) {
    float wave_len = 2 * M_PI / inv_freq[i];

    if (wave_len > low_freq_wave) {
      inv_freq_llama[i] = inv_freq[i] / factor;
    } else if (wave_len >= high_freq_wave) {
      float smooth =
          ((float)max_hidden_dimension / wave_len - low_freq_factor) /
          (high_freq_factor - low_freq_factor);

      inv_freq_llama[i] =
          (1.0f - smooth) * (inv_freq[i] / factor) + inv_freq[i] * smooth;
    }
  }

  cudaMalloc(&gpu_inv_freq, sizeof(float) * half_dim);
  cudaMemcpy(gpu_inv_freq, inv_freq_llama.data(), sizeof(float) * half_dim,
             cudaMemcpyHostToDevice);

  std::vector<float> cos_table(max_token_seq_num * head_dim);
  std::vector<float> sin_table(max_token_seq_num * head_dim);

  for (int pos = 0; pos < max_token_seq_num; ++pos) {
    for (int dim = 0; dim < half_dim; ++dim) {
      float angle = inv_freq_llama[dim] * pos;
      float c = std::cos(angle);
      float s = std::sin(angle);

      cos_table[pos * head_dim + 2 * dim] = c;
      cos_table[pos * head_dim + 2 * dim + 1] = c;
      sin_table[pos * head_dim + 2 * dim + 1] = s;
      sin_table[pos * head_dim + 2 * dim + 1] = s;
    }
  }

  cudaMalloc(&gpu_cos_table, sizeof(float) * max_token_seq_num * head_dim);
  cudaMalloc(&gpu_sin_table, sizeof(float) * max_token_seq_num * head_dim);
  cudaMemcpy(gpu_cos_table, cos_table.data(),
             sizeof(float) * max_token_seq_num * head_dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_sin_table, cos_table.data(),
             sizeof(float) * max_token_seq_num * head_dim,
             cudaMemcpyHostToDevice);
}

void free_rope_frequency() {
  if (gpu_inv_freq) {
    cudaFree(gpu_inv_freq);
    gpu_inv_freq = nullptr;
  }

  if (gpu_sin_table) {
    cudaFree(gpu_sin_table);
    gpu_sin_table = nullptr;
  }

  if (gpu_cos_table) {
    cudaFree(gpu_cos_table);
    gpu_cos_table = nullptr;
  }
}

__global__ void ropeKernel(__nv_bfloat16 *input, int head_dim, int proj_dim,
                           float *gpu_cos_table, float *gpu_sin_table) {
  int half_dim = head_dim / 2;
  int half_proj = proj_dim / 2;

  if (threadIdx.x >= half_proj) {
    return;
  }

  int head_idx = threadIdx.x / half_dim;
  int pair_idx = threadIdx.x % half_dim;

  int token_idx = blockIdx.x;

  int base = token_idx * proj_dim + head_idx * head_dim;

  int idx1 = base + pair_idx;
  int idx2 = base + pair_idx + half_dim;

  float x1 = (float)input[idx1];
  float x2 = (float)input[idx2];

  int table_idx = token_idx * head_dim + pair_idx * 2;
  float c = gpu_cos_table[table_idx];
  float s = gpu_sin_table[table_idx];

  input[idx1] = (__nv_bfloat16)(x1 * c - x2 * s);
  input[idx2] = (__nv_bfloat16)(x1 * s + x2 * c);
}

void rope(__nv_bfloat16 *input, int proj_dim, int token_num) {
  int num_thread = proj_dim / 2;
  if (num_thread > 2) {
    std::cout << "max thread num in a block is 1024" << std::endl;
    return;
  }

  ropeKernel<<<token_num, num_thread>>>(input, proj_dim, HEAD_DIM,
                                        gpu_cos_table, gpu_sin_table);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error)
              << " (code: " << error << ")" << std::endl;
  }
}

__global__ void residualAddKernel(__nv_bfloat16 *input,
                                  __nv_bfloat16 *input_embed) {
  int work_idx = threadIdx.x + blockIdx.x * 2048;

  input[work_idx] = input[work_idx] + input_embed[work_idx];
  input[work_idx + 1024] = input[work_idx] + input_embed[work_idx + 1024];
}

void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embed,
                 int token_num) {
  residualAddKernel<<<token_num, 1024>>>(input, input_embed);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error)
              << " (code: " << error << ")" << std::endl;
  }
}

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b) {
  int work_idx = blockIdx.x * 8196 + threadIdx.x;

  for (int i = 0; i < 8196; i += 1024) {
    a[work_idx + i] =
        (__nv_bfloat16)((float)a[work_idx + i] *
                            (1.0f / (1.0f + expf(-(float)a[work_idx + i]))) +
                        (float)b[work_idx + i]);
  }
}

void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int token_num) {
  siluKernel<<<token_num, 1024>>>(a, b);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error)
              << " (code: " << error << ")" << std::endl;
  }
}
