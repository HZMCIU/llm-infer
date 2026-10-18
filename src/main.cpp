#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>

#include "json.hpp"
#include "kernel.cuh"

constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;

constexpr int MAX_PROMPT_LEN = 512;

constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_LENGTH = 2048;

using json = nlohmann::json;

int checkGPUStatus() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found\n";
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // std::cout << "Device: " << prop.name << "\n";
  // std::cout << "Compute capability: " << prop.major << "." << prop.minor
  //           << "\n";
  // std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
  // std::cout << "SM count: " << prop.multiProcessorCount << "\n";
  // std::cout << "Max threads per block: " << prop.maxThreadsPerBlock
  //           << std::endl;
  // size_t free_mem;
  // size_t total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "Free memory: " << free_mem / B_TO_GB
  //           << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
  return 0;
}

struct Weights {
  __nv_bfloat16 *embed_tokens;
  __nv_bfloat16 *input_layernorm[N_LAYERS];
  __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
  __nv_bfloat16 *mlp_up_proj[N_LAYERS];
  __nv_bfloat16 *mlp_down_proj[N_LAYERS];
  __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
  __nv_bfloat16 *w_k[N_LAYERS];
  __nv_bfloat16 *w_o[N_LAYERS];
  __nv_bfloat16 *w_q[N_LAYERS];
  __nv_bfloat16 *w_v[N_LAYERS];
  __nv_bfloat16 *norm;
};

int loadWeight(Weights &weights) {
  if (checkGPUStatus() != 0) {
    return 1;
  }

  std::string file_path =
      "/home/yuqyda/Data/cuda/models/LLM-Research--Llama-3.2-1B/snapshots/"
      "master/model.safetensors";
  std::ifstream safetensors_file(file_path, std::ios_base::binary);

  if (!safetensors_file.is_open()) {
    std::cout << "Can't open model.safetensors file" << std::endl;
    safetensors_file.close();
    return 1;
  }

  uint64_t header_size;
  safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
  std::string header;
  header.resize(header_size);
  // std::cout << header_size << std::endl;
  safetensors_file.read(header.data(), header_size);

  // std::cout << header << std::endl;

  std::unordered_map<std::string, uint64_t> offsets;
  json header_json = json::parse(header);
  uint64_t max_offset = 0;
  for (auto &[key, value] : header_json.items()) {
    if (key == "__metadata__") {
      continue;
    }

    uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
    if (offset_end > max_offset) {
      max_offset = offset_end;
    }
    offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
  }

  // std::cout << "max_offset: " << max_offset << std::endl;
  void *model_weights;
  cudaMalloc(&model_weights, max_offset);

  std::vector<char> model_weights_cpu;
  model_weights_cpu.resize(max_offset);

  cudaMalloc(&model_weights, max_offset);
  model_weights_cpu.resize(max_offset);
  safetensors_file.read(model_weights_cpu.data(), max_offset);

  cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset,
             cudaMemcpyHostToDevice);
  safetensors_file.close();

  weights.embed_tokens =
      (__nv_bfloat16 *)((char *)model_weights +
                        offsets.at("model.embed_tokens.weight"));
  weights.norm = (__nv_bfloat16 *)((char *)model_weights +
                                   offsets.at("model.norm.weight"));

  for (int i = 0; i < N_LAYERS; ++i) {
    // std::cout << "load " << (i + 1) << "-th layer weight" << std::endl;
    weights.input_layernorm[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".input_layernorm.weight"));
    weights.mlp_down_proj[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".mlp.down_proj.weight"));
    weights.mlp_gate_proj[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".mlp.gate_proj.weight"));
    weights.mlp_up_proj[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".mlp.up_proj.weight"));
    weights.post_attn_layernorms[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".post_attention_layernorm.weight"));
    weights.w_k[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".self_attn.k_proj.weight"));
    weights.w_o[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".self_attn.o_proj.weight"));
    weights.w_q[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".self_attn.q_proj.weight"));
    weights.w_v[i] =
        (__nv_bfloat16 *)((char *)model_weights +
                          offsets.at("model.layers." + std::to_string(i) +
                                     ".self_attn.v_proj.weight"));
  }
  return 0;
}

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

  std::vector<__nv_bfloat16> host_input_token_embedding(MAX_PROMPT_LEN *
                                                        EMBEDDING_LENGTH);
  cudaMemcpy(host_input_token_embedding.data(), input_token_embedding,
             sizeof(__nv_bfloat16) * prompt_len * EMBEDDING_LENGTH,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < prompt_len; ++i) {
    printf("%d-th\n", i + 1);
    for (int j = 0; j < 2048; ++j) {
      printf("%f ", __bfloat162float(host_input_token_embedding[i * 2048 + j]));
    }
    printf("%s\n");
  }

  return 0;
}
