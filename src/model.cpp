#include "model.h"

#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "json.hpp"

#include "helper.h"
#include "kernel.cuh"

using json = nlohmann::json;

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
  // TODO fake load
  // max_offset = 2349985792;

  std::vector<char> model_weights_cpu;
  model_weights_cpu.resize(max_offset);

  CUDA_CHECK(cudaMalloc(&model_weights, max_offset));
  model_weights_cpu.resize(max_offset);
  safetensors_file.read(model_weights_cpu.data(), max_offset);

  CUDA_CHECK(cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset,
                        cudaMemcpyHostToDevice));
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

void freeWeight(Weights &weights) {
  CUDA_CHECK(cudaFree(weights.embed_tokens));
  CUDA_CHECK(cudaFree(weights.norm));

  for (int layer = 0; layer < N_LAYERS; ++layer) {
    CUDA_CHECK(cudaFree(weights.input_layernorm[layer]));
    CUDA_CHECK(cudaFree(weights.mlp_gate_proj[layer]));
    CUDA_CHECK(cudaFree(weights.mlp_up_proj[layer]));
    CUDA_CHECK(cudaFree(weights.mlp_down_proj[layer]));
    CUDA_CHECK(cudaFree(weights.post_attn_layernorms[layer]));
    CUDA_CHECK(cudaFree(weights.w_k[layer]));
    CUDA_CHECK(cudaFree(weights.w_o[layer]));
    CUDA_CHECK(cudaFree(weights.w_q[layer]));
    CUDA_CHECK(cudaFree(weights.w_v[layer]));
  }
}

void prefill(std::vector<int> &prompt, std::queue<std::vector<int>> &queue,
             int &prompt_len, std::vector<bool> &is_slot_free, int slot,
             int *gpu_input_tokens, nv_bfloat16 *input_embeddings,
             Weights &weights, nv_bfloat16 *hidden_state,
             nv_bfloat16 *rms_norms, nv_bfloat16 *&q_proj,
             nv_bfloat16 *buf_2048_1, cublasHandle_t cublas_handle,
             float &q_proj_alpha, float &q_proj_beta, float &k_proj_alpha,
             float &k_proj_beta, float &v_proj_alpha, float &v_proj_beta,
             nv_bfloat16 *prefill_attn_scores, float &attn_alpha,
             float &attn_beta, nv_bfloat16 *&attn_scores_v,
             float &attn_scores_v_alpha, float &attn_scores_v_beta,
             nv_bfloat16 *&o_proj, nv_bfloat16 *buf_2048_2, float &o_proj_alpha,
             float &o_proj_beta, float &gate_alpha, float &gate_beta,
             nv_bfloat16 *gate, float &up_alpha, float &up_beta,
             nv_bfloat16 *up, nv_bfloat16 *&down, float &down_alpha,
             float &down_beta, float &embed_alpha, float &embed_beta,
             nv_bfloat16 *embed_proj, std::vector<nv_bfloat16> &embed_proj_cpu,
             std::vector<std::vector<int>> &generated_tokens,
             std::vector<int> &last_generated_tokens,
             std::vector<int> &current_prompt_len,
             __nv_bfloat16 *k_proj_temp_buf, __nv_bfloat16 *v_proj_temp_buf,
             std::vector<int> &block_table, int *block_table_gpu,
             std::vector<int> &free_blocks, __nv_bfloat16 *kv_cache) {
  prompt = queue.front();
  prompt_len = prompt.size();
  queue.pop();
  is_slot_free[slot] = false;

  cudaMemcpy(gpu_input_tokens, prompt.data(), prompt_len * sizeof(int),
             cudaMemcpyHostToDevice);
  embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens,
                  prompt_len);

  cudaMemcpy(hidden_state, input_embeddings,
             prompt_len * EMBEDDING_LENGTH * sizeof(__nv_bfloat16),
             cudaMemcpyDeviceToDevice);

  for (int layer = 0; layer < N_LAYERS; ++layer) {
    rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer],
            prompt_len);

    q_proj = buf_2048_1;
    cublasStatus_t q_proj_staus = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH, prompt_len,
        EMBEDDING_LENGTH, &q_proj_alpha, weights.w_q[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
        &q_proj_beta, q_proj, CUDA_R_16BF, EMBEDDING_LENGTH, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cublasStatus_t k_proj_status = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, prompt_len,
        EMBEDDING_LENGTH, &k_proj_alpha, weights.w_k[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
        &k_proj_beta, k_proj_temp_buf, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cublasStatus_t v_proj_status = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, prompt_len,
        EMBEDDING_LENGTH, &v_proj_alpha, weights.w_v[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
        &v_proj_beta, v_proj_temp_buf, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    rope(q_proj, prompt_len, EMBEDDING_LENGTH);
    rope(k_proj_temp_buf, prompt_len, KV_DIM);

    for (int token_idx = 0; token_idx < prompt_len; token_idx += BLOCK_SIZE) {
      int num_tokens_to_copy = prompt_len - token_idx;
      if (num_tokens_to_copy > BLOCK_SIZE) {
        num_tokens_to_copy = BLOCK_SIZE;
      }

      int block_idx = token_idx / BLOCK_SIZE;
      int block = block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                              layer * MAX_BLOCKS_PER_SEQ + block_idx];

      if (block == -1) {
        int physical_block_idx = free_blocks.back();
        free_blocks.pop_back();
        block = physical_block_idx;
        block_table[slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                    layer * MAX_BLOCKS_PER_SEQ + block_idx] = block;
      } else {
        assert(false && "block must be -1 during prefill - what happended?");
      }

      __nv_bfloat16 *k_cache_ptr =
          (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES);
      __nv_bfloat16 *k_proj_ptr = k_proj_temp_buf + token_idx * KV_DIM;
      cudaMemcpy(k_cache_ptr, k_proj_ptr,
                 num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToDevice);

      __nv_bfloat16 *v_cache_ptr =
          (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_SIZE + V_OFFSET);
      __nv_bfloat16 *v_proj_ptr = v_proj_temp_buf + token_idx * KV_DIM;
      cudaMemcpy(v_cache_ptr, v_proj_ptr,
                 num_tokens_to_copy * KV_DIM * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToDevice);
    }

    for (int i = 0; i < NUM_Q_HEADS; ++i) {
      int k_head_idx = i / GQA_Q_TO_K_RATIO;
      __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
      __nv_bfloat16 *k_head = k_proj_temp_buf + k_head_idx * HEAD_DIM;
      __nv_bfloat16 *attn_score_head =
          prefill_attn_scores + prompt_len * prompt_len * i;

      cublasStatus_t attn_score_status = cublasGemmEx(
          cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, prompt_len, prompt_len,
          HEAD_DIM, &attn_alpha, k_head, CUDA_R_16BF, KV_DIM, q_head,
          CUDA_R_16BF, EMBEDDING_LENGTH, &attn_beta, attn_score_head,
          CUDA_R_16BF, prompt_len, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    causalMask(prefill_attn_scores, prompt_len);
    softmax(prefill_attn_scores, prompt_len);

    attn_scores_v = buf_2048_1;

    for (int i = 0; i < NUM_Q_HEADS; ++i) {
      int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;

      __nv_bfloat16 *attn_scores_head =
          prefill_attn_scores + i * prompt_len * prompt_len;
      __nv_bfloat16 *v_head = v_proj_temp_buf + v_head_idx * HEAD_DIM;
      __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

      cublasStatus_t attn_score_status = cublasGemmEx(
          cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HEAD_DIM, prompt_len,
          prompt_len, &attn_scores_v_alpha, v_head, CUDA_R_16BF, KV_DIM,
          attn_scores_head, CUDA_R_16BF, prompt_len, &attn_scores_v_beta,
          output_attn_scores_head, CUDA_R_16BF, EMBEDDING_LENGTH,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    o_proj = buf_2048_2;
    cublasStatus_t o_proj_staus = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH, prompt_len,
        EMBEDDING_LENGTH, &o_proj_alpha, weights.w_o[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH, attn_scores_v, CUDA_R_16BF, EMBEDDING_LENGTH,
        &o_proj_beta, o_proj, CUDA_R_16BF, EMBEDDING_LENGTH, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    residualAdd(hidden_state, o_proj, prompt_len);
    rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer],
            prompt_len);

    cublasStatus_t gate_status = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, prompt_len,
        EMBEDDING_LENGTH, &gate_alpha, weights.mlp_gate_proj[layer],
        CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
        &gate_beta, gate, CUDA_R_16BF, HIDDEN_DIM, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cublasStatus_t up_status = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, prompt_len,
        EMBEDDING_LENGTH, &up_alpha, weights.mlp_up_proj[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &up_beta,
        up, CUDA_R_16BF, HIDDEN_DIM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    silu(gate, up, prompt_len);

    down = buf_2048_2;
    cublasStatus_t down_status = cublasGemmEx(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH, prompt_len,
        HIDDEN_DIM, &down_alpha, weights.mlp_down_proj[layer], CUDA_R_16BF,
        HIDDEN_DIM, gate, CUDA_R_16BF, HIDDEN_DIM, &down_beta, down,
        CUDA_R_16BF, EMBEDDING_LENGTH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    residualAdd(hidden_state, down, prompt_len);
  }

  rmsNorm(hidden_state, rms_norms, weights.norm, prompt_len);

  cublasStatus_t embed_status = cublasGemmEx(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, prompt_len,
      EMBEDDING_LENGTH, &embed_alpha, weights.embed_tokens, CUDA_R_16BF,
      EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &embed_beta,
      embed_proj, CUDA_R_16BF, VOCAB_SIZE, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);

  cudaMemcpy(embed_proj_cpu.data(), embed_proj,
             sizeof(__nv_bfloat16) * prompt_len * VOCAB_SIZE,
             cudaMemcpyDeviceToHost);

  int last_token_offset = (prompt_len - 1) * VOCAB_SIZE;
  float max_token = (float)embed_proj_cpu[last_token_offset];
  int max_token_idx = 0;
  for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx) {
    if ((float)embed_proj_cpu[token_idx + last_token_offset] > max_token) {
      max_token = embed_proj_cpu[token_idx + last_token_offset];
      max_token_idx = token_idx;
    }
  }

  std::cout << "Output token: " << (float)max_token
            << ", token index: " << std::to_string(max_token_idx) << std::endl;

  generated_tokens[slot].push_back(max_token_idx);
  last_generated_tokens[slot] = max_token_idx;
  current_prompt_len[slot] = prompt_len;

  cudaMemcpy(block_table_gpu, block_table.data(),
             MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
             cudaMemcpyHostToDevice);
}

void TokenEmbedding(Weights &weights, TokenInfo &token_info,
                    LayerContext &layer_context) {
  std::vector<int> prompt = token_info.queue.front();
  int prompt_len = prompt.size();
  token_info.queue.pop();

  CUDA_CHECK(cudaMemcpy(token_info.input_token_gpu, prompt.data(),
                        prompt_len * sizeof(int), cudaMemcpyHostToDevice));
  embeddingGather(token_info.input_token_gpu, weights.embed_tokens,
                  layer_context.input, prompt_len);
  CUDA_CHECK(cudaMemcpy(layer_context.hidden_state, layer_context.input,
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * prompt_len,
                        cudaMemcpyDeviceToDevice));
}

// 1. 输出为人类可读的 .txt 文件 (如果指针在 GPU 上，自动拷回 CPU 并转 float)
void dump_bf16_to_txt(const std::string &filename, const __nv_bfloat16 *d_ptr,
                      size_t size, size_t cols = 4096,
                      bool is_device_ptr = true) {
  // 拷贝回 CPU
  std::vector<__nv_bfloat16> h_bf16(size);
  if (is_device_ptr) {
    cudaDeviceSynchronize();
    cudaMemcpy(h_bf16.data(), d_ptr, size * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
  } else {
    std::copy(d_ptr, d_ptr + size, h_bf16.begin());
  }

  // 打开文件并设置不使用科学计数法
  std::ofstream out(filename);
  out << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < size; ++i) {
    // 核心：使用 __bfloat162float 将 bf16 无损转为 float 再打印
    float val = __bfloat162float(h_bf16[i]);
    out << val;
    if ((i + 1) % cols == 0 || i == size - 1) {
      out << "\n";
    } else {
      out << " ";
    }
  }
  out.close();
}

int Prefill_(Weights &weights, TokenInfo &token_info,
             LayerContext &layer_context) {
  std::vector<int> prompt = token_info.queue.front();
  int prompt_len = prompt.size();

  TokenEmbedding(weights, token_info, layer_context);
  init_rope_frequency(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f,
                      8192);

  for (int layer = 0; layer < N_LAYERS; ++layer) {
    rmsNorm(layer_context.hidden_state, layer_context.input,
            weights.input_layernorm[layer], prompt_len);

    cublasStatus_t q_proj_staus =
        cublasGemmEx(layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

                     EMBEDDING_LENGTH, prompt_len, EMBEDDING_LENGTH,

                     &layer_context.q_proj_alpha, weights.w_q[layer],
                     CUDA_R_16BF, EMBEDDING_LENGTH,

                     layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

                     &layer_context.q_proj_beta, layer_context.q_proj,
                     CUDA_R_16BF, EMBEDDING_LENGTH,

                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasStatus_t k_proj_status = cublasGemmEx(
        layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

        KV_DIM, prompt_len, EMBEDDING_LENGTH,

        &layer_context.k_proj_alpha, weights.w_k[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH,

        layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

        &layer_context.k_proj_beta, layer_context.k_proj, CUDA_R_16BF, KV_DIM,

        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasStatus_t v_proj_status = cublasGemmEx(
        layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

        KV_DIM, prompt_len, EMBEDDING_LENGTH,

        &layer_context.v_proj_alpha, weights.w_v[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH,

        layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

        &layer_context.v_proj_beta, layer_context.v_proj, CUDA_R_16BF, KV_DIM,

        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    rope(layer_context.q_proj, prompt_len, EMBEDDING_LENGTH);
    rope(layer_context.k_proj, prompt_len, KV_DIM);

    for (int i = 0; i < NUM_Q_HEADS; ++i) {
      int k_head_idx = i / GQA_Q_TO_K_RATIO;
      __nv_bfloat16 *q_head = layer_context.q_proj + i * HEAD_DIM;
      __nv_bfloat16 *k_head = layer_context.k_proj + k_head_idx * HEAD_DIM;
      __nv_bfloat16 *attn_score_head =
          layer_context.attn_scores + prompt_len * prompt_len * i;

      cublasStatus_t attn_score_status = cublasGemmEx(
          layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

          prompt_len, prompt_len, HEAD_DIM,

          &layer_context.attn_alpha, k_head, CUDA_R_16BF, KV_DIM,

          q_head, CUDA_R_16BF, EMBEDDING_LENGTH,

          &layer_context.attn_beta, attn_score_head, CUDA_R_16BF, prompt_len,

          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    causalMask(layer_context.attn_scores, prompt_len);
    softmax(layer_context.attn_scores, prompt_len);

    for (int i = 0; i < NUM_Q_HEADS; ++i) {
      int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO;

      __nv_bfloat16 *attn_scores_head =
          layer_context.attn_scores + i * prompt_len * prompt_len;
      __nv_bfloat16 *v_head = layer_context.v_proj + v_head_idx * HEAD_DIM;
      __nv_bfloat16 *attn_scores_v_head =
          layer_context.attn_scores_v + i * HEAD_DIM;

      cublasStatus_t attn_score_status = cublasGemmEx(
          layer_context.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,

          HEAD_DIM, prompt_len, prompt_len,

          &layer_context.attn_scores_v_alpha, v_head, CUDA_R_16BF, KV_DIM,

          attn_scores_head, CUDA_R_16BF, prompt_len,

          &layer_context.attn_scores_v_beta, attn_scores_v_head, CUDA_R_16BF,
          EMBEDDING_LENGTH,

          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    cublasStatus_t o_proj_staus =
        cublasGemmEx(layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

                     EMBEDDING_LENGTH, prompt_len, EMBEDDING_LENGTH,

                     &layer_context.o_proj_alpha, weights.w_o[layer],
                     CUDA_R_16BF, EMBEDDING_LENGTH,

                     layer_context.attn_scores_v, CUDA_R_16BF, EMBEDDING_LENGTH,

                     &layer_context.o_proj_beta, layer_context.o_proj,
                     CUDA_R_16BF, EMBEDDING_LENGTH,

                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    residualAdd(layer_context.hidden_state, layer_context.o_proj, prompt_len);

    rmsNorm(layer_context.hidden_state, layer_context.input,
            weights.post_attn_layernorms[layer], prompt_len);

    cublasStatus_t gate_status = cublasGemmEx(
        layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

        HIDDEN_DIM, prompt_len, EMBEDDING_LENGTH,

        &layer_context.gate_alpha, weights.mlp_gate_proj[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH,

        layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

        &layer_context.gate_beta, layer_context.gate, CUDA_R_16BF, HIDDEN_DIM,

        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cublasStatus_t up_status = cublasGemmEx(
        layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

        HIDDEN_DIM, prompt_len, EMBEDDING_LENGTH,

        &layer_context.up_alpha, weights.mlp_up_proj[layer], CUDA_R_16BF,
        EMBEDDING_LENGTH,

        layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

        &layer_context.up_beta, layer_context.up, CUDA_R_16BF, HIDDEN_DIM,

        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    silu(layer_context.gate, layer_context.up, prompt_len);

    cublasStatus_t down_status =
        cublasGemmEx(layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     EMBEDDING_LENGTH, prompt_len, HIDDEN_DIM,

                     &layer_context.down_alpha, weights.mlp_down_proj[layer],
                     CUDA_R_16BF, HIDDEN_DIM,

                     layer_context.gate, CUDA_R_16BF, HIDDEN_DIM,

                     &layer_context.down_beta, layer_context.down, CUDA_R_16BF,
                     EMBEDDING_LENGTH,

                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    residualAdd(layer_context.hidden_state, layer_context.down, prompt_len);

    if (layer == 0) {
      dump_bf16_to_txt("/home/yuqyda/Data/cuda/llm-infer/test/a.txt",
                       layer_context.hidden_state,
                       prompt_len * EMBEDDING_LENGTH, EMBEDDING_LENGTH);
    }
  }

  rmsNorm(layer_context.hidden_state, layer_context.input, weights.norm,
          prompt_len);

  cublasStatus_t embed_status =
      cublasGemmEx(layer_context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,

                   VOCAB_SIZE, prompt_len, EMBEDDING_LENGTH,

                   &layer_context.embed_alpha, weights.embed_tokens,
                   CUDA_R_16BF, EMBEDDING_LENGTH,

                   layer_context.input, CUDA_R_16BF, EMBEDDING_LENGTH,

                   &layer_context.embed_beta, layer_context.embed_proj,
                   CUDA_R_16BF, VOCAB_SIZE,

                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  std::vector<__nv_bfloat16> hidden_state_cpu(sizeof(__nv_bfloat16) *
                                              EMBEDDING_LENGTH * prompt_len);
  CUDA_CHECK(cudaMemcpy(hidden_state_cpu.data(), layer_context.hidden_state,
                        sizeof(__nv_bfloat16) * EMBEDDING_LENGTH * prompt_len,
                        cudaMemcpyDeviceToHost));
  return 0;
}
