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
#include "model.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  cublasHandle_t cublas_handle;
  cublasStatus_t status = cublasCreate(&cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "cuBLAS init failed, status: " << status << std::endl;
    return 1;
  }

  Weights weights{};
  if (loadWeight(weights) != 0) {
    return 1;
  }

  init_rope_frequency(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f,
                      8192);

  __nv_bfloat16 *kv_cache;
  CUDA_CHECK(cudaMalloc(&kv_cache, KV_CACHE_SIZE_BYTES));
  std::vector<int> free_blocks(NUM_BLOCKS);
  std::iota(free_blocks.begin(), free_blocks.end(), 0);
  std::vector<int> block_table(MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ,
                               -1);

  int *block_table_gpu;
  CUDA_CHECK(
      cudaMalloc(&block_table_gpu,
                 MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int)));

  std::queue<std::vector<int>> queue;
  queue.push({128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30,
              128009, 128006, 78191, 128007, 271});

  // PROMPT 1 (Name a color.) - length 14
  queue.push({128000, 128006, 882, 128007, 271, 678, 264, 1933, 13, 128009,
              128006, 78191, 128007, 271});

  // PROMPT 2 (Say hello.) - length 13
  queue.push({128000, 128006, 882, 128007, 271, 46864, 24748, 13, 128009,
              128006, 78191, 128007, 271});

  // PROMPT 3 (Capital of France?) - length 14
  queue.push({128000, 128006, 882, 128007, 271, 64693, 315, 9822, 30, 128009,
              128006, 78191, 128007, 271});

  std::vector<bool> is_slot_free(BATCH_SIZE, true);

  std::vector<std::vector<int>> generated_tokens(BATCH_SIZE);
  std::vector<int> last_generated_tokens(BATCH_SIZE);
  std::vector<int> current_prompt_len(BATCH_SIZE, 0);

  std::vector<int> active_slots;
  std::vector<int> active_tokens;

  int *active_slots_gpu;
  CUDA_CHECK(cudaMalloc(&active_slots_gpu, BATCH_SIZE * sizeof(int)));
  int *seq_lens_gpu;
  CUDA_CHECK(cudaMalloc(&seq_lens_gpu, BATCH_SIZE * sizeof(int)));

  std::vector<int> prompt;
  int prompt_len;
  int *input_tokens_gpu;
  CUDA_CHECK(cudaMalloc(&input_tokens_gpu, MAX_PROMPT_LEN * sizeof(int)));
  __nv_bfloat16 *input_embeddings;
  CUDA_CHECK(
      cudaMalloc(&input_embeddings,
                 MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH));

  __nv_bfloat16 *hidden_state;
  CUDA_CHECK(cudaMalloc(&hidden_state, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) *
                                           EMBEDDING_LENGTH));

  __nv_bfloat16 *rms_norms;
  CUDA_CHECK(cudaMalloc(&rms_norms, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) *
                                        EMBEDDING_LENGTH));

  __nv_bfloat16 *buf_2048_1;
  CUDA_CHECK(cudaMalloc(&buf_2048_1, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) *
                                         EMBEDDING_LENGTH));
  __nv_bfloat16 *q_proj;
  float q_proj_alpha = 1.0f;
  float q_proj_beta = 0.0f;

  __nv_bfloat16 *k_proj_temp_buf;
  CUDA_CHECK(cudaMalloc(&k_proj_temp_buf,
                        MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16)));
  __nv_bfloat16 *v_proj_temp_buf;
  CUDA_CHECK(cudaMalloc(&v_proj_temp_buf,
                        MAX_PROMPT_LEN * KV_DIM * sizeof(__nv_bfloat16)));

  float k_proj_alpha = 1.0f;
  float k_proj_beta = 0.0f;

  float v_proj_alpha = 1.0f;
  float v_proj_beta = 0.0f;

  __nv_bfloat16 *prefill_attn_scores;
  CUDA_CHECK(cudaMalloc(&prefill_attn_scores, MAX_PROMPT_LEN * MAX_PROMPT_LEN *
                                                  sizeof(__nv_bfloat16) *
                                                  NUM_Q_HEADS));

  float attn_alpha = 1.0f / 8.0f;
  float attn_beta = 0.0f;

  __nv_bfloat16 *attn_scores_v;
  float attn_scores_v_alpha = 1.0f;
  float attn_scores_v_beta = 0.0f;

  __nv_bfloat16 *buf_2048_2;
  CUDA_CHECK(cudaMalloc(&buf_2048_2, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) *
                                         EMBEDDING_LENGTH));
  __nv_bfloat16 *o_proj;
  float o_proj_alpha = 1.0f;
  float o_proj_beta = 0.0f;

  __nv_bfloat16 *gate;
  CUDA_CHECK(
      cudaMalloc(&gate, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM));
  float gate_alpha = 1.0f;
  float gate_beta = 0.0f;

  __nv_bfloat16 *up;
  CUDA_CHECK(
      cudaMalloc(&up, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * HIDDEN_DIM));
  float up_alpha = 1.0f;
  float up_beta = 0.0f;

  __nv_bfloat16 *down;
  float down_alpha = 1.0f;
  float down_beta = 0.0f;

  __nv_bfloat16 *embed_proj;
  CUDA_CHECK(cudaMalloc(&embed_proj,
                        sizeof(__nv_bfloat16) * MAX_BUFFER_SIZE * VOCAB_SIZE));
  float embed_alpha = 1.0f;
  float embed_beta = 0.0f;

  std::vector<__nv_bfloat16> embed_proj_cpu;
  embed_proj_cpu.resize(MAX_BUFFER_SIZE * VOCAB_SIZE);

  int *last_tokens_gpu;
  CUDA_CHECK(cudaMalloc(&last_tokens_gpu, BATCH_SIZE * sizeof(int)));

  __nv_bfloat16 *k_proj_batched_buffer;
  CUDA_CHECK(cudaMalloc(&k_proj_batched_buffer,
                        BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM));

  __nv_bfloat16 *v_proj_batched_buffer;
  CUDA_CHECK(cudaMalloc(&v_proj_batched_buffer,
                        BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM));

  for (int slot = 0; slot < is_slot_free.size() && !queue.empty(); ++slot) {
    if (!is_slot_free[slot]) {
      continue;
    }

    prefill(prompt, queue, prompt_len, is_slot_free, slot, input_tokens_gpu,
            input_embeddings, weights, hidden_state, rms_norms, q_proj,
            buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta, k_proj_alpha,
            k_proj_beta, v_proj_alpha, v_proj_beta, prefill_attn_scores,
            attn_alpha, attn_beta, attn_scores_v, attn_scores_v_alpha,
            attn_scores_v_beta, o_proj, buf_2048_2, o_proj_alpha, o_proj_beta,
            gate_alpha, gate_beta, gate, up_alpha, up_beta, up, down,
            down_alpha, down_beta, embed_alpha, embed_beta, embed_proj,
            embed_proj_cpu, generated_tokens, last_generated_tokens,
            current_prompt_len, k_proj_temp_buf, v_proj_temp_buf, block_table,
            block_table_gpu, free_blocks, kv_cache);
  }

  while (true) {
    active_slots.clear();
    active_tokens.clear();
    for (int slot = 0; slot < BATCH_SIZE; ++slot) {
      if (is_slot_free[slot]) {
        if (queue.empty()) {
          continue;
        }
        generated_tokens[slot].clear();
        prefill(prompt, queue, prompt_len, is_slot_free, slot, input_tokens_gpu,
                input_embeddings, weights, hidden_state, rms_norms, q_proj,
                buf_2048_1, cublas_handle, q_proj_alpha, q_proj_beta,
                k_proj_alpha, k_proj_beta, v_proj_alpha, v_proj_beta,
                prefill_attn_scores, attn_alpha, attn_beta, attn_scores_v,
                attn_scores_v_alpha, attn_scores_v_beta, o_proj, buf_2048_2,
                o_proj_alpha, o_proj_beta, gate_alpha, gate_beta, gate,
                up_alpha, up_beta, up, down, down_alpha, down_beta, embed_alpha,
                embed_beta, embed_proj, embed_proj_cpu, generated_tokens,
                last_generated_tokens, current_prompt_len, k_proj_temp_buf,
                v_proj_temp_buf, block_table, block_table_gpu, free_blocks,
                kv_cache);
      }
      active_slots.push_back(slot);
      active_tokens.push_back(last_generated_tokens[slot]);
    }

    int active_slot_num = active_slots.size();
    if (active_slot_num == 0) {
      if (queue.empty()) {
        break;
      }
      continue;
    }

    cudaMemcpy(last_tokens_gpu, active_slots.data(),
               active_slot_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(active_slots_gpu, active_slots.data(),
               active_slot_num * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<int> seq_lens(active_slot_num);
    for (int slot = 0; slot < active_slot_num; ++slot) {
      int active_slot = active_slots[slot];
      seq_lens[slot] = current_prompt_len[active_slot] + 1;
    }
    cudaMemcpy(seq_lens_gpu, seq_lens.data(), seq_lens.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    embeddingGatherDecode(last_tokens_gpu, active_slot_num, hidden_state,
                          weights.embed_tokens);

    for (int layer = 0; layer < N_LAYERS; ++layer) {
      rmsNorm(hidden_state, rms_norms, weights.input_layernorm[layer],
              active_slot_num);
      q_proj = buf_2048_1;
      // q proj (num_prompts, 2048)
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   EMBEDDING_LENGTH, // m
                   active_slot_num,  // n
                   EMBEDDING_LENGTH, // k
                   &q_proj_alpha,
                   weights.w_q[layer], // A
                   CUDA_R_16BF,
                   EMBEDDING_LENGTH, // lda
                   rms_norms,        // B
                   CUDA_R_16BF,
                   EMBEDDING_LENGTH, // ldb
                   &q_proj_beta,
                   q_proj, // C
                   CUDA_R_16BF,
                   EMBEDDING_LENGTH, // ldc
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

      // k proj
      cublasGemmEx(
          cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
          KV_DIM,           // m = 512
          active_slot_num,  // n = num prompts
          EMBEDDING_LENGTH, // k = 2048
          &k_proj_alpha,
          weights.w_k[layer], // A
          CUDA_R_16BF,
          EMBEDDING_LENGTH, // lda 2048, because W_k is in memory as 512, 2048
          // so the gap between subsequent elements is 2048
          rms_norms, // B
          CUDA_R_16BF,
          EMBEDDING_LENGTH, // ldb, same reason for rms_norms
          &k_proj_beta,
          k_proj_batched_buffer, // TODO C
          CUDA_R_16BF,
          KV_DIM, // ldc = 512
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM,
                   active_slot_num, EMBEDDING_LENGTH, &v_proj_alpha,
                   weights.w_v[layer], CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms,
                   CUDA_R_16BF, EMBEDDING_LENGTH, &v_proj_beta,
                   v_proj_batched_buffer, CUDA_R_16BF, KV_DIM,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

      for (int slot = 0; slot < active_slot_num; ++slot) {
        int active_slot = active_slots[slot];
        ropeDecode(&q_proj[slot * EMBEDDING_LENGTH],
                   current_prompt_len[active_slot], EMBEDDING_LENGTH);
        ropeDecode(k_proj_batched_buffer + slot * KV_DIM,
                   current_prompt_len[active_slot], KV_DIM);
      }

      for (int slot = 0; slot < active_slot_num; ++slot) {
        int active_slot = active_slots[slot];
        int seq_len = current_prompt_len[active_slot]; // + generated tokens?
        int logical_block_idx = seq_len / BLOCK_SIZE;
        int token_in_block_idx = seq_len % BLOCK_SIZE;
        int block = block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                                layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
        if (token_in_block_idx == 0) {
          int physical_block_idx = free_blocks.back();
          free_blocks.pop_back();
          block = physical_block_idx;
          block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                      layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
        }
        __nv_bfloat16 *k_cache_ptr =
            (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES +
                              token_in_block_idx * KV_DIM *
                                  sizeof(__nv_bfloat16));
        __nv_bfloat16 *k_proj_ptr = k_proj_batched_buffer + slot * KV_DIM;
        cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice);

        __nv_bfloat16 *v_cache_ptr =
            (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES +
                              V_OFFSET +
                              token_in_block_idx * KV_DIM *
                                  sizeof(__nv_bfloat16));
        __nv_bfloat16 *v_proj_ptr = v_proj_batched_buffer + slot * KV_DIM;
        cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice);
      }

      cudaMemcpy(block_table_gpu, block_table.data(),
                 MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                 cudaMemcpyHostToDevice);

      pagedAttention(layer, active_slot_num, q_proj, kv_cache, block_table_gpu,
                     seq_lens_gpu, active_slots_gpu, buf_2048_1);

      o_proj = buf_2048_2;

      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH,
                   active_slot_num, EMBEDDING_LENGTH, &o_proj_alpha,
                   weights.w_o[layer], CUDA_R_16BF, EMBEDDING_LENGTH,
                   buf_2048_1, CUDA_R_16BF, EMBEDDING_LENGTH, &o_proj_beta,
                   o_proj, CUDA_R_16BF, EMBEDDING_LENGTH, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT);

      residualAdd(hidden_state, o_proj, active_slot_num);

      rmsNorm(hidden_state, rms_norms, weights.post_attn_layernorms[layer],
              active_slot_num);

      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM,
                   active_slot_num, EMBEDDING_LENGTH, &gate_alpha,
                   weights.mlp_gate_proj[layer], CUDA_R_16BF, EMBEDDING_LENGTH,
                   rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &gate_beta, gate,
                   CUDA_R_16BF, HIDDEN_DIM, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT);

      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   HIDDEN_DIM,       // m
                   active_slot_num,  // n
                   EMBEDDING_LENGTH, // k
                   &up_alpha, weights.mlp_up_proj[layer], CUDA_R_16BF,
                   EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH,
                   &up_beta, up, CUDA_R_16BF, HIDDEN_DIM, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT);

      silu(gate, up, active_slot_num);

      down = buf_2048_2;
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, EMBEDDING_LENGTH,
                   active_slot_num, HIDDEN_DIM, &down_alpha,
                   weights.mlp_down_proj[layer], CUDA_R_16BF, HIDDEN_DIM, gate,
                   CUDA_R_16BF, HIDDEN_DIM, &down_beta, down, CUDA_R_16BF,
                   EMBEDDING_LENGTH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

      residualAdd(hidden_state, down, active_slot_num);
    }

    rmsNorm(hidden_state, rms_norms, weights.norm, active_slot_num);

    cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE,
                 active_slot_num, EMBEDDING_LENGTH, &embed_alpha,
                 weights.embed_tokens, CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms,
                 CUDA_R_16BF, EMBEDDING_LENGTH, &embed_beta, embed_proj,
                 CUDA_R_16BF, VOCAB_SIZE, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(embed_proj_cpu.data(), embed_proj,
               sizeof(__nv_bfloat16) * active_slot_num * VOCAB_SIZE,
               cudaMemcpyDeviceToHost);

    float max_token = 0.0;
    int max_token_idx = 0;
    for (int slot = 0; slot < active_slot_num; ++slot) {
      int active_slot = active_slots[slot];
      max_token = (float)embed_proj_cpu[slot * VOCAB_SIZE];

      max_token_idx = 0;
      for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx) {
        if ((float)embed_proj_cpu[slot * VOCAB_SIZE + token_idx] > max_token) {
          max_token = embed_proj_cpu[slot * VOCAB_SIZE + token_idx];
          max_token_idx = token_idx;
        }
      }

      std::cout << "Output token: " << (float)max_token
                << ", token index: " << std::to_string(max_token_idx)
                << std::endl;

      if (max_token_idx == END_OF_TEXT_TOKEN_ID ||
          max_token_idx == EOT_ID_TOKEN_ID ||
          current_prompt_len[active_slot] == MAX_SEQ_LEN - 1) {
        is_slot_free[active_slot] = true;
        for (int layer = 0; layer < N_LAYERS; ++layer) {
          for (int logical_block_idx = 0;
               logical_block_idx < MAX_BLOCKS_PER_SEQ; ++logical_block_idx) {
            int block_idx = active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ +
                            layer * MAX_BLOCKS_PER_SEQ + logical_block_idx;
            if (block_table[block_idx] != -1) {
              free_blocks.push_back(block_table[block_idx]);
              block_table[block_idx] = -1;
            }
          }
        }
        cudaMemcpy(block_table_gpu, block_table.data(),
                   MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int),
                   cudaMemcpyHostToDevice);
      } else {
        last_generated_tokens[active_slot] = max_token_idx;
        generated_tokens[active_slot].push_back(max_token_idx);
        current_prompt_len[active_slot] = current_prompt_len[active_slot] + 1;
      }
    }
  }

  std::cout << "\nOk bye!" << std::endl;
  cublasDestroy(cublas_handle);
  cudaDeviceSynchronize();
  return 0;
}
