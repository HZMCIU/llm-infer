#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <queue>
#include <vector>

#include "const.h"

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

struct LayerContext {
  cublasHandle_t cublas_handle;

  __nv_bfloat16 *input;
  __nv_bfloat16 *hidden_state;

  __nv_bfloat16 *q_proj;
  __nv_bfloat16 *k_proj;
  __nv_bfloat16 *v_proj;

  float q_proj_alpha = 1.0f;
  float q_proj_beta = 0.0f;
  float k_proj_alpha = 1.0f;
  float k_proj_beta = 0.0f;
  float v_proj_alpha = 1.0f;
  float v_proj_beta = 0.0f;

  __nv_bfloat16 *attn_scores;
  __nv_bfloat16 *attn_scores_v;

  float attn_alpha = 1.0f / 8.0f;
  float attn_beta = 0.0f;
  float attn_scores_v_alpha = 1.0f;
  float attn_scores_v_beta = 0.0f;

  __nv_bfloat16 *o_proj;

  float o_proj_alpha = 1.0f;
  float o_proj_beta = 0.0f;

  __nv_bfloat16 *up;
  __nv_bfloat16 *gate;
  __nv_bfloat16 *down;

  float gate_alpha = 1.0f;
  float gate_beta = 0.0f;
  float up_alpha = 1.0f;
  float up_beta = 0.0f;
  float down_alpha = 1.0f;
  float down_beta = 0.0f;

  __nv_bfloat16 *embed_proj;

  float embed_alpha = 1.0f;
  float embed_beta = 0.0f;
};

struct TokenInfo {
  std::queue<std::vector<int>> queue;
  int *input_token_gpu;
  std::vector<bool> is_slot_free;
  int slot;
};

int loadWeight(Weights &weights);

void freeWeight(Weights &weights);

void MatMulProj(cublasHandle_t &cublas_handle, __nv_bfloat16 *y,
                __nv_bfloat16 *W, __nv_bfloat16 *x, int m, int n, int k,
                float alpha, float beta);

int Prefill_(Weights &weights, TokenInfo &token_info,
             LayerContext &layer_context);

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
             std::vector<int> &free_blocks, __nv_bfloat16 *kv_cache);
