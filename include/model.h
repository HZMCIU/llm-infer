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

  std::vector<std::vector<int>> generated_tokens;
};

int LoadWeight(Weights &weights);

void FreeWeight(Weights &weights);

int Prefill(Weights &weights, TokenInfo &token_info,
            LayerContext &layer_context);

int Decode(Weights &weights, TokenInfo &token_info,
           LayerContext &layer_context);
