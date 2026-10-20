#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int MAX_PROMPT_LEN = 512;

constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_LENGTH = 2048;

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

int loadWeight(Weights &weights);
