#include <cuda_bf16.h>

void embeddingGather(int *input_token, __nv_bfloat16 *model_token_embedding,
                     __nv_bfloat16 *input_token_embedding, int input_token_num);

void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output,
             __nv_bfloat16 *norm_weight, int num_token);

void init_rope_frequency(int head_dim, int max_token_seq_num, float rope_theta,
                         float factor, float low_freq_factor,
                         float high_freq_factor, int max_hidden_dimension);

void free_rope_frequency();

void rope(__nv_bfloat16 *input, int proj_dim, int token_num);

void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embed,
                 int token_num);

void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int token_num);
