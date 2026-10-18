#include <cuda_bf16.h>

void embeddingGather(int *input_token, __nv_bfloat16 *model_token_embedding,
                     __nv_bfloat16 *input_token_embedding, int input_token_num);
void rmsNorm(__nv_bfloat16 *input, __nv_bfloat16 *output,
             __nv_bfloat16 *norm_weight, int num_token);
