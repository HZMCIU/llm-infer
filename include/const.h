#pragma once
constexpr int MAX_NEW_TOKENS_GENERATED =
    20; // TODO: parameterize it with program arguments
constexpr int B_TO_MB = 1024 * 1024;
constexpr int B_TO_GB = 1024 * 1024 * 1024;
constexpr int N_LAYERS =
    16; // TODO: hardcoded for llama 3.2 1B, just like any other value for now
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int KV_DIM = 512;
constexpr int HEAD_DIM = 64;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_K_HEADS = 8;
constexpr int NUM_V_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4;
constexpr int VOCAB_SIZE = 128256;
constexpr int END_OF_TEXT_TOKEN_ID = 128001; // <|end_of_text|>
constexpr int EOT_ID_TOKEN_ID = 128009;      // <|eot_id|>
constexpr int MAX_SEQ_LEN = 2048;            // TODO: make it tunable
constexpr int BATCH_SIZE =
    2; // TODO: not even close to being good, it's just here to have batching
constexpr int MAX_PROMPT_LEN = 512; // TODO: arbitrary, tunable
constexpr int MAX_BUFFER_SIZE = std::max(MAX_PROMPT_LEN, BATCH_SIZE);
constexpr int BLOCK_SIZE =
    16; // TODO: tunable as well, defined the size of a single page in pagedattn
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2; // * 2 because K and V
constexpr size_t KV_CACHE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024; // TODO: 2GB
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE; // 2048 / 16 = 128
constexpr int NUM_BLOCKS =
    KV_CACHE_SIZE_BYTES / BLOCK_BYTES; // 2*1024*1024*1024/(16*512*2*2) = 65536
constexpr int MAX_SEQUENCES = BATCH_SIZE;

constexpr float SQRT_HEAD_DIM = 8;

#define WARP_FULL_MASK 0xffffffffffffffffULL
