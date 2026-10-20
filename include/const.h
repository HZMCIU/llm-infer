#pragma once
constexpr int MAX_PROMPT_LEN = 512;
constexpr int NUM_Q_HEADS = 32;

constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int KV_DIM = 512;

constexpr int BLOCK_SIZE = 16;
constexpr int V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
constexpr int BLOCK_BYTES = V_OFFSET * 2;
constexpr int MAX_SEQ_LEN = 2048;
constexpr int MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN / BLOCK_SIZE;

constexpr int GQA_Q_TO_K_RATIO = 4;
constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4;
constexpr int HEAD_DIM = 64;

constexpr int HIDDEN_DIM = 8192;

constexpr int VOCAB_SIZE = 128256;

constexpr int BATCH_SIZE = 2;
constexpr int MAX_SEQUENCES = BATCH_SIZE;
