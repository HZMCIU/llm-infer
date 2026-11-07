#include "tokenizer.h"

#include <gtest/gtest.h>
#include <vector>
using namespace std;

TEST(TokenizerTest, TokenizerEncode) {
  Tokenizer tokenizer;
  string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  string text =
      "What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
  tokenizer.BuildTokenizer(dir_path);
  vector<int> tokens;
  tokenizer.Encode(text, tokens);

  for (int &c : tokens) {
    printf("%d\n", c);
  }

  // 128000,   3923,    374,    220,     17,     10,     17,     30, 128009,
  // 128006,  78191, 128007
}

TEST(TokenizerTest, TokenizerDecode) {
  Tokenizer tokenizer;
  string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  tokenizer.BuildTokenizer(dir_path);
  printf("%s\n", tokenizer.Decode(654).c_str());
}
