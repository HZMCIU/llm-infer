#include "tokenizer.h"

#include <gtest/gtest.h>
#include <vector>
using namespace std;

TEST(TokenizerTest, TokenizerEncode) {
  Tokenizer tokenizer;
  string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  string text = R"(<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

Today Date: 07 Nov 2026

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

The key of life is<|eot_id|><|start_header_id|>assistant<|end_header_id|>)";
  tokenizer.BuildTokenizer(dir_path);
  vector<int> tokens;
  tokenizer.Encode(text, tokens);

  printf("[");
  for (int &c : tokens) {
    printf("%d, ", c);
  }
  printf("]\n");

  // 128000,   3923,    374,    220,     17,     10,     17,     30, 128009,
  // 128006,  78191, 128007
}

TEST(TokenizerTest, TokenizerDecode) {
  Tokenizer tokenizer;
  string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  tokenizer.BuildTokenizer(dir_path);

  vector<int> tokens = {
      128000, 128006, 9125,   128007, 271,    38766, 1303,  33025, 2696,  25,
      6790,   220,    508,    1419,   271,    15724, 2696,  25,    220,   2589,
      4723,   220,    508,    1627,   271,    2675,  527,   264,   11190, 18328,
      13,     128009, 128006, 882,    128007, 271,   791,   1401,  315,   2324,
      374,    128009, 128006, 78191,  128007, 271,   32,    3488,  430,   706,
      1027,   59674,  323,    36131,  555,    61787, 11,    18330, 6164,  11,
      323,    69122,  369,    24552,  2268,   8142,  1070,  374,   912,   832,
      45813,  4320,   11,     1618,   527,    1063,  315,   279,   1455,  5526,
      323,    28254,  58689,  315,    279,    1401,  315,   2324,  1473,  16,
      13,     3146,   1,      791,    5422,   315,   10919, 1,     96618, 1115,
      17571,  374,    3629,   5938,   449,    279,   4623,  430,   279,   1401,
      311,    23871,  11,     9096,   11,     323,   57383, 15812, 304,   99139,
      3021,   323,    30481,  369,    57669,  323,   3885,  627,   17,    13,
      3146,   1,      791,    5422,   315,    23754, 31514, 1,     96618, 1115,
      7434,   66159,  279,    12939,  315};
  for (int token : tokens) {
    printf("%s", tokenizer.Decode(token).c_str());
  }
  printf("\n");
}

TEST(TokenizerTest, TokenizerRenderPrompt) {
  Tokenizer tokenizer;
  string dir_path = "/home/yuqyda/Data/cuda/models/Llama-3.2-1B-Instruct";
  tokenizer.BuildTokenizer(dir_path);

  string input = "the key to life is";
  string result = tokenizer.RenderPrompt(input);

  cout << result << endl;
}
