#pragma once

#include <string>
#include <vector>

class Tokenizer {
public:
  unsigned int max_token_length_;
  unsigned int bos_token_id_;
  std::vector<unsigned int> eos_token_ids_;

  std::vector<std::string> vocab_;
  std::vector<float> merge_scores_;
  int vocab_size_;

  std::string user_prompt_ =
      R"(<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

Today Date: 07 Nov 2026

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>


)";

  void BuildTokenizer(const std::string &dir_path);

  std::string &Decode(int token);

  void Encode(std::string &text, std::vector<int> &tokens);

  int StrLookup(std::string &str);

  std::string RenderPrompt(std::string &text);

  bool CheckEosToken(unsigned int token);
};
