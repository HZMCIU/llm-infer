#pragma once

#include <string>
#include <vector>

class Tokenizer {
public:
  unsigned int max_token_length_;
  unsigned int bos_token_id_;
  unsigned int eos_token_id_;

  std::vector<std::string> vocab_;
  std::vector<float> merge_scores_;
  int vocab_size_;

  void BuildTokenizer(const std::string &dir_path);

  std::string &Decode(int token);

  void Encode(std::string &text, std::vector<int> &tokens);

  int StrLookup(std::string &str);
};
