#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "const.h"

void Tokenizer::BuildTokenizer(const std::string &dir_path) {
  std::string file_path = dir_path + "/tokenizer.bin";
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    std::cerr << "tokenizer.bin not exist" << std::endl;
    return;
  }

  this->vocab_size_ = VOCAB_SIZE;
  file.read((char *)&this->max_token_length_, sizeof(unsigned int));
  file.read((char *)&this->bos_token_id_, sizeof(unsigned int));

  int eos_token_ids_len;
  file.read((char *)&eos_token_ids_len, sizeof(int));
  this->eos_token_ids_.resize(eos_token_ids_len);
  for (int i = 0; i < eos_token_ids_len; ++i) {
    file.read((char *)&this->eos_token_ids_[i], sizeof(unsigned int));
  }

  this->vocab_.resize(this->vocab_size_);
  this->merge_scores_.resize(this->vocab_size_);

  int len;
  for (int i = 0; i < this->vocab_size_; ++i) {
    file.read((char *)&this->merge_scores_[i], sizeof(float));
    if (file.eof() || file.fail() || file.bad()) {
      this->vocab_[i] = "";
    } else {
      file.read((char *)&len, sizeof(int));
      this->vocab_[i].resize(len + 1);
      file.read(this->vocab_[i].data(), len);
    }
  }
}

std::string &Tokenizer::Decode(int token) { return this->vocab_[token]; }

int Tokenizer::StrLookup(std::string &str) {
  for (int i = 0; i < this->vocab_size_; ++i) {
    if (strcmp(str.c_str(), this->vocab_[i].c_str()) == 0) {
      return i;
    }
  }
  return -1;
}

void Tokenizer::Encode(std::string &text, std::vector<int> &tokens) {
  std::string str_buffer(this->vocab_size_ * 2 + 1 + 2, '\0');
  std::string special_token(128 * 2 + 1, '\0');

  int text_len = text.length();
  for (int i = 0; i < text_len; ++i) {
    int id, found_special_token = 0;

    str_buffer[0] = text[i];
    str_buffer[1] = 0;

    if (text[i] == '<') {
      int end_of_token_pos = -1;
      found_special_token = 0;
      for (int k = 0; i + k < text_len && k < 64; ++k) {
        if (text[i + k] == '>') {
          end_of_token_pos = k;
          break;
        }
      }

      if (end_of_token_pos != -1) {
        special_token = text.substr(i, end_of_token_pos + 1);

        id = StrLookup(special_token);

        if (id != -1) {
          i += end_of_token_pos;
          found_special_token = 1;
        }
      }
    }

    if (!found_special_token) {
      id = StrLookup(str_buffer);
    }

    if (id != -1) {
      tokens.push_back(id);
    } else {
      printf("Warning: unknown character code point %d in input, skipping.\n",
             str_buffer.c_str());
    }
  }

  while (true) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < tokens.size() - 1; ++i) {
      snprintf(str_buffer.data(), str_buffer.length(), "%s%s",
               this->vocab_[tokens[i]].c_str(),
               this->vocab_[tokens[i + 1]].c_str());
      int id = StrLookup(str_buffer);
      if (id != -1 && this->merge_scores_[id] > best_score) {
        best_score = this->merge_scores_[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break;
    }

    tokens[best_idx] = best_id;
    tokens.erase(tokens.begin() + best_idx + 1);
  }
}
