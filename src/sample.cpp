#include <algorithm>
#include <cmath>
#include <vector>

#include "const.h"
#include "sampler.h"

unsigned int Sampler::RandomU32(unsigned long long &state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return (state * 0x2545F4914F6CDD1Dull) >> 32;
}

float Sampler::RandomF32(unsigned long long &state) {
  return (RandomU32(state) >> 8) / 16777216.0f;
}

void Sampler::QuickSelect(std::vector<ProbIndex> &arr, int n, int k) {
  int l = 0, r = n - 1;
  while (l < r) {
    ProbIndex pivot = arr[k];
    int i = l, j = r;
    do {
      while (arr[i].prob > pivot.prob) {
        i++;
      }
      while (arr[j].prob < pivot.prob) {
        j--;
      }
      if (i <= j) {
        std::swap(arr[i], arr[j]);
        i++;
        j--;
      }
    } while (i <= j);

    if (j < k) {
      l = i;
    }
    if (i > k) {
      r = j;
    }
  }
}

int Sampler::SampleArgmax(std::vector<float> &logits) {
  int max_i = 0;
  float max_p = logits[0];
  for (int i = 0; i < VOCAB_SIZE; ++i) {
    if (logits[i] > max_p) {
      max_i = i;
      max_p = logits[i];
    }
  }
  return max_i;
}

void Sampler::BuildSampler(float temperature, float top_p, int top_k,
                           unsigned long long rng_seed) {
  this->temperature_ = temperature;
  this->top_p_ = top_p;
  this->top_k_ = (top_k == 0) ? VOCAB_SIZE : top_k;
  this->rng_state_ = rng_seed;
  this->prob_index_.resize(VOCAB_SIZE);
}

int Sampler::Sample(std::vector<float> &logits) {
  if (this->temperature_ == 0.0f || this->top_k_ == 1) {
    return SampleArgmax(logits);
  }

  for (int i = 0; i < VOCAB_SIZE; ++i) {
    this->prob_index_[i].prob = logits[i];
    this->prob_index_[i].index = i;
  }

  int n_cands = (this->top_k_ < VOCAB_SIZE) ? this->top_k_ : VOCAB_SIZE;
  if (n_cands < VOCAB_SIZE) {
    QuickSelect(this->prob_index_, VOCAB_SIZE, n_cands);
  }
  float max_logit = this->prob_index_[0].prob;
  for (int i = 0; i < n_cands; ++i) {
    if (this->prob_index_[i].prob > max_logit) {
      max_logit = this->prob_index_[i].prob;
    }
  }

  float prob_sum = 0.0f;
  for (int i = 0; i < n_cands; ++i) {
    float prob = std::expf((prob_index_[i].prob - max_logit) / temperature_);
    this->prob_index_[i].prob = prob;
    prob_sum += prob;
  }

  for (int i = 0; i < n_cands; ++i) {
    prob_index_[i].prob /= prob_sum;
  }

  if (this->top_p_ > 0.0f && this->top_p_ < 1.0f) {
    std::sort(this->prob_index_.begin(), this->prob_index_.begin() + n_cands,
              [](const ProbIndex &a, const ProbIndex &b) -> bool {
                return a.prob > b.prob;
              });
    float cumulative_prob = 0.0f;
    int last_idx = n_cands - 1;
    for (int i = 0; i < n_cands; ++i) {
      cumulative_prob += this->prob_index_[i].prob;
      if (cumulative_prob > this->top_p_) {
        last_idx = i;
        break;
      }
    }

    n_cands = last_idx + 1;
    prob_sum = cumulative_prob;
  }

  float coin = RandomF32(rng_state_) * prob_sum;
  float cdf = 0.0f;
  for (int i = 0; i < n_cands; ++i) {
    cdf += prob_index_[i].prob;
    if (coin < cdf) {
      return prob_index_[i].index;
    }
  }
  return prob_index_[n_cands - 1].index;
}
