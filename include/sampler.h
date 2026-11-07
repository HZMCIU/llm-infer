#pragma once
#include <vector>

struct ProbIndex {
  float prob;
  int index;
};

class Sampler {
public:
  float temperature_;
  float top_p_;
  int top_k_;
  unsigned long long rng_state_;
  std::vector<ProbIndex> prob_index_;

  unsigned int RandomU32(unsigned long long &state);
  float RandomF32(unsigned long long &state);
  void QuickSelect(std::vector<ProbIndex> &arr, int n, int k);
  int SampleArgmax(std::vector<float> &logits);
  void BuildSampler(float temperature, float top_p, int top_k,
                    unsigned long long rng_seed);
  int Sample(std::vector<float> &logits);
};
