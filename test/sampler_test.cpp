#include "const.h"
#include "sampler.h"
#include <gtest/gtest.h>
#include <vector>

TEST(SamplerTest, Sample) {
  Sampler sampler;
  sampler.BuildSampler(0.8, 0.92, 30, 1234);
  std::vector<float> logits(VOCAB_SIZE);
  for (int i = 0; i < VOCAB_SIZE; ++i) {
    logits[i] = 10;
  }
  for (int i = 0; i < 10; ++i) {
    printf("%d\n", sampler.Sample(logits));
  }
}
