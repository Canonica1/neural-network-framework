#pragma once

#include <cstddef>

namespace nn {
class AdamW {
  public:
    AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.01f)
        : learning_rate(lr), first_moment_decay(beta1), second_moment_decay(beta2), epsilon(eps),
          decay(weight_decay) {}

    template <class Model> void step(Model &model) {
        ++iteration;
        model.update_adamw(learning_rate, first_moment_decay, second_moment_decay, epsilon, decay,
                           iteration);
    }

    float lr() const { return learning_rate; }
    float beta1() const { return first_moment_decay; }
    float beta2() const { return second_moment_decay; }
    float eps() const { return epsilon; }
    float weight_decay() const { return decay; }
    std::size_t step_count() const { return iteration; }

  private:
    float learning_rate;
    float first_moment_decay;
    float second_moment_decay;
    float epsilon;
    float decay;
    std::size_t iteration = 0;
};
} // namespace nn
