#pragma once

namespace nn {
class Momentum {
  public:
    explicit Momentum(float lr, float coefficient = 0.9f)
        : learning_rate(lr), momentum_coefficient(coefficient) {}

    template <class Model> void step(Model &model) const {
        model.update_momentum(learning_rate, momentum_coefficient);
    }

    float lr() const { return learning_rate; }
    float coefficient() const { return momentum_coefficient; }

  private:
    float learning_rate;
    float momentum_coefficient;
};
} // namespace nn
