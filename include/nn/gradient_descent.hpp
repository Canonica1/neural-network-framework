#pragma once

namespace nn {
class GradientDescent {
  public:
    explicit GradientDescent(float lr) : learning_rate(lr) {}

    template <class Model> void step(Model &model) const { model.update(learning_rate); }

    float lr() const { return learning_rate; }

  private:
    float learning_rate;
};
} // namespace nn
