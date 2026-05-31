#pragma once

#include "nn/linalg.hpp"

#include <cstddef>

namespace nn {
class ReluLayer {
  public:
    ReluLayer() = default;

    Matrix predict(const Matrix &x) const { return x.cwiseMax(0.0f); }

    Matrix forward(const Matrix &x) {
        mask = (x.array() > 0.0f).cast<float>();
        return x.cwiseMax(0.0f);
    }

    Matrix backward(const Matrix &upstream_gradient) {
        return upstream_gradient.cwiseProduct(mask);
    }

    void update(float) {}
    void update_momentum(float, float) {}
    void update_adamw(float, float, float, float, float, std::size_t) {}
    void zero_grad() {}

  private:
    Matrix mask;
};
} // namespace nn
