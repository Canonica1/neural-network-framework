#pragma once

#include "nn/linalg.hpp"

#include <stdexcept>

namespace nn {
struct LossResult {
    float value;
    Matrix gradient;
};

class MSELoss {
  public:
    LossResult evaluate(const Matrix &prediction, const Matrix &expected) const {
        if (prediction.rows() != expected.rows() || prediction.cols() != expected.cols()) {
            throw std::invalid_argument("MSELoss: prediction and target sizes differ");
        }
        if (expected.cols() == 0) {
            throw std::invalid_argument("MSELoss: empty batch");
        }

        const Matrix diff = prediction - expected;
        const float batch = static_cast<float>(expected.cols());
        const float loss = diff.squaredNorm() / batch;
        return {loss, (2.0f / batch) * diff};
    }
};

using MSEloss = MSELoss;
} // namespace nn
