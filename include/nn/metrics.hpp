#pragma once

#include "nn/linalg.hpp"

#include <stdexcept>

namespace nn {
template <class VectorExpression> Index argmax(const VectorExpression &values) {
    Index index = 0;
    values.maxCoeff(&index);
    return index;
}

inline float accuracy(const Matrix &logits, const Matrix &expected) {
    if (logits.cols() != expected.cols()) {
        throw std::invalid_argument("accuracy: logits and expected labels have different sizes");
    }
    if (logits.cols() == 0) {
        return 0.0f;
    }

    Index correct = 0;
    for (Index col = 0; col < logits.cols(); ++col) {
        correct += argmax(logits.col(col)) == argmax(expected.col(col));
    }

    return static_cast<float>(correct) / static_cast<float>(logits.cols());
}
} // namespace nn
