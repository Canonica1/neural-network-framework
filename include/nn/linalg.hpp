#pragma once

#include <Eigen/Dense>
#include <vector>

namespace nn {
using Index = Eigen::Index;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

inline Matrix map_matrix(const std::vector<float> &values, Index rows, Index cols) {
    return Eigen::Map<const Matrix>(values.data(), rows, cols);
}
} // namespace nn
