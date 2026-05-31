#pragma once

#include "nn/linalg.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace nn {
class LinearLayer {
  public:
    LinearLayer(Index in_dim, Index out_dim)
        : weights(Matrix::Random(out_dim, in_dim) * std::sqrt(2.0f / static_cast<float>(in_dim))),
          biases(Vector::Zero(out_dim)), weight_gradient(Matrix::Zero(out_dim, in_dim)),
          bias_gradient(Vector::Zero(out_dim)), weight_velocity(Matrix::Zero(out_dim, in_dim)),
          bias_velocity(Vector::Zero(out_dim)), weight_first_moment(Matrix::Zero(out_dim, in_dim)),
          bias_first_moment(Vector::Zero(out_dim)),
          weight_second_moment(Matrix::Zero(out_dim, in_dim)),
          bias_second_moment(Vector::Zero(out_dim)) {}

    Matrix predict(const Matrix &x) const {
        Matrix output = weights * x;
        output.colwise() += biases;
        return output;
    }

    Matrix forward(const Matrix &x) {
        input_cache = x;
        return predict(x);
    }

    Matrix backward(const Matrix &upstream_gradient) {
        Matrix input_gradient = weights.transpose() * upstream_gradient;
        weight_gradient = upstream_gradient * input_cache.transpose();
        bias_gradient = upstream_gradient.rowwise().sum();
        return input_gradient;
    }

    void set_parameters(Matrix new_weights, Vector new_biases) {
        if (new_weights.rows() != weights.rows() || new_weights.cols() != weights.cols() ||
            new_biases.rows() != biases.rows()) {
            throw std::invalid_argument("LinearLayer: parameter sizes differ");
        }
        weights = std::move(new_weights);
        biases = std::move(new_biases);
        weight_gradient = Matrix::Zero(weights.rows(), weights.cols());
        bias_gradient = Vector::Zero(biases.rows());
        weight_velocity = Matrix::Zero(weights.rows(), weights.cols());
        bias_velocity = Vector::Zero(biases.rows());
        weight_first_moment = Matrix::Zero(weights.rows(), weights.cols());
        bias_first_moment = Vector::Zero(biases.rows());
        weight_second_moment = Matrix::Zero(weights.rows(), weights.cols());
        bias_second_moment = Vector::Zero(biases.rows());
    }

    const Matrix &weight_grad() const { return weight_gradient; }
    const Vector &bias_grad() const { return bias_gradient; }

    void update(float lr) {
        weights -= lr * weight_gradient;
        biases -= lr * bias_gradient;
    }

    void update_momentum(float lr, float momentum) {
        weight_velocity = momentum * weight_velocity + weight_gradient;
        bias_velocity = momentum * bias_velocity + bias_gradient;
        weights -= lr * weight_velocity;
        biases -= lr * bias_velocity;
    }

    void update_adamw(float lr, float beta1, float beta2, float eps, float weight_decay,
                      std::size_t step) {
        weight_first_moment = beta1 * weight_first_moment + (1.0f - beta1) * weight_gradient;
        bias_first_moment = beta1 * bias_first_moment + (1.0f - beta1) * bias_gradient;

        weight_second_moment = beta2 * weight_second_moment +
                               (1.0f - beta2) * weight_gradient.array().square().matrix();
        bias_second_moment =
            beta2 * bias_second_moment + (1.0f - beta2) * bias_gradient.array().square().matrix();

        const float first_correction = 1.0f - std::pow(beta1, static_cast<float>(step));
        const float second_correction = 1.0f - std::pow(beta2, static_cast<float>(step));

        const Matrix weight_m_hat = weight_first_moment / first_correction;
        const Vector bias_m_hat = bias_first_moment / first_correction;
        const Matrix weight_v_hat = weight_second_moment / second_correction;
        const Vector bias_v_hat = bias_second_moment / second_correction;

        weights -= lr * ((weight_m_hat.array() / (weight_v_hat.array().sqrt() + eps)).matrix() +
                         weight_decay * weights);
        biases -= lr * ((bias_m_hat.array() / (bias_v_hat.array().sqrt() + eps)).matrix() +
                        weight_decay * biases);
    }

    void zero_grad() {
        weight_gradient.setZero();
        bias_gradient.setZero();
    }

  private:
    Matrix weights;
    Vector biases;
    Matrix input_cache;
    Matrix weight_gradient;
    Vector bias_gradient;
    Matrix weight_velocity;
    Vector bias_velocity;
    Matrix weight_first_moment;
    Vector bias_first_moment;
    Matrix weight_second_moment;
    Vector bias_second_moment;
};
} // namespace nn
