#pragma once

#include "nn/any_layer.hpp"
#include "nn/gradient_descent.hpp"
#include "nn/linalg.hpp"
#include "nn/mse_loss.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace nn {
class Network {
  public:
    template <class Layer> void add(Layer layer) { blocks.emplace_back(std::move(layer)); }

    Matrix predict(const Matrix &x) const {
        Matrix cur = x;
        for (const auto &layer : blocks) {
            cur = layer.predict(cur);
        }
        return cur;
    }

    float train_batch(const Matrix &x, const Matrix &y, const MSELoss &loss, float lr) {
        GradientDescent optimizer(lr);
        return train_batch(x, y, loss, optimizer);
    }

    template <class Optimizer>
    float train_batch(const Matrix &x, const Matrix &y, const MSELoss &loss, Optimizer &optimizer) {
        const Matrix prediction = forward(x);
        const LossResult result = loss.evaluate(prediction, y);
        backward(result.gradient);
        optimizer.step(*this);
        return result.value;
    }

    Matrix forward(const Matrix &x) {
        Matrix result = x;
        for (auto &layer : blocks) {
            result = layer.forward(result);
        }
        return result;
    }

    Matrix backward(const Matrix &u) {
        Matrix result = u;
        for (std::size_t i = blocks.size(); i-- > 0;) {
            result = blocks[i].backward(result);
        }
        return result;
    }

    void update(float lr) {
        for (auto &layer : blocks) {
            layer.update(lr);
            layer.zero_grad();
        }
    }

    void update_momentum(float lr, float momentum) {
        for (auto &layer : blocks) {
            layer.update_momentum(lr, momentum);
            layer.zero_grad();
        }
    }

    void update_adamw(float lr, float beta1, float beta2, float eps, float weight_decay,
                      std::size_t step) {
        for (auto &layer : blocks) {
            layer.update_adamw(lr, beta1, beta2, eps, weight_decay, step);
            layer.zero_grad();
        }
    }

  private:
    std::vector<AnyLayer> blocks;
};
} // namespace nn
