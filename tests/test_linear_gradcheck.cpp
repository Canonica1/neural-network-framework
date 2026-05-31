#include "nn/adamw.hpp"
#include "nn/except.hpp"
#include "nn/gradient_descent.hpp"
#include "nn/linear_layer.hpp"
#include "nn/momentum.hpp"
#include "nn/mse_loss.hpp"
#include "nn/network.hpp"
#include "nn/relu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
[[maybe_unused]] float rel_err(float a, float b) {
    float denom = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) / denom;
}

void require(bool ok, const std::string &message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float eps, const std::string &message) {
    require(rel_err(actual, expected) <= eps, message);
}

void require_matrix_near(const nn::Matrix &actual, const nn::Matrix &expected, float eps,
                         const std::string &message) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
            message + ": shape mismatch");
    for (nn::Index row = 0; row < actual.rows(); ++row) {
        for (nn::Index col = 0; col < actual.cols(); ++col) {
            require_near(actual(row, col), expected(row, col), eps, message);
        }
    }
}

template <class F> void require_throw(F &&f, const std::string &message) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(message);
}

void test_mse_loss() {
    nn::Matrix prediction(2, 2);
    prediction << 1.0f, 2.0f, 3.0f, 4.0f;

    nn::Matrix target(2, 2);
    target << 0.0f, 2.0f, 1.0f, 6.0f;

    const nn::MSELoss loss;
    const nn::LossResult result = loss.evaluate(prediction, target);

    nn::Matrix expected_gradient(2, 2);
    expected_gradient << 1.0f, 0.0f, 2.0f, -2.0f;

    require_near(result.value, 4.5f, 1e-6f, "mse loss value");
    require_matrix_near(result.gradient, expected_gradient, 1e-6f, "mse loss gradient");

    nn::Matrix bad_target(1, 2);
    require_throw([&] { loss.evaluate(prediction, bad_target); }, "mse must reject bad shape");

    nn::Matrix empty(2, 0);
    require_throw([&] { loss.evaluate(empty, empty); }, "mse must reject empty batch");
}

void test_relu_layer() {
    nn::ReluLayer relu;

    nn::Matrix x(2, 3);
    x << -1.0f, 0.0f, 2.0f, 3.0f, -4.0f, 5.0f;

    nn::Matrix expected_forward(2, 3);
    expected_forward << 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 5.0f;

    require_matrix_near(relu.forward(x), expected_forward, 1e-6f, "relu forward");

    nn::Matrix upstream(2, 3);
    upstream << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

    nn::Matrix expected_backward(2, 3);
    expected_backward << 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 6.0f;

    require_matrix_near(relu.backward(upstream), expected_backward, 1e-6f, "relu backward");
}

float loss_for(nn::LinearLayer &layer, const nn::Matrix &x, const nn::Matrix &target,
               const nn::Matrix &weights, const nn::Vector &biases) {
    layer.set_parameters(weights, biases);
    return nn::MSELoss{}.evaluate(layer.predict(x), target).value;
}

void test_linear_gradcheck() {
    nn::LinearLayer layer(2, 2);

    nn::Matrix x(2, 3);
    x << 1.0f, -2.0f, 0.5f, 0.0f, 1.5f, -1.0f;

    nn::Matrix target(2, 3);
    target << 0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f;

    nn::Matrix weights(2, 2);
    weights << 0.2f, -0.4f, 0.1f, 0.3f;

    nn::Vector biases(2);
    biases << 0.05f, -0.2f;

    nn::Matrix bad_weights(1, 2);
    require_throw([&] { layer.set_parameters(bad_weights, biases); },
                  "linear layer must reject bad parameter shape");

    layer.set_parameters(weights, biases);
    const nn::LossResult result = nn::MSELoss{}.evaluate(layer.forward(x), target);
    layer.backward(result.gradient);

    const nn::Matrix analytic_weight_grad = layer.weight_grad();
    const nn::Vector analytic_bias_grad = layer.bias_grad();

    constexpr float eps = 1e-3f;
    for (nn::Index row = 0; row < weights.rows(); ++row) {
        for (nn::Index col = 0; col < weights.cols(); ++col) {
            nn::Matrix plus = weights;
            nn::Matrix minus = weights;
            plus(row, col) += eps;
            minus(row, col) -= eps;

            const float numerical = (loss_for(layer, x, target, plus, biases) -
                                     loss_for(layer, x, target, minus, biases)) /
                                    (2.0f * eps);

            require_near(analytic_weight_grad(row, col), numerical, 2e-2f,
                         "linear weight gradient");
        }
    }

    for (nn::Index row = 0; row < biases.rows(); ++row) {
        nn::Vector plus = biases;
        nn::Vector minus = biases;
        plus(row) += eps;
        minus(row) -= eps;

        const float numerical = (loss_for(layer, x, target, weights, plus) -
                                 loss_for(layer, x, target, weights, minus)) /
                                (2.0f * eps);

        require_near(analytic_bias_grad(row), numerical, 2e-2f, "linear bias gradient");
    }
}

void test_gradient_descent_optimizer() {
    struct Model {
        void update(float value) {
            lr = value;
            ++steps;
        }

        float lr = 0.0f;
        int steps = 0;
    };

    Model model;
    const nn::GradientDescent optimizer(0.25f);
    optimizer.step(model);

    require_near(model.lr, 0.25f, 1e-6f, "optimizer learning rate");
    require(model.steps == 1, "optimizer must call update once");
}

void test_momentum_optimizer() {
    struct Model {
        void update_momentum(float new_lr, float new_momentum) {
            lr = new_lr;
            momentum = new_momentum;
            ++steps;
        }

        float lr = 0.0f;
        float momentum = 0.0f;
        int steps = 0;
    };

    Model model;
    const nn::Momentum optimizer(0.1f, 0.9f);
    optimizer.step(model);

    require_near(model.lr, 0.1f, 1e-6f, "momentum learning rate");
    require_near(model.momentum, 0.9f, 1e-6f, "momentum coefficient");
    require(model.steps == 1, "momentum optimizer must call update_momentum once");
}

void test_adamw_optimizer() {
    struct Model {
        void update_adamw(float new_lr, float new_beta1, float new_beta2, float new_eps,
                          float new_weight_decay, std::size_t new_step) {
            lr = new_lr;
            beta1 = new_beta1;
            beta2 = new_beta2;
            eps = new_eps;
            weight_decay = new_weight_decay;
            step = new_step;
        }

        float lr = 0.0f;
        float beta1 = 0.0f;
        float beta2 = 0.0f;
        float eps = 0.0f;
        float weight_decay = 0.0f;
        std::size_t step = 0;
    };

    Model model;
    nn::AdamW optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    optimizer.step(model);
    optimizer.step(model);

    require_near(model.lr, 0.001f, 1e-6f, "adamw learning rate");
    require_near(model.beta1, 0.9f, 1e-6f, "adamw beta1");
    require_near(model.beta2, 0.999f, 1e-6f, "adamw beta2");
    require_near(model.eps, 1e-8f, 1e-6f, "adamw epsilon");
    require_near(model.weight_decay, 0.01f, 1e-6f, "adamw weight decay");
    require(model.step == 2, "adamw must pass growing step index");
    require(optimizer.step_count() == 2, "adamw must store step count");
}

void test_linear_momentum_update() {
    nn::LinearLayer layer(1, 1);

    nn::Matrix weights(1, 1);
    weights << 1.0f;

    nn::Vector biases(1);
    biases << 0.0f;

    nn::Matrix x(1, 1);
    x << 1.0f;

    nn::Matrix target(1, 1);
    target << 0.0f;

    layer.set_parameters(weights, biases);

    nn::LossResult first = nn::MSELoss{}.evaluate(layer.forward(x), target);
    layer.backward(first.gradient);
    layer.update_momentum(0.1f, 0.9f);

    nn::LossResult second = nn::MSELoss{}.evaluate(layer.forward(x), target);
    layer.backward(second.gradient);
    layer.update_momentum(0.1f, 0.9f);

    const float prediction = layer.predict(x)(0, 0);
    require_near(prediction, 0.0f, 1e-5f, "linear momentum must use previous velocity");
}

void test_linear_adamw_update() {
    nn::LinearLayer layer(1, 1);

    nn::Matrix weights(1, 1);
    weights << 1.0f;

    nn::Vector biases(1);
    biases << 0.0f;

    nn::Matrix x(1, 1);
    x << 1.0f;

    nn::Matrix target(1, 1);
    target << 0.0f;

    layer.set_parameters(weights, biases);

    nn::LossResult result = nn::MSELoss{}.evaluate(layer.forward(x), target);
    layer.backward(result.gradient);
    layer.update_adamw(0.1f, 0.0f, 0.0f, 1e-8f, 0.01f, 1);

    const float prediction = layer.predict(x)(0, 0);
    require_near(prediction, 0.799f, 1e-5f, "linear adamw update");
}

void test_network_training() {
    nn::Matrix x(2, 4);
    x << 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f;

    nn::Matrix y(1, 4);
    y << 0.0f, 1.0f, 1.0f, 2.0f;

    nn::Network net;
    net.add(nn::LinearLayer(2, 1));

    const nn::MSELoss loss;
    nn::AdamW optimizer(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    const float before = loss.evaluate(net.predict(x), y).value;
    for (int i = 0; i < 100; ++i) {
        net.train_batch(x, y, loss, optimizer);
    }
    const float after = loss.evaluate(net.predict(x), y).value;

    require(std::isfinite(after), "training loss must be finite");
    require(after < before, "training must reduce loss");
}
} // namespace

namespace nn {
void run_all_tests() {
    test_mse_loss();
    test_relu_layer();
    test_linear_gradcheck();
    test_gradient_descent_optimizer();
    test_momentum_optimizer();
    test_adamw_optimizer();
    test_linear_momentum_update();
    test_linear_adamw_update();
    test_network_training();
    std::cout << "all tests passed" << std::endl;
}
} // namespace nn

int main() {
    try {
        nn::run_all_tests();
    } catch (...) {
        return nn::react();
    }
    return 0;
}
