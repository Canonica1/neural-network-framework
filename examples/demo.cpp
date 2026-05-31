#include "nn/adamw.hpp"
#include "nn/except.hpp"
#include "nn/gradient_descent.hpp"
#include "nn/linear_layer.hpp"
#include "nn/metrics.hpp"
#include "nn/mnist.hpp"
#include "nn/momentum.hpp"
#include "nn/mse_loss.hpp"
#include "nn/network.hpp"
#include "nn/relu_layer.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {
using nn::Index;
using nn::Matrix;

constexpr Index epochs = 10;
constexpr Index batch_size = 128;
constexpr float lr = 0.001f;
constexpr float weight_decay = 0.01f;

struct Batch {
    Matrix inputs;
    Matrix targets;
};

std::pair<Matrix, Matrix> split_to_train_and_validate(const Matrix &data, Index train_size) {
    return {data.leftCols(train_size), data.rightCols(data.cols() - train_size)};
}

void print_dataset_info(const Matrix &X, const Matrix &Y) {
    std::cout << "Loaded: X " << X.rows() << "x" << X.cols() << ", Y " << Y.rows() << "x"
              << Y.cols() << '\n';
}

nn::Network make_network() {
    nn::Network net;
    net.add(nn::LinearLayer(784, 128));
    net.add(nn::ReluLayer{});
    net.add(nn::LinearLayer(128, 10));
    return net;
}

Batch make_batch(const Matrix &X, const Matrix &Y, const std::vector<Index> &idx, Index start,
                 Index size) {
    Matrix xb(X.rows(), size);
    Matrix yb(Y.rows(), size);

    for (Index j = 0; j < size; ++j) {
        const Index col = idx[static_cast<std::size_t>(start + j)];
        xb.col(j) = X.col(col);
        yb.col(j) = Y.col(col);
    }

    return {std::move(xb), std::move(yb)};
}

void print_epoch_stats(Index epoch, float average_loss, float validation_accuracy) {
    std::cout << "epoch " << epoch << " avg_loss=" << average_loss
              << " val_acc=" << validation_accuracy << '\n';
}

int run_training() {
    const std::filesystem::path dir = std::filesystem::path{"data"} / "mnist";
    auto train = nn::load_mnist_train_csv(dir / "mnist_train.csv");

    Matrix X = train.X;
    Matrix Y = train.Y;
    print_dataset_info(X, Y);

    const Index N = X.cols();
    const Index N_val = std::min(Index{10000}, N / 6);
    const Index N_tr = N - N_val;

    auto [Xtr, Xva] = split_to_train_and_validate(X, N_tr);
    auto [Ytr, Yva] = split_to_train_and_validate(Y, N_tr);

    nn::Network net = make_network();
    const nn::MSELoss loss;
    nn::AdamW optimizer(lr, 0.9f, 0.999f, 1e-8f, weight_decay);

    std::vector<Index> idx(static_cast<std::size_t>(N_tr));
    std::iota(idx.begin(), idx.end(), Index{0});
    std::mt19937 rng(42);

    for (Index ep = 0; ep < epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng);

        float sum_loss = 0.0f;
        Index batches = 0;

        for (Index start = 0; start < N_tr; start += batch_size) {
            const Index bs = std::min(batch_size, N_tr - start);
            const Batch batch = make_batch(Xtr, Ytr, idx, start, bs);
            sum_loss += net.train_batch(batch.inputs, batch.targets, loss, optimizer);
            ++batches;
        }

        const Matrix logits_va = net.predict(Xva);
        print_epoch_stats(ep, sum_loss / static_cast<float>(std::max(Index{1}, batches)),
                          nn::accuracy(logits_va, Yva));
    }

    return 0;
}
} // namespace

int main() {
    try {
        return run_training();
    } catch (...) {
        return nn::react();
    }
}
