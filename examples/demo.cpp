#include "nn/any_layer.hpp"
#include "nn/linear_layer.hpp"
#include "nn/relu_layer.hpp"
#include "nn/mse_loss.hpp"
#include "nn/network.hpp"
#include "nn/mnist.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static int argmax_col(const Eigen::VectorXf& v) {
    Eigen::Index idx;
    v.maxCoeff(&idx);
    return (int)idx;
}

static float accuracy(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& y_onehot) {
    int correct = 0;
    const int N = (int)logits.cols();
    for (int i = 0; i < N; ++i) {
        int p = argmax_col(logits.col(i));
        int t = argmax_col(y_onehot.col(i));
        correct += (p == t);
    }
    return float(correct) / float(N);
}

int main() {
    using nn::Matrix;

    const std::string dir = "data/mnist/";

    auto train = load_mnist_train_csv(dir + "mnist_train.csv");
    
    Matrix X = train.X;
    Matrix Y = train.Y;
    X.array() /= 255.0f;
    std::cout << "Loaded: X " << X.rows() << "x" << X.cols()
              << ", Y " << Y.rows() << "x" << Y.cols() << "\n";

    const int N = (int)X.cols();
    const int N_val = std::min(10000, N / 6);
    const int N_tr = N - N_val;

    Matrix Xtr = X.leftCols(N_tr);
    Matrix Ytr = Y.leftCols(N_tr);
    Matrix Xva = X.rightCols(N_val);
    Matrix Yva = Y.rightCols(N_val);

    nn::Network net;
    net.add(nn::AnyLayer{nn::LinearLayer(784, 128)});
    net.add(nn::AnyLayer{nn::ReluLayer{}});
    net.add(nn::AnyLayer{nn::LinearLayer(128, 10)});
    net.add(nn::AnyLayer{nn::MSEloss{}});

    const int epochs = 10;
    const int batch_size = 128;
    const float lr = 0.05f;

    std::vector<int> idx(N_tr);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    Matrix xb, yb;

    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng);

        float sum_loss = 0.0f;
        int batches = 0;

        for (int start = 0; start < N_tr; start += batch_size) {
            const int bs = std::min(batch_size, N_tr - start);

            xb.resize(784, bs);
            yb.resize(10, bs);
            for (int j = 0; j < bs; ++j) {
                const int col = idx[start + j];
                xb.col(j) = Xtr.col(col);
                yb.col(j) = Ytr.col(col);
            }

            net.set_target(yb);
            Matrix L = net.forward(xb);
            sum_loss += L(0, 0);
            ++batches;

            net.backward(Matrix::Ones(1, 1));
            net.update(lr);
            net.zero_grad();
        }

        Matrix logits_va = net.predict(Xva);
        float acc = accuracy(logits_va, Yva);

        std::cout << "epoch " << ep
                  << " avg_loss=" << (sum_loss / std::max(1, batches))
                  << " val_acc=" << acc << "\n";
    }

    return 0;
}