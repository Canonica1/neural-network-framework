#pragma once
#include <Eigen/Dense>
#include <cassert>

namespace nn {
using Matrix = Eigen::MatrixXf;

class SoftmaxCrossEntropyLoss {
public:
    Matrix predict(const Matrix& x) const { return x; }

    void set_target(const Matrix& y_true) { y = y_true; }

    Matrix forward(const Matrix& x) {
        assert(y.size() != 0 && "SoftmaxCrossEntropyLoss: target not set");
        assert(x.rows() == y.rows() && x.cols() == y.cols() && "SoftmaxCrossEntropyLoss: shape mismatch");

        const int C = static_cast<int>(x.rows());
        const int B = static_cast<int>(x.cols());
        (void)C;

        // stable softmax: z = x - max(x) per column
        Matrix z = x;
        z.colwise() -= x.colwise().maxCoeff();

        Matrix expz = z.array().exp().matrix();
        Eigen::RowVectorXf denom = expz.colwise().sum();
        probs = expz.array().rowwise() / denom.array();
        const float batch = static_cast<float>(B);
        const float eps = 1e-12f;
        const float loss = -(y.array() * (probs.array() + eps).log()).sum() / batch;

        grad = (probs - y) / batch;

        Matrix out(1, 1);
        out(0, 0) = loss;
        return out;
    }

    Matrix backward(const Matrix& u) {
        assert(u.rows() == 1 && u.cols() == 1 && "SoftmaxCrossEntropyLoss: backward expects 1x1 upstream");
        return u(0, 0) * grad;
    }

    void update(float) {}
    void zero_grad() {
        probs.resize(0, 0);
        grad.resize(0, 0);
    }

private:
    Matrix y;
    Matrix probs;
    Matrix grad;
};

}