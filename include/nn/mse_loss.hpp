#pragma once
#include <Eigen/Dense>

namespace nn {
    using Matrix = Eigen::MatrixXf;
    
    class MSEloss {
    public:
        Matrix predict(const Matrix& x) const {
            return x;
        }

        Matrix forward(const Matrix& x) {
            assert(y.size() != 0 && "MSEloss: target not set");
            diff = x - y;

            const float batch = static_cast<float>(y.cols());
            const float loss = diff.squaredNorm() / batch;

            Matrix out(1, 1);
            out(0, 0) = loss;
            return out;
        }

        Matrix backward(const Matrix& u) {
            const float batch = static_cast<float>(y.cols());
            return u(0, 0) * (2.0f / batch) * diff;
        }

        void update(float) {}
        void zero_grad() { diff.resize(0,0); }

        void set_target(const Matrix& y_true) { y = y_true; }
    private:
        Matrix y;
        Matrix diff;
    };
}