#pragma once
#include <Eigen/Dense>
#include <tuple>

namespace nn {
    using Matrix = Eigen::MatrixXf;
    using Vector = Eigen::VectorXf;
    class LinearLayer {
    public:
        LinearLayer(int in_dim, int out_dim)
            : A(Matrix::Random(out_dim, in_dim)),
          b(Vector::Zero(out_dim)),
          dA(Matrix::Zero(out_dim, in_dim)),
          db(Vector::Zero(out_dim)) {}

        Matrix predict(const Matrix& x) const {
            Matrix after_A = A * x;
            after_A.colwise() += b;
            return after_A;
        }
        Matrix forward(const Matrix& x) {
            x_cache = x;
            Matrix after_A = A * x;
            after_A.colwise() += b;
            return after_A;
        }
        Matrix backward(Matrix u) {
            Matrix grad_by_x = A.transpose() * u;
            dA = u * x_cache.transpose();
            db = u.rowwise().sum();
            return grad_by_x;
        }
        void update(float lr) {
            A -= lr * dA;
            b -= lr * db;
        }
        void zero_grad() {
            dA.setZero();
            db.setZero();
        }
        void set_target(const Matrix& y_true) {

        }
    private:
        Matrix A;
        Vector b;

        Matrix x_cache;

        Matrix dA;
        Vector db;
        
    };
}
