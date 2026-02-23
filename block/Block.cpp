#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <folly/Poly.h>

using Matrix = Eigen::MatrixXf;

struct ILayer {
    template <class Base>
    struct Interface : Base {
        Matrix predict(const Matrix& x) const { return folly::poly_call<0>(*this, x); }
        Matrix forward(const Matrix& x) { return folly::poly_call<1>(*this, x); }
        Matrix backward(const Matrix& grad) { return folly::poly_call<2>(*this, grad); }
        void update(float lr) { folly::poly_call<3>(*this, lr); }
    };

    template <class T>
    using Members = folly::PolyMembers<

    
        &T::predict,
        &T::forward,
        &T::backward,
        &T::update>;
};

using Layer = folly::Poly<ILayer>;

class NeuralNetwork {
    public:
        void addLayer(Layer block) {
            blocks.push_back(std::move(block));
        }
        Matrix predict(const Matrix &x) const {
            Matrix result = x;
            for (const auto& b: blocks) {
                result = b.predict(result);
            }
            return result;
        }
   private:
        std::vector<Layer> blocks;
};

class LinearLayer {
public:
    Matrix A;
    Matrix B;
    Matrix last_input;
    LinearLayer(int in, int out)
        : A(out, in),
          B(out, 1) {
        A.setZero();
        B.setZero();
    }
    Matrix predict(const Matrix& prev) const {
        return (A * prev) + B;
    }
    Matrix forward(const Matrix& prev) {
        last_input = prev;
        return (A * prev) + B;
    }
    Matrix backward(const Matrix& grad) {
        return grad;
    }
    void update(float lr) {
        (void)lr;
    }
};
