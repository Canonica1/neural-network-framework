#pragma once
#include <Eigen/Dense>

namespace nn {
    using Matrix = Eigen::MatrixXf;
    
    class ReluLayer {
    public:
        ReluLayer() = default;
        Matrix predict(const Matrix& x) const {
            return x.cwiseMax(0.0f);
        }
        Matrix forward(const Matrix& x) {
            mask = (x.array() > 0.0f).cast<float>(); 
            return x.cwiseMax(0.0f);
        }
        Matrix backward(Matrix u) {
            return u.cwiseProduct(mask);       
        }
        
        void update(float lr) {
        }
        void zero_grad() {}
        void set_target(const Matrix& y_true) {

        }
    private:
        Matrix mask;        
    };
}