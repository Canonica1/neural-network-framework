#pragma once
#include <Eigen/Dense>
#include "nn/any_layer.hpp"

namespace nn {
    using Matrix = Eigen::MatrixXf;
    
    class Network {
    public:
        void add(AnyLayer layer) {
            blocks.emplace_back(std::move(layer));
        }
        Matrix predict(const Matrix& x) const {
            Matrix cur = x;
            for (const auto& b : blocks) {
                cur = b.predict(cur);
            }
            return cur;
        }

        Matrix forward(const Matrix& x) {
            Matrix X = x;
            for (int i = 0; i < blocks.size(); i++) {
                X = blocks[i].forward(X);
            }
            return X;
        }

        Matrix backward(const Matrix& u) {
            Matrix X = u;
            for (std::size_t i = blocks.size(); i-- > 0; ) {
                X = blocks[i].backward(X);
            }
            return X;
        }

        void update(float lr) {
            for (int i = 0; i < blocks.size(); i++) {
                blocks[i].update(lr);
            }
        }
        void zero_grad() { 
            for (int i = 0; i < blocks.size(); i++) {
                blocks[i].zero_grad();
            }
        }
        
        void set_target(const Matrix& y_true) {
            for (auto& b : blocks) b.set_target(y_true);
    }
    private:
        std::vector<AnyLayer> blocks;
    };
}