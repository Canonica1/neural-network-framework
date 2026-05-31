#pragma once

#include "nn/linalg.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace nn {
class AnyLayer {
  public:
    template <class Layer>
    AnyLayer(Layer layer) : self(std::make_unique<Model<std::decay_t<Layer>>>(std::move(layer))) {}

    AnyLayer(const AnyLayer &other) : self(other.self->clone()) {}
    AnyLayer(AnyLayer &&) noexcept = default;

    AnyLayer &operator=(const AnyLayer &other) {
        if (this != &other) {
            self = other.self->clone();
        }
        return *this;
    }

    AnyLayer &operator=(AnyLayer &&) noexcept = default;

    Matrix predict(const Matrix &x) const { return self->predict(x); }
    Matrix forward(const Matrix &x) { return self->forward(x); }
    Matrix backward(const Matrix &u) { return self->backward(u); }
    void update(float lr) { self->update(lr); }
    void update_momentum(float lr, float momentum) { self->update_momentum(lr, momentum); }
    void update_adamw(float lr, float beta1, float beta2, float eps, float weight_decay,
                      std::size_t step) {
        self->update_adamw(lr, beta1, beta2, eps, weight_decay, step);
    }
    void zero_grad() { self->zero_grad(); }

  private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::unique_ptr<Concept> clone() const = 0;
        virtual Matrix predict(const Matrix &x) const = 0;
        virtual Matrix forward(const Matrix &x) = 0;
        virtual Matrix backward(const Matrix &u) = 0;
        virtual void update(float lr) = 0;
        virtual void update_momentum(float lr, float momentum) = 0;
        virtual void update_adamw(float lr, float beta1, float beta2, float eps, float weight_decay,
                                  std::size_t step) = 0;
        virtual void zero_grad() = 0;
    };

    template <class Layer> struct Model final : Concept {
        explicit Model(Layer layer) : layer(std::move(layer)) {}

        std::unique_ptr<Concept> clone() const override {
            return std::make_unique<Model<Layer>>(layer);
        }

        Matrix predict(const Matrix &x) const override { return layer.predict(x); }
        Matrix forward(const Matrix &x) override { return layer.forward(x); }
        Matrix backward(const Matrix &u) override { return layer.backward(u); }
        void update(float lr) override { layer.update(lr); }
        void update_momentum(float lr, float momentum) override {
            layer.update_momentum(lr, momentum);
        }
        void update_adamw(float lr, float beta1, float beta2, float eps, float weight_decay,
                          std::size_t step) override {
            layer.update_adamw(lr, beta1, beta2, eps, weight_decay, step);
        }
        void zero_grad() override { layer.zero_grad(); }

        Layer layer;
    };

    std::unique_ptr<Concept> self;
};
} // namespace nn
