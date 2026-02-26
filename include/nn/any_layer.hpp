#pragma once
#include <folly/Poly.h>
#include <Eigen/Dense>

namespace nn {
using Matrix = Eigen::MatrixXf;

struct ILayer {
  template <class Base>
  struct Interface : Base {
    Matrix predict(const Matrix& x) const { return folly::poly_call<0>(*this, x); }
    Matrix forward(const Matrix& x)       { return folly::poly_call<1>(*this, x); }
    Matrix backward(const Matrix& u)      { return folly::poly_call<2>(*this, u); }
    void update(float lr)                 { folly::poly_call<3>(*this, lr); }
    void zero_grad()                      { folly::poly_call<4>(*this); }
    void set_target(const Matrix& y_true) { folly::poly_call<5>(*this, y_true); }

  };

  template <class T>
  using Members = folly::PolyMembers<
      &T::predict,
      &T::forward,
      &T::backward,
      &T::update,
      &T::zero_grad,
      &T::set_target>;
};
using AnyLayer = folly::Poly<ILayer>;
}