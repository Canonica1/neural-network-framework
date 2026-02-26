#include <iostream>
#include <cmath>
#include <algorithm>
#include "nn/linear_layer.hpp"
#include "nn/relu_layer.hpp"

static float rel_err(float a, float b) {
    float denom = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) / denom;
}

int main() {
    std::cout << 123 << std::endl;
}