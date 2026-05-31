#pragma once

#include <exception>
#include <iostream>

namespace nn {
inline int react() {
    try {
        throw;
    } catch (const std::exception &error) {
        std::cerr << "Unhandled exception: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "Unhandled unknown exception\n";
    }
    return 1;
}
} // namespace nn
