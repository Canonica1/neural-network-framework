#pragma once

#include "nn/linalg.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {
struct MnistCSV {
    Matrix X;
    Matrix Y;
};

MnistCSV load_mnist_train_csv(const std::filesystem::path &path, Index limit = -1) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open MNIST csv file: " + path.string());
    }

    std::vector<float> xbuf;
    std::vector<uint8_t> labels;

    std::string line;
    Index loaded = 0;
    while (std::getline(input, line)) {
        std::stringstream row(line);
        std::string cell;

        if (!std::getline(row, cell, ',')) {
            throw std::runtime_error("invalid MNIST csv row: missing label");
        }

        const int lab = std::stoi(cell);
        labels.push_back(static_cast<uint8_t>(lab));
        for (Index pixel = 0; pixel < 784; ++pixel) {
            if (!std::getline(row, cell, ',')) {
                throw std::runtime_error("invalid MNIST csv row: missing pixel");
            }
            xbuf.push_back(static_cast<float>(std::stoi(cell)) / 255.0f);
        }

        ++loaded;
        if (limit > 0 && loaded >= limit) {
            break;
        }
    }

    const Index sample_count = static_cast<Index>(labels.size());
    MnistCSV out;
    out.X = map_matrix(xbuf, 784, sample_count);
    out.Y = Matrix::Zero(10, sample_count);
    for (Index i = 0; i < sample_count; ++i) {
        out.Y(static_cast<Index>(labels[static_cast<std::size_t>(i)]), i) = 1.0f;
    }
    return out;
}
} // namespace nn
