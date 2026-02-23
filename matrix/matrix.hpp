#include <cstddef>
#include <stdexcept>
#include <vector>

class matrix {
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;

    std::size_t index(std::size_t i, std::size_t j) const {
        return i * cols_ + j;
    }

public:
    matrix() = default;
    matrix(std::size_t rows, std::size_t cols, float value = 0.0f)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    matrix(const matrix&) = default;
    matrix(matrix&&) noexcept = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&) noexcept = default;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    float& operator()(std::size_t i, std::size_t j) {
        return data_[index(i, j)];
    }
    const float& operator()(std::size_t i, std::size_t j) const {
        return data_[index(i, j)];
    }

    matrix add(const matrix& in) const {
        if (in.rows() != rows_ || in.cols() != cols_) {
            throw std::runtime_error("add: bad dimensions");
        }
        matrix answer(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                answer(i, j) = (*this)(i, j) + in(i, j);
            }
        }
        return answer;
    }

    matrix mul(const matrix& in) const {
        if (in.cols() != rows_) {
            throw std::runtime_error("mul: bad dimensions");
        }
        matrix answer(in.rows(), cols_);
        for (std::size_t i = 0; i < in.rows(); ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                float sum = 0.0f;
                for (std::size_t r = 0; r < rows_; ++r) {
                    sum += in(i, r) * (*this)(r, j);
                }
                answer(i, j) = sum;
            }
        }
        return answer;
    }
};
