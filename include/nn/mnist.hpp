#include <Eigen/Dense>
#include <csv.hpp>
#include <vector>
#include <cstdint>

using Matrix = Eigen::MatrixXf;

struct MnistCSV { Matrix X; Matrix Y; };

MnistCSV load_mnist_train_csv(const std::string& path, int limit = -1) {
    csv::CSVFormat fmt;
    fmt.delimiter(',').header_row(0);

    csv::CSVReader reader(path, fmt);

    std::vector<float> xbuf;
    std::vector<uint8_t> labels;

    int n = 0;
    for (auto& row : reader) {
        int lab = row[0].get<int>();
        labels.push_back((uint8_t)lab);
        for (int j = 1; j <= 784; ++j) {
            xbuf.push_back(row[j].get<int>() / 255.0f);
        }
        if (limit > 0 && ++n >= limit) break;
    }

    const int N = (int)labels.size();
    MnistCSV out;
    out.X = Eigen::Map<const Matrix>(xbuf.data(), 784, N);
    out.Y = Matrix::Zero(10, N);
    for (int i = 0; i < N; ++i) out.Y((int)labels[i], i) = 1.0f;
    return out;
}