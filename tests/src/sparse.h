#ifndef SPARSE_H
#define SPARSE_H

#include <cstddef>
#include <vector>
#include <random>

#include "Eigen/Sparse"
#include "Eigen/Dense"

struct CompressedSparseData {
    std::size_t rows, cols;
    std::vector<double> values;
    std::vector<int> indices;
    std::vector<std::size_t> nzeros;
};

inline CompressedSparseData simulate_compressed_sparse(std::size_t nr, std::size_t nc) {
    std::mt19937_64 rng(nr * nc);
    std::uniform_real_distribution udist(0.0, 1.0);
    std::normal_distribution ndist;

    CompressedSparseData output;
    output.rows = nr;
    output.cols = nc;
    output.nzeros.resize(nc + 1);

    for (std::size_t c = 0; c < nc; ++c) {
        auto& count = output.nzeros[c + 1];
        for (std::size_t r = 0; r < nr; ++r) {
            if (udist(rng) < 0.2) {
                output.values.push_back(ndist(rng));
                output.indices.push_back(r);
                ++count;
            }
        }
    }

    for (std::size_t c = 0; c < nc; ++c) {
        output.nzeros[c + 1] += output.nzeros[c];
    }

    return output;
}

inline Eigen::SparseMatrix<double> create_sparse_matrix(const CompressedSparseData& data) {
    std::vector<Eigen::Triplet<double> > coefficients;
    for (std::size_t c = 0; c < data.cols; ++c) {
        const auto start = data.nzeros[c], end = data.nzeros[c + 1];
        for (std::size_t i = start; i < end; ++i) {
            coefficients.emplace_back(data.indices[i], c, data.values[i]);
        }
    }

    Eigen::SparseMatrix<double> output(data.rows, data.cols);
    output.setFromTriplets(coefficients.begin(), coefficients.end());
    return output;
}

inline Eigen::MatrixXd create_dense_matrix(const CompressedSparseData& data) {
    Eigen::MatrixXd output(data.rows, data.cols);
    output.setZero();

    for (std::size_t c = 0; c < data.cols; ++c) {
        const auto start = data.nzeros[c], end = data.nzeros[c + 1];
        for (std::size_t i = start; i < end; ++i) {
            output.coeffRef(data.indices[i], c) = data.values[i];
        }
    }

    return output;
}

#endif
