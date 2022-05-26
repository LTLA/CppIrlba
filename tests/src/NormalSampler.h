#ifndef NORMAL_SAMPLER_H
#define NORMAL_SAMPLER_H

#include "Eigen/Dense"
#include <random>

struct NormalSampler {
    NormalSampler(int seed) : engine(seed) {}
    double operator()() { return dist(engine); }
    std::mt19937_64 engine;
    std::normal_distribution<> dist;
};

inline Eigen::MatrixXd create_random_matrix(size_t nr, size_t nc, int seed = 42) {
    Eigen::MatrixXd A(nr, nc);
    NormalSampler norm(seed); 
    for (size_t i = 0; i < nc; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            A(j, i) = norm();
        }
    }
    return A;
}

inline Eigen::MatrixXcd create_random_complex_matrix(size_t nr, size_t nc, int seed = 42) {
    Eigen::MatrixXcd A(nr, nc);
    NormalSampler norm(seed); 
    for (size_t i = 0; i < nc; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            A(j, i) = std::complex<double>(norm(), norm());
        }
    }
    return A;
}

inline Eigen::VectorXd create_random_vector(size_t n, int seed = 50) {
    NormalSampler norm(seed);
    Eigen::VectorXd output(n);
    for (auto& o : output) {
        o = norm();
    }
    return output;
}

#endif
