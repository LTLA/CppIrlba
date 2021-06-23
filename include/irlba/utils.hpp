#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"
#include <random>

namespace irlba {

/**
 * Orthogonalizes a vector against a set of orthonormal column vectors in a matrix.
 */
class OrthogonalizeVector {
public:
    /**
     * @param max Expected maximum number of columns in `mat`, to be used to construct the workspace.
     */
    OrthogonalizeVector() : {}

    OrthogonalizeVector& set_size(int max) {
        tmp.resize(max);
        return *this;
    }
public:
    /**
     * @param mat A matrix where the left-most `ncols` columns are orthonormal vectors.
     * @param vec The vector of interest, of length equal to the number of rows in `mat`.
     * @param ncols Number of left-most columns of `mat` to use.
     *
     * @return `vec` is modified to contain `vec - mat0 * t(mat0) * vec`, where `mat0` is defined as the first `ncols` columns of `mat`.
     * This ensures that it is orthogonal to each column of `mat0`.
     */
    void operator()(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec, size_t ncols) {
        tmp.head(ncols).noalias() = mat.leftCols(ncols).adjoint() * vec;
        vec.noalias() -= mat.leftCols(ncols) * tmp.head(ncols);
        return;
    }
private:
    Eigen::VectorXd tmp;
};

class NormalSampler {
public:
    NormalSampler(size_t s) : generator(s) {}
    double operator()() { 
        return distr(generator);
    }
private:
    std::mt19937_64 generator;

#ifdef IRLBA_RANDOM_REPRODUCIBLE
    // Probably should use Boost's method here, to ensure we 
    // get the same implementation across compilers.
#else 
    std::normal_distribution<double> distr;
#endif
};

}

#endif
