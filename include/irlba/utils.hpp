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
    OrthogonalizeVector() {}

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
    void run(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec, size_t ncols) {
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

struct ConvergenceTest {
private:
    double tol;
    double svtol;
public:
    ConvergenceTest& set_tol(double t) {
        tol = t;
        return *this;
    }

    ConvergenceTest& set_svtol(double s) {
        svtol = s;
        return *this;
    }

    template<class V>
    ConvergenceTest& set_last(V&& l) {
        last = l;
        return *this;
    }

public:
    bool run(int desired, const Eigen::VectorXd& sv, const Eigen::VectorXd& residuals) {
        int counter = 0;
        double Smax = *std::max_element(sv.begin(), sv.end());

        for (int j = 0; j < sv.size(); ++j) {
            double ratio = std::abs(sv[j] - last[j]) / sv[j];
            if (std::abs(residuals[j]) < tol * Smax && ratio < svtol) {
                ++counter;
            } else {
                break;
            }
        }

        return counter >= desired;
    }
private:
    Eigen::VectorXd last;
};

}

#endif
