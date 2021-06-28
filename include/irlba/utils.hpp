#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"
#include <random>

namespace irlba {

/**
 * @brief Orthogonalize a vector against a set of orthonormal column vectors. 
 */
class OrthogonalizeVector {
public:
    /**
     * @param max Expected maximum number of columns in `mat`, to be used to construct the workspace.
     */
    OrthogonalizeVector() {}

    /**
     * Set the maximum size of the temporary vector.
     *
     * @param max Maximum size of the temporary vector, equivalent to the maximum `ncols` in `run()`.
     *
     * @return A reference to the `OrthogonalizeVector` instance.
     */
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

/**
 * @brief Sample from a normal distribution.
 *
 * This performs pseudo-random sampling from the normal distribution, using the 64-bit Mersenne Twister from the C++ `<random>` standard library as the random number engine.
 * However, the former may not be consistently implemented across different platforms;
 * users may prefer to use the Boost `normal_distribution` implementation to ensure reproducibility across compilers.
 */
class NormalSampler {
public:
    /**
     * @param s Seed value.
     */
    NormalSampler(uint_fast64_t s) : generator(s) {}

    /**
     * @return A pseudo-random value from a normal distribution.
     */
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

/**
 * @brief Test for IRLBA convergence.
 */
struct ConvergenceTest {
private:
    double tol = 1e-5;
    double svtol = -1;
public:
    /**
     * Set the tolerance on the residuals.
     *
     * @param t Tolerance, should be a positive number.
     *
     * @return A reference to this `ConvergenceTest` instance.
     */
    ConvergenceTest& set_tol(double t = 1e-5) {
        tol = t;
        return *this;
    }

    /**
     * Set the tolerance on the relative differences between singular values across iterations.
     *
     * @param s Tolerance, should be a positive number.
     * Alternatively -1, in which case the value supplied to `set_tol()` is used.
     *
     * @return A reference to this `ConvergenceTest` instance.
     */
    ConvergenceTest& set_svtol(double s = -1) {
        svtol = s;
        return *this;
    }

    /**
     * Memorize this iteration's singular values, for comparison to the next iteration's values.
     *
     * @param l A vector of singular values.
     *
     * @return A reference to this `ConvergenceTest` instance.
     */
    ConvergenceTest& set_last(const Eigen::VectorXd& l) {
        last = l;
        return *this;
    }

public:
    /**
     * Calculate the number of singular values/vectors that have converged.
     *
     * @param sv Vector of singular values.
     * @param residuals Vector of residuals of some sort?
     *
     * @return The number of singular values/vectors that have achieved convergence.
     *
     * Convergence of each singular value/vector is defined by two conditions.
     * The first is that the relative change from the previous singular value is less than the value set in `set_svtol()`.
     * The second is that the residual is less than the value set in `set_tol()` (scaled by the largest singular value, which approximates the spectral norm of the input matrix).
     */
    int run(const Eigen::VectorXd& sv, const Eigen::VectorXd& residuals) {
        int counter = 0;
        double Smax = *std::max_element(sv.begin(), sv.end());
        double svtol_actual = (svtol >= 0 ? svtol : tol);

        for (int j = 0; j < sv.size(); ++j) {
            double ratio = std::abs(sv[j] - last[j]) / sv[j];
            if (std::abs(residuals[j]) < tol * Smax && ratio < svtol_actual) {
                ++counter;
            }
        }

        return counter;
    }
private:
    Eigen::VectorXd last;
};

}

#endif
