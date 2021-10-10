#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"
#include <random>
#include "aarand/aarand.hpp"

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
 * Fill an **Eigen** vector with random normals via **aarand**.
 *
 * @param Vec Any **Eigen** vector class or equivalent proxy object.
 * @param Engine A (pseudo-)random number generator class that returns a random number when called with no arguments.
 *
 * @param vec Instance of a `Vec` class.
 * @param eng Instance of an `Engine` class.
 *
 * @return `vec` is filled with random draws from a standard normal distribution.
 */
template<class Vec, class Engine>
void fill_with_random_normals(Vec& vec, Engine& eng) {
    Eigen::Index i = 0;
    while (i < vec.size() - 1) {
        auto paired = aarand::standard_normal(eng);
        vec[i] = paired.first;
        vec[i + 1] = paired.second;
        i += 2;
    }

    if (i != vec.size()) {
        auto paired = aarand::standard_normal(eng);
        vec[i] = paired.first;
    }
    return;
}

/**
 * @cond
 */
template<class M>
struct ColumnVectorProxy {
    ColumnVectorProxy(M& m, int c) : mat(m), col(c) {}
    auto size () { return mat.rows(); }
    auto& operator[](int r) { return mat(r, col); }
    M& mat;
    int col;
};
/**
 * @endcond
 */
 
/** 
 * Fill a column of an **Eigen** matrix with random normals via **aarand**.
 *
 * @param Matrix Any **Eigen** matrix class or equivalent proxy object.
 * @param Engine A (pseudo-)random number generator class that returns a random number when called with no arguments.
 *
 * @param mat Instance of a `Matrix` class.
 * @param eng Instance of an `Engine` class.
 *
 * @return The `column` column of `mat` is filled with random draws from a standard normal distribution.
 */
template<class Matrix, class Engine>
void fill_with_random_normals(Matrix& mat, int column, Engine& eng) {
    ColumnVectorProxy proxy(mat, column);
    fill_with_random_normals(proxy, eng);
    return;
}

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
     * @param residuals Vector of residuals for each singular value/vector.
     *
     * @return The number of singular values/vectors that have achieved convergence.
     */
    int run(const Eigen::VectorXd& sv, const Eigen::VectorXd& residuals) {
        int counter = 0;
        double Smax = *std::max_element(sv.begin(), sv.end());
        double svtol_actual = (svtol >= 0 ? svtol : tol);

        for (int j = 0; j < sv.size(); ++j) {
            double ratio = std::abs(sv[j] - last[j]) / sv[j];
            if (std::abs(residuals[j]) < tol * Smax && // see the RHS of Equation 2.13 in Baglama and Reichel.
                    ratio < svtol_actual) {
                ++counter;
            }
        }

        return counter;
    }
private:
    Eigen::VectorXd last;
};

/**
 * Utility function to generate a default `NULL` argument for the random number generator input.
 *
 * @return A null pointer to an RNG.
 * Any RNG will do here, so we use the most common one.
 */
constexpr std::mt19937_64* null_rng() { return NULL; }

}

#endif
