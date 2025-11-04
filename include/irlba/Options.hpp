#ifndef IRLBA_OPTIONS_HPP
#define IRLBA_OPTIONS_HPP

#include <cstdint>
#include <optional>

/**
 * @file Options.hpp
 * @brief Options for IRLBA.
 */

namespace irlba {

/**
 * @brief Options for running IRLBA in `compute()` and `pca()`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class to store the initial values.
 */
template<class EigenVector_ = Eigen::VectorXd>
struct Options {
    /**
     * Set the tolerance to use to define invariant subspaces.
     * This is used as the lower bound for the L2 norm for the subspace vectors;
     * below this bound, vectors are treated as all-zero and are instead filled with random draws from a normal distribution.
     * If negative, we instead use the machine epsilon to the power of 0.8.
     */
    double invariant_subspace_tolerance = -1;

    /**
     * Tolerance on the residuals of the singular triplets.
     * Lower values improve the accuracy of the decomposition.
     * (See Equation 2.13 of Baglama and Reichel.)
     */
    double convergence_tolerance = 1e-5; 

    /**
     * Tolerance on the relative differences between singular values across iterations.
     * Lower values improve the accuracy of the decomposition.
     * If -1, the value in `Options::convergence_tolerance` is used.
     */
    double singular_value_ratio_tolerance = -1; 

    /**
     * Number of extra dimensions to define the working subspace.
     * Larger values can speed up convergence at the cost of increased memory usage.
     */
    int extra_work = 7;

    /**
     * Maximum number of restart iterations.
     * Larger values improve the chance of convergence.
     */
    int max_iterations = 1000;

    /**
     * Whether to perform an exact SVD if the matrix is too small (fewer than 6 elements in any dimension).
     * This is more efficient and avoids inaccuracies from an insufficient workspace.
     */
    bool exact_for_small_matrix = true;

    /**
     * Whether to perform an exact SVD if the requested number of singular values is too large (greater than or equal to half the smaller matrix dimension).
     * This is more efficient and avoids inaccuracies from an insufficient workspace.
     */
    bool exact_for_large_number = true;

    /**
     * Whether to automatically cap the requested number of singular values to the smaller dimension of the input matrix.
     * If false, an error is thrown instead.
     */
    bool cap_number = false;

    /**
     * Seed for the creation of random vectors, primarily during initialization of the IRLBA algorithm.
     */
    typename std::mt19937_64::result_type seed = std::mt19937_64::default_seed;

    /**
     * Pointer to the initial values of the first right singular vector.
     * This should have length equal to the number of columns of the input `matrix` in `compute()` or `pca()`.
     */
    std::optional<EigenVector_> initial;
};

}

#endif

