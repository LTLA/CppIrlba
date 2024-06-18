#ifndef IRLBA_OPTIONS_HPP
#define IRLBA_OPTIONS_HPP

/**
 * @file Options.hpp
 * @brief Options for IRLBA.
 */

namespace irlba {

/**
 * @brief Options for the IRLBA algorithm.
 */
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
};

}

#endif

