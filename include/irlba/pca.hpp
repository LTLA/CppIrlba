#ifndef IRLBA_PCA_HPP
#define IRLBA_PCA_HPP

#include <cmath>
#include <utility>
#include <stdexcept>

#include "compute.hpp"
#include "Matrix/simple.hpp"
#include "Matrix/centered.hpp"
#include "Matrix/scaled.hpp"

#include "Eigen/Dense"

/**
 * @file pca.hpp
 * @brief Perform PCA with IRLBA.
 */

namespace irlba {

/**
 * Perform a principal components analysis (PCA) using IRLBA for the underlying SVD.
 * This applies deferred centering and scaling via the `CenteredMatrix` and `ScaledMatrix` classes, respectively.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class for the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `OutputEigenMatrix_`.
 *
 * @param[in] matrix Input matrix where rows are observations and columns are features.
 * @param center Should the matrix be centered by column?
 * This should be set to `true` if the matrix is not already centered.
 * @param scale Should the matrix be scaled to unit variance for each column?
 * @param number Number of principal components to obtain.
 * @param[out] scores Output matrix for the principal component scores.
 * Each row corresponds to an observation while each column corresponds to a principal component.
 * On output, the number of rows is equal to the number of columns in `matrix`,
 * while the number of columns is equal to `number` (or less, if `Options::cap_number` is applied).
 * @param[out] rotation Output matrix in which to store the rotation matrix.
 * Each row corresponds to an feature while each column corresponds to a principal component.
 * On output, the number of rows is equal to the number of rows in `matrix`,
 * while the number of columns is equal to `number` (or less, if `Options::cap_number` is applied).
 * @param[out] variances Vector to store the variance explained by each principal component.
 * On output, the length of this vector is equal to `number` (or less, if `Options::cap_number` is applied).
 * @param options Further options.
 *
 * @return A pair where the first entry indicates whether the algorithm converged,
 * and the second entry indicates the number of restart iterations performed.
 *
 * Centering is performed by subtracting each element of `center` from the corresponding column of `mat`.
 * Scaling is performed by dividing each column of `mat` by the corresponding element of `scale` (after any centering has been applied).
 * Note that `scale=true` requires `center=true` to guarantee unit variance along each column. 
 * No scaling is performed when the variance of a column is zero, so as to avoid divide-by-zero errors. 
 */
template<class InputEigenMatrix_, class OutputEigenMatrix_, class EigenVector_>
std::pair<bool, int> pca(
    const InputEigenMatrix_& matrix,
    bool center,
    bool scale,
    Eigen::Index number,
    OutputEigenMatrix_& scores,
    OutputEigenMatrix_& rotation,
    EigenVector_& variances,
    const Options<EigenVector_>& options
) {
    const Eigen::Index nr = matrix.rows();
    const Eigen::Index nc = matrix.cols();

    EigenVector_ center0;
    if (center) {
        if (nr < 1) {
            throw std::runtime_error("cannot center with no observations");    
        }
        center0.resize(nc);
    }

    EigenVector_ scale0;
    if (scale) {
        if (nr < 2) {
            throw std::runtime_error("cannot scale with fewer than two observations");    
        }
        scale0.resize(nc);
    }

    if (center || scale) {
        EigenVector_ buffer;
        for (Eigen::Index i = 0; i < nc; ++i) {
            buffer = matrix.col(i);

            typename EigenVector_::Scalar mean = 0;
            if (center) {
                mean = buffer.sum() / nr;
                center0[i] = mean;
            }

            if (scale) {
                typename EigenVector_::Scalar var = 0;
                for (const auto x : buffer) {
                    var += (x - mean) * (x - mean);
                }

                if (var) {
                    scale0[i] = std::sqrt(var/(nr - 1));
                } else {
                    scale0[i] = 1;
                }
            }
        }
    }

    // Reduce the size of the binary by forcing all compute() calls to use virtual dispatch,
    // rather than realizing four new instances of the same function with different Matrix subclasses.
    std::unique_ptr<Matrix<EigenVector_, OutputEigenMatrix_> > ptr;
    ptr.reset(new SimpleMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(&matrix)> >(&matrix));

    if (center) {
        std::unique_ptr<Matrix<EigenVector_, OutputEigenMatrix_> > alt;
        alt.reset(new CenteredMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(ptr)>, I<decltype(&center0)> >(std::move(ptr), &center0));
        ptr.swap(alt);
    }

    if (scale) {
        std::unique_ptr<Matrix<EigenVector_, OutputEigenMatrix_> > alt;
        alt.reset(new ScaledMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(ptr)>, I<decltype(&scale0)> >(std::move(ptr), &scale0, true, true));
        ptr.swap(alt);
    }

    const auto stats = compute(*ptr, number, scores, rotation, variances, options);

    scores.array().rowwise() *= variances.adjoint().array();
    if (nc > 1) {
        const auto denom = nr - 1;
        for (auto& v : variances) {
            v = v * v / denom;
        }
    }

    return stats;
}

/**
 * @brief Results of the IRLBA-based PCA by `pca()`.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 */
template<class EigenMatrix_, class EigenVector_>
struct PcaResults {
    /**
     * Matrix of principal component scores. 
     * Each row corresponds to an observation while each column corresponds to a principal component.
     * The number of rows is equal to the number of columns in `matrix`,
     * while the number of columns is equal to `number` (or less, if `Options::cap_number` is applied).
     */
    EigenMatrix_ scores;

    /**
     * Rotation matrix.
     * Each row corresponds to an feature while each column corresponds to a principal component.
     * The number of rows is equal to the number of rows in `matrix`,
     * while the number of columns is equal to `number` (or less, if `Options::cap_number` is applied).
     */
    EigenMatrix_ rotation;

    /**
     * Variance explained by each principal component.
     * The length of this vector is equal to `number` (or less, if `Options::cap_number` is applied).
     */
    EigenVector_ variances;

    /**
     * The number of restart iterations performed.
     */
    int iterations;

    /**
     * Whether the algorithm converged.
     */
    bool converged;
};

/** 
 * Convenient overload of `pca()` that allocates memory for the output matrices of the PCA.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class for the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix where rows are observations and columns are features.
 * @param center Should the matrix be centered by column?
 * This should be set to `true` if the matrix is not already centered.
 * @param scale Should the matrix be scaled to unit variance for each column?
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class OutputEigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class InputEigenMatrix_>
PcaResults<OutputEigenMatrix_, EigenVector_> pca(const InputEigenMatrix_& matrix, bool center, bool scale, Eigen::Index number, const Options<EigenVector_>& options) {
    PcaResults<OutputEigenMatrix_, EigenVector_> output;
    const auto stats = pca(matrix, center, scale, number, output.scores, output.rotation, output.variances, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

}

#endif
