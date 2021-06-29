#ifndef IRLBA_IRLBA_HPP
#define IRLBA_IRLBA_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "lanczos.hpp"
#include <cmath>

/**
 * @file irlba.hpp
 *
 * Implements the main user-visible class for running IRLBA.
 */

namespace irlba {

/**
 * @brief Run IRLBA on an input matrix.
 *
 * Implements the Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA) for fast truncated singular value decomposition.
 * This is heavily derived from the C code in the [**irlba** package](https://github.com/bwlewis/irlba),
 * with refactoring into C++ to use Eigen instead of LAPACK for much of the matrix algebra.
 */
class Irlba {
private:
    Eigen::MatrixXd W, V, B;
    Eigen::MatrixXd Wtmp, Vtmp;

    Eigen::VectorXd initV, res, F;

    LanczosBidiagonalization lp;
    int number = 5, extra_work = 7;
    int maxit = 1000;

    ConvergenceTest convtest;

public:
    /**
     * Set the starting vector for V.
     *
     * @param v Arbitrary vector of length equal to the number of columns in the matrix supplied to `run()`.
     * 
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_init(const Eigen::VectorXd& v) {
        initV = v;
        return *this;
    }

    /**
     * Specify the number of singular values/vectors to obtain.
     * This should be less than the smaller dimension of the matrix supplied to `run()`.
     *
     * @param n Number of singular values/vectors of interest.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_number(int n = 5) {
        number = n;
        return *this;
    }

    /**
     * Set the maximum number of restart iterations.
     * In most cases, convergence will occur before reaching this limit.
     *
     * @param m Maximum number of iterations.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_maxit(int m = 1000) {
        maxit = m;
        return *this;
    }

    /**
     * Set the number of extra dimensions to define the working subspace.
     * Larger values can speed up convergence at the cost of more memory use.
     *
     * @param w Number of extra dimensions, added to the value specified in `set_number()` to obtain the working subspace dimension.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_work(int w = 7) {
        extra_work = w;
        return *this;
    }

    /**
     * Set the tolerance for detecting invariant subspaces. 
     *
     * @param e Positive tolerance value.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_invariant_tolerance(double e) {
         lp.set_eps(e);
         return *this;
    }

    /**
     * Set the tolerance for detecting invariant subspaces to its default value, 
     * namely the epsilon for double-precision values to the power of 0.8.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_invariant_tolerance() {
        lp.set_eps();
        return *this;
    }

    /**
     * Set the tolerance for convergence based on the residuals.
     * This is used to define a threshold against which the residuals of the decomposition are compared;
     * all values must be below the threshold for convergence to be considered.
     *
     * @param t Positive tolerance value.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_convergence_tolerance(double t = 1e-5) {
        convtest.set_tol(t);
        return *this;
    }

    /**
     * Set the tolerance for convergence based on the singular value ratios.
     * At each iteration, the absolute value of the relative difference in the singular values is compared to the tolerance; 
     * all values must be below the threshold for convergence to be considered.
     *
     * @param t Positive tolerance value.
     * Alternatively, a negative value, in which case the value from `set_convergence_tolerance()` is used instead.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_singular_value_ratio_tolerance(double t = -1) {
        convtest.set_svtol(t);
        return *this;
    }

public:
    /** 
     * Run IRLBA on an input matrix to perform an approximate SVD.
     *
     * If `center` (and optionally `scale`) are set accordingly, this can be used to perform an approximate PCA.
     *
     * If the smallest dimension of `mat` is below 6, this method falls back to performing an exact SVD.
     * 
     * @tparam M Matrix class that supports `cols()`, `rows()`, `*` and `adjoint()`.
     * This is most typically a class from the Eigen library.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam NORMSAMP A functor that, when called with no arguments, returns a random Normal value.
     *
     * @param mat Input matrix.
     * @param center A vector of length equal to the number of columns of `mat`.
     * Each value is to be subtracted from the corresponding column of `mat`.
     * Alternatively `false`, if no centering is to be performed.
     * @param scale A vector of length equal to the number of columns of `mat`.
     * Each value should be positive and is used to divide the corresponding column of `mat`.
     * @param norm An instance of a functor to generate normally distributed values.
     * Alternatively `false`, if no scaling is to be performed.
     * @param outU Output matrix where columns contain the first left singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of rows in `mat`.
     * @param outV Output matrix where columns contain the first right singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of columns in `mat`.
     * @param outS Vector to store the first singular values.
     * The length is set automatically as defined by `set_number()`.
     *
     * @return A pair where the first entry indicates whether the algorithm converged,
     * and the second entry indicates the number of restart iterations performed.
     */
    template<class M, class CENTER, class SCALE, class NORMSAMP>
    std::pair<bool, int> run(const M& mat, const CENTER& center, const SCALE& scale, NORMSAMP& norm, Eigen::MatrixXd& outU, Eigen::MatrixXd& outV, Eigen::VectorXd& outS) {
        const int smaller = std::min(mat.rows(), mat.cols());
        assert(number < smaller);

        // Falling back to an exact SVD for small matrices.
        if (smaller < 6) {
            exact(mat, center, scale, outU, outV, outS);
            return std::make_pair(true, 0);
        }

        const int work = std::min(number + extra_work, smaller);

        W.resize(mat.rows(), work);
        Wtmp.resize(mat.rows(), work);
        V.resize(mat.cols(), work);
        Vtmp.resize(mat.cols(), work);
        res.resize(work);
        F.resize(mat.cols());

        if (initV.size()) {
            assert(initV.size() == V.rows());
            V.col(0) = initV;
        } else {
            for (Eigen::Index i = 0; i < V.rows(); ++i) {
                V(i, 0) = norm();
            }
        }
        V.col(0) /= V.col(0).norm();

        B.resize(work, work);
        B.setZero(work, work);

        bool converged = false;
        int iter = 0, k =0;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(work, work, Eigen::ComputeThinU | Eigen::ComputeThinV);

        for (; iter < maxit; ++iter) {
            // Technically, this is only a 'true' Lanczos bidiagonalization
            // when k = 0. All other times, we're just recycling the machinery,
            // see the text below Equation 3.11 in Baglama and Reichel.
            lp.run(mat, W, V, B, center, scale, norm, k);

//            if (iter < 2) {
//                std::cout << "B is currently:\n" << B << std::endl;
//                std::cout << "W is currently:\n" << W << std::endl;
//                std::cout << "V is currently:\n" << V << std::endl;
//            }

            svd.compute(B);
            const auto& BS = svd.singularValues();
            const auto& BU = svd.matrixU();
            const auto& BV = svd.matrixV();

            // Checking for convergence.
            if (B(work-1, work-1) == 0) { // a.k.a. the final value of 'S' from the Lanczos iterations.
                converged = true;
                break;
            }

            F = lp.residuals();
            double R_F = F.norm();
            F /= R_F;

            // Computes the convergence criterion defined in on the LHS of Equation 2.13 of Baglama and Riechel.
            // We expose it here as we will be re-using the same values to update B, see below.
            res = R_F * BU.row(work - 1);

            int n_converged = 0;
            if (iter > 0) {
                n_converged = convtest.run(BS, res);
                if (n_converged >= number) {
                    converged = true;
                    break;
                }
            }
            convtest.set_last(BS);

            // Setting 'k'. This looks kinda weird, but this is deliberate,
            // see the text below Algorithm 3.1 of Baglama and Reichel.
            if (n_converged + number > k) {
                k = n_converged + number;
            }
            if (k > work - 3) {
                k = work - 3;
            }
            if (k < 1) {
                k = 1;
            }

            // Updating B, W and V.
            Vtmp.leftCols(k).noalias() = V * BV.leftCols(k);
            V.leftCols(k) = Vtmp.leftCols(k);

            V.col(k) = F; // See Equation 3.2 of Baglama and Reichel, where our 'V' is
                          // their 'P', and our 'F' is their 'p_{m+1}' (2.2).
                          // 'F' was orthogonal to the old 'V' and it so it
                          // should still be orthogonal to the new left-most
                          // columns of 'V'; the input expectations of 'lp'
                          // are still met.

            Wtmp.leftCols(k).noalias() = W * BU.leftCols(k);
            W.leftCols(k) = Wtmp.leftCols(k);

            B.setZero(work, work);
            for (int l = 0; l < k; ++l) {
                B(l, l) = BS[l];
                B(l, k) = res[l]; // this looks weird but is deliberate, see
                                  // Equation 3.6 of Baglama and Reichel.
                                  // By happy coincidence, this is the same
                                  // value used to determine convergence in
                                  // 2.13, so we can just re-use it.
            }
        }

        // See Equation 2.11 of Baglama and Reichel for how to get from B's
        // singular triplets to mat's singular triplets.
        outS.resize(number);
        outS = svd.singularValues().head(number);

        outU.resize(mat.rows(), number);
        outU.noalias() = W * svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV.noalias() = V * svd.matrixV().leftCols(number);

        return std::make_pair(converged, iter + 1);
    }
private:
    template<class M, class CENTER, class SCALE> 
    void exact(const M& mat, const CENTER& center, const SCALE& scale, Eigen::MatrixXd& outU, Eigen::MatrixXd& outV, Eigen::VectorXd& outS) {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(mat.rows(), mat.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        if constexpr(std::is_same<M, Eigen::MatrixXd>::value && !do_center && !do_scale) {
            svd.compute(mat);
        } else {
            Eigen::MatrixXd adjusted(mat);

            for (Eigen::Index i = 0; i < mat.cols(); ++i) {
                if (do_center) {
                    for (Eigen::Index j = 0; j < mat.rows(); ++j) {
                        adjusted(j, i) -= center[i];
                    }
                }
                if (do_scale) {
                    adjusted.col(i) /= scale[i];
                }
            }
            svd.compute(adjusted);
        }

        outS.resize(number);
        outS = svd.singularValues().head(number);

        outU.resize(mat.rows(), number);
        outU = svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV = svd.matrixV().leftCols(number);

        return;
    }
};

}

#endif
