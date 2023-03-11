#ifndef IRLBA_IRLBA_HPP
#define IRLBA_IRLBA_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "lanczos.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

/**
 * @file irlba.hpp
 *
 * @brief Implements the main user-visible class for running IRLBA.
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
public:
    /**
     * @brief Default parameter values.
     */
    struct Defaults {
        /**
         * See `set_number()` for more details.
         */
        static constexpr int number = 5;

        /**
         * See `set_work()` for more details.
         */
        static constexpr int extra_work = 7;

        /**
         * See `set_maxit()` for more details.
         */
        static constexpr int maxit = 1000;

        /**
         * See `set_seed()` for more details.
         */
        static constexpr uint64_t seed = std::mt19937_64::default_seed;
    };
private:
    LanczosBidiagonalization lp;

    int number = Defaults::number; 
    int extra_work = Defaults::extra_work;
    int maxit = Defaults::maxit;
    uint64_t seed = Defaults::seed;

    ConvergenceTest convtest;

public:
    /**
     * Specify the number of singular values/vectors to obtain.
     * This should be less than the smaller dimension of the matrix supplied to `run()`.
     *
     * @param n Number of singular values/vectors of interest.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_number(int n = Defaults::number) {
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
    Irlba& set_maxit(int m = Defaults::maxit) {
        maxit = m;
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
    Irlba& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
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
    Irlba& set_work(int w = Defaults::extra_work) {
        extra_work = w;
        return *this;
    }

    /**
     * See `LanczosBidiagonalization::set_epsilon()` for details.
     *
     * @param e Epsilon value.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_invariant_tolerance(double e = LanczosBidiagonalization::Defaults::epsilon) {
         lp.set_epsilon(e);
         return *this;
    }

    /**
     * See `ConvergenceTest::set_tol()` for details.
     *
     * @param t Positive tolerance value.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_convergence_tolerance(double t = ConvergenceTest::Defaults::tol) {
        convtest.set_tol(t);
        return *this;
    }

    /**
     * See `ConvergenceTest::set_svtol()` for details.
     *
     * @param t Positive tolerance value, or -1.
     *
     * @return A reference to the `Irlba` instance.
     */
    Irlba& set_singular_value_ratio_tolerance(double t = ConvergenceTest::Defaults::svtol) {
        convtest.set_svtol(t);
        return *this;
    }

public:
    /** 
     * Run IRLBA on an input matrix to perform an approximate SVD, with arbitrary centering and scaling operations.
     *
     * @tparam Input Matrix class, typically from the **Eigen** matrix manipulation library.
     * However, other classes are also supported, see the other `run()` methods for details.
     * @tparam Engine A (pseudo-)random number generator class, returning a randomly sampled value when called as a functor with no arguments.
     *
     * @param[in] mat Input matrix.
     * @param center Should the matrix be centered by column?
     * @param scale Should the matrix be scaled to unit variance for each column?
     * @param[out] outU Output matrix where columns contain the first left singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of rows in `mat`.
     * @param[out] outV Output matrix where columns contain the first right singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of columns in `mat`.
     * @param[out] outD Vector to store the first singular values.
     * The length is set automatically as defined by `set_number()`.
     * @param eng Pointer to an instance of a random number generator.
     * If set to `NULL`, a Mersenne Twister is used internally with the seed defined by `set_seed()`. 
     * @param[in] init Pointer to a vector of length equal to the number of columns of `mat`,
     * containing the initial values of the first right singular vector.
     *
     * @return A pair where the first entry indicates whether the algorithm converged,
     * and the second entry indicates the number of restart iterations performed.
     *
     * Centering is performed by subtracting each element of `center` from the corresponding column of `mat`.
     * Scaling is performed by dividing each column of `mat` by the corresponding element of `scale` (after any centering has been applied).
     * Note that `scale=true` requires `center=true` to guarantee unit variance along each column. 
     * No scaling is performed when the variance of a column is zero, so as to avoid divide-by-zero errors. 
     */
    template<class Input, class Engine = std::mt19937_64>
    std::pair<bool, int> run(
        const Input& mat, 
        bool center, 
        bool scale, 
        MatrixOf<typename Input::Scalar>& outU, 
        MatrixOf<typename Input::Scalar>& outV, 
        RealVectorOf<typename Input::Scalar>& outD, 
        Engine* eng = null_rng(),
        VectorOf<typename Input::Scalar>* init = NULL) 
    const {
        typedef VectorOf<typename Input::Scalar> Vector;
        typedef typename Input::Scalar Scalar;

        if (scale || center) {
            Vector center0, scale0;

            if (center) {
                if (mat.rows() < 1) {
                    throw std::runtime_error("cannot center with no observations");    
                }
                center0.resize(mat.cols());
            }

            if (scale) {
                if (mat.rows() < 2) {
                    throw std::runtime_error("cannot scale with fewer than two observations");    
                }
                scale0.resize(mat.cols());
            }

            for (Eigen::Index i = 0; i < mat.cols(); ++i) {
                Scalar mean = 0;
                if (center) {
                    mean = mat.col(i).sum() / mat.rows();
                    center0[i] = mean;
                }
                if (scale) {
                    Vector current = mat.col(i); // force the column to be a Vector, even if it's coming from a sparse matrix.
                    Scalar var = 0;
                    for (auto x : current) {
                        var += (x - mean)*(x - mean);
                    }

                    if (var) {
                        scale0[i] = std::sqrt(var/(mat.rows() - 1));
                    } else {
                        scale0[i] = 1;
                    }
                }
            }

            if (center) {
                Centered<Input> centered(&mat, &center0);
                if (scale) {
                    Scaled<decltype(centered)> centered_scaled(&centered, &scale0);
                    return run(centered_scaled, outU, outV, outD, eng, init);
                } else {
                    return run(centered, outU, outV, outD, eng, init);
                }
            } else {
                Scaled<Input> scaled(&mat, &scale0);
                return run(scaled, outU, outV, outD, eng, init);
            }
        } else {
            return run(mat, outU, outV, outD, eng, init);
        }
    }

    /** 
     * Run IRLBA on an input matrix to perform an approximate SVD.
     *
     * @tparam Input Matrix class, typically from the **Eigen** matrix manipulation library.
     * However, other classes are also supported, see below for details.
     * @tparam Engine A (pseudo-)random number generator class, returning a randomly sampled value when called as a functor with no arguments.
     *
     * @param[in] mat Input matrix.
     * @param[out] outU Output matrix where columns contain the first left singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of rows in `mat`.
     * @param[out] outV Output matrix where columns contain the first right singular vectors.
     * Dimensions are set automatically on output;
     * the number of columns is defined by `set_number()` and the number of rows is equal to the number of columns in `mat`.
     * @param[out] outD Vector to store the first singular values.
     * The length is set automatically as defined by `set_number()`.
     * @param eng Pointer to an instance of a random number generator.
     * If set to `NULL`, a Mersenne Twister is used internally with the seed defined by `set_seed()`. 
     * @param[in] init Pointer to a vector of length equal to the number of columns of `mat`,
     * containing the initial values of the first right singular vector.
     *
     * @return A pair where the first entry indicates whether the algorithm converged,
     * and the second entry indicates the number of restart iterations performed.
     *
     * Custom classes can be used to define modified matrices that cannot be efficiently realized into the standard **Eigen** classes.
     * We expect:
     * - A `rows()` method that returns the number of rows.
     * - A `cols()` method that returns the number of columns.
     * - One of the following for matrix-vector multiplication:
     *   - `multiply(rhs, out)`, which should compute the product of the matrix with `rhs`, a `Vector`-equivalent of length equal to the number of columns;
     *     and stores the result in `out`, an `Vector` of length equal to the number of rows.
     *   - A `*` method where the right-hand side is an `Vector` (or equivalent expression) of length equal to the number of columsn,
     *     and returns an `Vector`-equivalent of length equal to the number of rows.
     * - One of the following for matrix transpose-vector multiplication:
     *   - `adjoint_multiply(rhs, out)`, which should compute the product of the matrix transpose with `rhs`, a `Vector`-equivalent of length equal to the number of rows;
     *     and stores the result in `out`, an `Vector` of length equal to the number of columns.
     *   - An `adjoint()` method that returns an instance of any class that has a `*` method for matrix-vector multiplication.
     *     The method should accept an `Vector`-equivalent of length equal to the number of rows,
     *     and return an `Vector`-equvialent of length equal to the number of columns.
     * - A `realize()` method that returns an `MatrixOf<typename Input::Scalar>` object representing the modified matrix.
     *   This can be omitted if an `MatrixOf<typename Input::Scalar>` can be copy-constructed from the class.
     *
     * See the `Centered` and `Scaled` classes for more details.
     *
     * If the smallest dimension of `mat` is below 6, this method falls back to performing an exact SVD.
     */
    template<class Input, class Engine = std::mt19937_64>
    std::pair<bool, int> run(
        const Input& mat, 
        MatrixOf<typename Input::Scalar>& outU, 
        MatrixOf<typename Input::Scalar>& outV, 
        RealVectorOf<typename Input::Scalar>& outD, 
        Engine* eng = null_rng(),
        VectorOf<typename Input::Scalar>* init = NULL) 
    const {
        if (eng == NULL) {
            std::mt19937_64 rng(seed);
            return run_internal(mat, rng, outU, outV, outD, init);
        } else {
            return run_internal(mat, *eng, outU, outV, outD, init);
        }
    }

private:
    template<class Input, class Engine>
    std::pair<bool, int> run_internal(
        const Input& mat, 
        Engine& eng, 
        MatrixOf<typename Input::Scalar>& outU, 
        MatrixOf<typename Input::Scalar>& outV, 
        RealVectorOf<typename Input::Scalar>& outD, 
        VectorOf<typename Input::Scalar>* init)
    const {
        typedef MatrixOf<typename Input::Scalar> Matrix;
        typedef VectorOf<typename Input::Scalar> Vector;
        typedef RealVectorOf<typename Input::Scalar> RealVector;

        const int smaller = std::min(mat.rows(), mat.cols());
        if (number >= smaller) {
            throw std::runtime_error("requested number of singular values must be less than smaller matrix dimension");
        }

        // Falling back to an exact SVD for small matrices.
        if (smaller < 6) {
            exact(mat, outU, outV, outD);
            return std::make_pair(true, 0);
        }

        const int work = std::min(number + extra_work, smaller);

        Matrix V(mat.cols(), work);
        if (init) {
            if (init->size() != V.rows()) {
                throw std::runtime_error("initialization vector does not have expected number of rows");
            }
            V.col(0) = *init;
        } else {
            fill_with_random_normals(V, 0, eng);
        }
        V.col(0) /= V.col(0).norm();

        bool converged = false;
        int iter = 0, k =0;
        Eigen::JacobiSVD<Matrix> svd(work, work, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto lptmp = lp.initialize(mat);

        Matrix W(mat.rows(), work);
        Matrix Wtmp(mat.rows(), work);
        Matrix Vtmp(mat.cols(), work);

        Matrix B(work, work);
        B.setZero(work, work);
        Vector res(work);
        Vector F(mat.cols());

        RealVector prevS(work);

        for (; iter < maxit; ++iter) {
            // Technically, this is only a 'true' Lanczos bidiagonalization
            // when k = 0. All other times, we're just recycling the machinery,
            // see the text below Equation 3.11 in Baglama and Reichel.
            lp.run(mat, W, V, B, eng, lptmp, k);

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
            // Bottom-left element of B is the final value of 'S' from the Lanczos iterations.
            // Note that this still works for complex numbers, as 0+0i == 0.0.
            if (B(work-1, work-1) == 0.0) { 
                converged = true;
                break;
            }

            F = lptmp.residuals();
            double R_F = F.norm();
            F /= R_F;

            // Computes the convergence criterion defined in on the LHS of Equation 2.13 of Baglama and Riechel.
            // We expose it here as we will be re-using the same values to update B, see below.
            res = R_F * BU.row(work - 1);

            int n_converged = 0;
            if (iter > 0) {
                n_converged = convtest.run(BS, res, prevS);
                if (n_converged >= number) {
                    converged = true;
                    break;
                }
            }
            prevS = BS;

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
        outD.resize(number);
        outD = svd.singularValues().head(number);

        outU.resize(mat.rows(), number);
        outU.noalias() = W * svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV.noalias() = V * svd.matrixV().leftCols(number);

        return std::make_pair(converged, iter + 1);
    }

private:
    template<class Input>
    void exact(const Input& mat, MatrixOf<typename Input::Scalar>& outU, MatrixOf<typename Input::Scalar>& outV, RealVectorOf<typename Input::Scalar>& outD) const {
        typedef MatrixOf<typename Input::Scalar> Matrix;
        Eigen::BDCSVD<Matrix> svd(mat.rows(), mat.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        if constexpr(has_realize_method<Input>::value) {
            Matrix adjusted = mat.realize();
            svd.compute(adjusted);
        } else if constexpr(std::is_same<Matrix, Input>::value) {
            svd.compute(mat);
        } else {
            Matrix adjusted(mat);
            svd.compute(adjusted);
        }

        outD.resize(number);
        outD = svd.singularValues().head(number);

        outU.resize(mat.rows(), number);
        outU = svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV = svd.matrixV().leftCols(number);

        return;
    }

public:
    /**
     * Result of the IRLBA-based decomposition.
     *
     * @tparam Scalar Scalar type for the left/right singular vectors.
     */
    template<class Scalar>
    struct Results {
        /**
         * The left singular vectors, stored as columns of `U`.
         * The number of rows in `U` is equal to the number of rows in the input matrix,
         * and the number of columns is equal to the number of requested vectors.
         */
        MatrixOf<Scalar> U;

        /**
         * The right singular vectors, stored as columns of `U`.
         * The number of rows in `U` is equal to the number of columns in the input matrix,
         * and the number of columns is equal to the number of requested vectors.
         */
        MatrixOf<Scalar> V;

        /**
         * The requested number of singular values, ordered by decreasing value.
         * These correspond to the vectors in `U` and `V`.
         */
        RealVectorOf<Scalar> D;

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
     * Run IRLBA on an input matrix to perform an approximate SVD with centering and scaling.
     * 
     * @tparam Input Matrix class, typically from the **Eigen** matrix manipulation library.
     * However, other classes are also supported, see the other `run()` methods for details.
     * @tparam Engine A (pseudo-)random number generator class, returning a randomly sampled value when called as a functor with no arguments.
     *
     * @param[in] mat Input matrix.
     * @param center Should the matrix be centered by column?
     * @param scale Should the matrix be scaled to unit variance for each column?
     * @param eng Pointer to an instance of a random number generator.
     * If set to `NULL`, a Mersenne Twister is used internally with the seed defined by `set_seed()`. 
     * @param[in] init Pointer to a vector of length equal to the number of columns of `mat`,
     * containing the initial values of the first right singular vector.
     *
     * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
     */
    template<class Input, class Engine = std::mt19937_64>
    Results<typename Input::Scalar> run(const Input& mat, bool center, bool scale, Engine* eng = null_rng(), VectorOf<typename Input::Scalar>* init = NULL) {
        Results<typename Input::Scalar> output;
        auto stats = run(mat, center, scale, output.U, output.V, output.D, eng, init);
        output.converged = stats.first;
        output.iterations = stats.second;
        return output;
    }

    /** 
     * Run IRLBA on an input matrix to perform an approximate SVD, see the `run()` method for more details.
     *
     * @tparam Input Matrix class, typically from the **Eigen** matrix manipulation library.
     * However, other classes are also supported, see the other `run()` methods for details.
     * @tparam Engine A (pseudo-)random number generator class, returning a randomly sampled value when called as a functor with no arguments.
     *
     * @param[in] mat Input matrix.
     * @param eng Pointer to an instance of a random number generator.
     * If set to `NULL`, a Mersenne Twister is used internally with the seed defined by `set_seed()`. 
     * @param[in] init Pointer to a vector of length equal to the number of columns of `mat`,
     * containing the initial values of the first right singular vector.
     *
     * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
     */
    template<class Input, class Engine = std::mt19937_64>
    Results<typename Input::Scalar> run(const Input& mat, Engine* eng = null_rng(), VectorOf<typename Input::Scalar>* init = NULL) {
        Results<typename Input::Scalar> output;
        auto stats = run(mat, output.U, output.V, output.D, eng, init);
        output.converged = stats.first;
        output.iterations = stats.second;
        return output;
    }
};

}

#endif
