#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"
#include <random>
#include <utility>
#include "aarand/aarand.hpp"

namespace irlba {

template<class EigenMatrix_, class EigenVector_>
void orthogonalize_vector(const EigenMatrix_& mat, EigenVector_& vec, size_t ncols, EigenVector_& tmp) {
    tmp.head(ncols).noalias() = mat.leftCols(ncols).adjoint() * vec;
    vec.noalias() -= mat.leftCols(ncols) * tmp.head(ncols);
}

template<class EigenVector_, class Engine_>
void fill_with_random_normals(EigenVector_& vec, Engine_& eng) {
    Eigen::Index i = 1, limit = vec.size();
    while (i < limit) {
        auto paired = aarand::standard_normal<typename EigenVector_::Scalar>(eng);
        vec[i - 1] = paired.first;
        vec[i] = paired.second;
        i += 2;
    }

    if (i == limit) {
        auto paired = aarand::standard_normal(eng);
        vec[i - 1] = paired.first;
    }
}

template<class Matrix_>
struct ColumnVectorProxy {
    ColumnVectorProxy(Matrix_& m, Eigen::Index c) : mat(m), col(c) {}
    auto size () { return mat.rows(); }
    auto& operator[](Eigen::Index r) { return mat(r, col); }
    Matrix_& mat;
    Eigen::Index col;
    typedef typename Matrix_::Scalar Scalar;
};

template<class Matrix_, class Engine_>
void fill_with_random_normals(Matrix_& mat, Eigen::Index column, Engine_& eng) {
    ColumnVectorProxy proxy(mat, column);
    fill_with_random_normals(proxy, eng);
}

/**
 * Utility function to generate a default `NULL` argument for the random number generator input.
 *
 * @return A null pointer to an RNG.
 * Any RNG will do here, so we use the most common one.
 */
constexpr std::mt19937_64* null_rng() { return NULL; }

template<class Matrix_, typename = int>
struct has_multiply_method {
    static constexpr bool value = false;
};

template<class M>
struct has_multiply_method<M, decltype((void) (std::declval<M>() * std::declval<Eigen::VectorXd>()), 0)> {
    static constexpr bool value = true;
};

template<class M, typename = int>
struct has_adjoint_multiply_method {
    static constexpr bool value = false;
};

template<class M>
struct has_adjoint_multiply_method<M, decltype((void) (std::declval<M>().adjoint() * std::declval<Eigen::VectorXd>()), 0)> {
    static constexpr bool value = true;
};

template<class M, typename = int>
struct has_realize_method {
    static constexpr bool value = false;
};

template<class M>
struct has_realize_method<M, decltype((void) std::declval<M>().realize(), 0)> {
    static constexpr bool value = std::is_same<decltype(std::declval<M>().realize()), Eigen::MatrixXd>::value;
};

}

#endif
