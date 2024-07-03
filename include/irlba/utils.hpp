#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"
#include <random>
#include <utility>
#include "aarand/aarand.hpp"

namespace irlba {

namespace internal {

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

template<class Right_, class EigenVector_>
const EigenVector_& realize_rhs(const Right_& rhs, EigenVector_& buffer) {
    if constexpr(std::is_same<Right_, EigenVector_>::value) {
        return rhs;
    } else {
        buffer.noalias() = rhs;
        return buffer;
    }
}

}

}

#endif
