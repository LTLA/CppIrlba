#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include <random>
#include <utility>
#include <type_traits>

#include "aarand/aarand.hpp"
#include "Eigen/Dense"

namespace irlba {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

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

}

#endif
