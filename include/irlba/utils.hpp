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
    auto num_total = vec.size();
    const bool odd = num_total % 2;
    if (odd) {
        --num_total;
    }

    // Box-Muller gives us two random values at a time.
    typedef typename EigenVector_::Scalar Float;
    for (I<decltype(num_total)> i = 0; i < num_total; i += 2) {
        const auto paired = aarand::standard_normal<Float>(eng);
        vec[i] = paired.first;
        vec[i + 1] = paired.second;
    }

    if (odd) {
        // Adding the poor extra for odd total lengths.
        auto paired = aarand::standard_normal<Float>(eng);
        vec[num_total] = paired.first;
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
