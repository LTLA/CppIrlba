#ifndef IRLBA_UTILS_HPP
#define IRLBA_UTILS_HPP

#include "Eigen/Dense"

namespace irlba {

/*
 * Orthogonalizes a vector against a set of orthonormal column vectors in a matrix.
 */
class OrthogonalizeVector {
public:
    /**
     * @param mat A matrix of orthonormal column vectors.
     * @param vec The vector of interest, of length equal to `mat.cols()`.
     *
     * @return `vec` is modified to contain `vec - mat * t(mat) * vec`,
     * which is orthogonal to each column of `mat`.
     */
    void operator()(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec) {
        tmpmat.noalias() = mat.adjoint() * vec;
        vec.noalias() -= mat * tmpmat;
        return;
    }
private:
    Eigen::MatrixXd tmpmat;
};

}

#endif
