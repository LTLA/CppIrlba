#ifndef IRLBA_WRAPPERS_HPP
#define IRLBA_WRAPPERS_HPP

#include "utils.hpp"
#include "Eigen/Dense"

namespace irlba {

template<class Matrix>
struct CenteredWrapper {
    CenteredWrapper(const Matrix* m, const Eigen::VectorXd* c) : mat(m), center(c) {}
    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

    template<class Right>
    void multiply(Right&& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_multiply_method<Matrix>::value) {
            out.noalias() = *mat * rhs;
        } else {
            mat->multiply(rhs, out);
        }

        double beta = rhs.dot(*center);
        for (auto& o : out) {
            o -= beta;
        }
        return;
    }

    template<class Right>
    void adjoint_multiply(Right&& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_adjoint_multiply_method<Matrix>::value) {
            out.noalias() = mat->adjoint() * rhs;
        } else {
            mat->adjoint_multiply(rhs, out);
        }

        double beta = rhs.sum();
        out -= beta * (*center);
        return;
    }

    Eigen::MatrixXd realize() const {
        auto subtractor = [&](Eigen::MatrixXd& m) -> void {
            for (Eigen::Index c = 0; c < m.cols(); ++c) {
                for (Eigen::Index r = 0; r < m.rows(); ++r) {
                    m(r, c) -= (*center)[c];
                }
            }
        };

        if constexpr(has_realize_method<Matrix>::value) {
            Eigen::MatrixXd output = mat->realize();
            subtractor(output);
            return output;
        } else {
            Eigen::MatrixXd output(*mat);
            subtractor(output);
            return output;
        }
    }

    const Matrix* mat;
    const Eigen::VectorXd* center;
};

template<class Matrix>
struct ScaledWrapper {
    ScaledWrapper(const Matrix* m, const Eigen::VectorXd* s) : mat(m), scale(s) {}
    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

    template<class Right>
    void multiply(Right&& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_multiply_method<Matrix>::value) {
            out.noalias() = *mat * rhs.cwiseQuotient(*scale);
        } else {
            mat->multiply(rhs.cwiseQuotient(*scale), out);
        }
        return;
    }

    template<class Right>
    void adjoint_multiply(Right&& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_adjoint_multiply_method<Matrix>::value) {
            out.noalias() = mat->adjoint() * rhs;
        } else {
            mat->adjoint_multiply(rhs, out);
        }
        out.noalias() = out.cwiseQuotient(*scale);
        return;
    }

    Eigen::MatrixXd realize() const {
        auto scaler = [&](Eigen::MatrixXd& m) -> void {
            for (Eigen::Index c = 0; c < m.cols(); ++c) {
                for (Eigen::Index r = 0; r < m.rows(); ++r) {
                    m(r, c) /= (*scale)[c];
                }
            }
        };

        if constexpr(has_realize_method<Matrix>::value) {
            Eigen::MatrixXd output = mat->realize();
            scaler(output);
            return output;
        } else {
            Eigen::MatrixXd output(*mat);
            scaler(output);
            return output;
        }
    }

    const Matrix* mat;
    const Eigen::VectorXd* scale;
};

}

#endif
