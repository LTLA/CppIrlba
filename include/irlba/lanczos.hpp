#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include <cmath>
#include <limits>

namespace irlba {
    
class LanczosProcess {
public:
    LanczosProcess() {} 
public:
    static constexpr double default_eps = std::pow(std::numeric_limits<double>::epsilon, 0.8); // inherited from irlba.

    LanczosProcess& set_eps(double e = default_eps) {
        eps = e;
        return *this;
    }

public:
    template<class M, class CENTER, class SCALE, class NORMSAMP>
    void operator()(const M& mat, Eigen::MatrixXd& W, Eigen::MatrixXd& V, const CENTER& center, const SCALE& scale, NORMSAMP& norm, int work, int start, bool first) {
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        orthog.set_size(work);
        F.resize(mat.cols());
        W_next.resize(mat.rows());

        // Doing some preparatory work.
        if (start == 0 && first) {
            double d = V.col(0).norm();
            if (d < eps) {
                throw -1; // TODO: try refilling with another random normal?
            }
            V.col(0) /= d;
        }

        F.noalias() = V.col(start);

        if constexpr(do_scale) {
            F /= scale;
        }

        W_next.noalias() = mat * F;

        if constexpr(do_center) {
            double beta = F.dot(center);
            for (auto& w : W_next) {
                w -= beta;
            }
        }

        if (start && !first) {
            orthog(W, W_next, start);
        }

        double S = W_next.norm();
        if (S < eps && start == 0) {
            throw -4; // TODO: try refilling with another random normal? And what happens if S < eps and start > 0; div by zero?
        }
        W_next /= S;
        W.col(start) = W_next;

        // The Lanczos iterations themselves.
        for (int j = start; j < work; ++j) {
            F.noalias() = mat.adjoint() * W.col(j);

            // Centering and scaling, if requested.
            if constexpr(do_center) {
                F.noalias() -= center;
            }
            if constexpr(do_scale) {
                F.noalias() /= scale;
            }

            F.noalias() -= S * V.col(j); // equivalent to daxpy.
            orthog(V, F, j + 1);

            if (j + 1 < work) {
                double R_F = F.norm();

                if (R_F < eps) {
                    for (auto& f : F) { f = norm(); }
                    orthog(V, F, j + 1);
                    R_F = F.norm();
                    F /= R_F;
                    R_F = 0;
                } else {
                    F /= R_F;
                }

                V.col(j + 1) = F;

                B(j, j) = S;
                B(j, j + 1) = R_F;

                // Re-using 'F' as 'x', the temporary buffer used in irlb.c's
                // inner loop. 'F's original value will not be used in the
                // rest of the loop, so no harm, no foul.
                auto& x = F;

                // Applying the scaling.
                if constexpr(do_scale) {
                    x /= scale;
                }

                W_next.noalias() = mat * x;

                // Applying the centering.
                if constexpr(do_center) {
                    double beta = x.dot(center);
                    for (auto& x : W_next) {
                        x -= beta;
                    }
                }

                // One round of classical Gram-Schmidt. 
                W_next.noalias() -= R_F * W.col(j);

                // Full re-orthogonalization of W_{j+1}.
                orthog(W, W_next, j + 1);

                S = W_next.norm();
                if (S < eps) {
                    for (auto& w : W_next) { w = norm(); }
                    orthog(W, W_next, j + 1);
                    S = W_next.norm();
                    W_next /= S;
                    S = 0;
                } else {
                    W_next /= S;
                }

                W.col(j + 1) = W_next;
            } else {
                B(j, j) = S;
            }
        }

        return 0;
    }

private:
    OrthogonalizeVector orthog;
    double eps = default_eps;

    Eigen::VectorXd F; 
    Eigen::VectorXd W_next;
};

}

#endif
