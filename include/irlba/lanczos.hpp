#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include <cmath>
#include <limits>

namespace irlba {
    
class LanczosProcess {
public:
    static constexpr double default_eps = std::pow(std::numeric_limits<double>::epsilon(), 0.8); // inherited from irlba.

    LanczosProcess& set_eps(double e = default_eps) {
        eps = e;
        return *this;
    }

public:
    template<class M, class CENTER, class SCALE, class NORMSAMP>
    void run(const M& mat, Eigen::MatrixXd& W, Eigen::MatrixXd& V, Eigen::MatrixXd& B, const CENTER& center, const SCALE& scale, NORMSAMP& norm, int work, bool first) {
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        orthog.set_size(work);
        F.resize(mat.cols());
        W_next.resize(mat.rows());

        // Doing some preparatory work.
        if (first) {
            double d = V.col(0).norm();
            if (d < eps) {
                throw -1; 
            }
            V.col(0) /= d;
        }

        F = V.col(0);

        if constexpr(do_scale) {
            F = F.cwiseQuotient(scale);
        }

        W_next.noalias() = mat * F;

        if constexpr(do_center) {
            double beta = F.dot(center);
            for (auto& w : W_next) {
                w -= beta;
            }
        }

        double S = W_next.norm();
        if (S < eps) {
            throw -4;
        }
        W_next /= S;
        W.col(0) = W_next;

        // The Lanczos iterations themselves.
        for (int j = 0; j < work; ++j) {
            F.noalias() = mat.adjoint() * W.col(j);

            // Centering and scaling, if requested.
            if constexpr(do_center) {
                double beta = W.col(j).sum();
                F -= beta * center;
            }
            if constexpr(do_scale) {
                F = F.cwiseQuotient(scale);
            }

            F -= S * V.col(j); // equivalent to daxpy.
            orthog.run(V, F, j + 1);

            if (j + 1 < work) {
                double R_F = F.norm();

                if (R_F < eps) {
                    for (auto& f : F) { f = norm(); }
                    orthog.run(V, F, j + 1);
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
                    x = x.cwiseQuotient(scale);
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
                W_next -= R_F * W.col(j);

                // Full re-orthogonalization of W_{j+1}.
                orthog.run(W, W_next, j + 1);

                S = W_next.norm();
                if (S < eps) {
                    for (auto& w : W_next) { w = norm(); }
                    orthog.run(W, W_next, j + 1);
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

        return;
    }

public:
    const Eigen::VectorXd& finalF() const {
        return F;
    }
    
private:
    OrthogonalizeVector orthog;
    double eps = default_eps;

    Eigen::VectorXd F; 
    Eigen::VectorXd W_next;
};

}

#endif
