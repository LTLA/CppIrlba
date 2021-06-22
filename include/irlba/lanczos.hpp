#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"

namespace irlba {
    
class LanczosProcess {
public:
    template<class M>
    void operator()(const M& mat, Eigen::MatrixXd& W, Eigen::MatrixXd& V, double S) {
        for (int j = 0; j < work; ++j) {
            F.noalias() = mat.adjoint() * W.col(j);

            // Centering and scaling, if requested.
            if (center.size())  {
                F.noalias() -= center;
            }
            if (scale.size()) {
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
                if (scale.size()) {
                    x /= scale;
                }

                W_next.noalias() = mat * F;

                // Applying the centering.
                if (center.size()) {
                    double beta = F.dot(center);
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
                    W_next /= F.norm();
                    S = 0;
                } else {
                    W_next /= S;
                }

                W.col(j + 1) = W_next;
            } else {
                B(j, j) = S;
            }
        }
    }

private:
    int work;
    double eps;

    Eigen::VectorXd F, scale, center;
    Eigen::VectorXd W_next;

    OrthogonalizeVector orthog;

    NormalSampler norm;
};


}

#endif
