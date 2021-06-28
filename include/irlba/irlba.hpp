#ifndef IRLBA_IRLBA_HPP
#define IRLBA_IRLBA_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "lanczos.hpp"
#include <cmath>

namespace irlba {

class Irlba {
private:
    Eigen::MatrixXd W, V, B;
    Eigen::MatrixXd Wtmp, Vtmp;

    Eigen::VectorXd initV, res, F;

    LanczosProcess lp;
    int number = 5, extra_work = 7;
    int maxit = 1000;

    ConvergenceTest convtest;

public:
    Irlba& set_init(const Eigen::VectorXd& v) {
        initV = v;
        return *this;
    }

    Irlba& set_number(int n = 5) {
        number = n;
        return *this;
    }

    Irlba& set_maxit(int m = 1000) {
        maxit = m;
        return *this;
    }

    Irlba& set_work(int w = 7) {
        extra_work = w;
        return *this;
    }

public:
    template<class M, class CENTER, class SCALE, class NORMSAMP>
    std::pair<bool, int> run(const M& mat, const CENTER& center, const SCALE& scale, NORMSAMP& norm, Eigen::MatrixXd& outU, Eigen::MatrixXd& outV, Eigen::VectorXd& outS) {
        int work = number + extra_work;
        if (work > mat.rows()) {
            work = mat.rows();
        }
        if (work > mat.cols()) {
            work = mat.cols();
        }

        W.resize(mat.rows(), work);
        Wtmp.resize(mat.rows(), work);
        V.resize(mat.cols(), work);
        Vtmp.resize(mat.cols(), work);
        res.resize(work);
        F.resize(mat.cols());

        if (initV.size() == V.rows()) {
            V.col(0) = initV;
        } else {
            for (Eigen::Index i = 0; i < V.rows(); ++i) {
                V(i, 0) = norm();
            }
        }

        B.resize(work, work);
        B.setZero(work, work);

        bool converged = false;
        int iter = 0, k =0;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(work, work, Eigen::ComputeThinU | Eigen::ComputeThinV);

        for (; iter < maxit; ++iter) {
            lp.run(mat, W, V, B, center, scale, norm, work, k, iter==0);

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
            if (B(work-1, work-1) == 0) { // a.k.a. the final value of 'S' from the Lanczos iterations.
                converged = true;
                break;
            }

            // Not really sure what this is, but here we are.
            F = lp.finalF();
            double R_F = F.norm();
            F /= R_F;

            // irlba's original code uses 'j - 1' here, but it seems that 'j' must
            // be equal to 'work' when it exits the Lanczos iterations, so I'm
            // just going to use that instead. 
            res = R_F * BU.row(work - 1);

            int n_converged = 0;
            if (iter > 0) {
                n_converged = convtest.run(BS, res);
                if (n_converged >= number) {
                    converged = true;
                    break;
                }
            }
            convtest.set_last(BS);

            // Setting 'k'.
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
            V.col(k) = F;

            Wtmp.leftCols(k).noalias() = W * BU.leftCols(k);
            W.leftCols(k) = Wtmp.leftCols(k);

            B.setZero(work, work);
            for (int l = 0; l < k; ++l) {
                B(l, l) = BS[l];
                B(l, k) = res[l];
            }
        }

        outS.resize(number);
        outS = svd.singularValues().head(number);

        outU.resize(mat.rows(), number);
        outU.noalias() = W * svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV.noalias() = V * svd.matrixV().leftCols(number);

        return std::make_pair(converged, iter + 1);
    }
};

}

#endif
