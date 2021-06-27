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
    bool used = false;

    Eigen::VectorXd initV, res;

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
        W.resize(mat.rows(), work);
        V.resize(mat.cols(), work);
        res.resize(work);

        if (initV.size() == V.rows()) {
            V.col(0) = initV;
        } else {
            for (Eigen::Index i = 0; i < V.rows(); ++i) {
                V(i, 0) = norm();
            }
        }

        B.resize(work, work);
        if (used) {
            B.setZero(work, work);
        } else {
            used = true;
        }

        bool converged = false;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(work, work, Eigen::ComputeThinU | Eigen::ComputeThinV);

        for (int iter = 0; iter < maxit; ++iter) {
            lp.run(mat, W, V, B, center, scale, norm, work, iter==0);

            svd.compute(B);
            const auto& BS = svd.singularValues();
            const auto& BU = svd.matrixU();
            const auto& BV = svd.matrixV();

            if (iter > 0) {
                if (B(work-1, work-1) == 0) { // a.k.a. the final value of 'S' from the Lanczos iterations.
                    return std::make_pair(true, iter + 1);
                }
                
                // Not really sure what this is, but here we are.
                double R_F = lp.finalF().norm();

                // irlba's original code uses 'j - 1' here, but it seems that 'j' must
                // be equal to 'work' when it exits the Lanczos iterations, so I'm
                // just going to use that instead. 
                res = R_F * BU.row(work - 1);
                if (convtest.run(number, BS, res)) {
                    return std::make_pair(true, iter + 1);
                }
            }
            convtest.set_last(BS);
            for (auto x : BS) { std::cout << x << " "; }
            std::cout << std::endl;

            // Update the first column in V only, as everything else
            // gets overwritten as part of the Lanczos process anyway.
            V.col(0) = V * BV.col(0);
            std::cout << iter << std::endl;
        }

        outS.resize(number);
        outS = svd.singularValues();

        outU.resize(mat.rows(), number);
        outU.noalias() = W * svd.matrixU().leftCols(number);

        outV.resize(mat.cols(), number);
        outV.noalias() = V * svd.matrixV().leftCols(number);

        return std::make_pair(false, maxit);
    }
};

}

#endif
