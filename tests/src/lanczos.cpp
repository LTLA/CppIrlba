#include <gtest/gtest.h>
#include "irlba/lanczos.hpp"
#include "irlba/utils.hpp"
#include "Eigen/Dense"
#include "NormalSampler.h"

class LanczosTester : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nr = std::get<0>(param);
        nc = std::get<1>(param);
        work = std::get<2>(param);

        A = Eigen::MatrixXd(nr, nc);
        W = Eigen::MatrixXd(nr, work);
        V = Eigen::MatrixXd(nc, work);
        B = Eigen::MatrixXd(work, work);
        B.setZero();

        NormalSampler norm(42);
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                A(j, i) = norm();
            }
        }

        for (size_t i = 0; i < nc; ++i) {
            V(i) = norm();
        }
        V.col(0) /= V.col(0).norm();
    }

    size_t nr, nc, work;
    Eigen::MatrixXd A, W, V, B;
};

TEST_P(LanczosTester, Basic) {
    assemble(GetParam());
    irlba::LanczosBidiagonalization y;

    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, W, V, B, eng, init);

    // Check that vectors in W are self-orthogonal.
    Eigen::MatrixXd Wcheck = W.adjoint() * W;
    for (size_t i = 0; i < work; ++i) {
        for (size_t j = 0; j < work; ++j) {
            if (i==j) {
                EXPECT_FLOAT_EQ(Wcheck(i, j), 1);
            } else {
                EXPECT_TRUE(std::abs(Wcheck(i, j)) < 0.00000000001);
            }
        }
    }

    // Check that vectors in V are self-orthogonal.
    Eigen::MatrixXd Vcheck = V.adjoint() * V;
    for (size_t i = 0; i < work; ++i) {
        for (size_t j = 0; j < work; ++j) {
            if (i==j) {
                EXPECT_FLOAT_EQ(Vcheck(i, j), 1);
            } else {
                EXPECT_TRUE(std::abs(Vcheck(i, j)) < 0.00000000001);
            }
        }
    }
}

TEST_P(LanczosTester, Restart) {
    assemble(GetParam());
    irlba::LanczosBidiagonalization y;

    int mid = work / 2;
    Eigen::MatrixXd subW = W.leftCols(mid); 
    Eigen::MatrixXd subV = V.leftCols(mid); 
    Eigen::MatrixXd subB = B.topLeftCorner(mid,mid); 
    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, subW, subV, subB, eng, init);

    Eigen::MatrixXd copyW(nr, work);
    copyW.leftCols(mid) = subW;
    Eigen::MatrixXd copyV(nc, work);
    copyV.leftCols(mid) = subV;
    Eigen::MatrixXd copyB(work, work);
    copyB.setZero();
    copyB.topLeftCorner(mid,mid) = subB;
    copyV.col(mid) = init.residuals() / init.residuals().norm();
    y.run(A, copyW, copyV, copyB, eng, init, mid); //restarting from start = mid.

    // Numerically equivalent to a full compuation... except for B, where the
    // restart loses one of the superdiagonal elements (which is normally 
    // filled in by the residual error in the IRLBA loop, see Equation 3.6).
    std::mt19937_64 eng2(50);
    auto init2 = y.initialize(A);
    y.run(A, W, V, B, eng2, init2);

    for (size_t i = 0; i < copyW.cols(); ++i) {
        for (size_t j = 0; j < copyW.rows(); ++j) {
            EXPECT_FLOAT_EQ(copyW(j, i), W(j, i));
        }
    }

    for (size_t i = 0; i < copyV.cols(); ++i) {
        for (size_t j = 0; j < copyV.rows(); ++j) {
            EXPECT_FLOAT_EQ(copyV(j, i), V(j, i));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    LanczosTests,
    LanczosTester,
    ::testing::Combine(
        ::testing::Values(10, 20, 30), // number of rows
        ::testing::Values(10, 20, 30), // number of columns
        ::testing::Values(3, 5, 7) // workspace
    )
);
