#include <gtest/gtest.h>
#include "irlba/lanczos.hpp"
#include "irlba/utils.hpp"
#include "Eigen/Dense"
#include "NormalSampler.h"

class LanczosTester : public ::testing::Test {
protected:
    void SetUp () {
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

    size_t nr = 20, nc = 10, work = 7;
    Eigen::MatrixXd A, W, V, B;
};

TEST_F(LanczosTester, Basic) {
    irlba::LanczosBidiagonalization y;

    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, W, V, B, false, false, eng, init);

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

TEST_F(LanczosTester, Center) {
    irlba::LanczosBidiagonalization y;

    NormalSampler norm(50);
    Eigen::VectorXd center(A.cols());
    for (auto& c : center) { c = norm(); }

    Eigen::MatrixXd W2 = W;
    Eigen::MatrixXd V2 = V;
    Eigen::MatrixXd B2 = B;
    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, W2, V2, B2, center, false, eng, init);

    Eigen::MatrixXd Acopy = A;
    for (size_t i = 0; i < A.cols(); ++i) {
        for (size_t j = 0; j < A.rows(); ++j) {
            Acopy(j, i) -= center(i);
        }
    }

    Eigen::MatrixXd W3 = W;
    Eigen::MatrixXd V3 = V;
    Eigen::MatrixXd B3 = B;
    std::mt19937_64 eng2(50);
    auto init2 = y.initialize(A);
    y.run(Acopy, W3, V3, B3, false, false, eng2, init2);

    // Numerically equivalent values.
    for (size_t i = 0; i < W2.cols(); ++i) {
        for (size_t j = 0; j < W2.rows(); ++j) {
            EXPECT_FLOAT_EQ(W2(j, i), W3(j, i));
        }
    }

    for (size_t i = 0; i < V2.cols(); ++i) {
        for (size_t j = 0; j < V2.rows(); ++j) {
            EXPECT_FLOAT_EQ(V2(j, i), V3(j, i));
        }
    }

    for (size_t i = 0; i < B2.cols(); ++i) {
        for (size_t j = 0; j < B2.rows(); ++j) {
            EXPECT_FLOAT_EQ(B2(j, i), B3(j, i));
        }
    }
}

TEST_F(LanczosTester, CenterAndScale) {
    irlba::LanczosBidiagonalization y;

    NormalSampler norm(50);
    Eigen::VectorXd center(A.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(A.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    Eigen::MatrixXd W2 = W;
    Eigen::MatrixXd V2 = V;
    Eigen::MatrixXd B2 = B;
    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, W2, V2, B2, center, scale, eng, init);

    Eigen::MatrixXd Acopy = A;
    for (size_t i = 0; i < A.cols(); ++i) {
        for (size_t j = 0; j < A.rows(); ++j) {
            Acopy(j, i) -= center(i);
            Acopy(j, i) /= scale(i);
        }
    }

    Eigen::MatrixXd W3 = W;
    Eigen::MatrixXd V3 = V;
    Eigen::MatrixXd B3 = B;
    std::mt19937_64 eng2(50);
    auto init2 = y.initialize(A);
    y.run(Acopy, W3, V3, B3, false, false, eng2, init2);

    // Numerically equivalent values.
    for (size_t i = 0; i < W2.cols(); ++i) {
        for (size_t j = 0; j < W2.rows(); ++j) {
            EXPECT_FLOAT_EQ(W2(j, i), W3(j, i));
        }
    }

    for (size_t i = 0; i < V2.cols(); ++i) {
        for (size_t j = 0; j < V2.rows(); ++j) {
            EXPECT_FLOAT_EQ(V2(j, i), V3(j, i));
        }
    }

    for (size_t i = 0; i < B2.cols(); ++i) {
        for (size_t j = 0; j < B2.rows(); ++j) {
            EXPECT_FLOAT_EQ(B2(j, i), B3(j, i));
        }
    }
}

TEST_F(LanczosTester, Restart) {
    irlba::LanczosBidiagonalization y;

    Eigen::MatrixXd subW = W.leftCols(3); 
    Eigen::MatrixXd subV = V.leftCols(3); 
    Eigen::MatrixXd subB = B.topLeftCorner(3,3); 
    std::mt19937_64 eng(50);
    auto init = y.initialize(A);
    y.run(A, subW, subV, subB, false, false, eng, init);

    Eigen::MatrixXd copyW(nr, work);
    copyW.leftCols(3) = subW;
    Eigen::MatrixXd copyV(nc, work);
    copyV.leftCols(3) = subV;
    Eigen::MatrixXd copyB(work, work);
    copyB.setZero();
    copyB.topLeftCorner(3,3) = subB;
    copyV.col(3) = init.residuals() / init.residuals().norm();
    y.run(A, copyW, copyV, copyB, false, false, eng, init, 3); //restarting from start = 3.

    // Numerically equivalent to a full compuation... except for B, where the
    // restart loses one of the superdiagonal elements (which is normally 
    // filled in by the residual error in the IRLBA loop, see Equation 3.6).
    std::mt19937_64 eng2(50);
    auto init2 = y.initialize(A);
    y.run(A, W, V, B, false, false, eng2, init2);

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
