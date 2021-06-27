#include <gtest/gtest.h>
#include "irlba/irlba.hpp"
#include "irlba/utils.hpp"
#include "Eigen/Dense"

class IrlbaTester : public ::testing::Test {
protected:
    void SetUp () {
        A = Eigen::MatrixXd(nr, nc);

        irlba::NormalSampler norm(42);
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                A(j, i) = norm();
            }
        }
    }

    size_t nr = 20, nc = 10;
    Eigen::MatrixXd A, V;
};

TEST_F(IrlbaTester, Basic) {
    irlba::Irlba irb;

    irlba::NormalSampler norm(50);
    Eigen::MatrixXd U, V;
    Eigen::VectorXd S;
    irb.run(A, false, false, norm, U, V, S);
}
