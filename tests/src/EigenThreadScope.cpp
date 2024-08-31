#include <gtest/gtest.h>

#include "irlba/parallel.hpp"

TEST(EigenThreadScope, Basic) {
    int old = Eigen::nbThreads();

    {
        irlba::EigenThreadScope scope(10);

#ifdef IRLBA_CUSTOM_PARALLEL
#ifdef IRLBA_CUSTOM_PARALLEL_USES_OPENMP
        EXPECT_EQ(Eigen::nbThreads(), 10);
#else
        EXPECT_EQ(Eigen::nbThreads(), 1);
#endif
#else
#ifdef SUBPAR_NO_OPENMP_SIMPLE
        EXPECT_EQ(Eigen::nbThreads(), 1);
#else
        EXPECT_EQ(Eigen::nbThreads(), 10);
#endif
#endif
    }

    EXPECT_EQ(Eigen::nbThreads(), old);
}
