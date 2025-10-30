#include <gtest/gtest.h>

#include "irlba/parallel.hpp"

#ifdef IRLBA_CUSTOM_PARALLEL
#ifdef IRLBA_CUSTOM_PARALLEL_USES_OPENMP
#include "omp.h"
#endif
#else
#ifndef SUBPAR_NO_OPENMP_SIMPLE
#include "omp.h"
#endif
#endif

TEST(EigenThreadScope, Basic) {
    int old = Eigen::nbThreads();

    {
        irlba::EigenThreadScope scope(10);

#ifdef IRLBA_CUSTOM_PARALLEL
#ifdef IRLBA_CUSTOM_PARALLEL_USES_OPENMP
        EXPECT_EQ(Eigen::nbThreads(), std::min(10, omp_get_max_threads()));
#else
        EXPECT_EQ(Eigen::nbThreads(), 1);
#endif
#else
#ifdef SUBPAR_NO_OPENMP_SIMPLE
        EXPECT_EQ(Eigen::nbThreads(), 1);
#else
        EXPECT_EQ(Eigen::nbThreads(), std::min(10, omp_get_max_threads()));
#endif
#endif
    }

    EXPECT_EQ(Eigen::nbThreads(), old);
}
