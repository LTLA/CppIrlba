#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#include <vector>
#include <thread>

template<class Function_>
void test_parallelize(int nthreads, Function_ f) {
    std::vector<std::thread> jobs;
    jobs.reserve(nthreads);
    for (int w = 0; w < nthreads; ++w) {
        jobs.emplace_back(f, w);
    }
    for (auto& job : jobs) {
        job.join();
    }
}

#define IRLBA_CUSTOM_PARALLEL ::test_parallelize
#endif
