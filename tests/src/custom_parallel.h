#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#include <cmath>
#include <vector>
#include <thread>

template<class Function>
void parallelize(size_t nthreads, Function f) {
    std::vector<std::thread> jobs;
    for (size_t w = 0; w < nthreads; ++w) {
        jobs.emplace_back(f, w);
    }
    for (auto& job : jobs) {
        job.join();
    }
}

#define CUSTOM_PARALLEL parallelize
#endif
