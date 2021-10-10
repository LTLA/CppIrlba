#ifndef NORMAL_SAMPLER_H
#define NORMAL_SAMPLER_H

#include <random>

struct NormalSampler {
    NormalSampler(int seed) : engine(seed) {}
    double operator()() { return dist(engine); }
    std::mt19937_64 engine;
    std::normal_distribution<> dist;
};

#endif
