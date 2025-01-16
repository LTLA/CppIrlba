#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Sparse"

#include <thread>
#include <random>
#include <vector>

int main(int argc, char* argv[]) {
    CLI::App app{"Sparse matrix multiplication test"};
    int nrow;
    app.add_option("-r,--rows", nrow, "Number of rows")->default_val(10000);
    int ncol;
    app.add_option("-c,--columns", ncol, "Number of columns")->default_val(10000);
    int nthreads;
    app.add_option("-t,--threads", nthreads, "Number of threads")->default_val(4);
    double density;
    app.add_option("-d,--density", density, "Density of non-zero elements")->default_val(0.1);
    int iterations;
    app.add_option("-i,--iter", iterations, "Multiplication iterations")->default_val(100);
    CLI11_PARSE(app, argc, argv);

    // Simulating a sparse matrix.
    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> indptrs;
    std::vector<Eigen::Triplet<double> > triplets;
    size_t estimated = (static_cast<double>(nrow) * static_cast<double>(ncol) * density * 1.05);
    values.reserve(estimated);
    indices.reserve(estimated);
    triplets.reserve(estimated);
    indptrs.resize(ncol + 1);

    std::mt19937_64 rng(/* seed = */ (nrow + ncol * nthreads) * density);
    std::uniform_real_distribution udist;
    std::normal_distribution ndist;
    for (int c = 0; c < ncol; ++c) {
        for (int r = 0; r < nrow; ++r) {
            if (udist(rng) < density) {
                auto val = ndist(rng);
                values.push_back(val);
                indices.push_back(r);
                triplets.emplace_back(r, c, val);
            }
        }
        indptrs[c + 1] = values.size();
    }

    // Forcing OpenMP to use the specified number of threads, if relevant.
    irlba::EigenThreadScope scope(nthreads);

    // Constructing our matrices. We don't time the constructions to make life
    // simpler; besides, most of these matrices are constructed once and used
    // for hundreds of multiplications in IRLBA, so we don't really care too
    // much about the construction time.
    irlba::ParallelSparseMatrix imat(nrow, ncol, std::move(values), std::move(indices), std::move(indptrs), true, nthreads);

    Eigen::SparseMatrix<double> emat(nrow, ncol);
    emat.setFromTriplets(triplets.begin(), triplets.end());
    emat.makeCompressed(); triplets.clear(); triplets.shrink_to_fit(); // freeing up the memory.

    Eigen::VectorXd rhs(ncol);
    for (auto& r : rhs) {
        r = ndist(rng);
    }

    Eigen::VectorXd custom_out(nrow);
    ankerl::nanobench::Bench().run("custom", [&](){
        auto workspace = imat.workspace();
        for (int i = 0; i < iterations; ++i) {
            imat.multiply(rhs, workspace, custom_out);
        }
    });

    Eigen::VectorXd eigen_out(nrow);
    ankerl::nanobench::Bench().run("eigen", [&](){
        for (int i = 0; i < iterations; ++i) {
            eigen_out = emat * rhs;
        }
    });

    if (!custom_out.isApprox(eigen_out)) {
        throw std::runtime_error("differences in the results for regular multiplication");
    }

    Eigen::VectorXd arhs(nrow);
    for (auto& r : arhs) {
        r = ndist(rng);
    }

    Eigen::VectorXd custom_adj_out(ncol);
    ankerl::nanobench::Bench().run("custom-adjoint", [&](){
        auto aworkspace = imat.adjoint_workspace();
        for (int i = 0; i < iterations; ++i) {
            imat.adjoint_multiply(arhs, aworkspace, custom_adj_out);
        }
    });

    Eigen::VectorXd eigen_adj_out(ncol);
    ankerl::nanobench::Bench().run("eigen-adjoint", [&](){
        for (int i = 0; i < iterations; ++i) {
            eigen_adj_out = emat.adjoint() * arhs;
        }
    });

    if (!custom_adj_out.isApprox(eigen_adj_out)) {
        throw std::runtime_error("differences in the results for regular multiplication");
    }

    return 0;
}
