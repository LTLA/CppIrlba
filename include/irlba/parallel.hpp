#ifndef IRLBA_PARALLEL_HPP
#define IRLBA_PARALLEL_HPP

#include "utils.hpp"
#include <vector>
#include "Eigen/Dense"

#ifndef IRLBA_CUSTOM_PARALLEL
#ifdef CUSTOM_PARALLEL
#define IRLBA_CUSTOM_PARALLEL CUSTOM_PARALLEL
#endif
#endif

/**
 * @file parallel.hpp
 *
 * @brief Sparse matrix class with parallelized multiplication.
 */

namespace irlba {

/**
 * @brief Sparse matrix with customizable parallelization.
 *
 * This provides an alternative to `Eigen::SparseMatrix` for parallelized multiplication of compressed sparse matrices.
 * Unlike Eigen, this implementation is able to parallelize when the multiplication does not align well with the storage layout,
 * e.g., multiplication of a compressed sparse column matrix by a dense vector on the right hand side.
 * On construction, it also pre-allocates the rows and/or columns to each thread, aiming to balance the number of non-zero elements that each thread needs to process.
 * All subsequent multiplications can then use these allocations, which is useful for cases like `Irlba` where the cost of pre-allocation is abrogated by repeated multiplication calls.
 *
 * Some cursory testing indicates that the performance of this implementation is comparable to Eigen for OpenMP-based parallelization.
 * However, the real purpose of this class is to support custom parallelization schemes in cases where OpenMP is not available.
 * This is achieved by defining the `CUSTOM_PARALLEL` or `IRLBA_CUSTOM_PARALLEL` macros to the names of functions implementing a custom scheme.
 * Any such function should accept three arguments - the number of jobs, a lambda that accepts a start and end index on the job range, and the number of workers.
 * It should then distribute the jobs across workers by calling the lambda on any combination of non-verlapping intervals that covers the entire job range.
 *
 * @tparam column_major Whether the matrix should be in compressed sparse column format.
 * If `false`, this is assumed to be in row-major format.
 * @tparam ValueArray Array class containing numeric values for the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam IndexArray Array class containing integer values for the indices of the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam PointerArray Array class containing integer values for the pointers to the row/column boundaries.
 * Should support a read-only `[]` operator.
 */
template<
    bool column_major = true, 
    class ValueArray = std::vector<double>, 
    class IndexArray = std::vector<int>, 
    class PointerArray = std::vector<size_t> 
>
class ParallelSparseMatrix {
public:
    /**
     * Default constructor.
     * This object cannot be used for any operations.
     */
    ParallelSparseMatrix() {}

    /**
     * @param nr Number of rows.
     * @param nc Number of columns.
     * @param x Values of non-zero elements.
     * @param i Indices of non-zero elements.
     * Each entry corresponds to a value in `x`, so `i` should be an array of length equal to `x`.
     * If `column_major = true`, `i` should contain row indices; otherwise it should contain column indices.
     * @param p Pointers to the start of each column (if `column_major = true`) or row (otherwise).
     * This should be an ordered array of length equal to the number of columns or rows plus 1.
     * @param nt Number of threads to be used for multiplication.
     *
     * `x`, `i` and `p` represent the typical components of a compressed sparse column/row matrix.
     * Thus, entries in `i` should be sorted within each column/row, where the boundaries between columns/rows are defined by `p`.
     */
    ParallelSparseMatrix(size_t nr, size_t nc, std::vector<double> x, std::vector<int> i, std::vector<size_t> p, int nt) : 
        primary_dim(column_major ? nc : nr), 
        secondary_dim(column_major ? nr : nc), 
        nthreads(nt), 
        values(std::move(x)), 
        indices(std::move(i)), 
        ptrs(std::move(p)) 
    {
        if (nthreads > 1) {
            fragment_threads();
        }
    }

    /**
     * @return Number of rows in the matrix.
     */
    auto rows() const { 
        if constexpr(column_major) {
            return secondary_dim;
        } else {
            return primary_dim;
        }
    }

    /**
     * @return Number of columns in the matrix.
     */
    auto cols() const { 
        if constexpr(column_major) {
            return primary_dim;
        } else {
            return secondary_dim;
        }
    }

    /**
     * @return Non-zero elements in compressed sparse row/column format.
     * This is equivalent to `x` in the constructor.
     */
    const ValueArray& get_values() const {
        return values;
    }

    /**
     * @return Indices of non-zero elements, equivalent to `i` in the constructor.
     * These are row or column indices for compressed sparse row or column format, respectively, depending on `column_major`.
     */
    const IndexArray& get_indices() const {
        return indices;
    }

    /**
     * @return Pointers to the start of each row or column, equivalent to `p` in the constructor.
     */
    const PointerArray& get_pointers() const {
        return ptrs;
    }
private:
    size_t primary_dim, secondary_dim;
    int nthreads;
    ValueArray values;
    IndexArray indices;
    PointerArray ptrs;

    typedef typename std::remove_const<typename std::remove_reference<decltype(indices[0])>::type>::type IndexType;
    typedef typename std::remove_const<typename std::remove_reference<decltype(ptrs[0])>::type>::type PointerType;

private:
    std::vector<size_t> primary_starts, primary_ends;
    std::vector<std::vector<PointerType> > secondary_nonzero_starts;

    void fragment_threads() {
        auto total_nzeros = ptrs[primary_dim]; // last element - not using back() to avoid an extra requirement on PointerArray.
        PointerType per_thread = std::ceil(static_cast<double>(total_nzeros) / nthreads);
        
        // Splitting columns across threads so each thread processes the same number of nonzero elements.
        primary_starts.resize(nthreads);
        primary_ends.resize(nthreads);
        {
            size_t primary_counter = 0;
            PointerType sofar = per_thread;
            for (int t = 0; t < nthreads; ++t) {
                primary_starts[t] = primary_counter;
                while (primary_counter < primary_dim && ptrs[primary_counter + 1] <= sofar) {
                    ++primary_counter;
                }
                primary_ends[t] = primary_counter;
                sofar += per_thread;
            }
        }

        // Splitting rows across threads so each thread processes the same number of nonzero elements.
        secondary_nonzero_starts.resize(nthreads + 1, std::vector<PointerType>(primary_dim));
        {
            std::vector<PointerType> secondary_nonzeros(secondary_dim);
            for (PointerType i = 0; i < total_nzeros; ++i) { // don't using range for loop to avoid an extra requirement on IndexArray.
                ++(secondary_nonzeros[indices[i]]);
            }
            
            std::vector<IndexType> secondary_ends(nthreads);
            IndexType secondary_counter = 0;
            PointerType sofar = per_thread;
            PointerType cum_rows = 0;

            for (int t = 0; t < nthreads; ++t) {
                while (secondary_counter < secondary_dim && cum_rows <= sofar) {
                    cum_rows += secondary_nonzeros[secondary_counter];
                    ++secondary_counter;
                }
                secondary_ends[t] = secondary_counter;
                sofar += per_thread;
            }

            for (size_t c = 0; c < primary_dim; ++c) {
                auto primary_start = ptrs[c], primary_end = ptrs[c + 1];
                secondary_nonzero_starts[0][c] = primary_start;

                auto s = primary_start;
                for (int thread = 0; thread < nthreads; ++thread) {
                    while (s < primary_end && indices[s] < secondary_ends[thread]) { 
                        ++s; 
                    }
                    secondary_nonzero_starts[thread + 1][c] = s;
                }
            }
        }
    }

private:
    template<class Right>
    void indirect_multiply(const Right& rhs, Eigen::VectorXd& output) const {
        output.setZero();
        if (nthreads == 1) {
            for (size_t c = 0; c < primary_dim; ++c) {
                column_sum_product(ptrs[c], ptrs[c + 1], rhs.coeff(c), output); 
            }
            return;
        }

#ifndef IRLBA_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(nthreads, [&](int first, int last) -> void {
        for (int t = first; t < last; ++t) {
#endif

            auto starts = secondary_nonzero_starts[t];
            auto ends = secondary_nonzero_starts[t + 1];
            for (size_t c = 0; c < primary_dim; ++c) {
                column_sum_product(starts[c], ends[c], rhs.coeff(c), output);
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return;
    }

    void column_sum_product(PointerType start, PointerType end, double val, Eigen::VectorXd& output) const {
        for (PointerType s = start; s < end; ++s) {
            output.coeffRef(indices[s]) += values[s] * val;
        }
    }

private:
    template<class Right>
    void direct_multiply(const Right& rhs, Eigen::VectorXd& output) const {
        if (nthreads == 1) {
            for (size_t c = 0; c < primary_dim; ++c) {
                output.coeffRef(c) = column_dot_product(c, rhs);
            }
            return;
        }

#ifndef IRLBA_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(nthreads, [&](int first, int last) -> void {
        for (int t = first; t < last; ++t) {
#endif

            auto curstart = primary_starts[t];
            auto curend = primary_ends[t];
            for (size_t c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product(c, rhs);
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return;
    }

    template<class Right>
    double column_dot_product(size_t c, const Right& rhs) const {
        PointerType primary_start = ptrs[c], primary_end = ptrs[c + 1];
        double dot = 0;
        for (PointerType s = primary_start; s < primary_end; ++s) {
            dot += values[s] * rhs.coeff(indices[s]);
        }
        return dot;
    }

public:
    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& output) const {
        if constexpr(column_major) {
            indirect_multiply(rhs, output);
        } else {
            direct_multiply(rhs, output);
        }
    }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& output) const {
        if constexpr(column_major) {
            direct_multiply(rhs, output);
        } else {
            indirect_multiply(rhs, output);
        }
    }

public:
    /**
     * @return A dense copy of the sparse matrix data.
     */
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output(rows(), cols());
        output.setZero();

        if constexpr(column_major) {
            for (size_t c = 0; c < cols(); ++c) {
                size_t col_start = ptrs[c], col_end = ptrs[c + 1];
                for (size_t s = col_start; s < col_end; ++s) {
                    output.coeffRef(indices[s], c) = values[s];
                }
            }
        } else {
            for (size_t r = 0; r < rows(); ++r) {
                size_t row_start = ptrs[r], row_end = ptrs[r + 1];
                for (size_t s = row_start; s < row_end; ++s) {
                    output.coeffRef(r, indices[s]) = values[s];
                }
            }
        }

        return output;
    }
};

}

#endif
