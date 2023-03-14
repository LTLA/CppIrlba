#ifndef IRLBA_PARALLEL_HPP
#define IRLBA_PARALLEL_HPP

#include "utils.hpp"
#include <vector>
#include "Eigen/Dense"

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
 * This is achieved by defining `IRLBA_CUSTOM_PARALLEL` macro to the name of a function implementing a custom scheme.
 * Such a function should accept two arguments - an integer specifying the number of threads, and a lambda that accepts a thread number.
 * It should then loop over the number of threads and launch one job for each thread via the lambda.
 * Once all threads are complete, the function should return.
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

public:
    /**
     * This should only be called if `nt > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads,
     * specifying the first dimension along the primary extent (e.g., column for `column_major = true`) that each thread works on.
     */
    const std::vector<size_t>& get_primary_starts() const {
        return primary_starts;
    }

    /**
     * This should only be called if `nt > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads,
     * specifying the one-past-the-last dimension along the primary extent (e.g., column for `column_major = true`) that each thread works on.
     */
    const std::vector<size_t>& get_primary_ends() const {
        return primary_ends;
    }

    /**
     * Type of the elements inside a `PointerArray`.
     */
    typedef typename std::remove_const<typename std::remove_reference<decltype(ptrs[0])>::type>::type PointerType;

    /**
     * This should only be called if `nt > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads plus one.
     * Each inner vector is of length equal to the size of the primary extent (e.g., number of columns for `column_major = true`).
     * For thread `i`, the vectors `i` and `i + 1` define the ranges of non-zero elements assigned to that thread within each primary dimension.
     * This is guaranteed to contain all and only non-zero elements with indices in a contiguous range of secondary dimensions.
     */
    const std::vector<std::vector<PointerType> >& get_secondary_nonzero_starts() const {
        return secondary_nonzero_starts;
    }

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
                auto start = ptrs[c];
                auto end = ptrs[c + 1];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(indices[s]) += values[s] * val;
                }
            }
            return;
        }

#ifndef IRLBA_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(nthreads, [&](int t) -> void {
#endif

            const auto& starts = secondary_nonzero_starts[t];
            const auto& ends = secondary_nonzero_starts[t + 1];
            for (size_t c = 0; c < primary_dim; ++c) {
                auto start = starts[c];
                auto end = ends[c];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(indices[s]) += values[s] * val;
                }
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        });
#endif

        return;
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
        IRLBA_CUSTOM_PARALLEL(nthreads, [&](int t) -> void {
#endif

            auto curstart = primary_starts[t];
            auto curend = primary_ends[t];
            for (size_t c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product(c, rhs);
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        });
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
     * Workspace type for `multiply()`.
     * Currently this is a placeholder.
     */
    typedef bool Workspace;

    /**
     * @return Workspace for use in `multiply()`.
     */
    bool workspace() const {
        return false;
    }

    /**
     * Workspace type for `adjoint_multiply()`.
     * Currently this is a placeholder.
     */
    typedef bool AdjointWorkspace;

    /**
     * @return Workspace for use in `adjoint_multiply()`.
     */
    bool adjoint_workspace() const {
        return false;
    }

public:
    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param work The return value of `workspace()`.
     * @param[out] output The output vector to store the matrix product.
     * This is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Workspace& work, Eigen::VectorXd& output) const {
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
     * @param work The return value of `adjoint_workspace()`.
     * @param[out] output The output vector to store the matrix product.
     * This is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& output) const {
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

/**
 * @brief Restrict the number of available threads for Eigen.
 *
 * @details
 * Creating an instance of this class will call `Eigen::setNbThreads()` to control the number of available OpenMP threads in Eigen operations.
 * Destruction will then reset the number of available threads to its prior value.
 *
 * If OpenMP is available and `IRLBA_CUSTOM_PARALLEL`, Eigen is restricted to just one thread when an instance of this class is created.
 * This is done to avoid using OpenMP when a custom parallelization scheme has already been specified.
 *
 * If OpenMP is not available, this class has no effect.
 */ 
class EigenThreadScope {
public:
#ifdef _OPENMP
    /**
     * @cond
     */
    EigenThreadScope(int n) : previous(Eigen::nbThreads()) {
#ifndef IRLBA_CUSTOM_PARALLEL
        Eigen::setNbThreads(n);
#else
        Eigen::setNbThreads(1);
#endif
    }
    /**
     * @endcond
     */
#else
    /**
     * @param n Number of threads to be used by Eigen.
     */
    EigenThreadScope(int n) {}
#endif

    /**
     * @cond
     */
    EigenThreadScope(const EigenThreadScope&) = delete;
    EigenThreadScope(EigenThreadScope&&) = delete;
    EigenThreadScope& operator=(const EigenThreadScope&) = delete;
    EigenThreadScope& operator=(EigenThreadScope&&) = delete;

    ~EigenThreadScope() { 
#ifdef _OPENMP
        Eigen::setNbThreads(previous);
#endif
    }
    /**
     * @endcond
     */
private:
    int previous;
};




}

#endif
