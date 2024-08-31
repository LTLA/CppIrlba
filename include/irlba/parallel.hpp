#ifndef IRLBA_PARALLEL_HPP
#define IRLBA_PARALLEL_HPP

#include "utils.hpp"
#include <vector>
#include "Eigen/Dense"

#ifndef IRLBA_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

/**
 * @file parallel.hpp
 *
 * @brief Classes for parallelized multiplication.
 */

namespace irlba {

/**
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function to execute each task.
 *
 * @param num_tasks Number of tasks.
 * This is equal to the number of threads in the context of `ParallelSparseMatrix`.
 * @param run_task Function to execute each task within its own worker.
 *
 * By default, this is an alias to `subpar::parallelize_simple()`.
 * However, if the `IRLBA_CUSTOM_PARALLEL` function-like macro is defined, it is called instead. 
 * Any user-defined macro should accept the same arguments as `subpar::parallelize_simple()`.
 */
template<typename Task_, class Run_>
void parallelize(Task_ num_tasks, Run_ run_task) {
#ifndef IRLBA_CUSTOM_PARALLEL
    // Use cases here don't allocate or throw, so nothrow_ = true is fine.
    subpar::parallelize_simple<true>(num_tasks, std::move(run_task));
#else
    IRLBA_CUSTOM_PARALLEL(num_tasks, run_task);
#endif
}

/**
 * @brief Sparse matrix with customizable parallelization.
 *
 * This provides an alternative to `Eigen::SparseMatrix` for parallelized multiplication of compressed sparse matrices.
 * Unlike Eigen, this implementation is able to parallelize when the multiplication does not align well with the storage layout,
 * e.g., multiplication of a compressed sparse column matrix by a dense vector on the right hand side.
 * On construction, it also pre-allocates the rows and/or columns to each thread, aiming to balance the number of non-zero elements that each thread needs to process.
 * All subsequent multiplications can then use these allocations, which is useful for `compute()` where the cost of pre-allocation is abrogated by repeated multiplication calls.
 *
 * Some cursory testing indicates that the performance of this implementation is comparable to Eigen for OpenMP-based parallelization.
 * However, the real purpose of this class is to support custom parallelization schemes in cases where OpenMP is not available.
 * This is achieved by defining `IRLBA_CUSTOM_PARALLEL` macro to the name of a function implementing a custom scheme.
 * Such a function should accept two arguments - an integer specifying the number of threads, and a lambda that accepts a thread number.
 * It should then loop over the number of threads and launch one job for each thread via the lambda.
 * Once all threads are complete, the function should return.
 *
 * This class satisfies the `MockMatrix` interface and implements all of its methods/typedefs.
 *
 * @tparam ValueArray_ Array class containing numeric values for the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam IndexArray_ Array class containing integer values for the indices of the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam PointerArray_ Array class containing integer values for the pointers to the row/column boundaries.
 * Should support a read-only `[]` operator.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<
    class ValueArray_ = std::vector<double>, 
    class IndexArray_ = std::vector<int>, 
    class PointerArray_ = std::vector<size_t>,
    class EigenVector_ = Eigen::VectorXd
>
class ParallelSparseMatrix {
public:
    /**
     * Default constructor.
     * This object cannot be used for any operations.
     */
    ParallelSparseMatrix() {}

    /**
     * @param nrow Number of rows.
     * @param ncol Number of columns.
     * @param x Values of non-zero elements.
     * @param i Indices of non-zero elements.
     * Each entry corresponds to a value in `x`, so `i` should be an array of length equal to `x`.
     * If `column_major = true`, `i` should contain row indices; otherwise it should contain column indices.
     * @param p Pointers to the start of each column (if `column_major = true`) or row (otherwise).
     * This should be an ordered array of length equal to the number of columns or rows plus 1.
     * @param column_major Whether the matrix should be in compressed sparse column format.
     * If `false`, this is assumed to be in row-major format.
     * @param nthreads Number of threads to be used for multiplication.
     *
     * `x`, `i` and `p` represent the typical components of a compressed sparse column/row matrix.
     * Thus, entries in `i` should be sorted within each column/row, where the boundaries between columns/rows are defined by `p`.
     */
    ParallelSparseMatrix(Eigen::Index nrow, Eigen::Index ncol, ValueArray_ x, IndexArray_ i, PointerArray_ p, bool column_major, int nthreads) : 
        my_primary_dim(column_major ? ncol : nrow), 
        my_secondary_dim(column_major ? nrow : ncol), 
        my_nthreads(nthreads), 
        my_values(std::move(x)), 
        my_indices(std::move(i)), 
        my_ptrs(std::move(p)),
        my_column_major(column_major)
    {
        if (nthreads > 1) {
            fragment_threads();
        }
    }

public:
    /**
     * @return Number of rows in the matrix.
     */
    Eigen::Index rows() const { 
        if (my_column_major) {
            return my_secondary_dim;
        } else {
            return my_primary_dim;
        }
    }

    /**
     * @return Number of columns in the matrix.
     */
    Eigen::Index cols() const { 
        if (my_column_major) {
            return my_primary_dim;
        } else {
            return my_secondary_dim;
        }
    }

    /**
     * @return Non-zero elements in compressed sparse row/column format.
     * This is equivalent to `x` in the constructor.
     */
    const ValueArray_& get_values() const {
        return my_values;
    }

    /**
     * @return Indices of non-zero elements, equivalent to `i` in the constructor.
     * These are row or column indices for compressed sparse row or column format, respectively, depending on `column_major`.
     */
    const IndexArray_& get_indices() const {
        return my_indices;
    }

    /**
     * @return Pointers to the start of each row or column, equivalent to `p` in the constructor.
     */
    const PointerArray_& get_pointers() const {
        return my_ptrs;
    }

public:
    /**
     * Type of the elements inside a `PointerArray_`.
     */
    typedef typename std::remove_const<typename std::remove_reference<decltype(std::declval<PointerArray_>()[0])>::type>::type PointerType;

private:
    Eigen::Index my_primary_dim, my_secondary_dim;
    int my_nthreads;
    ValueArray_ my_values;
    IndexArray_ my_indices;
    PointerArray_ my_ptrs;
    bool my_column_major;

    typedef typename std::remove_const<typename std::remove_reference<decltype(std::declval<IndexArray_>()[0])>::type>::type IndexType;

    std::vector<size_t> my_primary_starts, my_primary_ends;
    std::vector<std::vector<PointerType> > my_secondary_nonzero_starts;

public:
    /**
     * This should only be called if `nthreads > 1` in the constructor, otherwise it will not be initialized.
     bool my_column_major;
     *
     * @return Vector of length equal to the number of threads,
     * specifying the first dimension along the primary extent (e.g., column for `column_major = true`) that each thread works on.
     */
    const std::vector<size_t>& get_primary_starts() const {
        return my_primary_starts;
    }

    /**
     * This should only be called if `nthreads > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads,
     * specifying the one-past-the-last dimension along the primary extent (e.g., column for `column_major = true`) that each thread works on.
     */
    const std::vector<size_t>& get_primary_ends() const {
        return my_primary_ends;
    }

    /**
     * This should only be called if `nthreads > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads plus one.
     * Each inner vector is of length equal to the size of the primary extent (e.g., number of columns for `column_major = true`).
     * For thread `i`, the vectors `i` and `i + 1` define the ranges of non-zero elements assigned to that thread within each primary dimension.
     * This is guaranteed to contain all and only non-zero elements with indices in a contiguous range of secondary dimensions.
     */
    const std::vector<std::vector<PointerType> >& get_secondary_nonzero_starts() const {
        return my_secondary_nonzero_starts;
    }

private:

    void fragment_threads() {
        auto total_nzeros = my_ptrs[my_primary_dim]; // last element - not using back() to avoid an extra requirement on PointerArray.
        PointerType per_thread = (total_nzeros / my_nthreads) + (total_nzeros % my_nthreads > 0); // i.e., ceiling.
        
        // Splitting columns across threads so each thread processes the same number of nonzero elements.
        my_primary_starts.resize(my_nthreads);
        my_primary_ends.resize(my_nthreads);
        {
            Eigen::Index primary_counter = 0;
            PointerType sofar = per_thread;
            for (int t = 0; t < my_nthreads; ++t) {
                my_primary_starts[t] = primary_counter;
                while (primary_counter < my_primary_dim && my_ptrs[primary_counter + 1] <= sofar) {
                    ++primary_counter;
                }
                my_primary_ends[t] = primary_counter;
                sofar += per_thread;
            }
        }

        // Splitting rows across threads so each thread processes the same number of nonzero elements.
        my_secondary_nonzero_starts.resize(my_nthreads + 1, std::vector<PointerType>(my_primary_dim));
        {
            std::vector<PointerType> secondary_nonzeros(my_secondary_dim);
            for (PointerType i = 0; i < total_nzeros; ++i) { // don't using range for loop to avoid an extra requirement on IndexArray.
                ++(secondary_nonzeros[my_indices[i]]);
            }
            
            std::vector<IndexType> secondary_ends(my_nthreads);
            IndexType secondary_counter = 0;
            PointerType sofar = per_thread;
            PointerType cum_rows = 0;

            for (int t = 0; t < my_nthreads; ++t) {
                while (secondary_counter < my_secondary_dim && cum_rows <= sofar) {
                    cum_rows += secondary_nonzeros[secondary_counter];
                    ++secondary_counter;
                }
                secondary_ends[t] = secondary_counter;
                sofar += per_thread;
            }

            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                auto primary_start = my_ptrs[c], primary_end = my_ptrs[c + 1];
                my_secondary_nonzero_starts[0][c] = primary_start;

                auto s = primary_start;
                for (int thread = 0; thread < my_nthreads; ++thread) {
                    while (s < primary_end && my_indices[s] < secondary_ends[thread]) { 
                        ++s; 
                    }
                    my_secondary_nonzero_starts[thread + 1][c] = s;
                }
            }
        }
    }

private:
    void indirect_multiply(const EigenVector_& rhs, EigenVector_& output) const {
        output.setZero();

        if (my_nthreads == 1) {
            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                auto start = my_ptrs[c];
                auto end = my_ptrs[c + 1];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(my_indices[s]) += my_values[s] * val;
                }
            }
            return;
        }

        parallelize(my_nthreads, [&](int t) -> void {
            const auto& starts = my_secondary_nonzero_starts[t];
            const auto& ends = my_secondary_nonzero_starts[t + 1];
            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                auto start = starts[c];
                auto end = ends[c];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(my_indices[s]) += my_values[s] * val;
                }
            }
        });

        return;
    }

    void direct_multiply(const EigenVector_& rhs, EigenVector_& output) const {
        if (my_nthreads == 1) {
            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }
            return;
        }

        parallelize(my_nthreads, [&](int t) -> void {
            auto curstart = my_primary_starts[t];
            auto curend = my_primary_ends[t];
            for (size_t c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }
        });

        return;
    }

    template<typename Scalar_>
    Scalar_ column_dot_product(size_t c, const EigenVector_& rhs) const {
        PointerType primary_start = my_ptrs[c], primary_end = my_ptrs[c + 1];
        Scalar_ dot = 0;
        for (PointerType s = primary_start; s < primary_end; ++s) {
            dot += my_values[s] * rhs.coeff(my_indices[s]);
        }
        return dot;
    }

    /**
     * @cond
     */
    // All MockMatrix interface methods, we can ignore this.
public:
    struct Workspace {
        EigenVector_ buffer;
    };

    Workspace workspace() const {
        return Workspace();
    }

    struct AdjointWorkspace {
        EigenVector_ buffer;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace();
    }

public:
    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& output) const {
        const auto& realized_rhs = internal::realize_rhs(rhs, work.buffer);
        if (my_column_major) {
            indirect_multiply(realized_rhs, output);
        } else {
            direct_multiply(realized_rhs, output);
        }
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector_& output) const {
        const auto& realized_rhs = internal::realize_rhs(rhs, work.buffer);
        if (my_column_major) {
            direct_multiply(realized_rhs, output);
        } else {
            indirect_multiply(realized_rhs, output);
        }
    }

public:
    template<class EigenMatrix_>
    EigenMatrix_ realize() const {
        auto nr = rows(), nc = cols();
        EigenMatrix_ output(nr, nc);
        output.setZero();

        if (my_column_major) {
            for (Eigen::Index c = 0; c < nc; ++c) {
                PointerType col_start = my_ptrs[c], col_end = my_ptrs[c + 1];
                for (PointerType s = col_start; s < col_end; ++s) {
                    output.coeffRef(my_indices[s], c) = my_values[s];
                }
            }
        } else {
            for (Eigen::Index r = 0; r < nr; ++r) {
                PointerType row_start = my_ptrs[r], row_end = my_ptrs[r + 1];
                for (PointerType s = row_start; s < row_end; ++s) {
                    output.coeffRef(r, my_indices[s]) = my_values[s];
                }
            }
        }

        return output;
    }
    /**
     * @endcond
     */
};

/**
 * @brief Restrict the number of available threads for Eigen.
 *
 * @details
 * Creating an instance of this class will call `Eigen::setNbThreads()` to control the number of available OpenMP threads in Eigen operations.
 * Destruction will then reset the number of available threads to its prior value.
 *
 * If the parallelization scheme is not OpenMP, `num_threads` is ignored and the number of Eigen threads is always set to 1 when an instance of this class is created.
 * This is done to avoid unintended parallelization via OpenMP when another scheme has already been specified.
 * We assume that OpenMP is not the parallelization scheme if:
 * - `IRLBA_CUSTOM_PARALLEL` is defined (see `parallelize()`) and the `IRLBA_CUSTOM_PARALLEL_USES_OPENMP` macro is not defined.
 * - `IRLBA_CUSTOM_PARALLEL` is not defined and OpenMP was not chosen by `subpar::parallelize_simple()`.
 *
 * If OpenMP is not available, the creation/destruction of a class instance has no effect.
 */ 
class EigenThreadScope {
#ifndef _OPENMP
public:
    EigenThreadScope([[maybe_unused]] int num_threads) {}

#else
public:
    /**
     * @param num_threads Number of threads to be used by Eigen.
     */
    EigenThreadScope([[maybe_unused]] int num_threads) : my_previous(Eigen::nbThreads()) {
#ifdef IRLBA_CUSTOM_PARALLEL
#ifdef IRLBA_CUSTOM_PARALLEL_USES_OPENMP
        Eigen::setNbThreads(num_threads);
#else
        Eigen::setNbThreads(1);
#endif
#else
#ifdef SUBPAR_USES_OPENMP_SIMPLE
        Eigen::setNbThreads(num_threads);
#else
        Eigen::setNbThreads(1);
#endif
#endif
    }

private:
    int my_previous;

public:
    ~EigenThreadScope() { 
        Eigen::setNbThreads(my_previous);
    }
#endif

public:
    /**
     * @cond
     */
    EigenThreadScope(const EigenThreadScope&) = delete;
    EigenThreadScope(EigenThreadScope&&) = delete;
    EigenThreadScope& operator=(const EigenThreadScope&) = delete;
    EigenThreadScope& operator=(EigenThreadScope&&) = delete;
    /**
     * @endcond
     */
};

}

#endif
