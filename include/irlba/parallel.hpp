#ifndef IRLBA_PARALLEL_HPP
#define IRLBA_PARALLEL_HPP

#include "utils.hpp"
#include <vector>
#include "Eigen/Dense"

/**
 * @file parallel.hpp
 *
 * @brief Classes for parallelized multiplication.
 */

namespace irlba {

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
 */
template<
    class ValueArray_ = std::vector<double>, 
    class IndexArray_ = std::vector<int>, 
    class PointerArray_ = std::vector<size_t> 
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
     * @tparam column_major Whether the matrix should be in compressed sparse column format.
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
            size_t primary_counter = 0;
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

            for (size_t c = 0; c < my_primary_dim; ++c) {
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
    template<class Right_, class EigenVector_>
    void indirect_multiply(const Right_& rhs, EigenVector_& output) const {
        output.setZero();

        if (my_nthreads == 1) {
            for (size_t c = 0; c < my_primary_dim; ++c) {
                auto start = my_ptrs[c];
                auto end = my_ptrs[c + 1];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(my_indices[s]) += my_values[s] * val;
                }
            }
            return;
        }

#ifndef IRLBA_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
#endif
        for (int t = 0; t < my_nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(my_nthreads, [&](int t) -> void {
#endif

            const auto& starts = my_secondary_nonzero_starts[t];
            const auto& ends = my_secondary_nonzero_starts[t + 1];
            for (size_t c = 0; c < my_primary_dim; ++c) {
                auto start = starts[c];
                auto end = ends[c];
                auto val = rhs.coeff(c);
                for (PointerType s = start; s < end; ++s) {
                    output.coeffRef(my_indices[s]) += my_values[s] * val;
                }
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        });
#endif

        return;
    }

    template<class Right_, class EigenVector_>
    void direct_multiply(const Right_& rhs, EigenVector_& output) const {
        if (my_nthreads == 1) {
            for (size_t c = 0; c < my_primary_dim; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }
            return;
        }

#ifndef IRLBA_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
#endif
        for (int t = 0; t < my_nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(my_nthreads, [&](int t) -> void {
#endif

            auto curstart = my_primary_starts[t];
            auto curend = my_primary_ends[t];
            for (size_t c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        });
#endif

        return;
    }

    template<typename Scalar_, class Right_>
    Scalar_ column_dot_product(size_t c, const Right_& rhs) const {
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
    typedef bool Workspace;

    bool workspace() const {
        return false;
    }

    typedef bool AdjointWorkspace;

    bool adjoint_workspace() const {
        return false;
    }

public:
    template<class Right_, class EigenVector_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& output) const {
        if (my_column_major) {
            indirect_multiply(rhs, output);
        } else {
            direct_multiply(rhs, output);
        }
    }

    template<class Right_, class EigenVector_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector_& output) const {
        if (my_column_major) {
            direct_multiply(rhs, output);
        } else {
            indirect_multiply(rhs, output);
        }
    }

public:
    template<class EigenMatrix_>
    EigenMatrix_ realize() const {
        auto nr = rows(), nc = cols();
        EigenMatrix_ output(nr, nc);
        output.setZero();

        if (my_column_major) {
            for (size_t c = 0; c < nc; ++c) {
                size_t col_start = my_ptrs[c], col_end = my_ptrs[c + 1];
                for (size_t s = col_start; s < col_end; ++s) {
                    output.coeffRef(my_indices[s], c) = my_values[s];
                }
            }
        } else {
            for (size_t r = 0; r < nr; ++r) {
                size_t row_start = my_ptrs[r], row_end = my_ptrs[r + 1];
                for (size_t s = row_start; s < row_end; ++s) {
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
 * If OpenMP is available and `IRLBA_CUSTOM_PARALLEL` is defined, Eigen is restricted to just one thread when an instance of this class is created.
 * This is done to avoid using OpenMP when a custom parallelization scheme has already been specified.
 *
 * If OpenMP is not available, this class has no effect.
 */ 
class EigenThreadScope {
public:
    /**
     * @param n Number of threads to be used by Eigen.
     */
    EigenThreadScope(int n) 
#ifdef _OPENMP
        : my_previous(Eigen::nbThreads()) {
#ifndef IRLBA_CUSTOM_PARALLEL
        Eigen::setNbThreads(n);
#else
        Eigen::setNbThreads(1);
#endif
    }
#else
    {}
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
        Eigen::setNbThreads(my_previous);
#endif
    }
    /**
     * @endcond
     */
private:
    int my_previous;
};

}

#endif
