#ifndef IRLBA_MATRIX_SPARSE_HPP
#define IRLBA_MATRIX_SPARSE_HPP

#include <vector>
#include <memory>
#include <cstddef>

#include "../utils.hpp"
#include "../parallel.hpp"
#include "interface.hpp"

#include "Eigen/Dense"
#include "sanisizer/sanisizer.hpp"

#ifndef IRLBA_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

/**
 * @file sparse.hpp
 * @brief Sparse matrix with parallelized multiplication.
 */

namespace irlba {

/**
 * @cond
 */
template<class ValueArray_, class IndexArray_, class PointerArray_ >
class ParallelSparseMatrixCore {
public:
    typedef I<decltype(std::declval<PointerArray_>()[0])> PointerType;

public:
    ParallelSparseMatrixCore(
        Eigen::Index nrow,
        Eigen::Index ncol,
        ValueArray_ x,
        IndexArray_ i,
        PointerArray_ p,
        bool column_major,
        int num_threads
    ) : 
        my_primary_dim(column_major ? ncol : nrow), 
        my_secondary_dim(column_major ? nrow : ncol), 
        my_num_threads(num_threads), 
        my_values(std::move(x)), 
        my_indices(std::move(i)), 
        my_ptrs(std::move(p)),
        my_column_major(column_major)
    {
        if (num_threads > 1) {
            const auto total_nzeros = my_ptrs[my_primary_dim]; // last element - not using back() to avoid an extra requirement on PointerArray.
            const PointerType per_thread_floor = total_nzeros / my_num_threads;
            const int per_thread_extra = total_nzeros % my_num_threads;

            // Note that we do a lot of 't + 1' incrementing, but this is guaranteed to fit in an int because 't + 1 <= my_num_threads'.
            // We just need 'my_num_threads + 1' to fit in a size_t for the various vector allocations.
            const auto nthreads_p1 = sanisizer::sum<std::size_t>(my_num_threads, 1);
            
            // Splitting primary dimension elements across threads so each thread processes the same number of nonzero elements.
            {
                sanisizer::resize(my_primary_boundaries, nthreads_p1);

                Eigen::Index primary_counter = 0;
                PointerType sofar = 0;
                for (int t = 0; t < my_num_threads; ++t) {
                    sofar += per_thread_floor + (t < per_thread_extra); // first few threads might get an extra element to deal with the remainder.
                    while (primary_counter < my_primary_dim && my_ptrs[primary_counter + 1] <= sofar) {
                        ++primary_counter;
                    }
                    my_primary_boundaries[t + 1] = primary_counter;
                }
            }

            // Splitting secondary dimension elements across threads so each thread processes the same number of nonzero elements.
            {
                auto secondary_nonzeros = sanisizer::create<std::vector<PointerType> >(my_secondary_dim);
                for (PointerType i = 0; i < total_nzeros; ++i) { // don't using range for loop to avoid an extra requirement on IndexArray.
                    ++(secondary_nonzeros[my_indices[i]]);
                }

                sanisizer::resize(my_secondary_boundaries, nthreads_p1);
                Eigen::Index secondary_counter = 0;
                PointerType sofar = 0;
                PointerType cum_secondary = 0;
                for (int t = 0; t < my_num_threads; ++t) {
                    sofar += per_thread_floor + (t < per_thread_extra); // first few threads might get an extra element to deal with the remainder.
                    while (secondary_counter < my_secondary_dim && cum_secondary <= sofar) {
                        cum_secondary += secondary_nonzeros[secondary_counter];
                        ++secondary_counter;
                    }
                    my_secondary_boundaries[t + 1] = secondary_counter;
                }

                sanisizer::resize(my_secondary_nonzero_boundaries, nthreads_p1);
                for (auto& starts : my_secondary_nonzero_boundaries) {
                    sanisizer::resize(starts, my_primary_dim);
                }

                for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                    const auto primary_start = my_ptrs[c], primary_end = my_ptrs[c + 1];
                    my_secondary_nonzero_boundaries[0][c] = primary_start;
                    auto s = primary_start;
                    for (int thread = 0; thread < my_num_threads; ++thread) {
                        const auto limit = my_secondary_boundaries[thread + 1];
                        while (s < primary_end && static_cast<Eigen::Index>(my_indices[s]) < limit) { // cast is safe as my_indices[s] < my_secondary_dim.
                            ++s; 
                        }
                        my_secondary_nonzero_boundaries[thread + 1][c] = s;
                    }
                }
            }
        }
    }

private:
    Eigen::Index my_primary_dim, my_secondary_dim;
    int my_num_threads;

    ValueArray_ my_values;
    IndexArray_ my_indices;
    PointerArray_ my_ptrs;
    bool my_column_major;

    std::vector<Eigen::Index> my_primary_boundaries;

    // In theory, it is possible that the IndexArray type (i.e., IndexType) is not large enough to hold my_secondary_dim.
    // So while it is safe to cast from the IndexType to Eigen::Index, it is not safe to go the other way;
    // hence we use an Eigen::Index to hold the secondary boundaries as the last entry is equal to my_secondary_dim.
    std::vector<Eigen::Index> my_secondary_boundaries;

    std::vector<std::vector<PointerType> > my_secondary_nonzero_boundaries;

public:
    Eigen::Index rows() const { 
        if (my_column_major) {
            return my_secondary_dim;
        } else {
            return my_primary_dim;
        }
    }

    Eigen::Index cols() const { 
        if (my_column_major) {
            return my_primary_dim;
        } else {
            return my_secondary_dim;
        }
    }

    const ValueArray_& get_values() const {
        return my_values;
    }

    const IndexArray_& get_indices() const {
        return my_indices;
    }

    const PointerArray_& get_pointers() const {
        return my_ptrs;
    }

    int get_num_threads() const {
        return my_num_threads;
    }

    bool get_column_major() const {
        return my_column_major;
    }

    const std::vector<Eigen::Index>& get_primary_boundaries() const {
        return my_primary_boundaries;
    }

    const std::vector<Eigen::Index>& get_secondary_boundaries() const {
        return my_secondary_boundaries;
    }

    const std::vector<std::vector<PointerType> >& get_secondary_nonzero_boundaries() const {
        return my_secondary_nonzero_boundaries;
    }

public:
    template<typename EigenVector_>
    void indirect_multiply(const EigenVector_& rhs, std::vector<std::vector<typename EigenVector_::Scalar> >& thread_buffers, EigenVector_& output) const {
        if (my_num_threads == 1) {
            output.setZero();
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

        parallelize(my_num_threads, [&](int t) -> void {
            const auto secondary_start = my_secondary_boundaries[t];
            const auto secondary_end = my_secondary_boundaries[t + 1];
            const auto secondary_len = secondary_end - secondary_start;

            // Using a separate buffer for the other threads to avoid false
            // sharing. On first use, each buffer is allocated within each
            // thread to give malloc a chance of using thread-specific arenas.
            typename EigenVector_::Scalar* optr;
            if (t != 0) {
                auto& curbuffer = thread_buffers[t - 1];
                sanisizer::resize(curbuffer, secondary_len);
                optr = curbuffer.data();
            } else {
                optr = output.data() + secondary_start;
            }
            std::fill_n(optr, secondary_len, static_cast<typename EigenVector_::Scalar>(0));

            const auto& nz_starts = my_secondary_nonzero_boundaries[t];
            const auto& nz_ends = my_secondary_nonzero_boundaries[t + 1];
            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                const auto nz_start = nz_starts[c];
                const auto nz_end = nz_ends[c];
                const auto val = rhs.coeff(c);
                for (PointerType s = nz_start; s < nz_end; ++s) {
                    optr[my_indices[s] - secondary_start] += my_values[s] * val;
                }
            }

            if (t != 0) {
                std::copy_n(optr, secondary_len, output.data() + secondary_start);
            }
        });

        return;
    }

public:
    template<typename EigenVector_>
    void direct_multiply(const EigenVector_& rhs, EigenVector_& output) const {
        if (my_num_threads == 1) {
            for (Eigen::Index c = 0; c < my_primary_dim; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }
            return;
        }

        parallelize(my_num_threads, [&](int t) -> void {
            const auto curstart = my_primary_boundaries[t];
            const auto curend = my_primary_boundaries[t + 1];
            for (auto c = curstart; c < curend; ++c) {
                output.coeffRef(c) = column_dot_product<typename EigenVector_::Scalar>(c, rhs);
            }
        });

        return;
    }

private:
    template<typename Scalar_, class EigenVector_>
    Scalar_ column_dot_product(Eigen::Index p, const EigenVector_& rhs) const {
        PointerType primary_start = my_ptrs[p], primary_end = my_ptrs[p + 1];
        Scalar_ dot = 0;
        for (PointerType s = primary_start; s < primary_end; ++s) {
            dot += my_values[s] * rhs.coeff(my_indices[s]);
        }
        return dot;
    }
};
/**
 * @endcond
 */

/**
 * @brief Workspace for multiplication of a `ParallelSparseMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam ValueArray_ Array class containing the non-zero values, see `ParallelSparseMatrix`.
 * @tparam IndexArray_ Array class containing indices of non-zero elements, see `ParallelSparseMatrix`.
 * @tparam PointerArray_ Array class containing the pointers to the row/column boundaries, see `ParallelSparseMatrix`.
 *
 * Typically constructed by `ParallelSparseMatrix::new_known_workspace()`.
 */
template<class EigenVector_, class ValueArray_, class IndexArray_, class PointerArray_ >
class ParallelSparseWorkspace final : public Workspace<EigenVector_> {
public:
    /**
     * @cond
     */
    ParallelSparseWorkspace(const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& core) :
        my_core(core)
    {
        if (my_core.get_num_threads() > 1 && my_core.get_column_major()) {
            my_thread_buffers.resize(my_core.get_num_threads() - 1);
        }
    }
    /**
     * @endcond
     */

private:
    const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& my_core;
    std::vector<std::vector<typename EigenVector_::Scalar> > my_thread_buffers;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        if (my_core.get_column_major()) {
            my_core.indirect_multiply(right, my_thread_buffers, output);
        } else {
            my_core.direct_multiply(right, output);
        }
    }
};

/**
 * @brief Workspace for multiplication of a transposed `ParallelSparseMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam ValueArray_ Array class containing the non-zero values, see `ParallelSparseMatrix`.
 * @tparam IndexArray_ Array class containing indices of non-zero elements, see `ParallelSparseMatrix`.
 * @tparam PointerArray_ Array class containing the pointers to the row/column boundaries, see `ParallelSparseMatrix`.
 *
 * Typically constructed by `ParallelSparseMatrix::new_known_adjoint_workspace()`.
 */
template<class EigenVector_, class ValueArray_, class IndexArray_, class PointerArray_ >
class ParallelSparseAdjointWorkspace final : public AdjointWorkspace<EigenVector_> {
public:
    /**
     * @cond
     */
    ParallelSparseAdjointWorkspace(const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& core) :
        my_core(core)
    {
        if (my_core.get_num_threads() > 1 && !my_core.get_column_major()) {
            my_thread_buffers.resize(my_core.get_num_threads() - 1);
        }
    }
    /**
     * @endcond
     */

private:
    const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& my_core;
    std::vector<std::vector<typename EigenVector_::Scalar> > my_thread_buffers;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        if (my_core.get_column_major()) {
            my_core.direct_multiply(right, output);
        } else {
            my_core.indirect_multiply(right, my_thread_buffers, output);
        }
    }
};

/**
 * @brief Workspace for realizing a `ParallelSparseMatrix`.
 *
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * @tparam ValueArray_ Array class containing the non-zero values, see `ParallelSparseMatrix`.
 * @tparam IndexArray_ Array class containing indices of non-zero elements, see `ParallelSparseMatrix`.
 * @tparam PointerArray_ Array class containing the pointers to the row/column boundaries, see `ParallelSparseMatrix`.
 *
 * Typically constructed by `ParallelSparseMatrix::new_known_realize_workspace()`.
 */
template<class EigenMatrix_, class ValueArray_, class IndexArray_, class PointerArray_ >
class ParallelSparseRealizeWorkspace final : public RealizeWorkspace<EigenMatrix_> {
public:
    /**
     * @cond
     */
    ParallelSparseRealizeWorkspace(const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& core) :
        my_core(core)
    {}
    /**
     * @endcond
     */

private:
    const ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_>& my_core;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        const auto nr = my_core.rows(), nc = my_core.cols();
        buffer.resize(nr, nc);
        buffer.setZero();

        const auto& ptrs = my_core.get_pointers();
        const auto& indices = my_core.get_indices();
        const auto& values = my_core.get_values();

        typedef I<decltype(std::declval<PointerArray_>()[0])> PointerType;
        if (my_core.get_column_major()) {
            for (Eigen::Index c = 0; c < nc; ++c) {
                PointerType col_start = ptrs[c], col_end = ptrs[c + 1];
                for (PointerType s = col_start; s < col_end; ++s) {
                    buffer.coeffRef(indices[s], c) = values[s];
                }
            }
        } else {
            for (Eigen::Index r = 0; r < nr; ++r) {
                PointerType row_start = ptrs[r], row_end = ptrs[r + 1];
                for (PointerType s = row_start; s < row_end; ++s) {
                    buffer.coeffRef(r, indices[s]) = values[s];
                }
            }
        }

        return buffer;
    }
};

/**
 * @brief Sparse matrix with customizable parallelization.
 *
 * This provides an alternative to `Eigen::SparseMatrix` for parallelized multiplication of compressed sparse matrices.
 * Unlike **Eigen**'s sparse matrix, this implementation is able to parallelize when the multiplication does not align well with the storage layout,
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
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of multiplication operations.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * Typically of the same scalar type as `EigenVector_`.
 * @tparam ValueArray_ Array class containing numeric values for the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam IndexArray_ Array class containing integer values for the indices of the non-zero values.
 * Should support a read-only `[]` operator.
 * @tparam PointerArray_ Array class containing integer values for the pointers to the row/column boundaries.
 * Should support a read-only `[]` operator.
 */
template<
    class EigenVector_,
    class EigenMatrix_,
    class ValueArray_, 
    class IndexArray_,
    class PointerArray_
>
class ParallelSparseMatrix final : public Matrix<EigenVector_, EigenMatrix_> {
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
     * @param num_threads Number of threads to be used for multiplication.
     *
     * `x`, `i` and `p` represent the typical components of a compressed sparse column/row matrix.
     * Thus, entries in `i` should be sorted within each column/row, where the boundaries between columns/rows are defined by `p`.
     */
    ParallelSparseMatrix(Eigen::Index nrow, Eigen::Index ncol, ValueArray_ x, IndexArray_ i, PointerArray_ p, bool column_major, int num_threads) : 
        my_core(nrow, ncol, std::move(x), std::move(i), std::move(p), column_major, num_threads)
    {}

private:
    ParallelSparseMatrixCore<ValueArray_, IndexArray_, PointerArray_> my_core;

public:
    /**
     * @return Number of rows in the matrix.
     */
    Eigen::Index rows() const { 
        return my_core.rows();
    }

    /**
     * @return Number of columns in the matrix.
     */
    Eigen::Index cols() const { 
        return my_core.cols();
    }

    /**
     * @return Non-zero elements in compressed sparse row/column format.
     * This is equivalent to `x` in the constructor.
     */
    const ValueArray_& get_values() const {
        return my_core.get_values();
    }

    /**
     * @return Indices of non-zero elements, equivalent to `i` in the constructor.
     * These are row or column indices for compressed sparse row or column format, respectively, depending on `column_major`.
     */
    const IndexArray_& get_indices() const {
        return my_core.get_indices();
    }

    /**
     * @return Pointers to the start of each row or column, equivalent to `p` in the constructor.
     */
    const PointerArray_& get_pointers() const {
        return my_core.get_pointers();
    }

    /**
     * Type of the elements inside a `PointerArray_`.
     */
    typedef I<decltype(std::declval<PointerArray_>()[0])> PointerType;

    /**
     * This should only be called if `num_threads > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads plus one.
     * The `t`-th and `t + 1`-th entries specifies the first and one-past-the-last elements along the primary dimension
     * (e.g., column for `column_major = true`) that each thread operates on.
     */
    const std::vector<Eigen::Index>& get_primary_boundaries() const {
        return my_core.get_primary_boundaries();
    }

    /**
     * This should only be called if `num_threads > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads plus one.
     * The `t`-th and `t + 1`-th entries specifies the first and one-past-the-last elements along the secondary dimension
     * (e.g., row for `column_major = true`) that each thread operates on.
     */
    const std::vector<Eigen::Index>& get_secondary_boundaries() const {
        return my_core.get_secondary_boundaries();
    }

    /**
     * This should only be called if `num_threads > 1` in the constructor, otherwise it will not be initialized.
     *
     * @return Vector of length equal to the number of threads plus one.
     * Each inner vector is of length equal to the extent of the primary dimension (e.g., number of columns for `column_major = true`).
     * For thread `t`, `secondary_nonzero_boundaries[t][i]` is the first non-zero element to be processed by this thread in the primary dimension element `i`,
     * while `boundaries[t + 1][i]` is one-past-the-last non-zero element to be processed. 
     * This is guaranteed to contain all and only non-zero elements with indices `i` where `secondary_boundaries[t] <= i < secondary_boundaries[t + 1]`.
     */
    const std::vector<std::vector<PointerType> >& get_secondary_nonzero_boundaries() const {
        return my_core.get_secondary_nonzero_boundaries();
    }

public:
    std::unique_ptr<Workspace<EigenVector_> > new_workspace() const {
        return new_known_workspace();
    }

    std::unique_ptr<AdjointWorkspace<EigenVector_> > new_adjoint_workspace() const {
        return new_known_adjoint_workspace();
    }

    std::unique_ptr<RealizeWorkspace<EigenMatrix_> > new_realize_workspace() const {
        return new_known_realize_workspace();
    }

public:
    /**
     * Overrides `Matrix::new_known_workspace()` to enable devirtualization.
     */
    std::unique_ptr<ParallelSparseWorkspace<EigenVector_, ValueArray_, IndexArray_, PointerArray_> > new_known_workspace() const {
        return std::make_unique<ParallelSparseWorkspace<EigenVector_, ValueArray_, IndexArray_, PointerArray_> >(my_core);
    }

    /**
     * Overrides `Matrix::new_known_adjoint_workspace()` to enable devirtualization.
     */
    std::unique_ptr<ParallelSparseAdjointWorkspace<EigenVector_, ValueArray_, IndexArray_, PointerArray_> > new_known_adjoint_workspace() const {
        return std::make_unique<ParallelSparseAdjointWorkspace<EigenVector_, ValueArray_, IndexArray_, PointerArray_> >(my_core);
    }

    /**
     * Overrides `Matrix::new_known_realize_workspace()` to enable devirtualization. 
     */
    std::unique_ptr<ParallelSparseRealizeWorkspace<EigenMatrix_, ValueArray_, IndexArray_, PointerArray_> > new_known_realize_workspace() const {
        return std::make_unique<ParallelSparseRealizeWorkspace<EigenMatrix_, ValueArray_, IndexArray_, PointerArray_> >(my_core);
    }

};

}

#endif
