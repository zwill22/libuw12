//
// Created by Zack Williams on 27/11/2020.
//

#ifndef UW12_PARALLEL_HPP
#define UW12_PARALLEL_HPP

#include <functional>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#endif

namespace uw12::parallel {
    /// \brief Run a for loop in parallel
    ///
    /// Calls a function which takes an integer as argument and calls it for each
    /// integer between start and stop (exclusive).
    ///
    /// Useful for looping through arrays and calculating each entry in parllel.
    /// Should not be used to manipulate the same result multiple times, such as summation. If this
    /// is required, the ::parallel_sum function should be used instead.
    ///
    /// \param start Initial integer value
    /// \param stop Final integer value (exclusive)
    /// \param func Function to call
    /// \param parallel Whether to run in parallel
    inline void parallel_for(size_t start, size_t stop,
                             const std::function<void(size_t)> &func,
                             bool parallel = true) {
#ifdef USE_TBB
        if (parallel) {
            const auto range = tbb::blocked_range(start, stop);

            const auto parallel_fn = [&func](const tbb::blocked_range<size_t> &block) {
                for (size_t i = block.begin(); i != block.end(); ++i) {
                    func(i);
                }
            };

            tbb::parallel_for(range, parallel_fn);
        } else
#elif USE_OMP
        if (parallel) {
#pragma omp parallel
            {
#pragma omp for
                for (size_t i = start; i < stop; ++i) {
                    func(i);
                }
            }
        } else
#endif
        {
            for (size_t i = start; i < stop; ++i) {
                func(i);
            }
        }
    }

    /// \brief Construct an object in parallel
    ///
    /// Construct an object of type `ReturnType` in parallel for each input integer
    /// in range [start, stop).
    ///
    /// \tparam ReturnType
    /// \param start Initial value
    /// \param stop Final value (excluded)
    /// \param identity Identity value
    /// \param func Parallel function
    /// \param parallel Run in parallel
    ///
    /// \return Object of type `ReturnType`
    template<typename ReturnType>
    ReturnType parallel_sum(const size_t start, const size_t stop,
                            const ReturnType &identity,
                            const std::function<ReturnType(size_t)> &func,
                            const bool parallel = true) {
#ifdef USE_TBB
        if (parallel) {
            const auto red_fn = [](ReturnType left, const ReturnType &right) -> ReturnType {
                left += right;
                return left;
            };

            const auto range = tbb::blocked_range(start, stop);

            const auto parallel_fn = [&func](const tbb::blocked_range<size_t> &blocked_range,
                                             ReturnType result) -> ReturnType {
                for (size_t i = blocked_range.begin(); i != blocked_range.end(); i++) {
                    result += func(i);
                }
                return result;
            };

            return tbb::parallel_reduce(range, identity, parallel_fn, red_fn);
        } else
#elif USE_OMP
        if (parallel) {
            ReturnType result(identity);

#pragma omp declare reduction(sum:ReturnType       \
                              : omp_out += omp_in) \
    initializer(omp_priv(omp_orig))
#pragma omp parallel reduction(sum : result)
            {
#pragma omp for
                for (size_t i = start; i < stop; ++i) {
                    result += func(i);
                }
            }

            return result;
        } else
#endif
        {
            ReturnType result(identity);
            for (size_t i = start; i < stop; ++i) {
                result += func(i);
            }
            return result;
        }
    }

    /// \brief Construct an object in parallel over two sets of integers
    ///
    /// \tparam ReturnType
    /// \param start1 Initial value for loop 1
    /// \param stop1 Final value (excluded) for loop 1
    /// \param start2 Initial value for loop 2
    /// \param stop2 Final value (excluded) for loop 2
    /// \param identity Identity value
    /// \param func Parallel function
    /// \param parallel Run in parallel
    ///
    /// \return Object of type `ReturnType`
    template<typename ReturnType>
    ReturnType parallel_sum_2d(
        const size_t start1, const size_t stop1, const size_t start2,
        const size_t stop2, const ReturnType &identity,
        const std::function<ReturnType(size_t, size_t)> &func,
        const bool parallel = true) {
        const auto parallel_fn = [&func, start2, stop2, &identity](const size_t i) -> ReturnType {
            ReturnType result(identity);
            for (size_t j = start2; j < stop2; ++j) {
                result += func(i, j);
            }
            return result;
        };

        return parallel_sum<ReturnType>(start1, stop1, identity, parallel_fn,
                                        parallel);
    }

    /// \brief Function to avoid deadlocks for mutex in parallel
    ///
    /// \tparam Func
    /// \param func Parallel mutex protected code
    template<typename Func>
    void isolate(Func &&func) {
#ifdef USE_TBB
        tbb::this_task_arena::isolate([&func] { func(); });
#else
        func();
#endif
    }
} // namespace uw12::parallel


#endif  // UW12_PARALLEL_HPP
