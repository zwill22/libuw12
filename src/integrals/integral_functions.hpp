//
// Created by Zack Williams on 27/11/2020.
//

#ifndef UW12_INTEGRAL_FUNCTIONS_HPP
#define UW12_INTEGRAL_FUNCTIONS_HPP

#include <functional>
#include <vector>

#include "../utils/linalg.hpp"
#include "../utils/parallel.hpp"

namespace uw12::integrals {
    /// Function which returns a matrix of two-index integrals
    using TwoIndexFn = std::function<linalg::Mat()>;

    /// Function which returns a matrix of three-index integrals for a single shell
    /// of third index
    using ThreeIndexFn = std::function<linalg::Mat(size_t)>;

    /// \brief Calculate the three-index Coulomb matrix \f$(\mu\nu | w | A)\f$
    ///
    /// Use a `three_index_fn` for each density-fitting basis shell to calculate
    /// the total three-index matrix
    ///
    /// \param three_index_fn Function to calculate integrals for each basis shell
    /// \param df_offsets offsets in full matrix for each df basis shell
    /// \param n_rows number of rows of the matrix (depends on the `three_index_fn`)
    /// \param n_df number of df basis functions
    ///
    /// \return Three-index matrix
    inline linalg::Mat coulomb_3idx(
        const ThreeIndexFn &three_index_fn,
        const std::vector<size_t> &df_offsets,
        const size_t n_rows,
        const size_t n_df
    ) {
        linalg::Mat result(n_rows, n_df);

        const auto n_df_sh = df_offsets.size();

        const auto parallel_fn = [&result, &three_index_fn,
                    &df_offsets](const size_t A) {
            const auto off_a = df_offsets[A];

            const auto shell_results = three_index_fn(A);

            linalg::assign_cols(result, shell_results, off_a);
        };

        parallel::parallel_for(0, n_df_sh, parallel_fn);

        return result;
    }
} // namespace uw12::integrals

#endif  // UW12_INTEGRAL_FUNCTIONS_HPP
