//
// Created by Zack Williams on 23/02/2024.
//

#include "../src/integrals/integral_functions.hpp"
#include "catch.hpp"

using namespace uw12_test;

TEST_CASE("Test integrals - Test integral functions") {
  const std::vector<size_t> df_shell_sizes = {1, 3, 5, 7, 9, 11};
  std::vector<size_t> df_offsets = {0};
  for (size_t idx = 0; idx < df_shell_sizes.size() - 1; ++idx) {
    df_offsets.push_back(df_offsets[idx] + df_shell_sizes[idx]);
  }

  constexpr auto n_ao = 5;
  constexpr auto n_row = n_ao * (n_ao + 1) / 2;

  size_t n_df = 0;
  for (const auto df : df_shell_sizes) {
    n_df += df;
  }

  const auto three_idx_fn = [n_row, &df_shell_sizes](const int A
                            ) -> uw12::linalg::Mat {
    return uw12::linalg::ones(n_row, df_shell_sizes[A]) * A;
  };

  // TODO implement exception handling across openmp threads
#ifndef USE_OMP
  REQUIRE_THROWS(
      uw12::integrals::coulomb_3idx(three_idx_fn, df_offsets, n_row + 1, n_df)
  );
  REQUIRE_THROWS(
      uw12::integrals::coulomb_3idx(three_idx_fn, df_offsets, n_row - 1, n_df)
  );
  REQUIRE_THROWS(
      uw12::integrals::coulomb_3idx(three_idx_fn, df_offsets, n_row, n_df - 1)
  );
#endif

  const auto result =
      uw12::integrals::coulomb_3idx(three_idx_fn, df_offsets, n_row, n_df);
  REQUIRE(uw12::linalg::n_rows(result) == n_row);
  REQUIRE(uw12::linalg::n_cols(result) == n_df);

  int shell_idx = 0;
  for (auto col_idx = 0; col_idx < n_df; ++col_idx) {
    for (auto idx = 0; idx < df_offsets.size(); ++idx) {
      if (col_idx >= df_offsets[idx]) {
        shell_idx = idx;
      }
    }

    const uw12::linalg::Mat col = uw12::linalg::col(result, col_idx);
    const uw12::linalg::Mat target = uw12::linalg::ones(n_row, 1) * shell_idx;

    CHECK(uw12::linalg::nearly_equal(col, target, epsilon));
  }
}
