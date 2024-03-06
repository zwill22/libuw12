//
// Created by Zack Williams on 06/03/2024.
//

#include "../src/utils/parallel.hpp"
#include "../src/utils/utils.hpp"
#include "catch.hpp"
#include "numerical_fock.hpp"

TEST_CASE("Test Fock - Test numerical Fock") {
  using uw12::utils::DensityMatrix;

  constexpr auto n_ao = 5;
  constexpr auto delta = 1e-4;

  const auto D1 = uw12::linalg::random(n_ao, n_ao, test::seed);
  const DensityMatrix D_mat = {0.5 * (D1 + uw12::linalg::transpose(D1))};

  const auto energy_function = [](const DensityMatrix& D) {
    double total = 0;
    for (const auto& Do : D) {
      const auto n_row = uw12::linalg::n_rows(Do);
      const auto n_col = uw12::linalg::n_cols(Do);
      for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
        for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
          const auto val = uw12::linalg::elem(Do, row_idx, col_idx);

          total += val * val;
        }
      }
    }
    return total;
  };

  const auto num_fock =
      fock::numerical_fock_matrix(energy_function, D_mat, delta);

  for (size_t sigma = 0; sigma < 1; ++sigma) {
    const auto analytic_fock = 2 * D_mat[sigma];
    CHECK(uw12::linalg::nearly_equal(
        analytic_fock, num_fock[sigma], test::epsilon
    ));
  }
}
