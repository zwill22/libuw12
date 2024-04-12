//
// Created by Zack Williams on 06/03/2024.
//

#include "catch.hpp"
#include "numerical_fock.hpp"
#include "utils/parallel.hpp"
#include "utils/utils.hpp"

using uw12::utils::DensityMatrix;
using uw12_test::epsilon;
using uw12_test::seed;

constexpr auto delta = 1e-4;

void run_test(const DensityMatrix& D_mat) {
  const auto n_spin = D_mat.size();

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
      uw12_test::fock::numerical_fock_matrix(energy_function, D_mat, delta);

  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const uw12::linalg::Mat analytic_fock = 2 * D_mat[sigma];
    CHECK(uw12::linalg::nearly_equal(analytic_fock, num_fock[sigma], epsilon));
  }
}

TEST_CASE("Test Fock - Test numerical Fock (Closed Shell)") {
  constexpr auto n_ao = 5;

  const auto D1 = uw12::linalg::random(n_ao, n_ao, seed);
  const DensityMatrix D_mat = {0.5 * (D1 + uw12::linalg::transpose(D1))};

  run_test(D_mat);
}

TEST_CASE("Test Fock - Test numerical Fock (Open Shell)") {
  constexpr auto n_ao = 7;

  const auto Dall = uw12::linalg::random(n_ao, 2 * n_ao, seed);

  const auto D1 = uw12::linalg::head_cols(Dall, n_ao);
  REQUIRE(uw12::linalg::n_cols(D1) == n_ao);
  REQUIRE(uw12::linalg::n_rows(D1) == n_ao);

  const auto D2 = uw12::linalg::tail_cols(Dall, n_ao);
  REQUIRE(uw12::linalg::n_cols(D2) == n_ao);
  REQUIRE(uw12::linalg::n_rows(D2) == n_ao);

  const DensityMatrix D_mat = {
      0.5 * (D1 + uw12::linalg::transpose(D1)),
      0.5 * (D2 + uw12::linalg::transpose(D2))
  };

  run_test(D_mat);
}
