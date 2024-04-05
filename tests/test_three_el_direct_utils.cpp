//
// Created by Zack Williams on 27/03/2024.
//

#include "../src/three_electron/direct_utils.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

auto setup_abs_projector(
    const size_t n_ao, const size_t n_ri, const int S_seed = test::seed
) {
  const auto s = uw12::linalg::random_pd(n_ao + n_ri, S_seed);

  return uw12::three_el::ri::calculate_abs_projectors(s, n_ao, n_ri);
}

TEST_CASE("Test three electron term - Direct Utils (X_AB)") {
  std::vector<size_t> n_occ = {3};
  std::vector<size_t> n_active = {2};

  for (int i = 0; i < 2; ++i) {
    constexpr size_t n_ao = 10;
    constexpr size_t n_df = 18;
    constexpr size_t n_ri = 25;

    const auto n_spin = n_occ.size();
    REQUIRE(n_active.size() == n_spin);

    const auto [W, V] = test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, test::seed - 1
    );

    const auto abs_projectors = setup_abs_projector(n_ao, n_ri);

    const auto Xab = uw12::three_el::calculate_xab(W,V, abs_projectors);

    REQUIRE(Xab.size() == n_spin);

    for (const auto & X: Xab) {
      CHECK(uw12::linalg::n_rows(X) == n_df);
      CHECK(uw12::linalg::n_cols(X) == n_df);
    }
    n_occ.push_back(2);
    n_active.push_back(1);
  }
}