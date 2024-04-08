//
// Created by Zack Williams on 27/03/2024.
//

#include "../src/three_electron/indirect_utils.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

TEST_CASE("Test three electron term - Indirect Utils") {
  constexpr size_t n_ao = 5;
  constexpr size_t n_df = 11;
  constexpr size_t n_ri = 17;
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  const auto abs_projectors = test::setup_abs_projector(n_ao, n_ri);

  for (int i = 0; i < 2; ++i) {
    const auto [W, V] = test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, test::seed - 1
    );

    const auto e = uw12::three_el::indirect_3el_energy(W, V, abs_projectors);

    const auto e2 = uw12::three_el::indirect_3el_energy(V, W, abs_projectors);

    CHECK_THAT(e, Catch::Matchers::WithinRel(e2, test::eps));
    n_occ.push_back(3);
    n_active.push_back(2);
  }
}
