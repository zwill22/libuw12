//
// Created by Zack Williams on 27/03/2024.
//

#include "../src/three_electron/indirect_utils.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

constexpr size_t n_ao = 5;
constexpr size_t n_df = 11;
constexpr size_t n_ri = 17;

const auto abs_projectors = test::setup_abs_projector(n_ao, n_ri);

TEST_CASE("Test three electron term - Indirect Utils (Indirect energy)") {
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

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

TEST_CASE("Test three electron term - Indirect Utils (Indirect Fock)") {
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  for (size_t i = 0; i < 2; ++i) {
    const auto n_spin = uw12::utils::spin_channels(n_occ);
    REQUIRE(uw12::utils::spin_channels(n_active) == n_spin);

    const auto [W, V] = test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, test::seed - 1
    );

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      const auto fock =
          uw12::three_el::indirect_3el_fock_matrix(W, V, abs_projectors, sigma);
      REQUIRE(uw12::linalg::n_rows(fock) == n_ao);
      REQUIRE(uw12::linalg::n_cols(fock) == n_ao);
      
      const auto fock2 =
          uw12::three_el::indirect_3el_fock_matrix(W, V, abs_projectors, sigma);

      CHECK(uw12::linalg::nearly_equal(fock, fock2, test::epsilon
      ));

    }
    n_occ.push_back(3);
    n_active.push_back(2);
  }
}