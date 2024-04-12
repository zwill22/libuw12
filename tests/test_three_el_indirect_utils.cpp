//
// Created by Zack Williams on 27/03/2024.
//

#include "catch.hpp"
#include "setup_integrals.hpp"
#include "three_electron/indirect_utils.hpp"

using uw12_test::setup_abs_projector;
using uw12_test::setup_base_integrals;
using uw12_test::setup_base_integrals_direct;

constexpr size_t n_ao = 5;
constexpr size_t n_df = 11;
constexpr size_t n_ri = 17;
constexpr auto W_seed = uw12_test::seed + 1;
constexpr auto V_seed = uw12_test::seed;

const auto abs_projectors = uw12_test::setup_abs_projector(n_ao, n_ri);

TEST_CASE("Test three electron term - Indirect Utils (Indirect energy)") {
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  for (int i = 0; i < 2; ++i) {
    const auto [W, V] = uw12_test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, W_seed
    );

    const auto e = uw12::three_el::indirect_3el_energy(W, V, abs_projectors);

    const auto e2 = uw12::three_el::indirect_3el_energy(V, W, abs_projectors);

    CHECK_THAT(e, Catch::Matchers::WithinRel(e2, uw12_test::eps));
    n_occ.push_back(3);
    n_active.push_back(2);
  }
}

TEST_CASE("Test three electron term - Indirect Utils (Indirect Fock)") {
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  const auto W2 = uw12::linalg::random_pd(n_df, W_seed);
  const auto W3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, W_seed);
  const auto W3_ri = uw12::linalg::random(n_ao * n_ri, n_df, W_seed);

  const auto W_base = uw12::integrals::BaseIntegrals(W3, W2, W3_ri);
  const auto W_base_direct = setup_base_integrals_direct(W_base, W2);

  const auto V2 = uw12::linalg::random_pd(n_df, V_seed);
  const auto V3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, V_seed);
  const auto V3_ri = uw12::linalg::random(n_ao * n_ri, n_df, V_seed);

  const auto V_base = uw12::integrals::BaseIntegrals(V3, V2, V3_ri);
  const auto V_base_direct = setup_base_integrals_direct(V_base, V2);

  const auto abs_projectors = setup_abs_projector(n_ao, n_ri);

  for (size_t i = 0; i < 2; ++i) {
    const auto n_spin = uw12::utils::spin_channels(n_occ);
    REQUIRE(uw12::utils::spin_channels(n_active) == n_spin);

    const auto [Co, active_Co] =
        uw12_test::setup_orbitals(n_occ, n_active, n_ao);

    const auto W = uw12::integrals::Integrals(W_base, Co, active_Co);
    const auto V = uw12::integrals::Integrals(V_base, Co, active_Co);

    const auto W_direct =
        uw12::integrals::Integrals(W_base_direct, Co, active_Co);
    const auto V_direct =
        uw12::integrals::Integrals(V_base_direct, Co, active_Co);

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      const auto fock =
          uw12::three_el::indirect_3el_fock_matrix(W, V, abs_projectors, sigma);
      REQUIRE(uw12::linalg::n_rows(fock) == n_ao);
      REQUIRE(uw12::linalg::n_cols(fock) == n_ao);

      const auto fock2 = uw12::three_el::indirect_3el_fock_matrix(
          W_direct, V_direct, abs_projectors, sigma
      );

      CHECK(uw12::linalg::nearly_equal(fock, fock2, uw12_test::epsilon));
    }
    n_occ.push_back(3);
    n_active.push_back(2);
  }
}