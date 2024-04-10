//
// Created by Zack Williams on 09/04/2024.
//

#include "../src/uw12.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

constexpr size_t n_ao = 5;
constexpr size_t n_df = 8;
constexpr size_t n_ri = 14;

constexpr auto W_seed = test::seed - 1;
constexpr auto V_seed = test::seed;
constexpr auto WV_seed = test::seed + 1;

const auto abs_projectors = test::setup_abs_projector(n_ao, n_ri);

auto setup_occupations(const std::vector<size_t>& n_occ) {
  uw12::utils::Occupations occ;
  for (const auto n : n_occ) {
    occ.push_back(uw12::linalg::ones(n));
  }

  return occ;
}

TEST_CASE("Test UW12") {
  const auto W = test::setup_base_integrals(n_ao, n_df, n_ri, W_seed);
  const auto V = test::setup_base_integrals(n_ao, n_df, n_ri, V_seed);
  const auto WV = test::setup_base_integrals(n_ao, n_df, W_seed);

  const std::vector<size_t> n_occ = {3};
  const std::vector<size_t> n_active = {2};

  const auto n_spin = uw12::utils::spin_channels(n_occ);
  const auto [occ_orb, active_orb] =
      test::setup_orbitals(n_occ, n_active, n_ao);

  const auto occ = setup_occupations(n_occ);

  const auto [fock_os, e_os] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 1.0, 0, 3
  );

  const auto [fock_ss, e_ss] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 0.0, 1.0, 3
  );

  const auto [fock, e] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 1.0, 0.5, 3
  );

  CHECK_THAT(e_os + 0.5 * e_ss, Catch::Matchers::WithinRel(e, test::epsilon));

  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const uw12::linalg::Mat mat1 = fock_os[sigma] + 0.5 * fock_ss[sigma];
    const auto mat2 = fock[sigma];
    CHECK(uw12::linalg::nearly_equal(mat1, mat2, test::epsilon));
  }
}
