//
// Created by Zack Williams on 09/04/2024.
//

#include "uw12.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

constexpr size_t n_ao = 5;
constexpr size_t n_df = 8;
constexpr size_t n_ri = 14;

constexpr auto W_seed = uw12_test::seed - 1;
constexpr auto V_seed = uw12_test::seed;
constexpr auto WV_seed = uw12_test::seed + 1;

const auto abs_projectors = uw12_test::setup_abs_projector(n_ao, n_ri);

const auto W = uw12_test::setup_base_integrals(n_ao, n_df, n_ri, W_seed);
const auto V = uw12_test::setup_base_integrals(n_ao, n_df, n_ri, V_seed);
const auto WV = uw12_test::setup_base_integrals(n_ao, n_df, W_seed);

auto setup_occupations(const std::vector<size_t>& n_occ) {
  uw12::utils::Occupations occ;
  for (const auto n : n_occ) {
    occ.push_back(uw12::linalg::ones(n));
  }

  return occ;
}

auto form_fock_full(
    const uw12::utils::Orbitals& occ_orb,
    const uw12::utils::Occupations& occ,
    const std::vector<size_t>& n_active
) {
  return uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 1.0, 0.5
  );
}

void run_test(
    const uw12::utils::Orbitals& occ_orb,
    const uw12::utils::Occupations& occ,
    const std::vector<size_t>& n_active
) {
  const auto n_spin = uw12::utils::spin_channels(occ_orb);
  REQUIRE(uw12::utils::spin_channels(occ) == n_spin);
  REQUIRE(uw12::utils::spin_channels(n_active) == n_spin);

  const auto [fock_os, e_os] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 1.0, 0, 3
  );

  const auto [fock_ss, e_ss] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 0.0, 1.0, 3
  );

  const auto [fock, e] = uw12::form_fock(
      W, V, WV, abs_projectors, occ_orb, occ, n_active, true, true, 1.0, 0.5, 3
  );

  CHECK_THAT(
      e_os + 0.5 * e_ss, Catch::Matchers::WithinRel(e, uw12_test::epsilon)
  );

  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const uw12::linalg::Mat mat1 = fock_os[sigma] + 0.5 * fock_ss[sigma];
    const auto mat2 = fock[sigma];
    CHECK(uw12::linalg::nearly_equal(mat1, mat2, uw12_test::epsilon));
  }
}

TEST_CASE("Test UW12 - Closed shell") {
  const std::vector<size_t> n_occ = {3};
  const std::vector<size_t> n_active = {2};

  const auto [occ_orb, active_orb] =
      uw12_test::setup_orbitals(n_occ, n_active, n_ao);

  const auto occ = setup_occupations(n_occ);

  run_test(occ_orb, occ, n_active);
}

TEST_CASE("Test UW12 - Open shell") {
  const std::vector<size_t> n_occ = {4, 3};
  const std::vector<size_t> n_active = {3, 2};

  const auto [occ_orb, active_orb] =
      uw12_test::setup_orbitals(n_occ, n_active, n_ao);

  const auto occ = setup_occupations(n_occ);

  run_test(occ_orb, occ, n_active);
}
TEST_CASE("Test UW12 - Check error handling") {
  const std::vector<size_t> n_occ = {4, 3};
  const std::vector<size_t> n_active = {3, 2};

  const auto [occ_orb, active_orb] =
      uw12_test::setup_orbitals(n_occ, n_active, n_ao);

  const auto occ = setup_occupations(n_occ);

  CHECK_THROWS(form_fock_full(occ_orb, occ, {3}));
  CHECK_THROWS(form_fock_full(occ_orb, {occ[0]}, n_active));
  CHECK_THROWS(form_fock_full({occ_orb[0]}, occ, n_active));

  const auto active_occ = setup_occupations(n_active);
  CHECK_THROWS(form_fock_full(active_orb, active_occ, n_occ));
  CHECK_THROWS(form_fock_full(active_orb, occ, n_active));
}
