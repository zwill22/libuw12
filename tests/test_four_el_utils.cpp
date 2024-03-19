//
// Created by Zack Williams on 18/03/2024.
//

#include "../src/four_electron/four_electron_utils.hpp"
#include "catch.hpp"

using test::epsilon;
using test::margin;
using test::seed;
using uw12::integrals::Integrals;

TEST_CASE("Test Four Electron Utils - Test energy spin factor") {
  using uw12::four_el::get_energy_spin_factor;

  CHECK_THROWS(get_energy_spin_factor(0, 0, 0, 0, 0));

  CHECK_THROWS(get_energy_spin_factor(1, 1, 0, 1, 0));
  CHECK_THROWS(get_energy_spin_factor(1, 0, 1, 1, 0));
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 1, 0),
      Catch::Matchers::WithinAbs(2, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 0, 1),
      Catch::Matchers::WithinAbs(2, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(3, margin)
  );

  CHECK_THROWS(get_energy_spin_factor(2, 2, 0, 1, 0));
  CHECK_THROWS(get_energy_spin_factor(2, 0, 2, 1, 0));
  CHECK_THAT(
      get_energy_spin_factor(2, 0, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(0.5, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 0, 1, 1, 0.5),
      Catch::Matchers::WithinAbs(1, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 1, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(1, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 1, 1, 1, 0.5),
      Catch::Matchers::WithinAbs(0.5, margin)
  );

  CHECK_THROWS(get_energy_spin_factor(3, 0, 0, 0, 0));
}

auto setup_base_integrals(
    const size_t n_ao, const size_t n_df, const int J_seed
) {
  const auto J2 = uw12::linalg::random_pd(n_df, J_seed);
  const auto J3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, J_seed);

  return uw12::integrals::BaseIntegrals(J3, J2);
}

auto setup_orbitals(
    const std::vector<size_t>& n_occ,
    const std::vector<size_t>& n_active,
    const size_t n_ao
) {
  const auto n_spin = n_occ.size();

  REQUIRE(0 < n_spin);
  REQUIRE(n_spin <= 2);
  REQUIRE(n_active.size() == n_spin);

  uw12::utils::Orbitals Co;
  uw12::utils::Orbitals active_Co;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto C = uw12::linalg::random(n_ao, n_occ[sigma], seed);
    REQUIRE(n_active[sigma] <= n_occ[sigma]);

    Co.push_back(C);
    active_Co.push_back(uw12::linalg::tail_cols(C, n_active[sigma], true));
  }

  return std::pair(Co, active_Co);
}

TEST_CASE("Test Four Electron Utils - Test t_ab") {
  constexpr size_t n_ao = 9;
  constexpr size_t n_df = 19;

  const auto W_base = setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = setup_base_integrals(n_ao, n_df, seed);

  std::vector<size_t> n_occ = {5};
  std::vector<size_t> n_active = {3};

  for (const size_t n_spin : {1, 2}) {
    if (n_spin == 1) {
      INFO("Closed Shell");
    } else {
      INFO("Open Shell");
    }

    const auto [Co, active_Co] = setup_orbitals(n_occ, n_active, n_ao);

    const auto W = Integrals(W_base, Co, active_Co);
    const auto V = Integrals(V_base, Co, active_Co);

    const auto t_ab = uw12::four_el::calculate_tab(W, V);
    const auto t_ab_u = uw12::four_el::calculate_tab(V, V);

    REQUIRE(t_ab.size() == n_spin);
    REQUIRE(t_ab_u.size() == n_spin);

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK(uw12::linalg::n_rows(t_ab[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(t_ab[sigma]) == n_df);
      CHECK_FALSE(uw12::linalg::is_symmetric(t_ab[sigma]));

      CHECK(uw12::linalg::n_rows(t_ab_u[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(t_ab_u[sigma]) == n_df);
      CHECK(uw12::linalg::is_symmetric(t_ab_u[sigma]));
    }

    n_occ.push_back(4);
    n_active.push_back(2);
  }
}

TEST_CASE("Test Four Electron Utils - Test t_ab_tilde") {
  constexpr size_t n_ao = 7;
  constexpr size_t n_df = 23;

  const auto W_base = setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = setup_base_integrals(n_ao, n_df, seed);

  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  for (const size_t n_spin : {1, 2}) {
    if (n_spin == 1) {
      INFO("Closed Shell");
    } else {
      INFO("Open Shell");
    }

    const auto [Co, active_Co] = setup_orbitals(n_occ, n_active, n_ao);

    const auto W = Integrals(W_base, Co, active_Co);
    const auto V = Integrals(V_base, Co, active_Co);

    const auto t_ab = uw12::four_el::calculate_tab(W, V);
    const auto t_ab_u = uw12::four_el::calculate_tab(V, V);

    const auto ttilde_ab = uw12::four_el::calculate_ttilde(W, V, t_ab);
    const auto ttilde_ab_u = uw12::four_el::calculate_ttilde(V, V, t_ab_u);

    REQUIRE(t_ab.size() == n_spin);
    REQUIRE(t_ab_u.size() == n_spin);
    REQUIRE(t_ab.size() == n_spin);
    REQUIRE(t_ab_u.size() == n_spin);

    const auto ttilde_ab_direct = uw12::four_el::calculate_ttilde(W, V);
    const auto ttilde_ab_u_direct = uw12::four_el::calculate_ttilde(V, V);

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK(uw12::linalg::n_rows(ttilde_ab[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(ttilde_ab[sigma]) == n_df);
      CHECK_FALSE(uw12::linalg::is_symmetric(ttilde_ab[sigma]));

      CHECK(uw12::linalg::n_rows(ttilde_ab_direct[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(ttilde_ab_direct[sigma]) == n_df);
      CHECK_FALSE(uw12::linalg::is_symmetric(ttilde_ab_direct[sigma]));
      CHECK(uw12::linalg::nearly_equal(
          ttilde_ab[sigma], ttilde_ab_direct[sigma], epsilon
      ));

      CHECK(uw12::linalg::n_rows(ttilde_ab_u[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(ttilde_ab_u[sigma]) == n_df);
      CHECK(uw12::linalg::is_symmetric(ttilde_ab_u[sigma]));

      CHECK(uw12::linalg::n_rows(ttilde_ab_u_direct[sigma]) == n_df);
      CHECK(uw12::linalg::n_cols(ttilde_ab_u_direct[sigma]) == n_df);
      CHECK(uw12::linalg::is_symmetric(ttilde_ab_u_direct[sigma]));
      CHECK(uw12::linalg::nearly_equal(
          ttilde_ab_u[sigma], ttilde_ab_u_direct[sigma], epsilon
      ));
    }

    n_occ.push_back(3);
    n_active.push_back(2);
  }
}
