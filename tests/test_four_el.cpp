//
// Created by Zack Williams on 18/03/2024.
//

#include "../src/four_electron/four_electron.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

using test::eps;
using test::epsilon;
using test::margin;
using test::seed;
using uw12::four_el::form_fock_four_el_df;

TEST_CASE("Test four electron term - Closed Shell (opposite spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto [fock, energy] = form_fock_four_el_df(W, V, true, true, 1.0, 0);
  REQUIRE((fock.size() == 1));

  SECTION("Check enegy is the same whether indirect term is calculated or not"
  ) {
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, false, true, 1.0, 0);

    REQUIRE((fock2.size() == 1));

    CHECK((uw12::linalg::nearly_equal(fock2[0], fock[0], epsilon)));
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check enegy is the same whether fock matrix is calculated or not") {
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, false, false, 1.0, 0);

    REQUIRE((fock2.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock2[0], fock0, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check scale = 0 returns zero energy") {
    const auto [fock2, energy2] = form_fock_four_el_df(W, V, false, true, 0, 0);

    REQUIRE((fock2.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock2[0], fock0, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinAbs(0, margin));
  }

  SECTION("Check scale factor") {
    constexpr auto scale_factor = 1.5;
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, false, true, scale_factor, 0);

    REQUIRE((fock2.size() == 1));

    const uw12::linalg::Mat mat2 = fock[0] * scale_factor;
    const auto target = energy * scale_factor;
    CHECK(uw12::linalg::nearly_equal(fock2[0], mat2, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinAbs(target, margin));
  }

  SECTION("Check symmetry of expresion") {
    const auto [fock2, energy2] =
        form_fock_four_el_df(V, W, false, true, 1.0, 0);

    REQUIRE((fock2.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock2[0], fock[0], epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }
}

TEST_CASE("Test four electron term - Closed Shell (same spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto [fock, energy] = form_fock_four_el_df(W, V, true, true, 0, 1.0);
  REQUIRE((fock.size() == 1));

  SECTION("Check for incorrect results when indirect term not calcuated") {
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, false, true, 0, 1.0);

    REQUIRE((fock2.size() == 1));

    CHECK_FALSE((uw12::linalg::nearly_equal(fock2[0], fock[0], epsilon)));
  }

  SECTION("Check enegy is the same whether fock matrix is calculated or not") {
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, true, false, 0, 1.0);

    REQUIRE((fock2.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock2[0], fock0, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check scale factor") {
    constexpr auto scale_factor = 1.5;
    const auto [fock2, energy2] =
        form_fock_four_el_df(W, V, true, true, 0, scale_factor);

    REQUIRE((fock2.size() == 1));

    const uw12::linalg::Mat mat2 = fock[0] * scale_factor;
    const auto target = energy * scale_factor;
    CHECK(uw12::linalg::nearly_equal(fock2[0], mat2, epsilon));
    CHECK_THAT(
        energy2, Catch::Matchers::WithinRel(target, eps * 10)
    );  // Linux test
  }
}

TEST_CASE("Test four electron term - Closed Shell (full expression)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto [fock, energy] = form_fock_four_el_df(W, V, true, true, 1.0, 0.5);
  REQUIRE((fock.size() == 1));

  SECTION("Check result are equal when calculated separately") {
    const auto [osfock, osenergy] =
        form_fock_four_el_df(W, V, false, true, 1.0, 0);
    REQUIRE((osfock.size() == 1));

    const auto [ssfock, ssenergy] =
        form_fock_four_el_df(W, V, true, true, 0, 1);
    REQUIRE((ssfock.size() == 1));

    const auto energy_total = osenergy + 0.5 * ssenergy;
    const uw12::linalg::Mat fock_total = osfock[0] + 0.5 * ssfock[0];
    CHECK(uw12::linalg::nearly_equal(fock_total, fock[0], epsilon));
    CHECK_THAT(energy_total, Catch::Matchers::WithinRel(energy, eps));
  }
}

TEST_CASE(
    "Test four electron term - equality of open and closed shell expressions"
) {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto W_base = test::setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = test::setup_base_integrals(n_ao, n_df, seed);

  const auto [Co, active_Co] = test::setup_orbitals(n_occ, n_active, n_ao);

  REQUIRE(Co.size() == 1);
  REQUIRE(active_Co.size() == 1);

  const uw12::utils::Orbitals Co_open = {Co[0], Co[0]};
  const uw12::utils::Orbitals active_Co_open = {active_Co[0], active_Co[0]};

  const auto W = uw12::integrals::Integrals(W_base, Co, active_Co);
  const auto V = uw12::integrals::Integrals(V_base, Co, active_Co);

  const auto W2 = uw12::integrals::Integrals(W_base, Co_open, active_Co_open);
  const auto V2 = uw12::integrals::Integrals(V_base, Co_open, active_Co_open);

  const auto [fock, energy] = form_fock_four_el_df(W, V, true, true, 1.0, 0.5);
  REQUIRE((fock.size() == 1));

  const auto [fock2, energy2] =
      form_fock_four_el_df(W2, V2, true, true, 1.0, 0.5);
  REQUIRE((fock2.size() == 2));
  for (size_t sigma = 0; sigma < 2; ++sigma) {
    CHECK(uw12::linalg::nearly_equal(fock2[sigma], fock[0], epsilon));
  }
  CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
}

TEST_CASE("Test four electron term - Closed Shell (Check cases)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  SECTION("All electron") {
    const std::vector<size_t> n_active = {5};

    const auto [W, V] =
        test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

    form_fock_four_el_df(W, V, true, true, 1.0, 0.5);
  }

  SECTION("No active orbitals") {
    const auto [W, V] =
        test::setup_integrals_pair(n_ao, n_df, n_occ, {0}, seed + 1);

    const auto [fock, energy] =
        form_fock_four_el_df(W, V, true, true, 1.0, 0.5);

    REQUIRE((fock.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock[0], fock0, epsilon));
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }

  SECTION("No occupied orbitals") {
    const auto [W, V] =
        test::setup_integrals_pair(n_ao, n_df, {0}, {0}, seed + 1);

    const auto [fock, energy] =
        form_fock_four_el_df(W, V, true, true, 1.0, 0.5);

    REQUIRE((fock.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock[0], fock0, epsilon));
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }
}