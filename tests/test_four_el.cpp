//
// Created by Zack Williams on 18/03/2024.
//

#include "catch.hpp"
#include "density_utils.hpp"
#include "four_electron/four_electron.hpp"
#include "multi_el_test_utils.hpp"
#include "setup_integrals.hpp"

using uw12::four_el::form_fock_four_el_df;
using uw12::integrals::Integrals;
using uw12_test::eps;
using uw12_test::epsilon;
using uw12_test::margin;
using uw12_test::seed;

void run_os_tests(const Integrals &W, const Integrals &V) {
  uw12_test::run_os_tests(W, V, form_fock_four_el_df);
}

void run_ss_test(const Integrals &W, const Integrals &V) {
  uw12_test::run_ss_test(W, V, form_fock_four_el_df);
}

void run_test_full(const Integrals &W, const Integrals &V) {
  uw12_test::run_test_full(W, V, form_fock_four_el_df);
}

TEST_CASE("Test four electron term - Closed Shell (opposite spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_os_tests(W, V);
}

TEST_CASE("Test four electron term - Closed Shell (same spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_ss_test(W, V);
}

TEST_CASE("Test four electron term - Closed Shell (full expression)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_test_full(W, V);
}

void check_equality_open_closed_spin(
    const uw12::integrals::BaseIntegrals &W_base,
    const uw12::integrals::BaseIntegrals &V_base,
    const uw12::utils::Orbitals &Co,
    const uw12::utils::Orbitals &active_Co
) {
  uw12_test::check_equality_open_closed_spin(
      W_base, V_base, Co, active_Co, form_fock_four_el_df
  );
}

TEST_CASE(
    "Test four electron term - equality of open and closed shell expressions"
) {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto W_base = uw12_test::setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = uw12_test::setup_base_integrals(n_ao, n_df, seed);

  const auto [Co, active_Co] = uw12_test::setup_orbitals(n_occ, n_active, n_ao);

  check_equality_open_closed_spin(W_base, V_base, Co, active_Co);
}

auto get_integral_fn(const size_t n_ao, const size_t n_df) {
  return [n_ao, n_df](
             const std::vector<size_t> &n_occ,
             const std::vector<size_t> &n_active,
             const int seed
         ) {
    return uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed);
  };
}

TEST_CASE("Test four electron term - Closed Shell (Check cases)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto integral_fn = get_integral_fn(n_ao, n_df);

  uw12_test::check_closed_shell_cases(integral_fn, form_fock_four_el_df, fock0);
}

TEST_CASE("Test four electron term - Open Shell (opposite spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_os_tests(W, V);
}

TEST_CASE("Test four electron term - Open Shell (same spin only)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_ss_test(W, V);
}

TEST_CASE("Test four electron term - Open Shell (full expression)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  run_test_full(W, V);
}

TEST_CASE("Test four electron term - Open Shell (All electron case)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {5, 4};

  const auto [W, V] =
      uw12_test::setup_integrals_pair(n_ao, n_df, n_occ, n_active, seed + 1);

  // TODO Add test
  form_fock_four_el_df(W, V, true, true, 1.0, 0.5);
}

TEST_CASE("Test four electron term - Open Shell (No active orbitals)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto integral_fn = get_integral_fn(n_ao, n_df);

  uw12_test::check_open_shell_no_active_orbitals(
      form_fock_four_el_df, integral_fn, fock0
  );
}

template <typename FockFn, typename IntegralFn>
void check_open_shell_no_occupied_orbitals(
    const FockFn &fock_fn,
    const IntegralFn &integral_fn,
    const uw12::linalg::Mat &fock0
) {
  SECTION("Alpha channel") {
    const std::vector<size_t> n_occ = {3, 0};
    const std::vector<size_t> n_active = {2, 0};

    const auto [W, V] = integral_fn(n_occ, n_active, seed + 1);

    const auto [os_fock, os_energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(os_fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(os_fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(os_energy, Catch::Matchers::WithinAbs(0, margin));

    const auto [ss_fock, ss_energy] = fock_fn(W, V, true, true, 0, 0.5);

    REQUIRE((uw12::utils::spin_channels(ss_fock) == 2));
    CHECK_FALSE(uw12::linalg::nearly_equal(ss_fock[0], fock0, epsilon));
    CHECK(uw12::linalg::nearly_equal(ss_fock[1], fock0, epsilon));
  }

  SECTION("Beta channel)") {
    const std::vector<size_t> n_occ = {0, 3};
    const std::vector<size_t> n_active = {0, 2};
    const auto [W, V] = integral_fn(n_occ, n_active, seed + 1);

    const auto [os_fock, os_energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(os_fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(os_fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(os_energy, Catch::Matchers::WithinAbs(0, margin));

    const auto [ss_fock, ss_energy] = fock_fn(W, V, true, true, 0, 0.5);

    REQUIRE((ss_fock.size() == 2));
    CHECK(uw12::linalg::nearly_equal(ss_fock[0], fock0, epsilon));
    CHECK_FALSE(uw12::linalg::nearly_equal(ss_fock[1], fock0, epsilon));
  }

  SECTION("Both channels") {
    const auto [W, V] = integral_fn({0, 0}, {0, 0}, seed + 1);

    const auto [fock, energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }
}

TEST_CASE("Test four electron term - Open Shell (No occupied orbitals)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;

  const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

  const auto integral_fn = get_integral_fn(n_ao, n_df);

  check_open_shell_no_occupied_orbitals(
      form_fock_four_el_df, integral_fn, fock0
  );
}

void test_four_el_fock_all_electron(
    const uw12::integrals::BaseIntegrals &W_base,
    const uw12::integrals::BaseIntegrals &V_base,
    const uw12::utils::DensityMatrix &D,
    const double threshold
) {
  uw12_test::test_multi_el_fock_all_electron(
      form_fock_four_el_df, W_base, V_base, D, threshold
  );
}

TEST_CASE("Test four electron term - Test Fock matrix (Closed Shell)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;
  constexpr auto threshold = 1e-3;

  const std::vector<size_t> n_occ = {5};

  const auto W_base = uw12_test::setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = uw12_test::setup_base_integrals(n_ao, n_df, seed);

  const auto D = uw12_test::density::random_density_matrix(n_occ, n_ao, seed);

  test_four_el_fock_all_electron(W_base, V_base, D, threshold);
}

TEST_CASE("Test four electron term - Test Fock matrix (Open Shell)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 23;
  constexpr auto threshold = 1e-3;

  const std::vector<size_t> n_occ = {5, 4};

  const auto W_base = uw12_test::setup_base_integrals(n_ao, n_df, seed + 1);
  const auto V_base = uw12_test::setup_base_integrals(n_ao, n_df, seed);

  const auto D = uw12_test::density::random_density_matrix(n_occ, n_ao, seed);

  test_four_el_fock_all_electron(W_base, V_base, D, threshold);
}
