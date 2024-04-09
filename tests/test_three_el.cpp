//
// Created by Zack Williams on 27/03/2024.
//

#include "../src/three_electron/three_electron.hpp"
#include "catch.hpp"
#include "density_utils.hpp"
#include "multi_el_test_utils.hpp"
#include "setup_integrals.hpp"

using uw12::integrals::Integrals;
using uw12::three_el::form_fock_three_el_term_df_ri;
using uw12::three_el::ri::ABSProjectors;

constexpr size_t n_ao = 15;
constexpr size_t n_df = 17;
constexpr size_t n_ri = 23;

constexpr auto threshold = 1e-3;

constexpr auto W_seed = test::seed + 1;
constexpr auto V_seed = test::seed;

const auto abs_projectors = test::setup_abs_projector(n_ao, n_ri);
const auto fock0 = uw12::linalg::zeros(n_ao, n_ao);

auto get_fock_fn(const ABSProjectors& abs_projectors) {
  return [&abs_projectors](
             const Integrals& X,
             const Integrals& Y,
             const bool indirect_term,
             const bool calculate_fock,
             const double scale_opp_spin,
             const double scale_same_spin
         ) {
    return form_fock_three_el_term_df_ri(
        X,
        Y,
        abs_projectors,
        indirect_term,
        calculate_fock,
        scale_opp_spin,
        scale_same_spin
    );
  };
}

auto get_integral_fn() {
  return [](const std::vector<size_t>& n_occ,
            const std::vector<size_t>& n_active,
            const int seed) {
    return test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, seed);
  };
}

void run_os_tests(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto fock_fn = get_fock_fn(abs_projectors);

  return test::run_os_tests(W, V, fock_fn);
}

void run_ss_tests(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto fock_fn = get_fock_fn(abs_projectors);

  return test::run_ss_test(W, V, fock_fn);
}

void run_test_full(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto fock_fn = get_fock_fn(abs_projectors);
  return test::run_test_full(W, V, fock_fn);
}

TEST_CASE("Test three electron term - Closed Shell (opposite spin only)") {
  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_os_tests(W, V, abs_projectors);
}

TEST_CASE("Test three electron term - Closed Shell (same spin only)") {
  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_ss_tests(W, V, abs_projectors);
}

TEST_CASE("Test three electron term - Closed Shell (full expression)") {
  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_test_full(W, V, abs_projectors);
}

TEST_CASE(
    "Test three electron term - equality of open and closed shell expressions"
) {
  const std::vector<size_t> n_occ = {5};
  const std::vector<size_t> n_active = {4};

  const auto W_base = test::setup_base_integrals(n_ao, n_df, n_ri, W_seed);
  const auto V_base = test::setup_base_integrals(n_ao, n_df, n_ri, V_seed);

  const auto [Co, active_Co] = test::setup_orbitals(n_occ, n_active, n_ao);

  const auto fock_fn = get_fock_fn(abs_projectors);

  test::check_equality_open_closed_spin(W_base, V_base, Co, active_Co, fock_fn);
}

TEST_CASE("Test three electron term - Closed Shell (Check cases)") {
  const auto integral_fn = get_integral_fn();

  const auto fock_fn = get_fock_fn(abs_projectors);

  test::check_closed_shell_cases(integral_fn, fock_fn, fock0);
}

TEST_CASE("Test three electron term - Open Shell (opposite spin only)") {
  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_os_tests(W, V, abs_projectors);
}

TEST_CASE("Test three electron term - Open Shell (same spin only)") {
  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_ss_tests(W, V, abs_projectors);
}

TEST_CASE("Test three electron term - Open Shell (full expression)") {
  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {4, 3};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  run_test_full(W, V, abs_projectors);
}

TEST_CASE("Test three electron term - Open Shell (All electron case)") {
  const std::vector<size_t> n_occ = {5, 4};
  const std::vector<size_t> n_active = {5, 4};

  const auto [W, V] =
      test::setup_integrals_pair(n_ao, n_df, n_ri, n_occ, n_active, W_seed);

  // TODO Add an actual test here
  form_fock_three_el_term_df_ri(W, V, abs_projectors, true, true, 1.0, 0.5);
}

TEST_CASE("Test three electron term - Open Shell (No active orbitals)") {
  const auto integral_fn = get_integral_fn();
  const auto fock_fn = get_fock_fn(abs_projectors);

  test::check_open_shell_no_active_orbitals(fock_fn, integral_fn, fock0);
}

template <typename FockFn, typename IntegralFn>
void check_open_shell_no_occupied_orbitals(
    const FockFn &fock_fn,
    const IntegralFn &integral_fn,
    const uw12::linalg::Mat &fock0
) {
  using test::epsilon;
  using test::margin;

  SECTION("Alpha channel") {
    const std::vector<size_t> n_occ = {3, 0};
    const std::vector<size_t> n_active = {2, 0};

    const auto [W, V] = integral_fn(n_occ, n_active, W_seed);

    const auto [os_fock, os_energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(os_fock) == 2));
    CHECK(uw12::linalg::nearly_equal(os_fock[0], fock0, epsilon));
    CHECK_FALSE(uw12::linalg::nearly_equal(os_fock[1], fock0, epsilon));

    CHECK_THAT(os_energy, Catch::Matchers::WithinAbs(0, margin));

    const auto [ss_fock, ss_energy] = fock_fn(W, V, true, true, 0, 0.5);

    REQUIRE((uw12::utils::spin_channels(ss_fock) == 2));
    CHECK_FALSE(uw12::linalg::nearly_equal(ss_fock[0], fock0, epsilon));
    CHECK(uw12::linalg::nearly_equal(ss_fock[1], fock0, epsilon));
  }

  SECTION("Beta channel)") {
    const std::vector<size_t> n_occ = {0, 3};
    const std::vector<size_t> n_active = {0, 2};
    const auto [W, V] = integral_fn(n_occ, n_active, W_seed);

    const auto [os_fock, os_energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(os_fock) == 2));
    CHECK_FALSE(uw12::linalg::nearly_equal(os_fock[0], fock0, epsilon));
    CHECK(uw12::linalg::nearly_equal(os_fock[1], fock0, epsilon));

    CHECK_THAT(os_energy, Catch::Matchers::WithinAbs(0, margin));

    const auto [ss_fock, ss_energy] = fock_fn(W, V, true, true, 0, 0.5);

    REQUIRE((ss_fock.size() == 2));
    CHECK(uw12::linalg::nearly_equal(ss_fock[0], fock0, epsilon));
    CHECK_FALSE(uw12::linalg::nearly_equal(ss_fock[1], fock0, epsilon));
  }

  SECTION("Both channels") {
    const auto [W, V] = integral_fn({0, 0}, {0, 0}, W_seed);

    const auto [fock, energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }
}

TEST_CASE("Test three electron term - Open Shell (No occupied orbitals)") {
  const auto integral_fn = get_integral_fn();
  const auto fock_fn = get_fock_fn(abs_projectors);

  check_open_shell_no_occupied_orbitals(fock_fn, integral_fn, fock0);
}

void test_three_el_fock_all_electron(
    const uw12::integrals::BaseIntegrals& W_base,
    const uw12::integrals::BaseIntegrals& V_base,
    const uw12::utils::DensityMatrix& D,
    const double threshold
) {
  const auto fock_fn = get_fock_fn(abs_projectors);

  test::test_multi_el_fock_all_electron(fock_fn, W_base, V_base, D, threshold);
}

TEST_CASE("Test three electron term - Test Fock matrix (Closed Shell)") {
  const std::vector<size_t> n_occ = {5};

  const auto W_base = test::setup_base_integrals(n_ao, n_df, n_ri, W_seed);
  const auto V_base = test::setup_base_integrals(n_ao, n_df, n_ri, V_seed);

  const auto D = density::random_density_matrix(n_occ, n_ao, test::seed);

  test_three_el_fock_all_electron(W_base, V_base, D, threshold);
}

TEST_CASE("Test three electron term - Test Fock matrix (Open Shell)") {
  const std::vector<size_t> n_occ = {5, 4};

  const auto W_base = test::setup_base_integrals(n_ao, n_df, n_ri, W_seed);
  const auto V_base = test::setup_base_integrals(n_ao, n_df, n_ri, V_seed);

  const auto D = density::random_density_matrix(n_occ, n_ao, test::seed);

  test_three_el_fock_all_electron(W_base, V_base, D, threshold);
}
