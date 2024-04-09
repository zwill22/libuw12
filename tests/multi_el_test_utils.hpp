//
// Created by Zack Williams on 08/04/2024.
//

#ifndef MULTI_ELECTRON_TEST_UTILS_HPP
#define MULTI_ELECTRON_TEST_UTILS_HPP

#include "../src/integrals/integrals.hpp"
#include "numerical_fock.hpp"

namespace test {

template <typename Fock>
auto fock_zero(const Fock &fock) {
  const auto n_ao = uw12::linalg::n_rows(fock[0]);

  for (const auto &f : fock) {
    REQUIRE(uw12::linalg::n_rows(f) == n_ao);
    REQUIRE(uw12::linalg::n_cols(f) == n_ao);
  }

  return uw12::linalg::zeros(n_ao, n_ao);
}

template <typename FockFn>
void run_os_tests(
    const uw12::integrals::Integrals &W,
    const uw12::integrals::Integrals &V,
    const FockFn &func
) {
  const auto n_spin = W.spin_channels();
  REQUIRE(V.spin_channels() == n_spin);

  const auto [fock, energy] = func(W, V, true, true, 1.0, 0);
  REQUIRE((fock.size() == n_spin));

  const auto fock0 = fock_zero(fock);

  SECTION("Check enegy is the same whether indirect term is calculated or not"
  ) {
    const auto [fock2, energy2] = func(W, V, false, true, 1.0, 0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon)));
    }
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check enegy is the same whether fock matrix is calculated or not") {
    const auto [fock2, energy2] = func(W, V, false, false, 1.0, 0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], fock0, epsilon)));
    }
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check scale = 0 returns zero energy") {
    const auto [fock2, energy2] = func(W, V, false, true, 0, 0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], fock0, epsilon)));
    }
    CHECK_THAT(energy2, Catch::Matchers::WithinAbs(0, margin));
  }

  SECTION("Check scale factor") {
    constexpr auto scale_factor = 1.5;
    const auto [fock2, energy2] = func(W, V, false, true, scale_factor, 0);

    REQUIRE((fock2.size() == n_spin));

    const uw12::linalg::Mat mat2 = fock[0] * scale_factor;
    const auto target = energy * scale_factor;
    CHECK(uw12::linalg::nearly_equal(fock2[0], mat2, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinAbs(target, margin));
  }

  SECTION("Check symmetry of expresion") {
    const auto [fock2, energy2] = func(V, W, false, true, 1.0, 0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon)));
    }
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }
}

template <typename FockFn>
void run_ss_test(
    const uw12::integrals::Integrals &W,
    const uw12::integrals::Integrals &V,
    const FockFn &func
) {
  const auto n_spin = W.spin_channels();
  REQUIRE(V.spin_channels() == n_spin);

  const auto [fock, energy] = func(W, V, true, true, 0, 1.0);
  REQUIRE((fock.size() == n_spin));

  const auto fock0 = fock_zero(fock);

  SECTION("Check for incorrect results when indirect term not calcuated") {
    const auto [fock2, energy2] = func(W, V, false, true, 0, 1.0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK_FALSE(
          (uw12::linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon))
      );
    }
  }

  SECTION("Check enegy is the same whether fock matrix is calculated or not") {
    const auto [fock2, energy2] = func(W, V, true, false, 0, 1.0);

    REQUIRE((fock2.size() == n_spin));

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], fock0, epsilon)));
    }
    REQUIRE((fock2.size() == n_spin));

    CHECK(uw12::linalg::nearly_equal(fock2[0], fock0, epsilon));
    CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
  }

  SECTION("Check scale factor") {
    constexpr auto scale_factor = 1.5;
    const auto [fock2, energy2] = func(W, V, true, true, 0, scale_factor);

    REQUIRE((fock2.size() == n_spin));

    const auto target = energy * scale_factor;

    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      const uw12::linalg::Mat mat2 = fock[sigma] * scale_factor;
      CHECK((uw12::linalg::nearly_equal(fock2[sigma], mat2, 10 * epsilon)));
    }
    CHECK_THAT(
        energy2, Catch::Matchers::WithinRel(target, eps * 10)
    );  // Linux test
  }
}

template <typename FockFn>
void run_test_full(
    const uw12::integrals::Integrals &W,
    const uw12::integrals::Integrals &V,
    const FockFn &func
) {
  const auto n_spin = W.spin_channels();
  REQUIRE(V.spin_channels() == n_spin);

  const auto [fock, energy] = func(W, V, true, true, 1.0, 0.5);
  REQUIRE((fock.size() == n_spin));

  SECTION("Check result are equal when calculated separately") {
    const auto [osfock, osenergy] = func(W, V, false, true, 1.0, 0);
    REQUIRE((osfock.size() == n_spin));

    const auto [ssfock, ssenergy] = func(W, V, true, true, 0, 1);
    REQUIRE((ssfock.size() == n_spin));

    const auto energy_total = osenergy + 0.5 * ssenergy;
    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      const uw12::linalg::Mat fock_total = osfock[sigma] + 0.5 * ssfock[sigma];
      CHECK(uw12::linalg::nearly_equal(fock_total, fock[sigma], epsilon));
    }
    CHECK_THAT(energy_total, Catch::Matchers::WithinRel(energy, eps));
  }
}

template <typename FockFn>
void check_equality_open_closed_spin(
    const uw12::integrals::BaseIntegrals &W_base,
    const uw12::integrals::BaseIntegrals &V_base,
    const uw12::utils::Orbitals &Co,
    const uw12::utils::Orbitals &active_Co,
    const FockFn &func
) {
  using uw12::integrals::Integrals;
  using uw12::utils::Orbitals;

  REQUIRE(uw12::utils::spin_channels(Co) == 1);
  REQUIRE(uw12::utils::spin_channels(active_Co) == 1);

  const Orbitals Co_open = {Co[0], Co[0]};
  const Orbitals active_Co_open = {active_Co[0], active_Co[0]};

  const auto W = Integrals(W_base, Co, active_Co);
  const auto V = Integrals(V_base, Co, active_Co);

  const auto W2 = Integrals(W_base, Co_open, active_Co_open);
  const auto V2 = Integrals(V_base, Co_open, active_Co_open);

  const auto [fock, energy] = func(W, V, true, true, 1.0, 0.5);
  REQUIRE((fock.size() == 1));

  const auto [fock2, energy2] = func(W2, V2, true, true, 1.0, 0.5);
  REQUIRE((fock2.size() == 2));
  for (size_t sigma = 0; sigma < 2; ++sigma) {
    CHECK(uw12::linalg::nearly_equal(fock2[sigma], fock[0], epsilon));
  }
  CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
}

template <typename FockFn, typename IntegralFn>
void check_closed_shell_cases(
    const IntegralFn &setup_integral_fn,
    const FockFn &fock_fn,
    const uw12::linalg::Mat &fock0
) {
  const std::vector<size_t> n_occ = {5};

  SECTION("All electron") {
    const std::vector<size_t> n_active = {5};

    const auto [W, V] = setup_integral_fn(n_occ, n_active, seed + 1);

    fock_fn(W, V, true, true, 1.0, 0.5);
  }

  SECTION("No active orbitals") {
    const auto [W, V] = setup_integral_fn(n_occ, {0}, seed + 1);

    const auto [fock, energy] = fock_fn(W, V, true, true, 1.0, 0.5);

    REQUIRE((fock.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock[0], fock0, epsilon));
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }

  SECTION("No occupied orbitals") {
    const auto [W, V] = setup_integral_fn({0}, {0}, seed + 1);

    const auto [fock, energy] = fock_fn(W, V, true, true, 1.0, 0.5);

    REQUIRE((fock.size() == 1));

    CHECK(uw12::linalg::nearly_equal(fock[0], fock0, epsilon));
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }
}

template <typename FockFn, typename IntegralFn>
void check_open_shell_no_active_orbitals(
    const FockFn &fock_fn,
    const IntegralFn &integral_fn,
    const uw12::linalg::Mat &fock0
) {
  const std::vector<size_t> n_occ = {5, 4};

  SECTION("Alpha channel") {
    const auto [W, V] = integral_fn(n_occ, {2, 0}, seed + 1);

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
    const auto [W, V] = integral_fn(n_occ, {0, 2}, seed + 1);

    const auto [os_fock, os_energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(os_fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(os_fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(os_energy, Catch::Matchers::WithinAbs(0, margin));

    const auto [ss_fock, ss_energy] = fock_fn(W, V, true, true, 0, 0.5);

    REQUIRE((uw12::utils::spin_channels(ss_fock) == 2));
    CHECK(uw12::linalg::nearly_equal(ss_fock[0], fock0, epsilon));
    CHECK_FALSE(uw12::linalg::nearly_equal(ss_fock[1], fock0, epsilon));
  }

  SECTION("Both channels)") {
    const auto [W, V] = integral_fn(n_occ, {0, 0}, seed + 1);

    const auto [fock, energy] = fock_fn(W, V, true, true, 1.0, 0);

    REQUIRE((uw12::utils::spin_channels(fock) == 2));
    for (size_t sigma = 0; sigma < 2; ++sigma) {
      CHECK(uw12::linalg::nearly_equal(fock[sigma], fock0, epsilon));
    }
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(0, margin));
  }
}

#endif  // MULTI_ELECTRON_TEST_UTILS_HPP
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

template <typename FockFn>
void test_multi_el_fock_all_electron(
    const FockFn &fock_fn,
    const uw12::integrals::BaseIntegrals &W_base,
    const uw12::integrals::BaseIntegrals &V_base,
    const uw12::utils::DensityMatrix &D,
    const double threshold,
    const double delta = 1e-4,
    const double rel_eps = 0.5
) {
  using uw12::integrals::Integrals;

  const auto n_spin = uw12::utils::spin_channels(D);

  const auto Co = density::calculate_orbitals_from_density(D, threshold);

  const auto W1 = Integrals(W_base, Co, Co);
  const auto V1 = Integrals(V_base, Co, Co);

  for (auto scale_same_spin : {0.0, 0.5}) {
    constexpr auto scale_opp_spin = 1.0;

    const auto &[analytic_fock, energy] =
        fock_fn(W1, V1, true, true, scale_opp_spin, scale_same_spin);

    REQUIRE((uw12::utils::spin_channels(analytic_fock) == n_spin));

    const auto energy_fn =
        [&W_base, &V_base, scale_opp_spin, &fock_fn, scale_same_spin, threshold](
            const uw12::utils::DensityMatrix &D_mat
        ) {
          const auto occ_orbitals =
              density::calculate_orbitals_from_density(D_mat, threshold);

          const auto W = Integrals(W_base, occ_orbitals, occ_orbitals);
          const auto V = Integrals(V_base, occ_orbitals, occ_orbitals);

          return fock_fn(W, V, true, true, scale_opp_spin, scale_same_spin).energy;
        };

    REQUIRE_THAT(energy_fn(D), Catch::Matchers::WithinRel(energy, margin));

    const auto num_fock = fock::numerical_fock_matrix(energy_fn, D, delta);
    REQUIRE((uw12::utils::spin_channels(num_fock) == n_spin));

    std::cout << "n spin: " << n_spin << '\n';
    std::cout << "Opposite spin scale: " << scale_opp_spin << '\n';
    std::cout << "Same spin scale: " << scale_same_spin << '\n';

    fock::check_fock(analytic_fock, num_fock, rel_eps);
  }
}

}  // namespace test
