
//
// Created by Zack Williams on 10/04/2024.
//

#include "../src/uw12.hpp"
#include "catch.hpp"
#include "test_data.hpp"

constexpr auto threshold = 100 * uw12_test::epsilon;

void run_test(const uw12_test::TestData& test_data) {
  const auto& W = test_data.W;
  const auto& V = test_data.V;
  const auto& WV = test_data.WV;

  const auto n_ao = W.get_number_ao();
  const auto n_df = W.get_number_df();
  const auto n_ri = W.get_number_ri();

  REQUIRE(V.get_number_ao() == n_ao);
  REQUIRE(V.get_number_df() == n_df);
  REQUIRE(V.get_number_ri() == n_ri);

  REQUIRE(WV.get_number_ao() == n_ao);
  REQUIRE(WV.get_number_df() == n_df);

  REQUIRE(uw12::linalg::n_rows(test_data.S) == n_ao + n_ri);
  REQUIRE(uw12::linalg::n_cols(test_data.S) == n_ao + n_ri);

  const auto n_spin = uw12::utils::spin_channels(test_data.orbitals);
  REQUIRE(uw12::utils::spin_channels(test_data.orbitals) == n_spin);
  REQUIRE(uw12::utils::spin_channels(test_data.occ) == n_spin);
  REQUIRE(uw12::utils::spin_channels(test_data.n_active) == n_spin);
  REQUIRE(uw12::utils::spin_channels(test_data.fock.fock) == n_spin);

  const auto abs_projectors =
      uw12::three_el::ri::calculate_abs_projectors(test_data.S, n_ao, n_ri);

  const auto [fock, energy] = uw12::form_fock(
      W,
      V,
      WV,
      abs_projectors,
      test_data.orbitals,
      test_data.occ,
      test_data.n_active,
      true,
      true,
      1.0,
      0.5
  );

  CHECK_THAT(
      energy, Catch::Matchers::WithinAbs(test_data.fock.energy, threshold)
  );

  REQUIRE(uw12::utils::spin_channels(fock) == n_spin);
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    CHECK(uw12::linalg::nearly_equal(
        fock[sigma], test_data.fock.fock[sigma], threshold
    ));
  }
}

TEST_CASE("Regression Test UW12 - Test Water") {
  constexpr size_t n_spin = 1;
  const auto test_data = uw12_test::TestData("water", "neutral", n_spin);

  run_test(test_data);
}

TEST_CASE("Regression Test UW12 - Test Water (cation)") {
  constexpr size_t n_spin = 2;
  const auto test_data = uw12_test::TestData("water", "cation", n_spin);

  run_test(test_data);
}
