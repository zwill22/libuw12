//
// Created by Zack Williams on 04/03/2024.
//

#include "../src/utils/utils.hpp"
#include "catch.hpp"
#include "density_utils.hpp"

using density::calculate_orbitals_from_density;
using density::orbitals_from_density;
using density::random_density_matrix;
using test::epsilon;
using test::seed;

using uw12::linalg::n_cols;
using uw12::linalg::n_elem;
using uw12::linalg::n_rows;
using uw12::linalg::nearly_equal;

TEST_CASE("Test Density Utils - Orbitals from density") {
  constexpr size_t n_ao = 10;
  std::vector<size_t> n_occ = {4};

  for (size_t i = 0; i < 2; ++i) {
    const auto n_spin = n_occ.size();
    const auto D = random_density_matrix(n_occ, n_ao, seed);
    REQUIRE((D.size() == n_spin));

    for (const auto &Do : D) {
      REQUIRE((n_rows(Do) == n_ao));
      REQUIRE((n_cols(Do) == n_ao));
    }

    SECTION("Orbitals from density") {
      std::vector<uw12::linalg::Mat> orbs = {};
      std::vector<uw12::linalg::Vec> occs = {};
      for (size_t sigma = 0; sigma < n_spin; ++sigma) {
        const auto [C, occ] = orbitals_from_density(D[sigma], 1e-6);
        REQUIRE((n_rows(C) == n_ao));
        REQUIRE((n_cols(C) == n_occ[sigma]));
        REQUIRE((n_elem(occ) == n_occ[sigma]));
        orbs.push_back(C);
        occs.push_back(occ);
      }

      const auto orbitals = calculate_orbitals_from_density(D, 1e-6);
      REQUIRE((orbitals.size() == n_spin));

      const auto orbitals2 =
          uw12::utils::occupation_weighted_orbitals(orbs, occs);
      REQUIRE((orbitals2.size() == n_spin));
      for (size_t sigma = 0; sigma < n_spin; ++sigma) {
        CHECK(nearly_equal(orbitals[sigma], orbitals2[sigma], epsilon));
      }
    }

    SECTION("Test invertability") {
      const auto orbitals = calculate_orbitals_from_density(D, 1e-6);
      REQUIRE((orbitals.size() == n_spin));

      const auto D2 = uw12::utils::construct_density(orbitals);
      REQUIRE((D2.size() == n_spin));
      for (size_t sigma = 0; sigma < n_spin; ++sigma) {
        CHECK(nearly_equal(D[sigma], D2[sigma], epsilon));
      }
    }

    n_occ.push_back(3);
  }

  SECTION("Zero density") {
    const auto D = uw12::linalg::zeros(n_ao, n_ao);

    const auto [C, occ] = orbitals_from_density(D, 1e-6);
    REQUIRE((n_rows(C) == n_ao));
    REQUIRE((n_cols(C) == 1));
    REQUIRE((n_elem(occ) == 1));

    CHECK(nearly_equal(C, uw12::linalg::zeros(n_ao, 1), epsilon));
    CHECK(nearly_equal(occ, uw12::linalg::ones(1), epsilon));
  }
}
