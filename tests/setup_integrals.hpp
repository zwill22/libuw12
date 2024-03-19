//
// Created by Zack Williams on 19/03/2024.
//

#ifndef SETUP_INTEGRALS_HPP
#define SETUP_INTEGRALS_HPP

#include "../src/utils/linalg.hpp"
#include "../src/utils/utils.hpp"
#include "catch.hpp"

namespace test {

/// Setup random base integrals instance given minimum number of parameters
///
/// @param n_ao Number of atomic orbitals
/// @param n_df Number of density-fitting orbitals
/// @param J_seed Seed for rng
///
/// @return Base integrals
inline auto setup_base_integrals(
    const size_t n_ao, const size_t n_df, const int J_seed
) {
  const auto J2 = uw12::linalg::random_pd(n_df, J_seed);
  const auto J3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, J_seed);

  return uw12::integrals::BaseIntegrals(J3, J2);
}

/// Generate a set of random occupied orbitals and active orbitals
///
/// @param n_occ Number of occupied orbitals in each spin channel
/// @param n_active Number of active orbitals in each spin channel
/// @param n_ao Number of atomic orbitals
///
/// @return Pair of occupied and active orbitals
inline auto setup_orbitals(
    const std::vector<size_t>& n_occ,
    const std::vector<size_t>& n_active,
    const size_t n_ao
) {
  const auto n_spin = n_occ.size();

  REQUIRE(n_spin > 0);
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

}  // namespace test

#endif  // SETUP_INTEGRALS_HPP
