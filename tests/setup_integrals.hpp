//
// Created by Zack Williams on 19/03/2024.
//

#ifndef SETUP_INTEGRALS_HPP
#define SETUP_INTEGRALS_HPP

#include "../src/utils/linalg.hpp"
#include "../src/utils/utils.hpp"
#include "../src/integrals/base_integrals.hpp"
#include "../src/integrals/integrals.hpp"
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
    const size_t n_ao, const size_t n_df, const int J_seed = seed
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
/// @param C_seed Seed for rng for orbitals
///
/// @return Pair of occupied and active orbitals
inline auto setup_orbitals(
    const std::vector<size_t>& n_occ,
    const std::vector<size_t>& n_active,
    const size_t n_ao,
    const int C_seed = seed
) {
  const auto n_spin = n_occ.size();

  REQUIRE(n_spin > 0);
  REQUIRE(n_spin <= 2);
  REQUIRE(n_active.size() == n_spin);

  uw12::utils::Orbitals Co;
  uw12::utils::Orbitals active_Co;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto C = uw12::linalg::random(n_ao, n_occ[sigma], C_seed);
    REQUIRE(n_active[sigma] <= n_occ[sigma]);

    Co.push_back(C);
    active_Co.push_back(uw12::linalg::tail_cols(C, n_active[sigma], true));
  }

  return std::pair(Co, active_Co);
}

inline auto setup_integrals_pair(
    const size_t n_ao,
    const size_t n_df,
    const std::vector<size_t>& n_occ,
    const std::vector<size_t>& n_active,
    const int W_seed = seed,
    const int V_seed = seed,
    const int C_seed = seed
) {
  const auto W_base = setup_base_integrals(n_ao, n_df, W_seed);
  const auto V_base = setup_base_integrals(n_ao, n_df, V_seed);

  const auto [Co, active_Co] = setup_orbitals(n_occ, n_active, n_ao, C_seed);

  const auto W = uw12::integrals::Integrals(W_base, Co, active_Co);
  const auto V = uw12::integrals::Integrals(V_base, Co, active_Co);

  return std::pair(W, V);
}

}  // namespace test

#endif  // SETUP_INTEGRALS_HPP
