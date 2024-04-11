//
// Created by Zack Williams on 01/03/2024.
//

#ifndef DENSITY_UTILS_HPP
#define DENSITY_UTILS_HPP

#include "../src/utils/utils.hpp"

namespace uw12_test::density {

/// Calculates a random density matrix with number of spin channels determined
/// by length of `n_occ` and size `n_ao * n_ao`.
///
/// @param n_occ Occupied orbitals for each spin channel
/// @param n_ao
/// @param seed
///
/// @return
inline uw12::utils::DensityMatrix random_density_matrix(
    const std::vector<size_t> &n_occ, const size_t n_ao, const int seed
) {
  using uw12::linalg::random;

  uw12::utils::Orbitals C;
  uw12::utils::Occupations occ;

  const auto n_spin = n_occ.size();
  const auto max = n_spin == 1 ? 2 : 1;

  for (auto sigma = 0; sigma < n_spin; sigma++) {
    C.push_back(random(n_ao, n_occ[sigma], seed));
    const uw12::linalg::Vec tmp = random(n_occ[sigma], 1, seed);
    occ.emplace_back(max * tmp);
  }

  const auto Co = uw12::utils::occupation_weighted_orbitals(C, occ);

  return uw12::utils::construct_density(Co);
}

/// Calculates the orbitals from a single density matrix `D`
///
/// @param D Density matrix for a single spin channel
/// @param epsilon Threshold for zero occupation
///
/// @return Orbitals and occupations
inline std::pair<uw12::linalg::Mat, uw12::linalg::Vec> orbitals_from_density(
    const uw12::linalg::Mat &D, const double epsilon
) {
  const auto n_ao = uw12::linalg::n_rows(D);
  assert(uw12::linalg::n_cols(D) == n_ao);
  assert(epsilon > 0);

  if (uw12::linalg::nearly_equal(
          D, uw12::linalg::zeros(n_ao, n_ao), uw12_test::epsilon
      )) {
    return {uw12::linalg::zeros(n_ao, 1), uw12::linalg::ones(1)};
  }

  const uw12::linalg::Mat D_neg = -0.5 * (D + uw12::linalg::transpose(D));

  const auto [occ_neg, C] =
      uw12::linalg::eigen_decomposition(D_neg, epsilon, epsilon);

  const uw12::linalg::Vec occ = -1 * occ_neg;

  return {C, occ};
}

/// Calculates the occupation weighted orbitals from a density matrix
///
/// @param D Density matrix
/// @param epsilon Threshold for zero occupation
///
/// @return Occupation weighted orbitals
inline uw12::utils::Orbitals calculate_orbitals_from_density(
    const uw12::utils::DensityMatrix &D, const double epsilon
) {
  uw12::utils::Orbitals orbitals;
  uw12::utils::Occupations occupations;

  const auto n_spin = D.size();

  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto [C, occ] = orbitals_from_density(D[sigma], epsilon);
    orbitals.push_back(C);
    occupations.push_back(occ);
  }

  return uw12::utils::occupation_weighted_orbitals(orbitals, occupations);
}
}  // namespace uw12_test::density

#endif
