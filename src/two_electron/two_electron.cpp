#include "two_electron.hpp"

#include "../integrals/base_integrals.hpp"
#include "../integrals/integrals.hpp"
#include "../utils/linalg.hpp"
#include "../utils/utils.hpp"

namespace uw12::two_el {
// Vector of length lambda for each spin channel of
// Sum_j ( lambda tilde | WV | jj) = O_lambda * ( lambda | WV | jj)
std::vector<linalg::Vec> calculate_WV_tilde_D(
    const std::vector<linalg::Vec> &WV_D, const integrals::Integrals &WV
) {
  const auto n_spin = WV_D.size();
  const auto &WV_vals = WV.get_df_vals();

  std::vector<linalg::Vec> result(n_spin);
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    assert(linalg::n_elem(WV_vals) == linalg::n_elem(WV_D[sigma]));

    result[sigma] = linalg::schur(WV_vals, WV_D[sigma]);
  }
  return result;
}

linalg::Mat calculate_direct_fock(
    const integrals::Integrals &WV, const linalg::Vec &WV_tilde_D
) {
  // TODO Implement fully direct mo version
  if (const auto &base = WV.get_base_integrals(); base.has_J3_0()) {
    return utils::square(base.get_J3_0() * WV.get_P2() * WV_tilde_D);
  }

  return utils::square(WV.get_J3() * WV_tilde_D);
}

utils::FockMatrixAndEnergy direct_fock(
    const integrals::Integrals &WV,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto n_spin = WV.spin_channels();
  const auto n_ao = WV.number_ao_orbitals();

  std::vector fock(n_spin, linalg::zeros(n_ao, n_ao));
  double energy = 0;

  // Sum_j (lambda | WV | jj) at each lambda
  const auto WV_D = WV.get_X_D();
  // Transform WV_D using df-eigenvalues O_lambda
  const auto WV_tilde_D = calculate_WV_tilde_D(WV_D, WV);

  for (size_t sigma = 0; sigma < n_spin; sigma++) {
    for (size_t sigmaprime = 0; sigmaprime < n_spin; sigmaprime++) {
      assert(n_ao > 0);

      const auto scale_factor =
          (sigma == sigmaprime) ? scale_same_spin : scale_opp_spin;

      const auto energy_spin_factor =
          (n_spin == 1) ? 2 * (scale_opp_spin + scale_same_spin) : scale_factor;

      if (energy_spin_factor == 0) {
        continue;
      }

      energy += 0.5 * energy_spin_factor *
                linalg::dot(WV_D[sigma], WV_tilde_D[sigmaprime]);

      if (calculate_fock) {
        const auto fock_spin_factor =
            0.5 * static_cast<double>(n_spin) * energy_spin_factor;

        fock[sigma] += fock_spin_factor *
                       calculate_direct_fock(WV, WV_tilde_D[sigmaprime]);
      }  // calculate_fock
    }    // sigmaprime
  }      // sigma

  return {fock, energy};
}

// #######################################################################
// #######################################################################
// #######################################################################

utils::FockMatrixAndEnergy indirect_fock(
    const integrals::Integrals &WV, const bool calculate_fock
) {
  const auto n_spin = WV.spin_channels();
  const auto n_ao = WV.number_ao_orbitals();

  const auto &WV_vals = WV.get_df_vals();
  const auto n_df = linalg::n_elem(WV_vals);

  std::vector fock(n_spin, linalg::zeros(n_ao, n_ao));
  double energy = 0;

  const auto &WV3idx_two_trans = WV.get_X3idx_two_trans();

  const auto energy_spin_factor = (n_spin == 1) ? 2 : 1;

  for (size_t sigma = 0; sigma < n_spin; sigma++) {
    const auto n_occ = WV.number_occ_orbitals(sigma);
    assert(n_occ == WV.number_active_orbitals(sigma));
    assert(n_ao > 0);
    assert(n_df > 0);
    if (n_occ == 0) {
      continue;
    }

    const linalg::Mat tmp = WV3idx_two_trans[sigma] * linalg::diagmat(WV_vals);
    energy -=
        0.5 * energy_spin_factor * linalg::dot(tmp, WV3idx_two_trans[sigma]);

    if (calculate_fock) {
      const auto &WV3idx_one_trans = WV.get_X3idx_one_trans();
      const linalg::Mat tmp2 =
          WV3idx_one_trans[sigma] * linalg::diagmat(WV_vals);
      const linalg::Mat WV3idx_one_trans_tilde =
          linalg::reshape(tmp2, n_ao, n_occ * n_df);

      // Reshape WV3idx_two_trans to size (n_ao, nj * na) multiplication sums
      // over nj and na indices returning a matrix of size (n_ao, n_ao)
      fock[sigma] =
          -linalg::reshape(WV3idx_one_trans[sigma], n_ao, n_occ * n_df) *
          linalg::transpose(WV3idx_one_trans_tilde);
    }
  }  // sigma

  return {fock, energy};
}

utils::FockMatrixAndEnergy form_fock_two_el_df(
    const integrals::BaseIntegrals &WV,
    const utils::Orbitals &active_Co,
    const bool indirect_term,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto integrals = integrals::Integrals(WV, active_Co, active_Co);

  auto fock =
      direct_fock(integrals, calculate_fock, scale_opp_spin, scale_same_spin);

  // TODO Implement near zero for double
  if (indirect_term && scale_same_spin != 0) {
    fock += scale_same_spin * indirect_fock(integrals, calculate_fock);
  }

  return fock;
}
}  // namespace uw12::two_el
