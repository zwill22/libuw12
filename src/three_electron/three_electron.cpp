//
// Created by Zack Williams on 25/03/2024.
//

#include "three_electron.hpp"

#include "../four_electron/four_electron_utils.hpp"
#include "../integrals/integrals.hpp"
#include "../utils/utils.hpp"
#include "direct_utils.hpp"
#include "indirect_utils.hpp"
#include "ri_utils.hpp"

namespace uw12::three_el {

using four_el::calculate_ttilde;
using four_el::get_energy_spin_factor;
using integrals::Integrals;
using ri::ABSProjectors;

// ##################### Direct Fock Contribution ########################

// Calculate the direct fock matrix contribution given by
// δE/δD_{αβ} = (A|w|αμ) [S^{-1}]_{μν} (νβ|r^{-1}|B) \tilde{t}_{AB}
//            + (A|w|αρ) [S^{-1}]_{ρν} (νβ|r^{-1}|B) \tilde{t}_{AB}
//            + (A|w|αμ) [S^{-1}]_{μσ} (σβ|r^{-1}|B) \tilde{t}_{AB}
//            + (A|w|αρ) [S^{-1}]_{μσ} (σβ|r^{-1}|B) \tilde{t}_{AB}
//            + X_{AB} (\tilde{A}|w|αk) (kβ|r^{-1}|\tilde{B})
//            + X_{AB} (\tilde{A}|w|iβ) (αi|r^{-1}|\tilde{B})
// for μ,ν in `ri`, ρ,σ in `ao`, A,B in `df`, i in active orbitals and k in all
// occupied orbitals.
utils::FockMatrix calculate_direct_fock_matrix(
    const Integrals& W,
    const Integrals& V,
    const utils::MatVec& x,
    const utils::MatVec& ttilde,
    const ABSProjectors& abs_projectors,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto n_spin = W.spin_channels();
  assert(V.spin_channels() == n_spin);
  assert(x.size() == n_spin);
  assert(ttilde.size() == n_spin);

  const auto n_ao = linalg::n_rows(abs_projectors.s_inv_ao_ao);

  const auto& W3idx_one_trans = W.get_X3idx_one_trans();
  const auto& V3idx_one_trans = V.get_X3idx_one_trans();

  utils::MatVec fock(n_spin, linalg::zeros(n_ao, n_ao));
  for (int sigma = 0; sigma < n_spin; ++sigma) {
    const auto n_active = W.number_active_orbitals(sigma);
    assert(n_active == V.number_active_orbitals(sigma));

    const auto fock_sp =
        calculate_ttilde_dxab(W, V, ttilde[sigma], abs_projectors);

    for (int sigmaprime = 0; sigmaprime < n_spin; ++sigmaprime) {
      const auto direct_fock_spin_factor =
          ((n_spin == 1) ? 0.5 : 1) *
          get_energy_spin_factor(
              n_spin, sigma, sigmaprime, scale_opp_spin, scale_same_spin
          );

      // X_{AB} d\tilde{t}_{AB} /dD_{αβ}^{σ}
      fock[sigma] += direct_fock_spin_factor * calculate_xab_dttilde(
                                                   W3idx_one_trans[sigma],
                                                   V3idx_one_trans[sigma],
                                                   x[sigmaprime],
                                                   W.get_df_vals(),
                                                   V.get_df_vals(),
                                                   n_active,
                                                   n_ao
                                               );

      // \tilde{t}_{AB} dX_{AB} /dD_{αβ}^{σ}
      fock[sigmaprime] += direct_fock_spin_factor * fock_sp;
    }
  }

  return fock;
}

// Calculates the direct fock matrix and energy contribution
auto direct_fock(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto x = calculate_xab(W, V, abs_projectors);

  const auto ttilde = calculate_ttilde(W, V);

  utils::FockMatrixAndEnergy fock;
  fock.energy =
      calculate_direct_energy(x, ttilde, scale_opp_spin, scale_same_spin);

  if (calculate_fock) {
    fock.fock = calculate_direct_fock_matrix(
        W, V, x, ttilde, abs_projectors, scale_opp_spin, scale_same_spin
    );
  } else {
    const auto n_ao = linalg::n_rows(abs_projectors.s_inv_ao_ao);
    const auto n_spin = W.spin_channels();
    for (int sigma = 0; sigma < n_spin; ++sigma) {
      fock.fock.push_back(linalg::zeros(n_ao, n_ao));
    }
  }

  return symmetrise_fock(fock);
}

// ###################### Indirect Fock Contribution ######################

// Calculate indirect fock matrix
// Calculation of which is split into three contributions, i, j, k with the
// index corresponding to the index differentiated. These are
// fock_i = (jk|w|αμ)S_μν^-1(νj|r^-1|kβ)
// fock_j = (αk|w|iμ)S_μν^-1(νβ|r^-1|ki)
// fock_k = (jβ|w|iμ)S_μν^-1(νj|r^-1|αi)
auto calculate_indirect_fock(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto n_spin = W.spin_channels();

  utils::FockMatrix fock;
  for (int sigma = 0; sigma < n_spin; ++sigma) {
    const auto fock_sigma =
        indirect_3el_fock_matrix(W, V, abs_projectors, sigma);

    fock.push_back(fock_sigma);
  }

  return fock;
}

// Calculate indirect fock contibution.
// Energy and fock matrix are calculated separately.
auto indirect_fock(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const bool calculate_fock
) {
  const double energy = indirect_3el_energy(W, V, abs_projectors);

  utils::FockMatrixAndEnergy fock;
  fock.energy = energy;

  if (calculate_fock) {
    fock.fock = calculate_indirect_fock(W, V, abs_projectors);
  } else {
    const auto nspin = W.spin_channels();
    const int nao = abs_projectors.s_inv_ao_ao.n_rows;

    for (int sigma = 0; sigma < nspin; ++sigma) {
      fock.fock.push_back(arma::zeros(nao, nao));
    }
  }

  return symmetrise_fock(fock);
}

// ######################################################################

utils::FockMatrixAndEnergy form_fock_three_el_term_df_ri(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const bool indirect_term,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  auto fock = direct_fock(
      W, V, abs_projectors, calculate_fock, scale_opp_spin, scale_same_spin
  );

  if (indirect_term && scale_same_spin != 0) {
    fock +=
        scale_same_spin * indirect_fock(W, V, abs_projectors, calculate_fock);
  }

  return fock;
}

}
