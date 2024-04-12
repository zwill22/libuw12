#include "four_electron.hpp"

#include "four_electron_utils.hpp"
#include "integrals/integrals.hpp"
#include "utils/linalg.hpp"
#include "utils/utils.hpp"

namespace uw12::four_el {

// Calculate direct fock matrix
linalg::Mat direct_fock_contribution(
    const linalg::Mat& W3idx_one_trans,
    const linalg::Mat& V3idx_one_trans,
    const linalg::Mat& ttilde,
    const double energy_spin_factor,
    const size_t n_ao,
    const size_t n_spin
) {
  using linalg::n_rows;

  const size_t n_occ = n_rows(W3idx_one_trans) / n_ao;
  assert(n_rows(W3idx_one_trans) == n_ao * n_occ);
  assert(n_rows(V3idx_one_trans) == n_ao * n_occ);

  // Transform W:
  // Wttilde_{alpha, i, A} = \sum_B W_{alpha, i, B} * ttilde_{BA}
  const linalg::Mat W3idx_one_trans_ttilde = W3idx_one_trans * ttilde;

  const auto n_df = linalg::n_cols(V3idx_one_trans);
  assert(linalg::n_cols(W3idx_one_trans) == n_df);
  assert(linalg::n_cols(W3idx_one_trans_ttilde) == n_df);

  const double fock_spin_factor = (n_spin == 1 ? 0.5 : 1) * energy_spin_factor;

  // F_{\alpha, \beta} = \sum_{Aj} V_{alpha, j, A} * Wtilde_{beta, j, A}
  return 2 * fock_spin_factor *
         linalg::reshape(V3idx_one_trans, n_ao, n_occ * n_df)  // alpha, jA
         * linalg::transpose(
               linalg::reshape(W3idx_one_trans_ttilde, n_ao, n_occ * n_df)
           );  // jA, beta
}

linalg::Mat direct_fock_matrix(
    const linalg::Mat& W3idx_one_trans,
    const linalg::Mat& V3idx_one_trans,
    const linalg::Mat& ttilde,
    const double energy_spin_factor,
    const int n_active,
    const int n_ao,
    const int n_spin
) {
  using linalg::n_rows;
  using linalg::tail_rows;

  const size_t n_occ = n_rows(W3idx_one_trans) / n_ao;
  assert(n_rows(W3idx_one_trans) == n_occ * n_ao);
  assert(n_rows(V3idx_one_trans) == n_occ * n_ao);

  auto fock = direct_fock_contribution(
      W3idx_one_trans, V3idx_one_trans, ttilde, energy_spin_factor, n_ao, n_spin
  );

  if (n_occ == n_active || n_active == 0) {
    return fock;
  }

  const auto frozen_fock = direct_fock_contribution(
      tail_rows(W3idx_one_trans, n_ao * n_active),
      tail_rows(V3idx_one_trans, n_ao * n_active),
      ttilde,
      energy_spin_factor,
      n_ao,
      n_spin
  );

  return 0.5 * (fock + frozen_fock);
}

// Direct fock matrix and energy
utils::FockMatrixAndEnergy direct_fock(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto n_ao = W.number_ao_orbitals();
  const auto n_spin = W.spin_channels();
  assert(n_spin == V.spin_channels());

  std::vector<linalg::Mat> fock(n_spin, linalg::zeros(n_ao, n_ao));
  double energy = 0;

  // t_{AB} (sum over occupied orbitals)
  const auto tab = calculate_tab(W, V);

  // ttilde_{AB} = W_A t_{AB} V_A (transform t_{AB})
  const auto ttildeab = calculate_ttilde(W, V, tab);

  // Each spin state
  for (size_t sigma = 0; sigma < n_spin; sigma++) {
    // Nonzero number of electrons in sigma
    const auto n_active = W.number_active_orbitals(sigma);
    assert(n_active == V.number_active_orbitals(sigma));

    // Each second spin state
    for (size_t sigmaprime = 0; sigmaprime < n_spin; sigmaprime++) {
      const auto energy_spin_factor = get_energy_spin_factor(
          n_spin, sigma, sigmaprime, scale_opp_spin, scale_same_spin
      );

      // Skip if energy factor is zero
      if (energy_spin_factor == 0) {
        continue;
      }

      // Calculate energy
      energy += 0.5 * energy_spin_factor *
                linalg::dot(ttildeab[sigmaprime], tab[sigma]);

      // Direct term Fock matrix
      if (calculate_fock) {
        fock[sigma] += direct_fock_matrix(
            W.get_X3idx_one_trans()[sigma],
            V.get_X3idx_one_trans()[sigma],
            ttildeab[sigmaprime],
            energy_spin_factor,
            n_active,
            n_ao,
            n_spin
        );
      }
    }  // sigmaprime
  }    // sigma

  return utils::symmetrise_fock({fock, energy});
}

// ####################################################################
// ####################################################################
// ####################################################################

double indirect_energy(
    const linalg::Mat& W4idx_four_trans,
    const linalg::Mat& V4idx_four_trans,
    const size_t n_occ,
    const size_t n_spin
) {
  using linalg::n_cols;
  using linalg::n_rows;
  using linalg::sub_mat;

  const size_t n_active = n_rows(W4idx_four_trans) / n_occ;

  assert(n_rows(W4idx_four_trans) == n_active * n_occ);
  assert(n_cols(W4idx_four_trans) == n_active * n_occ);
  assert(n_rows(V4idx_four_trans) == n_active * n_occ);
  assert(n_cols(V4idx_four_trans) == n_active * n_occ);

  const auto indirect_term =
      [&W4idx_four_trans, &V4idx_four_trans, n_spin, n_active](
          const size_t k, const size_t l
      ) -> double {
    const double energy_val = linalg::dot(
        sub_mat(
            W4idx_four_trans, l * n_active, k * n_active, n_active, n_active
        ),
        sub_mat(
            V4idx_four_trans, k * n_active, l * n_active, n_active, n_active
        )
    );

    return -0.5 * ((n_spin == 1) ? 2 : 1) * energy_val;
  };

  return parallel::parallel_sum_2d<double>(
      0, n_occ, 0, n_occ, 0, indirect_term
  );
}

linalg::Mat indirect_fock_matrix(
    const linalg::Mat& W4idx_three_trans,
    const linalg::Mat& V4idx_three_trans,
    const size_t n_ao
) {
  using linalg::n_cols;
  using linalg::n_rows;
  using linalg::sub_mat;
  using linalg::transpose;

  const size_t n_occ = n_rows(W4idx_three_trans) / n_ao;
  const size_t n_active = n_cols(W4idx_three_trans) / n_occ;
  assert(n_rows(W4idx_three_trans) == n_ao * n_occ);
  assert(n_cols(W4idx_three_trans) == n_occ * n_active);
  assert(n_rows(V4idx_three_trans) == n_ao * n_occ);
  assert(n_cols(V4idx_three_trans) == n_occ * n_active);
  assert(n_active <= n_occ);
  const auto n_core = n_occ - n_active;

  const auto indirect_term =
      [&W4idx_three_trans, &V4idx_three_trans, n_ao, n_active, n_core](
          const size_t i, const size_t k
      ) -> linalg::Mat {
    // Pick out ijkth components from the three transformed integrals

    // W4_k size alpha * nj
    const linalg::Mat W4_ik = sub_mat(
        W4idx_three_trans, (n_core + i) * n_ao, n_active * k, n_ao, n_active
    );

    const linalg::Mat V4_ik = linalg::reshape(
        linalg::tail(
            linalg::col(V4idx_three_trans, i + n_active * k), n_ao * n_active
        ),
        n_ao,
        n_active,
        true  // TODO Avoid copy
    );

    return -2 * W4_ik * transpose(V4_ik);
  };

  auto fock_contrib = parallel::parallel_sum_2d<linalg::Mat>(
      0, n_active, 0, n_occ, linalg::zeros(n_ao, n_ao), indirect_term
  );

  if (n_core == 0) {
    return fock_contrib;
  }

  const auto frozen_term =
      [&W4idx_three_trans, &V4idx_three_trans, n_ao, n_active](
          const size_t k, const size_t l
      ) -> linalg::Mat {
    const linalg::Mat fock_val =
        sub_mat(W4idx_three_trans, l * n_ao, k * n_active, n_ao, n_active) *
        transpose(
            sub_mat(V4idx_three_trans, k * n_ao, l * n_active, n_ao, n_active)
        );

    return -2 * fock_val;
  };

  const auto frozen_contrib = parallel::parallel_sum_2d<linalg::Mat>(
      0, n_occ, 0, n_occ, linalg::zeros(n_ao, n_ao), frozen_term
  );

  return 0.5 * (fock_contrib + frozen_contrib);
}

// This algorithm is N^5 scaling for the Fock matrix. Please improve.
// TODO Implement N^4 scaling indirect Fock method
utils::FockMatrixAndEnergy indirect_four_el_fock(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const bool calculate_fock
) {
  const auto n_ao = W.number_ao_orbitals();
  const auto n_spin = W.spin_channels();
  assert(n_spin == V.spin_channels());

  std::vector<linalg::Mat> fock(n_spin, linalg::zeros(n_ao, n_ao));
  double energy = 0;
  for (int sigma = 0; sigma < n_spin; sigma++) {
    const auto n_occ = W.number_occ_orbitals(sigma);
    assert(V.number_occ_orbitals(sigma) == n_occ);
    const auto n_active = W.number_active_orbitals(sigma);
    assert(n_active == V.number_active_orbitals(sigma));

    if (n_active == 0 || n_occ == 0 || n_ao == 0) {
      continue;
    }

    const auto W4idx_four_trans = W.get_X4idx_four_trans(sigma);
    const auto V4idx_four_trans = V.get_X4idx_four_trans(sigma);

    energy +=
        indirect_energy(W4idx_four_trans, V4idx_four_trans, n_occ, n_spin);

    if (calculate_fock) {
      const auto W4idx_three_trans = W.get_X4idx_three_trans(sigma);
      const auto V4idx_three_trans = V.get_X4idx_three_trans(sigma);

      fock[sigma] =
          indirect_fock_matrix(W4idx_three_trans, V4idx_three_trans, n_ao);
    }
  }  // sigma

  return utils::symmetrise_fock({fock, energy});
}

// ####################################################################
// ####################################################################
// ####################################################################

utils::FockMatrixAndEnergy form_fock_four_el_df(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const bool indirect_term,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  auto fock =
      direct_fock(W, V, calculate_fock, scale_opp_spin, scale_same_spin);

  if (indirect_term && !(scale_same_spin == 0)) {
    fock += scale_same_spin * indirect_four_el_fock(W, V, calculate_fock);
  }

  return fock;
}

}  // namespace uw12::four_el
