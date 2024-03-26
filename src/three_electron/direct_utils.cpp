//
// Created by Zack Williams on 25/03/2024.
//

#include "direct_utils.hpp"

#include "../four_electron/four_electron_utils.hpp"
#include "../utils/linalg.hpp"

namespace uw12::three_el {

using four_el::get_energy_spin_factor;
using integrals::Integrals;
using linalg::col;
using linalg::Mat;
using linalg::n_cols;
using linalg::n_rows;
using linalg::reshape;
using linalg::reshape_col;
using linalg::tail_cols;
using linalg::tail_rows;
using linalg::transpose;
using linalg::Vec;
using ri::ABSProjectors;
using utils::square;

// Calculate a component of the matrix X_AB using
// X_ab^i = (A|w|jμ) [S^{-1}]_{μν} (νj|r^{-1}|B)
//        = W_{jμ}^Α [S^{-1}]_{μν} V_{νj}^{B}
// for `projector` [S^{-1}]_{μν}.
auto calculate_xab_component(const Mat& W, const Mat& V, const Mat& projector) {
  const auto n_orb = n_cols(projector);
  const auto n_df = n_cols(V);
  const size_t n_active = n_rows(V) / n_orb;
  assert(n_active * n_orb == n_rows(V));

  assert(n_cols(W) == n_df);
  assert(n_rows(V) == n_orb * n_active);
  assert(n_rows(projector) * n_active == n_rows(W));

  Mat out(n_df, n_df);
  const auto func = [&W, &V, &projector, &out, n_orb, n_active](const int A) {
    const Mat V_proj = projector * linalg::reshape(col(V, A), n_orb, n_active);

    linalg::assign_cols(out, transpose(W) * linalg::vectorise(V_proj), A);
  };

  parallel::parallel_for(0, n_df, func);

  return out;
}

// Calculates the matrix X_{AB} = (A|w|jμ') S^{-1} (ν'j|r^{-1}|B)
// for μ'ν' in the union of ao and ri. This is done in four parts for each
// possible combination of ao and ri basis, these are:
// X_AB^1 = (A|w|jμ) [S^{-1}]_{μν} (νj|r^{-1}|B)
// X_AB^2 = (A|w|jρ) [S^{-1}]_{ρν} (νj|r^{-1}|B)
// X_AB^3 = (A|w|jρ) [S^{-1}]_{ρσ} (σj|r^{-1}|B)
// X_AB^4 = (A|w|jμ) [S^{-1}]_{μσ} (σj|r^{-1}|B)
// for μ,ν in 'ri' and ρ,σ in 'ao', A,B in `df`.
utils::MatVec calculate_xab(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  const auto n_df = linalg::n_elem(V.get_df_vals());

  const auto n_spin = W.spin_channels();
  assert(V.spin_channels() == n_spin);

  const auto& W3idx_one_trans = W.get_X3idx_one_trans();
  const auto& V3idx_one_trans = V.get_X3idx_one_trans();

  const auto& W3idx_one_trans_ri = W.get_X3idx_one_trans_ri();
  const auto& V3idx_one_trans_ri = V.get_X3idx_one_trans_ri();

  utils::MatVec x(n_spin, linalg::zeros(n_df, n_df));
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto n_active = W.number_active_orbitals(sigma);
    assert(n_active == V.number_active_orbitals(sigma));
#ifndef NDEBUG
    {
      const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);
      const auto n_occ = W.number_occ_orbitals(sigma);
      assert(V.number_occ_orbitals(sigma) == n_occ);
      assert(n_active <= n_occ);
      assert(n_rows(W3idx_one_trans[sigma]) == n_occ * n_ao);
      assert(n_cols(W3idx_one_trans[sigma]) == n_df);
      assert(n_rows(V3idx_one_trans[sigma]) == n_occ * n_ao);
      assert(n_cols(V3idx_one_trans[sigma]) == n_df);

      assert(n_rows(W3idx_one_trans_ri[sigma]) == n_active * n_ri);
      assert(n_cols(W3idx_one_trans_ri[sigma]) == n_df);
      assert(n_rows(V3idx_one_trans_ri[sigma]) == n_active * n_ri);
      assert(n_cols(V3idx_one_trans_ri[sigma]) == n_df);
    }
#endif

    // X_AB^1 = (A|w|jμ) [S^{-1}]_{μν} (νj|r^{-1}|B)
    x[sigma] += calculate_xab_component(
        W3idx_one_trans_ri[sigma],
        V3idx_one_trans_ri[sigma],
        abs_projectors.s_inv_ri_ri
    );

    // X_AB^2 = (A|w|jρ) [S^{-1}]_{ρν} (νj|r^{-1}|B)
    x[sigma] += calculate_xab_component(
        tail_rows(W3idx_one_trans[sigma], n_active * n_ao),
        V3idx_one_trans_ri[sigma],
        abs_projectors.s_inv_ao_ri
    );

    // X_AB^3 = (A|w|jρ) [S^{-1}]_{ρσ} (σj|r^{-1}|B)
    x[sigma] += calculate_xab_component(
        tail_rows(W3idx_one_trans[sigma], n_active * n_ao),
        tail_rows(V3idx_one_trans[sigma], n_active * n_ao),
        abs_projectors.s_inv_ao_ao
    );

    // X_AB^4 = (A|w|jμ) [S^{-1}]_{μσ} (σj|r^{-1}|B)
    x[sigma] += calculate_xab_component(
        W3idx_one_trans_ri[sigma],
        tail_rows(V3idx_one_trans[sigma], n_active * n_ao),
        abs_projectors.s_inv_ri_ao
    );
  }

  return x;
}

// Calculates the direct energy E = X_{AB} \tilde{t}_{AB}
double calculate_direct_energy(
    const utils::MatVec& x,
    const utils::MatVec& ttilde,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  const auto n_spin = utils::spin_channels(x);
  assert(ttilde.size() == n_spin);

  double energy = 0;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    for (size_t sigmaprime = 0; sigmaprime < n_spin; ++sigmaprime) {
      const double energy_spin_factor = get_energy_spin_factor(
          n_spin, sigma, sigmaprime, scale_opp_spin, scale_same_spin
      );

      energy -= energy_spin_factor * linalg::dot(x[sigma], ttilde[sigmaprime]);
    }
  }

  return energy;
}

// Calculate the contribution
// X_{AB} (\tilde{A}|w|αk) (kβ|r^{-1}|\tilde{B})
// + X_{AB} (\tilde{A}|w|iβ) (αi|r^{-1}|\tilde{B})
// for `v_tilde = (αk|r^{-1}|\tilde{B})` and
// `w = (αk|w|A) X_{AB}`
Mat calculate_xab_dttilde(
    const Mat& W3idx_one_trans,
    const Mat& V3idx_one_trans,
    const Mat& xab,
    const Vec& W_vals,
    const Vec& V_vals,
    const size_t n_active,
    const size_t n_ao
) {
  const auto n_df = linalg::n_elem(W_vals);
  const size_t n_occ = n_rows(W3idx_one_trans) / n_ao;
  assert(n_occ * n_ao == n_rows(W3idx_one_trans));
  assert(n_occ * n_ao == n_rows(V3idx_one_trans));

  const Mat w = W3idx_one_trans * linalg::diagmat(W_vals) * xab;
  const Mat v_tilde = V3idx_one_trans * linalg::diagmat(V_vals);

  const auto fock_fn = [&w, &v_tilde, n_active, n_ao, n_occ](const size_t A
                       ) -> Mat {
    const auto wA = reshape_col(w, A, n_ao, n_occ);
    const auto vA = reshape_col(v_tilde, A, n_ao, n_occ);

    Mat fock = -wA * transpose(vA);

    fock -= tail_cols(wA, n_active) * transpose(tail_cols(vA, n_active));

    return fock;
  };

  return parallel::parallel_sum<Mat>(
      0, n_df, linalg::zeros(n_ao, n_ao), fock_fn
  );
}

// Calculates the contribution
// (A|w|αμ) [S^{-1}]_{μσ} (σβ|r^{-1}|B) \tilde{t}_{AB}
//   + (A|w|αρ) [S^{-1}]_{μσ} (σβ|r^{-1}|B) \tilde{t}_{AB}
// using `s = (σβ|r^{-1}|B) \tilde{t}_{AB}^t`
Mat calculate_ttilde_dxab_s_term(
    const Integrals& W,
    const Integrals& V,
    const Mat& ttilde,
    const ABSProjectors& abs_projectors
) {
  const auto n_df_W = linalg::n_elem(W.get_df_vals());
  const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  // TODO Avoid direct calculation of three index objects
  const Mat s = V.get_J3() * transpose(ttilde);
  assert(n_rows(s) == n_ao * (n_ao + 1) / 2);
  assert(n_cols(s) == n_df_W);

  const auto fock_fn = [&s, &abs_projectors, &W, n_ao, n_ri](const size_t A
                       ) -> Mat {
    Mat fock = -square(col(W.get_J3(), A)) * abs_projectors.s_inv_ao_ao *
               square(col(s, A));

    fock -= transpose(linalg::reshape(col(W.get_J3_ri(), A), n_ri, n_ao)) *
            abs_projectors.s_inv_ri_ao * square(col(s, A));

    return fock;
  };

  return parallel::parallel_sum<Mat>(
      0, n_cols(s), linalg::zeros(n_ao, n_ao), fock_fn
  );
}

// Calculates the contribution
// (A|w|αμ) [S^{-1}]_{μν} (νβ|r^{-1}|B) \tilde{t}_{AB}
//  + (A|w|αρ) [S^{-1}]_{ρν} (νβ|r^{-1}|B) \tilde{t}_{AB}
// using `p = (νβ|r^{-1}|B) \tilde{t}_{AB}^t`
Mat calculate_ttilde_dxab_p_term(
    const Integrals& W,
    const Integrals& V,
    const Mat& ttilde,
    const ABSProjectors& abs_projectors
) {
  const auto n_df_W = linalg::n_elem(W.get_df_vals());
  const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);

  const Mat p = V.get_J3_ri() * ttilde.t();
  assert(n_rows(p) == n_ao * n_ri);
  assert(n_cols(p) == n_df_W);

  const auto fock_fn = [&W, &p, &abs_projectors, n_ao, n_ri](const size_t A
                       ) -> Mat {
    Mat fock = -square(col(W.get_J3(), A)) * abs_projectors.s_inv_ao_ri *
               linalg::reshape(col(p, A), n_ri, n_ao);

    fock -= transpose(linalg::reshape(col(W.get_J3_ri(), A), n_ri, n_ao)) *
            abs_projectors.s_inv_ri_ri * linalg::reshape(col(p, A), n_ri, n_ao);

    return fock;
  };

  return parallel::parallel_sum<Mat>(
      0, n_cols(p), linalg::zeros(n_ao, n_ao), fock_fn
  );
}

Mat calculate_ttilde_dxab(
    const Integrals& W,
    const Integrals& V,
    const Mat& ttilde,
    const ABSProjectors& abs_projectors
) {
  // TODO Split into cases to avoid J3(_ri) being calculated directly
  auto fock_sigma = calculate_ttilde_dxab_s_term(W, V, ttilde, abs_projectors);

  fock_sigma += calculate_ttilde_dxab_p_term(W, V, ttilde, abs_projectors);

  return fock_sigma;
}
}
