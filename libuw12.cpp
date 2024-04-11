//
// Created by Zack Williams on 12/02/2024.
//

#include "libuw12.hpp"

#include "src/integrals/base_integrals.hpp"
#include "src/three_electron/ri_utils.hpp"
#include "src/utils/utils.hpp"
#include "src/uw12.hpp"

namespace uw12 {
auto setup_mat(const double* X, const size_t n_row, const size_t n_col) {
  return linalg::mat(const_cast<double*>(X), n_row, n_col);
}

auto setup_base_integrals_ri(
    const double* X3,
    const double* X2,
    const double* X3_ri,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri
) {
  const auto X3_0 = setup_mat(X3, n_ao * (n_ao + 1) / 2, n_df);
  const auto X2_0 = setup_mat(X2, n_df, n_df);
  const auto X3_ri_0 = setup_mat(X3_ri, n_ao * n_ri, n_df);

  return integrals::BaseIntegrals(X3_0, X2_0, X3_ri_0);
}

auto setup_base_integrals(
    const double* X3, const double* X2, const size_t n_ao, const size_t n_df
) {
  const auto X3_0 = setup_mat(X3, n_ao * (n_ao + 1) / 2, n_df);
  const auto X2_0 = setup_mat(X2, n_df, n_df);

  return integrals::BaseIntegrals(X3_0, X2_0);
}

utils::Orbitals setup_orbitals(
    const double* C, const size_t n_ao, const size_t n_orb, const size_t n_spin
) {
  const auto C1 = setup_mat(C, n_ao, n_orb * n_spin);

  if (n_spin == 1) {
    return {C1};
  }

  assert(n_spin == 2);
  assert(linalg::n_cols(C1) == n_orb * 2);

  const auto C_alpha = linalg::head_cols(C1, n_orb);
  const auto C_beta = linalg::tail_cols(C1, n_orb);

  return {C_alpha, C_beta};
}

utils::Occupations setup_occupations(
    const double* occ,
    const size_t n_spin,
    const size_t n_occ_alpha,
    const size_t n_occ_beta
) {
  const linalg::Vec occ_vec = setup_mat(occ, n_occ_alpha + n_occ_beta, 1);

  if (n_spin == 1) {
    if (n_occ_beta != 0) {
      throw std::runtime_error("n_spin is 1, but n_occ_beta != 0");
    }

    return {occ_vec};
  }

  assert(n_spin == 2);

  const auto occ_alpha = linalg::head(occ_vec, n_occ_alpha);
  const auto occ_beta = linalg::tail(occ_vec, n_occ_beta);

  return {occ_alpha, occ_beta};
}

std::vector<size_t> get_n_active(
    const size_t n_spin, const size_t n_active_alpha, const size_t n_active_beta
) {
  if (n_spin == 1) {
    if (n_active_beta != 0) {
      throw std::runtime_error(
          "Number of spin channels is 1, but n_active_beta_provided"
      );
    }

    return {n_active_alpha};
  }

  assert(n_spin == 2);

  return {n_active_alpha, n_active_beta};
}

void sort_result(const utils::FockMatrixAndEnergy& result, double* fock) {
  const auto n_spin = utils::spin_channels(result.fock);
  const auto n_ao = linalg::n_rows(result.fock[0]);

  auto f_mat = linalg::mat(fock, n_ao, n_ao * n_spin);

  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    linalg::assign_cols(f_mat, result.fock[sigma], sigma * n_ao);
  }
}

utils::FockMatrixAndEnergy uw12_fock(
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri,
    const size_t n_orb,
    const size_t n_spin,
    const size_t n_occ_alpha,
    const size_t n_occ_beta,
    const size_t n_active_alpha,
    const size_t n_active_beta,
    const bool calculate_fock,
    const double scale_same_spin,
    const size_t print_level
) {
  if (n_spin != 1 && n_spin != 2) {
    throw std::runtime_error("Number of spin channels must be 1 or 2");
  }

  const auto W = setup_base_integrals_ri(W3, W2, W3_ri, n_ao, n_df, n_ri);
  const auto V = setup_base_integrals_ri(V3, V2, V3_ri, n_ao, n_df, n_ri);
  const auto WV = setup_base_integrals(WV3, WV2, n_ao, n_df);

  const auto S = setup_mat(S2, n_ao + n_ri, n_ao + n_ri);
  const auto abs_projectors =
      three_el::ri::calculate_abs_projectors(S, n_ao, n_ri);

  const auto orbitals = setup_orbitals(C, n_ao, n_orb, n_spin);
  const auto occupations =
      setup_occupations(occ, n_spin, n_occ_alpha, n_occ_beta);

  const auto n_active = get_n_active(n_spin, n_active_alpha, n_active_beta);

  const auto indirect_term = !utils::nearly_zero(scale_same_spin);

  constexpr auto scale_opp_spin = 1.0;

  return form_fock(
      W,
      V,
      WV,
      abs_projectors,
      orbitals,
      occupations,
      n_active,
      indirect_term,
      calculate_fock,
      scale_opp_spin,
      scale_same_spin,
      print_level
  );
}

double uw12_energy(
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri,
    const size_t n_orb,
    const size_t n_occ,
    const size_t n_active,
    const double scale_same_spin,
    const size_t print_level
) {
  return uw12_fock(
             W3,
             W2,
             W3_ri,
             V3,
             V2,
             V3_ri,
             WV3,
             WV2,
             S2,
             C,
             occ,
             n_ao,
             n_df,
             n_ri,
             n_orb,
             1,
             n_occ,
             0,
             n_active,
             0,
             false,
             scale_same_spin,
             print_level
  )
      .energy;
}

double uw12_energy(
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri,
    const size_t n_orb,
    const size_t n_occ_alpha,
    const size_t n_occ_beta,
    const size_t n_active_alpha,
    const size_t n_active_beta,
    const double scale_same_spin,
    const size_t print_level
) {
  return uw12_fock(
             W3,
             W2,
             W3_ri,
             V3,
             V2,
             V3_ri,
             WV3,
             WV2,
             S2,
             C,
             occ,
             n_ao,
             n_df,
             n_ri,
             n_orb,
             2,
             n_occ_alpha,
             n_occ_beta,
             n_active_alpha,
             n_active_beta,
             false,
             scale_same_spin,
             print_level
  )
      .energy;
}

double uw12_fock(
    double* fock,
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri,
    const size_t n_orb,
    const size_t n_occ,
    const size_t n_active,
    const double scale_same_spin,
    const size_t print_level
) {
  const auto result = uw12_fock(
      W3,
      W2,
      W3_ri,
      V3,
      V2,
      V3_ri,
      WV3,
      WV2,
      S2,
      C,
      occ,
      n_ao,
      n_df,
      n_ri,
      n_orb,
      1,
      n_occ,
      0,
      n_active,
      0,
      true,
      scale_same_spin,
      print_level
  );

  sort_result(result, fock);

  return result.energy;
}

double uw12_fock(
    double* fock,
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    const size_t n_ao,
    const size_t n_df,
    const size_t n_ri,
    const size_t n_orb,
    const size_t n_occ_alpha,
    const size_t n_occ_beta,
    const size_t n_active_alpha,
    const size_t n_active_beta,
    const double scale_same_spin,
    const size_t print_level
) {
  const auto result = uw12_fock(
      W3,
      W2,
      W3_ri,
      V3,
      V2,
      V3_ri,
      WV3,
      WV2,
      S2,
      C,
      occ,
      n_ao,
      n_df,
      n_ri,
      n_orb,
      2,
      n_occ_alpha,
      n_occ_beta,
      n_active_alpha,
      n_active_beta,
      true,
      scale_same_spin,
      print_level
  );

  sort_result(result, fock);

  return result.energy;
}

}  // namespace uw12