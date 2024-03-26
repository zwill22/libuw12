//
// Created by Zack Williams on 25/03/2024.
//

#ifndef RI_UTILS_HPP
#define RI_UTILS_HPP

#include "../utils/linalg.hpp"

namespace uw12::three_el::ri {

/// Structure containing the submatrices of the inverse overlap of the
/// combined `ao` and `ri` space
///
/// Contains the four sub-matrices \f$[S^{-1}]_{\mu\nu}\f$,
/// \f$[S^{-1}]_{\mu\sigma}\f$, \f$[S^{-1}]_{\rho\nu}\f$,
/// \f$[S^{-1}]_{\rho\sigma}\f$ for \f$\rho\sigma\f$ in `ao` and \f$\mu\nu\f$
/// in `ri` used in the Auxiliary Basis Set + (ABS+) scheme. Along with the
/// eigenvalues and eigenvectors of \f$S_{\mu\nu}\f$.
struct ABSProjectors {
  linalg::Mat s_inv_ri_ri;
  linalg::Mat s_inv_ri_ao;
  linalg::Mat s_inv_ao_ri;
  linalg::Mat s_inv_ao_ao;
  linalg::Mat s_vecs;
  linalg::Vec s_vals;
};

/// Calculates the ABSProjectors for a given pair of ao and ri basis sets
///
/// Eigenvalues of the overlap matrix are assumed to be zero for values less
/// than the maximum of `linear_dependency_threshold` and the maximum eigenvalue
/// multiplied by `eigen_ld_threshold`. Default values for linear dependency
/// taken from https://doi.org/10.1063/1.2712434
///
/// @param ao Atomic orbital basis set
/// @param ri Auxiliary RI basis set
/// @param eigen_ld_threshold ld threshold relative to greatest eigenvalue
/// @param linear_dependency_threshold ld threshold in the eigenvalues
///
/// @return ABSProjectors
inline ABSProjectors calculate_abs_projectors(
    const linalg::Mat& overlap_matrix,
    const size_t n_ao,
    const size_t n_ri,
    double eigen_ld_threshold = 1e-8,
    double linear_dependency_threshold = 1e-6
) {
  using linalg::head_rows;
  using linalg::tail_rows;
  using linalg::transpose;

  const auto [s_vals, s_vecs] = linalg::eigen_decomposition(
      overlap_matrix, linear_dependency_threshold, eigen_ld_threshold
  );

  assert(linalg::n_rows(s_vecs) == nao + nri);
  assert(linalg::n_cols(s_vecs) == linalg::n_elem(s_vals));

  const linalg::Mat inv = linalg::inv_sym_pd(linalg::diagmat(s_vals));

  return {
      tail_rows(s_vec, nri) * inv * transpose(tail_rows(s_vec, nri)),
      tail_rows(s_vec, nri) * inv * transpose(head_rows(s_vec, nao)),
      head_rows(s_vec, nao) * inv * transpose(tail_rows(s_vec, nri)),
      head_rows(s_vec, nao) * inv * transpose(head_rows(s_vec, nao))
  };
}

}  // namespace uw12::three_el::ri
#endif  // RI_UTILS_HPP
