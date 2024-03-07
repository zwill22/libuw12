//
// Created by Zack Williams on 23/02/2024.
//

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include <cassert>

#include "../utils/linalg.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"

namespace uw12::integrals::transformations {
/// \brief Transform one index of the three-index density-fitting integrals from
/// the ao basis to mo orbitals using orbital matrix `C`.
///
/// Transform one index of the matrix `J3` of three-index density-fitting
/// integrals \f$(\mu\nu | A)\f$ for ao indices \f$\mu, \nu\f$ to the space of
/// orbitals `k` resulting in a matrix of three-index integrals
/// \f$(\mu k | A)\f$ for df index `A`.
///
/// The density-fitting integrals `J3` are in matrix form with
/// `n_ao * (n_ao + 1) /2` rows and `nA` columns. The orbital coefficient matrix
/// `C` should be of size `n_ao * n_occ` for number of ao basis functions `n_ao`
/// and number of occupied orbitals `n_occ`. The resulting matrix is of size
/// `(n_ao * n_occ) * nA`.
///
/// @param J3 Three-index density-fitting integrals \f$(\mu\nu|A)\f$
/// @param C Orbital coefficients \f$C_{\mu k}\f$
///
/// @return One-index transformed df integrals \f$(\mu k|A)\f$
inline linalg::Mat mo_transform_one_index_full(
    const linalg::Mat &J3, const linalg::Mat &C
) {
  using namespace linalg;

  const auto n_ao = n_rows(C);
  const auto n_orb = n_cols(C);
  const auto n_df = n_cols(J3);
  if (n_rows(J3) != n_ao * (n_ao + 1) / 2) {
    throw std::runtime_error("J3 and C of incompatible sizes");
  }

  Mat result(n_ao * n_orb, n_df);

  const auto parallel_fn = [&result, &J3, &C](const size_t A) {
    assign_cols(result, vectorise(utils::square(col(J3, A)) * C), A);
  };

  parallel::parallel_for(0, n_df, parallel_fn);

  return result;
}

/// \brief Transform first index of a three-index density-fitting integrals
/// object from the ao basis using orbital matrix `C`.
///
/// Transform the first index of the matrix `J3` of three-index density-fitting
/// integrals \f$(\mu k| A)\f$ for ao index \f$\mu\f$ to orbitals `i` in mo space
/// resulting in a matrix of three-index integrals \f$(i k | A)\f$ for df
/// index `A`.
///
/// The density-fitting integrals `J3` are in matrix form with `n_ao * n2` rows
/// and `n3` columns. The orbital coefficient matrix `C` should be of size
/// `n_ao * n_i` for number of ao basis functions `n_ao` and number of `i`
/// orbitals `n_i`. The resulting matrix is of size
/// `(n_i * n2) * n3`.
///
/// No restriction is placed on size of `n2` and `n3` so may relate to any
/// orbitals space. Unlike ::mo_transform_one_index_full, each column in
/// J3 is a full matrix vectorised rather than the lower trianglar part
/// of a symmetric matrix.
///
/// @param J3 Three-index density-fitting integrals \f$(\mu k|A)\f$
/// @param C Orbital coefficients \f$C_{\mu i}\f$
///
/// @return First-index mo transformed df integrals \f$(i k|A)\f$
inline linalg::Mat transform_first_index(
    const linalg::Mat &J3, const linalg::Mat &C
) {
  using namespace linalg;

  const auto n1 = n_rows(C);
  const auto n3 = n_cols(J3);
  const auto n4 = n_cols(C);
  if (n_rows(J3) % n1 != 0) {
    throw std::runtime_error("number of rows of J3 not a multiple of n_ao");
  }
  const size_t n2 = n_rows(J3) / n1;

  const Mat C_t = transpose(C);

  Mat result(n4 * n2, n3);

  const auto parallel_fn = [&result, &J3, &C_t, n1, n2](const size_t A) {
    // reshaping the column of A3 without memory copy
    const auto A12 = reshape_col(J3, A, n1, n2);

    assign_cols(result, vectorise(C_t * A12), A);
  };

  parallel::parallel_for(0, n3, parallel_fn);

  return result;
}

/// \brief Transform second index of a three-index density-fitting integrals
/// object from the ao basis using orbital matrix `C`.
///
/// Transform the second index of the matrix `J3` of three-index density-fitting
/// integrals \f$(p \mu| A)\f$ for ao index \f$\mu\f$ to the space of orbitals
/// `i` resulting in a matrix of three-index integrals \f$(p i | A)\f$ for df
/// index `A`.
///
/// The density-fitting integrals `J3` are in matrix form with `n1 * n_ao` rows
/// and `n3` columns. The orbital coefficient matrix `C` should be of size
/// `n_ao * n_i` for number of ao basis functions `n_ao` and number of `i`
/// orbitals `n_i`. The resulting matrix is of size `(n1 * n_i) * n3`.
///
/// No restriction is placed on size of `n1` and `n3` so may relate to any
/// orbitals space. This function is not compatible with results of
/// ::mo_transform_one_index_full. Should only be used to transform the second
/// index an asymmetric matrix.
///
/// @param J3 Three-index density-fitting integrals \f$(a \mu|A)\f$
/// @param C Orbital coefficients \f$C_{\mu i}\f$
///
/// @return Second-index mo transformed df integrals \f$(a i|A)\f$
inline linalg::Mat transform_second_index(
    const linalg::Mat &J3, const linalg::Mat &C
) {
  using namespace linalg;

  const auto n2 = n_rows(C);
  const auto n3 = n_cols(J3);
  const auto n4 = n_cols(C);
  if (n_rows(J3) % n2 != 0) {
    throw std::runtime_error(
        "number of rows J3 is not a multiple of number of orbitals being "
        "transformed"
    );
  }
  assert(n_rows(J3) % n2 == 0);
  const size_t n1 = n_rows(J3) / n2;

  Mat result(n1 * n4, n3);

  const auto parallel_fn = [&result, &J3, &C, n2, n1](const size_t A) {
    const auto col_mat = reshape_col(J3, A, n1, n2);

    assign_cols(result, vectorise(col_mat * C), A);
  };

  parallel::parallel_for(0, n3, parallel_fn);

  return result;
}

/// Directly transform the three-index density-fitting integrals from
/// the ao basis to mo space using orbital matrices `C_left` and `C_right`.
///
/// Transform both ao indices of the matrix `J3` of three-index density-fitting
/// integrals \f$(\mu\nu | A)\f$ for ao indices \f$\mu, \nu\f$ to the space of
/// orbitals `k` and `i` resulting in a matrix of three-index integrals
/// \f$(i k | A)\f$ for df index `A`.
///
/// The density-fitting integrals `J3` are in matrix form with
/// `n_ao * (n_ao + 1) /2` rows and `nA` columns. The orbital coefficient
/// matrices `C_left` and `C_right` should be of sizes `n_ao * n_i` and
/// `n_ao * n_k` respectively. For number of ao basis functions `n_ao`
/// and number of `i` and `k` orbitals `n_i` and `n_k` respectively. The
/// resulting matrix is of size `(n_i * n_k) * nA`.
///
/// @param J3 Three-index density-fitting integrals \f$(\mu\nu|A)\f$
/// @param C_left Orbital coefficients \f$C_{\mu i}\f$
/// @param C_right Orbital coefficients \f$C_{\mu k}\f$
///
/// @return Two-index mo transformed df integrals \f$(i k|A)\f$
inline linalg::Mat mo_transform_two_index_full(
    const linalg::Mat &J3, const linalg::Mat &C_left, const linalg::Mat &C_right
) {
  if (linalg::n_rows(C_left) != linalg::n_rows(C_right)) {
    throw std::runtime_error(
        "Coefficient matrices with different numbers of ao functions"
    );
  }

  const auto tmp = mo_transform_one_index_full(J3, C_right);

  return transform_first_index(tmp, C_left);
}
}  // namespace uw12::integrals::transformations

#endif  // TRANSFORMATIONS_HPP
