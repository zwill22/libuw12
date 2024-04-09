//
// Created by Zack Williams on 01/12/2020.
//

#ifndef UW12_UTILS_HPP
#define UW12_UTILS_HPP

#include <vector>

#include "linalg.hpp"

namespace uw12::utils {
/// Generate a square symmetric matrix from a vector
///
/// Generate a square symmetric matrix of size `n * n` from a vector of size
/// `n * (n+1) /2` where the elements are the lower triangular elements of the
/// output matrix in column major ordering. Inverse of `lower`.
///
/// @param vec Vector of lower triangular elements of symmetric matrix
/// @param factor Off-diagonal factor
///
/// @return Resulting symmetric matrix
inline auto square(const linalg::Vec &vec, const double factor = 1) {
  const auto n_1 = linalg::n_elem(vec);
  const auto n_2 = static_cast<size_t>(std::sqrt(8 * n_1 - 1) / 2);

  if (n_2 * (n_2 + 1) / 2 != n_1) {
    throw std::logic_error("vector must be of length n(n+1)/2");
  }

  auto matrix = linalg::mat(n_2, n_2);
  size_t idx = 0;
  for (size_t i = 0; i < n_2; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      if (i != j) {
        matrix(i, j) = factor * vec(idx);
        matrix(j, i) = factor * vec(idx);
      } else {
        matrix(i, i) = vec(idx);
      }
      idx++;
    }
  }

  assert(idx == n_1);

  return matrix;
}

/// Compress symmetric matrix into vector of lower triangular elements
///
/// Store `n * n` symmetric matrix as a vector of lower triangular elements of
/// size `n * (n+1) /2`. Inverse of `square`.
///
/// @param mat
/// @param factor
/// @return
inline auto lower(const linalg::Mat &mat, const double factor = 1) {
  const auto n_row = linalg::n_rows(mat);
  if (!linalg::is_square(mat)) {
    throw std::logic_error("matrix is not square");
  }
  if (!linalg::is_symmetric(mat)) {
    throw std::logic_error("matrix is not symmetric");
  }

  linalg::Vec vec(n_row * (n_row + 1) / 2);
  size_t idx = 0;
  for (size_t i = 0; i < n_row; ++i) {
    for (size_t j = 0; j < i; ++j) {
      vec(idx++) = factor * mat(i, j);
    }
    vec(idx++) = mat(i, i);
  }

  assert(idx == n_row * (n_row + 1) / 2);

  return vec;
}

/// Matrix vector for storing multiple matrices in a single object
using MatVec = std::vector<linalg::Mat>;

/// Scalar multiplication of MatVec
/// {
inline auto operator*(const MatVec &object, const double factor) {
  MatVec result(object.size());
  for (size_t i = 0; i < object.size(); ++i) {
    result[i] = factor * object[i];
  }
  return result;
}

inline auto operator*(const double factor, const MatVec &object) {
  return object * factor;
}

/// }

/// Fock matrix addition {
inline auto &operator+=(MatVec &lhs, const MatVec &rhs) {
  const auto length = lhs.size();
  if (rhs.size() != length) {
    throw std::logic_error("containers are of different sizes");
  }

  for (size_t i = 0; i < length; ++i) {
    if (linalg::n_rows(lhs[i]) != linalg::n_rows(rhs[i]) ||
        linalg::n_cols(lhs[i]) != linalg::n_cols(rhs[i])) {
      throw std::logic_error("matrices are of different sizes");
    }

    lhs[i] = lhs[i] + rhs[i];
  }
  return lhs;
}

inline auto operator+(MatVec left, const MatVec &right) {
  left += right;
  return left;
}

/// }

/// OccupationVector for storing occupation numbers
using OccupationVector = linalg::Vec;

/// Object to store orbital coefficients for each spin channel
using Orbitals = MatVec;

/// Object to store occupation numbers for occupied orbitals for each spin
/// channel
using Occupations = std::vector<OccupationVector>;

/// Object to store the density matrix for each spin channel
using DensityMatrix = MatVec;

/// Object to store Fock matrix for each spin channel
using FockMatrix = MatVec;

/// Struct to store Fock matrix and energy contributions
struct FockMatrixAndEnergy {
  FockMatrix fock;
  double energy;
};

/// Scalar multiplication of FockMatrixAndEnergy
/// {
inline auto operator*(const FockMatrixAndEnergy &fock, const double factor)
    -> FockMatrixAndEnergy {
  return {factor * fock.fock, factor * fock.energy};
}

inline auto operator*(const double factor, const FockMatrixAndEnergy &fock) {
  return fock * factor;
}

/// }

/// FockMatrixAndEnergy addition {
inline auto operator+=(
    FockMatrixAndEnergy &lhs, const FockMatrixAndEnergy &rhs
) {
  lhs.fock += rhs.fock;
  lhs.energy += rhs.energy;

  return lhs;
}

inline auto operator+(FockMatrixAndEnergy lhs, const FockMatrixAndEnergy &rhs) {
  lhs += rhs;
  return lhs;
}

/// }

/// Check whether a value is approximately zero
///
/// \tparam T
/// @param value Value to check
/// @param epsilon Multiplicative factor to control numerical limit
///
/// @return Whether value is nearly zero
template <typename T>
auto nearly_zero(T value, int epsilon = 4) {
  return std::abs(value) <= epsilon * std::numeric_limits<T>::epsilon();
}

/// Calculate and check the number of spin channels of an object
///
/// \tparam T
/// @param object Object of size `n_spin`
///
/// @return Number of spin channels `n_spin`
template <typename T>
auto spin_channels(const T &object) {
  const size_t n_spin = object.size();
  if (n_spin < 1 || n_spin > 2) {
    throw std::logic_error("invalid number of spin channels");
  }

  return n_spin;
}

/// Make Fock matrix symmetric (inplace)
///
/// Using \f$0.5 (F + F^T)\f$, make the Fock matrix in a FockMatrixAndEnergy
/// object symmetric
///
/// @param fock A FockMatrixAndEnergy object
///
/// @return Symmetric FockMatrixAndEnergy
inline auto symmetrise_fock(FockMatrixAndEnergy fock) {
  const auto n_spin = spin_channels(fock.fock);

  for (size_t sigma = 0; sigma < n_spin; sigma++) {
    fock.fock[sigma] =
        0.5 * (fock.fock[sigma] + linalg::transpose(fock.fock[sigma]));
  }

  return fock;
}

/// Freeze core orbitals of an Orbitals object
///
/// Freeze the core orbitals out of an Orbitals object by removing these
/// orbitals from the object.
///
/// @param orbitals Orbitals object containing core and valence orbitals
/// @param n_active Number of active (spin) orbitals in each spin channel
///
/// @return Active Orbitals
inline auto freeze_core(
    const Orbitals &orbitals, const std::vector<size_t> &n_active
) {
  const auto n_spin = spin_channels(orbitals);
  if (spin_channels(n_active) != n_spin) {
    throw std::runtime_error(
        "Different number of spin channels in orbitals and n_active"
    );
  }

  Orbitals frozen_orbitals;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    frozen_orbitals.push_back(
        linalg::tail_cols(orbitals[sigma], n_active[sigma], true)
    );
  }

  return frozen_orbitals;
}

/// Construct the density matrix from the (occupation weighted) Orbitals
///
/// @param orbitals Orbitals object containing coefficients for each spin
/// channel
///
/// @return Density matrix for each spin channel
inline auto construct_density(const Orbitals &orbitals) {
  const auto n_spin = spin_channels(orbitals);

  const auto nao = linalg::n_rows(orbitals[0]);

  DensityMatrix result;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto &C_sigma = orbitals[sigma];

    if (const auto n_occ = linalg::n_cols(C_sigma); n_occ == 0) {
      result.push_back(linalg::zeros(nao, nao));
    } else {
      const auto n_electron_per_orbital = (n_spin == 1) ? 2 : 1;

      result.emplace_back(
          n_electron_per_orbital * C_sigma * linalg::transpose(C_sigma)
      );
    }
  }

  return result;
}

/// Construct occupation weighted orbitals from a set of orbitals and
/// occupations
///
/// @param orb Orbital coefficients
/// @param occ Occupation vectors
inline auto occupation_weighted_orbitals(
    const Orbitals &orb, const Occupations &occ
) {
  const auto n_spin = orb.size();
  if (occ.size() != n_spin) {
    throw std::runtime_error("orbitals and occupations must be of equal size");
  }

  Orbitals result(n_spin);
  const auto n_electron_per_orbital = (n_spin == 1) ? 2 : 1;
  for (size_t i = 0; i < n_spin; ++i) {
    if (!linalg::all_positive(occ[i])) {
      throw std::runtime_error("occupations must be positive");
    }
    const auto n_occ = occ[i].size();
    const auto C_occ = linalg::head_cols(orb[i], n_occ);
    const linalg::Vec weights = linalg::sqrt(occ[i] / n_electron_per_orbital);
    result[i] = C_occ * linalg::diagmat(weights);
  }

  return result;
}
}  // namespace uw12::utils

#endif  // UW12_UTILS_HPP
