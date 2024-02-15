//
// Created by Zack Williams on 01/12/2020.
//

#ifndef UW12_UTILS_HPP
#define UW12_UTILS_HPP

#include <vector>

#include "linalg.hpp"

namespace uw12 {

  /// Generate a random positive-definite square matrix of size `n_row * n_row`
  inline linalg::Mat random_pd(const size_t n_row, const int seed) {
    using namespace linalg;
    const Vec eigen = random(n_row, 1, seed) + ones(n_row);
    const Mat X = random(n_row, n_row, seed);

    return X * transpose(X) + id(n_row);
  }

/// Matrix vector for storing multiple matrices in a single object
using MatVec = std::vector<linalg::Mat>;

/// Scalar multiplication of MatVec
/// {
inline MatVec operator*(const MatVec &object, const double a) {
  MatVec result(object.size());
  for (int i = 0; i < object.size(); ++i) {
    result[i] = a * object[i];
  }
  return result;
}

inline MatVec operator*(const double a, const MatVec &object) {
  return object * a;
}
/// }

/// Fock matrix addition {
inline MatVec &operator+=(MatVec &lhs, const MatVec &rhs) {
  const auto n = lhs.size();
  if (rhs.size() != n) {
    throw std::logic_error("containers are of different sizes");
  }

  for (int i = 0; i < n; ++i) {
    lhs[i] = lhs[i] + rhs[i];
  }
  return lhs;
}

inline MatVec operator+(MatVec left, const MatVec &right) {
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
inline FockMatrixAndEnergy operator*(const FockMatrixAndEnergy &fock,
                                     const double a) {
  return {a * fock.fock, a * fock.energy};
}

inline FockMatrixAndEnergy operator*(const double a,
                                     const FockMatrixAndEnergy &fock) {
  return fock * a;
}
/// }

/// FockMatrixAndEnergy addition {
inline FockMatrixAndEnergy operator+=(FockMatrixAndEnergy &lhs,
                                      const FockMatrixAndEnergy &rhs) {
  lhs.fock += rhs.fock;
  lhs.energy += rhs.energy;

  return lhs;
}

inline FockMatrixAndEnergy operator+(FockMatrixAndEnergy lhs,
                                     const FockMatrixAndEnergy &rhs) {
  lhs += rhs;
  return lhs;
}
/// }

/// \brief Check whether a value is approximately zero
///
/// \tparam T
/// \param a Value to check
/// \param epsilon Multiplicative factor to control numerical limit
///
/// \return Whether value is nearly zero
template <typename T>
bool nearly_zero(T a, int epsilon = 4) {
  return std::abs(a) <= epsilon * std::numeric_limits<T>::epsilon();
}

/// \brief Calculate and check the number of spin channels of an object
///
/// \tparam T
/// \param object Object of size `n_spin`
///
/// \return Number of spin channels `n_spin`
template <typename T>
size_t spin_channels(const T &object) {
  const size_t n_spin = object.size();
  if (n_spin < 1 || n_spin > 2) {
    throw std::logic_error("invalid number of spin channels");
  }

  return n_spin;
}

/// \brief Make Fock matrix symmetric (inplace)
///
/// Using \f$0.5 (F + F^T)\f$, make the Fock matrix in a FockMatrixAndEnergy
/// object symmetric
///
/// \param fock A FockMatrixAndEnergy object
///
/// \return Symmetric FockMatrixAndEnergy
inline FockMatrixAndEnergy symmetrise_fock(FockMatrixAndEnergy fock) {
  const auto n_spin = spin_channels(fock.fock);

  for (size_t sigma = 0; sigma < n_spin; sigma++) {
    fock.fock[sigma] =
        0.5 * (fock.fock[sigma] + linalg::transpose(fock.fock[sigma]));
  }

  return fock;
}

/// \brief Freeze core orbitals of an Orbitals object
///
/// Freeze the core orbitals out of an Orbitals object by removing these
/// orbitals from the object.
///
/// \param orbitals Orbitals object containing core and valence orbitals
/// \param n_core Number of core (spatial) orbitals
///
/// \return Active Orbitals
inline Orbitals freeze_core(const Orbitals &orbitals, const size_t n_core) {
  Orbitals frozen_orbitals;
  for (const auto &channel : orbitals) {
    const auto n_orb = linalg::n_cols(channel);
    if (n_core > n_orb) {
      throw std::logic_error(
          "insufficient orbitals for number of core orbitals");
    }

    frozen_orbitals.push_back(linalg::tail_cols(channel, n_orb - n_core, true));
  }

  return frozen_orbitals;
}

/// \brief Construct the density matrix from the (occupation weighted) Orbitals
///
/// \param orbitals Orbitals object containing coefficients for each spin
/// channel
///
/// \return Density matrix for each spin channel
inline DensityMatrix construct_density(const Orbitals &Co) {
  const auto n_spin = spin_channels(Co);

  const auto nao = linalg::n_rows(Co[0]);

  DensityMatrix result;
  for (int sigma = 0; sigma < n_spin; ++sigma) {
    const auto &C_sigma = Co[sigma];

    const auto n_occ = linalg::n_cols(C_sigma);

    if (n_occ == 0) {
      result.push_back(linalg::zeros(nao, nao));
    } else {
      const auto n_electron_per_orbital = (n_spin == 1) ? 2 : 1;

      result.push_back(n_electron_per_orbital * C_sigma *
                       linalg::transpose(C_sigma));
    }
  }

  return result;
}

/// \brief Construct occupation weighted orbitals from a set of orbitals and
/// occupations
///
/// \param orb Orbital coefficients
/// \param occ Occupation vectors
inline Orbitals occupation_weighted_orbitals(const Orbitals &orb,
                                             const Occupations &occ) {
  const auto n_spin = orb.size();
  if (occ.size() != n_spin) {
    throw std::runtime_error("orbitals and occupations must be of equal size");
  }

  Orbitals result(n_spin);
  const auto oo_n_electron_per_orbital = (n_spin == 1) ? 0.5 : 1.0;
  for (auto i = 0; i < n_spin; ++i) {
    if (!linalg::all_positive(occ[i])) {
      throw std::runtime_error("occupations must be positive");
    }
    const auto n_occ = occ[i].size();
    const auto C_occ = linalg::head_cols(orb[i], n_occ);
    const linalg::Vec weights =
        linalg::sqrt(occ[i] * oo_n_electron_per_orbital);
    result[i] = C_occ * linalg::diagmat(weights);
  }

  return result;
}

}  // namespace uw12

#endif  // UW12_UTILS_HPP
