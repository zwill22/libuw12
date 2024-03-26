//
// Created by Zack Williams on 25/03/2024.
//

#ifndef THREE_ELECTRON_INDIRECT_UTILS_HPP
#define THREE_ELECTRON_INDIRECT_UTILS_HPP

#include "../integrals/integrals.hpp"
#include "ri_utils.hpp"

namespace uw12::three_el {
/// Calculates the indirect three electron energy
///
/// The indirect three electron energy is calculated using ABS+RI as:
/// \f[
/// E_c^{3el, -} = \sum_{ij} Y_{ij}
/// = \sum_{ij} \left( \sum_{AB} \tilde{t}_{AB}^{ij} X_{AB}^{ij} \right)
/// \f]
/// for (active) occupied indices \f$i,j\f$, and density-fitting indices
/// \f$A,B\f$, and where
/// \f[
/// \tilde{t}_{AB}^{ij} = \sum_k (\tilde{A}| w_{12} | jk)
/// (ki |r_{12}^{-1}|\tilde{B})
/// \f]
/// for (complete) occupied indices \f$k\f$, and
/// \f[
/// X_{AB}^{ij} = \sum_{\mu'\nu'} (A| w_{12}| i\mu') [S^{-1}]_{\mu'\nu'}
/// (\nu'j| r_{12}^{-1} |B)
/// \f]
/// with ri indices \f$\mu'\nu'\f$ over the complete ao+ri space.
///
/// Calculation of the energy is parallelised over indices \f$i, j\f$.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
/// @param abs_projectors RI projectors \f$S^{-1}\f$
///
/// @return Indirect three-electron energy
double indirect_3el_energy(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const ri::ABSProjectors& abs_projectors
);

/// Generalised version of indirect energy function with multiple integrals as
/// inputs (useful for testing)
double indirect_3el_energy(
    const integrals::Integrals& W1,
    const integrals::Integrals& V1,
    const integrals::Integrals& W2,
    const integrals::Integrals& V2,
    const ri::ABSProjectors& abs_projectors
);

/// Calculates the indirect fock matrix contribution
///
/// The indirect fock matrix contribution
/// \f[
/// F_{\alpha\beta}^{\sigma} = \frac{d E_c^{3el,-}}{d D_{\alpha\beta}^{\sigma}}
/// \f]
/// has three contributions from the three occupied orbital indices \f$i,j,k\f$.
/// These are given by:
/// \f{align*}{
/// f_{\alpha\beta}^{\sigma} (i) = \sum_{jk} \sum_{\mu'\nu'}
/// (jk|w_{12}|\alpha \mu') [S^{-1}]_{\mu'\nu'} (\nu' j| r_{12}^{-1}|k\beta)
/// \newline
/// f_{\alpha\beta}^{\sigma} (j) = \sum_{ik} \sum_{\mu'\nu'}
/// (\alpha k|w_{12}|i \mu') [S^{-1}]_{\mu'\nu'} (\nu' \beta| r_{12}^{-1}|ki)
/// \newline
/// f_{\alpha\beta}^{\sigma} (k) = \sum_{ij} \sum_{\mu'\nu'}
/// (j\beta|w_{12}|i \mu') [S^{-1}]_{\mu'\nu'} (\nu' j| r_{12}^{-1}|\alpha i)
/// \f}
/// Contributions are parallelised over the occupied indices.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
/// @param abs_projectors RI projectors \f$S^{-1}\f$
/// @param sigma Spin index
///
/// @return Unsymmetrised indirect fock contribution for single spin channel
linalg::Mat indirect_3el_fock_matrix(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const ri::ABSProjectors& abs_projectors,
    size_t sigma
);

}  // namespace uw12::three_el

#endif  // THREE_ELECTRON_INDIRECT_UTILS_HPP
