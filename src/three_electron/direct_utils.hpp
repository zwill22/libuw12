//
// Created by Zack Williams on 25/03/2024.
//

#ifndef THREE_ELECTRON_DIRECT_UTILS_HPP
#define THREE_ELECTRON_DIRECT_UTILS_HPP

#include "../integrals/integrals.hpp"
#include "../utils/utils.hpp"
#include "ri_utils.hpp"

namespace uw12::three_el {

/// Calculate matrices \f$X_{AB}^{\sigma}\f$ given by:
/// \f[
/// X_{AB} = \sum_{j \mu'} (A | w_{12} | j \mu')
/// \sum_{\nu'} [ S^{-1} ]_{\mu' \nu'} (\nu' j | r_{12}^{-1} | B)
/// \f]
/// for (active) occupied orbitals \f$j\f$, ri orbitals \f$\mu', \nu'\f$, and
/// density-fitting indices \f$A, B\f$. The ri indices run over the ao and
/// auxilliary ri basis sets.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
/// @param abs_projectors RI projectors \f$S^{-1}\f$
///
/// @return Matrices \f$X_{AB}^{\sigma}\f$
utils::MatVec calculate_xab(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const ri::ABSProjectors& abs_projectors
);

/// \brief Calculates the direct three-el energy
///
/// Direct three-el energy is given by:
/// \f[
/// E_c^{3el, +} = \sum_{\sigma} \sum_{AB} X_{AB}^{\sigma}
/// \tilde{t}_{AB}^{\sigma} \f] for matrices \f$X_{AB}^{\sigma}\f$ and
/// \f$\tilde{t}_{AB}^{\sigma}\f$.
///
/// @param x Matrices \f$X_{AB}^{\sigma}\f$
/// @param ttilde Matrices \f$\tilde{t}_{AB}^{\sigma}\f$
/// @param scale_opp_spin Opposite-spin scale factor \f$\kappa_0\f$
/// @param scale_same_spin Same-spin scale factor \f$\kappa_1\f$
///
/// @return Direct three-el energy
double calculate_direct_energy(
    const utils::MatVec& x,
    const utils::MatVec& ttilde,
    double scale_opp_spin,
    double scale_same_spin
);

/// Fock contribution from the derivative of \f$\tilde{t}_{AB}^{\sigma}\f$
///
/// Calculates the direct three-electron fock contribution corresponding to the
/// derivative of \f$\tilde{t}_{AB}^{\sigma}\f$ with matrix
/// \f$X_{AB}^{\sigma'}\f$, given by:
/// \f[
/// F_{\alpha\beta}^{\sigma\sigma'} = \sum_{AB} X_{AB}^{\sigma'}
/// \frac{d \tilde{t}_{AB}^{\sigma}}{d D_{\alpha_\beta}}
/// \f]
///
/// @param W3idx_one_trans Integrals \f$(j \rho | w_{12} |A)\f$
/// @param V3idx_one_trans Integrals \f$(j \rho | r_{12}^{-1} |A)\f$
/// @param xab Matrix \f$X_{AB}^{\sigma'}\f$
/// @param W_vals Vector of df eigenvalues for \f$(A| w_{12} |B)\f$
/// @param V_vals Vector of df eigenvalues for \f$(A| r_{12}^{-1} |B)\f$
/// @param n_active Number of active orbitals
/// @param n_spin Number of spin channels @param n_ao Number of atomic orbitals
///
/// @return Fock matrix contribution
linalg::Mat calculate_xab_dttilde(
    const linalg::Mat& W3idx_one_trans,
    const linalg::Mat& V3idx_one_trans,
    const linalg::Mat& xab,
    const linalg::Vec& W_vals,
    const linalg::Vec& V_vals,
    size_t n_active,
    size_t n_ao
);

/// Fock contribution from the derivative of \f$X_{AB}^{\sigma}\f$
///
/// Calculates the direct three-electron fock contribution corresponding to the
/// derivative of \f$X_{AB}^{\sigma}\f$ with matrix
/// \f$\tilde{t}_{AB}^{\sigma}\f$, given by:
/// \f[
/// F_{\alpha\beta}^{\sigma\sigma'} = \sum_{AB} \tilde{t}_{AB}^{\sigma'}
/// \frac{d X_{AB}^{\sigma}}{d D_{\alpha_\beta}}
/// \f]
///
/// @param W Integrals for \f$w_{12}\f$
/// @param W Integrals for \f$r_{12}^{-1}\f$
/// @param ttilde Matrix \f$\tilde{t}_{AB}^{\sigma'}\f$
/// @param abs_projectors RI projectors \f$S^{-1}\f$
///
/// @return Fock matrix contribution
linalg::Mat calculate_ttilde_dxab(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const linalg::Mat& ttilde,
    const ri::ABSProjectors& abs_projectors
);

}  // namespace uw12::three_el

#endif  // THREE_ELECTRON_DIRECT_UTILS_HPP
