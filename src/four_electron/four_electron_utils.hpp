#ifndef UW12_FOUR_ELECTRON_UTILS_HPP
#define UW12_FOUR_ELECTRON_UTILS_HPP

#include "../integrals/integrals.hpp"
#include "../utils/utils.hpp"

namespace uw12::four_el {

/// Calculate energy spin factor given by:
/// \f{align*}{
/// f_{n_{spin} = 1} &= 2 * (\kappa_0 + \kappa_1) \newline
/// f_{n_{spin} = 2} &= \kappa_{\delta_{\sigma \sigma'}}
/// \f}
///
/// @param n_spin Number of spin channels
/// @param sigma Spin \f$\sigma\f$
/// @param sigmaprime  Spin \f$\sigma'\f$
/// @param scale_opp_spin Opposite-spin scale factor \f$\kappa_0\f$
/// @param scale_same_spin Same-spin scale factor \f$\kappa_1\f$
///
/// @return Energy spin factor
double get_energy_spin_factor(
    size_t n_spin,
    size_t sigma,
    size_t sigmaprime,
    double scale_opp_spin,
    double scale_same_spin
);

/// Calculate matrices \f$t_{AB}^{\sigma}\f$ given by:
/// \f[
/// t_{AB} = \sum_{j k} (A | w_{12} | k j) (j k | r_{12}^{-1} | B)
/// \f]
/// for (active) occupied orbitals \f$j\f$, occupied orbitals \f$k\f$,
/// and (eigen) density-fitting indices \f$A, B\f$.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
///
/// @return Matrices \f$t_{AB}^{\sigma}\f$
utils::MatVec calculate_tab(
    const integrals::Integrals& W, const integrals::Integrals& V
);

/// Calculate matrices \f$\tilde{t}_{AB}^{\sigma}\f$ given by:
/// \f[
/// \tilde{t}_{AB} = \sum_{j k} (\tilde{A} | w_{12} | k j)
/// (j k | r_{12}^{-1} | \tilde{B}) = w_A t_{AB} v_{B}
/// \f]
/// for (active) occupied orbitals \f$j\f$, occupied orbitals \f$k\f$,
/// and density-fitting indices \f$A, B\f$.
///
/// Transformed integrals are given by:
/// \f[
/// (\tilde{A} | w_{12} | k j) = w_A (A | w_{12} | kj )
/// \f]
/// where \f$w_A\f$ are the density-fitting eigenvalues of \f$(A|w_{12}|B)\f$,
/// and similarly for \f$v_B\f$.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
/// @param tab \f$\sum_{j k} (A | w_{12} | k j) (j k | r_{12}^{-1} | B)\f$
///
/// @return Matrices \f$\tilde{t}_{AB}^{\sigma}\f$
/// {
utils::MatVec calculate_ttilde(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const utils::MatVec& tab
);

utils::MatVec calculate_ttilde(
    const integrals::Integrals& W, const integrals::Integrals& V
);
/// }

}  // namespace uw12::four_el

#endif
