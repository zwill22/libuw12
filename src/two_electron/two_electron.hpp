#ifndef UW12_two_electron_HPP
#define UW12_two_electron_HPP

#include "../integrals/integrals.hpp"
#include "../utils/utils.hpp"

namespace uw12::two_el {
/// Find the contribution to the energy and Fock matrix of the two
/// electron term of UW12.
///
/// The two electron UW12 Fock matrix and energy are calculated using
/// density fitting. In this method, the direct (+) and indirect (-)
/// contributions are calculated separately with the energies given by
/// \f{align*}{
/// E_{c, 2el,+}^{UW12} &= \frac{1}{2} \sum_{C} \sum_{ij} \left( ii \left|
/// w_{12}^{s_{ij}} r_{12}^{-1} \right| C \right) \left( \tilde{C} \left|
/// w_{12}^{s_{i j}} r_{12}^{-1} \right| jj \right) \newline
/// E_{c, 2el,-}^{UW12} &= - \frac{1}{2} \sum_{C} \sum_{ij} \left( ij \left|
/// w_{12}^{1} r_{12}^{-1} \right| C \right) \left( \tilde{C} \left| w_{12}^{1}
/// r_{12}^{-1} \right| ij \right),
/// \f}
/// where:
/// \f[
/// \left( \tilde{A} \left| w_{12}^{s_{i j}} r_{12}^{-1} \right| ij \right) =
/// \sum_A M_{AB}^{-1} \left( B \left| w_{12}^{s_{i j}} r_{12}^{-1} \right| ij
/// \right)
/// \f]
/// The corresponding Fock matrix contributions are given by:
/// \f{align*}{
/// \frac{\partial E_{c, 2el,+}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}} &=
/// \sum_{C} \sum_{j} \left( \alpha \beta \left|
/// w_{12}^{\delta_{\sigma_{j} \sigma}} r_{12}^{-1} \right| C \right) \left(
/// \tilde{C} \left| w_{12}^{\delta_{\sigma_{j} \sigma}} r_{12}^{-1} \right|
/// jj \right) \newline
/// \frac{\partial E_{c, 2el, -}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}} &=
/// - \sum_{j} \delta_{\sigma_{j} \sigma} \sum_{C} \left( \alpha j \left|
/// w_{12}^{1} r_{12}^{-1} \right| C \right) \left( \tilde{C} \left| w_{12}^{1}
/// r_{12}^{-1} \right| \beta j \right)
/// \f}
///
/// The indirect term contains only same-spin contributions. Therefore, for
/// \f$w^{s=1}(r) = 0\f$ (opposite spin only), there is no indirect
/// contribution.
///
/// @param WV Integrals \f$(ab|r^{-1}w(r)|A)\f$ and \f$(A|r^{-1}w(r)|B)\f$
/// @param active_Co Frozen core occupation weighted orbitals
/// @param indirect_term Whether to compute the indirect term
/// @param calculate_fock Whether to calculate the Fock matrix contribution
/// @param scale_opp_spin Scale factor for \f$ w^0 (r) \f$
/// @param scale_same_spin Scale factor for \f$ w^1 (r) \f$
///
/// @return Two electron UW12 Fock matrix and energy contribution
utils::FockMatrixAndEnergy form_fock_two_el_df(
    const integrals::BaseIntegrals &WV,
    const utils::Orbitals &active_Co,
    bool indirect_term,
    bool calculate_fock,
    double scale_opp_spin,
    double scale_same_spin
);
}  // namespace uw12::two_el

#endif  // UW12_two_electron_HPP
