#ifndef UW12_four_electron_HPP
#define UW12_four_electron_HPP

#include "integrals/integrals.hpp"

namespace uw12::four_el {

/// \brief Find the contribution to the energy and Fock matrix of the
/// four electron term of UW12.
///
/// Calculates the four electron UW12 Fock matrix and energy using density
/// fitting integrals.
///
/// In this method, the direct (+) and indirect (-) contributions are calculated
/// separately with the energies given by
/// \f{align*}{
/// E_{c, 4el,+}^{UW12} &= \frac{1}{2} \sum_{ijkl} \left( ik \left|
/// w_{12}^{s_{ij}} \right| jl \right) \left( ki \left| r_{12}^{-1} \right| lj
/// \right) \newline
/// &= \frac{1}{2} \sum_{Cik} \left( ik \left| w_{12}^{s_{ij}} \right| C \right)
/// \sum_{D} \left( ki \left| r_{12}^{-1} \right| D \right) \sum_{jl} \left(
/// \tilde{D} \left| r_{12}^{-1} \right| lj \right) \left( \tilde{C}  \left|
/// w_{12}^{s_{ij}} \right| jl \right) \newline
/// E_{c, 4 el, -}^{UW12} &= \frac{1}{2} \sum_{ijkl} \left( il \left| w_{12}^{1}
/// \right| jk \right) \left( ki \left| r_{12}^{-1} \right| lj \right),
/// \f}
/// where the indirect integrals must be calculated separately using density
/// fitting and
/// \f[
/// \left( \tilde{C} \left| x \right| i j \right) = \sum_{D} \left( C
/// \left| X \right\vert D \right)^{-1} \left(D \left| X \right| i j \right)
/// \f]
/// for potential \f$X\f$. For the frozen-core approximation, orbital indices i
/// and j run over active (valence) orbitals only, while k and l include the
/// core orbitals in the summation.
///
/// The corresponding Fock contributions are given by
/// \f{align*}{
/// \frac{\partial E_{c, 4 el, +}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}}
/// &= \sum_{C k} \left( \alpha k \left| w_{12}^{s_{ij}} \right| C \right)
/// \sum_{D} \left( k\beta \left| r_{12}^{-1} \right| D \right) \sum_{jl}
/// \left( \tilde{D} \left| r_{12}^{-1} \right| lj \right) \left( \tilde{C}
/// \left| w_{12}^{s_{ij}} \right| jl \right) \newline
/// &\qquad + \sum_{C i} \left( i \beta \left| w_{12}^{s_{ij}} \right| C \right)
/// \sum_{D} \left( \alpha i \left| r_{12}^{-1} \right| D \right) \sum_{jl}
/// \left( \tilde{D} \left| r_{12}^{-1} \right| lj \right) \left( \tilde{C}
/// \left| w_{12}^{s_{ij}} \right| jl \right) \newline
/// \frac{\partial E_{c, 4 el,-}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}}
/// &= - \sum_{jkl} \delta_{\sigma_{j} \sigma} \left( \alpha l \left|
/// w_{12}^{1} \right| jk \right) \left( k \beta \left| r_{12}^{-1} \right|
/// l j\right) - \sum_{ijk} \delta_{\sigma_{j} \sigma} \left( \alpha i \left|
/// w_{12}^{1} \right| kj \right) \left( j \beta \left| r_{12}^{-1} \right|
/// ik \right)
/// \f}
/// where the indirect four component integral must be calculated separately.
/// In the full-core Unsold-W12, both terms in each of these equations are
/// equal. However, for the frozen-core approximation they are not, since the
/// summations are now over different orbitals.
///
/// Since the four component integrals must be constructed separately for the
/// indirect term, this algorithm scales as \f$N^5\f$. However, the indirect
/// term is for same spin only. Therefore, for \f$w^{s=1}(r) = 0\f$ (opposite
/// spin only), this term is zero, and the algorithm is \f$N^4\f$.
///
/// \return Four electron UW12 Fock matrix and energy
utils::FockMatrixAndEnergy form_fock_four_el_df(
    const integrals::Integrals & W,   ///< \f$w(r)\f$ integrals
    const integrals::Integrals & V,   ///<\f$r^{-1}\f$ integrals
    bool indirect_term,    ///< Whether to calculate the indirect term
    bool calculate_fock,   ///< Whether to calculate the Fock matrix
    double scale_opp_spin, ///< Scale for opposite spin
    double scale_same_spin ///< Scale for same spin
);

}

#endif
