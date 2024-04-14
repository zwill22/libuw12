//
// Created by Zack Williams on 09/04/2024.
//

#ifndef UW12_HPP
#define UW12_HPP

#include "integrals/base_integrals.hpp"
#include "three_electron/ri_utils.hpp"
#include "utils/utils.hpp"

/// @mainpage
/// Functionality relating to the UW12 correlation method
///
/// The UW12 correlation energy is defined as:
/// \f[
/// E_c^{UW12} = \frac{1}{2} \sum_{ij ab} \left\langle i j \left\vert
/// w^{s_{ij}}_{12} \right\vert \overline{ab} \right\rangle \left\langle a b
/// \left\vert r_{12}^{-1}\right\vert i j \right\rangle
/// \f]
/// Where \f$\vert \overline{ab} \rangle = \vert ab \rangle - \vert ba
/// \rangle\f$, \f$ij\f$ are occupied orbitals, \f$ab\f$ are unoccupied
/// (virtual) orbitals, \f$ r_{12} = \vert \vec{r}_1 - \vec{r}_2 \vert \f$, with
/// two-electron geminal operator \f$ w^s_{12} \f$ for total spin \f$s_{ij} =
/// \delta_{\sigma_i \sigma_j}\f$.
///
/// By removing the sum over the virtual orbitals, the energy can be rewritten
/// as
/// \f[
/// E_c^{UW12} = E_{c, 2el}^{UW12} + E_{c, 3el}^{UW12} + E_{c, 4el}^{UW12},
/// \f]
/// for two, three and four electron terms defined by:
/// \f{align*}{
/// E_{c, 2el}^{UW12} &= \frac{1}{2} \sum_{ij} \left\langle \overline{ij}
/// \left\vert w^{s_{ij}}_{12} r_{12}^{-1}\right\vert i j \right\rangle \newline
/// E_{c, 3el}^{UW12} &= - \sum_{ijk} \left\langle \overline{ij} k \left\vert
/// w^{s_{ij}}_{12} r_{23}^{-1}\right\vert k j i \right\rangle \newline
/// E_{c, 4 el}^{UW12} &= \frac{1}{2} \sum_{ij kl} \left\langle i j \left\vert
/// w^{s_{ij}}_{12}\right\vert \overline{kl} \right\rangle \left\langle kl
/// \left\vert r_{12}^{-1} \right\vert ij \right\rangle
/// \f}
///
/// In the frozen core approximation, indices i,j run only over the correlated
/// (active) orbitals; whilst k,l indices summed over all occupied orbitals
/// including the frozen core orbitals.
///
/// ### Fock contributions
///
/// Since UW12 is self-consistent, the Fock matrix elements
/// \f$F_{\alpha \beta}^{\sigma}\f$ may also be calculated, where
/// \f[
/// F_{\alpha \beta}^{\sigma} = \frac{1}{2} \left[ \frac{\partial E}{\partial
/// D_{\alpha \beta}^{\sigma}} + \frac{\partial E}{\partial
/// D_{\beta \alpha}^{\sigma}} \right],
/// \f]
/// for real symmetric density matrix \f$D_{\alpha \beta}^{\sigma}\f$. The
/// contributions for which are given by:
/// \f{align*}{
/// \frac{\partial E_{c, 2 el}^{UW 12}}{\partial D_{\alpha \beta}^{\sigma}}
/// &= \sum_{j} \left\langle \overline{\alpha j} \left\vert
/// w^{\delta_{\sigma_j \sigma}}_{12} r_{12}^{-1}\right\vert \beta j
/// \right\rangle \newline
/// \frac{\partial E_{c, 3 el}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}}
/// &= - \sum_{jk} \left\langle \overline{\alpha j} k \left\vert w_{12}
/// r_{23}^{-1} \right\vert k j \beta \right\rangle - \sum_{i k} \left\langle
/// \overline{i \alpha} k \left\vert w_{12} r_{23}^{-1}\right\vert k \beta i
/// \right\rangle - \sum_{i j} \left\langle \overline{i j} \alpha
/// \left\vert w_{12} r_{23}^{-1}\right\vert \beta j i \right\rangle \newline
/// \frac{\partial E_{c, 4el}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}} &=
/// \sum_{jkl} \left\langle \alpha j \left\vert w^{\delta_{\sigma_j
/// \sigma}}_{12} \right\vert \overline{k l} \right\rangle \left\langle k l
/// \left\vert r_{12}^{-1}\right\vert \beta j \right\rangle + \sum_{ijk}
/// \left\langle i j \left\vert w^{\delta_{\sigma_j \sigma}}_{12} \right\vert
/// \overline{\alpha k} \right\rangle \left\langle k \beta \left\vert
/// r_{12}^{-1} \right\vert i j \right\rangle
/// \f}
///
/// In the full core version, the two terms in the four electron term are the
/// same. This is a result of the symmetry between orbitals i,j and k,l.
/// Similarly for the direct portion of the first and third terms in the three
/// electron Fock contribution. However, for frozen core the terms are no longer
/// the same and must be calculated separately.
///
/// ### Geminal Function
///
/// The geminal function \f$w_{12}\f$ is given as a function of the
/// inter-electron distance \f$r_{12}\f$. The function is implemented as a sum
/// of Gaussian terms
/// \f[
/// w^s (r) = \sum_i c_i^s \exp \left[ - \gamma_i r^2 \right]
/// \f]
/// specified by coefficients \f$c_i^s\f$ and exponents \f$\gamma_i\f$. The
/// structure of the theory allows two sets of coefficients for opposite
/// (\f$s = 0\f$) and same (\f$s = 1\f$) spin contributions. Though the standard
/// implementation has \f$w^{s=1} (r) = \kappa w^{s=0} (r)\f$ for same spin
/// scale factor \f$\kappa_1\f$.
namespace uw12 {

/// Find the contribution to the energy and Fock matrix of UW12
///
/// This function calculates the UW12 energy and Fock matrix contribution
/// by calculating the two, three and four electron components. The two and four
/// electron components are calculated using density fitting, the integrals for
/// which are given to this function. The three electron term is calculated
/// using density-fitted RI.
///
/// @param W Density-fitting integrals and eigenvalues for \f$w_{12}\f$
/// @param V Density-fitting integrals and eigenvalues for \f$v_{12}\f$
/// @param WV Density-fitting integrals for \f$w_{12}v_{12}\f$ (No RI)
/// @param abs_projectors Projectors for ABS+ (inverse overlaps matrices between
///                       ri and ao space)
/// @param orbitals Orbital coefficients in the form of a std::vector of
///                 matrices with number of rows specified by the size number
///                 of ao basis functions and number of columns specified by
///                 number of (occupied) orbitals with one matrix for each
///                 spin channel. The first `n_occ` columns are treated as the
///                 occupied orbital coefficients, with `n_occ` determined by
///                 `occupations`.
/// @param occupations Occupation numbers for the orbitals in the form of a
///                    std::vector of Vecs of size `n_occ` for each spin channel
/// @param n_active Vector of the number of active orbitals in each spin channel
/// @param indirect_term Calculate the indirect (exchange) term
/// @param calculate_fock Calculate the Fock matrix contribution
/// @param scale_opp_spin Scale \f$w^0 (r)\f$ by a factor \f$\kappa_0\f$
/// @param scale_same_spin Scale \f$w^1 (r)\f$ by a factor \f$\kappa_1\f$
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return Fock matrix and energy for UW12
utils::FockMatrixAndEnergy form_fock(
    const integrals::BaseIntegrals &W,
    const integrals::BaseIntegrals &V,
    const integrals::BaseIntegrals &WV,
    const three_el::ri::ABSProjectors &abs_projectors,
    const utils::Orbitals &orbitals,
    const utils::Occupations &occupations,
    const std::vector<size_t> &n_active,
    bool indirect_term,
    bool calculate_fock,
    double scale_opp_spin,
    double scale_same_spin,
    size_t print_level = 0
);

}  // namespace uw12

#endif
