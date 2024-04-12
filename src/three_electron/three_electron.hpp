//
// Created by Zack Williams on 25/03/2024.
//

#ifndef THREE_ELECTRON_HPP
#define THREE_ELECTRON_HPP

#include "integrals/integrals.hpp"
#include "ri_utils.hpp"

namespace uw12::three_el {

/// Calculates the UW12 three electron energy and Fock matrix
/// contribution using a density-fitted RI approximation
///
/// This algorithm calculates the UW12 three-electron energy given by:
/// \f[
/// E_{c, 3el}^{UW12} = - \left\langle \overline{ij} k \left\vert
/// w^{s_{ij}}_{12} r_{23}^{-1}\right\vert k j i \right\rangle
/// \f]
/// (repeated index summation)
///
/// Using the RI approximation, these integrals may be approximated as
/// \f[
/// E_{c, 3el}^{UW12, RI} = - \left\langle \overline{ij} \left\vert
/// w^{s_{ij}}_{12} \right\vert k q \right\rangle
/// \left\langle k q \left\vert r_{12}^{-1} \right\vert ij \right\rangle,
/// \f]
/// using the resolution of the identity \f$| q ><q | \approx 1\f$.
///
/// ### Resolution of the identity
///
/// To construct the resolution of the identity, an ABS+ approach is used:
/// https://doi.org/10.1016/j.cplett.2004.07.061
/// In this approach, the orthonormal basis q is constructed from the union of
/// the ao basis and an auxiliary basis set (ABS) `ri`. This basis is not
/// constructed explicitly but calculated using the inverse overlap for the
/// combined ao and ri space. The projector may be written as
/// \f$|q \rangle \langle q | = |\tilde{\mu} \rangle
/// [S^{-1}]_{\tilde{\mu}\tilde{\nu}} \langle \tilde{\nu} |\f$
/// for indices \f$\tilde{\mu},\tilde{\nu}\f$ in the union of the two basis
/// sets. After computing the inverse of \f$S\f$ for the full union using
/// singular value decomposition to remove linearly dependent functions, the
/// inverse may be split into four sub-matrices for each possible component
/// basis combination: \f[ | q \rangle \langle q | = | \mu \rangle
/// [S^{-1}]_{\mu\nu} \langle \nu |
/// + | \rho \rangle [S^{-1}]_{\rho\nu} \langle \nu |
/// + | \mu \rangle [S^{-1}]_{\mu \sigma} \langle \sigma |
/// + | \rho \rangle [S^{-1}]_{\rho\sigma} \langle \sigma |
/// \f]
/// for \f$\mu, \nu\f$ in `ri` and \f$\rho, \sigma\f$ in `ao`. The projectors
/// \f$S^{-1}\f$ are taken from `abs_projectors`.
///
/// In order to evaluate the integrals efficiently, a density-fitting approach
/// is used to approximate the two-electron integrals. Therefore only
/// three-index objects must be stored.
///
/// ### Direct Energy
///
/// The direct energy is then given by:
/// \f[
/// E_{c, 3el, +}^{UW12} = - X_{AB} \tilde{t}_{AB}
/// \f]
/// for density fitting basis indices \f$A,B\f$, where:
/// \f{align*}{
/// X_{AB} &= (A|w_{12}|j \mu) [S^{-1}]_{\mu\nu} (\nu j| r_{12}^{-1} |B)
/// + (A|w_{12}|j \rho) [S^{-1}]_{\rho\nu} (\nu j| r_{12}^{-1} |B) \newline
/// &+ (A|w_{12}|j \mu) [S^{-1}]_{\mu\sigma} (\sigma j| r_{12}^{-1} |B)
/// + (A|w_{12}|j \rho) [S^{-1}]_{\rho\sigma} (\sigma j| r_{12}^{-1} |B)
/// \f}
/// for active (occupied) orbitals `j`. Similarly:
/// \f[
/// \tilde{t}_{AB} = (\tilde{A} | w_{12} | ik ) ( ki | r_{12}^{-1} | \tilde{B})
/// \f]
/// for active occupied orbitals `i` and all occupied orbitals `k` with
/// \f$(\tilde{A} | w_{12} | ik ) = (A|w_{12}|B)^{-1} (B|w_{12}|ik)\f$.
/// Three-index integrals \f$(A|x_{12}|j\rho)\f$ and \f$(A|x_{12}|ik)\f$ are the
/// one and two mo-transformed integrals respectively, while the three-index
/// mo-transformed `ri` integrals are \f$(A|w_{12}|j \mu)\f$.
///
/// ### Direct Fock
///
/// The direct fock matrix contribution is calculated from:
/// \f[
/// \frac{\partial E_{c, 3el,+}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}} =
/// \frac{\partial X_{AB}}{\partial D_{\alpha \beta}^{\sigma}} \tilde{t}_{AB}
/// + X_{AB} \frac{\partial \tilde{t}_{AB}}{\partial D_{\alpha \beta}^{\sigma}}
/// \f]
/// where
/// \f{align*}{
/// \frac{\partial X_{AB}}{\partial D_{\alpha \beta}^{\sigma}} &=
/// (A|w_{12}|\alpha \mu) [S^{-1}]_{\mu\nu} (\nu \beta| r_{12}^{-1} |B)
/// + (A|w_{12}|\alpha \rho) [S^{-1}]_{\rho\nu} (\nu \beta| r_{12}^{-1} |B)
/// \newline
/// &+ (A|w_{12}|\alpha \mu) [S^{-1}]_{\mu\sigma} (\sigma \beta| r_{12}^{-1} |B)
/// + (A|w_{12}|\alpha \rho) [S^{-1}]_{\rho\sigma} (\sigma \beta| r_{12}^{-1}
/// |B) \f} and \f[ \frac{\partial \tilde{t}_{AB}}{\partial D_{\alpha
/// \beta}^{\sigma}} =
/// (\tilde{A} | w_{12} | \alpha k ) ( k \beta | r_{12}^{-1} | \tilde{B})
/// +  (\tilde{A} | w_{12} | i \beta ) ( \alpha i | r_{12}^{-1} | \tilde{B})
/// \f]
///
/// ### Indirect Energy
///
/// The indirect energy is given by:
/// \f[
/// E_{c, 3el, -}^{UW12} = - \sum_{ij} \sum_{AB} X_{AB}^{ij} \tilde{t}_{AB}^{ij}
/// \f]
/// for density fitting basis indices \f$A,B\f$, where:
/// \f{align*}{
/// X_{AB}^{ij} &= \sum_{\mu \nu}
/// (A|w_{12}|i \mu) [S^{-1}]_{\mu\nu} (\nu j| r_{12}^{-1} |B)
/// + (A|w_{12}|i \rho) [S^{-1}]_{\rho\nu} (\nu j| r_{12}^{-1} |B) \newline
/// &+ (A|w_{12}|i \mu) [S^{-1}]_{\mu\sigma} (\sigma j| r_{12}^{-1} |B)
/// + (A|w_{12}|i \rho) [S^{-1}]_{\rho\sigma} (\sigma j| r_{12}^{-1} |B)
/// \f}
/// \f[
/// \tilde{t}_{AB}^{ij} = \sum_{k} (\tilde{A} | w_{12} | jk )
/// ( ki | r_{12}^{-1} | \tilde{B})
/// \f]
/// This operation is performed using parallelisation over indices i,j.
///
/// ### Indirect Fock
///
/// Writing
/// \f$|q \rangle\langle q| = |\mu' \rangle[S^{-1}]_{\mu'\nu'}\langle \nu'|\f$,
/// where \f$\mu',\nu'\f$ represent the complete set of ri basis function (ao
/// and aux indices), the indirect fock contribution is given by (in four-index
/// notation):
/// \f{align*}{
/// \frac{\partial E_{c, 3el,-}^{UW12}}{\partial D_{\alpha \beta}^{\sigma}} &=
/// \sum_{jk} \sum_{\mu' \nu'} (jk | w_{12} | \alpha \mu' ) [S^{-1}]_{\mu'\nu'}
/// (\nu' j | r_{12}^{-1} | k \beta) \newline
/// &+ \sum_{ik} \sum_{\mu'\nu'} (\alpha k | w_{12} | i \mu')
/// [S^{-1}]_{\mu'\nu'}
/// (\nu' \beta | r_{12}^{-1} | k i) \newline
/// &+ \sum_{ij} \sum_{\mu'\nu'}( j \beta | w_{12} | i\mu')[S^{-1}]_{\mu' \nu'}
/// (\nu' j | r_{12}^{-1} | \alpha i )
/// \f}
/// Where we refer to these separate terms as the i, j, and k contributions
/// corresponding to the index which has been removed from the summation by
/// differentiation. Each of these terms is made up of four separate terms
/// corresponding to the four combinations of ri basis components:
/// \f[
/// |\mu' \rangle[S^{-1}]_{\mu'\nu'}\langle \nu'| =
/// | \mu \rangle [S^{-1}]_{\mu\nu} \langle \nu |
/// + | \rho \rangle [S^{-1}]_{\rho\nu} \langle \nu |
/// + | \mu \rangle [S^{-1}]_{\mu \sigma} \langle \sigma |
/// + | \rho \rangle [S^{-1}]_{\rho\sigma} \langle \sigma |
/// \f]
/// for \f$\mu, \nu\f$ in aux, \f$\rho, \sigma\f$ in ao, and \f$\mu', \nu'\f$
/// in the union of the two. Four-index integrals are evaluated from the
/// density fitting integrals for a single pair ij, ik, or jk with
/// parallelisation over these.
///
/// @param W Integrals for \f$w_{12}\f$
/// @param V Integrals for \f$r_{12}^{-1}\f$
/// @param abs_projectors projectors \f$S^{-1}\f$ for each subspace
/// @param indirect_term calculate the indirect term
/// @param calculate_fock calculate the fock matrix contribution
/// @param scale_opp_spin scale factor for opposite spin contribution
/// @param scale_same_spin scale factor for same spin contribution
///
/// @return Fock matrix and energy contributions for the three electron term
utils::FockMatrixAndEnergy form_fock_three_el_term_df_ri(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const ri::ABSProjectors& abs_projectors,
    bool indirect_term,
    bool calculate_fock,
    double scale_opp_spin,
    double scale_same_spin
);

}  // namespace uw12::three_el

#endif  // THREE_ELECTRON_HPP
