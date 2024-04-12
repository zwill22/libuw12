//
// Created by Zack Williams on 12/02/2024.
//

#ifndef LIBUW12_HPP
#define LIBUW12_HPP

#include "src/integrals/base_integrals.hpp"
#include "src/three_electron/ri_utils.hpp"
#include "src/utils/utils.hpp"

namespace uw12 {

/// A Class which manages all two- and three- index density-fitting integrals
using integrals::BaseIntegrals;

/// A Struct which contains the four projectors corresponding to each ABS+ subset
using three_el::ri::ABSProjectors;

/// A vector storing occupation numbers for each spin channel
using utils::Occupations;

/// A vector storing orbital coefficients for each spin channel
using utils::Orbitals;

/// Initialise a set of BaseIntegrals for a given inter-electron potential
/// x_{12} using arrays for each integrals
///
/// @param X3 Ptr to three index integrals \f$(\rho\sigma | x_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param X2 Ptr to two index integrals \f$(A | x_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param X3_ri  Ptr to three index RI integrals \f$(\mu\rho|x_{12}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param n_ao Number of atomic orbitals
/// @param n_df Number of density-fitting orbitals
/// @param n_ri Number of auxilliary RI basis functions
/// @param copy_data Copy data when creating BaseIntegrals (default: false)
///
/// @return BaseIntegrals
BaseIntegrals setup_base_integrals(
    const double* X3,
    const double* X2,
    const double* X3_ri,
    size_t n_ao,
    size_t n_df,
    size_t n_ri,
    bool copy_data = false
);

/// Initialise a set of BaseIntegrals for a given inter-electron potential
/// x_{12} using arrays for each integrals.
///
/// @param X3 Ptr to three index integrals \f$(\rho\sigma | x_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param X2 Ptr to two index integrals \f$(A | x_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param n_ao Number of atomic orbitals
/// @param n_df Number of density-fitting orbitals
/// @param copy_data Copy data when creating BaseIntegrals (default: false)
///
/// @return BaseIntegrals (no RI)
BaseIntegrals setup_base_integrals(
    const double* X3,
    const double* X2,
    size_t n_ao,
    size_t n_df,
    bool copy_data = false
);

/// Setup the projectors for the ABS+ RI method
///
/// @param S Ptr to overlap matrix \f$(\mu | \nu)\f$ for the combined ao and
///          auxilliary RI space array of size `(n_ao + n_ri) * (n_ao + n_ri)`
/// @param n_ao Number of atomic orbitals
/// @param n_ri Number of auxilliary RI orbitals
///
/// @return ABSProjectors
ABSProjectors setup_abs_projectors(const double* S, size_t n_ao, size_t n_ri);

/// Setup orbitals from memory
///
/// @param C Ptr to orbital coefficients of size `n_ao * (2 * n_orb)`
/// @param n_ao Number of atomic orbitals
/// @param n_orb Number of molecular orbitals provided (per spin channel
/// @param n_spin Number of spin channels
///
/// @return Orbitals
Orbitals setup_orbitals(
    const double* C, size_t n_ao, size_t n_orb, size_t n_spin
);

/// Setup orbital occupations (Closed shell)
/// Occupations must correspond to the first `n_occ` orbitals in `C`
///
/// @param occ Occupation vector as an array of size `n_occ`
/// @param n_occ Number of occupied orbitals
///
/// @return Occupations
Occupations setup_occupations(const double* occ, size_t n_occ);

/// Setup orbital occupations (Open shell)
/// Occupations must correspond to orbital coefficient in `C`
///
/// @param occ Occupation vector as an array of size `n_occ`
/// @param n_occ_alpha Number of occupied orbitals in spin channel alpha
/// @param n_occ_beta Number of occupied orbitals in spin channel beta
///
/// @return Occupations
Occupations setup_occupations(
    const double* occ, size_t n_occ_alpha, size_t n_occ_beta
);

/// Calculate the UW12 energy only
///
/// @param W BaseIntegrals for \f$w_{12}\f$ (must include RI)
/// @param V BaseIntegrals for \f$r_{12}^{-1}\f$ (must include RI)
/// @param WV BaseIntegrals for \f$w_{12} r_{12}^{-1}\f$ (no RI)
/// @param abs_projectors Projectors for ABS+
/// @param orbitals Vector of orbitals
/// @param occ Occupation vectors for each spin channel
/// @param n_active Vector of number of active orbitals for each spin channel
/// @param scale_opp_spin Scale factor for osUW12
/// @param scale_same_spin Scale factor for ssUW12
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 energy
double uw12_energy(
    const BaseIntegrals& W,
    const BaseIntegrals& V,
    const BaseIntegrals& WV,
    const ABSProjectors& abs_projectors,
    const Orbitals& orbitals,
    const Occupations& occ,
    const std::vector<size_t>& n_active,
    double scale_opp_spin,
    double scale_same_spin,
    size_t print_level = 0
);

/// Calculate the UW12 fock matrix and energy
///
/// @param fock Resulting fock matrix for UW12
/// @param W BaseIntegrals for \f$w_{12}\f$ (must include RI)
/// @param V BaseIntegrals for \f$r_{12}^{-1}\f$ (must include RI)
/// @param WV BaseIntegrals for \f$w_{12} r_{12}^{-1}\f$ (no RI)
/// @param abs_projectors Projectors for ABS+
/// @param orbitals Vector of orbitals
/// @param occ Occupation vectors for each spin channel
/// @param n_active Vector of number of active orbitals for each spin channel
/// @param scale_opp_spin Scale factor for osUW12
/// @param scale_same_spin Scale factor for ssUW12
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 Fock matrix and energy
double uw12_fock(
    double* fock,
    const BaseIntegrals& W,
    const BaseIntegrals& V,
    const BaseIntegrals& WV,
    const ABSProjectors& abs_projectors,
    const Orbitals& orbitals,
    const Occupations& occ,
    const std::vector<size_t>& n_active,
    double scale_opp_spin,
    double scale_same_spin,
    size_t print_level = 0
);

}  // namespace uw12

#endif  // LIBUW12_HPP
