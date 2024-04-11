//
// Created by Zack Williams on 12/02/2024.
//

#ifndef LIBUW12_HPP
#define LIBUW12_HPP

namespace uw12 {

/// Calculate the UW12 energy only (Closed shell)
///
/// @param W3 Ptr to three index integrals \f$(\rho\sigma | w_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param W2 Ptr to two index integrals \f$(A | w_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param W3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param V3 Ptr to three index integrals \f$(\rho\sigma|r_{12}^{-1}|A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param V2 Ptr to two index integrals \f$(A | r_{12}^{-1} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param V3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param WV3  Ptr to three index integrals
///             \f$(\rho\sigma|w_{12}r_{12}^{-1}|A)\f$ given by an array of size
///             `n_ao * (n_ao + 1) /2 * n_df`
/// @param WV2  Ptr to two index integrals \f$(A|w_{12}r_{12}^{-1}|B)\f$ given
///             by an array of size `n_df * n_df`
/// @param S2 Ptr to overlap matrix \f$(\mu | \nu)\f$ for the combined ao and
///           auxilliary RI space array of size `(n_ao + n_ri) * (n_ao + n_ri)`
/// @param C  Ptr to orbital coefficients of size `n_ao * n_orb`
/// @param occ Ptr to array of occupation number of size `n_occ`
/// @param n_ao Number of atomic orbital basis functions
/// @param n_df Number of density-fitting basis functions
/// @param n_ri Number of auxilliary RI basis functions
/// @param n_orb Number of molecular orbitals provided
/// @param n_occ Number of occupied orbitals
/// @param n_active Number of active orbitals
/// @param scale_same_spin Scale factor for same spin UW12 (if zero, an osUW12
///                        calculation is performed)
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 energy
double uw12_energy(
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    size_t n_ao,
    size_t n_df,
    size_t n_ri,
    size_t n_orb,
    size_t n_occ,
    size_t n_active,
    double scale_same_spin,
    size_t print_level = 0
);

/// Calculate the UW12 energy only (Closed shell)
///
/// @param W3 Ptr to three index integrals \f$(\rho\sigma | w_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param W2 Ptr to two index integrals \f$(A | w_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param W3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param V3 Ptr to three index integrals \f$(\rho\sigma|r_{12}^{-1}|A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param V2 Ptr to two index integrals \f$(A | r_{12}^{-1} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param V3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param WV3  Ptr to three index integrals
///             \f$(\rho\sigma|w_{12}r_{12}^{-1}|A)\f$ given by an array of size
///             `n_ao * (n_ao + 1) /2 * n_df`
/// @param WV2  Ptr to two index integrals \f$(A|w_{12}r_{12}^{-1}|B)\f$ given
///             by an array of size `n_df * n_df`
/// @param S2 Ptr to overlap matrix \f$(\mu | \nu)\f$ for the combined ao and
///           auxilliary RI space array of size `(n_ao + n_ri) * (n_ao + n_ri)`
/// @param C  Ptr to orbital coefficients of size `n_ao * (2 * n_orb)`
/// @param occ Ptr to array of occupation number of size `n_occ_alpha +
///            n_occ_beta`
/// @param n_ao Number of atomic orbital basis functions
/// @param n_df Number of density-fitting basis functions
/// @param n_ri Number of auxilliary RI basis functions
/// @param n_orb Number of molecular orbitals provided in each spin channel
/// @param n_occ_alpha Number of occupied orbitals in spin channel alpha
/// @param n_occ_beta Number of occupied orbitals in spin channel beta
/// @param n_active_alpha Number of active orbitals in spin channel alpha
/// @param n_active_beta Number of active orbitals in spin channel beta
/// @param scale_same_spin  Scale factor for same spin UW12 (if zero, an osUW12
///                         calculation is performed)
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 energy
double uw12_energy(
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    size_t n_ao,
    size_t n_df,
    size_t n_ri,
    size_t n_orb,
    size_t n_occ_alpha,
    size_t n_occ_beta,
    size_t n_active_alpha,
    size_t n_active_beta,
    double scale_same_spin,
    size_t print_level = 0
);

/// Calculate the UW12 fock matrix and energy (Closed shell)
///
/// @param fock Ptr to Fock matrix of size `n_ao * n_ao` to be filled
/// @param W3 Ptr to three index integrals \f$(\rho\sigma | w_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param W2 Ptr to two index integrals \f$(A | w_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param W3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param V3 Ptr to three index integrals \f$(\rho\sigma|r_{12}^{-1}|A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param V2 Ptr to two index integrals \f$(A | r_{12}^{-1} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param V3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param WV3  Ptr to three index integrals
///             \f$(\rho\sigma|w_{12}r_{12}^{-1}|A)\f$ given by an array of size
///             `n_ao * (n_ao + 1) /2 * n_df`
/// @param WV2  Ptr to two index integrals \f$(A|w_{12}r_{12}^{-1}|B)\f$ given
///             by an array of size `n_df * n_df`
/// @param S2 Ptr to overlap matrix \f$(\mu | \nu)\f$ for the combined ao and
///           auxilliary RI space array of size `(n_ao + n_ri) * (n_ao + n_ri)`
/// @param C  Ptr to orbital coefficients of size `n_ao * n_orb`
/// @param occ Ptr to array of occupation number of size `n_occ`
/// @param n_ao Number of atomic orbital basis functions
/// @param n_df Number of density-fitting basis functions
/// @param n_ri Number of auxilliary RI basis functions
/// @param n_orb Number of molecular orbitals provided
/// @param n_occ Number of occupied orbitals
/// @param n_active Number of active orbitals
/// @param scale_same_spin Scale factor for same spin UW12 (if zero, an osUW12
///                        calculation is performed)
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 energy
double uw12_fock(
    double* fock,
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    size_t n_ao,
    size_t n_df,
    size_t n_ri,
    size_t n_orb,
    size_t n_occ,
    size_t n_active,
    double scale_same_spin,
    size_t print_level
);

/// Calculate the UW12 fock matrix and energy (Open shell)
///
/// @param fock Ptr to Fock matrix of size `2 * n_ao * n_ao` to be filled
/// @param W3 Ptr to three index integrals \f$(\rho\sigma | w_{12} | A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param W2 Ptr to two index integrals \f$(A | w_{12} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param W3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param V3 Ptr to three index integrals \f$(\rho\sigma|r_{12}^{-1}|A)\f$
///           given by an array of size `n_ao * (n_ao + 1) /2 * n_df`
/// @param V2 Ptr to two index integrals \f$(A | r_{12}^{-1} | B)\f$ given by
///           an array of size `n_df * n_df`
/// @param V3_ri  Ptr to three index RI integrals \f$(\mu\rho|r_{12}^{-1}|A)\f$
///               given by an array of size `n_ao * n_ri * n_df`
/// @param WV3  Ptr to three index integrals
///             \f$(\rho\sigma|w_{12}r_{12}^{-1}|A)\f$ given by an array of size
///             `n_ao * (n_ao + 1) /2 * n_df`
/// @param WV2  Ptr to two index integrals \f$(A|w_{12}r_{12}^{-1}|B)\f$ given
///             by an array of size `n_df * n_df`
/// @param S2 Ptr to overlap matrix \f$(\mu | \nu)\f$ for the combined ao and
///           auxilliary RI space array of size `(n_ao + n_ri) * (n_ao + n_ri)`
/// @param C  Ptr to orbital coefficients of size `n_ao * (2 * n_orb)`
/// @param occ Ptr to array of occupation number of size `n_occ_alpha +
///            n_occ_beta`
/// @param n_ao Number of atomic orbital basis functions
/// @param n_df Number of density-fitting basis functions
/// @param n_ri Number of auxilliary RI basis functions
/// @param n_orb Number of molecular orbitals provided in each spin channel
/// @param n_occ_alpha Number of occupied orbitals in spin channel alpha
/// @param n_occ_beta Number of occupied orbitals in spin channel beta
/// @param n_active_alpha Number of active orbitals in spin channel alpha
/// @param n_active_beta Number of active orbitals in spin channel beta
/// @param scale_same_spin  Scale factor for same spin UW12 (if zero, an osUW12
///                         calculation is performed)
/// @param print_level Adjust details printed (0-3) default: 0 (silent)
///
/// @return UW12 energy
double uw12_fock(
    double* fock,
    const double* W3,
    const double* W2,
    const double* W3_ri,
    const double* V3,
    const double* V2,
    const double* V3_ri,
    const double* WV3,
    const double* WV2,
    const double* S2,
    const double* C,
    const double* occ,
    size_t n_ao,
    size_t n_df,
    size_t n_ri,
    size_t n_orb,
    size_t n_occ_alpha,
    size_t n_occ_beta,
    size_t n_active_alpha,
    size_t n_active_beta,
    double scale_same_spin,
    size_t print_level
);

// TODO: More interface options

}  // namespace uw12

#endif  // LIBUW12_HPP
