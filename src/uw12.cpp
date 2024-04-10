//
// Created by Zack Williams on 09/04/2024.
//

#include "uw12.hpp"

#include "four_electron/four_electron.hpp"
#include "three_electron/three_electron.hpp"
#include "two_electron/two_electron.hpp"
#include "utils/print.hpp"

namespace uw12 {

using Vec = linalg::Vec;
using Matrix = linalg::Mat;
using integrals::BaseIntegrals;
using three_el::ri::ABSProjectors;
using utils::FockMatrixAndEnergy;
using utils::Orbitals;
using utils::spin_channels;

FockMatrixAndEnergy form_fock(
    const BaseIntegrals &W,
    const BaseIntegrals &V,
    const BaseIntegrals &WV,
    const ABSProjectors &abs_projectors,
    const Orbitals &orbitals,
    const utils::Occupations &occupations,
    const std::vector<size_t> &n_active,
    const bool indirect_term,
    const bool calculate_fock,
    const double scale_opp_spin,
    const double scale_same_spin,
    const size_t print_level
) {
  if (print_level > 2) {
    print::print_header("UW12 Calculation");
  }

  const size_t n_spin = spin_channels(orbitals);
  if (spin_channels(occupations) != n_spin) {
    throw std::runtime_error(
        "Different number of spin channels for orbitals and occupations"
    );
  }
  if (spin_channels(n_active) != n_spin) {
    throw std::runtime_error(
        "Different number of spin channels in n_active and orbitals"
    );
  }

  const auto Co = utils::occupation_weighted_orbitals(orbitals, occupations);
  const auto active_Co = utils::freeze_core(Co, n_active);

  // ----------------------- Two electron term -------------------------------

  // Calculate two electron contribution using density fitting
  const auto fock_two_el = two_el::form_fock_two_el_df(
      WV,
      active_Co,
      indirect_term,
      calculate_fock,
      scale_opp_spin,
      scale_same_spin
  );

  if (print_level > 1) {
    print::print_result("Two electron energy", fock_two_el.energy);
  }

  // ---------------------- Three electron term ------------------------------

  const auto W_ints = integrals::Integrals(W, Co, active_Co);
  const auto V_ints = integrals::Integrals(V, Co, active_Co);

  const auto fock_three_el = three_el::form_fock_three_el_term_df_ri(
      W_ints,
      V_ints,
      abs_projectors,
      indirect_term,
      calculate_fock,
      scale_opp_spin,
      scale_same_spin
  );

  if (print_level > 1) {
    print::print_result("Three electron energy", fock_three_el.energy);
  }

  // ----------------------- Four electron term ------------------------------

  // Calculate four electron contribution using density fitting
  const auto fock_four_el = four_el::form_fock_four_el_df(
      W_ints,
      V_ints,
      indirect_term,
      calculate_fock,
      scale_opp_spin,
      scale_same_spin
  );

  if (print_level > 1) {
    print::print_result("Four electron energy", fock_four_el.energy);
  }

  // -------------------------------------------------------------------------

  // Compute totals
  auto fock = fock_two_el + fock_three_el + fock_four_el;

  if (print_level > 2) {
    print::print_character_line('=', 48);
  }
  if (print_level > 0) {
    print::print_result("Total UW12 energy", fock.energy);
  }

  return fock;
}

}  // namespace uw12
