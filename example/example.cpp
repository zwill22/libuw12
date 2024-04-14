#include <libuw12.hpp>

#include <iostream>
#include <iomanip>

// Get filepath to UW12 root directory, script requires test_data to be
// subdirectory of UW12_ROOT
#define STRING(x) #x
#define XSTRING(x) STRING(x)
constexpr char uw12_root[] = XSTRING(UW12_ROOT);

using uw12::integrals::BaseIntegrals;
using uw12::three_el::ri::ABSProjectors;
using uw12::utils::Occupations;
using uw12::utils::Orbitals;

constexpr auto epsilon = 1e-6;

template <typename Object>
auto get_mem_loc(const Object& object) {
  return uw12::linalg::mem_ptr(object);
}

BaseIntegrals setup_base_integrals(
  const std::string& filepath,
  const std::string& prefix,
  const bool ri = true
) {
  const auto X3_src = uw12::linalg::load_csv(filepath + prefix + "3.csv");
  const auto X2_src = uw12::linalg::load_csv(filepath + prefix + "2.csv");

  // Get pointers to memory locations
  const auto X3 = get_mem_loc(X3_src);
  const auto X2 = get_mem_loc(X2_src);

  // Get the sizes of each
  const auto n_row = uw12::linalg::n_rows(X3_src);
  const auto n_ao = static_cast<size_t>(std::sqrt(8 * n_row - 1) / 2);
  assert(n_ao * (n_ao + 1) / 2 == n_row);

  const auto n_df = uw12::linalg::n_cols(X3_src);
  assert(uw12::linalg::n_rows(X2_src) == n_df);
  assert(uw12::linalg::n_cols(X2_src) == n_df);

  if (!ri) {
    return uw12::setup_base_integrals(X3, X2, n_ao, n_df);
  }

  // Setup RI integrals in the same way
  const auto X3_ri_src = uw12::linalg::load_csv(filepath + prefix + "3_ri.csv");
  const auto X3_ri = get_mem_loc(X3_ri_src);
  assert(uw12::linalg::n_cols(X3_ri_src) == n_df);
  const auto n_ri = uw12::linalg::n_rows(X3_ri_src) / n_ao;
  assert(n_ao * n_ri == uw12::linalg::n_rows(X3_ri_src));

  return uw12::setup_base_integrals(X3, X2, X3_ri, n_ao, n_df, n_ri);
}

ABSProjectors setup_abs_projectors(
  const std::string& filepath,
  const size_t n_ao,
  const size_t n_ri
) {
  // Get overlap matrix from external source
  const auto S_src = uw12::linalg::load_csv(filepath + "S.csv");

  // Get pointer to memory locations
  const auto S = get_mem_loc(S_src);

  assert(uw12::linalg::n_rows(S_src) == n_ao + n_ri);
  assert(uw12::linalg::n_cols(S_src) == n_ao + n_ri);

  return uw12::setup_abs_projectors(S, n_ao, n_ri);
}

// Restricted orbitals example
Orbitals setup_orbitals(const std::string& filepath) {
  const auto C_src = uw12::linalg::load_csv(filepath + "orbitals.csv");

  // Get pointer to memory locations
  const auto C = get_mem_loc(C_src);

  const auto n_ao = uw12::linalg::n_rows(C_src);
  const auto n_orb = uw12::linalg::n_cols(C_src);

  return uw12::setup_orbitals(C, n_ao, n_orb, 1);
}

// Restricted orbitals example
Occupations setup_occupations(const std::string& filepath) {
  const auto occ_src = uw12::linalg::load_csv(filepath + "occ.csv");

  // Get pointer to memory locations
  const auto occ = get_mem_loc(occ_src);

  const auto n_occ = uw12::linalg::n_elem(occ_src);

  return uw12::setup_occupations(occ, n_occ);
}

inline char separator() {
#ifdef _WIN32
  return '\\';
#else
  return '/';
#endif
}

int main() {
  std::cout << std::string(64, '=') << '\n';
  std::cout << "Sample UW12 Calculation on water molecule" << '\n';
  std::cout << std::string(64, '=') << '\n';

  // Get all integral data from external source
  const auto sep = separator();
  const std::string file_root = uw12_root;
  const std::string filepath = file_root +  "test_data" + sep + "water" + sep;
  std::cout << " Location of external data: \n  " << filepath << "\n\n";

  // Setup BaseIntegrals with these integrals for each inter-electron potential
  const auto W = setup_base_integrals(filepath, "W");
  const auto V = setup_base_integrals(filepath, "V");
  const auto WV = setup_base_integrals(
    filepath,
    "WV",
    false
  ); // No RI needed for WV integrals

  // get numbers
  const auto n_ao = W.get_number_ao();
  assert(V.get_number_ao() == n_ao);
  assert(WV.get_number_ao() == n_ao);
  std::cout << "Number of atomic orbitals:\t\t" << n_ao << '\n';

  const auto n_df = W.get_number_df();
  assert(V.get_number_df() == n_df);
  assert(WV.get_number_df() == n_df);
  std::cout << "Number of density-fitting orbitals:\t" << n_df << '\n';

  const auto n_ri = W.get_number_ri();
  assert(V.get_number_ri() == n_ri);
  std::cout << "Number of auxiliary RI orbitals:\t" << n_ri << '\n';

  // Setup ABS Projectors by providing overlap matrix S for the combined ao and
  // ri basis
  const auto abs_projectors = setup_abs_projectors(filepath, n_ao, n_ri);

  const auto orbitals = setup_orbitals(filepath);
  const auto occupations = setup_occupations(filepath);

  // Specify desired number of active orbitals for each spin channels
  const std::vector<size_t> n_active = {4};

  // Checks sizes
  const auto n_spin = orbitals.size();
  assert(n_spin == 1 || n_spin == 2);
  assert(occupations.size() == n_spin);
  assert(n_active.size() == n_spin);
  std::cout << "Number of spin channels: \t\t" << n_spin << "\n\n";

  // Cannot have more active orbitals than the number of occupied orbitals in
  // each spin channel
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    assert(uw12::linalg::n_elem(occupations[sigma]) >= n_active[sigma]);
  }

  constexpr double scale_opp_spin = 1.0;
  std::cout << "osUW12 scale factor:\t\t\t" << scale_opp_spin << '\n';

  constexpr double scale_same_spin = 0.0; // osUW12 calculation
  std::cout << "ssUW12 scale factor:\t\t\t" << scale_same_spin << '\n';

  // Calculate osUW12 energy
  const auto energy = uw12::uw12_energy(
    W,
    V,
    WV,
    abs_projectors,
    orbitals,
    occupations,
    n_active,
    scale_opp_spin,
    scale_same_spin,
    3
  );

  std::cout << " Calculation complete" << '\n';
  std::cout << std::string(48, '=') << '\n';
  std::cout << '\n';
  std::cout << std::string(48, '-') << '\n';
  std::cout << "osUW12 energy: " << energy << '\n';
  std::cout << std::string(48, '-') << '\n';
  std::cout << '\n';
  std::cout << '\n';
  std::cout << std::string(64, '=') << '\n';

  // IF Fock matrix is also required
  const auto fock_size = n_ao * n_ao * n_spin;

  // Initialise some memory
  double fock[fock_size];

  const auto energy2 = uw12::uw12_fock(
    fock,
    W,
    V,
    WV,
    abs_projectors,
    orbitals,
    occupations,
    n_active,
    scale_opp_spin,
    scale_same_spin
  );

  assert(std::abs(energy - energy2) < epsilon);

  return 0;
}
