//
// Created by Zack Williams on 10/04/2024.
//

#ifndef TEST_DATA_HPP
#define TEST_DATA_HPP

#include "../src/config.hpp"
#include "../src/integrals/base_integrals.hpp"
#include "../src/utils/utils.hpp"

namespace uw12_test {

inline char separator() {
#ifdef _WIN32
  return '\\';
#else
  return '/';
#endif
}

template <typename Output>
Output load(
    const std::string& identifier,
    const std::string& filepath,
    const size_t n_spin
) {
  using uw12::linalg::load_csv;

  const auto sep = separator();

  if (n_spin == 1) {
    return {load_csv(filepath + sep + identifier + ".csv")};
  }

  if (n_spin == 2) {
    return {
        load_csv(filepath + sep + identifier + "_alpha.csv"),
        load_csv(filepath + sep + identifier + "_beta.csv")
    };
  }

  throw std::runtime_error("invalid number of spin channels");
}

inline uw12::utils::Orbitals load_orbitals(
    const std::string& filepath, const size_t n_spin
) {
  return load<uw12::utils::Orbitals>("orbitals", filepath, n_spin);
}

inline uw12::utils::Occupations load_occ(
    const std::string& filepath, const size_t n_spin
) {
  return load<uw12::utils::Occupations>("occ", filepath, n_spin);
}

inline uw12::utils::FockMatrix load_fock(
    const std::string& filepath, const size_t n_spin
) {
  return load<uw12::utils::FockMatrix>("fock", filepath, n_spin);
}

inline std::vector<size_t> load_n_active(
    const std::string& filepath, const std::string& suffix
) {
  const auto result =
      load<std::vector<uw12::linalg::Mat>>("n_active_" + suffix, filepath, 1);
  assert(result.size() == 1);

  const auto mat = result[0];

  std::vector<size_t> output = {};
  for (const auto elem: mat) {
    output.push_back(elem);
  }
  assert(output.size() > 0);
  assert(output.size() <= 2);

  return output;
}

inline double load_energy(
    const std::string& filepath, const std::string& suffix
) {
  const auto energy_mat =
      load<std::vector<uw12::linalg::Vec>>("energy_" + suffix, filepath, 1);
  assert(energy_mat.size() == 1);
  const auto col = energy_mat[0];
  assert(uw12::linalg::n_elem(col) == 1);

  return uw12::linalg::elem(col, 0);
}

struct TestData {
  uw12::integrals::BaseIntegrals W;
  uw12::integrals::BaseIntegrals V;
  uw12::integrals::BaseIntegrals WV;
  uw12::linalg::Mat S;
  uw12::utils::Orbitals orbitals;
  uw12::utils::Occupations occ;
  std::vector<size_t> n_active;
  uw12::utils::FockMatrixAndEnergy fock;

  TestData(
      const std::string& molecule,
      const std::string& suffix,
      const size_t n_spin
  ) {
    using uw12::linalg::load_csv;

    const auto sep = separator();

    const std::string root_path = source_dir;
    const auto filepath = root_path + sep + "test_data" + sep + molecule;

    const auto W2 = load_csv(filepath + sep + "W2.csv");
    const auto W3 = load_csv(filepath + sep + "W3.csv");
    const auto W3_ri = load_csv(filepath + sep + "W3_ri.csv");

    W = uw12::integrals::BaseIntegrals(W3, W2, W3_ri);

    const auto V2 = load_csv(filepath + sep + "V2.csv");
    const auto V3 = load_csv(filepath + sep + "V3.csv");
    const auto V3_ri = load_csv(filepath + sep + "V3_ri.csv");

    V = uw12::integrals::BaseIntegrals(V3, V2, V3_ri);

    const auto WV2 = load_csv(filepath + sep + "WV2.csv");
    const auto WV3 = load_csv(filepath + sep + "WV3.csv");

    WV = uw12::integrals::BaseIntegrals(WV3, WV2);
    S = load_csv(filepath + sep + "S.csv");

    orbitals = load_orbitals(filepath, n_spin);
    occ = load_occ(filepath, n_spin);
    n_active = load_n_active(filepath, suffix);

    fock = uw12::utils::FockMatrixAndEnergy();
    fock.energy = load_energy(filepath, suffix);
    fock.fock = load_fock(filepath, n_spin);
  }
};

}  // namespace uw12_test

#endif  // TEST_DATA_HPP
