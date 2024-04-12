//
// Created by Zack Williams on 06/03/2024.
//

#ifndef NUMERICAL_FOCK_HPP
#define NUMERICAL_FOCK_HPP

#include "utils/parallel.hpp"
#include "utils/utils.hpp"

namespace uw12_test::fock {

/// A function which calculates the numerical gradient with respect to vector
/// vec for an energy function fn
///
/// @tparam EnergyFn A function of type f(vec) -> double
/// @param vec
/// @param energy_fn
/// @param delta
///
/// @return gradient in vector form
template <typename EnergyFn>
auto numerical_gradient(
    const uw12::linalg::Vec& vec, const EnergyFn& energy_fn, const double delta
) {
  using uw12::linalg::n_elem;
  using uw12::linalg::set_elem;

  const auto n_var = n_elem(vec);

  auto result = uw12::linalg::vec(n_var);

  const auto func = [&result, &vec, &energy_fn, delta](const size_t idx) {
    auto perturbed_vec = vec;
    const auto val = uw12::linalg::elem(perturbed_vec, idx);

    set_elem(perturbed_vec, idx, val + delta);

    const auto result_plus = energy_fn(perturbed_vec);

    set_elem(perturbed_vec, idx, val - delta);
    const auto result_minus = energy_fn(perturbed_vec);

    const auto factor = 1 / (2 * delta);
    const auto value = factor * (result_plus - result_minus);

    set_elem(result, idx, value);
  };

  uw12::parallel::parallel_for(0, n_var, func);

  return result;
}

/// Calculate the Fock matrix numerically using the energy
///
/// @tparam EnergyFn Function f(vec) -> double
/// @param energy_function E(D) -> energy
/// @param D_mat Density matrix
/// @param delta variation parameter
///
/// @return Fock matrix
template <typename EnergyFn>
auto numerical_fock_matrix(
    const EnergyFn& energy_function,
    const uw12::utils::DensityMatrix& D_mat,
    const double delta
) {
  using uw12::linalg::Vec;
  using uw12::utils::square;

  const auto n_spin = D_mat.size();
  if (n_spin != 1 && n_spin != 2) {
    throw std::runtime_error("Invalid number of spin channels");
  }

  uw12::utils::FockMatrix fock;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto energy_func = [&energy_function, &D_mat, sigma](const Vec& d_vec
                             ) -> double {
      auto D_new = D_mat;
      D_new[sigma] = square(d_vec);

      return energy_function(D_new);
    };

    const auto initial_d_vec = uw12::utils::lower(D_mat[sigma]);

    const auto F_vec = numerical_gradient(initial_d_vec, energy_func, delta);

    fock.push_back(square(F_vec, 0.5));
  }

  return fock;
}

inline void check_fock(
    const uw12::utils::FockMatrix& analytic_fock,
    const uw12::utils::FockMatrix& num_fock,
    const double rel_eps
) {
  const auto n_spin = analytic_fock.size();
  if (num_fock.size() != n_spin) {
    throw std::runtime_error(
        "Analytic and numeric Fock matrices have different numbers of spin "
        "channels"
    );
  }

  double max_rel_diff = 0;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto fock1 = analytic_fock[sigma];
    const auto fock2 = num_fock[sigma];
    const auto n_ao = uw12::linalg::n_rows(fock1);
    assert(uw12::linalg::n_cols(fock1) == n_ao);
    if (uw12::linalg::n_rows(fock2) != n_ao) {
      throw std::runtime_error(
          "Analytic and numeric Fock matrices are of different sizes"
      );
    }
    assert(uw12::linalg::n_cols(fock2) == n_ao);
    for (size_t col_idx = 0; col_idx < n_ao; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_ao; ++row_idx) {
        const auto target = uw12::linalg::elem(fock1, row_idx, col_idx);
        const auto elem = uw12::linalg::elem(fock2, row_idx, col_idx);

        if (const auto rel_diff = std::abs((elem - target) / target);
            rel_diff > max_rel_diff) {
          max_rel_diff = rel_diff;
        }
      }
    }
  }

  std::cout << "Maximum relative Diff: " << max_rel_diff << '\n';
  std::cout << "Threshold: " << rel_eps << '\n';

  if (max_rel_diff > rel_eps) {
    throw std::runtime_error("Relative difference outside threshold");
  }
}

}  // namespace uw12_test::fock
#endif  // NUMERICAL_FOCK_HPP
