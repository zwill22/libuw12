//
// Created by Zack Williams on 06/03/2024.
//

#ifndef NUMERICAL_FOCK_HPP
#define NUMERICAL_FOCK_HPP

#include "../src/utils/utils.hpp"

namespace fock {

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

template <typename EnergyFn>
auto numerical_fock_matrix(
    const EnergyFn& energy_function,
    const uw12::utils::DensityMatrix& D_mat,
    const double delta
) {
  using uw12::linalg::empty;
  using uw12::linalg::join_matrices;
  using uw12::linalg::n_elem;
  using uw12::linalg::Vec;
  using uw12::utils::lower;

  const auto n_spin = D_mat.size();
  if (n_spin != 1 && n_spin != 2) {
    throw std::runtime_error("Invalid number of spin channels");
  }

  const auto energy_func = [&energy_function,
                            n_spin](const Vec& d_vec) -> double {
    const auto n_total = uw12::linalg::n_elem(d_vec);
    const auto n_sub = n_total / n_spin;

    uw12::utils::DensityMatrix D_new;
    for (auto sigma = 0; sigma < n_spin; ++sigma) {
      const auto start = sigma * n_sub;
      const auto d_vec_this = uw12::linalg::sub_vec(d_vec, start, n_sub);
      D_new.push_back(uw12::utils::square(d_vec_this));
    }

    return energy_function(D_new);
  };

  const auto initial_d_vec = [&D_mat, n_spin] {
    Vec d_vec;
    for (auto sigma = 0; sigma < n_spin; ++sigma) {
      d_vec = empty(d_vec) ? lower(D_mat[sigma])
                           : join_matrices(d_vec, lower(D_mat[sigma]));
    }
    return d_vec;
  }();

  const auto F_vec = numerical_gradient(initial_d_vec, energy_func, delta);

  const auto n_total = n_elem(F_vec);
  const auto n_sub = n_total / n_spin;

  uw12::utils::FockMatrix fock;
  for (auto sigma = 0; sigma < n_spin; ++sigma) {
    constexpr double off_diag_factor = 0.5;
    const auto start = sigma * n_sub;
    const auto F_vec_this = uw12::linalg::sub_vec(F_vec, start, n_sub);
    fock.push_back(uw12::utils::square(F_vec_this, off_diag_factor));
  }

  return fock;
}
}  // namespace fock
#endif  // NUMERICAL_FOCK_HPP
