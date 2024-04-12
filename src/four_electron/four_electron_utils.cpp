#include "four_electron_utils.hpp"

#include "integrals/integrals.hpp"
#include "utils/linalg.hpp"
#include "utils/utils.hpp"

namespace uw12::four_el {

double get_energy_spin_factor(
    const size_t n_spin,
    const size_t sigma,
    const size_t sigmaprime,
    const double scale_opp_spin,
    const double scale_same_spin
) {
  if (sigma >= n_spin || sigmaprime >= n_spin) {
    throw std::runtime_error("Invalid spin channel");
  }
  if (n_spin == 1) {
    return 2 * (scale_opp_spin + scale_same_spin);
  }

  if (n_spin == 2) {
    const auto same_spin = sigma == sigmaprime;

    return same_spin ? scale_same_spin : scale_opp_spin;
  }

  throw std::runtime_error("Invalid number of spin channels");
}

// calculate the matrix t_AB = (A|w|ik)(ki|r^{-1}|B)
utils::MatVec calculate_tab(
    const integrals::Integrals& W, const integrals::Integrals& V
) {
  const auto n_spin = W.spin_channels();
  assert(V.spin_channels() == n_spin);

  const auto& W3idx_two_trans = W.get_X3idx_two_trans();
  const auto& V3idx_two_trans = V.get_X3idx_two_trans();

  std::vector<linalg::Mat> tab(n_spin);
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    tab[sigma] =
        linalg::transpose(W3idx_two_trans[sigma]) * V3idx_two_trans[sigma];
  }

  return tab;
}

utils::MatVec calculate_ttilde(
    const integrals::Integrals& W,
    const integrals::Integrals& V,
    const utils::MatVec& tab
) {
  using linalg::diagmat;

  const auto n_spin = tab.size();

  const auto& W_vals = W.get_df_vals();
  const auto& V_vals = V.get_df_vals();

  std::vector<linalg::Mat> ttilde(n_spin);
  for (int sigma = 0; sigma < n_spin; ++sigma) {
    ttilde[sigma] = diagmat(W_vals) * tab[sigma] * diagmat(V_vals);
  }

  return ttilde;
}

// calculate the matrix \tilde{t}_AB = (\tilde{A}|w|ik)(ki|r^{-1}|\tilde{B})
// for (\tilde{A}|w|ik) = (A|w|B)^{-1} (B|w|ik)
utils::MatVec calculate_ttilde(
    const integrals::Integrals& W, const integrals::Integrals& V
) {
  const auto tab = calculate_tab(W, V);

  return calculate_ttilde(W, V, tab);
}

}  // namespace uw12::four_el
