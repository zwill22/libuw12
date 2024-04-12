//
// Created by Zack Williams on 25/03/2024.
//

#include "indirect_utils.hpp"

#include "integrals/integrals.hpp"
#include "ri_utils.hpp"
#include "utils/linalg.hpp"

namespace uw12::three_el {

using namespace linalg;
using integrals::Integrals;
using ri::ABSProjectors;

// Calculate X for a single ij pair:
// X_AB^ij = Σ_μν (A|w|iμ) [S^-1]_μν (νj|r^-1|B)
Mat calculate_xij(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const size_t n_core,
    const size_t i,
    const size_t j,
    const size_t sigma
) {
  const auto n_df_W = n_elem(W.get_df_vals());
  const auto n_df_V = n_elem(V.get_df_vals());
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);

  const auto& W3idx_one_trans = W.get_X3idx_one_trans()[sigma];
  const auto& V3idx_one_trans = V.get_X3idx_one_trans()[sigma];

  const auto& W3idx_one_trans_ri = W.get_X3idx_one_trans_ri()[sigma];
  const auto& V3idx_one_trans_ri = V.get_X3idx_one_trans_ri()[sigma];

  Mat out = transpose(sub_mat(W3idx_one_trans_ri, i * n_ri, 0, n_ri, n_df_W)) *
            abs_projectors.s_inv_ri_ri *
            sub_mat(V3idx_one_trans_ri, j * n_ri, 0, n_ri, n_df_V);

  assert(n_rows(out) == n_df_W);
  assert(n_cols(out) == n_df_V);

  out += transpose(sub_mat(W3idx_one_trans_ri, i * n_ri, 0, n_ri, n_df_V)) *
         abs_projectors.s_inv_ri_ao *
         sub_mat(V3idx_one_trans, (j + n_core) * n_ao, 0, n_ao, n_df_V);

  out +=
      transpose(sub_mat(W3idx_one_trans, (i + n_core) * n_ao, 0, n_ao, n_df_W)
      ) *
      abs_projectors.s_inv_ao_ri *
      sub_mat(V3idx_one_trans_ri, j * n_ri, 0, n_ri, n_df_V);

  out +=
      transpose(sub_mat(W3idx_one_trans, (i + n_core) * n_ao, 0, n_ao, n_df_W)
      ) *
      abs_projectors.s_inv_ao_ao *
      sub_mat(V3idx_one_trans, (j + n_core) * n_ao, 0, n_ao, n_df_V);

  return out;
}

// Calculate ttilde for a given ij pair:
// ttilde_AB^ij = Σ_k (jk|w|Atilde) (Btilde|r^-1|ki)
Mat calculate_ttilde_ij(
    const Integrals& W,
    const Integrals& V,
    const size_t n_active,
    const size_t i,
    const size_t j,
    const size_t sigma
) {
  const auto n_occ = W.number_occ_orbitals(sigma);
  assert(n_occ == V.number_occ_orbitals(sigma));
  const auto n_df_W = n_elem(W.get_df_vals());
  const auto n_df_V = n_elem(V.get_df_vals());

  const auto& W3idx_two_trans_sigma = W.get_X3idx_two_trans()[sigma];
  const auto& V3idx_two_trans_sigma = V.get_X3idx_two_trans()[sigma];

  const auto W_tmp = reshape(W3idx_two_trans_sigma, n_active, n_occ * n_df_W);

  const Mat W3idx_trans = reshape(row(W_tmp, j), n_occ, n_df_W, true) *
                          linalg::diagmat(W.get_df_vals());

  const auto V_tmp = reshape(V3idx_two_trans_sigma, n_active, n_occ * n_df_V);

  const Mat V3idx_trans = reshape(row(V_tmp, i), n_occ, n_df_V, true) *
                          linalg::diagmat(V.get_df_vals());

  return transpose(W3idx_trans) * V3idx_trans;
}

// Indirect energy calculated as:
// E_c^- = Σ_ij Y_ij = Σ_ij ( Σ_ΑΒ ttilde_AB^ij X_AB^ij )
// where ttilde_AB^ij = Σ_k (jk|w|Atilde) (Btilde|r^-1|ki)
// X_AB^ij = Σ_μν (A|w|iμ) s_μ (μj|r^-1|B)
// with parallelisation over indices i,j.
double indirect_3el_energy(
    const Integrals& W, const Integrals& V, const ABSProjectors& abs_projectors
) {
  const auto n_spin = W.spin_channels();
  assert(V.spin_channels() == n_spin);

  double energy = 0;
  for (size_t sigma = 0; sigma < n_spin; ++sigma) {
    const auto n_occ = W.number_occ_orbitals(sigma);
    assert(n_occ == V.number_occ_orbitals(sigma));
    const auto n_active = W.number_active_orbitals(sigma);
    assert(n_active == V.number_active_orbitals(sigma));
    assert(n_occ >= n_active);
    const auto n_core = n_occ - n_active;

    const auto indirect_energy =
        [&W, &V, &abs_projectors, n_active, n_core, sigma, n_spin](
            const size_t i, const size_t j
        ) -> double {
      const auto xij = calculate_xij(W, V, abs_projectors, n_core, i, j, sigma);

      const auto ttilde_ij = calculate_ttilde_ij(W, V, n_active, i, j, sigma);

      return (n_spin == 1 ? 2 : 1) * dot(xij, ttilde_ij);
    };

    energy += parallel::parallel_sum_2d<double>(
        0, n_active, 0, n_active, 0, indirect_energy
    );
  }

  return energy;
}

// Calculates the fock contribution for a single pair jk
// fock_i = (jk|x|αμ)S_μν^-1(νj|y|kβ)
// this is split into four separate contributions for the four blocks of S_μν
template <typename X3Fn>
Mat fock_i(
    const Integrals& X,
    const Integrals& Y,
    const ABSProjectors& abs_projectors,
    const X3Fn& X3_fn,
    const X3Fn& X3_ri_fn,
    const size_t j,
    const size_t k,
    const size_t sigma
) {
  const auto n_df_Y = n_elem(Y.get_df_vals());
  const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  const auto n_occ = X.number_occ_orbitals(sigma);
  const auto n_active = X.number_active_orbitals(sigma);
  assert(n_occ == Y.number_occ_orbitals(sigma));
  assert(n_active == Y.number_active_orbitals(sigma));
  assert(n_occ >= n_active);
  const auto n_core = n_occ - n_active;

  const Vec row =
      transpose(linalg::row(X.get_X3idx_two_trans()[sigma], k * n_active + j));

  const Vec X3tilde_two_trans_kj = linalg::schur(row, X.get_df_vals());

  const Vec tmp1 = X3_ri_fn(X, X3tilde_two_trans_kj);
  const Mat X_jk_am = linalg::reshape(tmp1, n_ri, n_ao);

  const Mat Y3tilde_one_trans_k =
      linalg::diagmat(Y.get_df_vals()) *
      transpose(
          sub_mat(Y.get_X3idx_one_trans()[sigma], k * n_ao, 0, n_ao, n_df_Y)
      );

  const Mat Y_vj_kb =
      sub_mat(Y.get_X3idx_one_trans_ri()[sigma], n_ri * j, 0, n_ri, n_df_Y) *
      Y3tilde_one_trans_k;

  Mat out = transpose(X_jk_am) * abs_projectors.s_inv_ri_ri * Y_vj_kb;

  const Mat Y_pj_kb =
      sub_mat(
          Y.get_X3idx_one_trans()[sigma], (j + n_core) * n_ao, 0, n_ao, n_df_Y
      ) *
      Y3tilde_one_trans_k;

  out += transpose(X_jk_am) * abs_projectors.s_inv_ri_ao * Y_pj_kb;

  const Vec X_jk_ap = X3_fn(X, X3tilde_two_trans_kj);

  out += utils::square(X_jk_ap) * abs_projectors.s_inv_ao_ri * Y_vj_kb;

  out += utils::square(X_jk_ap) * abs_projectors.s_inv_ao_ao * Y_pj_kb;

  return out;
}

// Calculates the fock contribution for a single pair ik
// fock_k = (jβ|w|iμ)S_μν^-1(νj|r^-1|αi)
// this is split into four separate contributions for the four blocks of S_μν
Mat fock_k(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const size_t i,
    const size_t j,
    const size_t sigma
) {
  const auto& W_vals = W.get_df_vals();
  const auto& V_vals = V.get_df_vals();

  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  const auto n_ri = n_rows(abs_projectors.s_inv_ri_ri);
  const auto n_df_W = n_elem(W_vals);
  const auto n_df_V = n_elem(V_vals);

  const auto& W3idx_one_trans = W.get_X3idx_one_trans()[sigma];
  const auto& V3idx_one_trans = V.get_X3idx_one_trans()[sigma];

  const auto& W3idx_one_trans_ri = W.get_X3idx_one_trans_ri()[sigma];
  const auto& V3idx_one_trans_ri = V.get_X3idx_one_trans_ri()[sigma];

  const auto n_occ = W.number_occ_orbitals(sigma);
  assert(n_occ == V.number_occ_orbitals(sigma));
  const auto n_active = W.number_active_orbitals(sigma);
  assert(n_active == V.number_active_orbitals(sigma));
  assert(n_active <= n_occ);
  const auto n_core = n_occ - n_active;

  const Mat W3tilde_one_trans_j =
      linalg::diagmat(W_vals) *
      transpose(sub_mat(W3idx_one_trans, n_ao * (j + n_core), 0, n_ao, n_df_W));

  const Mat W_jb_im = sub_mat(W3idx_one_trans_ri, n_ri * i, 0, n_ri, n_df_W) *
                      W3tilde_one_trans_j;

  const Mat V3tilde_one_trans_i =
      linalg::diagmat(V_vals) *
      transpose(sub_mat(V3idx_one_trans, n_ao * (i + n_core), 0, n_ao, n_df_V));

  const Mat V_vj_ai = sub_mat(V3idx_one_trans_ri, n_ri * j, 0, n_ri, n_df_V) *
                      V3tilde_one_trans_i;

  Mat out = transpose(W_jb_im) * abs_projectors.s_inv_ri_ri * V_vj_ai;

  const Mat V_pj_ai =
      sub_mat(V3idx_one_trans, n_ao * (j + n_core), 0, n_ao, n_df_V) *
      V3tilde_one_trans_i;

  out += transpose(W_jb_im) * abs_projectors.s_inv_ri_ao * V_pj_ai;

  const Mat W_jb_ip =
      sub_mat(W3idx_one_trans, (n_core + i) * n_ao, 0, n_ao, n_df_W) *
      W3tilde_one_trans_j;

  out += transpose(W_jb_ip) * abs_projectors.s_inv_ao_ri * V_vj_ai;

  out += transpose(W_jb_ip) * abs_projectors.s_inv_ao_ao * V_pj_ai;

  return out;
}

std::function<Vec(const Integrals&, const Vec&)> get_X3_func(
    const Integrals& X_int
) {
  if (const auto& X = X_int.get_base_integrals(); X.has_J3_0()) {
    return [](const Integrals& Y_int, const Vec& vec) -> Vec {
      const auto& Y = Y_int.get_base_integrals();

      return Y.get_J3_0() * Y.get_P2() * vec;
    };
  }

  return [](const Integrals& Y_int, const Vec& vec) -> Vec {
    const auto& Y = Y_int.get_base_integrals();

    return Y.get_J3() * vec;
  };
}

std::function<Vec(const Integrals&, const Vec&)> get_X3_ri_fn(
    const Integrals& X_int
) {
  if (const auto& X = X_int.get_base_integrals(); X.has_J3_ri_0()) {
    return [](const Integrals& Y_int, const Vec& vec) -> Vec {
      const auto& Y = Y_int.get_base_integrals();

      return Y.get_J3_ri_0() * Y.get_P2() * vec;
    };
  }

  return [](const Integrals& Y_int, const Vec& vec) -> Vec {
    const auto& Y = Y_int.get_base_integrals();

    return Y.get_J3_ri() * vec;
  };
}

Mat indirect_3el_fock_matrix(
    const Integrals& W,
    const Integrals& V,
    const ABSProjectors& abs_projectors,
    const size_t sigma
) {
  const auto n_ao = n_rows(abs_projectors.s_inv_ao_ao);
  const auto n_occ = W.number_occ_orbitals(sigma);
  const auto n_active = W.number_active_orbitals(sigma);

  const auto W3_fn = get_X3_func(W);
  const auto W3_ri_fn = get_X3_ri_fn(W);

  // i term
  const auto i_fock = [&W, &V, &abs_projectors, &W3_fn, &W3_ri_fn, sigma](
                          const size_t j, const size_t k
                      ) -> Mat {
    return fock_i(W, V, abs_projectors, W3_fn, W3_ri_fn, j, k, sigma);
  };

  auto fock_sigma = parallel::parallel_sum_2d<Mat>(
      0, n_active, 0, n_occ, zeros(n_ao, n_ao), i_fock
  );

  const auto V3_fn = get_X3_func(V);
  const auto V3_ri_fn = get_X3_ri_fn(V);

  // j term - this uses the same function as i swapping V and W terms
  const auto j_fock = [&V, &W, &abs_projectors, &V3_fn, &V3_ri_fn, sigma](
                          const size_t i, const size_t k
                      ) -> Mat {
    return fock_i(V, W, abs_projectors, V3_fn, V3_ri_fn, i, k, sigma);
  };

  fock_sigma += parallel::parallel_sum_2d<Mat>(
      0, n_active, 0, n_occ, zeros(n_ao, n_ao), j_fock
  );

  // k term
  const auto k_fock =
      [&W, &V, &abs_projectors, sigma](const size_t i, const size_t j) -> Mat {
    return fock_k(W, V, abs_projectors, i, j, sigma);
  };

  fock_sigma += parallel::parallel_sum_2d<Mat>(
      0, n_active, 0, n_active, zeros(n_ao, n_ao), k_fock);

  return fock_sigma;
}

}
