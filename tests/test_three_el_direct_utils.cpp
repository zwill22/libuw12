//
// Created by Zack Williams on 27/03/2024.
//

#include "../src/four_electron/four_electron_utils.hpp"
#include "../src/three_electron/direct_utils.hpp"
#include "catch.hpp"
#include "setup_integrals.hpp"

auto setup_abs_projector(
    const size_t n_ao, const size_t n_ri, const int S_seed = test::seed
) {
  const auto s = uw12::linalg::random_pd(n_ao + n_ri, S_seed);

  return uw12::three_el::ri::calculate_abs_projectors(s, n_ao, n_ri);
}

TEST_CASE("Test three electron term - Direct Utils (X_AB)") {
  std::vector<size_t> n_occ = {3};
  std::vector<size_t> n_active = {2};

  for (int i = 0; i < 2; ++i) {
    constexpr size_t n_ao = 10;
    constexpr size_t n_df = 18;
    constexpr size_t n_ri = 25;

    const auto n_spin = n_occ.size();
    REQUIRE(n_active.size() == n_spin);

    const auto [W, V] = test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, test::seed - 1
    );

    const auto abs_projectors = setup_abs_projector(n_ao, n_ri);

    const auto Xab = uw12::three_el::calculate_xab(W, V, abs_projectors);

    REQUIRE(Xab.size() == n_spin);

    for (const auto& X : Xab) {
      CHECK(uw12::linalg::n_rows(X) == n_df);
      CHECK(uw12::linalg::n_cols(X) == n_df);
    }

    n_occ.push_back(2);
    n_active.push_back(1);
  }
}

TEST_CASE("Test three electron term - Direct Utils (energy)") {
  std::vector<size_t> n_occ = {4};
  std::vector<size_t> n_active = {3};

  for (int i = 0; i < 2; ++i) {
    constexpr size_t n_ao = 11;
    constexpr size_t n_df = 19;
    constexpr size_t n_ri = 27;

    const auto n_spin = n_occ.size();
    REQUIRE(n_active.size() == n_spin);

    const auto [W, V] = test::setup_integrals_pair(
        n_ao, n_df, n_ri, n_occ, n_active, test::seed - 1
    );

    const auto abs_projectors = setup_abs_projector(n_ao, n_ri);

    const auto Xab = uw12::three_el::calculate_xab(W, V, abs_projectors);
    REQUIRE(Xab.size() == n_spin);

    const auto ttilde_ab = uw12::four_el::calculate_ttilde(W, V);
    REQUIRE(ttilde_ab.size() == n_spin);

    const auto e_os =
        uw12::three_el::calculate_direct_energy(Xab, ttilde_ab, 1.0, 0.0);
    const auto e_ss =
        uw12::three_el::calculate_direct_energy(Xab, ttilde_ab, 0.0, 1.0);

    SECTION("OS Multiplicity") {
      const auto e_os2 =
          uw12::three_el::calculate_direct_energy(Xab, ttilde_ab, 1.6, 0.0);
      CHECK_THAT(1.6 * e_os, Catch::Matchers::WithinRel(e_os2, test::eps));
    }

    SECTION("SS Multiplicity") {
      const auto e_ss2 =
          uw12::three_el::calculate_direct_energy(Xab, ttilde_ab, 0.0, 1.4);
      CHECK_THAT(1.4 * e_ss, Catch::Matchers::WithinRel(e_ss2, test::eps));
    }

    const auto e =
        uw12::three_el::calculate_direct_energy(Xab, ttilde_ab, 1.0, 0.5);
    CHECK_THAT(e_os + 0.5 * e_ss, Catch::Matchers::WithinRel(e, test::eps));

    n_occ.push_back(3);
    n_active.push_back(2);
  }
}

TEST_CASE("Test three electron term - Direct Utils (X_AB Dttilde)") {
  constexpr size_t n_ao = 11;
  constexpr size_t n_df = 19;
  constexpr size_t n_ri = 27;
  constexpr size_t n_occ = 4;
  constexpr size_t n_active = 3;

  const auto [W, V] = test::setup_integrals_pair(
      n_ao, n_df, n_ri, {n_occ}, {n_active}, test::seed - 1
  );

  const auto W3_imA = W.get_X3idx_one_trans()[0];
  const auto V3_imA = V.get_X3idx_one_trans()[0];

  const auto abs_projectors = setup_abs_projector(n_ao, n_ri);
  const auto Xab = uw12::three_el::calculate_xab(W, V, abs_projectors)[0];

  const auto& W_vals = W.get_df_vals();
  const auto& V_vals = V.get_df_vals();

  const auto x_dttilde = uw12::three_el::calculate_xab_dttilde(
      W3_imA, V3_imA, Xab, W_vals, V_vals, n_active, n_ao
  );

  CHECK_THROWS(uw12::three_el::calculate_xab_dttilde(
      W3_imA, V3_imA, Xab, W_vals, V_vals, n_active, n_ao - 1
  ));
  CHECK_THROWS(uw12::three_el::calculate_xab_dttilde(
      uw12::linalg::head_rows(W3_imA, n_occ * (n_ao - 1)),
      V3_imA,
      Xab,
      W_vals,
      V_vals,
      n_active,
      n_ao
  ));
  CHECK_THROWS(uw12::three_el::calculate_xab_dttilde(
      W3_imA,
      uw12::linalg::head_rows(V3_imA, n_occ * (n_ao - 1)), Xab,
      W_vals,
      V_vals,
      n_active,
      n_ao
  ));
}
