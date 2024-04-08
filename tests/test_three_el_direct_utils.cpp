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
  REQUIRE(uw12::linalg::n_rows(x_dttilde) == n_ao);
  REQUIRE(uw12::linalg::n_cols(x_dttilde) == n_ao);

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
      uw12::linalg::head_rows(V3_imA, n_occ * (n_ao - 1)),
      Xab,
      W_vals,
      V_vals,
      n_active,
      n_ao
  ));
}

auto setup_base_integrals_direct(
    const uw12::integrals::BaseIntegrals& W, const uw12::linalg::Mat& W2
) {
  const auto& W3 = W.get_J3_0();
  const auto& W3_ri = W.get_J3_ri_0();

  const auto n_ao = W.get_number_ao();
  const auto n_df = W.get_number_df();
  const auto n_ri = W.get_number_ri();

  const auto W2_func = [W2] { return W2; };

  const auto W3_func = [W3](const size_t A) { return W3; };

  const auto W3_ri_func = [W3_ri](const size_t A) { return W3_ri; };

  const auto df_sizes = std::vector({n_df});

  return uw12::integrals::BaseIntegrals(
      W2_func, W3_func, W3_ri_func, df_sizes, n_ao, n_df, n_ri, false, false
  );
}

TEST_CASE("Test three electron term - Direct Utils (ttilde dX_AB)") {
  constexpr size_t n_ao = 10;
  constexpr size_t n_df = 18;
  constexpr size_t n_ri = 25;
  std::vector<size_t> n_occ = {3};
  std::vector<size_t> n_active = {2};

  constexpr auto W_seed = test::seed + 1;
  constexpr auto V_seed = test::seed;

  const auto W2 = uw12::linalg::random_pd(n_df, W_seed);
  const auto W3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, W_seed);
  const auto W3_ri = uw12::linalg::random(n_ao * n_ri, n_df, W_seed);

  const auto W_base = uw12::integrals::BaseIntegrals(W3, W2, W3_ri);
  const auto W_base_direct = setup_base_integrals_direct(W_base, W2);

  CHECK(uw12::linalg::nearly_equal(W2, W_base_direct.two_index(), test::epsilon)
  );
  CHECK(uw12::linalg::nearly_equal(
      W_base.get_J3(), W_base_direct.get_J3(), test::epsilon
  ));
  CHECK(uw12::linalg::nearly_equal(
      W_base.get_J3_ri(), W_base_direct.get_J3_ri(), test::epsilon
  ));

  const auto V2 = uw12::linalg::random_pd(n_df, V_seed);
  const auto V3 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, V_seed);
  const auto V3_ri = uw12::linalg::random(n_ao * n_ri, n_df, V_seed);

  const auto V_base = uw12::integrals::BaseIntegrals(V3, V2, V3_ri);
  const auto V_base_direct = setup_base_integrals_direct(V_base, V2);

  CHECK(uw12::linalg::nearly_equal(V2, V_base_direct.two_index(), test::epsilon)
  );
  CHECK(uw12::linalg::nearly_equal(
      V_base.get_J3(), V_base_direct.get_J3(), test::epsilon
  ));
  CHECK(uw12::linalg::nearly_equal(
      V_base.get_J3_ri(), V_base_direct.get_J3_ri(), test::epsilon
  ));

  const auto abs_projectors = setup_abs_projector(n_ao, n_ri);

  for (size_t i = 0; i < 2; ++i) {
    const auto n_spin = n_occ.size();
    REQUIRE(n_active.size() == n_spin);

    const auto [Co, active_Co] = test::setup_orbitals(n_occ, n_active, n_ao);
    REQUIRE(Co.size() == n_spin);
    REQUIRE(active_Co.size() == n_spin);

    const auto W = uw12::integrals::Integrals(W_base, Co, active_Co);
    const auto V = uw12::integrals::Integrals(V_base, Co, active_Co);

    const auto ttilde = uw12::four_el::calculate_ttilde(W, V);
    REQUIRE(ttilde.size() == n_spin);

    const auto W_direct =
        uw12::integrals::Integrals(W_base_direct, Co, active_Co);
    const auto V_direct =
        uw12::integrals::Integrals(V_base_direct, Co, active_Co);

    SECTION("Test ttilde") {
      const auto ttilde_direct =
          uw12::four_el::calculate_ttilde(W_direct, V_direct);
      REQUIRE(ttilde_direct.size() == n_spin);

      for (size_t sigma = 0; sigma < n_spin; ++sigma) {
        CHECK(uw12::linalg::nearly_equal(
            ttilde[sigma], ttilde_direct[sigma], test::epsilon
        ));
      }
    }

    for (const auto& tt : ttilde) {
      const auto ttilde_dxab =
          uw12::three_el::calculate_ttilde_dxab(W, V, tt, abs_projectors);
      REQUIRE(uw12::linalg::n_rows(ttilde_dxab) == n_ao);
      REQUIRE(uw12::linalg::n_cols(ttilde_dxab) == n_ao);

      const auto ttilde_dxab_direct = uw12::three_el::calculate_ttilde_dxab(
          W_direct, V_direct, tt, abs_projectors
      );
      REQUIRE(uw12::linalg::n_rows(ttilde_dxab_direct) == n_ao);
      REQUIRE(uw12::linalg::n_cols(ttilde_dxab_direct) == n_ao);

      CHECK(uw12::linalg::nearly_equal(
          ttilde_dxab, ttilde_dxab_direct, test::epsilon
      ));
    }

    n_occ.push_back(2);
    n_active.push_back(1);
  }
}
