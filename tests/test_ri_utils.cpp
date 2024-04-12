//
// Created by Zack Williams on 26/03/2024.
//

#include "catch.hpp"
#include "three_electron/ri_utils.hpp"
#include "utils/linalg.hpp"

using uw12::linalg::n_cols;
using uw12::linalg::n_rows;

TEST_CASE("Test RI utils - Test ABS projectors") {
  constexpr size_t n_ao = 10;
  constexpr size_t n_ri = 24;

  const auto s = uw12::linalg::random_pd(n_ao + n_ri, uw12_test::seed);

  REQUIRE(uw12::linalg::n_rows(s) == n_ao + n_ri);
  REQUIRE(uw12::linalg::n_cols(s) == n_ao + n_ri);

  const auto abs_proj =
      uw12::three_el::ri::calculate_abs_projectors(s, n_ao, n_ri);

  const auto s_inv_ao_ao = abs_proj.s_inv_ao_ao;
  REQUIRE(n_rows(s_inv_ao_ao) == n_ao);
  REQUIRE(n_cols(s_inv_ao_ao) == n_ao);

  const auto s_inv_ao_ri = abs_proj.s_inv_ao_ri;
  REQUIRE(n_rows(s_inv_ao_ri) == n_ao);
  REQUIRE(n_cols(s_inv_ao_ri) == n_ri);

  const auto s_inv_ri_ao = abs_proj.s_inv_ri_ao;
  REQUIRE(n_rows(s_inv_ri_ao) == n_ri);
  REQUIRE(n_cols(s_inv_ri_ao) == n_ao);

  const auto s_inv_ri_ri = abs_proj.s_inv_ri_ri;
  REQUIRE(n_rows(s_inv_ri_ri) == n_ri);
  REQUIRE(n_cols(s_inv_ri_ri) == n_ri);

  CHECK_THROWS(uw12::three_el::ri::calculate_abs_projectors(s, n_ao + 1, n_ri));
  CHECK_THROWS(uw12::three_el::ri::calculate_abs_projectors(s, n_ao - 1, n_ri));
  CHECK_THROWS(uw12::three_el::ri::calculate_abs_projectors(s, n_ao, n_ri + 1));
  CHECK_THROWS(uw12::three_el::ri::calculate_abs_projectors(s, n_ao, n_ri - 1));
}
