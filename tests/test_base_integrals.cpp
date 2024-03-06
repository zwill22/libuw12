//
// Created by Zack Williams on 25/02/2024.
//

#include "../src/integrals/base_integrals.hpp"
#include "catch.hpp"

using namespace uw12::integrals;
using namespace test;

void check_df_vals(const BaseIntegrals &base_integrals, const size_t n_df) {
  const auto &P2 = base_integrals.get_P2();
  REQUIRE(uw12::linalg::n_rows(P2) == n_df);
  REQUIRE(uw12::linalg::n_cols(P2) == n_df);
  CHECK(uw12::linalg::is_square(P2));

  const auto &df_vals = base_integrals.get_df_vals();
  REQUIRE(uw12::linalg::n_elem(df_vals) == n_df);
}

void check_df_offsets(
    const BaseIntegrals &base_integrals, const std::vector<size_t> &df_sizes
) {
  const auto &df_size = base_integrals.get_df_sizes();
  REQUIRE(df_sizes.size() == df_size.size());
  for (size_t i = 0; i < df_sizes.size(); ++i) {
    CHECK(df_sizes[i] == df_size[i]);
  }

  const auto df_offsets = base_integrals.get_df_offsets();
  REQUIRE(df_offsets.size() == df_sizes.size());
  size_t offset = 0;
  for (size_t i = 0; i < df_sizes.size(); ++i) {
    CHECK(df_offsets[i] == offset);
    offset += df_sizes[i];
  }
}

TEST_CASE("Test integrals - base integrals") {
  constexpr size_t n_ao = 10;
  constexpr size_t n_df = 22;
  constexpr size_t n_ri = 33;

  const std::vector<size_t> df_sizes = {1, 3, 5, 1, 3, 5, 1, 3};

  size_t total = 0;
  for (const auto size : df_sizes) {
    total += size;
  }
  REQUIRE(total == n_df);

  auto J20 = uw12::linalg::random_pd(n_df, seed);
  auto J30 = uw12::linalg::random(n_ao * (n_ao + 1) / 2, n_df, seed);
  auto J3ri0 = uw12::linalg::random(n_ao * n_ri, n_df, seed);

  const TwoIndexFn two_index_fn = [&J20]() -> uw12::linalg::Mat { return J20; };

  const ThreeIndexFn three_index_fn = [&df_sizes, &J30](const size_t A
                                      ) -> uw12::linalg::Mat {
    constexpr auto n_row = n_ao * (n_ao + 1) / 2;
    const auto n_col = df_sizes[A];

    size_t offset = 0;
    for (size_t i = 0; i < A; ++i) {
      offset += df_sizes[i];
    }

    return uw12::linalg::sub_mat(J30, 0, offset, n_row, n_col);
  };

  const ThreeIndexFn three_index_ri_fn = [&df_sizes, &J3ri0](const size_t A
                                         ) -> uw12::linalg::Mat {
    constexpr auto n_row = n_ao * n_ri;
    const auto n_col = df_sizes[A];

    size_t offset = 0;
    for (size_t i = 0; i < A; ++i) {
      offset += df_sizes[i];
    }

    return uw12::linalg::sub_mat(J3ri0, 0, offset, n_row, n_col);
  };

  SECTION("Default constructor") {
    const BaseIntegrals base_integrals;

    CHECK_THROWS(base_integrals.two_index());
    CHECK_THROWS(base_integrals.three_index(0));
    CHECK_THROWS(base_integrals.three_index_ri(0));

    CHECK_FALSE(base_integrals.has_two_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_ri_fn());

    CHECK_THROWS(base_integrals.get_P2());
    CHECK_THROWS(base_integrals.get_df_sizes());
    CHECK_THROWS(base_integrals.get_df_offsets());
    CHECK_THROWS(base_integrals.get_df_vals());

    CHECK_THROWS(base_integrals.get_J3_0());
    CHECK_THROWS(base_integrals.get_J3());
    CHECK_THROWS(base_integrals.get_J3_ri_0());
    CHECK_THROWS(base_integrals.get_J3_ri());

    CHECK(base_integrals.get_number_ao() == 0);
    CHECK(base_integrals.get_number_df() == 0);
    CHECK(base_integrals.get_number_ri() == 0);

    CHECK_FALSE(base_integrals.storing_ao());
    CHECK_FALSE(base_integrals.storing_ri());

    CHECK_FALSE(base_integrals.has_P2());
    CHECK_FALSE(base_integrals.has_df_vals());
    CHECK_FALSE(base_integrals.has_J3_0());
    CHECK_FALSE(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK_FALSE(base_integrals.has_J3_ri());
  }

  SECTION("Standard constructor") {
    const auto base_integrals = BaseIntegrals(
        two_index_fn,
        three_index_fn,
        three_index_ri_fn,
        df_sizes,
        n_ao,
        n_df,
        n_ri
    );

    const auto J2 = base_integrals.two_index();
    REQUIRE(uw12::linalg::n_rows(J2) == n_df);
    REQUIRE(uw12::linalg::n_cols(J2) == n_df);
    CHECK(uw12::linalg::is_square(J2));
    CHECK(uw12::linalg::is_symmetric(J2));

    for (size_t A = 0; A < df_sizes.size(); ++A) {
      const auto J3_A = base_integrals.three_index(A);
      REQUIRE(uw12::linalg::n_rows(J3_A) == n_ao * (n_ao + 1) / 2);
      REQUIRE(uw12::linalg::n_cols(J3_A) == df_sizes[A]);
    }

    for (size_t A = 0; A < df_sizes.size(); ++A) {
      const auto J3_ri_A = base_integrals.three_index_ri(A);
      REQUIRE(uw12::linalg::n_rows(J3_ri_A) == n_ao * n_ri);
      REQUIRE(uw12::linalg::n_cols(J3_ri_A) == df_sizes[A]);
    }

    CHECK(base_integrals.has_two_index_fn());
    CHECK(base_integrals.has_three_index_fn());
    CHECK(base_integrals.has_three_index_ri_fn());

    check_df_vals(base_integrals, n_df);
    check_df_offsets(base_integrals, df_sizes);

    CHECK_THROWS(base_integrals.get_J3_0());
    CHECK_THROWS(base_integrals.get_J3_ri_0());

    const auto &J3 = base_integrals.get_J3();
    REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
    REQUIRE(uw12::linalg::n_cols(J3) == n_df);

    const auto &J3_ri = base_integrals.get_J3_ri();
    REQUIRE(uw12::linalg::n_rows(J3_ri) == n_ao * n_ri);
    REQUIRE(uw12::linalg::n_cols(J3_ri) == n_df);

    CHECK(base_integrals.get_number_ao() == n_ao);
    CHECK(base_integrals.get_number_df() == n_df);
    CHECK(base_integrals.get_number_ri() == n_ri);

    CHECK(base_integrals.storing_ao());
    CHECK(base_integrals.storing_ri());

    CHECK(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK_FALSE(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK(base_integrals.has_J3_ri());
    {
      const auto base_integrals2 = BaseIntegrals(
          two_index_fn,
          three_index_fn,
          three_index_ri_fn,
          df_sizes,
          n_ao,
          n_df,
          n_ri,
          false
      );

      CHECK_FALSE(base_integrals2.storing_ao());
      CHECK(base_integrals2.storing_ri());
      CHECK_THROWS(base_integrals2.get_J3_0());
      CHECK_THROWS(base_integrals2.get_J3_ri_0());

      const auto &P2_2 = base_integrals2.get_P2();
      CHECK(uw12::linalg::nearly_equal(base_integrals.get_P2(), P2_2, epsilon));

      const auto &df_vals_2 = base_integrals2.get_df_vals();
      CHECK(uw12::linalg::nearly_equal(
          base_integrals.get_df_vals(), df_vals_2, epsilon
      ));

      const auto &J3_2 = base_integrals2.get_J3();
      CHECK(uw12::linalg::nearly_equal(J3, J3_2, epsilon));

      const auto &J3_ri_2 = base_integrals2.get_J3_ri();
      CHECK(uw12::linalg::nearly_equal(J3_ri, J3_ri_2, epsilon));
    }
    {
      const auto base_integrals3 = BaseIntegrals(
          two_index_fn,
          three_index_fn,
          three_index_ri_fn,
          df_sizes,
          n_ao,
          n_df,
          n_ri,
          false,
          false
      );

      CHECK_FALSE(base_integrals3.storing_ao());
      CHECK_FALSE(base_integrals3.storing_ri());
      CHECK_THROWS(base_integrals3.get_J3_0());
      CHECK_THROWS(base_integrals3.get_J3_ri_0());

      const auto &P2_3 = base_integrals3.get_P2();
      CHECK(uw12::linalg::nearly_equal(base_integrals.get_P2(), P2_3, epsilon));

      const auto &df_vals_3 = base_integrals3.get_df_vals();
      CHECK(uw12::linalg::nearly_equal(
          base_integrals.get_df_vals(), df_vals_3, epsilon
      ));

      const auto &J3_3 = base_integrals3.get_J3();
      CHECK(uw12::linalg::nearly_equal(J3, J3_3, epsilon));

      const auto &J3_ri_3 = base_integrals3.get_J3_ri();
      CHECK(uw12::linalg::nearly_equal(J3_ri, J3_ri_3, epsilon));
    }
  }

  SECTION("Standard constructor (no RI)") {
    const auto base_integrals =
        BaseIntegrals(two_index_fn, three_index_fn, df_sizes, n_ao, n_df);

    const auto J2 = base_integrals.two_index();
    REQUIRE(uw12::linalg::n_rows(J2) == n_df);
    REQUIRE(uw12::linalg::n_cols(J2) == n_df);
    CHECK(uw12::linalg::is_square(J2));
    CHECK(uw12::linalg::is_symmetric(J2));

    for (size_t A = 0; A < df_sizes.size(); ++A) {
      const auto J3_A = base_integrals.three_index(A);
      REQUIRE(uw12::linalg::n_rows(J3_A) == n_ao * (n_ao + 1) / 2);
      REQUIRE(uw12::linalg::n_cols(J3_A) == df_sizes[A]);
    }

    REQUIRE_THROWS(base_integrals.three_index_ri(0));

    CHECK(base_integrals.has_two_index_fn());
    CHECK(base_integrals.has_three_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_ri_fn());

    check_df_vals(base_integrals, n_df);
    check_df_offsets(base_integrals, df_sizes);

    CHECK_THROWS(base_integrals.get_J3_0());
    CHECK_THROWS(base_integrals.get_J3_ri_0());
    CHECK_THROWS(base_integrals.get_J3_ri());

    const auto &J3 = base_integrals.get_J3();
    REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
    REQUIRE(uw12::linalg::n_cols(J3) == n_df);

    CHECK(base_integrals.get_number_ao() == n_ao);
    CHECK(base_integrals.get_number_df() == n_df);
    CHECK(base_integrals.get_number_ri() == 0);

    CHECK(base_integrals.storing_ao());
    CHECK_FALSE(base_integrals.storing_ri());

    CHECK(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK_FALSE(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK_FALSE(base_integrals.has_J3_ri());
    {
      const auto base_integrals2 = BaseIntegrals(
          two_index_fn, three_index_fn, df_sizes, n_ao, n_df, false
      );

      CHECK_FALSE(base_integrals2.storing_ao());
      CHECK_FALSE(base_integrals2.storing_ri());
      CHECK_THROWS(base_integrals2.get_J3_0());
      CHECK_THROWS(base_integrals2.get_J3_ri_0());
      CHECK_THROWS(base_integrals2.get_J3_ri());

      const auto &P2_2 = base_integrals2.get_P2();
      CHECK(uw12::linalg::nearly_equal(base_integrals.get_P2(), P2_2, epsilon));

      const auto &df_vals_2 = base_integrals2.get_df_vals();
      CHECK(uw12::linalg::nearly_equal(
          base_integrals.get_df_vals(), df_vals_2, epsilon
      ));

      const auto &J3_2 = base_integrals2.get_J3();
      CHECK(uw12::linalg::nearly_equal(J3, J3_2, epsilon));
    }
  }

  SECTION("J3_0 constructor") {
    const auto base_integrals = BaseIntegrals(
        J30, J20, three_index_ri_fn, df_sizes, true, n_ao, n_df, n_ri
    );

    CHECK_THROWS(base_integrals.two_index());
    CHECK_THROWS(base_integrals.three_index(0));

    for (size_t A = 0; A < df_sizes.size(); ++A) {
      const auto J3_ri_A = base_integrals.three_index_ri(A);
      REQUIRE(uw12::linalg::n_rows(J3_ri_A) == n_ao * n_ri);
      REQUIRE(uw12::linalg::n_cols(J3_ri_A) == df_sizes[A]);
    }

    CHECK_FALSE(base_integrals.has_two_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_fn());
    CHECK(base_integrals.has_three_index_ri_fn());

    check_df_vals(base_integrals, n_df);
    check_df_offsets(base_integrals, df_sizes);

    const auto &J3_0 = base_integrals.get_J3_0();
    CHECK(uw12::linalg::nearly_equal(J3_0, J30, epsilon));

    CHECK_THROWS(base_integrals.get_J3_ri_0());

    const auto &J3 = base_integrals.get_J3();
    REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
    REQUIRE(uw12::linalg::n_cols(J3) == n_df);

    const auto &J3_ri = base_integrals.get_J3_ri();
    REQUIRE(uw12::linalg::n_rows(J3_ri) == n_ao * n_ri);
    REQUIRE(uw12::linalg::n_cols(J3_ri) == n_df);

    CHECK(base_integrals.get_number_ao() == n_ao);
    CHECK(base_integrals.get_number_df() == n_df);
    CHECK(base_integrals.get_number_ri() == n_ri);

    CHECK(base_integrals.storing_ao());
    CHECK_FALSE(base_integrals.storing_ri());

    CHECK(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK(base_integrals.has_J3_ri());
    {
      const auto base_integrals2 = BaseIntegrals(
          J30, J20, three_index_ri_fn, df_sizes, false, n_ao, n_df, n_ri, false
      );

      CHECK(base_integrals2.storing_ao());
      CHECK_FALSE(base_integrals2.storing_ri());

      const auto &J3_0_2 = base_integrals2.get_J3_0();
      CHECK(uw12::linalg::nearly_equal(J3_0, J3_0_2, epsilon));

      CHECK_THROWS(base_integrals2.get_J3_ri_0());
      CHECK_THROWS(base_integrals2.get_J3_ri());
    }
  }

  SECTION("J3_0 constructor (RI_0)") {
    const auto base_integrals = BaseIntegrals(J30, J20, J3ri0);

    CHECK_THROWS(base_integrals.two_index());
    CHECK_THROWS(base_integrals.three_index(0));
    CHECK_THROWS(base_integrals.three_index_ri(0));

    CHECK_FALSE(base_integrals.has_two_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_ri_fn());

    check_df_vals(base_integrals, n_df);
    CHECK_THROWS(base_integrals.get_df_sizes());
    CHECK_THROWS(base_integrals.get_df_offsets());

    const auto &J3_0 = base_integrals.get_J3_0();
    CHECK(uw12::linalg::nearly_equal(J3_0, J30, epsilon));

    const auto &J3_ri_0 = base_integrals.get_J3_ri_0();
    CHECK(uw12::linalg::nearly_equal(J3_ri_0, J3ri0, epsilon));

    const auto &J3 = base_integrals.get_J3();
    REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
    REQUIRE(uw12::linalg::n_cols(J3) == n_df);

    const auto &J3_ri = base_integrals.get_J3_ri();
    REQUIRE(uw12::linalg::n_rows(J3_ri) == n_ao * n_ri);
    REQUIRE(uw12::linalg::n_cols(J3_ri) == n_df);

    CHECK(base_integrals.get_number_ao() == 0);
    CHECK(base_integrals.get_number_df() == 0);
    CHECK(base_integrals.get_number_ri() == 0);

    CHECK(base_integrals.storing_ao());
    CHECK(base_integrals.storing_ri());

    CHECK(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK(base_integrals.has_J3_ri_0());
    CHECK(base_integrals.has_J3_ri());
  }

  SECTION("J3_0 constructor - No RI") {
    const auto base_integrals = BaseIntegrals(J30, J20);

    CHECK_THROWS(base_integrals.two_index());
    CHECK_THROWS(base_integrals.three_index(0));
    CHECK_THROWS(base_integrals.three_index_ri(0));

    CHECK_FALSE(base_integrals.has_two_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_ri_fn());

    check_df_vals(base_integrals, n_df);
    CHECK_THROWS(base_integrals.get_df_sizes());
    CHECK_THROWS(base_integrals.get_df_offsets());

    const auto &J3_0 = base_integrals.get_J3_0();
    CHECK(uw12::linalg::nearly_equal(J3_0, J30, epsilon));

    CHECK_THROWS(base_integrals.get_J3_ri_0());

    const auto &J3 = base_integrals.get_J3();
    REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
    REQUIRE(uw12::linalg::n_cols(J3) == n_df);

    CHECK_THROWS(base_integrals.get_J3_ri());

    CHECK(base_integrals.get_number_ao() == 0);
    CHECK(base_integrals.get_number_df() == 0);
    CHECK(base_integrals.get_number_ri() == 0);

    CHECK(base_integrals.storing_ao());
    CHECK_FALSE(base_integrals.storing_ri());

    CHECK(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK_FALSE(base_integrals.has_J3_ri());
  }

  SECTION("J3 constructor - with df_vals") {
    const auto &[vals, vecs] = uw12::linalg::eigen_system(J20);
    const uw12::linalg::Mat P2 =
        vecs * uw12::linalg::p_inv(uw12::linalg::diagmat(vals));
    const uw12::linalg::Mat J3 = J30 * P2;

    const auto base_integrals = BaseIntegrals(J3, vals);

    CHECK_THROWS(base_integrals.two_index());
    CHECK_THROWS(base_integrals.three_index(0));
    CHECK_THROWS(base_integrals.three_index_ri(0));

    CHECK_FALSE(base_integrals.has_two_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_fn());
    CHECK_FALSE(base_integrals.has_three_index_ri_fn());

    CHECK_THROWS(base_integrals.get_P2());

    const auto &df_vals = base_integrals.get_df_vals();
    CHECK(uw12::linalg::nearly_equal(df_vals, vals, epsilon));

    CHECK_THROWS(base_integrals.get_df_sizes());
    CHECK_THROWS(base_integrals.get_df_offsets());

    CHECK_THROWS(base_integrals.get_J3_0());
    CHECK_THROWS(base_integrals.get_J3_ri_0());
    CHECK_THROWS(base_integrals.get_J3_ri());

    const auto &J3_2 = base_integrals.get_J3();
    CHECK(uw12::linalg::nearly_equal(J3_2, J3, epsilon));

    CHECK(base_integrals.get_number_ao() == 0);
    CHECK(base_integrals.get_number_df() == 0);
    CHECK(base_integrals.get_number_ri() == 0);

    CHECK(base_integrals.storing_ao());
    CHECK_FALSE(base_integrals.storing_ri());

    CHECK_FALSE(base_integrals.has_P2());
    CHECK(base_integrals.has_df_vals());
    CHECK_FALSE(base_integrals.has_J3_0());
    CHECK(base_integrals.has_J3());
    CHECK_FALSE(base_integrals.has_J3_ri_0());
    CHECK_FALSE(base_integrals.has_J3_ri());
  }

  SECTION("Test construction fails") {
    CHECK_THROWS(BaseIntegrals(two_index_fn, three_index_fn, {}, n_ao, n_df));
    CHECK_THROWS(BaseIntegrals(two_index_fn, three_index_fn, df_sizes, 0, n_df)
    );
    CHECK_THROWS(BaseIntegrals(two_index_fn, three_index_fn, df_sizes, n_ao, 0)
    );
    CHECK_THROWS(
        BaseIntegrals(two_index_fn, three_index_fn, df_sizes, n_ao, n_df + 1)
    );
    CHECK_THROWS(BaseIntegrals(
        two_index_fn, three_index_fn, three_index_ri_fn, df_sizes, n_ao, n_df, 0
    ));
  }
}
