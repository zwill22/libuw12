//
// Created by Zack Williams on 25/02/2024.
//

#include "../src/integrals/base_integrals.hpp"

#include "catch.hpp"

using namespace uw12::integrals;

constexpr auto seed = 2;

TEST_CASE("Test integrals - base integrals") {
    constexpr auto n_ao = 10;
    constexpr auto n_df = 22;
    constexpr auto n_ri = 33;

    const std::vector df_sizes = {1, 3, 5, 1, 3, 5, 1, 3};

    int total = 0;
    for (const auto size: df_sizes) {
        total += size;
    }
    REQUIRE(total == n_df);

    const TwoIndexFn two_index_fn = []()-> uw12::linalg::Mat {
        return uw12::linalg::random_pd(n_df, seed);
    };

    const ThreeIndexFn three_index_fn = [&df_sizes](const int A) -> uw12::linalg::Mat {
        constexpr auto n_row = n_ao * (n_ao + 1) / 2;
        const auto n_col = df_sizes[A];

        return uw12::linalg::random(n_row, n_col, seed);
    };

    const ThreeIndexFn three_index_ri_fn = [&df_sizes](const int A) -> uw12::linalg::Mat {
        constexpr auto n_row = n_ao * n_ri;
        const auto n_col = df_sizes[A];

        return uw12::linalg::random(n_row, n_col, seed);
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
            two_index_fn, three_index_fn, three_index_ri_fn, df_sizes, n_ao, n_df, n_ri);

        const auto J2 = base_integrals.two_index();
        REQUIRE(uw12::linalg::n_rows(J2) == n_df);
        REQUIRE(uw12::linalg::n_cols(J2) == n_df);
        CHECK(uw12::linalg::is_square(J2));
        CHECK(uw12::linalg::is_symmetric(J2));

        for (int A = 0; A < df_sizes.size(); ++A) {
            const auto J3_A = base_integrals.three_index(A);
            REQUIRE(uw12::linalg::n_rows(J3_A) == n_ao * (n_ao + 1) / 2);
            REQUIRE(uw12::linalg::n_cols(J3_A) == df_sizes[A]);
        }

        for (int A = 0; A < df_sizes.size(); ++A) {
            const auto J3_ri_A = base_integrals.three_index_ri(A);
            REQUIRE(uw12::linalg::n_rows(J3_ri_A) == n_ao * n_ri);
            REQUIRE(uw12::linalg::n_cols(J3_ri_A) == df_sizes[A]);
        }

        CHECK(base_integrals.has_two_index_fn());
        CHECK(base_integrals.has_three_index_fn());
        CHECK(base_integrals.has_three_index_ri_fn());

        const auto & P2 = base_integrals.get_P2();
        REQUIRE(uw12::linalg::n_rows(P2) == n_df);
        REQUIRE(uw12::linalg::n_cols(P2) == n_df);
        CHECK(uw12::linalg::is_square(J2));
        CHECK(uw12::linalg::is_symmetric(J2));

        const auto & df_size = base_integrals.get_df_sizes();
        REQUIRE(df_sizes.size() == df_size.size());
        for (int i = 0; i < df_sizes.size(); ++i) {
            CHECK(df_sizes[i] == df_size[i]);
        }

        const auto df_offsets = base_integrals.get_df_offsets();
        REQUIRE(df_offsets.size() == df_sizes.size());
        int offset = 0;
        for (int i = 0; i < df_sizes.size(); ++i) {
            CHECK(df_offsets[i] == offset);
            offset += df_sizes[i];
        }

        const auto & df_vals = base_integrals.get_df_vals();
        REQUIRE(uw12::linalg::n_elem(df_vals) == n_df);

        CHECK_THROWS(base_integrals.get_J3_0());
        CHECK_THROWS(base_integrals.get_J3_ri_0());

        const auto & J3 = base_integrals.get_J3();
        REQUIRE(uw12::linalg::n_rows(J3) == n_ao * (n_ao + 1) / 2);
        REQUIRE(uw12::linalg::n_cols(J3) == n_df);

        const auto & J3_ri = base_integrals.get_J3_ri();
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
    }
}
