//
// Created by Zack Williams on 23/02/2024.
//

#include "../src/integrals/transformations.hpp"

#include "catch.hpp"

using namespace uw12;
using namespace uw12::integrals::transformations;

TEST_CASE("Test integrals - MO transformations") {
    constexpr size_t n_ao = 11;
    constexpr size_t n_df = 20;
    constexpr size_t n_orb = 7;

    const auto J3 = linalg::random(n_ao * (n_ao + 1) / 2, n_df, seed);
    const auto C = linalg::random(n_ao, n_orb, seed);

    INFO("TEST one transform first index full");

    REQUIRE_THROWS(mo_transform_one_index_full(linalg::head_rows(J3, n_ao), C));

    const auto J3_mjA = mo_transform_one_index_full(J3, C);

    REQUIRE(linalg::n_rows(J3_mjA) == n_ao * n_orb);
    REQUIRE(linalg::n_cols(J3_mjA) == n_df);

    for (size_t col_idx = 0; col_idx < n_df; ++col_idx) {
        const linalg::Mat mat1 = linalg::reshape_col(J3_mjA, col_idx, n_ao, n_orb);
        const linalg::Mat mat2 = utils::square(linalg::col(J3, col_idx)) * C;

        CHECK(linalg::nearly_equal(mat1, mat2, epsilon));
    }

    INFO("TEST one tranform first index");
    REQUIRE_THROWS(transform_first_index(J3, linalg::head_rows(C, n_orb)));

    constexpr size_t n_occ = n_orb - 1;
    const auto Co = linalg::tail_cols(C, n_occ);

    const auto J3_ijA = transform_first_index(J3_mjA, Co);

    REQUIRE(linalg::n_rows(J3_ijA) == n_orb * n_occ);
    REQUIRE(linalg::n_cols(J3_ijA) == n_df);

    for (size_t col_idx = 0; col_idx < n_df; ++col_idx) {
        const linalg::Mat mat1 = linalg::reshape_col(J3_ijA, col_idx, n_occ, n_orb);
        const linalg::Mat mat2 = linalg::transpose(Co) * linalg::reshape_col(J3_mjA, col_idx, n_ao, n_orb);

        CHECK(linalg::nearly_equal(mat1, mat2, epsilon));
    }

    INFO("Test two index transform");

    REQUIRE_THROWS(mo_transform_two_index_full(J3, C, linalg::head_rows(C, n_orb)));

    const auto J3_ijB = mo_transform_two_index_full(J3, Co, C);

    CHECK(linalg::nearly_equal(J3_ijA, J3_ijB, epsilon));
}

TEST_CASE("Test integrals - MO transformations (RI)") {
    constexpr size_t n_ao = 11;
    constexpr size_t n_ri = 25;
    constexpr size_t n_df = 20;
    constexpr size_t n_orb = 7;

    const auto J3_pvA = linalg::random(n_ri * n_ao, n_df, seed);
    const auto C = linalg::random(n_ao, n_orb, seed);

    REQUIRE_THROWS(transform_second_index(linalg::head_rows(J3_pvA, n_ri), C));

    const auto J3_piA = transform_second_index(J3_pvA, C);

    REQUIRE(linalg::n_rows(J3_piA) == n_ri * n_orb);
    REQUIRE(linalg::n_cols(J3_piA) == n_df);

    for (size_t col_idx = 0; col_idx < n_df; ++col_idx) {
        const linalg::Mat mat1 = linalg::reshape_col(J3_piA, col_idx, n_ri, n_orb);
        const linalg::Mat mat2 = linalg::reshape_col(J3_pvA, col_idx, n_ri, n_ao) * C;

        CHECK(linalg::nearly_equal(mat1, mat2, epsilon));
    }
}
