//
// Created by Zack Williams on 22/02/2024.
//

#include "../src/utils/utils.hpp"

#include "catch.hpp"

using namespace uw12;
using namespace utils;

using Catch::Matchers::WithinAbs;

constexpr double margin = 1e-10;
constexpr int seed = 22;

TEST_CASE("Test utils - Matrix utils") {

    SECTION("Test square matrix") {

        constexpr size_t n = 6;

        constexpr int n2 = n * (n+1) / 2;

        const linalg::Vec vec = linalg::random(n2, 1, seed);

        // Error for non-triagular number
        REQUIRE_THROWS(square(linalg::ones(13)));


        const auto sq = square(vec);
        REQUIRE(linalg::n_rows(sq) == n);
        REQUIRE(linalg::n_cols(sq) == n);

        REQUIRE(linalg::is_square(sq));
        REQUIRE(linalg::is_symmetric(sq));

        size_t col_idx = 0;
        size_t row_idx = 0;
        for (int i = 0; i < n2; ++i) {
            CHECK_THAT(linalg::elem(sq, row_idx, col_idx), WithinAbs(linalg::elem(vec, i), margin));
            col_idx++;
            if (col_idx > row_idx) {
                row_idx++;
                col_idx = 0;
            }
        }

        CHECK(linalg::nearly_equal(lower(sq), vec, margin));
    }

    SECTION("Test lower matrix") {
        constexpr size_t n = 6;

        const auto sq = linalg::random_pd(n, seed);

        CHECK_THROWS(lower(linalg::random(n, n+1, seed)));
        CHECK_THROWS(lower(linalg::random(n, n, seed)));

        for (const auto factor: {1.0, 2.0, 3.5}) {
            const auto vec = lower(sq, factor);
            constexpr int n2 = n * (n+1) / 2;
            REQUIRE(linalg::n_elem(vec) == n2);

            size_t col_idx = 0;
            size_t row_idx = 0;
            for (int i = 0; i < n2; ++i) {
                const auto target = linalg::elem(vec, i) / (col_idx == row_idx ? 1.0: factor);
                CHECK_THAT(linalg::elem(sq, row_idx, col_idx), WithinAbs(target, margin));
                col_idx++;
                if (col_idx > row_idx) {
                    row_idx++;
                    col_idx = 0;
                }
            }
        }
    }
}

TEST_CASE("Test utils - MatVec") {
    const auto mat1 = linalg::random(10, 20, seed);

    const auto mat2 = linalg::random(13, 16, seed);

    const MatVec mat_vec = {&mat1, &mat2};

    SECTION("Multiplication") {
        constexpr auto factor = 2;
        const auto multiple = mat_vec * factor;
        const auto multiple2 = factor * mat_vec;

        REQUIRE(multiple.size() == multiple2.size());

        for (int i = 0; i < multiple.size(); ++i) {
            const linalg::Mat mat = factor * mat_vec[i];

            CHECK(linalg::nearly_equal(multiple[i], multiple2[i], margin));
            CHECK(linalg::nearly_equal(multiple[i], mat, margin));
        }
    }
}