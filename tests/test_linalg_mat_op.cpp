//
// Created by Zack Williams on 18/02/2024.
//

#include "../src/utils/linalg.hpp"

#include "catch.hpp"

using namespace uw12;

using Catch::Matchers::WithinAbs;

constexpr auto margin = 1e-10;
constexpr int seed = 22;

TEST_CASE("Test linear algebra - Test matrix properties") {
    constexpr size_t n_row = 12;
    constexpr size_t n_col = 14;

    const auto mat = linalg::random(n_row, n_col, seed);
    REQUIRE(linalg::n_rows(mat) == n_row);
    REQUIRE(linalg::n_cols(mat) == n_col);

    SECTION("Test norm") {
        const auto norm = linalg::norm(mat);

        INFO("2-Norm of matrix = " << norm);

        // Check upper and lower bound on 2-norm
        // ||A||_{max} \leq ||A||_2 \leq ||A||_F
        // where:
        // ||A||_{max} = \max_{ij} | a_{ij} |
        // ||A||_{F} = \sqrt{\sum_ij | a_{ij} |^2}
        double max_norm = 0;
        double total = 0;
        for (size_t col_index = 0; col_index < n_col; ++col_index) {
            for (size_t row_index = 0; row_index < n_row; ++row_index) {
                const auto elem = linalg::elem(mat, row_index, col_index);
                total += elem*elem;
                if (const auto abs = std::abs(elem); abs > max_norm) {
                    max_norm = abs;
                }
            }
        }
        INFO("max-norm of matrix = " << max_norm);

        const auto fro_norm = std::sqrt(total);
        INFO("F-Norm of matrix = " << fro_norm);
        CHECK(norm >= max_norm);
        CHECK(norm <= fro_norm);

        // Another upper bound for matrix of size mxn
        // ||A||_2 \leq \sqrt{mn} ||A||_{max}
        CHECK(norm <= std::sqrt(n_row * n_col) * max_norm);
    }

    SECTION("Test vectorise") {
        const auto vec = linalg::vectorise(mat);

        const auto n = linalg::n_elem(mat);
        REQUIRE(linalg::n_elem(vec) == n);
        REQUIRE(n == n_row * n_col);

        for (size_t i = 0; i < n; ++i) {
            const size_t row_idx = i % n_row;
            const size_t col_idx = i / n_row;

            CHECK_THAT(linalg::elem(mat, row_idx, col_idx), WithinAbs(linalg::elem(vec, i), margin));
        }
    }

    SECTION("Test trace") {
        REQUIRE_THROWS(linalg::trace(mat));

        const auto square = linalg::head_cols(mat, n_row);
        REQUIRE(linalg::n_rows(square) == n_row);
        REQUIRE(linalg::n_cols(square) == n_row);

        const auto trace = linalg::trace(square);

        double total = 0;
        for (size_t i = 0; i < n_row; ++i) {
            total += linalg::elem(square, i, i);
        }

        CHECK_THAT(total, WithinAbs(trace, margin));
    }

    SECTION("Test diagmat") {
        const auto vec = linalg::col(mat, 0);

        const auto diagmat = linalg::diagmat(vec);

        REQUIRE(linalg::n_elem(vec) == n_row);
        REQUIRE(linalg::n_cols(diagmat) == n_row);
        REQUIRE(linalg::n_rows(diagmat) == n_row);

        for (size_t col_idx = 0; col_idx < n_row; ++col_idx) {
            for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
                double target = 0;
                if (row_idx == col_idx) {
                    target = linalg::elem(vec, row_idx);
                }

                CHECK_THAT(linalg::elem(diagmat, row_idx, col_idx), WithinAbs(target, margin));
            }
        }

    }

    SECTION("Test sqrt") {
        for (size_t i = 0; i < n_col; ++i) {
            const auto col = linalg::col(mat, i);
            REQUIRE(linalg::all_positive(col));
        }

        const linalg::Mat neg = -1 * mat;

        REQUIRE_THROWS(linalg::sqrt(neg));

        const auto sqrt = linalg::sqrt(mat);
        for (size_t col_idx = 0; col_idx < n_row; ++col_idx) {
            for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
                double target = std::sqrt(linalg::elem(mat, row_idx, col_idx));

                CHECK_THAT(linalg::elem(sqrt, row_idx, col_idx), WithinAbs(target, margin));
            }
        }
    }
}
