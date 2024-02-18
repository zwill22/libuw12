//
// Created by Zack Williams on 18/02/2024.
//

#include "../src/utils/linalg.hpp"

#include "catch.hpp"
#include <iostream>

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
}
