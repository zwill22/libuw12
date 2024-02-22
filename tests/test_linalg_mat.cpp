//
// Created by Zack Williams on 15/02/2024.
//


#include "../src/utils/linalg.hpp"

#include "catch.hpp"

using namespace uw12;

using Catch::Matchers::WithinAbs;

constexpr auto margin = 1e-10;
constexpr int seed = 22;

TEST_CASE("Test linear algebra - Test Matrix initialisation") {
    SECTION("Test memory initialiser") {
        std::vector<double> vector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

        const auto n_elem = vector.size();

        constexpr size_t n_row = 3;
        constexpr size_t n_col = 4;

        REQUIRE(n_row * n_col == n_elem);

        const auto mat = linalg::mat(vector.data(), n_row, n_col, true);

        CHECK(linalg::n_elem(mat) == n_elem);
        CHECK(linalg::n_rows(mat) == n_row);
        CHECK(linalg::n_cols(mat) == n_col);
        CHECK_FALSE(linalg::is_square(mat));

        for (size_t i = 0; i < n_elem; ++i) {
            const size_t row_index = i % n_row;
            const size_t col_index = i / n_row;
            REQUIRE(row_index < n_row);
            REQUIRE(col_index < n_col);

            CHECK_THAT(linalg::elem(mat, row_index, col_index), WithinAbs(vector[i], margin));
        }
    }

    SECTION("Check size initialiser") {
        constexpr size_t n_row = 4;
        constexpr size_t n_col = 3;
        constexpr auto n_elem = n_row * n_col;

        auto mat = linalg::mat(n_row, n_col);
        CHECK(linalg::n_elem(mat) == n_elem);
        CHECK(linalg::n_rows(mat) == n_row);
        CHECK(linalg::n_cols(mat) == n_col);
        CHECK_FALSE(linalg::is_square(mat));
    }

    SECTION("Test matrix of ones") {
        constexpr size_t n_row = 3;
        constexpr size_t n_col = 2;
        constexpr auto n_elem = n_row * n_col;

        const auto mat = linalg::ones(n_row, n_col);
        CHECK(linalg::n_elem(mat) == n_elem);
        CHECK(linalg::n_rows(mat) == n_row);
        CHECK(linalg::n_cols(mat) == n_col);
        CHECK_FALSE(linalg::is_square(mat));

        for (size_t i = 0; i < linalg::n_elem(mat); ++i) {
            const size_t row_index = i % n_row;
            const size_t col_index = i / n_row;

            REQUIRE(row_index < n_row);
            REQUIRE(col_index < n_col);

            CHECK_THAT(linalg::elem(mat, row_index, col_index), WithinAbs(1, margin));
        }
    }

    SECTION("Test matrix of ones") {
        constexpr size_t n_rows = 5;
        constexpr size_t n_cols = 9;
        constexpr auto n_elem = n_rows * n_cols;

        const auto mat = linalg::zeros(n_rows, n_cols);
        CHECK(linalg::n_elem(mat) == n_elem);
        CHECK(linalg::n_rows(mat) == n_rows);
        CHECK(linalg::n_cols(mat) == n_cols);
        CHECK_FALSE(linalg::is_square(mat));

        for (size_t i = 0; i < linalg::n_elem(mat); ++i) {
            const size_t row_index = i % n_rows;
            const size_t col_index = i / n_rows;

            REQUIRE(row_index < n_rows);
            REQUIRE(col_index < n_cols);

            CHECK_THAT(linalg::elem(mat, row_index, col_index), WithinAbs(0, margin));
        }
    }

    SECTION("Test identity matrix") {
        constexpr size_t n = 11;

        const auto mat = linalg::id(n);
        REQUIRE(linalg::n_elem(mat) == n*n);
        REQUIRE(linalg::n_rows(mat) == n);
        REQUIRE(linalg::n_cols(mat) == n);
        REQUIRE((linalg::is_square(mat)));

        for (size_t i = 0; i < n * n; ++i) {
            const size_t row_index = i % n;
            const size_t col_index = i / n;

            REQUIRE(row_index < n);
            REQUIRE(col_index < n);

            double target = 0;
            if (row_index == col_index) {
                target = 1;
            }
            CHECK_THAT(linalg::elem(mat, row_index, col_index), WithinAbs(target, margin));
        }
    }


    SECTION("Test random matrices") {
        constexpr size_t n_row = 11;
        constexpr size_t n_col = 7;

        const auto mat = linalg::random(n_row, n_col, seed);
        REQUIRE(linalg::n_elem(mat) == n_row * n_col);
        REQUIRE(linalg::n_rows(mat) == n_row);
        REQUIRE(linalg::n_cols(mat) == n_col);

        const auto mat2 = linalg::random(n_row, n_col, seed);
        CHECK(linalg::nearly_equal(mat, mat2, margin));
    }

    SECTION("Random positive definite matrices") {
        constexpr size_t n = 12;

        // Random positive definite symmetric matrix
        const auto mat_pd = linalg::random_pd(n, seed);

        REQUIRE(linalg::n_elem(mat_pd) == n * n);
        REQUIRE(linalg::n_rows(mat_pd) == n);
        REQUIRE(linalg::n_cols(mat_pd) == n);
        REQUIRE(linalg::is_square(mat_pd));
        CHECK(linalg::is_symmetric(mat_pd));
    }
}

TEST_CASE("Test linear algebra - Test Matrix operations") {
    constexpr size_t n_row = 10;
    constexpr size_t n_col = 15;

    const auto mat1 = linalg::random(n_row, n_col, seed);

    SECTION("Check dot product") {
        const auto dot = linalg::dot(mat1, mat1);

        double sum = 0;
        for (size_t col_index = 0; col_index < n_col; ++col_index) {
            for (size_t row_index = 0; row_index < n_row; ++row_index) {
                sum += linalg::elem(mat1, row_index, col_index) * linalg::elem(mat1, row_index, col_index);
            }
        }

        CHECK_THAT(dot, WithinAbs(sum, margin));

        CHECK_THROWS(linalg::dot(mat1, linalg::transpose(mat1)));
        CHECK_THROWS(linalg::dot(mat1, linalg::random(n_col, n_col, seed)));

    }

    SECTION("Check reshape") {
        constexpr size_t n_row2 = 3;
        constexpr size_t n_col2 = n_row * n_col / n_row2;
        REQUIRE(n_row2 * n_col2 == n_row * n_col);
        const auto mat2 = linalg::reshape(mat1, n_row2, n_col2);
        const auto n_elem = linalg::n_elem(mat2);
        REQUIRE(n_elem == linalg::n_elem(mat1));

        for (size_t i = 0; i < n_elem; ++i) {
            const size_t row_idx1 = i % n_row;
            const size_t col_idx1 = i / n_row;

            const size_t row_idx2 = i % n_row2;
            const size_t col_idx2 = i / n_row2;

            CHECK_THAT(linalg::elem(mat1, row_idx1, col_idx1), WithinAbs(linalg::elem(mat2, row_idx2, col_idx2), margin));
        }

        CHECK_THROWS(linalg::reshape(mat1, n_row, n_col2));
    }

    SECTION("Check reshape col") {
        constexpr size_t n_row2 = 2;
        constexpr size_t n_col2 = n_row / n_row2;
        REQUIRE(n_col2 * n_row2 == n_row);

        for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
            const auto new_mat = linalg::reshape_col(mat1, col_idx, n_row2, n_col2);

            REQUIRE(linalg::n_elem(new_mat) == n_row);
            for (size_t idx = 0; idx < n_row; ++idx) {
                const size_t row_idx2 = idx % n_row2;
                const size_t col_idx2 = idx / n_row2;

                CHECK_THAT(linalg::elem(mat1, idx, col_idx), WithinAbs(linalg::elem(new_mat, row_idx2, col_idx2), margin));
            }
        }

        CHECK_THROWS(linalg::reshape_col(mat1, n_col, n_row2, n_col2));
        CHECK_THROWS(linalg::reshape_col(mat1, 0, n_row, n_col));
    }

    SECTION("Check pseudoinverse") {
        const auto mat_inv = linalg::p_inv(mat1);
        REQUIRE(linalg::n_elem(mat_inv) == n_row * n_col);
        REQUIRE(linalg::n_rows(mat_inv) == n_col);
        REQUIRE(linalg::n_cols(mat_inv) == n_row);

        const linalg::Mat product = mat1 * mat_inv;
        REQUIRE(linalg::n_rows(product) == n_row);
        REQUIRE(linalg::n_cols(product) == n_row);

        for (int col_index = 0; col_index < n_row; ++col_index) {
            for (int row_index = 0; row_index < n_row; ++row_index) {
                double target = 0;
                if (col_index == row_index) {
                    target = 1;
                }

                CHECK_THAT(linalg::elem(product, row_index, col_index), WithinAbs(target, margin));
            }
        }
    }

    SECTION("Check inverse of symmetric pd matrix") {
        constexpr size_t n = 11;

        const auto mat_pd = linalg::random_pd(n, seed);
        REQUIRE(linalg::n_elem(mat_pd) == n * n);
        REQUIRE(linalg::n_rows(mat_pd) == n);
        REQUIRE(linalg::n_cols(mat_pd) == n);
        REQUIRE(linalg::is_square(mat_pd));
        REQUIRE(linalg::is_symmetric(mat_pd));

        const auto inv = linalg::inv_sym_pd(mat_pd);
        CHECK_THROWS(linalg::inv_sym_pd(mat1));
        REQUIRE(linalg::n_elem(inv) == n * n);
        REQUIRE(linalg::n_rows(inv) == n);
        REQUIRE(linalg::n_cols(inv) == n);
        REQUIRE(linalg::is_square(inv));

        const linalg::Mat product = mat_pd * inv;
        REQUIRE(linalg::n_rows(product) == n);
        REQUIRE(linalg::n_cols(product) == n);
        REQUIRE(linalg::is_square(product));

        for (int col_index = 0; col_index < n; ++col_index) {
            for (int row_index = 0; row_index < n; ++row_index) {
                double target = 0;
                if (col_index == row_index) {
                    target = 1;
                }

                CHECK_THAT(linalg::elem(product, row_index, col_index), WithinAbs(target, 1000 * margin));
            }
        }
    }

    SECTION("Check symmetry") {
        REQUIRE_FALSE(linalg::is_symmetric(mat1));

        auto id = linalg::id(10);
        REQUIRE((linalg::is_symmetric(id)));

        linalg::set_elem(id, 4, 4, 6912);
        REQUIRE(linalg::is_symmetric(id));

        linalg::set_elem(id, 3, 6, 375.9);
        REQUIRE_FALSE(linalg::is_symmetric(id));

        linalg::set_elem(id, 6, 3, 375.9);
        REQUIRE(linalg::is_symmetric(id));

        REQUIRE_THROWS(linalg::set_elem(id, 4, 12, 32));
        REQUIRE_THROWS(linalg::set_elem(id, 12, 3, 32));
    }
}

TEST_CASE("Test Linear Algebra - Matrix slicing") {
    constexpr size_t n_row = 15;
    constexpr size_t n_col = 20;
    const auto mat = linalg::random(n_row, n_col, seed);

    SECTION("Submat") {
        CHECK_THROWS(linalg::sub_mat(mat, 0, 0, n_row + 1, n_col));
        CHECK_THROWS(linalg::sub_mat(mat, 0, 0, n_row + 1, n_col));


        CHECK_THROWS(linalg::sub_mat(mat, n_row, 0, 1, n_col));
        CHECK_THROWS(linalg::sub_mat(mat, 0, n_col, n_row, 1));

        const auto mat2 = linalg::sub_mat(mat, 0, 0, n_row, n_col);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat, mat2, margin));

        const auto mat3 = linalg::sub_mat(mat, 2, 4, 6, 8);
        REQUIRE(linalg::n_rows(mat3) == 6);
        REQUIRE(linalg::n_cols(mat3) == 8);

        for (int col_index = 0; col_index < 8; ++col_index) {
            for (size_t row_index = 0; row_index < 6; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index + 2, col_index + 4), margin));
            }
        }
    }

    SECTION("Column") {
        CHECK_THROWS(linalg::col(mat, n_col));

        for (size_t col_index = 0; col_index < n_col; ++col_index) {
            const auto col = linalg::col(mat, col_index);
            REQUIRE(linalg::n_elem(col) == n_row);

            for (size_t idx = 0; idx < n_row; ++idx) {
                CHECK_THAT(linalg::elem(col, idx),
                    WithinAbs(linalg::elem(mat, idx, col_index), margin));
            }
        }
    }

    SECTION("Row") {
        CHECK_THROWS(linalg::row(mat, n_row));

        for (size_t row_index = 0; row_index < n_row; ++row_index) {
            const auto row = linalg::row(mat, row_index);
            REQUIRE(linalg::n_elem(row) == n_col);

            for (size_t idx = 0; idx < n_col; ++idx) {
                CHECK_THAT(linalg::elem(row, 0, idx),
                    WithinAbs(linalg::elem(mat, row_index, idx), margin));
            }
        }
    }


    SECTION("Rows") {
        CHECK_THROWS(linalg::rows(mat, 0, n_row + 1));
        CHECK_THROWS(linalg::rows(mat, n_row, 2));

        const auto mat2 = linalg::rows(mat, 0, n_row);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat2, mat, margin));

        const auto mat3 = linalg::rows(mat, 2, 8);
        REQUIRE(linalg::n_rows(mat3) == 8);
        REQUIRE(linalg::n_cols(mat3) == n_col);

        for (int col_index = 0; col_index < n_col; ++col_index) {
            for (size_t row_index = 0; row_index < 8; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index + 2, col_index), margin));
            }
        }
    }

    SECTION("Head Cols") {
        CHECK_THROWS(linalg::head_cols(mat, n_col + 1));

        const auto mat2 = linalg::head_cols(mat, n_col);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat2, mat, margin));

        const auto mat3 = linalg::head_cols(mat, 5);
        REQUIRE(linalg::n_rows(mat3) == n_row);
        REQUIRE(linalg::n_cols(mat3) == 5);

        for (int col_index = 0; col_index < 5; ++col_index) {
            for (size_t row_index = 0; row_index < n_row; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index, col_index), margin));
            }
        }
    }

    SECTION("Tail Cols") {
        CHECK_THROWS(linalg::tail_cols(mat, n_col + 1));

        const auto mat2 = linalg::tail_cols(mat, n_col);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat2, mat, margin));

        const auto mat3 = linalg::tail_cols(mat, 5);
        REQUIRE(linalg::n_rows(mat3) == n_row);
        REQUIRE(linalg::n_cols(mat3) == 5);

        for (int col_index = 0; col_index < 5; ++col_index) {
            for (size_t row_index = 0; row_index < n_row; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index, col_index + n_col - 5), margin));
            }
        }
    }

        SECTION("Head Rows") {
        CHECK_THROWS(linalg::head_rows(mat, n_row + 1));

        const auto mat2 = linalg::head_rows(mat, n_row);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat, mat2, margin));

        const auto mat3 = linalg::head_rows(mat, 5);
        REQUIRE(linalg::n_rows(mat3) == 5);
        REQUIRE(linalg::n_cols(mat3) == n_col);

        for (int col_index = 0; col_index < n_col; ++col_index) {
            for (size_t row_index = 0; row_index < 5; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index, col_index), margin));
            }
        }
    }

    SECTION("Tail Rows") {
        CHECK_THROWS(linalg::tail_rows(mat, n_row + 1));

        const auto mat2 = linalg::tail_rows(mat, n_row);
        REQUIRE(linalg::n_rows(mat2) == n_row);
        REQUIRE(linalg::n_cols(mat2) == n_col);
        CHECK(linalg::nearly_equal(mat, mat2, margin));

        const auto mat3 = linalg::tail_rows(mat, 5);
        REQUIRE(linalg::n_rows(mat3) == 5);
        REQUIRE(linalg::n_cols(mat3) == n_col);

        for (int col_index = 0; col_index < n_col; ++col_index) {
            for (size_t row_index = 0; row_index < 5; ++row_index) {
                CHECK_THAT(linalg::elem(mat3, row_index, col_index),
                    WithinAbs(linalg::elem(mat, row_index + n_row - 5, col_index), margin));
            }
        }
    }

}
