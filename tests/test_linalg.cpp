//
// Created by Zack Williams on 14/02/2024.
//

#include <csignal>

#include "../src/utils/linalg.hpp"

#include "catch.hpp"

using namespace uw12;

void check_equal(const linalg::Vec &vec, const std::vector<double> &vector) {
    REQUIRE(vector.size() == linalg::n_elem(vec));

    for (size_t i = 0; i < vector.size(); ++i) {
        CHECK(vec[i] == Catch::Approx(vector[i]));
    }
}

std::vector<double> slice(
    const std::vector<double> &vector,
    const size_t start_index,
    const size_t n_elem
) {
    if (n_elem > vector.size()) {
        throw std::logic_error("Subvector must be subset of parent vector");
    }
    if (start_index >= vector.size()) {
        throw std::logic_error("Starting index outside of parent vector range");
    }
    if (start_index + n_elem > vector.size()) {
        throw std::logic_error("Final index outside of parent vector range");
    }

    std::vector<double> new_vector({});
    for (int i = 0; i < n_elem; ++i) {
        const auto index = start_index + i;
        if (index >= vector.size()) {
            throw std::logic_error("Index outside of parent vector range");
        }
        new_vector.push_back(vector[start_index + i]);
    }

    if (new_vector.size() > vector.size()) {
        throw std::logic_error("Subvector cannot be larger than parent vector");
    }

    return new_vector;
}

TEST_CASE("Test linear algebra library - Test Vectors") {
    SECTION("Test vector construction") {
        const std::vector vector = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        const auto n_elem = vector.size();

        const auto vec = linalg::vec(vector);

        check_equal(vec, vector);

        const auto vec2 = [n_elem, &vector] {
            auto new_vec = linalg::vec(n_elem);
            for (size_t i = 0; i < n_elem; ++i) {
                new_vec[i] = vector[i];
            }
            return new_vec;
        }();

        check_equal(vec2, vector);
    }

    SECTION("Test vector of ones") {
        constexpr auto n_elem = 10;

        const std::vector<double> vector(n_elem, 1);
        const auto ones = linalg::ones(n_elem);

        check_equal(ones, vector);

        const auto vec = linalg::vec(vector);
        check_equal(vec, vector);
    }

    SECTION("Test vector of zeros") {
        constexpr auto n_elem = 20;

        const std::vector<double> vector(n_elem, 0);
        const auto zeros = linalg::zeros(n_elem);

        check_equal(zeros, vector);

        const auto vec = linalg::vec(vector);
        check_equal(vec, vector);
    }

    SECTION("Check positivity") {
        const auto positive = linalg::vec({1, 2, 3, 4, 5});

        CHECK(linalg::all_positive(positive));

        const linalg::Vec negative = -1 * positive;

        CHECK_FALSE(linalg::all_positive(negative));

        const auto one = linalg::vec({1, 2, 3, 4, -5, 6});
        CHECK_FALSE(linalg::all_positive(one));

        const auto zero = linalg::zeros(10);
        CHECK_FALSE(linalg::all_positive(zero));
    }

    SECTION("Test Schur product") {
        const auto vec = linalg::vec({1, 2, 3, 4, 5});
        const auto vec2 = linalg::vec({6, 7, 8, 9, 10});

        const auto product = linalg::schur(vec, vec2);

        const std::vector<double> manual = [&vec, &vec2] {
            const auto n_elem = linalg::n_elem(vec);

            std::vector<double> new_vec(n_elem);
            for (size_t i = 0; i < n_elem; ++i) {
                new_vec[i] = linalg::elem(vec, i) * linalg::elem(vec2, i);
            }
            return new_vec;
        }();

        check_equal(product, manual);

        const auto vec3 = linalg::ones(4);
        CHECK_THROWS(linalg::schur(vec, vec3));
    }

    SECTION("Test memory") {
        const auto vec = linalg::vec({1, 2, 4, 5, 6});

        const auto ptr = linalg::mem_ptr(vec);
        const auto n_elem = linalg::n_elem(vec);

        const std::vector vector(ptr, ptr + n_elem);

        check_equal(vec, vector);
    }

    SECTION("Check empty") {
        const std::vector<double> vector(0);

        const auto vec = linalg::vec(vector);

        CHECK(linalg::empty(vec));

        const std::vector<double> vector2(1);
        const auto vec2 = linalg::vec(vector2);

        CHECK_FALSE(linalg::empty(vec2));
    }

    SECTION("Check max absolute value") {
        const auto vec = linalg::vec({1, 2, 3, 4, 5});
        CHECK(linalg::max_abs(vec) == Catch::Approx(5));

        const auto vec2 = linalg::vec({0, 0, 0, 0});
        CHECK(linalg::max_abs(vec2) == Catch::Approx(0));

        const auto vec3 = linalg::vec({0, 1, -1, 2, -3});
        CHECK(linalg::max_abs(vec3) == Catch::Approx(3));
    }
}

TEST_CASE("Test linear algebra library - Test vector slicing") {
    const std::vector<double> vector = {1, 2, 3, 4, 5, 6};
    const auto vec = linalg::vec(vector);

    SECTION("Check subvec") {
        const auto subvec = linalg::sub_vec(vec, 0, 3);
        const auto subvector = slice(vector, 0, 3);
        check_equal(subvec, subvector);

        const auto subvec2 = linalg::sub_vec(vec, 3, 3);
        const auto subvector2 = slice(vector, 3, 3);
        check_equal(subvec2, subvector2);

        CHECK_THROWS(linalg::sub_vec(vec, 0, 7));
        CHECK_THROWS(slice(vector, 0, 7));

        CHECK_THROWS(linalg::sub_vec(vec, 4, 3));
        CHECK_THROWS(slice(vector, 4, 3));

        CHECK_THROWS(linalg::sub_vec(vec, 6, 1));
        CHECK_THROWS(slice(vector, 6, 1));
    }

    SECTION("Check head") {
        const auto head = linalg::head(vec, 3);
        const auto subvector = slice(vector, 0, 3);
        check_equal(head, subvector);

        const auto head2 = linalg::head(vec, 6);
        check_equal(head2, vector);

        CHECK_THROWS(linalg::head(vec, 7));
    }

    SECTION("Check tail") {
        constexpr size_t n_rows = 3;

        const auto tail = linalg::tail(vec, n_rows);
        const auto subvector = slice(vector, vector.size() - n_rows, n_rows);
        check_equal(tail, subvector);

        const auto tail2 = linalg::tail(vec, 6);
        check_equal(tail2, vector);

        CHECK_THROWS(linalg::tail(vec, 7));
    }
}

TEST_CASE("Test linear algebra library - Test Matrix initialisation") {
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

            CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(vector[i]));
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

            CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(1));
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

            CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(0));
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

            if (row_index == col_index) {
                CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(1));
            } else {
                CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(0));
            }
        }
    }


    SECTION("Test random matrices") {
        constexpr int seed = 22;

        constexpr size_t n_row = 11;
        constexpr size_t n_col = 7;

        const auto mat = linalg::random(n_row, n_col, seed);
        REQUIRE(linalg::n_elem(mat) == n_row * n_col);
        REQUIRE(linalg::n_rows(mat) == n_row);
        REQUIRE(linalg::n_cols(mat) == n_col);

        const auto mat2 = linalg::random(n_row, n_col, seed);
        for (int col_index = 0; col_index < n_col; ++col_index) {
            for (int row_index = 0; row_index < n_row; ++row_index) {
                CHECK(linalg::elem(mat, row_index, col_index) == Catch::Approx(linalg::elem(mat2, row_index, col_index)
                ));
            }
        }
    }
}
