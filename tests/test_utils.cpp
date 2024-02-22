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

        constexpr int n2 = n * (n + 1) / 2;

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
            constexpr int n2 = n * (n + 1) / 2;
            REQUIRE(linalg::n_elem(vec) == n2);

            size_t col_idx = 0;
            size_t row_idx = 0;
            for (int i = 0; i < n2; ++i) {
                const auto target = linalg::elem(vec, i) / (col_idx == row_idx ? 1.0 : factor);
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

    const MatVec mat_vec = {mat1, mat2};

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

    SECTION("Addition") {
        const auto sum = mat_vec + mat_vec;
        REQUIRE(mat_vec.size() == sum.size());

         for (int i = 0; i < mat_vec.size(); ++i) {
             const linalg::Mat target = mat_vec[i] + mat_vec[i];
             CHECK(linalg::nearly_equal(sum[i], target, margin));
         }

         const auto one = MatVec({mat1});
         REQUIRE_THROWS(mat_vec + one);

         const auto reverse = MatVec({mat2, mat1});
         REQUIRE_THROWS(mat_vec + reverse);
    }
}

TEST_CASE("Test utils - Fock Matrix and Energy") {
    constexpr auto n = 12;

    const auto matrix = linalg::random(n, n, seed);

    const FockMatrix fock = {matrix, matrix};
    constexpr auto energy = -0.52;

    const FockMatrixAndEnergy fock_energy = {fock, energy};

    SECTION("Multiplication") {
        constexpr auto factor = 32.6;

        const auto fock2 = factor * fock_energy;
        for (const auto &mat1: fock2.fock) {
            const linalg::Mat mat2 = factor * matrix;
            CHECK(linalg::nearly_equal(mat1, mat2, margin));
        }
        CHECK_THAT(fock2.energy, WithinAbs(factor * fock_energy.energy, margin));

        const auto fock3 = fock_energy * factor;
        for (const auto &mat1: fock3.fock) {
            const linalg::Mat mat3 = factor * matrix;
            CHECK(linalg::nearly_equal(mat1, mat3, margin));
        }
        CHECK_THAT(fock3.energy, WithinAbs(factor * fock_energy.energy, margin));

    }

    SECTION("Addition") {
        const auto sum = fock_energy + fock_energy;
        REQUIRE(sum.fock.size() == fock_energy.fock.size());

        for (int i = 0; i < sum.fock.size(); ++i) {
            const linalg::Mat target = fock_energy.fock[i] + fock_energy.fock[i];
            CHECK(linalg::nearly_equal(sum.fock[i], target, margin));
        }
        CHECK_THAT(sum.energy, WithinAbs(fock_energy.energy * 2, margin));

        const FockMatrixAndEnergy one = {{matrix}, -0.356};
        REQUIRE_THROWS(fock_energy + one);

        FockMatrixAndEnergy fock_matrix_and_energy = fock_energy;
        fock_matrix_and_energy += fock_energy;
        REQUIRE(fock_matrix_and_energy.fock.size() == sum.fock.size());

        for (int i = 0; i < sum.fock.size(); ++i) {
            CHECK(linalg::nearly_equal(sum.fock[i], fock_matrix_and_energy.fock[i], margin));
        }
        CHECK_THAT(sum.energy, WithinAbs(fock_matrix_and_energy.energy, margin));
    }

    SECTION("Symmetrise Fock") {
        for (const auto &mat: fock_energy.fock) {
            REQUIRE_FALSE(linalg::is_symmetric(mat));
        }

        const auto fock_matrix_and_energy = symmetrise_fock(fock_energy);
        REQUIRE(fock_matrix_and_energy.fock.size() == fock_energy.fock.size());

        for (const auto &mat: fock_matrix_and_energy.fock) {
            REQUIRE(linalg::is_symmetric(mat));
        }
        REQUIRE_THAT(fock_matrix_and_energy.energy, WithinAbs(fock_energy.energy, margin));

        const auto fock_matrix_and_energy2 = symmetrise_fock(fock_matrix_and_energy);
        REQUIRE(fock_matrix_and_energy.fock.size() == fock_matrix_and_energy2.fock.size());

        for (int i = 0; i < fock_matrix_and_energy.fock.size(); ++i) {
            CHECK(linalg::nearly_equal(fock_matrix_and_energy.fock[i], fock_matrix_and_energy2.fock[i], margin));
        }
        CHECK_THAT(fock_matrix_and_energy.energy, WithinAbs(fock_energy.energy, margin));

    }
}

TEST_CASE("Test utils - Template functions") {
    SECTION("Nearly zero") {
        CHECK(nearly_zero(0));
        CHECK(nearly_zero(1e-16));
        CHECK(nearly_zero(-1e-16));
        CHECK_FALSE(nearly_zero(1));
        CHECK_FALSE(nearly_zero(-1));
    }

    SECTION("Spin channels") {
        const std::vector<double> vector0 = {};
        const std::vector vector1 = {0.1};
        const std::vector vector2 = {0.3, 0.4};
        const std::vector vector3 = {0.5, 0.6, 0.7};

        CHECK_THROWS(spin_channels(vector0));
        CHECK(spin_channels(vector1) == 1);
        CHECK(spin_channels(vector2) == 2);
        CHECK_THROWS(spin_channels(vector3));
    }
}
