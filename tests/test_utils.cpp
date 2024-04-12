//
// Created by Zack Williams on 22/02/2024.
//

#include "catch.hpp"
#include "utils/utils.hpp"

using namespace uw12;
using namespace utils;
using namespace uw12_test;

using Catch::Matchers::WithinAbs;

TEST_CASE("Test utils - Matrix utils") {
  SECTION("Test square matrix") {
    constexpr size_t n = 6;

    constexpr size_t n2 = n * (n + 1) / 2;

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
    for (size_t i = 0; i < n2; ++i) {
      CHECK_THAT(
          linalg::elem(sq, row_idx, col_idx),
          WithinAbs(linalg::elem(vec, i), margin)
      );
      col_idx++;
      if (col_idx > row_idx) {
        row_idx++;
        col_idx = 0;
      }
    }

    CHECK(linalg::nearly_equal(lower(sq), vec, epsilon));
  }

  SECTION("Test lower matrix") {
    constexpr size_t n = 6;

    const auto sq = linalg::random_pd(n, seed);

    CHECK_THROWS(lower(linalg::random(n, n + 1, seed)));
    CHECK_THROWS(lower(linalg::random(n, n, seed)));

    for (const auto factor : {1.0, 2.0, 3.5}) {
      const auto vec = lower(sq, factor);
      constexpr size_t n2 = n * (n + 1) / 2;
      REQUIRE(linalg::n_elem(vec) == n2);

      size_t col_idx = 0;
      size_t row_idx = 0;
      for (size_t i = 0; i < n2; ++i) {
        const auto target =
            linalg::elem(vec, i) / (col_idx == row_idx ? 1.0 : factor);
        CHECK_THAT(
            linalg::elem(sq, row_idx, col_idx), WithinAbs(target, margin)
        );
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

    for (size_t i = 0; i < multiple.size(); ++i) {
      const linalg::Mat mat = factor * mat_vec[i];

      CHECK(linalg::nearly_equal(multiple[i], multiple2[i], epsilon));
      CHECK(linalg::nearly_equal(multiple[i], mat, epsilon));
    }
  }

  SECTION("Addition") {
    const auto sum = mat_vec + mat_vec;
    REQUIRE(mat_vec.size() == sum.size());

    for (size_t i = 0; i < mat_vec.size(); ++i) {
      const linalg::Mat target = mat_vec[i] + mat_vec[i];
      CHECK(linalg::nearly_equal(sum[i], target, epsilon));
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
    for (const auto &mat1 : fock2.fock) {
      const linalg::Mat mat2 = factor * matrix;
      CHECK(linalg::nearly_equal(mat1, mat2, epsilon));
    }
    CHECK_THAT(fock2.energy, WithinAbs(factor * fock_energy.energy, margin));

    const auto fock3 = fock_energy * factor;
    for (const auto &mat1 : fock3.fock) {
      const linalg::Mat mat3 = factor * matrix;
      CHECK(linalg::nearly_equal(mat1, mat3, epsilon));
    }
    CHECK_THAT(fock3.energy, WithinAbs(factor * fock_energy.energy, margin));
  }

  SECTION("Addition") {
    const auto sum = fock_energy + fock_energy;
    REQUIRE(sum.fock.size() == fock_energy.fock.size());

    for (size_t i = 0; i < sum.fock.size(); ++i) {
      const linalg::Mat target = fock_energy.fock[i] + fock_energy.fock[i];
      CHECK(linalg::nearly_equal(sum.fock[i], target, epsilon));
    }
    CHECK_THAT(sum.energy, WithinAbs(fock_energy.energy * 2, margin));

    const FockMatrixAndEnergy one = {{matrix}, -0.356};
    REQUIRE_THROWS(fock_energy + one);

    FockMatrixAndEnergy fock_matrix_and_energy = fock_energy;
    fock_matrix_and_energy += fock_energy;
    REQUIRE(fock_matrix_and_energy.fock.size() == sum.fock.size());

    for (size_t i = 0; i < sum.fock.size(); ++i) {
      CHECK(linalg::nearly_equal(
          sum.fock[i], fock_matrix_and_energy.fock[i], epsilon
      ));
    }
    CHECK_THAT(sum.energy, WithinAbs(fock_matrix_and_energy.energy, margin));
  }

  SECTION("Symmetrise Fock") {
    for (const auto &mat : fock_energy.fock) {
      REQUIRE_FALSE(linalg::is_symmetric(mat));
    }

    const auto fock_matrix_and_energy = symmetrise_fock(fock_energy);
    REQUIRE(fock_matrix_and_energy.fock.size() == fock_energy.fock.size());

    for (const auto &mat : fock_matrix_and_energy.fock) {
      REQUIRE(linalg::is_symmetric(mat));
    }
    REQUIRE_THAT(
        fock_matrix_and_energy.energy, WithinAbs(fock_energy.energy, margin)
    );

    const auto fock_matrix_and_energy2 =
        symmetrise_fock(fock_matrix_and_energy);
    REQUIRE(
        fock_matrix_and_energy.fock.size() ==
        fock_matrix_and_energy2.fock.size()
    );

    for (size_t i = 0; i < fock_matrix_and_energy.fock.size(); ++i) {
      CHECK(linalg::nearly_equal(
          fock_matrix_and_energy.fock[i],
          fock_matrix_and_energy2.fock[i],
          epsilon
      ));
    }
    CHECK_THAT(
        fock_matrix_and_energy.energy, WithinAbs(fock_energy.energy, margin)
    );
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

TEST_CASE("Test utils - Orbitals") {
  constexpr auto nao = 20;
  constexpr auto n_orb = 4;
  const auto mat = linalg::random(nao, n_orb, seed);

  const Orbitals orbitals = {mat, mat};

  const auto n_spin = spin_channels(orbitals);
  REQUIRE(n_spin == 2);
  const std::vector<size_t> n_active = {3, 3};

  SECTION("Freeze core") {
    CHECK_THROWS(freeze_core(orbitals, {5}));
    CHECK_THROWS(freeze_core(orbitals, {5, 5}));
    for (const auto n_a : n_active) {
      REQUIRE(n_orb - n_a >= 0);
    }

    const auto frozen_orbitals = freeze_core(orbitals, n_active);
    REQUIRE(spin_channels(frozen_orbitals) == n_spin);
    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
      const auto &orb = frozen_orbitals[sigma];
      CHECK(linalg::n_rows(orb) == nao);
      CHECK(linalg::n_cols(orb) == n_active[sigma]);
    }
  }

  SECTION("Construct density") {
    const auto D = construct_density(orbitals);

    REQUIRE(spin_channels(D) == n_spin);
    for (const auto &d : D) {
      CHECK(linalg::n_rows(d) == nao);
      CHECK(linalg::n_cols(d) == nao);
    }

    const auto mat0 = linalg::mat(nao, 0);
    const auto D0 = construct_density({mat0});
    for (const auto &d0 : D0) {
      REQUIRE(linalg::n_rows(d0) == nao);
      REQUIRE(linalg::n_cols(d0) == nao);
      CHECK(linalg::nearly_equal(d0, linalg::zeros(nao, nao), epsilon));
    }
  }

  SECTION("Occ weighted orbitals") {
    constexpr auto n_occ = n_orb - 2;
    const OccupationVector occ1 = linalg::ones(n_occ);
    const Occupations occ = {occ1, occ1};

    REQUIRE_THROWS(occupation_weighted_orbitals(orbitals, {occ1}));

    const Occupations neg_occ = {-occ1, -occ1};
    REQUIRE_THROWS(occupation_weighted_orbitals(orbitals, neg_occ));

    const OccupationVector big_occ1 = linalg::ones(n_orb + 1);
    const Occupations big_occ = {big_occ1, big_occ1};
    REQUIRE_THROWS(occupation_weighted_orbitals(orbitals, big_occ));

    const auto Co = occupation_weighted_orbitals(orbitals, occ);
    REQUIRE(spin_channels(Co) == n_spin);

    for (const auto &co : Co) {
      CHECK(linalg::n_rows(co) == nao);
      CHECK(linalg::n_cols(co) == n_occ);
    }
  }
}
