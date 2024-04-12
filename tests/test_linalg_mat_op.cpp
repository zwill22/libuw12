//
// Created by Zack Williams on 18/02/2024.
//

#include "catch.hpp"
#include "utils/linalg.hpp"

using namespace uw12;
using namespace uw12_test;

using Catch::Matchers::WithinAbs;

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
        total += elem * elem;
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

  SECTION("Check empty") {
    const auto mat0 = linalg::ones(0, 0);
    CHECK(linalg::empty(mat0));

    CHECK_FALSE(linalg::empty(mat));
  }
}

TEST_CASE("Test linear algebra - Test matrix manipulations") {
  constexpr size_t n_row = 14;
  constexpr size_t n_col = 16;

  const auto mat = linalg::random(n_row, n_col, seed);
  REQUIRE(linalg::n_rows(mat) == n_row);
  REQUIRE(linalg::n_cols(mat) == n_col);

  SECTION("Test vectorise") {
    const auto vec = linalg::vectorise(mat);

    const auto n = linalg::n_elem(mat);
    REQUIRE(linalg::n_elem(vec) == n);
    REQUIRE(n == n_row * n_col);

    for (size_t i = 0; i < n; ++i) {
      const size_t row_idx = i % n_row;
      const size_t col_idx = i / n_row;

      CHECK_THAT(
          linalg::elem(mat, row_idx, col_idx),
          WithinAbs(linalg::elem(vec, i), margin)
      );
    }
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

        CHECK_THAT(
            linalg::elem(diagmat, row_idx, col_idx), WithinAbs(target, margin)
        );
      }
    }
  }

  SECTION("Test sqrt") {
    auto mat2 = linalg::random_pd(n_col, seed);
    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_col; ++row_idx) {
        if (const auto elem = linalg::elem(mat2, row_idx, col_idx); elem < 0) {
          linalg::set_elem(mat2, row_idx, col_idx, -1 * elem);
        }
      }
    }

    for (size_t i = 0; i < n_col; ++i) {
      const auto col = linalg::col(mat2, i);
      REQUIRE(linalg::all_positive(col));
    }

    const linalg::Mat neg = -1 * mat2;

    REQUIRE_THROWS(linalg::sqrt(neg));

    const auto sqrt = linalg::sqrt(mat2);

    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_col; ++row_idx) {
        double target = std::sqrt(linalg::elem(mat2, row_idx, col_idx));

        CHECK_THAT(
            linalg::elem(sqrt, row_idx, col_idx), WithinAbs(target, margin)
        );
      }
    }
  }

  SECTION("Test each col") {
    const linalg::Vec vec = linalg::random(n_row, 1, seed);

    REQUIRE_THROWS(linalg::each_col(mat, linalg::ones(n_row + 1)));
    REQUIRE_THROWS(linalg::each_col(mat, linalg::ones(n_row - 1)));

    const auto mat2 = linalg::each_col(mat, vec);
    REQUIRE(linalg::n_elem(vec) == n_row);
    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
        const auto target =
            linalg::elem(mat, row_idx, col_idx) * linalg::elem(vec, row_idx);
        CHECK_THAT(
            linalg::elem(mat2, row_idx, col_idx), WithinAbs(target, margin)
        );
      }
    }
  }

  SECTION("Test join matrices") {
    REQUIRE_THROWS(linalg::join_matrices(mat, linalg::head_rows(mat, n_row - 1))
    );

    const auto joined = linalg::join_matrices(mat, mat);
    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
        const auto target = linalg::elem(mat, row_idx, col_idx % n_col);
        CHECK_THAT(
            linalg::elem(joined, row_idx, col_idx), WithinAbs(target, margin)
        );
      }
    }
  }

  SECTION("Test assign cols") {
    auto mat2 = mat;
    REQUIRE_THROWS(linalg::assign_cols(mat2, mat, 1));
    REQUIRE_THROWS(linalg::assign_cols(mat2, linalg::col(mat, 0), n_col + 1));
    REQUIRE_THROWS(
        linalg::assign_cols(mat2, linalg::head_rows(mat, n_row - 1), 0)
    );

    linalg::assign_cols(mat2, linalg::tail_cols(mat, 3), 1);
    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      const auto col1 = linalg::col(mat2, col_idx);
      auto col2 = linalg::col(mat, col_idx);
      if (0 < col_idx && col_idx < 4) {
        col2 = linalg::col(mat, col_idx + n_col - 4);
      }
      CHECK(linalg::nearly_equal(col1, col2, margin));
    }
  }

  SECTION("Test assign rows") {
    auto mat2 = mat;
    REQUIRE_THROWS(linalg::assign_rows(mat2, mat, 1));
    REQUIRE_THROWS(linalg::assign_rows(mat2, linalg::row(mat, 0), n_row + 1));
    REQUIRE_THROWS(
        linalg::assign_rows(mat2, linalg::head_cols(mat, n_col - 1), 0)
    );

    linalg::assign_rows(mat2, linalg::tail_rows(mat, 3), 2);
    for (size_t col_idx = 0; col_idx < n_col; ++col_idx) {
      for (size_t row_idx = 0; row_idx < n_row; ++row_idx) {
        const auto elem = linalg::elem(mat2, row_idx, col_idx);
        auto target = linalg::elem(mat, row_idx, col_idx);
        if (row_idx >= 2 && row_idx <= 4) {
          target = linalg::elem(mat, row_idx + n_row - 5, col_idx);
        }
        CHECK_THAT(elem, WithinAbs(target, margin));
      }
    }
  }
}

TEST_CASE("Test linear algebra - Eigendecompositions") {
  constexpr size_t n_row = 10;

  REQUIRE_THROWS(linalg::eigen_system(linalg::random(n_row, n_row, seed)));
  REQUIRE_THROWS(linalg::eigen_system(linalg::random(n_row, 4, seed)));

  const auto mat = linalg::random_pd(n_row, seed);

  const auto &[vals, vecs] = linalg::eigen_system(mat);

  REQUIRE(linalg::n_rows(vals) == n_row);
  REQUIRE(linalg::n_rows(vecs) == n_row);
  REQUIRE(linalg::n_cols(vecs) == n_row);

  const auto &[vals2, vecs2] = linalg::eigen_decomposition(mat, 0, 0);

  CHECK(linalg::nearly_equal(vals, vals2, margin));
  CHECK(linalg::nearly_equal(vecs, vecs2, margin));

  const auto &[vals3, vecs3] = linalg::eigen_decomposition(mat, 1e-6, 1e-8);

  REQUIRE(linalg::n_elem(vals3) == linalg::n_cols(vecs3));
  REQUIRE(linalg::n_elem(vals3) <= linalg::n_elem(vals));
}

// TODO Memory test
// TODO CSV test
