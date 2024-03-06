//
// Created by Zack Williams on 14/02/2024.
//

#include <csignal>

#include "../src/utils/linalg.hpp"
#include "catch.hpp"

using namespace uw12;
using namespace test;

void check_equal(const linalg::Vec &vec, const std::vector<double> &vector) {
  REQUIRE(vector.size() == linalg::n_elem(vec));

  for (size_t i = 0; i < vector.size(); ++i) {
    CHECK_THAT(vec[i], Catch::Matchers::WithinAbs(vector[i], margin));
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
  for (size_t i = 0; i < n_elem; ++i) {
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

TEST_CASE("Test linear algebra - Test Vectors") {
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
    CHECK_THAT(linalg::max_abs(vec), Catch::Matchers::WithinAbs(5, margin));

    const auto vec2 = linalg::vec({0, 0, 0, 0});
    CHECK_THAT(linalg::max_abs(vec2), Catch::Matchers::WithinAbs(0, margin));

    const auto vec3 = linalg::vec({0, 1, -1, 2, -3});
    CHECK_THAT(linalg::max_abs(vec3), Catch::Matchers::WithinAbs(3, margin));
  }
}

TEST_CASE("Test linear algebra - Test vector slicing") {
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

  SECTION("Assign rows") {
    auto vec2 = vec;
    const auto n = linalg::n_elem(vec);
    REQUIRE_THROWS(linalg::assign_rows(vec2, {}, n));
    REQUIRE_THROWS(linalg::assign_rows(vec2, vec, 1));

    linalg::assign_rows(vec2, vec.tail(2), 1);
    for (size_t idx = 0; idx < n; ++idx) {
      const auto elem = linalg::elem(vec2, idx);
      auto target = linalg::elem(vec, idx);
      if (idx >= 1 && idx <= 2) {
        target = linalg::elem(vec, idx + n - 3);
      }
      CHECK_THAT(elem, Catch::Matchers::WithinAbs(target, margin));
    }
  }
}
