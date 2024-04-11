//
// Created by Zack Williams on 12/02/2024.
//

#include <complex>

#include "../src/utils/parallel.hpp"
#include "catch.hpp"

using namespace uw12::parallel;
using namespace uw12_test;

TEST_CASE("Test Parallel - parallel_for") {
  INFO("Check parallel for returns the same result in parallel and not");

  constexpr size_t n = 10000;

  auto values = std::vector<double>(n);

  const auto func = [&values](const size_t index) {
    values[index] = std::sqrt(index);
  };

  parallel_for(0, n, func, true);

  auto values2 = std::vector<double>(n);

  const auto func2 = [&values2](const size_t index) {
    values2[index] += std::sqrt(index);
  };

  parallel_for(0, n, func2, false);

  for (size_t i = 0; i < n; ++i) {
    CHECK_THAT(values[i], Catch::Matchers::WithinAbs(values2[i], margin));
  }
}

TEST_CASE("Test Parallel - parallel_sum") {
  constexpr size_t start = 0;
  constexpr size_t stop = 1000;

  constexpr double identity = 0;

  const std::function func = [](const size_t val) -> double {
    return std::sqrt(val);
  };

  const auto parallel_result = parallel_sum(start, stop, identity, func, true);
  const auto sequential_result =
      parallel_sum(start, stop, identity, func, false);

  INFO("Parallel sum = " << parallel_result);
  INFO("Sequential sum = " << sequential_result);

  REQUIRE_THAT(
      sequential_result, Catch::Matchers::WithinAbs(parallel_result, margin)
  );
}

TEST_CASE("Test Parallel - parallel_sum_2d") {
  constexpr size_t start1 = 1;
  constexpr size_t stop1 = 200;

  constexpr size_t start2 = 1;
  constexpr size_t stop2 = 100;

  constexpr double identity = 0;

  const std::function func = [](const size_t val, const size_t val2) -> double {
    return std::sqrt(val / val2);
  };

  const auto parallel_result =
      parallel_sum_2d(start1, stop1, start2, stop2, identity, func, true);
  const auto sequential_result =
      parallel_sum_2d(start1, stop1, start2, stop2, identity, func, false);

  INFO("Parallel sum = " << parallel_result);
  INFO("Sequential sum = " << sequential_result);

  REQUIRE_THAT(
      sequential_result, Catch::Matchers::WithinAbs(parallel_result, margin)
  );
}
