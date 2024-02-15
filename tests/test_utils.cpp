//
// Created by Zack Williams on 15/02/2024.
//

#include "../src/utils/utils.hpp"

#include "catch.hpp"

using namespace uw12;

TEST_CASE("Test random positive definite matrices") {
    constexpr size_t n = 12;
    constexpr int seed = 22;

    // Random positive definite symmetric matrix
    const auto mat_pd = random_pd(n, seed);

    REQUIRE(linalg::n_elem(mat_pd) == n * n);
    REQUIRE(linalg::n_rows(mat_pd) == n);
    REQUIRE(linalg::n_cols(mat_pd) == n);
    REQUIRE(linalg::is_square(mat_pd));
    CHECK(linalg::is_symmetric(mat_pd));
}
