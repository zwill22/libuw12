//
// Created by Zack Williams on 04/03/2024.
//

#include "catch.hpp"
#include "check_fock.hpp"
#include "../src/utils/utils.hpp"

using namespace fock;
using namespace test;

TEST_CASE("Orbitals from density") {
    constexpr size_t n_ao = 10;
    std::vector<size_t> n_occ = {4};

    for (size_t i = 0; i < 2; ++i) {
        const auto n_spin = n_occ.size();
        const auto D = random_density_matrix(n_occ, n_ao, seed);
        REQUIRE(D.size() == n_spin);

        for (const auto &Do: D) {
            REQUIRE(linalg::n_rows(Do) == n_ao);
            REQUIRE(linalg::n_cols(Do) == n_ao);
        }

        SECTION("Orbitals from density") {
            std::vector<linalg::Mat> orbs = {};
            std::vector<linalg::Vec> occs = {};
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const auto [C, occ] = orbitals_from_density(D[sigma], 1e-6);
                REQUIRE(linalg::n_rows(C) == n_ao);
                REQUIRE(linalg::n_cols(C) == n_occ[sigma]);
                REQUIRE(linalg::n_elem(occ) == n_occ[sigma]);
                orbs.push_back(C);
                occs.push_back(occ);
            }

            const auto orbitals = calculate_orbitals_from_density(D, 1e-6);
            REQUIRE(orbitals.size() == n_spin);

            const auto orbitals2 = utils::occupation_weighted_orbitals(orbs, occs);
            REQUIRE(orbitals2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(orbitals[sigma], orbitals2[sigma], epsilon));
            }
        }

        SECTION("Test invertability") {
            const auto orbitals = calculate_orbitals_from_density(D, 1e-6);
            REQUIRE(orbitals.size() == n_spin);

            const auto D2 = utils::construct_density(orbitals);
            REQUIRE(D2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(D[sigma], D2[sigma], epsilon));
            }
        }

        n_occ.push_back(3);
    }


    SECTION("Zero density") {
        const auto D = linalg::zeros(n_ao, n_ao);

        const auto [C, occ] = orbitals_from_density(D, 1e-6);
        REQUIRE(linalg::n_rows(C) == n_ao);
        REQUIRE(linalg::n_cols(C) == 1);
        REQUIRE(linalg::n_elem(occ) == 1);

        CHECK(linalg::nearly_equal(C, linalg::zeros(n_ao, 1), epsilon));
        CHECK(linalg::nearly_equal(occ, linalg::ones(1), epsilon));
    }
}
