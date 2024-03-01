//
// Created by Zack Williams on 29/02/2024.
//

#include "catch.hpp"

#include "../src/two_electron/two_electron.hpp"

#include "../src/integrals/integrals.hpp"
#include "../src/utils/linalg.hpp"

using namespace uw12;

constexpr auto seed = 2;
constexpr auto margin = 1e-10;
constexpr auto epsilon = 1e-10;

TEST_CASE("Test Two Electron term - Closed Shell") {
    constexpr size_t n_ao = 7;
    constexpr size_t n_occ = 5;

    const std::vector<size_t> df_sizes = {1, 3, 1, 3, 1, 3, 5};

    size_t n_df = 0;
    for (const auto size: df_sizes) {
        n_df += size;
    }

    auto J20 = linalg::random_pd(n_df, seed);
    auto J30 = linalg::random(n_ao * (n_ao + 1) / 2, n_df, seed);

    const integrals::TwoIndexFn two_index_fn = [J20]()-> linalg::Mat {
        return J20;
    };

    const integrals::ThreeIndexFn three_index_fn = [&df_sizes, J30, n_ao](const size_t A) -> linalg::Mat {
        const auto n_row = n_ao * (n_ao + 1) / 2;
        const auto n_col = df_sizes[A];

        size_t offset = 0;
        for (size_t i = 0; i < A; ++i) {
            offset += df_sizes[i];
        }

        return linalg::sub_mat(J30, 0, offset, n_row, n_col);
    };


    const auto base_integrals = integrals::BaseIntegrals(
        two_index_fn, three_index_fn, df_sizes, n_ao, n_df, true);

    const auto C = linalg::random(n_ao, n_occ, seed);

    const auto fock0 = linalg::zeros(n_ao, n_ao);

    const auto &[fock, energy] = two_el::form_fock_two_el_df(
        base_integrals, {C}, false, true, 1.0, 0);

    REQUIRE(fock.size() == 1);

    SECTION("Direct fock") {
        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C}, false, false, 1.0, 0);
            REQUIRE(fock.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock0, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }
        INFO("Test scale = 0 results in zero energy"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C}, false, false, 0, 0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock0, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Test J3_0 implementation"); {
            const auto base2 = integrals::BaseIntegrals(J30, J20);

            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base2, {C}, false, true, 1.0, 0.0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }

        INFO("Test multiplicity of scale factor"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C}, false, true, 1.5, 0.0);
            REQUIRE(fock2.size() == 1);

            const linalg::Mat mat2 = fock[0] * 1.5;
            CHECK(linalg::nearly_equal(fock2[0], mat2, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(1.5 * energy, margin));
        }

        INFO("Test symmetry of spin factors"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C}, false, true, 0.0, 1.0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }

        INFO("Test that results are the same when indirect_term = true for ss_scale = 0"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C}, true, true, 1.0, 0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }

        INFO("Check result are the same for open and closed shell with same orbitals"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, {C, C}, true, true, 1.0, 0);

            REQUIRE(fock2.size() == 2);
            for (size_t sigma = 0; sigma < 2; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[0], epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }
    }


    SECTION("Indirect fock") {
        const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
            base_integrals, {C}, true, true, 0, 1.0);
        CHECK(fock2.size() == 1);
        CHECK_FALSE(linalg::nearly_equal(fock2[0], fock[0], epsilon));

        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, {C}, true, false, 0, 1.0);
            REQUIRE(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock3[0], fock0, epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(energy2, margin));
        }

        INFO("Check zero"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, {C}, true, true, 0, 0.0);
            CHECK(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock3[0], fock0, epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Check same spin multiplicity"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, {C}, true, true, 0, 1.5);
            CHECK(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0] * 1.5, fock3[0], epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(energy2 * 1.5, margin));
        }

        INFO("Check combined calculations give same results"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, {C}, true, true, 1.0, 1.0);
            CHECK(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock[0] + fock2[0], fock3[0], epsilon));
            CHECK_THAT(energy + energy2, Catch::Matchers::WithinAbs(energy3, margin));
        }

        INFO("Check result are the same for open and closed shell with same orbitals"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, {C, C}, true, true, 0, 1.0);

            REQUIRE(fock3.size() == 2);
            for (size_t sigma = 0; sigma < 2; ++sigma) {
                CHECK(linalg::nearly_equal(fock3[sigma], fock2[0], epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(energy2, margin));
        }
    }
}

TEST_CASE("Test Two Electron term - Open Shell") {
    constexpr size_t n_ao = 7;
    constexpr size_t n_occ_a = 5;
    constexpr size_t n_occ_b = 4;

    const std::vector<size_t> df_sizes = {1, 3, 1, 3, 1, 3, 5};

    size_t n_df = 0;
    for (const auto size: df_sizes) {
        n_df += size;
    }

    auto J20 = linalg::random_pd(n_df, seed);
    auto J30 = linalg::random(n_ao * (n_ao + 1) / 2, n_df, seed);

    const integrals::TwoIndexFn two_index_fn = [J20]()-> linalg::Mat {
        return J20;
    };

    const integrals::ThreeIndexFn three_index_fn = [&df_sizes, J30, n_ao](const size_t A) -> linalg::Mat {
        const auto n_row = n_ao * (n_ao + 1) / 2;
        const auto n_col = df_sizes[A];

        size_t offset = 0;
        for (size_t i = 0; i < A; ++i) {
            offset += df_sizes[i];
        }

        return linalg::sub_mat(J30, 0, offset, n_row, n_col);
    };


    const auto base_integrals = integrals::BaseIntegrals(
        two_index_fn, three_index_fn, df_sizes, n_ao, n_df, true);

    const auto Ca = linalg::random(n_ao, n_occ_a, seed);
    const auto Cb = linalg::random(n_ao, n_occ_b, seed);

    const utils::Orbitals orbitals = {Ca, Cb};
    const auto n_spin = orbitals.size();

    const auto fock0 = linalg::zeros(n_ao, n_ao);

    const auto &[fock, energy] = two_el::form_fock_two_el_df(
        base_integrals, orbitals, false, true, 1.0, 0);

    REQUIRE(fock.size() == n_spin);

    SECTION("Direct fock") {
        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, false, 1.0, 0);
            REQUIRE(fock.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock0, epsilon));
            }

            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }
        INFO("Test scale = 0 results in zero energy"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, false, 0, 0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock0, epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Test J3_0 implementation"); {
            const auto base2 = integrals::BaseIntegrals(J30, J20);

            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base2, orbitals, false, true, 1.0, 0.0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }

        INFO("Test multiplicity of scale factor"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, true, 1.5, 0.0);
            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[sigma] * 1.5, epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(1.5 * energy, margin));
        }

        INFO("Test assymmetry of spin factors"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, true, 0.0, 1.0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK_FALSE(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
            }
            CHECK(std::abs(energy2 - energy) > margin);
        }

        INFO("Test that results are the same when indirect_term = true for ss_scale = 0"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 1.0, 0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(energy, margin));
        }
    }


    SECTION("Indirect fock") {
        const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
            base_integrals, orbitals, true, true, 0, 1.0);
        CHECK(fock2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK_FALSE(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
        }

        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, false, 0, 1.0);
            REQUIRE(fock3.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock3[sigma], fock0, epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(energy2, margin));
        }

        INFO("Check zero"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 0, 0.0);
            REQUIRE(fock3.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock3[sigma], fock0, epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Check same spin multiplicity"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 0, 1.5);
            REQUIRE(fock3.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock3[sigma], 1.5 * fock2[sigma], epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(energy2 * 1.5, margin));
        }

        INFO("Check combined calculations give same results"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 1.0, 1.0);
            REQUIRE(fock3.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(
                    fock3[sigma], fock[sigma] + fock2[sigma], epsilon));
            }
            CHECK_THAT(energy + energy2, Catch::Matchers::WithinAbs(energy3, margin));
        }

        INFO("Check empty spin channel"); {
            const auto C1 = linalg::random(n_ao, 1, seed);
            const auto C0 = linalg::random(n_ao, 0, seed);

            const utils::Orbitals empty_orb = {C1, C0};

            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, empty_orb, true, true, 1.0,
                1.0);
            REQUIRE(fock3.size() == n_spin);
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(0, margin));
            // One electron -- no SIE

            INFO("Test opposite spin only has no energy contribution"); {
                const auto &[fock4, energy4] = two_el::form_fock_two_el_df(
                    base_integrals, empty_orb, true, true, 1.0,
                    0.0);
                REQUIRE(fock4.size() == n_spin);
                CHECK(linalg::nearly_equal(fock4[0], fock0, epsilon));
                // Beta spin fock non-zero due to effect of alpha spin electron
                CHECK_FALSE(linalg::nearly_equal(fock4[1], fock0, epsilon));
                // But energy is zero due to no opposite spin electron pairs to interact
                CHECK_THAT(energy4, Catch::Matchers::WithinAbs(0, margin));
            }

            INFO("Test same spin contribution is equal to fock3"); {
                const auto &[fock4, energy4] = two_el::form_fock_two_el_df(
                    base_integrals, empty_orb, true, true, 0,
                    1.0);
                REQUIRE(fock4.size() == n_spin);
                CHECK(linalg::nearly_equal(fock4[0], fock3[0], epsilon));
                CHECK(linalg::nearly_equal(fock4[1], fock0, epsilon));
                CHECK_THAT(energy4, Catch::Matchers::WithinAbs(0, margin));
            }
        }
    }
}
