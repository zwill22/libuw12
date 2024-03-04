//
// Created by Zack Williams on 29/02/2024.
//

#include "catch.hpp"

#include "check_fock.hpp"

#include "../src/two_electron/two_electron.hpp"

#include "../src/integrals/integrals.hpp"
#include "../src/utils/linalg.hpp"

using namespace uw12;
using namespace test;

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
        constexpr auto n_row = n_ao * (n_ao + 1) / 2;
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
    const utils::Orbitals active_Co = {C};

    const auto fock0 = linalg::zeros(n_ao, n_ao);

    const auto &[fock, energy] = two_el::form_fock_two_el_df(
        base_integrals, {C}, false, true, 1.0, 0);

    REQUIRE(fock.size() == 1);

    SECTION("Direct fock") {
        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, false, false, 1.0, 0);
            REQUIRE(fock.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock0, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }
        INFO("Test scale = 0 results in zero energy"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, false, false, 0, 0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock0, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Test J3_0 implementation"); {
            const auto base2 = integrals::BaseIntegrals(J30, J20);

            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base2, active_Co, false, true, 1.0, 0.0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }

        INFO("Test multiplicity of scale factor"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, false, true, 1.5, 0.0);
            REQUIRE(fock2.size() == 1);

            const linalg::Mat mat2 = fock[0] * 1.5;
            CHECK(linalg::nearly_equal(fock2[0], mat2, epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(1.5 * energy, eps));
        }

        INFO("Test symmetry of spin factors"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, false, true, 0.0, 1.0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }

        INFO("Test that results are the same when indirect_term = true for ss_scale = 0"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, true, true, 1.0, 0);

            REQUIRE(fock2.size() == 1);
            CHECK(linalg::nearly_equal(fock2[0], fock[0], epsilon));
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }

        INFO("Check result are the same for open and closed shell with same orbitals"); {
            const utils::Orbitals Co = {C, C};
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, Co, true, true, 1.0, 0);

            REQUIRE(fock2.size() == 2);
            for (size_t sigma = 0; sigma < 2; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[0], epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }
    }


    SECTION("Indirect fock") {
        const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
            base_integrals, active_Co, true, true, 0, 1.0);
        CHECK(fock2.size() == 1);
        CHECK_FALSE(linalg::nearly_equal(fock2[0], fock[0], epsilon));

        INFO("Check energy is the same whether or not the Fock matrix is calculated"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, true, false, 0, 1.0);
            REQUIRE(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock3[0], fock0, epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinRel(energy2, eps));
        }

        INFO("Check zero"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, true, true, 0, 0.0);
            CHECK(fock3.size() == 1);
            CHECK(linalg::nearly_equal(fock3[0], fock0, epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(0, margin));
        }

        INFO("Check same spin multiplicity"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, true, true, 0, 1.5);
            CHECK(fock3.size() == 1);

            const linalg::Mat mat2 = 1.5 * fock2[0];
            CHECK(linalg::nearly_equal(fock3[0], mat2, epsilon));
            CHECK_THAT(energy3, Catch::Matchers::WithinRel(energy2 * 1.5, eps));
        }

        INFO("Check combined calculations give same results"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, active_Co, true, true, 1.0, 1.0);
            CHECK(fock3.size() == 1);

            const linalg::Mat total_fock = fock[0] + fock2[0];
            CHECK(linalg::nearly_equal(total_fock, fock3[0], epsilon));
            CHECK_THAT(energy + energy2, Catch::Matchers::WithinRel(energy3, eps));
        }

        INFO("Check result are the same for open and closed shell with same orbitals"); {
            const utils::Orbitals Co = {C, C};
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, Co, true, true, 0, 1.0);

            REQUIRE(fock3.size() == 2);
            for (size_t sigma = 0; sigma < 2; ++sigma) {
                CHECK(linalg::nearly_equal(fock3[sigma], fock2[0], epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinRel(energy2, eps));
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
        constexpr auto n_row = n_ao * (n_ao + 1) / 2;
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

            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
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
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
        }

        INFO("Test multiplicity of scale factor"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, true, 1.5, 0.0);
            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const linalg::Mat mat2 = 1.5 * fock[sigma];
                CHECK(linalg::nearly_equal(fock2[sigma], mat2, epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(1.5 * energy, eps));
        }

        INFO("Test assymmetry of spin factors"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, false, true, 0.0, 1.0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK_FALSE(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
            }
            CHECK(std::abs(energy2 - energy) > eps);
        }

        INFO("Test that results are the same when indirect_term = true for ss_scale = 0"); {
            const auto &[fock2, energy2] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 1.0, 0);

            REQUIRE(fock2.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                CHECK(linalg::nearly_equal(fock2[sigma], fock[sigma], epsilon));
            }
            CHECK_THAT(energy2, Catch::Matchers::WithinRel(energy, eps));
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
            CHECK_THAT(energy3, Catch::Matchers::WithinRel(energy2, eps));
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
                const linalg::Mat mat2 = 1.5 * fock2[sigma];

                CHECK(linalg::nearly_equal(fock3[sigma], mat2, epsilon));
            }
            CHECK_THAT(energy3, Catch::Matchers::WithinRel(energy2 * 1.5, eps));
        }

        INFO("Check combined calculations give same results"); {
            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, orbitals, true, true, 1.0, 1.0);
            REQUIRE(fock3.size() == n_spin);
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const linalg::Mat mat2 = fock[sigma] + fock2[sigma];
                CHECK(linalg::nearly_equal(fock3[sigma], mat2, epsilon));
            }
            CHECK_THAT(energy + energy2, Catch::Matchers::WithinRel(energy3, eps));
        }

        INFO("Check empty spin channel"); {
            const auto C1 = linalg::random(n_ao, 1, seed);
            const auto C0 = linalg::random(n_ao, 0, seed);

            const utils::Orbitals empty_orb = {C1, C0};

            const auto &[fock3, energy3] = two_el::form_fock_two_el_df(
                base_integrals, empty_orb, true, true, 1.0,
                1.0);
            REQUIRE(fock3.size() == n_spin);
            CHECK_THAT(energy3, Catch::Matchers::WithinAbs(0, 100 * margin));
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
                CHECK_THAT(energy4, Catch::Matchers::WithinAbs(0, 100 * margin));
            }
        }
    }
}

TEST_CASE("Test two-electron - Check Fock norm") {
    constexpr size_t n_ao = 7;
    constexpr size_t n_df = 17;

    const auto J20 = linalg::random_pd(n_df, seed);
    const auto J30 = linalg::random(n_ao * (n_ao + 1) / 2, n_df, seed);

    const auto base_integrals = integrals::BaseIntegrals(J30, J20);

    std::vector<size_t> n_occ = {4};
    for (auto i = 0; i < 2; ++i) {
        for (const auto scale_same_spin: {0.0, 0.5}) {
            constexpr auto scale_opp_spin = 1.0;
            if (std::abs(scale_opp_spin + scale_same_spin) < 1e-10) {
                continue;
            }
            std::cout << "Opposite spin scale = " << scale_opp_spin << std::endl;
            std::cout << "Same spin scale = " << scale_same_spin << std::endl;

            const auto two_el_func = [&base_integrals, scale_opp_spin, scale_same_spin](
                const utils::DensityMatrix &D) -> utils::FockMatrixAndEnergy {
                const auto active_Co = fock::calculate_orbitals_from_density(D, 1e-3);

                return two_el::form_fock_two_el_df(
                    base_integrals, active_Co, true, true, scale_opp_spin, scale_same_spin);
            };

            const auto D = fock::random_density_matrix(n_occ, n_ao, seed);

            fock::check_energy_derivative_along_fock(two_el_func, D);
        }

        n_occ.push_back(3);
    }
}
