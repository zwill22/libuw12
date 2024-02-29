//
// Created by Zack Williams on 26/02/2024.
//

#include "../src/integrals/integrals.hpp"

#include "catch.hpp"

using namespace uw12::utils;
using namespace uw12::linalg;
using namespace uw12::integrals;

constexpr auto epsilon = 1e-10;
constexpr auto seed = 2;

TEST_CASE("Test integrals - Closed shell") {
    constexpr size_t n_ao = 11;
    constexpr size_t n_df = 25;
    constexpr size_t n_ri = 32;
    constexpr size_t n_occ = 5;
    constexpr size_t n_active = 3;

    const std::vector<size_t> df_sizes = {1, 3, 5, 1, 3, 5, 7};

    size_t total = 0;
    for (const auto size: df_sizes) {
        total += size;
    }
    REQUIRE(total == n_df);

    auto J20 = random_pd(n_df, seed);
    auto J30 = random(n_ao * (n_ao + 1) / 2, n_df, seed);
    auto J3ri0 = random(n_ao * n_ri, n_df, seed);


    const TwoIndexFn two_index_fn = [&J20]()-> Mat {
        return J20;
    };

    const ThreeIndexFn three_index_fn = [&df_sizes, &J30](const size_t A) -> Mat {
        constexpr auto n_row = n_ao * (n_ao + 1) / 2;
        const auto n_col = df_sizes[A];

        size_t offset = 0;
        for (size_t i = 0; i < A; ++i) {
            offset += df_sizes[i];
        }

        return sub_mat(J30, 0, offset, n_row, n_col);
    };

    const ThreeIndexFn three_index_ri_fn = [&df_sizes, &J3ri0](const int A) -> Mat {
        constexpr auto n_row = n_ao * n_ri;
        const auto n_col = df_sizes[A];

        size_t offset = 0;
        for (size_t i = 0; i < A; ++i) {
            offset += df_sizes[i];
        }

        return sub_mat(J3ri0, 0, offset, n_row, n_col);
    };


    const auto base_integrals = BaseIntegrals(
        two_index_fn, three_index_fn, three_index_ri_fn, df_sizes, n_ao, n_df, n_ri,
        true, true);


    const auto C = random(n_ao, n_occ, seed);

    const auto Cactive = tail_cols(C, n_active);

    const Orbitals occ_orbitals = {C};
    const Orbitals active_orbitals = {Cactive};

    const auto integrals = Integrals(base_integrals, occ_orbitals, active_orbitals);

    const auto n_spin = integrals.spin_channels();
    const auto &X3pkA = integrals.get_X3idx_one_trans();
    const auto &X3miA = integrals.get_X3idx_one_trans_ri();
    REQUIRE(X3miA.size() == n_spin);
    REQUIRE(X3pkA.size() == n_spin);

    const auto &X3ikA = integrals.get_X3idx_two_trans();
    REQUIRE(X3ikA.size() == n_spin);

    MatVec X4pkjl = {};
    MatVec X4ikjl = {};
    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
        X4pkjl.push_back(integrals.get_X4idx_three_trans(sigma));
        X4ikjl.push_back(integrals.get_X4idx_four_trans(sigma));
    }

    const auto X_D = integrals.get_X_D();

    CHECK(nearly_equal(base_integrals.get_P2(), integrals.get_P2(), epsilon));
    CHECK(nearly_equal(base_integrals.get_df_vals(), integrals.get_df_vals(), epsilon));
    CHECK(nearly_equal(base_integrals.get_J3(), integrals.get_J3(), epsilon));
    CHECK(nearly_equal(base_integrals.get_J3_ri(), integrals.get_J3_ri(), epsilon));

    CHECK(integrals.number_ao_orbitals() == n_ao);
    CHECK(integrals.number_occ_orbitals(0) == n_occ);
    CHECK(integrals.number_active_orbitals(0) == n_active);

    SECTION("Two MO transform (direct)") {
        const auto &base2 = integrals.get_base_integrals();

        const auto integrals2 = Integrals(
            base2, occ_orbitals, active_orbitals, false, true);

        const auto &X3ikA2 = integrals2.get_X3idx_two_trans();
        REQUIRE(X3ikA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3ikA[sigma], X3ikA2[sigma], epsilon));
        }

        const auto &X3pkA2 = integrals2.get_X3idx_one_trans();
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3pkA[sigma], X3pkA2[sigma], epsilon));
        }
    }

    SECTION("J3_0") {
        const auto base2 = BaseIntegrals(J30, J20, J3ri0);

        const auto integrals2 = Integrals(base2, occ_orbitals, active_orbitals, true, false);

        const auto &X3pkA2 = integrals2.get_X3idx_one_trans();
        REQUIRE(X3pkA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3pkA[sigma], X3pkA2[sigma], epsilon));
        }

        const auto &X3miA2 = integrals2.get_X3idx_one_trans_ri();
        REQUIRE(X3miA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3miA[sigma], X3miA2[sigma], epsilon));
        }

        const auto X_D2 = integrals2.get_X_D();
        REQUIRE(X_D2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X_D[sigma], X_D2[sigma], epsilon));
        }

        const auto integrals3 = Integrals(base2, occ_orbitals, active_orbitals, false, false);
        const auto &X3ikA2 = integrals3.get_X3idx_two_trans();
        REQUIRE(X3ikA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3ikA[sigma], X3ikA2[sigma], epsilon));
        }
    }

    SECTION("Check errors") {
        CHECK_THROWS(Integrals(base_integrals, occ_orbitals, {Cactive, Cactive}));
        CHECK_THROWS(Integrals(base_integrals, occ_orbitals, {}));
        CHECK_THROWS(Integrals(base_integrals, {}, active_orbitals));
        CHECK_THROWS(Integrals(base_integrals, {}, {}));
        CHECK_THROWS(Integrals(base_integrals, {C, C, C}, {C, C, C}));
    }

    SECTION("Direct") {
        const auto base_integrals2 = BaseIntegrals(
            two_index_fn, three_index_fn, three_index_ri_fn, df_sizes, n_ao, n_df, n_ri,
            false, false);

        const auto integrals2 = Integrals(base_integrals2, occ_orbitals, active_orbitals, false);

        const auto &X3ikA2 = integrals2.get_X3idx_two_trans();
        REQUIRE(X3ikA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3ikA[sigma], X3ikA2[sigma], epsilon));
        }

        const auto &X3pkA2 = integrals2.get_X3idx_one_trans();
        REQUIRE(X3pkA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3pkA[sigma], X3pkA2[sigma], epsilon));
        }

        const auto &X3miA2 = integrals2.get_X3idx_one_trans_ri();
        REQUIRE(X3miA2.size() == n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X3miA[sigma], X3miA2[sigma], epsilon));
        }

        const auto X_D2 = integrals2.get_X_D();
        REQUIRE(X_D2.size() == X_D.size());
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            CHECK(nearly_equal(X_D[sigma], X_D2[sigma], epsilon));
        }
    }
}
