#include "two_electron.hpp"

#include "../utils/linalg.hpp"
#include "../utils/utils.hpp"
#include "../integrals/base_integrals.hpp"
#include "../integrals/integrals.hpp"

namespace uw12::two_el {
    // Vector of length lambda for each spin channel of
    // Sum_j ( lambda tilde | WV | jj) = O_lambda * ( lambda | WV | jj)
    std::vector<linalg::Vec> calculate_WV_tilde_D(
        const std::vector<linalg::Vec> &WV_D,
        const integrals::BaseIntegrals &WV
    ) {
        const auto n_spin = WV_D.size();
        const auto &WV_vals = WV.get_df_vals();

        std::vector<linalg::Vec> result(n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            assert(linalg::n_elem(WV_vals) == linalg::n_elem(WV_D[sigma]));

            result[sigma] = linalg::schur(WV_vals, WV_D[sigma]);
        }
        return result;
    }

    linalg::Mat calculate_direct_fock(
        const integrals::BaseIntegrals &WV,
        const linalg::Vec &WV_tilde_D
    ) {
        // TODO Implement fully direct mo version
        if (WV.has_J3_0()) {
            return utils::square(WV.get_J3_0() * WV.get_P2() * WV_tilde_D);
        }

        return utils::square(WV.get_J3() * WV_tilde_D);

    }

    std::vector<linalg::Vec> calculate_X_D_direct(
        const integrals::BaseIntegrals &base_integrals,
        const utils::DensityMatrix &D
    ) {
        const auto n_spin = D.size();

        const auto offsets = base_integrals.get_df_offsets();
        const auto n_df = base_integrals.get_number_df();
        assert(offsets.size() < n_df);

        std::vector<linalg::Vec> density(n_spin);
        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            density[sigma] = utils::lower(D[sigma], 2);
        }

        std::vector out(n_spin, linalg::Vec(n_df));

        const auto parallel_fn = [&out, &base_integrals, &offsets,
                    &density](const size_t A) {
            const auto off_a = offsets[A];

            const auto shell_results = base_integrals.three_index(A);

            for (size_t sigma = 0; sigma < out.size(); ++sigma) {
                linalg::assign_rows(out[sigma], linalg::transpose(shell_results) * density[sigma], off_a);
            }
        };

        parallel::parallel_for(0, offsets.size(), parallel_fn);

        for (auto &integrals: out) {
            integrals = 0.5 * static_cast<double>(n_spin) * linalg::transpose(base_integrals.get_P2()) * integrals;
        }

        return out;
    }

    // Transform base_integrals eigenvectors (lambda| w |alpha beta)
    // over density to give total contribution for each eigen-component
    // WV_D = Sum_j ( lambda | w | jj ) for each lambda for each spin channel
    // TODO Move function for testing
    std::vector<linalg::Vec> calculate_X_D(
        const integrals::BaseIntegrals &base_integrals,
        const utils::DensityMatrix &D
    ) {
        using namespace linalg;
        // TODO Add test to check all versions give same answer

        const auto n_spin = D.size();
        std::vector<Vec> result(n_spin);
        if (base_integrals.has_J3_0()) {
            const auto &WV3_0 = base_integrals.get_J3_0();
            const auto &P2 = base_integrals.get_P2();
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const auto D_sigma = utils::lower(D[sigma], 2);

                assert(n_rows(D_sigma) == n_rows(WV3_0));
                result[sigma] = 0.5 * static_cast<double>(n_spin) * transpose(P2) * transpose(WV3_0) * D_sigma;
            }
        } else if (base_integrals.has_J3()) {
            const auto &WV3 = base_integrals.get_J3();
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const auto D_sigma = utils::lower(D[sigma], 2);

                assert(n_rows(D_sigma) == n_rows(WV3));
                result[sigma] = 0.5 * static_cast<double>(n_spin) * transpose(WV3) * D_sigma;
            }
        } else {
            result = calculate_X_D_direct(base_integrals, D);
        }

        return result;
    }

    utils::FockMatrixAndEnergy direct_fock(
        const integrals::BaseIntegrals &WV,
        const utils::DensityMatrix &D,
        const bool calculate_fock,
        const double scale_opp_spin,
        const double scale_same_spin
    ) {
        const auto n_spin = D.size();
        const auto n_ao = WV.get_number_ao();

        std::vector fock(n_spin, linalg::zeros(n_ao, n_ao));
        double energy = 0;

        // Sum_j (lambda | WV | jj) at each lambda
        const auto WV_D = calculate_X_D(WV, D);
        // Transform WV_D using df-eigenvalues O_lambda
        const auto WV_tilde_D = calculate_WV_tilde_D(WV_D, WV);

        for (size_t sigma = 0; sigma < n_spin; sigma++) {
            for (size_t sigmaprime = 0; sigmaprime < n_spin; sigmaprime++) {
                if (n_ao == 0) {
                    continue;
                }

                const auto energy_spin_factor =
                        (n_spin == 1)
                            ? 2 * (scale_opp_spin + scale_same_spin)
                            : ((sigma == sigmaprime) ? scale_same_spin : scale_opp_spin);

                if (energy_spin_factor == 0) {
                    continue;
                }

                const auto fock_spin_factor = 0.5 * static_cast<double>(n_spin) * energy_spin_factor;

                energy += 0.5 * energy_spin_factor * linalg::dot(WV_D[sigma], WV_tilde_D[sigmaprime]);

                if (calculate_fock) {
                    fock[sigma] += fock_spin_factor * calculate_direct_fock(WV, WV_tilde_D[sigmaprime]);
                } // calculate_fock
            } // sigmaprime
        } // sigma

        return {fock, energy};
    }

    // #######################################################################
    // #######################################################################
    // #######################################################################

    utils::FockMatrixAndEnergy indirect_fock(
        const integrals::Integrals &WV,
        const bool calculate_fock
        ) {
        const auto n_spin = WV.spin_channels();
        const auto n_ao = WV.number_ao_orbitals();

        const auto &WV_vals = WV.get_df_vals();
        const auto n_df = linalg::n_elem(WV_vals);

        std::vector fock(n_spin, linalg::zeros(n_ao, n_ao));
        double energy = 0;

        const auto &WV3idx_two_trans = WV.get_X3idx_two_trans();

        for (size_t sigma = 0; sigma < n_spin; sigma++) {
            const auto n_occ = WV.number_occ_orbitals(sigma);
            assert(n_occ == WV.number_active_orbitals(sigma));

            if (n_ao == 0 || n_occ == 0 || n_df == 0) {
                continue;
            }

            energy -= 0.5 * ((n_spin == 1) ? 2 : 1) * linalg::dot(
                        WV3idx_two_trans[sigma] * linalg::diagmat(WV_vals), WV3idx_two_trans[sigma]);

            if (calculate_fock) {
                const auto &WV3idx_one_trans = WV.get_X3idx_one_trans();
                const linalg::Mat WV3idx_one_trans_tilde = linalg::reshape(
                    WV3idx_one_trans[sigma] * linalg::diagmat(WV_vals), n_ao, n_occ * n_df);

                // Reshape WV3idx_two_trans to size (n_ao, nj * na) multiplication sums
                // over nj and na indices returning a matrix of size (n_ao, n_ao)
                fock[sigma] = -linalg::reshape(WV3idx_one_trans[sigma], n_ao, n_occ * n_df)
                              * linalg::transpose(WV3idx_one_trans_tilde);
            }
        } // sigma

        return {fock, energy};
    }

    utils::FockMatrixAndEnergy form_fock_two_el_df(
        const integrals::BaseIntegrals &WV,
        const utils::Orbitals &active_Co,
        const bool indirect_term,
        const bool calculate_fock,
        const double scale_opp_spin,
        const double scale_same_spin
    ) {
        const auto active_D = utils::construct_density(active_Co);

        auto fock = direct_fock(WV, active_D, calculate_fock, scale_opp_spin, scale_same_spin);

        // TODO Implement near zero for double
        if (indirect_term && scale_same_spin != 0) {
            const auto WV_integrals = integrals::Integrals(WV, active_Co, active_Co,
                                                           calculate_fock, calculate_fock);

            fock += scale_same_spin * indirect_fock(WV_integrals, calculate_fock);
        }

        return fock;
    }
} // uw12::two_el
