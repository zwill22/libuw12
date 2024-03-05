//
// Created by Zack Williams on 01/03/2024.
//

#ifndef CHECK_FOCK_HPP
#define CHECK_FOCK_HPP

#include <iostream>

#include "../src/utils/utils.hpp"

using namespace uw12;

namespace fock {
    inline utils::DensityMatrix random_density_matrix(
    const std::vector<size_t> & n_occ,
    const size_t n_ao,
    const int seed
    ) {
        utils::Orbitals C;
        utils::Occupations occ;

        const auto n_spin = n_occ.size();
        const auto max = n_spin == 1 ? 2 : 1;

        for (auto sigma = 0; sigma < n_spin; sigma++) {
            C.push_back(linalg::random(n_ao, n_occ[sigma], seed));
            const linalg::Vec tmp = linalg::random(n_occ[sigma], 1, seed);
            occ.emplace_back(max * tmp);
        }

        const auto Co = utils::occupation_weighted_orbitals(C, occ);

        return utils::construct_density(Co);
    }

    inline std::pair<linalg::Mat, linalg::Vec> orbitals_from_density(const linalg::Mat &D, const double epsilon) {
        const auto n_ao = linalg::n_rows(D);
        assert(linalg::n_cols(D) == n_ao);
        assert(epsilon > 0);

        if (linalg::nearly_equal(D, linalg::zeros(n_ao, n_ao), test::epsilon)) {
            return {linalg::zeros(n_ao, 1), linalg::ones(1)};
        }

        const linalg::Mat D_neg = -0.5 * (D + linalg::transpose(D));

        const auto [occ_neg, C] = linalg::eigen_decomposition(
            D_neg, epsilon, epsilon);

        const linalg::Vec occ = -1 * occ_neg;

        return {C, occ};
    }

    inline utils::Orbitals calculate_orbitals_from_density(const utils::DensityMatrix &D, const double epsilon) {
        utils::Orbitals orbitals;
        utils::Occupations occupations;

        const auto n_spin = D.size();

        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            const auto [C, occ] = orbitals_from_density(D[sigma], epsilon);
            orbitals.push_back(C);
            occupations.push_back(occ);
        }

        return utils::occupation_weighted_orbitals(orbitals, occupations);
    }

    inline double analytic_fock_norm(
        const utils::FockMatrix &F
    ) {
        const auto n_spin = F.size();

        double norm_sq = 0;

        for (int sigma = 0; sigma < n_spin; ++sigma) {
            const auto norm = linalg::norm(F[sigma]);
            norm_sq += norm * norm;
        }

        return std::sqrt(norm_sq);
    }

    template<typename Function>
    double numerical_fock_norm(
        const utils::FockMatrix &F,
        const utils::DensityMatrix &D,
        const Function &func,
        const double h
    ) {
        const auto fock_norm = analytic_fock_norm(F);

        utils::DensityMatrix D_plus;
        utils::DensityMatrix D_minus;
        const auto n_spin = F.size();
        if (D.size() != n_spin) {
            throw std::runtime_error("Density and Fock matrices have different numbers of spin channels");
        }

        for (size_t sigma = 0; sigma < n_spin; ++sigma) {
            D_plus.emplace_back(D[sigma] + h * F[sigma] / fock_norm);
            D_minus.emplace_back(D[sigma] - h * F[sigma] / fock_norm);
        }


        return (func(D_plus).energy - func(D_minus).energy) / (2 * h);
    }

    template<typename Function>
    void check_energy_derivative_along_fock(
        const Function &func,
        const utils::DensityMatrix &D,
        const double h = 1e-4,
        const double factor = 1e3,
        const bool print_results = false
    ) {
        const auto &[F, energy] = func(D);

        const auto fock_norm = analytic_fock_norm(F);
        std::cout << "Analytic Fock norm = " << fock_norm << std::endl;

        if (print_results) {
            double hrel = 1e-12;
            while (hrel < 1e4) {
                const auto fock_norm_num = numerical_fock_norm(F, D, func, hrel);
                std::cout << "Numerical Fock norm = " << fock_norm_num << std::endl;

                const auto rel_error = std::abs(fock_norm_num - fock_norm) / fock_norm;
                std::cout << "Relative Fock error = " << rel_error << std::endl;

                std::cout << "Relative threshold = " << factor * hrel << std::endl;

                hrel *= 10;
            }
        }

        const auto fock_norm_num = numerical_fock_norm(F, D, func, h);
        std::cout << "Numerical Fock norm = " << fock_norm_num << std::endl;

        const auto rel_error = std::abs(fock_norm_num - fock_norm) / fock_norm;
        std::cout << "Relative Fock error = " << rel_error << std::endl;

        std::cout << "Relative threshold = " << factor * h << std::endl;

        if (rel_error > factor * h) {
            throw std::runtime_error("Analytic and numerical Fock norms not equal");
        }
    }
}

#endif //CHECK_FOCK_HPP
