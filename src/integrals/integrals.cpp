//
// Created by Zack Williams on 30/11/2020.
//

#include "integrals.hpp"

#include <cassert>
#include <iostream>

#include "base_integrals.hpp"
#include "transformations.hpp"

namespace uw12::integrals {
    using utils::MatVec;

    template<typename IntegralFunc, typename TransformFunc>
    std::vector<linalg::Mat> calculate_integrals_direct(
        const BaseIntegrals &base_integrals,
        const IntegralFunc &integral_fn,
        const TransformFunc &transform_fn,
        const std::vector<size_t> &n_rows
    ) {
        const auto offsets = base_integrals.get_df_offsets();

        const auto n_df = base_integrals.get_number_df();

        std::vector<linalg::Mat> out(n_rows.size());
        for (size_t i = 0; i < n_rows.size(); ++i) {
            out[i] = linalg::mat(n_rows[i], n_df);
        }

        const auto parallel_fn = [&out, &offsets, &transform_fn, &integral_fn](const size_t A) {
            const auto off_a = offsets[A];

            const auto shell_results = integral_fn(A);

            for (size_t sigma = 0; sigma < out.size(); ++sigma) {
                linalg::assign_cols(out[sigma], transform_fn(shell_results, sigma), off_a);
            }
        };

        parallel::parallel_for(0, offsets.size(), parallel_fn);

        for (auto &integrals: out) {
            integrals *= base_integrals.get_P2();
        }

        return out;
    }

    std::vector<linalg::Mat> calculate_X3idx_one_trans_direct(
        const BaseIntegrals &base_integrals,
        const utils::Orbitals &orbitals
    ) {
        const auto integral_fn = [&base_integrals](const size_t A) {
            return base_integrals.three_index(A);
        };

        const auto transform_fn = [&orbitals](const linalg::Mat &matrix, const size_t sigma) {
            return transformations::mo_transform_one_index_full(matrix, orbitals[sigma]);
        };

        const auto nao = base_integrals.get_number_ao();

        const auto n_rows = [&orbitals, nao]() {
            std::vector<size_t> out;
            for (const auto &orb: orbitals) {
                out.push_back(linalg::n_cols(orb) * nao);
            }
            return out;
        }();

        return calculate_integrals_direct(base_integrals, integral_fn, transform_fn, n_rows);
    }

    std::vector<linalg::Mat> calculate_X3idx_one_trans_direct_ri(
        const BaseIntegrals &base_integrals,
        const utils::Orbitals &active_orbitals
    ) {
        const auto integral_fn = [&base_integrals](const size_t A) {
            return base_integrals.three_index_ri(A);
        };

        const auto transform_fn = [&active_orbitals](const linalg::Mat &matrix, const size_t sigma) {
            return transformations::transform_second_index(matrix, active_orbitals[sigma]);
        };

        const auto n_ri = base_integrals.get_number_ri();
        const auto n_rows = [&active_orbitals, n_ri]() {
            std::vector<size_t> out;
            for (const auto &orb: active_orbitals) {
                out.push_back(linalg::n_cols(orb) * n_ri);
            }
            return out;
        }();

        return calculate_integrals_direct(base_integrals, integral_fn, transform_fn, n_rows);
    }

    std::vector<linalg::Mat> calculate_X3idx_two_trans_direct(
        const BaseIntegrals &base_integrals,
        const utils::Orbitals &occ_orbitals,
        const utils::Orbitals &active_orbitals
    ) {
        assert(occ_orbitals.size() == active_orbitals.size());

        const auto integral_fn = [&base_integrals](const size_t A) {
            return base_integrals.three_index(A);
        };

        const auto transform_fn = [&occ_orbitals, &active_orbitals](
            const linalg::Mat &matrix, const size_t sigma) {
            return transformations::mo_transform_two_index_full(
                matrix, active_orbitals[sigma], occ_orbitals[sigma]);
        };

        const auto n_rows = [&occ_orbitals, &active_orbitals]() {
            std::vector<size_t> out(occ_orbitals.size());
            for (size_t i = 0; i < occ_orbitals.size(); ++i) {
                out[i] = linalg::n_cols(occ_orbitals[i]) * linalg::n_cols(active_orbitals[i]);
            }
            return out;
        }();

        return calculate_integrals_direct(base_integrals, integral_fn, transform_fn, n_rows);
    }

    std::vector<linalg::Vec> calculate_X_D_direct(
        const BaseIntegrals &base_integrals,
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

    Integrals::Integrals(
        const BaseIntegrals &_base_integrals,
        const utils::Orbitals &_occ_orbitals,
        const utils::Orbitals &_active_orbitals,
        const bool _store_one_trans,
        const bool _store_one_trans_ri
    )
        : base_integrals(std::make_shared<BaseIntegrals>(_base_integrals)),
          occ_orbitals(_occ_orbitals),
          active_orbitals(_active_orbitals),
          one_trans(std::make_shared<MatVec>()),
          one_trans_ri(std::make_shared<MatVec>()),
          two_trans(std::make_shared<MatVec>()),
          one_trans_lock(std::make_shared<std::mutex>()),
          one_trans_ri_lock(std::make_shared<std::mutex>()),
          two_trans_lock(std::make_shared<std::mutex>()),
          store_one_trans(_store_one_trans),
          store_one_trans_ri(_store_one_trans_ri) {
        if (_occ_orbitals.size() == 1 || _occ_orbitals.size() == 2) {
            if (_occ_orbitals.size() != _active_orbitals.size()) {
                throw std::runtime_error("different number of spin channels in occupied and active orbitals");
            }
        } else {
            throw std::runtime_error("Invalid number of spin channels");
        }
    }

    const MatVec &Integrals::get_X3idx_one_trans() const {
        std::lock_guard lock_guard(*one_trans_lock);

        if (one_trans->empty()) {
            if (!store_one_trans) {
                std::cerr << "one transformed ao integrals have been requested and will "
                        "be stored"
                        << std::endl;
            }
            const auto n_spin = occ_orbitals.size();

            MatVec out(n_spin);
            parallel::isolate([&] {
                if (base_integrals->has_J3_0()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::mo_transform_one_index_full(
                                         base_integrals->get_J3_0(), occ_orbitals[sigma]) *
                                     base_integrals->get_P2();
                    }
                } else if (base_integrals->has_J3() || base_integrals->storing_ao()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::mo_transform_one_index_full(
                            base_integrals->get_J3(), occ_orbitals[sigma]);
                    }
                } else {
                    out = calculate_X3idx_one_trans_direct(*base_integrals, occ_orbitals);
                }
            });

            *one_trans = out;
        }

        return *one_trans;
    }

    const MatVec &Integrals::get_X3idx_one_trans_ri() const {
        std::lock_guard lock_guard(*one_trans_ri_lock);

        if (one_trans_ri->empty()) {
            if (!store_one_trans_ri) {
                std::cerr << "one transformed ri integrals have been requested and will "
                        "be stored"
                        << std::endl;
            }
            const auto n_spin = active_orbitals.size();

            MatVec out(n_spin);
            parallel::isolate([&] {
                if (base_integrals->has_J3_ri_0()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] =
                                transformations::transform_second_index(
                                    base_integrals->get_J3_ri_0(), active_orbitals[sigma]) *
                                base_integrals->get_P2();
                    }
                } else if (base_integrals->has_J3_ri() || base_integrals->storing_ri()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::transform_second_index(
                            base_integrals->get_J3_ri(), active_orbitals[sigma]);
                    }
                } else {
                    out = calculate_X3idx_one_trans_direct_ri(*base_integrals,
                                                              active_orbitals);
                }
            });

            *one_trans_ri = out;
        }

        return *one_trans_ri;
    }

    const MatVec &Integrals::get_X3idx_two_trans() const {
        std::lock_guard lock_guard(*two_trans_lock);

        if (two_trans->empty()) {
            const auto n_spin = active_orbitals.size();

            MatVec out(n_spin);
            parallel::isolate([&] {
                if (!one_trans->empty() || store_one_trans) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::transform_first_index(
                            get_X3idx_one_trans()[sigma], active_orbitals[sigma]);
                    }
                } else if (base_integrals->has_J3_0()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::mo_transform_two_index_full(
                                         base_integrals->get_J3_0(), active_orbitals[sigma],
                                         occ_orbitals[sigma]) *
                                     base_integrals->get_P2();
                    }
                } else if (base_integrals->has_J3() || base_integrals->storing_ao()) {
                    for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                        out[sigma] = transformations::mo_transform_two_index_full(
                            base_integrals->get_J3(), active_orbitals[sigma],
                            occ_orbitals[sigma]);
                    }
                } else {
                    out = calculate_X3idx_two_trans_direct(*base_integrals, occ_orbitals,
                                                           active_orbitals);
                }
            });

            *two_trans = out;
        }

        return *two_trans;
    }

    linalg::Mat Integrals::get_X4idx_three_trans(const size_t sigma) const {
        return get_X3idx_one_trans()[sigma] *
               linalg::diagmat(base_integrals->get_df_vals()) *
               linalg::transpose(get_X3idx_two_trans()[sigma]);
    }

    linalg::Mat Integrals::get_X4idx_four_trans(const size_t sigma) const {
        const auto X3idx_two_trans_sigma = get_X3idx_two_trans()[sigma];

        return X3idx_two_trans_sigma *
               linalg::diagmat(base_integrals->get_df_vals()) *
               linalg::transpose(X3idx_two_trans_sigma);
    }

    std::vector<linalg::Vec> Integrals::get_X_D() const {
        // TODO Add test to check all versions give same answer
        using namespace linalg;

        const auto n_spin = active_orbitals.size();
        const auto D = utils::construct_density(active_orbitals);
        assert(D.size() == n_spin);

        std::vector<Vec> result(n_spin);
        if (base_integrals->has_J3_0()) {
            const auto &WV3_0 = base_integrals->get_J3_0();
            const auto &P2 = base_integrals->get_P2();
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const auto D_sigma = utils::lower(D[sigma], 2);

                assert(linalg::n_rows(D_sigma) == linalg::n_rows(WV3_0));
                result[sigma] = 0.5 * static_cast<double>(n_spin) * transpose(P2) * transpose(WV3_0) * D_sigma;
            }
        } else if (base_integrals->has_J3()) {
            const auto &WV3 = base_integrals->get_J3();
            for (size_t sigma = 0; sigma < n_spin; ++sigma) {
                const auto D_sigma = utils::lower(D[sigma], 2);

                assert(linalg::n_rows(D_sigma) == linalg::n_rows(WV3));
                result[sigma] = 0.5 * static_cast<double>(n_spin) * transpose(WV3) * D_sigma;
            }
        } else {
            result = calculate_X_D_direct(*base_integrals, D);
        }

        return result;
    }

    const linalg::Mat &Integrals::get_P2() const {
        return base_integrals->get_P2();
    }

    const linalg::Vec &Integrals::get_df_vals() const {
        return base_integrals->get_df_vals();
    }

    const linalg::Mat &Integrals::get_J3() const {
        return base_integrals->get_J3();
    }

    const linalg::Mat &Integrals::get_J3_ri() const {
        return base_integrals->get_J3_ri();
    }

    size_t Integrals::spin_channels() const {
        const auto n_spin = active_orbitals.size();
        assert(occ_orbitals.size() == n_spin);
        assert(n_spin == 1 || n_spin == 2);

        return n_spin;
    }

    size_t Integrals::number_ao_orbitals() const {
        const auto n_ao = linalg::n_rows(active_orbitals[0]);
        assert(linalg::n_rows(occ_orbitals[0]) == n_ao);

        return n_ao;
    }

    size_t Integrals::number_occ_orbitals(const size_t sigma) const {
        return linalg::n_cols(occ_orbitals[sigma]);
    }

    size_t Integrals::number_active_orbitals(const size_t sigma) const {
        return linalg::n_cols(active_orbitals[sigma]);
    }

    const BaseIntegrals &Integrals::get_base_integrals() const {
        return *base_integrals;
    }
} // namespace uw12::integrals
