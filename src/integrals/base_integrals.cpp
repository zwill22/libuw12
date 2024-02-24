#include "base_integrals.hpp"

namespace uw12::integrals {
    BaseIntegrals::BaseIntegrals(
        TwoIndexFn two_index_fn_,
        ThreeIndexFn three_index_fn_,
        ThreeIndexFn three_index_ri_fn_,
        std::vector<int> df_sizes_,
        const int n_ao_,
        const int n_df_,
        const int n_ri_,
        const bool store_ao_integrals_,
        const bool store_ri_integrals_,
        const bool two_index_fn_provided_,
        const bool three_index_fn_provided_,
        const bool three_index_ri_fn_provided_,
        const std::shared_ptr<linalg::Mat> &J3_0_,
        const std::shared_ptr<linalg::Mat> &J3_,
        const std::shared_ptr<linalg::Mat> &J2_,
        const std::shared_ptr<linalg::Vec> &df_vals_,
        const std::shared_ptr<linalg::Mat> &J3_ri_0_
    ) : two_index_fn(std::move(two_index_fn_)),
        three_index_fn(std::move(three_index_fn_)),
        three_index_fn_ri(std::move(three_index_ri_fn_)),
        n_ao(n_ao_),
        n_df(n_df_),
        n_ri(n_ri_),
        df_sizes(std::move(df_sizes_)),
        J3_0(J3_0_),
        J3(J3_),
        J3_ri_0(J3_ri_0_),
        J3_ri(nullptr),
        P2(nullptr),
        df_vals(df_vals_),
        J3_lock(std::make_shared<std::mutex>()),
        J3_ri_lock(std::make_shared<std::mutex>()),
        eig_lock(std::make_shared<std::mutex>()),
        two_index_fn_provided(two_index_fn_provided_),
        three_index_fn_provided(three_index_fn_provided_),
        three_index_ri_fn_provided(three_index_ri_fn_provided_),
        store_ao_integrals(store_ao_integrals_),
        store_ri_integrals(store_ri_integrals_) {
        // If three index integrals are required
        if (three_index_fn_provided || three_index_ri_fn_provided) {
            if (df_sizes.empty()) {
                throw std::runtime_error("no df_sizes provided, cannot calculate three index integrals");
            }

            if (n_df == 0 || n_ao == 0) {
                throw std::runtime_error("Number of basis functions is zero, cannot calculate basis functions");
            }

            if (three_index_ri_fn_provided && n_ri == 0) {
                throw std::runtime_error("Three index ri integrals requested but number of ri basis functions is zero");
            }
        }

        // Check that the total number of df functions is equal to the sum of the sizes of all df shells
        if (const auto n_df_sh = df_sizes.size(); n_df_sh > 0) {
            if (n_df_sh > n_df) {
                throw std::runtime_error("More shells than basis functions in df basis");
            }

            int total = 0;
            for (const auto n_sh: df_sizes) {
                total += n_sh;
            }

            if (total != n_df) {
                throw std::runtime_error("total df basis functions inconsistent with number in shells");
            }
        }

        if (J2_) {
            if (P2) {
                throw std::runtime_error("density-fitting projector provided as well as J2idx");
            }
            if (df_vals) {
                throw std::runtime_error("density-fitting eigenvalue provided as well as J2idx");
            }

            const auto &[vals, vecs] = linalg::eigen_system(*J2_);

            P2 = std::make_shared<linalg::Mat>(vecs * linalg::p_inv(linalg::diagmat(vals)));

            df_vals = std::make_shared<linalg::Vec>(vals);
        }

        if (!J3_0) {
            J3_0 = std::make_shared<linalg::Mat>();
        }
        if (!J3) {
            J3 = std::make_shared<linalg::Mat>();
        }
        if (!J3_ri) {
            J3_ri = std::make_shared<linalg::Mat>();
        }
        if (!P2) {
            P2 = std::make_shared<linalg::Mat>();
        }
        if (!df_vals) {
            df_vals = std::make_shared<linalg::Vec>();
        }
        if (!J3_ri_0) {
            J3_ri_0 = std::make_shared<linalg::Mat>();
        }
    }

    BaseIntegrals::BaseIntegrals(
        const TwoIndexFn &two_index_fn,
        const ThreeIndexFn &three_index_fn,
        const ThreeIndexFn &three_index_ri_fn,
        const std::vector<int> &df_sizes,
        const int n_ao,
        const int n_df,
        const int n_ri,
        const bool store_ao_integrals,
        const bool store_ri_integrals
    )
        : BaseIntegrals(
            two_index_fn, three_index_fn, three_index_ri_fn,
            df_sizes, n_ao, n_df, n_ri, store_ao_integrals,
            store_ri_integrals, true, true,
            true
        ) {
    }

    BaseIntegrals::BaseIntegrals(
        const TwoIndexFn &two_index_fn,
        const ThreeIndexFn &three_index_fn,
        const std::vector<int> &df_sizes,
        const int n_ao,
        const int n_df,
        const bool store_ao_integrals
    )
        : BaseIntegrals(
            two_index_fn, three_index_fn, nullptr,
            df_sizes, n_ao, n_df, 0, store_ao_integrals, false,
            true, true, false
        ) {
    }

    BaseIntegrals::BaseIntegrals(
        const linalg::Mat &J3_0,
        const linalg::Mat &J2,
        const ThreeIndexFn &three_index_ri_fn,
        const std::vector<int> &df_sizes,
        const bool use_ri,
        const int n_ao,
        const int n_df,
        const int n_ri,
        const bool store_ri_integrals
    )
        : BaseIntegrals(
            nullptr, nullptr, three_index_ri_fn, df_sizes,
            n_ao, n_df, n_ri, true, store_ri_integrals,
            false, false, use_ri,
            std::make_shared<linalg::Mat>(J3_0), nullptr, std::make_shared<linalg::Mat>(J2)
        ) {
    }

    BaseIntegrals::BaseIntegrals(
        const linalg::Mat &J3_0,
        const linalg::Mat &J2,
        const linalg::Mat &J3_ri_0
    )
        : BaseIntegrals(
            nullptr, nullptr, nullptr, {}, 0, 0, 0,
            true, true, false, false,
            false, std::make_shared<linalg::Mat>(J3_0), nullptr,
            std::make_shared<linalg::Mat>(J2), nullptr, std::make_shared<linalg::Mat>(J3_ri_0)
        ) {
    }

    BaseIntegrals::BaseIntegrals(
        const linalg::Mat &J3_0,
        const linalg::Mat &J2
    )
        : BaseIntegrals(
            J3_0, J2, nullptr, {}, false, 0, 0, 0, false
        ) {
    }

    BaseIntegrals::BaseIntegrals(
        const linalg::Mat &J3,
        const linalg::Vec &df_vals
    )
        : BaseIntegrals(
            nullptr, nullptr, nullptr, {}, 0, 0, 0,
            true, false, false, false,
            false, nullptr, std::make_shared<linalg::Mat>(J3), nullptr,
            std::make_shared<linalg::Vec>(df_vals)
        ) {
    }

    linalg::Mat BaseIntegrals::two_index() const {
        if (!two_index_fn_provided) {
            throw std::runtime_error("no two index function provided");
        }

        return two_index_fn();
    }

    linalg::Mat BaseIntegrals::three_index(const int A) const {
        if (!three_index_fn_provided) {
            throw std::runtime_error("no three index function provided");
        }

        return three_index_fn(A);
    }

    linalg::Mat BaseIntegrals::three_index_ri(const int A) const {
        if (!three_index_ri_fn_provided) {
            throw std::runtime_error("no three index function provided");
        }

        return three_index_fn_ri(A);
    }

    bool BaseIntegrals::has_two_index_fn() const {
        return two_index_fn_provided;
    }

    bool BaseIntegrals::has_three_index_fn() const {
        return three_index_fn_provided;
    }

    bool BaseIntegrals::has_three_index_ri_fn() const {
        return three_index_ri_fn_provided;
    }

    void BaseIntegrals::set_J2_values() const {
        if (!two_index_fn_provided) {
            throw std::runtime_error("no two index integral function provided");
        }

        const auto J2 = two_index_fn();

        const auto &[vals, vecs] = linalg::eigen_system(J2);

        *P2 = vecs * linalg::p_inv(linalg::diagmat(vals));
        *df_vals = vals;
    }

    const linalg::Mat &BaseIntegrals::get_P2() const {
        std::lock_guard lock_guard(*eig_lock);

        if (linalg::empty(*P2)) {
            if (!linalg::empty(*df_vals)) {
                throw std::runtime_error("density-fitting eigenvalues already available");
            }

            parallel::isolate([&] { set_J2_values(); });
        }

        return *P2;
    }

    const linalg::Vec &BaseIntegrals::get_df_vals() const {
        std::lock_guard lock_guard(*eig_lock);

        if (linalg::empty(*df_vals)) {
            if (!linalg::empty(*P2)) {
                throw std::runtime_error("P2 is not empty but df values are");
            }
            parallel::isolate([&] { set_J2_values(); });
        }

        return *df_vals;
    }

    const std::vector<int> &BaseIntegrals::get_df_sizes() const {
        if (df_sizes.empty()) {
            throw std::runtime_error("df_sizes requested but none available");
        }

        int total = 0;
        for (const auto &shell_size: df_sizes) {
            total += shell_size;
        }

        if (total != n_df) {
            throw std::runtime_error("total number of df basis functions not equal to total sum of shell sizes");
        }

        return df_sizes;
    }


    std::vector<int> BaseIntegrals::get_df_offsets() const {
        const auto sizes = get_df_sizes();
        const auto n_shells = sizes.size();

        int total = 0;
        std::vector<int> df_offsets(n_shells);

        for (int i = 0; i < n_shells; ++i) {
            df_offsets[i] = total;
            total += sizes[i];
        }

        if (total != n_df) {
            throw std::runtime_error("total number of df basis functions must equal total sum of shell sizes");
        }

        return df_offsets;
    }

    const linalg::Mat &BaseIntegrals::get_J3_0() const {
        std::lock_guard lock_guard(*J3_lock);
        if (linalg::empty(*J3_0)) {
            throw std::runtime_error("J3_0 not stored");
        }

        return *J3_0;
    }

    const linalg::Mat &BaseIntegrals::get_J3() const {
        std::lock_guard lock_guard(*J3_lock);
        if (linalg::empty(*J3)) {
            const auto &P = get_P2();

            if (linalg::empty(*J3_0)) {
                if (!three_index_fn_provided) {
                    throw std::runtime_error("three-centre integrals requested, but no function provided to calculate");
                }

                if (!store_ao_integrals) {
                    std::cerr << "three-center ao integrals requested this may be memory inefficient" << std::endl;
                }

                if (n_ao == 0) {
                    throw std::runtime_error("three-centre integrals requested but n_ao is zero");
                }

                if (n_df == 0) {
                    throw std::runtime_error("three-centre integrals requested but n_df is zero");
                }

                if (df_sizes.empty()) {
                    throw std::runtime_error("three-index integrals requested but no df offsets provided");
                }

                const auto n_row = n_ao * (n_ao + 1) / 2;

                parallel::isolate([&] {
                    *J3 = coulomb_3idx(three_index_fn, get_df_offsets(), n_row, n_df) * P;
                });
            } else {
                *J3 = *J3_0 * P;
            }
        }

        return *J3;
    }

    const linalg::Mat &BaseIntegrals::get_J3_ri_0() const {
        std::lock_guard lock_guard(*J3_ri_lock);

        if (linalg::empty(*J3_ri_0)) {
            throw std::runtime_error("J3_ri_0 not stored");
        }
        return *J3_ri_0;
    }

    const linalg::Mat &BaseIntegrals::get_J3_ri() const {
        std::lock_guard lock_guard(*J3_ri_lock);
        if (linalg::empty(*J3_ri)) {
            const auto &P = get_P2();

            if (linalg::empty(*J3_ri_0)) {
                if (!three_index_ri_fn_provided) {
                    throw std::runtime_error("ri integrals requested, but no function provided to calculate");
                }

                if (!store_ri_integrals) {
                    std::cerr << "three-center ri integrals requested this may be memory inefficient" << std::endl;
                }

                if (n_ao == 0) {
                    throw std::runtime_error("ri integrals requested but number of ri basis functions is zero");
                }

                if (n_df == 0) {
                    throw std::runtime_error("ri integrals requested but number of df basis functions is zero");
                }

                if (n_ri == 0) {
                    throw std::runtime_error("ri integrals requested but number of ri basis functions is zero");
                }

                if (df_sizes.empty()) {
                    throw std::runtime_error("three-index integrals requested but no df offsets provided");
                }

                const auto n_row = n_ao * n_ri;

                parallel::isolate([&] {
                    *J3_ri = coulomb_3idx(three_index_fn_ri, get_df_offsets(), n_row, n_df) * P;
                });
            } else {
                *J3_ri = *J3_ri_0 * P;
            }
        }

        return *J3_ri;
    }

    int BaseIntegrals::get_number_ao() const {
        return n_ao;
    }

    int BaseIntegrals::get_number_df() const {
        return n_df;
    }

    int BaseIntegrals::get_number_ri() const {
        return n_ri;
    }

    bool BaseIntegrals::storing_ao() const {
        return store_ao_integrals;
    }

    bool BaseIntegrals::storing_ri() const {
        return store_ri_integrals;
    }

    bool BaseIntegrals::has_P2() const {
        if (linalg::empty(*P2)) {
            return false;
        }

        return true;
    }

    bool BaseIntegrals::has_df_vals() const {
        if (linalg::empty(*df_vals)) {
            return false;
        }

        return true;
    }

    bool BaseIntegrals::has_J3_0() const {
        if (linalg::empty(*J3_0)) {
            return false;
        }

        return true;
    }

    bool BaseIntegrals::has_J3() const {
        if (linalg::empty(*J3)) {
            return false;
        }

        return true;
    }

    bool BaseIntegrals::has_J3_ri() const {
        if (linalg::empty(*J3_ri)) {
            return false;
        }

        return true;
    }

    bool BaseIntegrals::has_J3_ri_0() const {
        if (linalg::empty(*J3_ri_0)) {
            return false;
        }

        return true;
    }
} // namespace uw12::integrals
