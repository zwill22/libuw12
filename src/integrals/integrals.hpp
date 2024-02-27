//
// Created by Zack Williams on 30/11/2020.
//

#ifndef UW12_INTEGRALS_HPP
#define UW12_INTEGRALS_HPP

#include "base_integrals.hpp"
#include "../utils/linalg.hpp"
#include "../utils/utils.hpp"

namespace uw12::integrals {
    /// \brief Class for calculating and managing all integrals for df-UW12
    ///
    /// This class calculates and stores mo transformed UW12 integrals. The
    /// class is initialised using a `BaseIntegrals` object which manages all
    /// untransformed three-index integrals and two-index df integrals.
    ///
    /// All mo transformed integrals may be accessed through this class, while
    /// functions to access untransformed integrals in `BaseIntegrals` are also
    /// provided. No integrals are calculated until requested.
    ///
    /// Integrals are calculated via differing methods depending on which integrals
    /// are stored and which integrals storage is set to true. For example two-index
    /// mo transformed integrals may be calculated using `get_X3idx_two_trans`. If
    /// `store_one_trans` is true then the one-index transformed integrals are
    /// calculated and stored before transforming the second-index. Else the
    /// two transformed integrals are calculated from 3idx ao integral from
    /// `BaseIntegrals` or directly depending on the settings for
    /// `storing_ao_integrals` in `BaseIntegrals`.
    class Integrals {
    public:
        /// Constructor for `Integrals` class using a `BaseIntegrals` class and the
        /// active and full occupied orbitals.
        Integrals(
            const BaseIntegrals &_base_integrals,
            const utils::Orbitals &_occ_orbitals,
            const utils::Orbitals &_active_orbitals,
            bool _store_one_trans = true,
            bool _store_one_trans_ri = true
        );

        /// Calculate the one mo-index transformed three index integrals
        /// \f$(\rho k| A)\f$. For `occ_orbitals` index k. Returns a matrix of size
        /// (nao * n_occ, ndf) for each spin channel.
        const utils::MatVec &get_X3idx_one_trans() const;

        /// Calculate the one mo-index transformed three index ri integrals
        /// \f$(\mu i| A)\f$. For `active_orbitals` index i. Returns a matrix of size
        /// (nao * n_occ, ndf) for each spin channel.
        const utils::MatVec &get_X3idx_one_trans_ri() const;

        /// Calculate the two mo-index transformed three index integrals
        /// \f$(i k| A)\f$. For `active_orbitals` index i and `occ_orbitals` index k.
        /// Returns a matrix of size (nao * n_occ, ndf) for each spin channel.
        const utils::MatVec &get_X3idx_two_trans() const;

        /// Calculate the three mo-index transformed four-index integrals
        /// \f$(\rho k|j l)\f$ for spin channel `sigma`. For `active_orbitals` index
        /// j and `occ_orbitals` indices k,l.
        linalg::Mat get_X4idx_three_trans(size_t sigma) const;

        /// Calculate the four mo-index transformed four-index integrals
        /// \f$(i k|j l)\f$ for spin channel `sigma`. For `active_orbitals` indices
        /// i,j and `occ_orbitals` indices k,l.
        linalg::Mat get_X4idx_four_trans(size_t sigma) const;

        /// Wrapper functions to access integrals from `BaseIntegrals` {
        const linalg::Mat &get_P2() const;

        const linalg::Vec &get_df_vals() const;

        const linalg::Mat &get_J3() const;

        const linalg::Mat &get_J3_ri() const;

        /// }

        /// Calculate the number of spin channels based on the orbitals provided
        size_t spin_channels() const;

        /// Determine the number of ao basis functions
        size_t number_ao_orbitals() const;

        /// Get the number of occupied orbitals in channel sigma
        size_t number_occ_orbitals(size_t sigma) const;

        /// Get the number of active occupied orbitals in channel sigma
        size_t number_active_orbitals(size_t sigma) const;

        /// Obtain the base integrals stored in the class
        const BaseIntegrals &get_base_integrals() const;

    private:
        std::shared_ptr<BaseIntegrals> base_integrals;

        utils::Orbitals occ_orbitals;
        utils::Orbitals active_orbitals;

        std::shared_ptr<utils::MatVec> one_trans;
        std::shared_ptr<utils::MatVec> one_trans_ri;
        std::shared_ptr<utils::MatVec> two_trans;

        std::shared_ptr<std::mutex> one_trans_lock;
        std::shared_ptr<std::mutex> one_trans_ri_lock;
        std::shared_ptr<std::mutex> two_trans_lock;

        bool store_one_trans;
        bool store_one_trans_ri;
    };
} // namespace uw12::integrals

#endif  // UW12_INTEGRALS_HPP
