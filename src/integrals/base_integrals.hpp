#ifndef UW12_BASE_INTEGRALS_HPP
#define UW12_BASE_INTEGRALS_HPP

#include <memory>
#include <mutex>

#include "integral_functions.hpp"

namespace uw12::integrals {
    /// \brief Class for calculating and storing ao/ri integrals for density-fitted
    /// UW12 calculations
    ///
    /// Density-fitting a two-electron integral is usually performed using:
    /// \f[
    /// (\mu\nu|\rho\sigma) = \sum_{AB} J_{\mu\nu, A}^{0} [ J_2^{-1} ]_{A,B}
    /// J_{B, \rho\sigma}^{0}
    /// \f]
    /// where \f$J_{\mu\nu, A}^{0} = (\mu\nu | A)\f$ are the three-index integrals
    /// in the space of the density-fitting basis `A`.
    ///
    /// For numerical stability, the two-index integrals in the UW12 approximation
    /// are decomposed as:
    /// \f[
    /// J_2^{-1} &= J_2^{-1} J_2 J_2^{-1} = J_2^{-1} Q \Lambda Q^{T} J_2^{-1}
    /// \f]
    /// where \f$\Lambda, Q\f$ are the matrices of eigenvalues and eigenvectors of
    /// \f$J_2\f$.
    /// By setting \f$P_2 = J_2^{-1} Q = Q \Lambda^{-1}\f$. The two-electron
    /// integrals may be evaluated as:
    /// \f[
    /// (\mu\nu | \rho\sigma) = \sum_{A} J_{\mu\nu}^A \Lambda_{A} J_{\rho\sigma}^B
    /// \f]
    /// where \f$J_{\mu\nu}^A = \sum_B J_{\mu\nu, B}^0 P_{BA}\f$ are the three-index
    /// integrals in the eigen-space of `A`.
    ///
    /// This class can be used to calculate and store the integrals
    /// `J3` (\f$J_{\mu\nu}^{A}\f$), `P2` (\f$J_2^{-1} Q\f$), and the eigenvalues
    /// `df_vals` (\f$\Lambda_A\f$).
    /// In addition, if using the density-fitted RI approach, the RI analogue of
    /// `J3` may also be accessed.
    ///
    /// If the original `J3_0` integrals are given, these will be stored and used to
    /// calculate the transformed `J3` integrals when requested.
    ///
    /// Unless supplied, no three-index integrals are stored unless requested.
    class BaseIntegrals {
    public:
        /// Default constructor
        BaseIntegrals();

        /// Standard constructor using functions to calculate integrals
        BaseIntegrals(
            const TwoIndexFn &two_index_fn,
            const ThreeIndexFn &three_index_fn,
            const ThreeIndexFn &three_index_ri_fn,
            const std::vector<size_t> &df_sizes,
            size_t n_ao,
            size_t n_df,
            size_t n_ri,
            bool store_ao_integrals = true,
            bool store_ri_integrals = true
        );

        /// Wrapper for above constructor with no ri
        BaseIntegrals(
            const TwoIndexFn &two_index_fn,
            const ThreeIndexFn &three_index_fn,
            const std::vector<size_t> &df_sizes,
            size_t n_ao,
            size_t n_df,
            bool store_ao_integrals = true
        );


        /// Constructor used if the standard three-index integrals `J3_0` and the
        /// two-index density-fitting integrals are already calculated. In this case,
        /// `J2` is immediately decomposed into `P2` and `df_vals`, and `J3_0` is
        /// stored.
        BaseIntegrals(
            const linalg::Mat &J3_0,
            const linalg::Mat &J2,
            const ThreeIndexFn &three_index_ri,
            const std::vector<size_t> &df_sizes,
            bool use_ri,
            size_t n_ao,
            size_t n_df,
            size_t n_ri,
            bool store_ri_integrals = false
        );

        /// Constructor used if the standard three-index integrals `J3_0` and the
        /// two-index density-fitting integrals are already calculated. As well as the ri integrals
        /// J3_ri_0. In this case, `J2` is immediately decomposed into `P2` and `df_vals`, and `J3_0` is
        /// stored.
        BaseIntegrals(const linalg::Mat &J3_0, const linalg::Mat &J2,
                      const linalg::Mat &J3_ri_0);

        /// Wrapper for previous constructor without ri
        BaseIntegrals(const linalg::Mat &J3_0, const linalg::Mat &J2);

        /// Constructor if the three-index integrals in the density-fitting eigenspace
        /// are already available. Since no basis sets or projection matrix `P2` can
        /// be specified, this constructor cannot be used for RI.
        BaseIntegrals(const linalg::Mat &J3, const linalg::Vec &df_vals);

        /// Call the two_index_fn to return the two-index density fitting integrals
        /// \f$(A|w|B)\f$, the resulting square matrix should be of size `n_df * n_df`
        /// where `n_df` is the number of density_fitting basis functions
        [[nodiscard]] linalg::Mat two_index() const;

        /// \brief Call the three_index_fn to obtain the three index density-fitting
        /// integrals in the ao basis for the density-fitting basis shell index `A`
        ///
        /// The resulting matrix should have `n_ao * (n_ao + 1) / 2` rows and `nA`
        /// columns, where  `n_ao` is the number of ao basis functions and `nA` is the
        /// number of basis functions in density-fitting basis shell `A`.
        ///
        /// Each column contains the upper triangular elements of a real symmetric
        /// matrix of size `n_ao * n_ao`, the full matrix may be obtained using
        /// `linalg::square`.
        ///
        /// \param A Index of density-fitting shell
        ///
        /// \return Density-fitting integrals \f$(\rho\sigma | w | A)\f$
        [[nodiscard]] linalg::Mat three_index(size_t A) const;

        /// \brief Call the three_index_fn to obtain the three index density-fitting
        /// ri integrals in the ao basis and ri basis for the density-fitting basis
        /// shell index `A`
        ///
        /// The resulting matrix should have `n_ri * n_ao` rows and `nA` columns,
        /// where `n_ao` is the number of ao basis functions, `n_ri` is the number
        /// of ri basis functions and `nA` is the number of basis functions in
        /// density-fitting basis shell `A`.
        ///
        /// Each column contains a matrix of size `n_ri * n_ao`.
        ///
        /// \param A Index of density-fitting shell
        ///
        /// \return Density-fitting ri integrals \f$(\mu\rho | w | A)\f$
        [[nodiscard]] linalg::Mat three_index_ri(size_t A) const;

        /// Check whether a `two_index_fn` is provided
        [[nodiscard]] bool has_two_index_fn() const;

        /// Check whether a `three_index_fn` is provided
        [[nodiscard]] bool has_three_index_fn() const;

        /// Check whether a `three_index_ri_fn` is provided
        [[nodiscard]] bool has_three_index_ri_fn() const;

        /// Return the projection matrix `P2`. If not already calculated, the
        /// two-index matrix `J2` is calculated and decomposed into `P2` and
        /// `df_vals`.
        [[nodiscard]] const linalg::Mat &get_P2() const;

        /// Obtain the vector of density-fitting basis set shell sizes.
        [[nodiscard]] const std::vector<size_t> &get_df_sizes() const;

        /// Obtain the vector of density-fitting basis set shell offsets. This vector
        /// is determined using `df_sizes` and is of length `n_sh` for number of df
        /// basis set shells. Th vector determines the first column index in the final
        /// `J3` matrix where the first column of a given df shell is assigned.
        [[nodiscard]] std::vector<size_t> get_df_offsets() const;

        /// Return the density-fitting eigenvalues `df_vals`. If not already
        /// calculated, the two-index matrix `J2` is calculated and decomposed into
        /// `P2` and `df_vals`.
        [[nodiscard]] const linalg::Vec &get_df_vals() const;

        /// Return the three-index integrals `J3` in the original space of
        /// density-fitting function. If stored, a reference to the shared pointer is
        /// returned, otherwise an error is thrown.
        [[nodiscard]] const linalg::Mat &get_J3_0() const;

        /// Return the three-index integrals `J3` in the space of density-fitting
        /// eigenvalues. If stored, a reference to the shared pointer is returned.
        /// If not stored, the integrals are calculated. If `J3_0` is stored, this
        /// is transformed to the space of `df_vals`.
        [[nodiscard]] const linalg::Mat &get_J3() const;

        /// Return the three-index integrals `J3_ri` in the original space of
        /// density-fitting functions. If stored, a reference to the shared pointer is
        /// returned, otherwise an error is thrown.
        [[nodiscard]] const linalg::Mat &get_J3_ri_0() const;

        /// Return the three-index ri integrals `J3_ri` in the space of
        /// density-fitting eigenvalues. If stored, a reference to the shared pointer
        /// is returned. If not stored, the integrals are calculated and stored.
        ///
        /// The size of the `ri` basis can be large, storing these integrals should
        /// be avoided for large systems.
        [[nodiscard]] const linalg::Mat &get_J3_ri() const;

        /// Get the number of ao basis functions
        [[nodiscard]] size_t get_number_ao() const;

        /// Get the number of df basis functions
        [[nodiscard]] size_t get_number_df() const;

        /// Get the number of ri basis functions
        [[nodiscard]] size_t get_number_ri() const;

        /// Check whether `ri` integrals should be stored. If false a warning is
        /// given when calling `get_J3_ri`
        [[nodiscard]] bool storing_ri() const;

        /// Check whether `ao` integrals should be stored. If false a warning is
        /// given when calling `get_J3_ao`.
        [[nodiscard]] bool storing_ao() const;

        /// Check whether `P2` is stored
        [[nodiscard]] bool has_P2() const;

        /// Check whether `df_vals` is stored
        [[nodiscard]] bool has_df_vals() const;

        /// Check whether the standard density-fitting integrals in the space of
        /// the density-fitting basis set.
        [[nodiscard]] bool has_J3_0() const;

        /// Check whether `J3` is stored
        [[nodiscard]] bool has_J3() const;

        /// Check whether `J3_ri` is stored
        [[nodiscard]] bool has_J3_ri() const;

        /// Check whether `J3_ri_0` is stored
        [[nodiscard]] bool has_J3_ri_0() const;

    private:
        BaseIntegrals(
            TwoIndexFn two_index_fn_,
            ThreeIndexFn three_index_fn_,
            ThreeIndexFn three_index_ri_fn_,
            const std::vector<size_t> & df_sizes_,
            size_t n_ao_,
            size_t n_df_,
            size_t n_ri_,
            bool store_ao_integrals_,
            bool store_ri_integrals_,
            bool two_index_fn_provided_,
            bool three_index_fn_provided_,
            bool three_index_ri_fn_provided_,
            const std::shared_ptr<linalg::Mat> &J3_0_ = nullptr,
            const std::shared_ptr<linalg::Mat> &J3_ = nullptr,
            const std::shared_ptr<linalg::Mat> &J2_ = nullptr,
            const std::shared_ptr<linalg::Vec> &df_vals_ = nullptr,
            const std::shared_ptr<linalg::Mat> &J3_ri_0_ = nullptr
        );

        void set_J2_values() const;

        TwoIndexFn two_index_fn;
        ThreeIndexFn three_index_fn;
        ThreeIndexFn three_index_fn_ri;

        size_t n_ao{};
        size_t n_df{};
        size_t n_ri{};

        /// Vector of df basis set shell sizes where the length is the number of df
        /// basis set shells
        std::vector<size_t> df_sizes;

        /// Three-index integrals in the original df basis space
        std::shared_ptr<linalg::Mat> J3_0;
        /// Three-index integrals in the space of `df_vals`
        std::shared_ptr<linalg::Mat> J3;
        /// Three-index ri integrals in the original basis space
        std::shared_ptr<linalg::Mat> J3_ri_0;
        /// Three-index ri integrals in the space of `df_vals`
        std::shared_ptr<linalg::Mat> J3_ri;
        /// Projector from original df basis space to space of `df_vals`
        std::shared_ptr<linalg::Mat> P2;
        /// Density-fitting eigenvalues
        std::shared_ptr<linalg::Vec> df_vals;

        /// Locks to prevent parallel calculation of matrices
        std::shared_ptr<std::mutex> J3_lock;
        std::shared_ptr<std::mutex> J3_ri_lock;
        std::shared_ptr<std::mutex> eig_lock;

        /// Booleans check that integral functions have been provided
        bool two_index_fn_provided{};
        bool three_index_fn_provided{};
        bool three_index_ri_fn_provided{};

        /// Whether storing ao integrals is expected
        bool store_ao_integrals{};
        /// Whether storing ri integrals is expected
        bool store_ri_integrals{};
    };
} // namespace uw12::integrals

#endif // UW12_BASE_INTEGRALS_HPP
