//
// Created by Zack Williams on 27/11/2020.
//

#ifndef UW12_LINALG_HPP
#define UW12_LINALG_HPP

#ifdef USE_ARMA
#include <armadillo>
#elif USE_EIGEN
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
#endif
#include <cassert>
#include <cmath>
#include <vector>

namespace uw12::linalg {
#ifdef USE_ARMA
/// Matrix object in column major ordering
using Mat = arma::mat;

/// Column vector (vec)
using Vec = arma::vec;
#elif USE_EIGEN
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
#endif

inline size_t n_elem(const Vec &vec) {
#ifdef USE_ARMA
  return vec.n_elem;
#elif USE_EIGEN
  return vec.size();
#endif
}

/// Calculate the number of elements in a matrix
inline size_t n_elem(const Mat &mat) {
#ifdef USE_ARMA
  return mat.n_elem;
#elif USE_EIGEN
  return mat.size();
#endif
}

/// Calculate the number of rows in a matrix
inline size_t n_rows(const Mat &mat) {
#ifdef USE_ARMA
  return mat.n_rows;
#elif USE_EIGEN
  return mat.rows();
#endif
}

/// Calculate the number of columns in a matrix
inline size_t n_cols(const Mat &mat) {
#ifdef USE_ARMA
  return mat.n_cols;
#elif USE_EIGEN
  return mat.cols();
#endif
}

/// Initialise a vec of size `n_el`
inline Vec vec(const size_t n_el) {
#ifdef USE_ARMA
  return arma::vec(n_el);
#elif USE_EIGEN
  return Vec(n_el);
#endif
}

/// Initialise a vec from a std::vector<double>
inline Vec vec(const std::vector<double> &vector) {
#ifdef USE_ARMA
  return vector;
#elif USE_EIGEN
  const auto ptr = const_cast<double *>(vector.data());
  const auto n = vector.size();

  return Eigen::Map<Vec>(ptr, n);
#endif
}

/// Initialise a vec of ones of size `n_el`
inline Vec ones(const size_t n_el) {
#ifdef USE_ARMA
  return arma::ones(n_el);
#elif USE_EIGEN
  return Vec::Ones(n_el);
#endif
}

/// Initialise a vec of zeros of size `n_el`
inline Vec zeros(const size_t n_el) {
#ifdef USE_ARMA
  return arma::zeros(n_el);
#elif USE_EIGEN
  return Vec::Zero(n_el);
#endif
}

inline double elem(const Vec &vec, const size_t index) {
#ifdef USE_ARMA
  return vec(index);
#elif USE_EIGEN
  return vec(index);
#endif
}

/// \brief Initialise a `Mat` object given a memory location and the size
///
/// Initialise a matrix of size `n_row * n_col` given these values and a
/// pointer to the first entry of the final matrix.
///
/// No checking is done by the function to check whether the is an
/// `n_row * n_col` array at the given location so ensure memory is assigned
/// before using this function.
/// TODO: Require memory checks whenever this function is called
///
/// \param mem Pointer to first entry in array
/// \param n_row Number of rows in final matrix
/// \param n_col Number of columns in final matrix
/// \param copy Whether to copy memory or construct matrix pointing to original
///             memory
///
/// \return Mat object of size `n_row * n_col`
inline Mat mat(
    double *mem, const size_t n_row, const size_t n_col, const bool copy = false
) {
#ifdef USE_ARMA
  return {mem, n_row, n_col, true, true};  // CheckMem
#elif USE_EIGEN
  return Eigen::Map<Mat>(mem, n_row, n_col);
#endif
}

/// \brief Returns the (`row_index`, `col_index`) element of `mat`. (For
/// testing) \param mat Matrix \param row_index \param col_index \return Element
/// in position (`row_index`, `col_index`)
inline double elem(
    const Mat &mat, const size_t row_index, const size_t col_index
) {
#ifdef USE_ARMA
  return mat(row_index, col_index);
#elif USE_EIGEN
  return mat(row_index, col_index);
#endif
}

/// \brief Sets the (`row_index`, `col_index`) element of `mat` to `value`. (For
/// testing)
/// \param mat Matrix
/// \param row_index
/// \param col_index
/// \param value
inline void set_elem(
    Mat &mat, const size_t row_index, const size_t col_index, const double value
) {
  if (row_index >= n_rows(mat) || col_index >= n_cols(mat)) {
    throw std::logic_error("Index outside scope of matrix");
  }

  mat(row_index, col_index) = value;
}

/// \brief Sets the `index` element of `vec` to `value`. (For testing)
/// \param vec vec
/// \param index
/// \param value
inline void set_elem(Vec &vec, const size_t index, const double value) {
  set_elem(vec, index, 0, value);
}

/// Initialise a matrix of size `n_row` by `n_col`
inline Mat mat(const size_t n_row, const size_t n_col) {
#ifdef USE_ARMA
  return arma::mat(n_row, n_col);
#elif USE_EIGEN
  return Mat(n_row, n_col);
#endif
}

/// Initialise a matrix of ones size `n_row` by `n_col`
inline Mat ones(const size_t n_row, const size_t n_col) {
#ifdef USE_ARMA
  return arma::ones(n_row, n_col);
#elif USE_EIGEN
  return Mat::Ones(n_row, n_col);
#endif
}

/// Initialise a matrix of zeros of size `n_row` by `n_col`
inline Mat zeros(const size_t n_row, const size_t n_col) {
#ifdef USE_ARMA
  return arma::zeros(n_row, n_col);
#elif USE_EIGEN
  return Mat::Zero(n_row, n_col);
#endif
}

/// Generate an identity matrix of size `n` by `n`
inline Mat id(const size_t n) {
#ifdef USE_ARMA
  return arma::eye(n, n);
#elif USE_EIGEN
  return Mat::Identity(n, n);
#endif
}

/// Compute the transpose of a matrix
inline Mat transpose(const Mat &mat) {
#ifdef USE_ARMA
  return mat.t();
#elif USE_EIGEN
  return mat.transpose();
#endif
}

/// Generate a random matrix of size `n_row * n_col`
inline Mat random(const size_t n_row, const size_t n_col, const int seed) {
#ifdef USE_ARMA
  arma::arma_rng::set_seed(seed);
  return arma::randu(n_row, n_col);
#elif USE_EIGEN
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(0, 1);
  const auto uniform = [&] { return distribution(generator); };

  const Mat mat = Mat::NullaryExpr(n_row, n_col, uniform);

  return mat;
#endif
}

/// Check whether a matrix is square
inline bool is_square(const Mat &mat) {
#ifdef USE_ARMA
  return mat.is_square();
#elif USE_EIGEN
  const auto n_row = n_rows(mat);
  const auto n_col = n_cols(mat);

  return n_row == n_col;
#endif
}

/// Check whether a matrix is symmetric to within a threshold
inline bool is_symmetric(const Mat &mat, const double threshold = 1e-10) {
#ifdef USE_ARMA
  return mat.is_symmetric(threshold);
#elif USE_EIGEN
  const auto n = n_rows(mat);
  if (n != n_cols(mat)) {
    return false;
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < i; j++) {
      if (std::abs(mat(i, j) - mat(j, i)) > threshold) {
        return false;
      }
    }
  }

  return true;
#endif
}

/// Check whether all elements of a vec are non-negative
inline bool all_positive(const Vec &vec) {
#ifdef USE_ARMA
  return all(vec > 0);
#elif USE_EIGEN
  return (vec.array() > 0).all();
#endif
}

/// Element-wise product of two vectors (Schur-product)
inline Vec schur(const Vec &vec1, const Vec &vec2) {
  if (n_elem(vec1) != n_elem(vec2)) {
    throw std::logic_error(
        "Cannot calculate Schur product for vectors of different sizes"
    );
  }
#ifdef USE_ARMA
  return vec1 % vec2;
#elif USE_EIGEN
  return vec1.cwiseProduct(vec2);
#endif
}

/// Matrix dot product
inline double dot(const Mat &mat1, const Mat &mat2) {
  const auto n_col = n_cols(mat1);
  if (n_col != n_cols(mat2)) {
    throw std::logic_error("Matrices must have same number of columns");
  }

  if (n_rows(mat1) != n_rows(mat2)) {
    throw std::logic_error("Matrices must have same number of rows");
  }
#ifdef USE_ARMA
  return arma::dot(mat1, mat2);
#elif USE_EIGEN
  double result = 0;
  for (size_t i = 0; i < n_col; ++i) {
    result += mat1.col(i).dot(mat2.col(i));
  }

  return result;
#endif
}

/// \brief Reshape an object into a matrix of size `n_row * n_col`
///
/// \param mat Input object (must have `n_row * n_col` elements)
/// \param n_row Number of rows of output matrix
/// \param n_col Number of columns of output matrix
/// \param copy_data Whether to copy the matrix data or point to original data
///
/// \return Output matrix of size `n_row * n_col`
inline Mat reshape(
    const Mat &mat,
    const size_t n_row,
    const size_t n_col,
    const bool copy_data = false
) {
  if (n_row * n_col != n_elem(mat)) {
    throw std::logic_error(
        "Reshaping a matrix must preserve total number of elements"
    );
  }
#ifdef USE_ARMA
  constexpr bool strict = true;

  return {
      const_cast<double *>(mat.memptr()), n_row, n_col, copy_data, strict
  };  // CheckMem

#elif USE_EIGEN
  if (!copy_data) {
    return Eigen::Map<Mat>(const_cast<double *>(mat.data()), n_row, n_col);
  } else {
    auto mat2 = mat;

    mat2.resize(n_row, n_col);

    return mat2;
  }
#endif
}

/// \brief Reshape a column of a matrix to a new matrix
///
/// \param mat Matrix containing chosen column (must have `n_row * n_col`
/// rows) \param col_idx Index of chosen column (must be a column of matrix
/// `mat`) \param n_row Number of rows of output matrix \param n_col Number of
/// columns of output matrix \param copy_data Whether to copy the data
///
/// \return Matrix of size `n_row * n_col`
inline Mat reshape_col(
    const Mat &mat,
    const size_t col_idx,
    const size_t n_row,
    const size_t n_col,
    const bool copy_data = false
) {
  if (n_row * n_col != n_rows(mat)) {
    throw std::logic_error(
        "Reshaped column of a matrix must preserve number of elements"
    );
  }
  if (col_idx >= n_cols(mat)) {
    throw std::logic_error("Column index must be in matrix");
  }
#ifdef USE_ARMA
  constexpr bool strict = true;

  return {
      const_cast<double *>(mat.colptr(col_idx)), n_row, n_col, copy_data, strict
  };  // CheckMem
#elif USE_EIGEN
  if (!copy_data) {
    return Eigen::Map<Mat>(
        const_cast<double *>(mat.col(col_idx).data()), n_row, n_col
    );
  } else {
    Mat out = mat.col(col_idx);
    out.resize(n_row, n_col);
    return out;
  }
#endif
}

/// Calculate the pseudo-inverse of matrix `mat` using the given `threshold`.
inline Mat p_inv(const Mat &mat, const double threshold = 1e-10) {
#ifdef USE_ARMA
  return pinv(mat, threshold);
#elif USE_EIGEN
  auto decomposition = mat.completeOrthogonalDecomposition();

  decomposition.setThreshold(threshold);

  return decomposition.pseudoInverse();
#endif
}

/// Generate a random positive-definite square matrix of size `n_row * n_row`
inline Mat random_pd(const size_t n_row, const int seed) {
  const Vec eigen = random(n_row, 1, seed) + ones(n_row);
  Mat X = random(n_row, n_row, seed);

#ifdef USE_ARMA
  X = 0.5 * (X - X.t());

  const Mat U = expmat(X);
  return U * diagmat(eigen) * U.t();
#elif USE_EIGEN
  X = 0.5 * (X - X.transpose());

  const Mat U = Eigen::exp(X.array());

  return U * eigen.asDiagonal() * U.transpose();
#endif
}

/// Calculate the inverse of a symmetric positive definite matrix
inline Mat inv_sym_pd(const Mat &mat) {
#ifdef USE_ARMA
  return inv_sympd(mat);
#elif USE_EIGEN
  if (!is_square(mat)) {
    throw std::logic_error("Matrix is not square");
  }
  return mat.inverse();
#endif
}

/// \brief Output a contiguous sub-vector of the input vector
///
/// \param vec Input vector
/// \param row1 First row of sub-vector
/// \param n_row Number of rows of sub-vector
///
/// \return Sub-vector of size `n_row`
inline Vec sub_vec(const Vec &vec, const size_t row1, const size_t n_row) {
  if (n_row > n_rows(vec)) {
    throw std::logic_error("sub vec cannot be larger than parent vec");
  }
  if (row1 >= n_rows(vec)) {
    throw std::logic_error("starting index cannot be outside parent vector");
  }
  if (row1 + n_row > n_rows(vec)) {
    throw std::logic_error("Final index outside of parent vector range");
  }

#ifdef USE_ARMA
  return vec.subvec(row1, row1 + n_row - 1);
#elif USE_EIGEN
  return vec.segment(row1, n_row);
#endif
}

/// \brief Output a sub-matrix of input matrix `mat`
///
/// \param mat Input matrix
/// \param row1 Index of first row of sub-matrix
/// \param col1 Index of first column of sub-matrix
/// \param n_row Number of rows of sub-matrix
/// \param n_col Number of columns of sub-matrix
///
/// \return Sub-matrix of size `n_row * n_col`
inline Mat sub_mat(
    const Mat &mat,
    const size_t row1,
    const size_t col1,
    const size_t n_row,
    const size_t n_col
) {
  if (row1 >= n_rows(mat)) {
    throw std::logic_error("Start row not in parent matrix");
  }
  if (col1 >= n_cols(mat)) {
    throw std::logic_error("Start column not in parent matrix");
  }
  if (row1 + n_row > n_rows(mat)) {
    throw std::logic_error("submatrix row index must be in parent index");
  }
  if (col1 + n_col > n_cols(mat)) {
    throw std::logic_error("submatrix column index must be in parent index");
  }
#ifdef USE_ARMA
  return mat.submat(row1, col1, arma::size(n_row, n_col));
#elif USE_EIGEN
  return mat.array().block(row1, col1, n_row, n_col);
#endif
}

/// Calculate the norm a matrix
inline double norm(const Mat &mat) {
#ifdef USE_ARMA
  return arma::norm(mat);
#elif USE_EIGEN
  const Eigen::JacobiSVD svd(mat);

  return svd.singularValues().array().abs().maxCoeff();
#endif
}

/// \brief Get a column of input matrix `mat`
///
/// \param mat Input matrix
/// \param col_idx Index of column
/// \param copy_data Whether to copy the column data
///
/// \return Column vector corresponding to `col_idx` column of matrix `mat`
inline Vec col(
    const Mat &mat, const size_t col_idx, const bool copy_data = false
) {
  if (col_idx >= n_cols(mat)) {
    throw std::logic_error("column index must be in parent matrix");
  }
#ifdef USE_ARMA
  const auto n_row = n_rows(mat);

  return reshape_col(mat, col_idx, n_row, 1, copy_data);
#elif USE_EIGEN
  return mat.col(col_idx);
#endif
}

/// \brief Get a row of input matrix `mat`
///
/// \param mat Input matrix
/// \param row_idx Index of row in `mat`
///
/// \return Matrix of size `1 * n_col` corresponding to `row_idx` row of `mat`
inline Mat row(const Mat &mat, const size_t row_idx) {
  if (row_idx >= n_rows(mat)) {
    throw std::logic_error("row index must be in parent matrix");
  }
#ifdef USE_ARMA
  return mat.row(row_idx);
#elif USE_EIGEN
  return mat.row(row_idx);
#endif
}

/// \brief Get sub-matrix of multiple rows of input matrix
///
/// \param mat Input matrix
/// \param row_idx Index of first row
/// \param n_row Number of rows in sub-matrix
///
/// \return sub-matrix of the `n_row` rows of `mat`
inline Mat rows(const Mat &mat, const size_t row_idx, const size_t n_row) {
  const auto n_col = n_cols(mat);

  return sub_mat(mat, row_idx, 0, n_row, n_col);
}

/// \brief Get first `n_row` of vector `vec`
///
/// \param vec Input vector
/// \param n_row Number of rows in sub-vector
///
/// \return Sub-vector of first `n_row` of input vector
inline Vec head(const Vec &vec, const size_t n_row) {
  if (n_row > n_elem(vec)) {
    throw std::logic_error(
        "Cannot take more head rows from vector than total number of rows"
    );
  }
#ifdef USE_ARMA
  return vec.head(n_row);
#elif USE_EIGEN
  return vec.head(n_row);
#endif
}

/// \brief Get last `n_row` of vector `vec`
///
/// \param vec Input vector
/// \param n_row Number of rows in sub-vector
///
/// \return Sub-vector of last `n_row` of input vector
inline Vec tail(const Vec &vec, const size_t n_row) {
  if (n_row > n_elem(vec)) {
    throw std::logic_error(
        "Cannot take more tail rows from vector than total number of rows"
    );
  }
#ifdef USE_ARMA
  return vec.tail(n_row);
#elif USE_EIGEN
  return vec.tail(n_row);
#endif
}

/// \brief Get sub-matrix of the first `n_col` of input matrix
///
/// \param mat Input matrix
/// \param n_col Number of columns in sub-matrix
/// \param copy_data Whether to copy dat to make new matrix
///
/// \return Sub-matrix of first `n_col` of `mat`
inline Mat head_cols(
    const Mat &mat, const size_t n_col, const bool copy_data = false
) {
  const auto n_row = n_rows(mat);
  if (n_col > n_cols(mat)) {
    throw std::logic_error(
        "Cannot have submatix with more columns than parent matrix"
    );
  }

#ifdef USE_ARMA
  constexpr auto strict = true;

  return {const_cast<double *>(mat.memptr()), n_row, n_col, copy_data, strict};

#elif USE_EIGEN
  if (!copy_data) {
    return Eigen::Map<Mat>(const_cast<double *>(mat.data()), n_row, n_col);
  } else {
    return mat.leftCols(n_col);
  }
#endif
}

/// \brief Get sub-matrix of the last `n_col` of input matrix
///
/// \param mat Input matrix
/// \param n_col Number of columns in sub-matrix
/// \param copy_data Whether to copy dat to make new matrix
///
/// \return Sub-matrix of last `n_col` of `mat`
inline Mat tail_cols(
    const Mat &mat, const size_t n_col, const bool copy_data = false
) {
  if (n_col > n_cols(mat)) {
    throw std::logic_error(
        "Cannot have submatix with more columns than parent matrix"
    );
  }

  const auto n_row = n_rows(mat);
  const auto n_col_all = n_cols(mat);

  assert(n_col_all >= n_col);
  const auto col1 = n_col_all - n_col;

#ifdef USE_ARMA
  constexpr auto strict = true;

  return {
      const_cast<double *>(mat.colptr(col1)), n_row, n_col, copy_data, strict
  };

#elif USE_EIGEN
  if (!copy_data) {
    return Eigen::Map<Mat>(
        const_cast<double *>(mat.col(col1).data()), n_row, n_col
    );
  } else {
    return mat.rightCols(n_col);
  }
#endif
}

/// \brief Get sub-matrix of the first `n_row` of input matrix
///
/// \param mat Input matrix
/// \param n_row Number of row in sub-matrix
///
/// \return Sub-matrix of first `n_row` of `mat`
inline Mat head_rows(const Mat &mat, const size_t n_row) {
  if (n_row > n_rows(mat)) {
    throw std::logic_error(
        "Cannot have submatix with more rows than parent matrix"
    );
  }
#ifdef USE_ARMA
  return mat.head_rows(n_row);
#elif USE_EIGEN
  const auto n_col = n_cols(mat);

  return sub_mat(mat, 0, 0, n_row, n_col);
#endif
}

/// \brief Get sub-matrix of the last `n_row` of input matrix
///
/// \param mat Input matrix
/// \param n_row Number of row in sub-matrix
///
/// \return Sub-matrix of last `n_row` of `mat`
inline Mat tail_rows(const Mat &mat, const size_t n_row) {
  if (n_row > n_rows(mat)) {
    throw std::logic_error(
        "Cannot have submatix with more rows than parent matrix"
    );
  }
#ifdef USE_ARMA
  return mat.tail_rows(n_row);
#elif USE_EIGEN
  const auto n_col = n_cols(mat);
  const auto n_row_all = n_rows(mat);

  assert(n_row_all >= n_row);
  const auto row1 = n_row_all - n_row;

  return sub_mat(mat, row1, 0, n_row, n_col);
#endif
}

/// Convert an `n_row * n_col` matrix into a vector of length `n_row *
/// n_col`
inline Vec vectorise(const Mat &mat) {
#ifdef USE_ARMA
  return arma::vectorise(mat);
#elif USE_EIGEN
  return Eigen::Map<Vec>(const_cast<double *>(mat.data()), n_elem(mat));
#endif
}

/// Compute the trace of matrix `mat`
inline double trace(const Mat &mat) {
  if (n_rows(mat) != n_cols(mat)) {
    throw std::logic_error("Matrix not square");
  }
#ifdef USE_ARMA
  return arma::trace(mat);
#elif USE_EIGEN
  return mat.trace();
#endif
}

/// Generate a diagonal matrix of size `n_el * n_el` from a vector of size
/// `n_el`
inline Mat diagmat(const Vec &vec) {
#ifdef USE_ARMA
  return arma::diagmat(vec);
#elif USE_EIGEN
  return vec.asDiagonal();
#endif
}

/// Calculate the element-wise square-root of the input object
inline Mat sqrt(const Mat &mat) {
  for (size_t i = 0; i < n_cols(mat); ++i) {
    const auto col = linalg::col(mat, i);
    if (!all_positive(col)) {
      throw std::logic_error("Cannot compute square root of negative elements");
    }
  }
#ifdef USE_ARMA
  return arma::sqrt(mat);
#elif USE_EIGEN
  return mat.array().sqrt();
#endif
}

/// \brief Assign a matrix to columns of matrix `mat`
///
/// \param mat Matrix for column assignment (non-const)
/// \param input Input matrix (must have same number of rows as `mat`)
/// \param offset First column index for assignment
inline void assign_cols(Mat &mat, const Mat &input, const size_t offset) {
  const auto n = n_cols(input);
  if (offset >= n_cols(mat)) {
    throw std::logic_error("offset greater than number of columns of mat");
  }
  if (n_rows(mat) != n_rows(input)) {
    throw std::logic_error("Matrices have different number of rows");
  }
  if (n_cols(mat) < offset + n) {
    throw std::logic_error("Columns outside boundary of matrix");
  }
  assert(n_cols(mat) >= offset + n - 1);
  assert(n_rows(mat) == n_rows(input));

#ifdef USE_ARMA
  mat.cols(offset, offset + n - 1) = input;
#elif USE_EIGEN
  for (size_t i = 0; i < n; ++i) {
    mat.col(offset + i) = input.col(i);
  }
#endif
}

/// \brief Assign a matrix to rows of matrix `mat`
///
/// \param mat Matrix for row assignment (non-const)
/// \param input Input matrix (must have same number of cols as `mat`)
/// \param offset First rows index for assignment
inline void assign_rows(Mat &mat, const Mat &input, const size_t offset) {
  const auto n = n_rows(input);
  if (offset >= n_rows(mat)) {
    throw std::logic_error("offset greater than number of rows of mat");
  }
  if (n_cols(mat) != n_cols(input)) {
    throw std::logic_error("Matrices have different number of columns");
  }
  if (n_rows(mat) < offset + n) {
    throw std::logic_error("Rows outside boundary of matrix");
  }

#ifdef USE_ARMA
  mat.rows(offset, offset + n - 1) = input;
#elif USE_EIGEN
  for (size_t i = 0; i < n; ++i) {
    mat.row(offset + i) = input.row(i);
  }
#endif
}

/// \brief Assign a matrix to rows of vector `vec`
///
/// \param vec Vector for row assignment (non-const)
/// \param input Input vector
/// \param offset First index for assignment
inline void assign_rows(Vec &vec, const Vec &input, const size_t offset) {
  const auto n = n_elem(input);
  if (offset >= n_elem(vec)) {
    throw std::logic_error("offset greater thansize of vec");
  }
  if (n_elem(vec) < offset + n) {
    throw std::logic_error("assignment outside boundary of vec");
  }

#ifdef USE_ARMA
  vec.rows(offset, offset + n - 1) = input;
#elif USE_EIGEN
  for (size_t i = 0; i < n; ++i) {
    vec(offset + i) = input(i);
  }
#endif
}

/// Obtain a pointer to first element of vector
inline const double *mem_ptr(const Vec &vec) {
#ifdef USE_ARMA
  return const_cast<double *>(vec.memptr());
#elif USE_EIGEN
  return vec.data();
#endif
}

/// Obtain a pointer to first element of matrix
inline const double *mem_ptr(const Mat &mat) {
#ifdef USE_ARMA
  return const_cast<double *>(mat.memptr());
#elif USE_EIGEN
  return mat.data();
#endif
}

/// Determine whether a vector is empty
inline bool empty(const Vec &vec) {
#ifdef USE_ARMA
  return vec.empty();
#elif USE_EIGEN
  return (n_elem(vec) == 0);
#endif
}

/// Determine whether a matrix is empty
inline bool empty(const Mat &mat) {
#ifdef USE_ARMA
  return mat.empty();
#elif USE_EIGEN
  return (n_elem(mat) == 0);
#endif
}

inline double max_abs(const Vec &vec) {
#ifdef USE_ARMA
  return max(arma::abs(vec));
#elif USE_EIGEN
  return vec.array().abs().maxCoeff();
#endif
}

/// Check whether two matrices are equal to within a tolerance `epsilon`
inline bool nearly_equal(
    const Mat &mat1, const Mat &mat2, const double epsilon
) {
#ifdef USE_ARMA
  return approx_equal(mat1, mat2, "absdiff", epsilon);
#elif USE_EIGEN
  return mat1.array().isApprox(mat2.array(), epsilon);
#endif
}

/// \brief Multiply each column in a matrix by a column vector element-wise
///
/// For each column in matrix `mat` multiply element-wise by vector `vec`
///
/// \param mat Matrix of size `n_row * n_col`
/// \param vec Vector of length `n_row`
///
/// \return Matrix of size `n_row * n_col`
inline Mat each_col(const Mat &mat, const Vec &vec) {
  if (n_elem(vec) != n_rows(mat)) {
    throw std::logic_error("size of vector does not match matrix");
  }
#ifdef USE_ARMA
  return mat.each_col() % vec;
#elif USE_EIGEN
  return mat.array().colwise() * vec.array();
#endif
}

/// Create a cube from two matrices
inline Mat join_matrices(const Mat &mat1, const Mat &mat2) {
  if (n_rows(mat1) != n_rows(mat2)) {
    throw std::logic_error("matrices do not have the same number of rows");
  }
#ifdef USE_ARMA
  return join_horiz(mat1, mat2);
#elif USE_EIGEN
  const auto n_row = n_rows(mat1);
  assert(n_rows(mat2) == n_row);

  const auto n_col_1 = n_cols(mat1);
  const auto n_col_2 = n_cols(mat2);

  Mat out(n_row, n_col_1 + n_col_2);

  out.leftCols(n_col_1) = mat1;
  out.rightCols(n_col_2) = mat2;

  return out;
#endif
}

/// Load a matrix from a csv file located at `file_path`
inline Mat load_csv(const std::string &filepath) {
#ifdef USE_ARMA
  Mat A;

  const bool load_ok = A.load(arma::csv_name(filepath));

  if (!load_ok) {
    throw std::runtime_error("cannot read data from file");
  }

  return A;
#elif USE_EIGEN
  std::ifstream in_data;
  in_data.open(filepath);

  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(in_data, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Eigen::Map<Mat>(values.data(), values.size() / rows, rows).transpose();
#endif
}

/// \brief Eigen-decomposition of a symmetric/hermitian matrix
///
/// Calculate the eigenvalues and eigenvectors of an `n * n` matrix. Results
/// are returned as a std::pair<Vec, Mat> containing a vector of `n` eigenvalues
/// and a matrix of `n` eigenvectors in each column.
///
/// \param mat Input matrix (must be square and symmetric)
///
/// \return Eigenvalues and eigenvectors of matrix
inline std::pair<Vec, Mat> eigen_system(const Mat &mat) {
  if (!is_square(mat)) {
    throw std::logic_error("matrix must be square");
  }
  if (!is_symmetric(mat)) {
    throw std::logic_error("matrix must be symmetric");
  }

  Vec vals;
  Mat vecs;

#ifdef USE_ARMA
  const auto success = arma::eig_sym(vals, vecs, mat);
  if (!success) {
    throw std::runtime_error("eigen-decomposition failed");
  }
#elif USE_EIGEN
  Eigen::SelfAdjointEigenSolver<Mat> eigen_solver(mat);

  if (eigen_solver.info() != Eigen::Success) {
    throw std::runtime_error("eigen-decomposition failed");
  }

  vals = eigen_solver.eigenvalues();
  vecs = eigen_solver.eigenvectors();
#endif

  assert(n_cols(vecs) == n_elem(vals));
  assert(n_rows(vecs) == n_rows(mat));

  return {vals, vecs};
}

/// \brief Eigen-decomposition of a symmetric/hermitian matrix removing linear
/// dependent eigenvalues
///
/// \param matrix Input matrix
/// \param linear_dependency_threshold Absolute threshold for linear-dependence
/// \param eigen_ld_threshold Eigenvalue linear dependence threshold
/// (eigenvalues smaller than this threshold multiplied by the maximum
/// eigenvalue are removed)
///
/// \return Eigenvalues (as a vector) Eigenvectors as a matrix
inline std::pair<Vec, Mat> eigen_decomposition(
    const Mat &matrix,
    const double linear_dependency_threshold,
    const double eigen_ld_threshold
) {
  const auto [vals, vecs] = eigen_system(matrix);

  if (linear_dependency_threshold > 0 || eigen_ld_threshold > 0) {
    const auto max_eig = max_abs(vals);
    const auto threshold =
        std::max(max_eig * eigen_ld_threshold, linear_dependency_threshold);

    // Throw away vectors with zero eigenvalues
#ifdef USE_ARMA
    const arma::uvec non_zero = find(arma::abs(vals) > threshold);
    const arma::vec eigenvalues = vals.rows(non_zero);
    const arma::mat eigenvectors = vecs.cols(non_zero);
#elif USE_EIGEN
    const Eigen::VectorXi non_zero =
        (vals.array().abs() > threshold).cast<int>();

    const auto n = non_zero.sum();
    Vec eigenvalues(n);
    Mat eigenvectors(vecs.rows(), n);
    int idx = 0;
    for (int i = 0; i < vals.size(); ++i) {
      if (non_zero[i]) {
        eigenvalues(idx) = vals(i);
        eigenvectors.col(idx) = vecs.col(i);
        idx++;
      }
    }
#endif

    return {eigenvalues, eigenvectors};
  }

  return {vals, vecs};
}
}  // namespace uw12::linalg

#endif  // UW12_LINALG_HPP
