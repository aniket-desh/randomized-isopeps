#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>

namespace py = pybind11;

#include <iostream>
void printMatrix(double *A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[j * rows + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void setup(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n) {
  auto buf = A_input.request();
  double *A = static_cast<double *>(buf.ptr);

  buf = tau_input.request();
  double *tau = static_cast<double *>(buf.ptr);

  int lwork = -1;
  int info = 0;
  double work_size = 0.0;

  // Query the optimal workspace size
  LAPACK_dgeqrf(&m, &n, A, &m, nullptr, &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  std::vector<double> work(lwork);

  // Compute QR factorization
  LAPACK_dgeqrf(&m, &n, A, &m, tau, work.data(), &lwork, &info);

  // Invert triangular part
  char uplo = 'U';
  char diag = 'N';
  LAPACK_dtrtri(&uplo, &diag, &n, A, &m, &info);
}

void add_cols(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n, int k) {
  auto buf = A_input.request();
  double *A = static_cast<double *>(buf.ptr);
  double *new_data = A + m*n;

  buf = tau_input.request();
  double *tau = static_cast<double *>(buf.ptr);
  double *new_tau = tau + n;

  // Determine size of work array
  char side = 'L';
  char trans = 'T';
  double work_size;
  int lwork = -1;
  int info;
  LAPACK_dormqr(&side, &trans, &m, &k, &n, A, &m, tau, new_data, &m, &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  std::vector<double> work(lwork);

  // Multiply new data by Q'
  LAPACK_dormqr(&side, &trans, &m, &k, &n, A, &m, tau, new_data, &m, work.data(), &lwork, &info);

  // Determine size of work array
  int mn = m - n;
  lwork = -1;
  LAPACK_dgeqrf(&mn, &k, new_data+n, &m, nullptr, &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  if (lwork > work.size()) {
    work = std::vector<double>(lwork);
  }
  
  // Compute QR factorization of lower right
  LAPACK_dgeqrf(&mn, &k, new_data+n, &m, new_tau, work.data(), &lwork, &info);

  // Invert bottom right of the inverse triangular matrix
  char uplo = 'U';
  char diag = 'N';
  LAPACK_dtrtri(&uplo, &diag, &k, new_data+n, &m, &info);

  // Compute the top right of the inverse triangular matrix inv(R)_12 = -inv(R_11) * R_12 * inv(R_22)
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, k, -1.0, A, m, new_data, m);
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, n, k, 1.0, new_data+n, m, new_data, m);
}

void extract_q(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n) {
  auto buf = A_input.request();
  double *A = static_cast<double *>(buf.ptr);

  buf = tau_input.request();
  double *tau = static_cast<double *>(buf.ptr);

  double lwork_dbl;
  int lwork = -1;
  int info;
  LAPACK_dorgqr(&m, &n, &n, A, &m, tau, &lwork_dbl, &lwork, &info);
  lwork = static_cast<int>(lwork_dbl);
  std::vector<double> work(lwork);
  LAPACK_dorgqr(&m, &n, &n, A, &m, tau, work.data(), &lwork, &info);
}

double get_error_estimate(py::array_t<double> A_input, int m, int n) {
  auto buf = A_input.request();
  double *A = static_cast<double *>(buf.ptr);

  double output = 0.0;
  for (int row = 0; row < n; ++row) {
    double rownormsq = 0.0;
    for (int col = row; col < n; ++col) {
      rownormsq += A[row + m*col] * A[row + m*col];
    }
    output += 1.0 / rownormsq / n;
  }
  return sqrt(output);
}

PYBIND11_MODULE(libincrementalqr, m) {
    m.def("setup", &setup, "Compute an in place QR decomposition of a data buffer and invert the triangular matrix");
    m.def("add_cols", &add_cols, "Append columns to end of matrix, update QR factorization, and update triangular matrix inverse");
    m.def("extract_q", &extract_q, "Get the Q factor");
    m.def("get_error_estimate", &get_error_estimate, "Randomized QB approximation leave-one-out error estimate");
}