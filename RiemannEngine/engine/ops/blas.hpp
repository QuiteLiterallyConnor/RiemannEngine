#ifndef RIEMANN_ENGINE_OPS_BLAS_HPP
#define RIEMANN_ENGINE_OPS_BLAS_HPP

#include <algorithm>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

namespace riemann {

template <typename T>
void cpu_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transpose_a,
              CBLAS_TRANSPOSE transpose_b, int rows, int columns,
              int reduction, T alpha, const T* a, const T* b, T beta,
              T* output) {
  const bool row_major = layout == CblasRowMajor;
  const bool a_transposed = transpose_a != CblasNoTrans;
  const bool b_transposed = transpose_b != CblasNoTrans;
  const auto a_at = [&](int row, int col) {
    if (row_major)
      return a_transposed ? a[col * rows + row] : a[row * reduction + col];
    return a_transposed ? a[col + row * reduction] : a[row + col * rows];
  };
  const auto b_at = [&](int row, int col) {
    if (row_major)
      return b_transposed ? b[col * reduction + row] : b[row * columns + col];
    return b_transposed ? b[col + row * columns] : b[row + col * reduction];
  };
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < columns; ++col) {
      T sum = T(0);
      for (int k = 0; k < reduction; ++k)
        sum += a_at(row, k) * b_at(k, col);
      const auto index =
          row_major ? row * columns + col : row + col * rows;
      output[index] = alpha * sum + beta * output[index];
    }
  }
}

template <typename T>
void cpu_add(int count, const T* a, const T* b, T* output) {
  for (int i = 0; i < count; ++i)
    output[i] = a[i] + b[i];
}

template <typename T>
void cpu_mul(int count, const T* a, const T* b, T* output) {
  for (int i = 0; i < count; ++i)
    output[i] = a[i] * b[i];
}

template <typename T>
void cpu_div(int count, const T* a, const T* b, T* output) {
  for (int i = 0; i < count; ++i)
    output[i] = a[i] / b[i];
}

template <typename T>
void cpu_axpy(int count, T alpha, const T* input, T* output) {
  for (int i = 0; i < count; ++i)
    output[i] += alpha * input[i];
}

} // namespace riemann

#endif
