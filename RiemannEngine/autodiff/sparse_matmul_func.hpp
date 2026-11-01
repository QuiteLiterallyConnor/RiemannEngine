#ifndef RIEMANN_ENGINE_AUTODIFF_SPARSE_MATMUL_FUNC_HPP
#define RIEMANN_ENGINE_AUTODIFF_SPARSE_MATMUL_FUNC_HPP

#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace riemann {
namespace detail {

inline at::Tensor coo_spmm_index_add(const at::Tensor& rows,
                                    const at::Tensor& cols,
                                    const at::Tensor& vals, int64_t dim_i,
                                    int64_t dim_j, const at::Tensor& mat) {
  TORCH_CHECK(rows.dim() == 1 && cols.dim() == 1 && vals.dim() == 1,
              "COO rows, columns, and values must be one-dimensional");
  TORCH_CHECK((rows.scalar_type() == at::kInt ||
               rows.scalar_type() == at::kLong) &&
                  (cols.scalar_type() == at::kInt ||
                   cols.scalar_type() == at::kLong),
              "COO row and column indices must be int32 or int64");
  TORCH_CHECK(rows.numel() == cols.numel() && rows.numel() == vals.numel(),
              "COO rows, columns, and values must have equal lengths");
  TORCH_CHECK(dim_i >= 0 && dim_j >= 0,
              "COO matrix dimensions must be non-negative");
  TORCH_CHECK(mat.dim() == 2 && mat.size(0) == dim_j,
              "dense matrix must have shape [dim_j, features]");

  const auto device = mat.device();
  const auto row_indices = rows.to(device, at::kLong);
  const auto column_indices = cols.to(device, at::kLong);
  const auto values = vals.to(device, mat.scalar_type());

  auto edge_features =
      mat.index_select(0, column_indices) * values.unsqueeze(1);
  auto output = at::zeros({dim_i, mat.size(1)}, mat.options());
  return output.index_add(0, row_indices, edge_features);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
coo_spmm_average_impl(const at::Tensor& rows, const at::Tensor& cols,
                      int64_t dim_i, int64_t dim_j, const at::Tensor& mat) {
  TORCH_CHECK(rows.dim() == 1 && cols.dim() == 1,
              "COO rows and columns must be one-dimensional");
  TORCH_CHECK((rows.scalar_type() == at::kInt ||
               rows.scalar_type() == at::kLong) &&
                  (cols.scalar_type() == at::kInt ||
                   cols.scalar_type() == at::kLong),
              "COO row and column indices must be int32 or int64");
  TORCH_CHECK(rows.numel() == cols.numel(),
              "COO rows and columns must have equal lengths");
  TORCH_CHECK(dim_i >= 0 && dim_j >= 0,
              "COO matrix dimensions must be non-negative");
  TORCH_CHECK(mat.dim() == 2 && mat.size(0) == dim_j,
              "dense matrix must have shape [dim_j, features]");

  const auto device = mat.device();
  const auto source_rows = rows.to(device, at::kLong);
  const auto source_cols = cols.to(device, at::kLong);

  const auto sort_result = at::sort(source_rows, 0);
  const auto sorted_rows = std::get<0>(sort_result);
  const auto sort_indices = std::get<1>(sort_result);
  const auto sorted_cols = source_cols.index_select(0, sort_indices);
  auto counts = at::zeros({dim_i}, mat.options().dtype(at::kLong))
                    .index_add(0, sorted_rows,
                               at::ones_like(sorted_rows));
  auto weights =
      counts.index_select(0, sorted_rows).to(mat.scalar_type()).reciprocal();
  auto output = coo_spmm_index_add(sorted_rows, sorted_cols, weights, dim_i,
                                  dim_j, mat);
  auto coo = at::stack({sorted_rows, sorted_cols}, 0);
  return {output, coo, weights};
}

} // namespace detail

template <typename th_int_type>
inline torch::Tensor coo_spmm(torch::Tensor const& rows,
                              torch::Tensor const& cols,
                              torch::Tensor const& vals, int64_t dim_i,
                              int64_t dim_j, torch::Tensor const& mat,
                              int64_t /*spmm_algorithm_id*/,
                              bool /*is_sorted*/) {
  return detail::coo_spmm_index_add(rows, cols, vals, dim_i, dim_j, mat);
}

template <typename th_int_type>
inline std::vector<torch::Tensor>
coo_spmm_average(torch::Tensor const& rows, torch::Tensor const& cols,
                 int64_t dim_i, int64_t dim_j, torch::Tensor const& mat,
                 int64_t /*spmm_algorithm_id*/) {
  auto result =
      detail::coo_spmm_average_impl(rows, cols, dim_i, dim_j, mat);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
}

inline torch::Tensor spmm_cpu(torch::Tensor const& rows,
                              torch::Tensor const& cols,
                              torch::Tensor const& vals, int64_t dim_i,
                              int64_t dim_j, torch::Tensor const& mat) {
  return detail::coo_spmm_index_add(rows, cols, vals, dim_i, dim_j, mat);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spmm_average_cpu(torch::Tensor const& rows, torch::Tensor const& cols,
                 int64_t dim_i, int64_t dim_j, torch::Tensor const& mat) {
  return detail::coo_spmm_average_impl(rows, cols, dim_i, dim_j, mat);
}

inline torch::Tensor spmm_dispatch(torch::Tensor const& rows,
                                   torch::Tensor const& cols,
                                   torch::Tensor const& vals, int64_t dim_i,
                                   int64_t dim_j, torch::Tensor const& mat,
                                   int64_t cuda_spmm_alg, bool is_sorted) {
#ifdef CPU_ONLY
  TORCH_CHECK(!mat.is_cuda(), "CUDA not available in CPU_ONLY build");
#endif
  return coo_spmm<int32_t>(rows, cols, vals, dim_i, dim_j, mat,
                           cuda_spmm_alg, is_sorted);
}

struct SPMMFunction : public torch::autograd::Function<SPMMFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            at::Tensor rows, at::Tensor cols, at::Tensor vals,
                            int64_t dim_i, int64_t dim_j, at::Tensor mat,
                            int64_t cuda_spmm_alg = 1) {
    ctx->saved_data["dim_i"] = dim_i;
    ctx->saved_data["dim_j"] = dim_j;
    ctx->saved_data["cuda_spmm_alg"] = cuda_spmm_alg;
    ctx->save_for_backward({rows, cols, vals});
    return spmm_dispatch(rows, cols, vals, dim_i, dim_j, mat,
                         cuda_spmm_alg, false);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext* ctx,
           torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto grad_mat = spmm_dispatch(
        saved[1], saved[0], saved[2],
        ctx->saved_data["dim_j"].toInt(),
        ctx->saved_data["dim_i"].toInt(), grad_outputs[0].contiguous(),
        ctx->saved_data["cuda_spmm_alg"].toInt(), false);
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), grad_mat, at::Tensor()};
  }
};

struct SPMMAverageFunction
    : public torch::autograd::Function<SPMMAverageFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            at::Tensor rows, at::Tensor cols, int64_t dim_i,
                            int64_t dim_j, at::Tensor mat,
                            int64_t cuda_spmm_alg = 1) {
    ctx->saved_data["dim_i"] = dim_i;
    ctx->saved_data["dim_j"] = dim_j;
    ctx->saved_data["cuda_spmm_alg"] = cuda_spmm_alg;
    auto result =
        detail::coo_spmm_average_impl(rows, cols, dim_i, dim_j, mat);
    ctx->save_for_backward({std::get<1>(result), std::get<2>(result)});
    return std::get<0>(result);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext* ctx,
           torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto coo = saved[0];
    const auto grad_mat = spmm_dispatch(
        coo.index({1}), coo.index({0}), saved[1],
        ctx->saved_data["dim_j"].toInt(),
        ctx->saved_data["dim_i"].toInt(), grad_outputs[0].contiguous(),
        ctx->saved_data["cuda_spmm_alg"].toInt(), false);
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            grad_mat, at::Tensor()};
  }
};

} // namespace riemann

#endif
