#ifndef RIEMANN_ENGINE_OPS_CONVOLUTION_CPU_HPP
#define RIEMANN_ENGINE_OPS_CONVOLUTION_CPU_HPP

#include "engine/grid/maps.hpp"

#include <cstring>
#include <vector>

namespace riemann {

inline bool has_contiguous_identity_mapping(const cpu_in_maps& inputs,
                                            const cpu_out_maps& outputs,
                                            int input_rows,
                                            int output_rows) {
  if (input_rows <= 0 || input_rows != output_rows || inputs.size() != 1 ||
      outputs.size() != 1 || inputs.front().size() !=
                                  static_cast<size_t>(input_rows) ||
      outputs.front().size() != static_cast<size_t>(output_rows))
    return false;
  for (int row = 0; row < input_rows; ++row)
    if (inputs.front()[row] != static_cast<index_type>(row) ||
        outputs.front()[row] != static_cast<index_type>(row))
      return false;
  return true;
}

template <typename Dtype, typename Itype>
void sparse_convolution_forward_cpu_kernel(
    const Dtype* input, int input_channels, Dtype* output, int output_channels,
    int input_rows, int output_rows, const Dtype* weights,
    const cpu_in_maps& input_maps, const cpu_out_maps& output_maps) {
  (void)sizeof(Itype);
  for (size_t kernel = 0; kernel < input_maps.size(); ++kernel) {
    const auto& in_rows = input_maps[kernel];
    const auto& out_rows = output_maps[kernel];
    const auto* kernel_weights =
        weights + kernel * input_channels * output_channels;
    for (size_t pair = 0; pair < in_rows.size(); ++pair) {
      const auto in_row = static_cast<int64_t>(in_rows[pair]);
      const auto out_row = static_cast<int64_t>(out_rows[pair]);
      if (in_row < 0 || in_row >= input_rows || out_row < 0 ||
          out_row >= output_rows)
        continue;
      const auto* feature = input + in_row * input_channels;
      auto* result = output + out_row * output_channels;
      for (int in_channel = 0; in_channel < input_channels; ++in_channel)
        for (int out_channel = 0; out_channel < output_channels; ++out_channel)
          result[out_channel] += feature[in_channel] *
              kernel_weights[in_channel * output_channels + out_channel];
    }
  }
}

template <typename Dtype, typename Itype>
void sparse_convolution_backward_cpu_kernel(
    const Dtype* input, Dtype* grad_input, int input_channels,
    const Dtype* grad_output, int output_channels, int input_rows,
    int output_rows, const Dtype* weights, Dtype* grad_weights,
    const cpu_in_maps& input_maps, const cpu_out_maps& output_maps) {
  (void)sizeof(Itype);
  for (size_t kernel = 0; kernel < input_maps.size(); ++kernel) {
    const auto& in_rows = input_maps[kernel];
    const auto& out_rows = output_maps[kernel];
    const auto* kernel_weights =
        weights + kernel * input_channels * output_channels;
    auto* kernel_grad =
        grad_weights + kernel * input_channels * output_channels;
    for (size_t pair = 0; pair < in_rows.size(); ++pair) {
      const auto in_row = static_cast<int64_t>(in_rows[pair]);
      const auto out_row = static_cast<int64_t>(out_rows[pair]);
      if (in_row < 0 || in_row >= input_rows || out_row < 0 ||
          out_row >= output_rows)
        continue;
      const auto* feature = input + in_row * input_channels;
      const auto* upstream = grad_output + out_row * output_channels;
      auto* input_grad = grad_input + in_row * input_channels;
      for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
        for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
          input_grad[in_channel] +=
              upstream[out_channel] *
              kernel_weights[in_channel * output_channels + out_channel];
          kernel_grad[in_channel * output_channels + out_channel] +=
              feature[in_channel] * upstream[out_channel];
        }
      }
    }
  }
}

} // namespace riemann

#endif
