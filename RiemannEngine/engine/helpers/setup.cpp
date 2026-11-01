#include "helpers/setup.hpp"

#include <cmath>

namespace riemann {

std::pair<int64_t, int64_t>
calculate_fan_in_and_fan_out(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() >= 2,
              "Fan in/out require a tensor with at least two dimensions");
  if (tensor.dim() == 2) return {tensor.size(1), tensor.size(0)};
  const auto kernel_volume = tensor.size(0);
  return {tensor.size(1) * kernel_volume, tensor.size(2) * kernel_volume};
}

at::Tensor& kaiming_normal_(at::Tensor& tensor, double a,
                            const std::string& mode,
                            const std::string& nonlinearity) {
  TORCH_CHECK(mode == "fan_in" || mode == "fan_out",
              "Mode must be 'fan_in' or 'fan_out'");
  const auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
  const auto fan = mode == "fan_in" ? fan_in : fan_out;
  TORCH_CHECK(fan > 0, "Fan must be positive");
  double gain = std::sqrt(2.0 / (1.0 + a * a));
  if (nonlinearity == "linear" || nonlinearity == "sigmoid" ||
      nonlinearity.rfind("conv", 0) == 0)
    gain = 1.0;
  else if (nonlinearity == "tanh")
    gain = 5.0 / 3.0;
  else if (nonlinearity == "relu")
    gain = std::sqrt(2.0);
  else if (nonlinearity == "selu")
    gain = 0.75;
  torch::NoGradGuard guard;
  return tensor.normal_(0.0, gain / std::sqrt(static_cast<double>(fan)));
}

} // namespace riemann
