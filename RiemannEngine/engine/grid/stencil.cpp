#include "engine/grid/stencil.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace riemann {

at::Tensor detail::normalize_region_offsets(const at::Tensor &offsets,
                                           int dimension) {
  TORCH_CHECK(dimension > 0, "CUSTOM region dimension must be positive");
  TORCH_CHECK(offsets.defined(), "region_offsets must be defined for CUSTOM");
  TORCH_CHECK(offsets.layout() == at::kStrided && offsets.dim() == 2,
              "region_offsets must be a dense [N, D] tensor");
  TORCH_CHECK(offsets.size(0) > 0 &&
                  offsets.size(0) <= std::numeric_limits<int>::max(),
              "region_offsets must have a non-empty, representable volume");
  TORCH_CHECK(offsets.size(1) == dimension,
              "region_offsets dimension mismatch");
  TORCH_CHECK(offsets.scalar_type() == at::kInt ||
                  offsets.scalar_type() == at::kLong,
              "region_offsets must contain int32 or int64 spatial offsets");
  TORCH_CHECK(offsets.device().is_cpu() || offsets.device().is_cuda(),
              "region_offsets must be on CPU or CUDA");
  auto host = offsets.to(at::kCPU).contiguous();
  if (host.scalar_type() == at::kLong) {
    TORCH_CHECK(host.min().item<int64_t>() >=
                        std::numeric_limits<int32_t>::min() &&
                    host.max().item<int64_t>() <=
                        std::numeric_limits<int32_t>::max(),
                "region_offsets values must fit int32");
  }
  return host.to(at::kInt).clone();
}

namespace {

int region_volume(RegionType::Type region_type,
                  const stride_type &kernel_size,
                  const at::Tensor &region_offsets, int dimension) {
  switch (region_type) {
  case RegionType::HYPER_CUBE: {
    int volume = 1;
    for (auto k : kernel_size) {
      TORCH_CHECK(k > 0, "kernel_size must be positive");
      volume *= k;
    }
    return volume;
  }
  case RegionType::HYPER_CROSS: {
    int volume = 1;
    for (auto k : kernel_size) {
      TORCH_CHECK(k > 0, "kernel_size must be positive");
      TORCH_CHECK(k % 2 == 1, "kernel_size must be odd for HYPER_CROSS");
      volume += (k - 1);
    }
    return volume;
  }
  case RegionType::CUSTOM:
    return region_offsets.size(0);
  }

  throw std::runtime_error("Unknown region type");
}

stride_type to_stride_vector(torch::IntArrayRef values) {
  stride_type v;
  v.reserve(values.size());
  for (auto x : values) {
    TORCH_CHECK(x > 0 && x <= std::numeric_limits<uint32_t>::max(),
                "kernel size, stride and dilation must be positive uint32 values");
    v.push_back(static_cast<uint32_t>(x));
  }
  return v;
}

} // namespace

KernelGenerator::KernelGenerator(
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride,
    torch::IntArrayRef dilation,
    bool is_transpose,
    RegionType::Type region_type,
    at::Tensor region_offsets,
    bool expand_coordinates,
    std::vector<RegionType::Type> axis_types,
    int dimension)
    : kernel_size_(to_stride_vector(kernel_size)),
      kernel_stride_(to_stride_vector(stride)),
      kernel_dilation_(to_stride_vector(dilation)),
      is_transpose_(is_transpose),
      region_type_(region_type),
      region_offsets_(region_offsets),
      expand_coordinates_(expand_coordinates),
      axis_types_(axis_types),
      D_(dimension),
      kernel_volume_(0) {
  initialize();
}

void KernelGenerator::initialize() {
  if (D_ <= 0) {
    D_ = kernel_size_.size();
  }

  // scalars broadcast across every spatial axis
  if (kernel_size_.size() == 1 && D_ > 1) {
    kernel_size_.assign(D_, kernel_size_[0]);
  }
  if (kernel_stride_.size() == 1 && D_ > 1) {
    kernel_stride_.assign(D_, kernel_stride_[0]);
  }
  if (kernel_dilation_.size() == 1 && D_ > 1) {
    kernel_dilation_.assign(D_, kernel_dilation_[0]);
  }

  TORCH_CHECK(kernel_size_.size() == static_cast<size_t>(D_),
              "kernel_size dimension mismatch");
  TORCH_CHECK(kernel_stride_.size() == static_cast<size_t>(D_),
              "kernel_stride dimension mismatch");
  TORCH_CHECK(kernel_dilation_.size() == static_cast<size_t>(D_),
              "kernel_dilation dimension mismatch");

  if (region_type_ == RegionType::CUSTOM) {
    TORCH_CHECK(axis_types_.empty(), "axis_types must be empty for CUSTOM");
    region_offsets_ = detail::normalize_region_offsets(region_offsets_, D_);
  }

  if (region_type_ == RegionType::HYPER_CUBE ||
      region_type_ == RegionType::HYPER_CROSS) {
    TORCH_CHECK(axis_types_.empty(),
                "axis_types must be empty for HYPER_CUBE/HYPER_CROSS");
    if (region_offsets_.defined()) {
      TORCH_CHECK(region_offsets_.numel() == 0,
                  "region_offsets must be empty for HYPER_CUBE/HYPER_CROSS");
    }
  }

  kernel_volume_ =
      region_volume(region_type_, kernel_size_, region_offsets_, D_);

  requires_strided_coordinates_ =
      std::all_of(kernel_stride_.begin(), kernel_stride_.end(),
                  [](auto s) { return s == 1; });
}

bool KernelGenerator::requires_strided_coordinates() const {
  return requires_strided_coordinates_;
}

KernelGenerator::KernelParams
KernelGenerator::get_kernel(const stride_type &tensor_stride,
                            bool is_transpose) const {
  TORCH_CHECK(tensor_stride.size() == static_cast<size_t>(D_),
              "tensor_stride dimension mismatch");
  for (auto s : tensor_stride)
    TORCH_CHECK(s > 0, "tensor_stride must be positive");
  auto &cache = cache_[is_transpose ? 1 : 0];
  auto it = cache.find(tensor_stride);
  if (it != cache.end()) {
    auto params = it->second;
    params.region_offsets = params.region_offsets.clone();
    return params;
  }

  KernelParams params;
  params.kernel_size = kernel_size_;
  params.kernel_stride = kernel_stride_;
  params.kernel_dilation = kernel_dilation_;
  params.expand_coordinates = expand_coordinates_;
  params.D = D_;

  stride_type up_stride(D_, 1);
  if (is_transpose) {
    up_stride = kernel_stride_;
  }

  auto [final_region_type, final_offsets, volume] = convert_region_type_impl(
      region_type_, tensor_stride, kernel_size_, up_stride, kernel_dilation_,
      region_offsets_, axis_types_, D_, true);
  (void)volume;

  params.region_type = final_region_type;
  params.region_offsets = final_offsets;

  cache[tensor_stride] = params;
  params.region_offsets = params.region_offsets.clone();

  return params;
}

std::tuple<RegionType::Type, at::Tensor, int>
KernelGenerator::convert_region_type_impl(
    RegionType::Type region_type,
    const stride_type &tensor_stride,
    const stride_type &kernel_size,
    const stride_type &up_stride,
    const stride_type &dilation,
    const at::Tensor &region_offsets,
    const std::vector<RegionType::Type> &axis_types,
    int dimension,
    bool center) const {
  switch (region_type) {
  case RegionType::HYPER_CUBE: {
    at::Tensor empty_offsets = torch::empty({0, dimension}, torch::kInt32);
    int volume = 1;
    for (auto k : kernel_size)
      volume *= k;
    return {region_type, empty_offsets, volume};
  }
  case RegionType::HYPER_CROSS: {
    at::Tensor empty_offsets = torch::empty({0, dimension}, torch::kInt32);
    int volume = 1;
    for (auto k : kernel_size)
      volume += (k - 1);
    return {region_type, empty_offsets, volume};
  }
  case RegionType::CUSTOM: {
    auto offsets = detail::normalize_region_offsets(region_offsets, dimension);
    int volume = offsets.size(0);
    return {RegionType::CUSTOM, offsets, volume};
  }
  }

  throw std::runtime_error("Unknown or unimplemented region type");
}

} // namespace riemann
