#include "config.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace riemann {
namespace adaptive {

AxisAlignedRegion AxisAlignedRegion::everything(size_t dimension) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  AxisAlignedRegion region;
  region.lo.assign(dimension, -inf);
  region.hi.assign(dimension, inf);
  return region;
}

void AdaptiveVoxelConfig::validate() const {
  const size_t dim = base_voxel_size.size();
  if (dim == 0 || dim > kMaxDimension)
    throw std::invalid_argument("AdaptiveVoxelConfig: base_voxel_size needs 1.." +
                                std::to_string(kMaxDimension) + " entries");
  for (double v : base_voxel_size)
    if (!(v > 0.0) || !std::isfinite(v))
      throw std::invalid_argument(
          "AdaptiveVoxelConfig: base_voxel_size entries must be positive finite numbers");
  if (!origin.empty()) {
    if (origin.size() != dim)
      throw std::invalid_argument("AdaptiveVoxelConfig: origin has the wrong dimension");
    for (double v : origin)
      if (!std::isfinite(v))
        throw std::invalid_argument("AdaptiveVoxelConfig: origin entries must be finite");
  }
  if (max_depth > kMaxSupportedDepth)
    throw std::invalid_argument("AdaptiveVoxelConfig: max_depth exceeds the supported maximum (" +
                                std::to_string(kMaxSupportedDepth) + ")");
  if (!min_voxel_size.empty()) {
    if (min_voxel_size.size() != dim)
      throw std::invalid_argument("AdaptiveVoxelConfig: min_voxel_size has the wrong dimension");
    const double want = std::ldexp(1.0, static_cast<int>(max_depth));
    for (size_t d = 0; d < dim; ++d) {
      if (!(min_voxel_size[d] > 0.0) || !std::isfinite(min_voxel_size[d]))
        throw std::invalid_argument(
            "AdaptiveVoxelConfig: min_voxel_size entries must be positive finite numbers");
      const double got = base_voxel_size[d] / min_voxel_size[d];
      if (std::fabs(got - want) > 1e-9 * want)
        throw std::invalid_argument(
            "AdaptiveVoxelConfig: base_voxel_size / min_voxel_size must equal 2^max_depth");
    }
  }
}

} // namespace adaptive
} // namespace riemann
