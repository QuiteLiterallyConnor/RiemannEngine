#ifndef RIEMANN_ENGINE_NORMAL_DEVICE_HPP
#define RIEMANN_ENGINE_NORMAL_DEVICE_HPP

#include "../splat/gpu.hpp"

namespace riemann {
namespace gaussian {
namespace detail {

constexpr int kRasterTileSize = gpu::kTileSide;
using gpu::covariance_cuda;
using gpu::rasterize_backward_cuda;
using gpu::rasterize_forward_cuda;
using gpu::transform_cuda;

}
}
}

#endif
