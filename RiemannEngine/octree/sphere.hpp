#ifndef RIEMANN_OCTREE_SPHERE_HPP
#define RIEMANN_OCTREE_SPHERE_HPP

#include "tensor.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define RIEMANN_OCTREE_HD __host__ __device__
#else
#define RIEMANN_OCTREE_HD
#endif

namespace riemann {
namespace adaptive {

// Voxelizes a sphere with the fewest leaves reaching `target_accuracy_percent`.
//
// Leaves are cubes refined from a single root cube of side 2 * radius by
// splitting the largest boundary cells first. A leaf counts as occupied when
// its center lies inside the sphere; the volume estimate is the total occupied
// leaf volume (see sphere_volume_estimate) and
// accuracy = 100 * (1 - |estimate - exact| / exact).
//
// The result has one float channel: 1 for occupied leaves, 0 otherwise. When
// achieved_accuracy_percent is non-null it receives the accuracy reached.
// Throws std::invalid_argument on bad arguments and std::runtime_error when
// the target is unreachable within max_voxels leaves / max_depth levels.
AdaptiveSparseTensor voxelize_sphere(double radius, double target_accuracy_percent,
                                     std::array<double, 3> center = {0.0, 0.0, 0.0},
                                     int64_t max_voxels = 10'000'000, uint32_t max_depth = 10,
                                     double *achieved_accuracy_percent = nullptr);

// Same refinement and result as voxelize_sphere, computed on the GPU. Features
// of the returned tensor live on the current CUDA device. Throws
// std::runtime_error in CPU_ONLY builds or when CUDA is unavailable.
AdaptiveSparseTensor voxelize_sphere_cuda(double radius, double target_accuracy_percent,
                                          std::array<double, 3> center = {0.0, 0.0, 0.0},
                                          int64_t max_voxels = 10'000'000,
                                          uint32_t max_depth = 10,
                                          double *achieved_accuracy_percent = nullptr);

// Total volume of occupied leaves in a tensor made by voxelize_sphere[_cuda].
double sphere_volume_estimate(const AdaptiveSparseTensor &sphere);

namespace sphere_detail {

void validate_args(double radius, double target_accuracy_percent, int64_t max_voxels,
                   uint32_t max_depth);
AdaptiveVoxelConfig make_config(double radius, const std::array<double, 3> &center,
                                uint32_t max_depth);

constexpr double kPi = 3.14159265358979323846;

// Cell classification shared by the CPU and CUDA voxelizers so both make
// identical decisions. Works in a frame where the sphere center sits at
// (r, r, r) and the bounding cube [0, 2r)^3 is the level-0 cell; a cell is
// identified by (x, y, z, level).
struct SphereFrame {
  double r;

  RIEMANN_OCTREE_HD double side() const { return 2.0 * r; }
  RIEMANN_OCTREE_HD double exact_volume() const { return (4.0 / 3.0) * kPi * r * r * r; }
  RIEMANN_OCTREE_HD double accuracy(double volume) const {
    return 100.0 * (1.0 - fabs(volume - exact_volume()) / exact_volume());
  }
  RIEMANN_OCTREE_HD double cell_size(int level) const { return ldexp(side(), -level); }

  // The cell straddles the sphere surface (neither fully inside nor outside).
  RIEMANN_OCTREE_HD bool is_boundary(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    const int64_t c[3] = {x, y, z};
    double near2 = 0.0, far2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double lo = static_cast<double>(c[axis]) * size;
      const double hi = lo + size;
      const double clamped = r < lo ? lo : (r > hi ? hi : r);
      const double near_d = clamped - r;
      const double d_lo = fabs(lo - r), d_hi = fabs(hi - r);
      const double far_d = d_lo > d_hi ? d_lo : d_hi;
      near2 += near_d * near_d;
      far2 += far_d * far_d;
    }
    return near2 < r * r && far2 > r * r;
  }

  // Occupied when the cell center lies inside the sphere.
  RIEMANN_OCTREE_HD bool occupied(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    const int64_t c[3] = {x, y, z};
    double dist2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double d = (static_cast<double>(c[axis]) + 0.5) * size - r;
      dist2 += d * d;
    }
    return dist2 <= r * r;
  }

  RIEMANN_OCTREE_HD double contribution(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    return occupied(x, y, z, level) ? size * size * size : 0.0;
  }
};

} // namespace sphere_detail

} // namespace adaptive
} // namespace riemann

#endif
