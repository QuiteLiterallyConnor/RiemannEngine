#ifndef RIEMANN_SPLAT_FOURDGS_HPP
#define RIEMANN_SPLAT_FOURDGS_HPP

#include "splat/model4d.hpp"

#include <cstdint>
#include <string>

namespace riemann {
namespace gaussian {

struct FourDGSExportOptions {
  // Model time [time_start, time_end) maps to scene seconds [0, duration).
  double time_start = 0.0;
  double time_end = 1.0;
  double seconds_per_time_unit = 1.0;
  double cutoff = 1e-6;
  int64_t chunk_size = 4096;
};

struct FourDGSExportResult {
  int64_t gaussians = 0;
  int64_t bytes = 0;
  int64_t clamped_rgb_components = 0;
  int64_t clamped_sh_coefficients = 0;
  // Measured maxima against source values AFTER format-required clamping.
  double max_position_error = 0.0;
  double max_scale_log_error = 0.0;
  double max_rotation_error = 0.0;
  double max_rgb_error = 0.0;
  double max_opacity_error = 0.0;
  double max_velocity_error = 0.0; // scene units / second
  double max_time_error = 0.0;     // seconds
  double max_sigma_log_error = 0.0;
  double max_sh_error = 0.0;       // coefficient units
};

// Exports the avala-ai/4dgs version-1 draft, gaussian-birth/uniform-v1.
// Includes all SH bands, deflate streams, an index and summary CRC. Normalizes
// wxyz -> xyzw quaternions; quantization is lossy. RGB is clamped to [0,1] and
// higher SH coefficients to [-4,4], with explicit counts in the result and file
// metadata. Source tensors are never changed. Rejects invalid/unrepresentable
// inputs and existing destinations; publishes the completed file by rename.
FourDGSExportResult save_four_dgs(const Gaussian4DModel &model,
                                 const std::string &path,
                                 const FourDGSExportOptions &options = {});

} // namespace gaussian
} // namespace riemann

#endif
