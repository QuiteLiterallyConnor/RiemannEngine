#include "splat/harmonics.hpp"

#include <vector>

namespace riemann {
namespace gaussian {
namespace {

// Real SH constants in the standard 3DGS ordering.
constexpr double kC1 = 0.4886025119029199;
constexpr double kC2[5] = {1.0925484305920792, -1.0925484305920792,
                           0.31539156525252005, -1.0925484305920792,
                           0.5462742152960396};
constexpr double kC3[7] = {-0.5900435899266435, 2.890611442640554,
                           -0.4570457994644658, 0.3731763325901154,
                           -0.4570457994644658, 1.445305721320277,
                           -0.5900435899266435};

} // namespace

at::Tensor sh_basis(int degree, const at::Tensor &directions) {
  TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
              "directions must have shape [N, 3]");
  const int64_t terms = sh_coefficient_count(degree);
  const auto x = directions.select(1, 0);
  const auto y = directions.select(1, 1);
  const auto z = directions.select(1, 2);

  std::vector<at::Tensor> rows;
  rows.reserve(static_cast<size_t>(terms));
  rows.push_back(at::full_like(x, kShC0));
  if (degree > 0) {
    rows.push_back(-kC1 * y);
    rows.push_back(kC1 * z);
    rows.push_back(-kC1 * x);
  }
  if (degree > 1) {
    const auto xx = x * x, yy = y * y, zz = z * z;
    const auto xy = x * y, yz = y * z, xz = x * z;
    rows.push_back(kC2[0] * xy);
    rows.push_back(kC2[1] * yz);
    rows.push_back(kC2[2] * (2.0 * zz - xx - yy));
    rows.push_back(kC2[3] * xz);
    rows.push_back(kC2[4] * (xx - yy));
    if (degree > 2) {
      rows.push_back(kC3[0] * y * (3.0 * xx - yy));
      rows.push_back(kC3[1] * xy * z);
      rows.push_back(kC3[2] * y * (4.0 * zz - xx - yy));
      rows.push_back(kC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
      rows.push_back(kC3[4] * x * (4.0 * zz - xx - yy));
      rows.push_back(kC3[5] * z * (xx - yy));
      rows.push_back(kC3[6] * x * (xx - 3.0 * yy));
    }
  }
  return at::stack(rows, 1);
}

at::Tensor eval_sh_color(int degree, const at::Tensor &sh,
                         const at::Tensor &directions) {
  const int64_t terms = sh_coefficient_count(degree);
  TORCH_CHECK(sh.dim() == 3 && sh.size(2) == 3 && sh.size(1) >= terms,
              "sh must have shape [N, K, 3] with K >= (degree + 1)^2");
  TORCH_CHECK(sh.size(0) == directions.size(0),
              "sh and directions must have the same length");
  const auto basis = sh_basis(degree, directions);
  const auto active = sh.slice(1, 0, terms);
  const auto radiance = (basis.unsqueeze(-1) * active).sum(1);
  return (radiance + 0.5).clamp_min(0.0);
}

} // namespace gaussian
} // namespace riemann
