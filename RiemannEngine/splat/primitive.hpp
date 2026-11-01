#ifndef RIEMANN_SPLAT_PRIMITIVE_HPP
#define RIEMANN_SPLAT_PRIMITIVE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace riemann {
namespace gaussian {

template <typename T> using Vector3 = std::array<T, 3>;
template <typename T> using Quaternion = std::array<T, 4>; // (w, x, y, z)
template <typename T> using Matrix3 = std::array<std::array<T, 3>, 3>;

// A single anisotropic 3D Gaussian: world-space mean, unit quaternion
// orientation, per-axis standard deviations, opacity, color, and optional
// per-splat metadata. Covariance is R diag(scale^2) R^T, built on demand.
template <typename T> class Gaussian3D {
  static_assert(std::is_floating_point<T>::value,
                "Gaussian3D requires a floating-point scalar");

public:
  Vector3<T> mean{T(0), T(0), T(0)};
  Quaternion<T> orientation{T(1), T(0), T(0), T(0)};
  Vector3<T> scale{T(1), T(1), T(1)};
  T opacity = T(1);
  Vector3<T> color{T(0), T(0), T(0)};
  std::vector<T> features;
  std::optional<Vector3<T>> velocity;
  std::optional<T> timestamp;
  std::optional<int64_t> semantic_class;
  std::optional<int64_t> object_id;

  static constexpr T kMinimumScale = T(1e-6);

  Quaternion<T> normalized_orientation() const {
    T squared = T(0);
    for (T component : orientation) {
      if (!std::isfinite(component))
        throw std::invalid_argument("Gaussian orientation must be finite");
      squared += component * component;
    }
    if (!(squared > T(0)) || !std::isfinite(squared))
      throw std::invalid_argument(
          "Gaussian orientation must have nonzero norm");
    const T reciprocal = T(1) / std::sqrt(squared);
    Quaternion<T> unit{};
    for (size_t i = 0; i < unit.size(); ++i)
      unit[i] = orientation[i] * reciprocal;
    return unit;
  }

  void validate(T minimum_scale = kMinimumScale) const {
    if (!(minimum_scale > T(0)) || !std::isfinite(minimum_scale))
      throw std::invalid_argument(
          "minimum scale must be positive and finite");
    for (T component : mean)
      if (!std::isfinite(component))
        throw std::invalid_argument("Gaussian mean must be finite");
    for (T component : scale)
      if (!std::isfinite(component) || component < minimum_scale)
        throw std::invalid_argument(
            "Gaussian scales must be finite and above minimum_scale");
    for (T component : color)
      if (!std::isfinite(component))
        throw std::invalid_argument("Gaussian color must be finite");
    for (T component : features)
      if (!std::isfinite(component))
        throw std::invalid_argument("Gaussian features must be finite");
    if (!std::isfinite(opacity) || opacity < T(0) || opacity > T(1))
      throw std::invalid_argument(
          "Gaussian opacity must be finite and in [0, 1]");
    if (velocity)
      for (T component : *velocity)
        if (!std::isfinite(component))
          throw std::invalid_argument("Gaussian velocity must be finite");
    if (timestamp && !std::isfinite(*timestamp))
      throw std::invalid_argument("Gaussian timestamp must be finite");
    (void)normalized_orientation();
  }

  Matrix3<T> rotation_matrix() const {
    const Quaternion<T> q = normalized_orientation();
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    Matrix3<T> r;
    r[0] = {T(1) - T(2) * (y * y + z * z), T(2) * (x * y - z * w),
            T(2) * (x * z + y * w)};
    r[1] = {T(2) * (x * y + z * w), T(1) - T(2) * (x * x + z * z),
            T(2) * (y * z - x * w)};
    r[2] = {T(2) * (x * z - y * w), T(2) * (y * z + x * w),
            T(1) - T(2) * (x * x + y * y)};
    return r;
  }

  Matrix3<T> covariance() const {
    validate();
    const Matrix3<T> r = rotation_matrix();
    Matrix3<T> sigma{};
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        for (size_t a = 0; a < 3; ++a)
          sigma[i][j] += r[i][a] * scale[a] * scale[a] * r[j][a];
    return sigma;
  }

  T determinant() const {
    validate();
    return scale[0] * scale[0] * scale[1] * scale[1] * scale[2] * scale[2];
  }

  Matrix3<T> inverse_covariance() const {
    validate();
    const Matrix3<T> r = rotation_matrix();
    Matrix3<T> precision{};
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        for (size_t a = 0; a < 3; ++a)
          precision[i][j] += r[i][a] / (scale[a] * scale[a]) * r[j][a];
    return precision;
  }

  T mahalanobis_squared(const Vector3<T> &point) const {
    const Matrix3<T> precision = inverse_covariance();
    Vector3<T> offset{};
    for (size_t i = 0; i < 3; ++i) {
      if (!std::isfinite(point[i]))
        throw std::invalid_argument("evaluation point must be finite");
      offset[i] = point[i] - mean[i];
    }
    T quadratic = T(0);
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        quadratic += offset[i] * precision[i][j] * offset[j];
    return std::max(T(0), quadratic);
  }

  T density(const Vector3<T> &point) const {
    const T two_pi = T(6.283185307179586476925286766559);
    const T amplitude =
        T(1) / (std::pow(two_pi, T(1.5)) * scale[0] * scale[1] * scale[2]);
    return amplitude * std::exp(T(-0.5) * mahalanobis_squared(point));
  }

  T overlap_integral(const Gaussian3D<T> &other) const {
    validate();
    other.validate();
    const Matrix3<T> ca = covariance();
    const Matrix3<T> cb = other.covariance();
    Matrix3<T> s{};
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        s[i][j] = ca[i][j] + cb[i][j];

    const T det = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1]) -
                  s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0]) +
                  s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);
    if (!(det > T(0)) || !std::isfinite(det))
      throw std::runtime_error("combined covariance is not positive definite");

    Matrix3<T> inv{};
    inv[0][0] = (s[1][1] * s[2][2] - s[1][2] * s[2][1]) / det;
    inv[0][1] = (s[0][2] * s[2][1] - s[0][1] * s[2][2]) / det;
    inv[0][2] = (s[0][1] * s[1][2] - s[0][2] * s[1][1]) / det;
    inv[1][0] = (s[1][2] * s[2][0] - s[1][0] * s[2][2]) / det;
    inv[1][1] = (s[0][0] * s[2][2] - s[0][2] * s[2][0]) / det;
    inv[1][2] = (s[0][2] * s[1][0] - s[0][0] * s[1][2]) / det;
    inv[2][0] = (s[1][0] * s[2][1] - s[1][1] * s[2][0]) / det;
    inv[2][1] = (s[0][1] * s[2][0] - s[0][0] * s[2][1]) / det;
    inv[2][2] = (s[0][0] * s[1][1] - s[0][1] * s[1][0]) / det;

    Vector3<T> offset{};
    for (size_t i = 0; i < 3; ++i)
      offset[i] = mean[i] - other.mean[i];
    T quadratic = T(0);
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        quadratic += offset[i] * inv[i][j] * offset[j];
    const T two_pi = T(6.283185307179586476925286766559);
    return std::exp(T(-0.5) * quadratic) /
           (std::pow(two_pi, T(1.5)) * std::sqrt(det));
  }

  Vector3<T> sample(std::mt19937_64 &generator) const {
    validate();
    std::normal_distribution<T> standard(T(0), T(1));
    Vector3<T> local{scale[0] * standard(generator),
                     scale[1] * standard(generator),
                     scale[2] * standard(generator)};
    const Matrix3<T> r = rotation_matrix();
    Vector3<T> world = mean;
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        world[i] += r[i][j] * local[j];
    return world;
  }

  std::pair<Vector3<T>, Vector3<T>> bounds(T confidence_radius = T(3)) const {
    validate();
    if (!(confidence_radius > T(0)) || !std::isfinite(confidence_radius))
      throw std::invalid_argument(
          "confidence radius must be positive and finite");
    const Matrix3<T> sigma = covariance();
    Vector3<T> lower{}, upper{};
    for (size_t axis = 0; axis < 3; ++axis) {
      const T reach = confidence_radius * std::sqrt(sigma[axis][axis]);
      lower[axis] = mean[axis] - reach;
      upper[axis] = mean[axis] + reach;
    }
    return {lower, upper};
  }
};

using Gaussian3Df = Gaussian3D<float>;
using Gaussian3Dd = Gaussian3D<double>;

} // namespace gaussian
} // namespace riemann

#endif
