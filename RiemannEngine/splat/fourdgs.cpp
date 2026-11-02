#include "splat/fourdgs.hpp"
#include "splat/harmonics.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

namespace riemann {
namespace gaussian {
namespace {

using Bytes = std::vector<uint8_t>;
constexpr std::array<uint8_t, 8> kMagic{0x89, '4', 'D', 'G', 'S', '1', 13, 10};

void integer(Bytes &bytes, uint64_t value, int width) {
  for (int i = 0; i < width; ++i)
    bytes.push_back(uint8_t(value >> (8 * i)));
}

void real(Bytes &bytes, double value) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  integer(bytes, bits, 8);
}

void text(Bytes &bytes, const std::string &value) {
  TORCH_CHECK(value.size() <= UINT32_MAX, "4dgs string is too long");
  integer(bytes, value.size(), 4);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void mapping(Bytes &bytes, const std::map<std::string, std::string> &values) {
  Bytes block;
  for (const auto &entry : values) {
    text(block, entry.first);
    text(block, entry.second);
  }
  integer(bytes, block.size(), 4);
  bytes.insert(bytes.end(), block.begin(), block.end());
}

std::string decimal(double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream.precision(17);
  stream << value;
  return stream.str();
}

std::string bound_decimal(double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream.precision(17);
  stream << std::scientific << (value == 0 ? 0 : std::nextafter(
      value, std::numeric_limits<double>::infinity()));
  const auto number = stream.str();
  const auto exponent = number.find('e');
  const auto magnitude = number.substr(exponent + 2);
  return number.substr(0, exponent + 2) +
         std::string(3 - magnitude.size(), '0') + magnitude;
}

uint32_t crc32(const Bytes &bytes) {
  uint32_t crc = 0xffffffffu;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int i = 0; i < 8; ++i)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

Bytes framed(uint8_t opcode, const Bytes &body) {
  Bytes record;
  integer(record, opcode, 1);
  integer(record, body.size(), 8);
  record.insert(record.end(), body.begin(), body.end());
  return record;
}

int64_t bin(double value, double step, double &max_error, bool logarithmic = false) {
  const double rounded = std::round(value / step);
  TORCH_CHECK(std::isfinite(rounded) && rounded >= INT32_MIN && rounded <= INT32_MAX,
              "4dgs quantization exceeds signed 32-bit bins");
  const double decoded = rounded * step;
  const double reconstructed = logarithmic ? std::exp(decoded) : decoded;
  const float narrowed = static_cast<float>(reconstructed);
  TORCH_CHECK(std::isfinite(narrowed) && (!logarithmic || narrowed > 0),
              "4dgs decoded attribute cannot be represented as finite float32");
  const double error = logarithmic ? std::abs(std::log(double(narrowed)) - value)
                                   : std::abs(double(narrowed) - value);
  max_error = std::max(max_error, error);
  return static_cast<int64_t>(rounded);
}

// Attribute structures are bare inside a Chunk, not opcode-0x06 records.
Bytes attribute(uint8_t id, int channels, const std::vector<int64_t> &values) {
  TORCH_CHECK(values.size() % channels == 0, "invalid 4dgs attribute shape");
  uint64_t largest = 0;
  for (const auto value : values) {
    const uint64_t zigzag = value >= 0 ? uint64_t(value) * 2
                                      : uint64_t(-(value + 1)) * 2 + 1;
    largest = std::max(largest, zigzag);
  }
  TORCH_CHECK(largest <= UINT32_MAX, "4dgs symbol exceeds 32 bits");
  const int width = largest <= 255 ? 1 : largest <= 65535 ? 2 : 4;
  Bytes shuffled(values.size() * width);
  for (size_t i = 0; i < values.size(); ++i) {
    const auto value = values[i];
    const uint64_t zigzag = value >= 0 ? uint64_t(value) * 2
                                      : uint64_t(-(value + 1)) * 2 + 1;
    for (int plane = 0; plane < width; ++plane)
      shuffled[size_t(plane) * values.size() + i] = uint8_t(zigzag >> (8 * plane));
  }
  Bytes stream;
  integer(stream, id, 1);
  integer(stream, width, 1);
  integer(stream, 0, 1); // raw integer mode, with per-stream deflate codec 0
  integer(stream, 0, 1);
  integer(stream, channels, 1);
  integer(stream, values.size() / channels, 4);
  if (values.empty()) {
    integer(stream, 0, 8);
    return stream;
  }
  TORCH_CHECK(shuffled.size() <= INT_MAX, "4dgs stream exceeds codec capacity");
  int size = 0;
  std::unique_ptr<unsigned char, decltype(&std::free)> compressed(
      stbi_zlib_compress(shuffled.data(), int(shuffled.size()), &size, 8), &std::free);
  TORCH_CHECK(compressed && size > 0, "4dgs deflate compression failed");
  integer(stream, uint64_t(size), 8);
  stream.insert(stream.end(), compressed.get(), compressed.get() + size);
  return stream;
}

struct Index {
  uint64_t offset, length;
  uint32_t count;
  std::vector<std::array<uint64_t, 3>> bands;
};

} // namespace

FourDGSExportResult save_four_dgs(const Gaussian4DModel &source,
                                 const std::string &path,
                                 const FourDGSExportOptions &options) {
  torch::NoGradGuard guard;
  const double duration = (options.time_end - options.time_start) *
                          options.seconds_per_time_unit;
  TORCH_CHECK(std::isfinite(options.time_start) && std::isfinite(options.time_end) &&
                  std::isfinite(options.seconds_per_time_unit) &&
                  options.seconds_per_time_unit > 0 && std::isfinite(duration) &&
                  duration > 0 && duration <= std::numeric_limits<float>::max(),
              "4dgs export requires a finite positive duration and time scale");
  TORCH_CHECK(std::isfinite(options.cutoff) && options.cutoff > 0 &&
                  options.cutoff < 1, "4dgs cutoff must be in (0,1)");
  TORCH_CHECK(options.chunk_size > 0 && options.chunk_size <= 65536,
              "4dgs chunk_size must be in [1,65536]");
  TORCH_CHECK(source.size() <= UINT32_MAX, "too many Gaussians for this 4dgs exporter");
  const std::filesystem::path destination(path);
  TORCH_CHECK(!std::filesystem::exists(destination),
              "refusing to overwrite 4dgs destination: ", path);
  const auto model = source.to(torch::Device(torch::kCPU));
  for (const auto &parameter : model.parameters())
    TORCH_CHECK(at::isfinite(parameter).all().item<bool>(),
                "4dgs export requires finite model parameters");
  const auto rotation_norm = model.rotation_raw().norm(2, {1});
  TORCH_CHECK(at::isfinite(rotation_norm).all().item<bool>() &&
                  (rotation_norm > 1e-12).all().item<bool>(),
              "4dgs export requires finite nonzero quaternion norms");
  const auto scales = model.scales();
  const auto effective_sigma = model.time_scales().clamp_min(1e-6);
  const auto sigmas = effective_sigma * options.seconds_per_time_unit;
  TORCH_CHECK(at::isfinite(scales).all().item<bool>() &&
                  (scales > 0).all().item<bool>() &&
                  at::isfinite(sigmas).all().item<bool>() &&
                  (sigmas > 0).all().item<bool>(),
              "4dgs scales and temporal sigmas must be finite and positive");

  FourDGSExportResult result;
  result.gaussians = model.size();
  const auto rgb = 0.5 + kShC0 * model.f_dc();
  result.clamped_rgb_components = ((rgb < 0) | (rgb > 1)).sum().item<int64_t>();
  const auto rest = model.f_rest().contiguous();
  result.clamped_sh_coefficients =
      ((rest < -4) | (rest > 4)).sum().item<int64_t>();
  // Fine-profile grids. Keep position bins representable for large scenes.
  const double median = model.size() ? scales.median().item<double>() : 0.01;
  const double extent = model.size()
                            ? (model.xyz().amax(0) - model.xyz().amin(0))
                                  .max().item<double>() : 0;
  const double pos_step = std::max({median * 0.04, extent / 1e9, 1e-9});
  constexpr double scale_step = 0.01, rot_step = 0.001;
  constexpr double rgb_step = 1.0 / 255, alpha_step = 1.0 / 255;
  constexpr double time_step = 0.001, sigma_step = 0.01;
  const double motion_step = pos_step / 0.5;
  const auto origin_tensor = model.size() ? model.xyz().amin(0) : at::zeros({3});
  const auto maximum = model.size() ? model.xyz().amax(0) : at::zeros({3});
  std::array<double, 3> origin{};
  for (int i = 0; i < 3; ++i)
    origin[i] = origin_tensor[i].item<double>();

  const auto temporary = std::filesystem::path(path + ".tmp-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()));
  TORCH_CHECK(!std::filesystem::exists(temporary), "temporary export path already exists");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{temporary};
  std::ofstream file(temporary, std::ios::binary);
  TORCH_CHECK(file.good(), "cannot create 4dgs file: ", path);
  uint64_t offset = 0;
  auto write = [&](const Bytes &bytes) {
    file.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    TORCH_CHECK(file.good(), "failed writing 4dgs file: ", path);
    offset += bytes.size();
  };
  write(Bytes(kMagic.begin(), kMagic.end()));
  Bytes header;
  text(header, "");
  text(header, "RiemannEngine gaussian-birth exporter");
  real(header, duration);
  integer(header, model.size(), 8);
  real(header, options.cutoff);
  text(header, "gaussian-birth");
  for (const auto value : origin) real(header, value);
  for (int i = 0; i < 3; ++i) real(header, maximum[i].item<double>());
  integer(header, model.sh_degree(), 1);
  integer(header, 2, 1);
  mapping(header, {{"visibility_profile", "gaussian"},
                   {"color_space", "srgb-rec709-display"},
                   {"coordinate_system", "source world coordinates, unchanged"},
                   {"riemann.time_start", decimal(options.time_start)},
                   {"riemann.seconds_per_time_unit", decimal(options.seconds_per_time_unit)},
                   {"riemann.clamped_rgb_components", std::to_string(result.clamped_rgb_components)},
                   {"riemann.clamped_sh_coefficients", std::to_string(result.clamped_sh_coefficients)}});
  write(framed(1, header));
  Bytes quantization;
  text(quantization, "uniform-v1");
  for (const auto value : origin) real(quantization, value);
  for (const auto value : {pos_step, scale_step, rot_step, rgb_step,
                           alpha_step, motion_step, time_step, sigma_step})
    real(quantization, value);
  integer(quantization, 1, 1);
  const auto quantization_parameters = quantization;
  auto quantization_record = [&]() {
    Bytes body = quantization_parameters;
    mapping(body, {{"pos", bound_decimal(result.max_position_error)},
                  {"scale_rel", bound_decimal(result.max_scale_log_error)},
                  {"rot", bound_decimal(result.max_rotation_error)},
                  {"rgb", bound_decimal(result.max_rgb_error)},
                  {"alpha", bound_decimal(result.max_opacity_error)},
                  {"motion", bound_decimal(result.max_velocity_error)},
                  {"time", bound_decimal(result.max_time_error)},
                  {"sigma_rel", bound_decimal(result.max_sigma_log_error)},
                  {"sh", "0"}});
    return framed(3, body);
  };
  const auto quantization_offset = offset;
  const auto placeholder = quantization_record();
  write(placeholder);
  Bytes windows;
  integer(windows, 1, 4);
  real(windows, 0);
  real(windows, duration);
  write(framed(4, windows));

  const auto xyz = model.xyz().contiguous();
  const auto raw_scale = model.scale_raw().contiguous();
  const auto rotations = model.rotations().contiguous();
  const auto colors = rgb.clamp(0, 1).contiguous();
  const auto opacity = model.opacities().contiguous();
  const auto velocity = model.velocity().contiguous();
  const auto centers = model.time_center().contiguous();
  const auto raw_sigma = at::log(effective_sigma.to(at::kDouble)).contiguous();
  std::vector<Index> index;
  const double k = std::sqrt(-2 * std::log(options.cutoff));
  for (int64_t begin = 0; begin < model.size(); begin += options.chunk_size) {
    const auto count = std::min(options.chunk_size, model.size() - begin);
    std::array<std::vector<int64_t>, 11> attributes;
    std::array<std::vector<int64_t>, 3> bands;
    for (int64_t row = begin; row < begin + count; ++row) {
      const auto sigma_bin = bin(raw_sigma.data_ptr<double>()[row] +
                                      std::log(options.seconds_per_time_unit),
                                  sigma_step, result.max_sigma_log_error, true);
      const double decoded_sigma = std::exp(sigma_bin * sigma_step);
      const double half = std::clamp(k * decoded_sigma, 0.02, 2.0);
      const double life = std::clamp(std::ceil(std::log2(half / 0.5)), -4.0, 2.0);
      const double velocity_step = motion_step * std::pow(2, -life);
      const double target = std::min(time_step, 0.05 * decoded_sigma);
      const double time_class =
          std::clamp(std::floor(std::log2(target / time_step)), -10.0, 0.0);
      const double mu_step = time_step * std::pow(2, time_class);
      attributes[7].push_back(bin((centers.data_ptr<float>()[row] - options.time_start) *
                                      options.seconds_per_time_unit,
                                  mu_step, result.max_time_error));
      attributes[8].push_back(sigma_bin);
      attributes[9].push_back(0);
      attributes[10].push_back(0);
      std::array<int64_t, 3> color_bins{};
      for (int c = 0; c < 3; ++c) {
        double position_error = 0;
        const auto position_bin = bin(xyz.data_ptr<float>()[row * 3 + c] - origin[c],
                                       pos_step, position_error);
        const double decoded_position = double(float(origin[c] + position_bin * pos_step));
        TORCH_CHECK(std::isfinite(decoded_position), "4dgs position overflow");
        result.max_position_error = std::max(result.max_position_error,
            std::abs(decoded_position - xyz.data_ptr<float>()[row * 3 + c]));
        attributes[0].push_back(position_bin);
        attributes[1].push_back(bin(raw_scale.data_ptr<float>()[row * 3 + c],
                                    scale_step, result.max_scale_log_error, true));
        attributes[6].push_back(bin(velocity.data_ptr<float>()[row * 3 + c] /
                                        options.seconds_per_time_unit,
                                    velocity_step, result.max_velocity_error));
        color_bins[c] = bin(colors.data_ptr<float>()[row * 3 + c], rgb_step,
                             result.max_rgb_error);
      }
      attributes[4].insert(attributes[4].end(),
                            {color_bins[1], color_bins[0] - color_bins[1],
                             color_bins[2] - color_bins[1]});
      attributes[5].push_back(bin(opacity.data_ptr<float>()[row], alpha_step,
                                  result.max_opacity_error));
      std::array<double, 4> q{};
      for (int c = 0; c < 4; ++c)
        q[c] = rotations.data_ptr<float>()[row * 4 + (c + 1) % 4];
      const int largest = int(std::max_element(q.begin(), q.end(),
          [](double a, double b) { return std::abs(a) < std::abs(b); }) - q.begin());
      const double sign = q[largest] < 0 ? -1 : 1;
      double sum = 0, unused_error = 0;
      std::array<double, 4> decoded_q{};
      for (int c = 0; c < 4; ++c) {
        if (c == largest) continue;
        const auto rotation_bin = bin(q[c] * sign, rot_step, unused_error);
        attributes[3].push_back(rotation_bin);
        decoded_q[c] = rotation_bin * rot_step;
        sum += decoded_q[c] * decoded_q[c];
      }
      decoded_q[largest] = std::sqrt(std::max(0.0, 1 - sum));
      double norm = 0;
      for (const auto value : decoded_q) norm += value * value;
      for (int c = 0; c < 4; ++c)
        result.max_rotation_error = std::max(result.max_rotation_error,
            std::abs(double(float(decoded_q[c] / std::sqrt(norm))) - q[c] * sign));
      attributes[2].push_back(largest);
      for (int band = 1; band <= model.sh_degree(); ++band) {
        const int start = band * band - 1, width = 2 * band + 1;
        for (int c = 0; c < 3; ++c)
          for (int coefficient = start; coefficient < start + width; ++coefficient) {
            const double value = std::clamp(
                double(rest.data_ptr<float>()[(row * rest.size(1) + coefficient) * 3 + c]),
                -4.0, 4.0);
            const int64_t code = int64_t(std::round((value + 4) * 255 / 8));
            bands[band - 1].push_back(code);
            result.max_sh_error = std::max(result.max_sh_error,
                                           std::abs(-4 + code * 8.0 / 255 - value));
          }
      }
    }
    constexpr std::array<int, 11> channels{3, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1};
    Bytes streams;
    for (int id = 0; id < 11; ++id) {
      const auto stream = attribute(uint8_t(id), channels[id], attributes[id]);
      streams.insert(streams.end(), stream.begin(), stream.end());
    }
    Bytes chunk;
    real(chunk, 0);
    real(chunk, duration);
    integer(chunk, 0, 4);
    integer(chunk, count, 4);
    text(chunk, "");
    integer(chunk, streams.size(), 8);
    integer(chunk, streams.size(), 8);
    chunk.insert(chunk.end(), streams.begin(), streams.end());
    const auto chunk_record = framed(5, chunk);
    Index entry{offset, chunk_record.size(), uint32_t(count), {}};
    write(chunk_record);
    for (int band = 1; band <= model.sh_degree(); ++band) {
      Bytes body;
      integer(body, band, 1);
      const auto stream = attribute(7, 3 * (2 * band + 1), bands[band - 1]);
      body.insert(body.end(), stream.begin(), stream.end());
      const auto record = framed(7, body);
      entry.bands.push_back({uint64_t(band), offset, record.size()});
      write(record);
    }
    index.push_back(std::move(entry));
  }
  const uint64_t summary_start = index.empty() ? 0 : offset;
  Bytes summary;
  for (const auto &entry : index) {
    Bytes body;
    real(body, 0);
    real(body, duration);
    integer(body, entry.offset, 8);
    integer(body, entry.length, 8);
    integer(body, entry.count, 4);
    integer(body, entry.bands.size(), 4);
    for (const auto &band : entry.bands) {
      integer(body, band[0], 1);
      integer(body, band[1], 8);
      integer(body, band[2], 8);
    }
    const auto record = framed(8, body);
    summary.insert(summary.end(), record.begin(), record.end());
  }
  write(summary);
  Bytes footer;
  integer(footer, summary_start, 8);
  integer(footer, 0, 8);
  integer(footer, summary.empty() ? 0 : crc32(summary), 4);
  write(framed(2, footer));
  write(Bytes(kMagic.begin(), kMagic.end()));
  // Fixed-width decimals let us publish measured float32 decode errors in front matter.
  const auto measured = quantization_record();
  TORCH_CHECK(measured.size() == placeholder.size(), "4dgs bounds size changed");
  file.seekp(std::streamoff(quantization_offset));
  TORCH_CHECK(file.good(), "failed seeking 4dgs quantization record");
  file.write(reinterpret_cast<const char *>(measured.data()),
             std::streamsize(measured.size()));
  file.flush();
  TORCH_CHECK(file.good(), "failed to flush 4dgs file: ", path);
  file.close();
  TORCH_CHECK(!file.fail(), "failed to close 4dgs file: ", path);
  std::filesystem::rename(temporary, destination);
  result.bytes = int64_t(offset);
  return result;
}

} // namespace gaussian
} // namespace riemann
