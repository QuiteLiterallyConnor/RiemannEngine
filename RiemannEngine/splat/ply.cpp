#include "splat/ply.hpp"

#include "splat/harmonics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace riemann {
namespace gaussian {
namespace {

constexpr float kC0 = static_cast<float>(kShC0);
constexpr float kLogitEpsilon = 1e-6f;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("PLY: " + message);
}

size_t scalar_width(const std::string &type) {
  if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
    return 1;
  if (type == "short" || type == "ushort" || type == "int16" ||
      type == "uint16")
    return 2;
  if (type == "int" || type == "uint" || type == "float" || type == "int32" ||
      type == "uint32" || type == "float32")
    return 4;
  if (type == "double" || type == "float64")
    return 8;
  fail("unsupported property type '" + type + "'");
}

struct Field {
  std::string name;
  std::string type;
  size_t offset = 0;
};

struct ElementBlock {
  std::string name;
  int64_t count = 0;
  size_t stride = 0;
  size_t file_offset = 0;
  std::vector<Field> fields;

  const Field *find(const std::string &field) const {
    for (const auto &f : fields)
      if (f.name == field)
        return &f;
    return nullptr;
  }

  const Field &require(const std::string &field) const {
    const Field *f = find(field);
    if (!f)
      fail("element '" + name + "' is missing property '" + field + "'");
    return *f;
  }
};

template <typename T> T unpack(const char *row, size_t offset) {
  T value;
  std::memcpy(&value, row + offset, sizeof(T));
  return value;
}

float unpack_float(const char *row, const Field &field) {
  if (field.type == "float" || field.type == "float32")
    return unpack<float>(row, field.offset);
  if (field.type == "double" || field.type == "float64")
    return static_cast<float>(unpack<double>(row, field.offset));
  fail("property '" + field.name + "' must be a float");
}

float mix(float low, float high, float t) { return low + (high - low) * t; }

void renormalize(float *q) {
  const float n =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12f) {
    q[0] = 1.f;
    q[1] = q[2] = q[3] = 0.f;
    return;
  }
  for (int i = 0; i < 4; ++i)
    q[i] /= n;
}

float activated_to_logit(float p) {
  p = std::min(std::max(p, kLogitEpsilon), 1.f - kLogitEpsilon);
  return std::log(p / (1.f - p));
}

// Number of f_rest_* properties -> SH degree, or -1 if the layout is unknown.
int degree_for_rest_total(int64_t total) {
  switch (total) {
  case 0:
    return 0;
  case 9:
    return 1;
  case 24:
    return 2;
  case 45:
    return 3;
  default:
    return -1;
  }
}

int64_t rest_property_count(const ElementBlock &element) {
  int64_t count = 0;
  while (element.find("f_rest_" + std::to_string(count)))
    ++count;
  return count;
}

// Compressed SH bytes: 0 -> -4, 255 -> 4, otherwise (v + 0.5) / 256 * 8 - 4.
float decode_sh_byte(uint8_t v) {
  const float t = v == 0 ? 0.f : (v == 255 ? 1.f : (v + 0.5f) / 256.f);
  return t * 8.f - 4.f;
}

struct ParsedFile {
  std::ifstream stream;
  std::vector<ElementBlock> elements;

  ElementBlock *element(const std::string &name) {
    for (auto &e : elements)
      if (e.name == name)
        return &e;
    return nullptr;
  }

  std::vector<char> read_block(const ElementBlock &element) {
    std::vector<char> bytes(element.stride *
                            static_cast<size_t>(element.count));
    stream.seekg(static_cast<std::streamoff>(element.file_offset));
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream && !bytes.empty())
      fail("failed reading element '" + element.name + "'");
    return bytes;
  }
};

ParsedFile parse_header(const std::string &path) {
  ParsedFile parsed;
  parsed.stream.open(path, std::ios::binary);
  if (!parsed.stream)
    fail("cannot open '" + path + "'");

  auto next_line = [&](std::string &line) {
    if (!std::getline(parsed.stream, line))
      return false;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    return true;
  };

  std::string line;
  if (!next_line(line) || line != "ply")
    fail("missing magic header in '" + path + "'");

  bool little_endian = false, closed = false;
  while (next_line(line)) {
    std::istringstream words(line);
    std::string keyword;
    words >> keyword;
    if (keyword == "format") {
      std::string format;
      words >> format;
      little_endian = format == "binary_little_endian";
    } else if (keyword == "element") {
      ElementBlock element;
      words >> element.name >> element.count;
      if (!words || element.count < 0)
        fail("malformed element line");
      parsed.elements.push_back(std::move(element));
    } else if (keyword == "property") {
      if (parsed.elements.empty())
        fail("property before element");
      Field field;
      words >> field.type >> field.name;
      if (field.type == "list")
        fail("list properties are not supported");
      ElementBlock &element = parsed.elements.back();
      field.offset = element.stride;
      element.stride += scalar_width(field.type);
      element.fields.push_back(std::move(field));
    } else if (keyword == "end_header") {
      closed = true;
      break;
    }
  }
  if (!closed)
    fail("missing end_header");
  if (!little_endian)
    fail("only binary_little_endian is supported");

  size_t cursor = static_cast<size_t>(parsed.stream.tellg());
  for (auto &element : parsed.elements) {
    element.file_offset = cursor;
    cursor += element.stride * static_cast<size_t>(element.count);
  }
  parsed.stream.seekg(0, std::ios::end);
  if (static_cast<size_t>(parsed.stream.tellg()) < cursor)
    fail("file is truncated");
  parsed.stream.clear();
  return parsed;
}

// Raw (pre-activation) parameters accumulated during decoding; rest holds the
// file's channel-major [3][K-1] layout per vertex.
struct RawSplats {
  int64_t n = 0;
  int degree = 0;
  std::vector<float> pos, dc, opacity, scale, rot, rest;
};

at::Tensor to_tensor(std::vector<float> &data, at::IntArrayRef shape) {
  return torch::from_blob(data.data(), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

void decode_compressed(ParsedFile &file, const ElementBlock &vertex,
                       bool load_sh, RawSplats &g) {
  const ElementBlock *chunk = file.element("chunk");
  if (!chunk)
    fail("compressed file has no chunk element");
  constexpr int64_t kChunkVertices = 256;
  if (chunk->count < (g.n + kChunkVertices - 1) / kChunkVertices)
    fail("too few chunks for vertex count");
  const std::vector<char> vertex_bytes = file.read_block(vertex);
  const std::vector<char> chunk_bytes = file.read_block(*chunk);

  const char *bound_names[18] = {
      "min_x",       "min_y",       "min_z",       "max_x",      "max_y",
      "max_z",       "min_scale_x", "min_scale_y", "min_scale_z",
      "max_scale_x", "max_scale_y", "max_scale_z", "min_r",      "min_g",
      "min_b",       "max_r",       "max_g",       "max_b"};
  const Field *bound_fields[18];
  for (int i = 0; i < 18; ++i)
    bound_fields[i] = &chunk->require(bound_names[i]);
  const Field &packed_position = vertex.require("packed_position");
  const Field &packed_rotation = vertex.require("packed_rotation");
  const Field &packed_scale = vertex.require("packed_scale");
  const Field &packed_color = vertex.require("packed_color");

  for (int64_t i = 0; i < g.n; ++i) {
    const char *row = vertex_bytes.data() + vertex.stride * i;
    const char *bounds =
        chunk_bytes.data() + chunk->stride * (i / kChunkVertices);
    float c[18];
    for (int k = 0; k < 18; ++k)
      c[k] = unpack_float(bounds, *bound_fields[k]);

    // Positions use 11/10/11-bit interpolation between the chunk bounds.
    const uint32_t p = unpack<uint32_t>(row, packed_position.offset);
    g.pos[3 * i + 0] = mix(c[0], c[3], ((p >> 21) & 2047u) / 2047.f);
    g.pos[3 * i + 1] = mix(c[1], c[4], ((p >> 11) & 1023u) / 1023.f);
    g.pos[3 * i + 2] = mix(c[2], c[5], (p & 2047u) / 2047.f);

    // 2-bit index of the omitted (largest) component, then three 10-bit
    // components scaled by sqrt(2).
    const uint32_t r = unpack<uint32_t>(row, packed_rotation.offset);
    const float root2 = 1.41421356237f;
    const float qa = (((r >> 20) & 1023u) / 1023.f - 0.5f) * root2;
    const float qb = (((r >> 10) & 1023u) / 1023.f - 0.5f) * root2;
    const float qc = ((r & 1023u) / 1023.f - 0.5f) * root2;
    const float qm = std::sqrt(std::max(0.f, 1.f - qa * qa - qb * qb - qc * qc));
    float q[4];
    switch (r >> 30) {
    case 0:
      q[0] = qm; q[1] = qa; q[2] = qb; q[3] = qc;
      break;
    case 1:
      q[0] = qa; q[1] = qm; q[2] = qb; q[3] = qc;
      break;
    case 2:
      q[0] = qa; q[1] = qb; q[2] = qm; q[3] = qc;
      break;
    default:
      q[0] = qa; q[1] = qb; q[2] = qc; q[3] = qm;
      break;
    }
    renormalize(q);
    std::copy(q, q + 4, &g.rot[4 * i]);

    // Chunk scale bounds are log-scales.
    const uint32_t s = unpack<uint32_t>(row, packed_scale.offset);
    g.scale[3 * i + 0] = mix(c[6], c[9], ((s >> 21) & 2047u) / 2047.f);
    g.scale[3 * i + 1] = mix(c[7], c[10], ((s >> 11) & 1023u) / 1023.f);
    g.scale[3 * i + 2] = mix(c[8], c[11], (s & 2047u) / 2047.f);

    // Packed color is RGB in [min, max] and alpha (already activated).
    const uint32_t k = unpack<uint32_t>(row, packed_color.offset);
    const float rgb[3] = {mix(c[12], c[15], ((k >> 24) & 255u) / 255.f),
                          mix(c[13], c[16], ((k >> 16) & 255u) / 255.f),
                          mix(c[14], c[17], ((k >> 8) & 255u) / 255.f)};
    for (int j = 0; j < 3; ++j)
      g.dc[3 * i + j] = (rgb[j] - 0.5f) / kC0;
    g.opacity[i] = activated_to_logit((k & 255u) / 255.f);
  }

  const ElementBlock *sh = file.element("sh");
  if (load_sh && sh) {
    const int64_t count = rest_property_count(*sh);
    g.degree = degree_for_rest_total(count);
    if (g.degree < 0) {
      g.degree = 0;
    } else if (sh->count != g.n) {
      fail("sh element count mismatch");
    } else if (count > 0) {
      const std::vector<char> sh_bytes = file.read_block(*sh);
      std::vector<size_t> offsets(static_cast<size_t>(count));
      for (int64_t j = 0; j < count; ++j)
        offsets[static_cast<size_t>(j)] =
            sh->require("f_rest_" + std::to_string(j)).offset;
      g.rest.resize(static_cast<size_t>(g.n * count));
      for (int64_t i = 0; i < g.n; ++i)
        for (int64_t j = 0; j < count; ++j)
          g.rest[static_cast<size_t>(i * count + j)] =
              decode_sh_byte(static_cast<uint8_t>(
                  sh_bytes[sh->stride * i + offsets[static_cast<size_t>(j)]]));
    }
  }
}

void decode_standard(ParsedFile &file, const ElementBlock &vertex,
                     bool load_sh, RawSplats &g) {
  const std::vector<char> vertex_bytes = file.read_block(vertex);
  const Field &px = vertex.require("x"), &py = vertex.require("y"),
              &pz = vertex.require("z");
  const Field &po = vertex.require("opacity");
  const Field *sc[3] = {&vertex.require("scale_0"), &vertex.require("scale_1"),
                        &vertex.require("scale_2")};
  const Field *rt[4] = {&vertex.require("rot_0"), &vertex.require("rot_1"),
                        &vertex.require("rot_2"), &vertex.require("rot_3")};
  const Field *dc[3] = {vertex.find("f_dc_0"), vertex.find("f_dc_1"),
                        vertex.find("f_dc_2")};

  int64_t rest_count = 0;
  std::vector<const Field *> rest_fields;
  if (load_sh) {
    rest_count = rest_property_count(vertex);
    g.degree = degree_for_rest_total(rest_count);
    if (g.degree < 0) {
      g.degree = 0;
      rest_count = 0;
    }
    for (int64_t j = 0; j < rest_count; ++j)
      rest_fields.push_back(&vertex.require("f_rest_" + std::to_string(j)));
    g.rest.resize(static_cast<size_t>(g.n * rest_count));
  }

  for (int64_t i = 0; i < g.n; ++i) {
    const char *row = vertex_bytes.data() + vertex.stride * i;
    g.pos[3 * i + 0] = unpack_float(row, px);
    g.pos[3 * i + 1] = unpack_float(row, py);
    g.pos[3 * i + 2] = unpack_float(row, pz);
    for (int j = 0; j < 3; ++j) {
      g.scale[3 * i + j] = unpack_float(row, *sc[j]);
      g.dc[3 * i + j] = dc[j] ? unpack_float(row, *dc[j]) : 0.f;
    }
    float q[4];
    for (int j = 0; j < 4; ++j)
      q[j] = unpack_float(row, *rt[j]);
    renormalize(q);
    std::copy(q, q + 4, &g.rot[4 * i]);
    g.opacity[i] = unpack_float(row, po);
    for (int64_t j = 0; j < rest_count; ++j)
      g.rest[static_cast<size_t>(i * rest_count + j)] =
          unpack_float(row, *rest_fields[static_cast<size_t>(j)]);
  }
}

GaussianModel read_model(const std::string &path, bool load_sh) {
  ParsedFile file = parse_header(path);
  const ElementBlock *vertex = file.element("vertex");
  if (!vertex)
    fail("no vertex element");

  RawSplats g;
  g.n = vertex->count;
  g.pos.resize(3 * g.n);
  g.dc.resize(3 * g.n);
  g.opacity.resize(g.n);
  g.scale.resize(3 * g.n);
  g.rot.resize(4 * g.n);

  if (vertex->find("packed_position"))
    decode_compressed(file, *vertex, load_sh, g);
  else
    decode_standard(file, *vertex, load_sh, g);

  const int64_t per_channel = sh_coefficient_count(g.degree) - 1;
  at::Tensor f_rest;
  if (per_channel == 0 || g.rest.empty())
    f_rest = at::zeros({g.n, per_channel, 3}, at::kFloat);
  else
    f_rest = to_tensor(g.rest, {g.n, 3, per_channel}).transpose(1, 2)
                 .contiguous();
  return GaussianModel(to_tensor(g.pos, {g.n, 3}), to_tensor(g.dc, {g.n, 3}),
                       f_rest, to_tensor(g.opacity, {g.n, 1}),
                       to_tensor(g.scale, {g.n, 3}),
                       to_tensor(g.rot, {g.n, 4}));
}

} // namespace

GaussianTensor load_gaussian_ply(const std::string &path) {
  return read_model(path, /*load_sh=*/false).to_gaussian_tensor();
}

GaussianModel load_gaussian_model_ply(const std::string &path) {
  return read_model(path, /*load_sh=*/true);
}

void save_gaussian_ply(const GaussianModel &model, const std::string &path) {
  const auto cpu = model.to(torch::kCPU);
  const int64_t n = cpu.size();
  const int64_t per_channel = cpu.f_rest().size(1);
  const int64_t rest_count = 3 * per_channel;

  std::ofstream file(path, std::ios::binary);
  if (!file)
    fail("cannot write '" + path + "'");

  std::ostringstream header;
  header << "ply\nformat binary_little_endian 1.0\nelement vertex " << n
         << "\n";
  for (const char *name :
       {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"})
    header << "property float " << name << "\n";
  for (int64_t i = 0; i < rest_count; ++i)
    header << "property float f_rest_" << i << "\n";
  for (const char *name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0",
                           "rot_1", "rot_2", "rot_3"})
    header << "property float " << name << "\n";
  header << "end_header\n";
  const std::string header_text = header.str();
  file.write(header_text.data(),
             static_cast<std::streamsize>(header_text.size()));

  const auto normals = at::zeros({n, 3}, at::kFloat);
  const auto rest =
      rest_count == 0
          ? at::zeros({n, 0}, at::kFloat)
          : cpu.f_rest().detach().transpose(1, 2).contiguous().reshape(
                {n, rest_count});
  const auto rows =
      at::cat({cpu.xyz().detach(), normals, cpu.f_dc().detach(), rest,
               cpu.opacity_raw().detach(), cpu.scale_raw().detach(),
               cpu.rotation_raw().detach()},
              1)
          .contiguous();
  file.write(reinterpret_cast<const char *>(rows.data_ptr<float>()),
             static_cast<std::streamsize>(rows.numel() * sizeof(float)));
  if (!file)
    fail("failed writing '" + path + "'");
}

void save_gaussian_ply(const GaussianTensor &gaussians,
                       const std::string &path) {
  save_gaussian_ply(GaussianModel::from_tensor(gaussians), path);
}

} // namespace gaussian
} // namespace riemann
