// Shared helpers for voxelizing Wavefront OBJ meshes with the adaptive engine:
// a minimal OBJ loader, surface rasterization + flood-fill solid filling,
// fixed- and adaptive-resolution (lossless octree) volume voxelization, and
// voxel writers (cube-mesh OBJ + JSON) for visualization. Used by
// test_voxelize_obj.cpp, bench_voxelize_obj.cpp and view_voxels.py.
#ifndef RIEMANN_TESTS_ADAPTIVE_OBJ_VOXELIZE_HPP
#define RIEMANN_TESTS_ADAPTIVE_OBJ_VOXELIZE_HPP

#include "RiemannEngine/dynamic/dynamic.hpp"
#include "third_party/json.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace riemann_test {
namespace objvox {

struct Mesh {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int64_t, 3>> triangles;  // fan-triangulated faces
};

// Minimal Wavefront OBJ reader: keeps `v` positions and `f` faces (any arity,
// v/vt/vn and negative indices supported); everything else is ignored.
inline Mesh load_obj(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open OBJ file: " + path);
    Mesh mesh;
    std::string line;
    std::vector<int64_t> face;
    while (std::getline(in, line)) {
        const char* s = line.c_str();
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
            char* end = nullptr;
            const double x = std::strtod(s + 2, &end);
            const double y = std::strtod(end, &end);
            const double z = std::strtod(end, &end);
            mesh.vertices.push_back({x, y, z});
        } else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            face.clear();
            const char* p = s + 2;
            while (*p) {
                while (*p == ' ' || *p == '\t') ++p;
                char* end = nullptr;
                const long long idx = std::strtoll(p, &end, 10);
                if (end == p) break;
                const int64_t v =
                    idx > 0 ? idx - 1 : static_cast<int64_t>(mesh.vertices.size()) + idx;
                if (v < 0)
                    throw std::runtime_error("OBJ face references vertex before any 'v' line");
                face.push_back(v);
                p = end;
                while (*p && *p != ' ' && *p != '\t') ++p;  // skip /vt/vn suffix
            }
            for (size_t i = 2; i < face.size(); ++i)
                mesh.triangles.push_back({face[0], face[i - 1], face[i]});
        }
    }
    for (const auto& t : mesh.triangles)
        for (int64_t v : t)
            if (v >= static_cast<int64_t>(mesh.vertices.size()))
                throw std::runtime_error("OBJ face references out-of-range vertex");
    return mesh;
}

// Finest-resolution grid derived from the mesh bounds and a user-chosen voxel
// budget: `resolution` voxels along the longest bounding-box axis.
struct GridSpec {
    std::array<double, 3> lo{};
    std::array<double, 3> hi{};
    double voxel = 0.0;             // finest voxel edge length (cubic voxels)
    std::array<int64_t, 3> dims{};  // finest-level cells per axis
};

inline GridSpec make_grid(const Mesh& mesh, int64_t resolution) {
    if (mesh.vertices.empty())
        throw std::runtime_error("mesh has no vertices");
    if (resolution < 1)
        throw std::runtime_error("resolution must be >= 1");
    GridSpec g;
    g.lo = g.hi = mesh.vertices.front();
    for (const auto& v : mesh.vertices)
        for (int d = 0; d < 3; ++d) {
            g.lo[d] = std::min(g.lo[d], v[d]);
            g.hi[d] = std::max(g.hi[d], v[d]);
        }
    double longest = 0.0;
    for (int d = 0; d < 3; ++d) longest = std::max(longest, g.hi[d] - g.lo[d]);
    if (longest <= 0.0)
        throw std::runtime_error("mesh is degenerate (zero extent)");
    g.voxel = longest / static_cast<double>(resolution);
    for (int d = 0; d < 3; ++d)
        g.dims[d] = std::max<int64_t>(
            1, static_cast<int64_t>(std::ceil((g.hi[d] - g.lo[d]) / g.voxel)));
    return g;
}

// Dense occupancy grid codes produced by rasterize_surface / fill_interior.
enum : uint8_t { kEmpty = 0, kSurface = 1, kInterior = 2 };

// Marks every finest-level cell the mesh surface passes through as kSurface.
// Each triangle is sampled on a barycentric lattice with spacing <= half a
// voxel, so consecutive samples never skip a cell and the marked shell is
// 26-connected (it cannot be crossed by a 6-connected flood fill).
// Returns a uint8 tensor of shape [dims[0], dims[1], dims[2]].
inline at::Tensor rasterize_surface(const Mesh& mesh, const GridSpec& grid) {
    const int64_t X = grid.dims[0], Y = grid.dims[1], Z = grid.dims[2];
    auto cells = torch::zeros({X, Y, Z}, torch::kUInt8);
    uint8_t* c = cells.data_ptr<uint8_t>();
    const double spacing = grid.voxel * 0.5;
    const double inv_voxel = 1.0 / grid.voxel;
    auto cell_of = [&](double p, int d) {
        const auto q = static_cast<int64_t>(std::floor((p - grid.lo[d]) * inv_voxel));
        return std::min<int64_t>(std::max<int64_t>(q, 0), grid.dims[d] - 1);
    };
    for (const auto& tri : mesh.triangles) {
        const auto& a = mesh.vertices[tri[0]];
        const auto& b = mesh.vertices[tri[1]];
        const auto& v2 = mesh.vertices[tri[2]];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3] = {v2[0] - a[0], v2[1] - a[1], v2[2] - a[2]};
        double longest_edge = 0.0;
        for (int d = 0; d < 3; ++d) {
            longest_edge = std::max(longest_edge, std::fabs(e1[d]));
            longest_edge = std::max(longest_edge, std::fabs(e2[d]));
            longest_edge = std::max(longest_edge, std::fabs(e2[d] - e1[d]));
        }
        const int64_t n = std::max<int64_t>(
            1, static_cast<int64_t>(std::ceil(longest_edge / spacing)));
        const double inv = 1.0 / static_cast<double>(n);
        for (int64_t i = 0; i <= n; ++i)
            for (int64_t j = 0; j <= n - i; ++j) {
                const double u = i * inv, v = j * inv;
                const int64_t x = cell_of(a[0] + e1[0] * u + e2[0] * v, 0);
                const int64_t y = cell_of(a[1] + e1[1] * u + e2[1] * v, 1);
                const int64_t z = cell_of(a[2] + e1[2] * u + e2[2] * v, 2);
                c[(x * Y + y) * Z + z] = kSurface;
            }
    }
    return cells;
}

// Solid fill: flood-fills empty space 6-connected from the grid boundary;
// every empty cell the flood cannot reach is enclosed by the surface and is
// marked kInterior. Modifies `cells` in place. Open meshes (holes wider than
// a voxel) leak and stay hollow, which is the correct answer for them.
inline void fill_interior(at::Tensor& cells) {
    const int64_t X = cells.size(0), Y = cells.size(1), Z = cells.size(2);
    uint8_t* c = cells.data_ptr<uint8_t>();
    constexpr uint8_t kOutside = 3;
    std::vector<int64_t> stack;
    auto push = [&](int64_t x, int64_t y, int64_t z) {
        const int64_t i = (x * Y + y) * Z + z;
        if (c[i] == kEmpty) {
            c[i] = kOutside;
            stack.push_back(i);
        }
    };
    for (int64_t x = 0; x < X; ++x)
        for (int64_t y = 0; y < Y; ++y) {
            push(x, y, 0);
            push(x, y, Z - 1);
        }
    for (int64_t x = 0; x < X; ++x)
        for (int64_t z = 0; z < Z; ++z) {
            push(x, 0, z);
            push(x, Y - 1, z);
        }
    for (int64_t y = 0; y < Y; ++y)
        for (int64_t z = 0; z < Z; ++z) {
            push(0, y, z);
            push(X - 1, y, z);
        }
    while (!stack.empty()) {
        const int64_t i = stack.back();
        stack.pop_back();
        const int64_t z = i % Z, y = (i / Z) % Y, x = i / (Y * Z);
        if (x > 0) push(x - 1, y, z);
        if (x + 1 < X) push(x + 1, y, z);
        if (y > 0) push(x, y - 1, z);
        if (y + 1 < Y) push(x, y + 1, z);
        if (z > 0) push(x, y, z - 1);
        if (z + 1 < Z) push(x, y, z + 1);
    }
    const int64_t n = cells.numel();
    for (int64_t i = 0; i < n; ++i)
        c[i] = c[i] == kEmpty ? kInterior : (c[i] == kOutside ? kEmpty : c[i]);
}

// Fixed resolution: every occupied (surface or interior) finest-level cell as
// one uniform grid.voxel-sized voxel. Feature column 0 is occupancy (always
// 1), column 1 is 1 for surface cells and 0 for interior cells.
inline riemann::adaptive::AdaptiveSparseTensor voxelize_fixed(const at::Tensor& cells,
                                                              const GridSpec& grid) {
    riemann::adaptive::AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {grid.voxel, grid.voxel, grid.voxel};
    cfg.origin = {grid.lo[0], grid.lo[1], grid.lo[2]};
    cfg.max_depth = 0;
    auto coords = (cells != kEmpty).nonzero();
    auto codes = cells.index({coords.select(1, 0), coords.select(1, 1), coords.select(1, 2)});
    auto feats = torch::stack({torch::ones({coords.size(0)}, torch::kFloat64),
                               (codes == kSurface).to(torch::kFloat64)},
                              1);
    return riemann::adaptive::AdaptiveSparseTensor::from_voxels(
        coords, torch::zeros({coords.size(0)}, torch::kInt32), feats, cfg);
}

// Adaptive resolution (lossless octree): the finest grid is padded to a
// multiple of 2^max_depth, then every 2x2x2 block whose cells are all occupied
// is merged into one parent, recursively up to level 0 (edge grid.voxel *
// 2^max_depth). Leaves cover exactly the same cells as voxelize_fixed, so the
// shape and total volume are identical; solid interiors become a few large
// cells while the surface keeps the finest resolution. Feature column 0 is
// occupancy, column 1 is the fraction of the leaf's finest cells that are
// surface cells.
inline riemann::adaptive::AdaptiveSparseTensor voxelize_adaptive(const at::Tensor& cells,
                                                                 const GridSpec& grid,
                                                                 uint32_t max_depth) {
    using torch::indexing::Slice;
    const int64_t block = int64_t(1) << max_depth;
    std::array<int64_t, 3> padded{};
    for (int d = 0; d < 3; ++d) padded[d] = (grid.dims[d] + block - 1) / block * block;

    // occ[l] / surf[l]: per-cell occupancy (all finest cells occupied) and
    // surface-cell count at level l; level max_depth is the finest grid.
    std::vector<at::Tensor> occ(max_depth + 1), surf(max_depth + 1);
    occ[max_depth] = torch::zeros({padded[0], padded[1], padded[2]}, torch::kUInt8);
    surf[max_depth] = torch::zeros({padded[0], padded[1], padded[2]}, torch::kFloat64);
    occ[max_depth].index_put_({Slice(0, grid.dims[0]), Slice(0, grid.dims[1]),
                               Slice(0, grid.dims[2])},
                              (cells != kEmpty).to(torch::kUInt8));
    surf[max_depth].index_put_({Slice(0, grid.dims[0]), Slice(0, grid.dims[1]),
                                Slice(0, grid.dims[2])},
                               (cells == kSurface).to(torch::kFloat64));
    for (uint32_t l = max_depth; l-- > 0;) {
        const auto s = occ[l + 1].sizes();
        const std::vector<int64_t> blocks = {s[0] / 2, 2, s[1] / 2, 2, s[2] / 2, 2};
        occ[l] = occ[l + 1].view(blocks).amin({1, 3, 5});
        surf[l] = surf[l + 1].view(blocks).sum({1, 3, 5});
    }

    std::vector<at::Tensor> coords, levels, feats;
    for (uint32_t l = 0; l <= max_depth; ++l) {
        auto leaf = occ[l].to(torch::kBool);
        if (l > 0) {
            auto parent_full = occ[l - 1]
                                   .repeat_interleave(2, 0)
                                   .repeat_interleave(2, 1)
                                   .repeat_interleave(2, 2)
                                   .to(torch::kBool);
            leaf = leaf & ~parent_full;
        }
        auto c = leaf.nonzero();
        const int64_t n = c.size(0);
        const double cells_per_leaf = std::pow(8.0, double(max_depth - l));
        auto surface_fraction =
            surf[l].index({c.select(1, 0), c.select(1, 1), c.select(1, 2)}) / cells_per_leaf;
        coords.push_back(c);
        levels.push_back(torch::full({n}, static_cast<int32_t>(l), torch::kInt32));
        feats.push_back(
            torch::stack({torch::ones({n}, torch::kFloat64), surface_fraction}, 1));
    }

    const double base = grid.voxel * static_cast<double>(block);
    riemann::adaptive::AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {base, base, base};
    cfg.min_voxel_size = {grid.voxel, grid.voxel, grid.voxel};
    cfg.origin = {grid.lo[0], grid.lo[1], grid.lo[2]};
    cfg.max_depth = max_depth;
    return riemann::adaptive::AdaptiveSparseTensor::from_voxels(
        torch::cat(coords), torch::cat(levels), torch::cat(feats), cfg);
}


// Writes each voxel as a cube (8 vertices, 6 quad faces) in OBJ format, which
// any mesh viewer (Blender, MeshLab, ...) can open directly. A face is
// omitted when the neighbouring cell across it is covered by a leaf of equal
// or larger size (it can never be seen), and fully enclosed voxels are
// skipped, so solid volumes only emit their visible boundary.
inline void write_voxels_obj(const riemann::adaptive::AdaptiveSparseTensor& tensor,
                             const std::string& path) {
    auto centers = tensor.centers().contiguous();
    auto sizes = tensor.voxel_sizes().contiguous();
    auto coords = tensor.coordinates().contiguous();
    auto levels = tensor.levels().contiguous();
    const double* c = centers.data_ptr<double>();
    const double* s = sizes.data_ptr<double>();
    const int64_t* cd = coords.data_ptr<int64_t>();
    const int32_t* ld = levels.data_ptr<int32_t>();
    const int64_t n = tensor.size();
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write voxel OBJ file: " + path);
    out << "# voxelization: " << n << " voxels as cubes (hidden faces culled)\n";
    static const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                     {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    static const int face[6][4] = {{1, 4, 3, 2}, {5, 6, 7, 8}, {1, 2, 6, 5},
                                   {2, 3, 7, 6}, {3, 4, 8, 7}, {4, 1, 5, 8}};
    static const int normal[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0},
                                     {1, 0, 0},  {0, 1, 0}, {-1, 0, 0}};
    auto neighbour_covered = [&](int64_t i, const int* dir) {
        const uint32_t level = static_cast<uint32_t>(ld[i]);
        const int64_t q[3] = {cd[i * 3 + 0] + dir[0], cd[i * 3 + 1] + dir[1],
                              cd[i * 3 + 2] + dir[2]};
        for (uint32_t up = 0; up <= level; ++up)
            if (tensor.index_of({q[0] >> up, q[1] >> up, q[2] >> up}, level - up) >= 0)
                return true;
        return false;
    };
    std::string buf;
    buf.reserve(1 << 20);
    char line[192];
    long long written = 0;  // vertices emitted so far (OBJ indices are 1-based)
    for (int64_t i = 0; i < n; ++i) {
        bool visible[6];
        bool any = false;
        for (int f = 0; f < 6; ++f) any |= (visible[f] = !neighbour_covered(i, normal[f]));
        if (!any) continue;
        for (const auto& k : corner) {
            std::snprintf(line, sizeof(line), "v %.9g %.9g %.9g\n",
                          c[i * 3 + 0] + (k[0] - 0.5) * s[i * 3 + 0],
                          c[i * 3 + 1] + (k[1] - 0.5) * s[i * 3 + 1],
                          c[i * 3 + 2] + (k[2] - 0.5) * s[i * 3 + 2]);
            buf += line;
        }
        for (int f = 0; f < 6; ++f) {
            if (!visible[f]) continue;
            std::snprintf(line, sizeof(line), "f %lld %lld %lld %lld\n",
                          written + face[f][0], written + face[f][1], written + face[f][2],
                          written + face[f][3]);
            buf += line;
        }
        written += 8;
        if (buf.size() > (1u << 20) - 1024) {
            out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
            buf.clear();
        }
    }
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    if (!out)
        throw std::runtime_error("failed while writing voxel OBJ file: " + path);
}

// Writes the voxel set as JSON in the style of tests/data/voxels/*.json:
// grid_info plus one record per voxel with integer coordinates (at the voxel's
// own level), center position, edge length, refinement level and the fraction
// of the voxel's finest cells that lie on the mesh surface (feature column 1).
inline void write_voxels_json(const riemann::adaptive::AdaptiveSparseTensor& tensor,
                              const GridSpec& grid, const std::string& path) {
    auto coords = tensor.coordinates().contiguous();
    auto levels = tensor.levels().contiguous();
    auto centers = tensor.centers().contiguous();
    auto sizes = tensor.voxel_sizes().contiguous();
    auto features = tensor.features().to(torch::kFloat64).cpu().contiguous();
    const int64_t* cd = coords.data_ptr<int64_t>();
    const int32_t* ld = levels.data_ptr<int32_t>();
    const double* ct = centers.data_ptr<double>();
    const double* sz = sizes.data_ptr<double>();
    const double* ft = features.data_ptr<double>();
    const int64_t channels = features.size(1);
    const int64_t n = tensor.size();

    nlohmann::json doc;
    doc["grid_info"] = {
        {"active_voxels", n},
        {"exported_voxels", n},
        {"dimensions", {grid.dims[0], grid.dims[1], grid.dims[2]}},
        {"origin_m", {grid.lo[0], grid.lo[1], grid.lo[2]}},
        {"voxel_size_m", grid.voxel},
    };
    auto& voxels = doc["voxels"] = nlohmann::json::array();
    for (int64_t i = 0; i < n; ++i)
        voxels.push_back({
            {"coordinates", {cd[i * 3 + 0], cd[i * 3 + 1], cd[i * 3 + 2]}},
            {"level", ld[i]},
            {"size_m", sz[i * 3 + 0]},
            {"position_m", {ct[i * 3 + 0], ct[i * 3 + 1], ct[i * 3 + 2]}},
            {"intensity", 1.0},
            {"surface", channels > 1 ? ft[i * channels + 1] : 1.0},
        });

    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write voxel JSON file: " + path);
    out << doc.dump(2) << "\n";
    if (!out)
        throw std::runtime_error("failed while writing voxel JSON file: " + path);
}

}  // namespace objvox
}  // namespace riemann_test

#endif
