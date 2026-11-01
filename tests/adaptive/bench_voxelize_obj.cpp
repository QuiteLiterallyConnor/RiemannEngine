// OBJ volume voxelization benchmark (not part of the correctness suite).
// Loads a Wavefront OBJ mesh, voxelizes its solid volume at a configurable
// voxel budget as a fixed-resolution grid and as an adaptive (lossless
// octree) tensor, prints per-stage timings and writes cube-mesh OBJ + JSON
// voxel files for visualization (python tests/adaptive/view_voxels.py).
//
// Run: voxelize_obj_bench [resolution] [max_depth] [obj_path] [out_dir]
//   resolution  finest voxels along the longest bbox axis (default 128)
//   max_depth   adaptive octree depth; coarsest voxel = 2^max_depth finest
//               voxels per edge                         (default 3)
//   obj_path    input mesh (default tests/data/chair.obj)
//   out_dir     where voxel files are written           (default .)
#include "obj_voxelize.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>

#ifndef RIEMANN_TEST_DATA_DIR
#define RIEMANN_TEST_DATA_DIR "tests/data"
#endif

using namespace riemann_test::objvox;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Mean over `reps` runs after one warm-up run.
double time_ms(const std::function<void()>& f, int reps = 3) {
    f();
    auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) f();
    return ms_since(t0) / reps;
}

}  // namespace

int main(int argc, char** argv) {
    const int64_t resolution = argc > 1 ? std::atoll(argv[1]) : 128;
    const uint32_t depth = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 3;
    const std::string obj_path =
        argc > 3 ? argv[3] : std::string(RIEMANN_TEST_DATA_DIR) + "/chair.obj";
    const std::filesystem::path out_dir = argc > 4 ? argv[4] : ".";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "obj=" << obj_path << " resolution=" << resolution << " depth=" << depth
              << "\n";

    auto t0 = Clock::now();
    const Mesh mesh = load_obj(obj_path);
    std::cout << "load_obj_ms            " << ms_since(t0) << "\n";
    std::cout << "vertices               " << mesh.vertices.size() << "\n";
    std::cout << "triangles              " << mesh.triangles.size() << "\n";

    const auto grid = make_grid(mesh, resolution);
    std::cout << "finest_voxel_size      " << grid.voxel << "\n";
    std::cout << "grid_dims              " << grid.dims[0] << " x " << grid.dims[1] << " x "
              << grid.dims[2] << "\n";

    at::Tensor surface_cells;
    std::cout << "rasterize_surface_ms   "
              << time_ms([&] { surface_cells = rasterize_surface(mesh, grid); }) << "\n";
    at::Tensor cells;
    std::cout << "fill_interior_ms       "
              << time_ms([&] {
                     cells = surface_cells.clone();
                     fill_interior(cells);
                 })
              << "\n";
    const int64_t surface = (cells == kSurface).sum().item<int64_t>();
    const int64_t interior = (cells == kInterior).sum().item<int64_t>();
    std::cout << "surface_cells          " << surface << "\n";
    std::cout << "interior_cells         " << interior << "\n";

    std::cout << "voxelize_fixed_ms      " << time_ms([&] { voxelize_fixed(cells, grid); })
              << "\n";
    const auto fixed = voxelize_fixed(cells, grid);
    std::cout << "fixed_voxels           " << fixed.size() << "\n";

    std::cout << "voxelize_adaptive_ms   "
              << time_ms([&] { voxelize_adaptive(cells, grid, depth); }) << "\n";
    const auto adaptive = voxelize_adaptive(cells, grid, depth);
    std::cout << "adaptive_voxels        " << adaptive.size() << "\n";
    const auto counts = adaptive.counts_by_level();
    for (size_t l = 0; l < counts.size(); ++l)
        std::cout << "  level " << l << "              " << counts[l] << "\n";
    std::cout << "voxel_ratio            "
              << double(fixed.size()) / double(std::max<int64_t>(1, adaptive.size()))
              << "x fewer adaptive voxels\n";
    std::cout << "volume_fixed           " << fixed.volumes().sum().item<double>() << "\n";
    std::cout << "volume_adaptive        " << adaptive.volumes().sum().item<double>() << "\n";
    std::cout << "feature_bytes_fixed    " << fixed.features().nbytes() << "\n";
    std::cout << "feature_bytes_adaptive " << adaptive.features().nbytes() << "\n";

    std::filesystem::create_directories(out_dir);
    const std::string stem = std::filesystem::path(obj_path).stem().string();
    const std::string tag = "_r" + std::to_string(resolution);
    struct Output {
        std::string name;
        const riemann::adaptive::AdaptiveSparseTensor* tensor;
    } outputs[] = {{stem + "_fixed" + tag, &fixed},
                   {stem + "_adaptive" + tag + "_d" + std::to_string(depth), &adaptive}};
    for (const auto& o : outputs) {
        const auto obj_file = (out_dir / (o.name + ".obj")).string();
        const auto json_file = (out_dir / (o.name + ".json")).string();
        t0 = Clock::now();
        write_voxels_obj(*o.tensor, obj_file);
        write_voxels_json(*o.tensor, grid, json_file);
        std::cout << "write_ms               " << ms_since(t0) << "  -> " << json_file
                  << " (+ .obj)\n";
    }
    return 0;
}
