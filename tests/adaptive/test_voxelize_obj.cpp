// OBJ-mesh volume voxelization tests: load tests/data/chair.obj, produce a
// solid fixed-resolution and adaptive-resolution (lossless octree)
// voxelization, benchmark the stages and emit voxel files (cube-mesh OBJ +
// JSON) for visualization with view_voxels.py.
//
// The voxel budget is configurable via environment variables:
//   RIEMANN_CHAIR_RESOLUTION   voxels along the longest bbox axis (default 64)
//   RIEMANN_CHAIR_DEPTH        adaptive octree depth               (default 3)
#include "obj_voxelize.hpp"
#include "utils/test_framework.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

#ifndef RIEMANN_TEST_DATA_DIR
#define RIEMANN_TEST_DATA_DIR "tests/data"
#endif

namespace {

using namespace riemann_test::objvox;
using riemann::adaptive::AdaptiveSparseTensor;
using Clock = std::chrono::steady_clock;

int64_t env_int(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoll(v) : fallback;
}

double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Axis-aligned box [0,sx]x[0,sy]x[0,sz] as quads with v/vt/vn-style indices
// (the same OBJ constructs chair.obj uses); `open_top` omits the +z face.
Mesh box_mesh(double sx, double sy, double sz, bool open_top = false) {
    auto path = std::filesystem::temp_directory_path() / "riemann_objvox_box.obj";
    {
        std::ofstream out(path);
        out << "# box\n"
            << "v 0 0 0\nv " << sx << " 0 0\nv " << sx << " " << sy << " 0\nv 0 " << sy
            << " 0\n"
            << "v 0 0 " << sz << "\nv " << sx << " 0 " << sz << "\nv " << sx << " " << sy
            << " " << sz << "\nv 0 " << sy << " " << sz << "\n"
            << "f 1/1/1 4/2/1 3/3/1 2/4/1\n"
            << "f 1/1/3 2/2/3 6/3/3 5/4/3\n"
            << "f 2/1/4 3/2/4 7/3/4 6/4/4\n"
            << "f 3/1/5 4/2/5 8/3/5 7/4/5\n"
            << "f 4/1/6 1/2/6 5/3/6 8/4/6\n";
        if (!open_top) out << "f 5/1/2 6/2/2 7/3/2 8/4/2\n";
    }
    auto mesh = load_obj(path.string());
    std::filesystem::remove(path);
    return mesh;
}

at::Tensor solid_cells(const Mesh& mesh, const GridSpec& grid) {
    auto cells = rasterize_surface(mesh, grid);
    fill_interior(cells);
    return cells;
}

double total_volume(const AdaptiveSparseTensor& t) { return t.volumes().sum().item<double>(); }

// True when the finest-level cell `coordinate` is covered by some leaf of the
// adaptive tensor (the leaf may sit at any coarser level).
bool covered_by_adaptive(const AdaptiveSparseTensor& t, std::array<int64_t, 3> coordinate,
                         uint32_t finest_level) {
    for (uint32_t level = finest_level + 1; level-- > 0;) {
        const uint32_t shift = finest_level - level;
        if (t.index_of({coordinate[0] >> shift, coordinate[1] >> shift,
                        coordinate[2] >> shift},
                       level) >= 0)
            return true;
    }
    return false;
}

}  // namespace

TEST(voxelize_obj_unit_cube_fixed_is_solid) {
    auto mesh = box_mesh(1, 1, 1);
    CHECK_EQ(mesh.vertices.size(), size_t(8));
    CHECK_EQ(mesh.triangles.size(), size_t(12));  // six quads, fan-triangulated

    auto grid = make_grid(mesh, 4);
    CHECK_NEAR(grid.voxel, 0.25, 1e-12);
    auto cells = solid_cells(mesh, grid);
    CHECK_EQ((cells == kSurface).sum().item<int64_t>(), int64_t(4 * 4 * 4 - 2 * 2 * 2));
    CHECK_EQ((cells == kInterior).sum().item<int64_t>(), int64_t(2 * 2 * 2));

    auto fixed = voxelize_fixed(cells, grid);
    CHECK_EQ(fixed.size(), int64_t(64));
    CHECK(torch::equal(fixed.levels(), torch::zeros({64}, torch::kInt32)));
    CHECK_NEAR(total_volume(fixed), 1.0, 1e-12);
    CHECK_NEAR(fixed.features().select(1, 1).sum().item<double>(), 56.0, 1e-12);

    // Hidden-face culling: only the 6 * 4 * 4 outer faces remain.
    auto obj = std::filesystem::temp_directory_path() / "riemann_objvox_cube_voxels.obj";
    write_voxels_obj(fixed, obj.string());
    auto reread = load_obj(obj.string());
    std::filesystem::remove(obj);
    CHECK_EQ(reread.triangles.size(), size_t(2 * 6 * 4 * 4));
    CHECK_EQ(reread.vertices.size(), size_t(8 * 56));  // enclosed voxels skipped
}

TEST(voxelize_obj_unit_cube_adaptive_merges_solid_blocks) {
    auto mesh = box_mesh(1, 1, 1);
    auto grid = make_grid(mesh, 8);
    auto cells = solid_cells(mesh, grid);
    CHECK_EQ(voxelize_fixed(cells, grid).size(), int64_t(512));

    // Depth 2: four-cell blocks, the solid 8^3 cube is exactly 8 such blocks.
    auto a2 = voxelize_adaptive(cells, grid, 2);
    CHECK_EQ(a2.size(), int64_t(8));
    CHECK_EQ(a2.counts_by_level()[0], int64_t(8));
    CHECK_NEAR(total_volume(a2), 1.0, 1e-12);

    // Depth 3: the whole cube collapses into a single level-0 voxel.
    auto a3 = voxelize_adaptive(cells, grid, 3);
    CHECK_EQ(a3.size(), int64_t(1));
    CHECK_NEAR(a3.voxel_sizes().max().item<double>(), 1.0, 1e-12);
    CHECK(!a3.has_overlapping_leaves());
}

TEST(voxelize_obj_adaptive_handles_unaligned_grid) {
    // 4 x 4 x 2 finest cells; padding to 4^3 must not create phantom voxels.
    auto mesh = box_mesh(1, 1, 0.5);
    auto grid = make_grid(mesh, 4);
    CHECK_EQ(grid.dims[2], int64_t(2));
    auto cells = solid_cells(mesh, grid);
    auto fixed = voxelize_fixed(cells, grid);
    auto adaptive = voxelize_adaptive(cells, grid, 2);
    CHECK_EQ(fixed.size(), int64_t(32));
    CHECK_EQ(adaptive.size(), int64_t(4));  // four 2x2x2 blocks at level 1
    CHECK_EQ(adaptive.counts_by_level()[1], int64_t(4));
    CHECK_NEAR(total_volume(adaptive), 0.5, 1e-12);
    CHECK_NEAR(total_volume(fixed), 0.5, 1e-12);
}

TEST(voxelize_obj_open_mesh_is_not_filled) {
    // Without a lid the flood fill reaches the inside, so only the walls stay.
    auto mesh = box_mesh(1, 1, 1, /*open_top=*/true);
    auto grid = make_grid(mesh, 4);
    auto cells = solid_cells(mesh, grid);
    CHECK_EQ((cells == kInterior).sum().item<int64_t>(), int64_t(0));
    CHECK_EQ(voxelize_fixed(cells, grid).size(), int64_t(64 - 8 - 4));
}

TEST(voxelize_chair_obj_fixed_and_adaptive_benchmark) {
    const std::string obj_path = std::string(RIEMANN_TEST_DATA_DIR) + "/chair.obj";
    if (!std::filesystem::exists(obj_path))
        throw riemann_test::Skip("tests/data/chair.obj not found");

    const int64_t resolution = env_int("RIEMANN_CHAIR_RESOLUTION", 64);
    const uint32_t depth = static_cast<uint32_t>(env_int("RIEMANN_CHAIR_DEPTH", 3));
    std::cout << "    chair.obj volume voxelization (resolution=" << resolution
              << ", depth=" << depth << ")\n";

    auto t0 = Clock::now();
    auto mesh = load_obj(obj_path);
    std::cout << "      load_obj          " << elapsed_ms(t0) << " ms  ("
              << mesh.vertices.size() << " vertices, " << mesh.triangles.size()
              << " triangles)\n";
    CHECK(!mesh.triangles.empty());

    auto grid = make_grid(mesh, resolution);
    t0 = Clock::now();
    auto cells = rasterize_surface(mesh, grid);
    std::cout << "      rasterize_surface " << elapsed_ms(t0) << " ms\n";
    t0 = Clock::now();
    fill_interior(cells);
    const int64_t surface = (cells == kSurface).sum().item<int64_t>();
    const int64_t interior = (cells == kInterior).sum().item<int64_t>();
    std::cout << "      fill_interior     " << elapsed_ms(t0) << " ms  (" << surface
              << " surface + " << interior << " interior cells)\n";
    CHECK(surface > 0);
    CHECK(interior > 0);  // the chair is closed, so it must have a solid inside

    t0 = Clock::now();
    auto fixed = voxelize_fixed(cells, grid);
    std::cout << "      voxelize_fixed    " << elapsed_ms(t0) << " ms  (" << fixed.size()
              << " voxels)\n";
    CHECK_EQ(fixed.size(), surface + interior);

    t0 = Clock::now();
    auto adaptive = voxelize_adaptive(cells, grid, depth);
    std::cout << "      voxelize_adaptive " << elapsed_ms(t0) << " ms  (" << adaptive.size()
              << " voxels)\n";
    auto counts = adaptive.counts_by_level();
    for (size_t l = 0; l < counts.size(); ++l)
        std::cout << "        level " << l << "         " << counts[l] << " voxels\n";
    CHECK(!adaptive.has_overlapping_leaves());
    CHECK(adaptive.size() < fixed.size());
    // Lossless: identical volume and every fixed voxel lies under an adaptive leaf.
    CHECK_NEAR(total_volume(adaptive), total_volume(fixed), total_volume(fixed) * 1e-9);
    auto fc = fixed.coordinates().contiguous();
    const int64_t* cd = fc.data_ptr<int64_t>();
    for (int64_t i = 0; i < fixed.size(); ++i)
        CHECK(covered_by_adaptive(adaptive, {cd[i * 3], cd[i * 3 + 1], cd[i * 3 + 2]},
                                  depth));

    // Voxel files for visualization: python tests/adaptive/view_voxels.py
    const auto out_dir = std::filesystem::current_path();
    const std::string tag = "_r" + std::to_string(resolution);
    struct Output {
        std::string name;
        const AdaptiveSparseTensor* tensor;
    } outputs[] = {{"chair_fixed" + tag, &fixed},
                   {"chair_adaptive" + tag + "_d" + std::to_string(depth), &adaptive}};
    for (const auto& o : outputs) {
        const auto obj_file = (out_dir / (o.name + ".obj")).string();
        const auto json_file = (out_dir / (o.name + ".json")).string();
        t0 = Clock::now();
        write_voxels_obj(*o.tensor, obj_file);
        write_voxels_json(*o.tensor, grid, json_file);
        std::cout << "      wrote " << json_file << " (+ .obj) in " << elapsed_ms(t0)
                  << " ms\n";
        CHECK(std::filesystem::file_size(obj_file) > 0);
        CHECK(std::filesystem::file_size(json_file) > 0);
    }
}
