#include "bindings.hpp"

#include "octree/config.hpp"
#include "octree/ops.hpp"
#include "octree/stencil.hpp"
#include "octree/tensor.hpp"

#include <array>
#include <cstdint>
#include <sstream>

namespace py = pybind11;
using namespace riemann::adaptive;

namespace riemann_py {

void bind_adaptive(py::module_ &m) {
  // --------------------------------------------------------------------
  // Enums
  // --------------------------------------------------------------------
  py::enum_<Reduction>(m, "Reduction")
      .value("Mean", Reduction::Mean)
      .value("Sum", Reduction::Sum)
      .value("Max", Reduction::Max);

  py::enum_<RefinePolicy>(m, "RefinePolicy")
      .value("Replicate", RefinePolicy::Replicate)
      .value("Split", RefinePolicy::Split);

  py::enum_<InterpMode>(m, "InterpMode")
      .value("Nearest", InterpMode::Nearest)
      .value("Multilinear", InterpMode::Multilinear);

  py::enum_<AdaptiveConvMode>(m, "AdaptiveConvMode")
      .value("CenterSampled", AdaptiveConvMode::CenterSampled)
      .value("VolumeWeighted", AdaptiveConvMode::VolumeWeighted)
      .value("OverlapWeighted", AdaptiveConvMode::OverlapWeighted);

  // --------------------------------------------------------------------
  // Configuration
  // --------------------------------------------------------------------
  py::class_<AxisAlignedRegion>(m, "AxisAlignedRegion",
                                "Axis-aligned half-open box [lo, hi)")
      .def(py::init<>())
      .def(py::init([](std::vector<double> lo, std::vector<double> hi) {
             AxisAlignedRegion region;
             region.lo = std::move(lo);
             region.hi = std::move(hi);
             return region;
           }),
           py::arg("lo"), py::arg("hi"))
      .def_readwrite("lo", &AxisAlignedRegion::lo)
      .def_readwrite("hi", &AxisAlignedRegion::hi)
      .def_static("everything", &AxisAlignedRegion::everything,
                  py::arg("dimension"));

  py::class_<AdaptiveVoxelConfig>(m, "AdaptiveVoxelConfig")
      .def(py::init<>())
      .def(py::init([](std::vector<double> base_voxel_size,
                       uint32_t max_depth, std::vector<double> origin,
                       std::vector<double> min_voxel_size) {
             AdaptiveVoxelConfig config;
             config.base_voxel_size = std::move(base_voxel_size);
             config.max_depth = max_depth;
             config.origin = std::move(origin);
             config.min_voxel_size = std::move(min_voxel_size);
             return config;
           }),
           py::arg("base_voxel_size"), py::arg("max_depth") = 0,
           py::arg("origin") = std::vector<double>{},
           py::arg("min_voxel_size") = std::vector<double>{})
      .def_readwrite("base_voxel_size", &AdaptiveVoxelConfig::base_voxel_size)
      .def_readwrite("min_voxel_size", &AdaptiveVoxelConfig::min_voxel_size)
      .def_readwrite("origin", &AdaptiveVoxelConfig::origin)
      .def_readwrite("max_depth", &AdaptiveVoxelConfig::max_depth)
      .def_property_readonly("dimension", &AdaptiveVoxelConfig::dimension)
      .def("validate", &AdaptiveVoxelConfig::validate);

  py::class_<AdaptiveKernelConfig>(m, "AdaptiveKernelConfig")
      .def(py::init<>())
      .def(py::init([](int kernel_size, AdaptiveConvMode mode,
                       int output_level_shift) {
             AdaptiveKernelConfig config;
             config.kernel_size = kernel_size;
             config.mode = mode;
             config.output_level_shift = output_level_shift;
             return config;
           }),
           py::arg("kernel_size") = 3,
           py::arg("mode") = AdaptiveConvMode::CenterSampled,
           py::arg("output_level_shift") = 0)
      .def_readwrite("kernel_size", &AdaptiveKernelConfig::kernel_size)
      .def_readwrite("mode", &AdaptiveKernelConfig::mode)
      .def_readwrite("output_level_shift",
                     &AdaptiveKernelConfig::output_level_shift);

  // --------------------------------------------------------------------
  // AdaptiveSparseTensor
  // --------------------------------------------------------------------
  py::class_<AdaptiveSparseTensor>(m, "AdaptiveSparseTensor")
      .def_static("from_points", &AdaptiveSparseTensor::from_points,
                  py::arg("points"), py::arg("features"), py::arg("config"),
                  py::call_guard<py::gil_scoped_release>())
      .def_static("from_voxels", &AdaptiveSparseTensor::from_voxels,
                  py::arg("coordinates"), py::arg("levels"),
                  py::arg("features"), py::arg("config"))
      .def("__len__", &AdaptiveSparseTensor::size)
      .def_property_readonly("size", &AdaptiveSparseTensor::size)
      .def_property_readonly("dimension", &AdaptiveSparseTensor::dimension)
      .def_property_readonly("channels", &AdaptiveSparseTensor::channels)
      .def_property_readonly("config", &AdaptiveSparseTensor::config)
      .def_property_readonly("device", &AdaptiveSparseTensor::device)
      .def("coordinates", &AdaptiveSparseTensor::coordinates)
      .def("levels", &AdaptiveSparseTensor::levels)
      .def("voxel_sizes", &AdaptiveSparseTensor::voxel_sizes)
      .def("volumes", &AdaptiveSparseTensor::volumes)
      .def("centers", &AdaptiveSparseTensor::centers)
      .def("bounds", &AdaptiveSparseTensor::bounds)
      .def_property("features", &AdaptiveSparseTensor::features,
                    &AdaptiveSparseTensor::set_features)
      .def("counts_by_level", &AdaptiveSparseTensor::counts_by_level)
      .def("index_of", &AdaptiveSparseTensor::index_of_key,
           py::arg("coordinates"), py::arg("level"))
      .def("topology_version", &AdaptiveSparseTensor::topology_version)
      .def("has_overlapping_leaves",
           &AdaptiveSparseTensor::has_overlapping_leaves)
      .def("has_source_points", &AdaptiveSparseTensor::has_source_points)
      .def("refine", &AdaptiveSparseTensor::refine, py::arg("region"),
           py::arg("target_level"),
           py::arg("policy") = RefinePolicy::Replicate)
      .def("coarsen", &AdaptiveSparseTensor::coarsen, py::arg("region"),
           py::arg("target_level"), py::arg("reduction") = Reduction::Mean)
      .def("refine_by_density", &AdaptiveSparseTensor::refine_by_density,
           py::arg("points_per_leaf_threshold"))
      .def("refine_by_variance", &AdaptiveSparseTensor::refine_by_variance,
           py::arg("feature_variance_threshold"))
      .def("refine_if", &AdaptiveSparseTensor::refine_if,
           py::arg("criterion"),
           "criterion(leaf, lo, hi, level) -> bool; refines leaves for which "
           "it returns True")
      .def("revoxelize_from_points",
           &AdaptiveSparseTensor::revoxelize_from_points)
      .def("with_features", &AdaptiveSparseTensor::with_features,
           py::arg("features"))
      .def("to", &AdaptiveSparseTensor::to, py::arg("device"))
      .def("resample_uniform", &AdaptiveSparseTensor::resample_uniform,
           py::arg("level"), py::arg("reduction") = Reduction::Mean)
      .def("resample_to", &AdaptiveSparseTensor::resample_to,
           py::arg("reference"), py::arg("reduction") = Reduction::Mean)
      .def("__add__",
           [](const AdaptiveSparseTensor &a, const AdaptiveSparseTensor &b) {
             return a + b;
           })
      .def("__repr__", [](const AdaptiveSparseTensor &self) {
        std::ostringstream out;
        out << "AdaptiveSparseTensor(N=" << self.size()
            << ", C=" << self.channels() << ", D=" << self.dimension()
            << ")";
        return out.str();
      });

  // --------------------------------------------------------------------
  // Operations
  // --------------------------------------------------------------------
  m.def("concatenate", &concatenate, py::arg("a"), py::arg("b"),
        "Concatenates features of two tensors sharing one topology");
  m.def("convolution", &adaptive_convolution, py::arg("input"),
        py::arg("weight"), py::arg("config"),
        py::call_guard<py::gil_scoped_release>(),
        "Adaptive convolution with weight [K^D, Cin, Cout]");
  m.def("convolution_transpose", &adaptive_convolution_transpose,
        py::arg("input"), py::arg("weight"), py::arg("config"),
        py::arg("reference"), py::call_guard<py::gil_scoped_release>());
  m.def("global_avg_pool", &adaptive_global_avg_pool, py::arg("input"),
        py::arg("volume_weighted") = true,
        py::call_guard<py::gil_scoped_release>());
  m.def("pool", &adaptive_pool, py::arg("input"), py::arg("target_level"),
        py::arg("reduction") = Reduction::Mean,
        py::call_guard<py::gil_scoped_release>());
  m.def("interpolate", &riemann::adaptive::interpolate, py::arg("input"),
        py::arg("points"), py::arg("mode") = InterpMode::Nearest,
        py::call_guard<py::gil_scoped_release>());

  // --------------------------------------------------------------------
  // Kernel-map cache control
  // --------------------------------------------------------------------
  m.def("clear_kernel_map_cache",
        []() { KernelMapCache::instance().clear(); });
  m.def("kernel_map_cache_stats", []() {
    auto stats = KernelMapCache::instance().stats();
    py::dict result;
    result["hits"] = stats.hits;
    result["misses"] = stats.misses;
    result["entries"] = stats.entries;
    return result;
  });
  m.def("set_kernel_map_cache_capacity",
        [](size_t max_entries) {
          KernelMapCache::instance().set_capacity(max_entries);
        },
        py::arg("max_entries"));
}

} // namespace riemann_py
