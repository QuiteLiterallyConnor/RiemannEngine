#include "bindings.hpp"

#include "gaussian_voxel/occupancy.hpp"
#include "gaussian_voxel/voxelize.hpp"

namespace py = pybind11;
using namespace riemann::gaussian_voxel;
using namespace riemann::occupancy;

namespace riemann_py {

void bind_gaussian_voxel(py::module_ &voxel_module, py::module_ &occupancy_module) {
  py::enum_<FeatureAggregation>(voxel_module, "FeatureAggregation")
      .value("WeightedSum", FeatureAggregation::WeightedSum)
      .value("NormalizedWeightedMean",
             FeatureAggregation::NormalizedWeightedMean);

  py::enum_<OccupancyModel>(voxel_module, "OccupancyModel")
      .value("Poisson", OccupancyModel::Poisson)
      .value("ClampedLinear", OccupancyModel::ClampedLinear);

  py::class_<GaussianVoxelOptions>(voxel_module, "GaussianVoxelOptions")
      .def(py::init<>())
      .def_readwrite("support_radius", &GaussianVoxelOptions::support_radius)
      .def_readwrite("opacity_density_scale",
                     &GaussianVoxelOptions::opacity_density_scale)
      .def_readwrite("max_associations",
                     &GaussianVoxelOptions::max_associations)
      .def_readwrite("feature_aggregation",
                     &GaussianVoxelOptions::feature_aggregation)
      .def_readwrite("occupancy_model", &GaussianVoxelOptions::occupancy_model);

  py::class_<GaussianVoxelOutput>(voxel_module, "GaussianVoxelOutput")
      .def_readonly("voxels", &GaussianVoxelOutput::voxels)
      .def_readonly("density", &GaussianVoxelOutput::density)
      .def_readonly("occupancy", &GaussianVoxelOutput::occupancy);

  voxel_module.def(
      "rasterize",
      [](const riemann::gaussian::GaussianTensor &gaussians,
         const riemann::adaptive::AdaptiveSparseTensor &reference,
         const GaussianVoxelOptions &options,
         c10::optional<at::Tensor> confidence) {
        return riemann::gaussian_voxel::rasterize(
            gaussians, reference, options,
            confidence.value_or(at::Tensor()));
      },
      py::arg("gaussians"), py::arg("reference"),
      py::arg("options") = GaussianVoxelOptions{},
      py::arg("confidence") = py::none(),
      py::call_guard<py::gil_scoped_release>(),
      "Rasterizes Gaussian features onto a fixed adaptive topology.");

  py::class_<LidarRayOptions>(occupancy_module, "LidarRayOptions")
      .def(py::init<>())
      .def_readwrite("max_ray_distance", &LidarRayOptions::max_ray_distance)
      .def_readwrite("max_ray_voxels", &LidarRayOptions::max_ray_voxels);

  py::class_<LidarOccupancyTargets>(occupancy_module, "LidarOccupancyTargets")
      .def_readonly("occupied", &LidarOccupancyTargets::occupied)
      .def_readonly("free", &LidarOccupancyTargets::free)
      .def_readonly("observed", &LidarOccupancyTargets::observed)
      .def("labels", &LidarOccupancyTargets::labels);

  occupancy_module.def(
      "from_lidar",
      [](const at::Tensor &points, const at::Tensor &sensor_origins,
         const at::Tensor &ray_batch_ids, int64_t batch_size,
         const riemann::adaptive::AdaptiveSparseTensor &reference,
         const LidarRayOptions &options,
         c10::optional<at::Tensor> sensor_to_world) {
        return riemann::occupancy::from_lidar(
            points, sensor_origins, ray_batch_ids, batch_size, reference,
            options, sensor_to_world.value_or(at::Tensor()));
      },
      py::arg("points"), py::arg("sensor_origins"),
      py::arg("ray_batch_ids"), py::arg("batch_size"), py::arg("reference"),
      py::arg("options") = LidarRayOptions{},
      py::arg("sensor_to_world") = py::none(),
      py::call_guard<py::gil_scoped_release>(),
      "Builds occupied/free/unknown labels from batched LiDAR rays.");

  occupancy_module.def(
      "masked_binary_cross_entropy",
      &riemann::occupancy::masked_binary_cross_entropy,
      py::arg("predicted_occupancy"), py::arg("targets"),
      py::arg("epsilon") = 1e-6,
      "Computes binary cross entropy only over observed LiDAR voxels.");
}

} // namespace riemann_py
