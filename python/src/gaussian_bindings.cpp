#include "bindings.hpp"

#include "splat/camera.hpp"
#include "splat/dnerf.hpp"
#include "splat/fourdgs.hpp"
#include "splat/model.hpp"
#include "splat/model4d.hpp"
#include "splat/ply.hpp"
#include "splat/render.hpp"
#include "splat/soup.hpp"
#include "splat/trainer.hpp"
#include "splat/trainer4d.hpp"

#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace riemann::gaussian;

namespace riemann_py {

void bind_gaussian(py::module_ &m) {
  // --------------------------------------------------------------------
  // Camera
  // --------------------------------------------------------------------
  py::class_<Camera>(m, "Camera",
                     "Pinhole camera, OpenCV/COLMAP convention "
                     "(+x right, +y down, +z forward)")
      .def(py::init<>())
      .def_readwrite("width", &Camera::width)
      .def_readwrite("height", &Camera::height)
      .def_readwrite("fx", &Camera::fx)
      .def_readwrite("fy", &Camera::fy)
      .def_readwrite("cx", &Camera::cx)
      .def_readwrite("cy", &Camera::cy)
      .def_readwrite("world_to_camera", &Camera::world_to_camera)
      .def_readwrite("near_plane", &Camera::near_plane)
      .def("rotation", &Camera::rotation)
      .def("translation", &Camera::translation)
      .def("position", &Camera::position)
      .def_static("from_fov", &Camera::from_fov, py::arg("width"),
                  py::arg("height"), py::arg("fov_x_radians"),
                  py::arg("world_to_camera"))
      .def_static("look_at", &Camera::look_at, py::arg("eye"),
                  py::arg("target"), py::arg("up"), py::arg("width"),
                  py::arg("height"), py::arg("fov_x_radians"));

  // --------------------------------------------------------------------
  // Render options and output
  // --------------------------------------------------------------------
  py::class_<RenderOptions>(m, "RenderOptions")
      .def(py::init<>())
      .def_readwrite("background", &RenderOptions::background)
      .def_readwrite("sh_degree", &RenderOptions::sh_degree)
      .def_readwrite("tile_size", &RenderOptions::tile_size)
      .def_readwrite("chunk_size", &RenderOptions::chunk_size)
      .def_readwrite("use_cuda_kernels", &RenderOptions::use_cuda_kernels);

  py::class_<RenderOutput>(m, "RenderOutput")
      .def_readonly("image", &RenderOutput::image)
      .def_readonly("depth", &RenderOutput::depth)
      .def_readonly("alpha", &RenderOutput::alpha)
      .def_readonly("means2d", &RenderOutput::means2d)
      .def_readonly("visible", &RenderOutput::visible)
      .def_readonly("radii", &RenderOutput::radii);

  // --------------------------------------------------------------------
  // GaussianTensor
  // --------------------------------------------------------------------
  py::class_<GaussianTensor>(m, "GaussianTensor")
      .def(py::init([](at::Tensor positions, at::Tensor rotations,
                       at::Tensor scales, at::Tensor opacities,
                       at::Tensor colors, c10::optional<at::Tensor> features,
                       c10::optional<at::Tensor> velocities) {
             return GaussianTensor(
                 std::move(positions), std::move(rotations),
                 std::move(scales), std::move(opacities), std::move(colors),
                 features.value_or(at::Tensor()),
                 velocities.value_or(at::Tensor()));
           }),
           py::arg("positions"), py::arg("rotations"), py::arg("scales"),
           py::arg("opacities"), py::arg("colors"),
           py::arg("features") = py::none(),
           py::arg("velocities") = py::none())
      .def_static(
          "empty",
          [](int64_t count, at::ScalarType dtype, at::Device device,
             int64_t feature_channels, bool with_velocities) {
            return GaussianTensor::empty(
                count, at::TensorOptions().dtype(dtype).device(device),
                feature_channels, with_velocities);
          },
          py::arg("count"), py::arg("dtype") = at::kFloat,
          py::arg("device") = at::Device(at::kCPU),
          py::arg("feature_channels") = 0,
          py::arg("with_velocities") = false)
      .def("__len__", &GaussianTensor::size)
      .def_property_readonly("size", &GaussianTensor::size)
      .def_property_readonly("feature_channels",
                             &GaussianTensor::feature_channels)
      .def_property_readonly("has_velocities", &GaussianTensor::has_velocities)
      .def_property_readonly("device", &GaussianTensor::device)
      .def_property_readonly("dtype", &GaussianTensor::dtype)
      .def_property_readonly("positions", &GaussianTensor::positions)
      .def_property_readonly("rotations", &GaussianTensor::rotations)
      .def_property_readonly("scales", &GaussianTensor::scales)
      .def_property_readonly("opacities", &GaussianTensor::opacities)
      .def_property_readonly("colors", &GaussianTensor::colors)
      .def_property_readonly("features", &GaussianTensor::features)
      .def_property_readonly("velocities", &GaussianTensor::velocities)
      .def("covariances", &GaussianTensor::covariances)
      .def("select", &GaussianTensor::select, py::arg("indices"))
      .def("remove", &GaussianTensor::remove, py::arg("indices"))
      .def("filter_opacity", &GaussianTensor::filter_opacity,
           py::arg("minimum_opacity"))
      .def("filter_bounds", &GaussianTensor::filter_bounds, py::arg("lower"),
           py::arg("upper"))
      .def("append", &GaussianTensor::append, py::arg("other"))
      .def_static("concatenate", &GaussianTensor::concatenate,
                  py::arg("tensors"))
      .def("clone", &GaussianTensor::clone)
      .def("to", &GaussianTensor::to, py::arg("device"))
      .def("enable_gradients", &GaussianTensor::enable_gradients)
      .def("project_constraints", &GaussianTensor::project_constraints)
      .def("transform", &GaussianTensor::transform, py::arg("rotation"),
           py::arg("translation"),
           "Quaternions use (w, x, y, z); rotation may be [4] or [N, 4], "
           "translation [3] or [N, 3]")
      .def("__repr__", [](const GaussianTensor &self) {
        std::ostringstream out;
        out << "GaussianTensor(N=" << self.size()
            << ", device=" << self.device() << ")";
        return out.str();
      });

  // --------------------------------------------------------------------
  // GaussianModel
  // --------------------------------------------------------------------
  py::class_<GaussianModel>(m, "GaussianModel")
      .def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor, at::Tensor>(),
           py::arg("xyz"), py::arg("f_dc"), py::arg("f_rest"),
           py::arg("opacity_raw"), py::arg("scale_raw"),
           py::arg("rotation_raw"))
      .def_static("from_tensor", &GaussianModel::from_tensor,
                  py::arg("gaussians"))
      .def_static("from_points", &GaussianModel::from_points,
                  py::arg("positions"), py::arg("colors"),
                  py::arg("sh_degree") = 3,
                  py::arg("initial_opacity") = 0.1)
      .def("__len__", &GaussianModel::size)
      .def_property_readonly("size", &GaussianModel::size)
      .def_property_readonly("sh_degree", &GaussianModel::sh_degree)
      .def_property_readonly("device", &GaussianModel::device)
      .def_property_readonly("xyz", &GaussianModel::xyz)
      .def_property_readonly("f_dc", &GaussianModel::f_dc)
      .def_property_readonly("f_rest", &GaussianModel::f_rest)
      .def_property_readonly("opacity_raw", &GaussianModel::opacity_raw)
      .def_property_readonly("scale_raw", &GaussianModel::scale_raw)
      .def_property_readonly("rotation_raw", &GaussianModel::rotation_raw)
      .def("positions", &GaussianModel::positions)
      .def("rotations", &GaussianModel::rotations)
      .def("scales", &GaussianModel::scales)
      .def("opacities", &GaussianModel::opacities)
      .def("colors", &GaussianModel::colors)
      .def("sh_coefficients", &GaussianModel::sh_coefficients)
      .def("to_gaussian_tensor", &GaussianModel::to_gaussian_tensor)
      .def("parameters", &GaussianModel::parameters,
           "Fixed order: xyz, f_dc, f_rest, opacity_raw, scale_raw, "
           "rotation_raw")
      .def("set_parameters", &GaussianModel::set_parameters,
           py::arg("parameters"))
      .def("enable_gradients", &GaussianModel::enable_gradients)
      .def("to", &GaussianModel::to, py::arg("device"))
      .def("clone", &GaussianModel::clone)
      .def("select", &GaussianModel::select, py::arg("indices"))
      .def("__repr__", [](const GaussianModel &self) {
        std::ostringstream out;
        out << "GaussianModel(N=" << self.size()
            << ", sh_degree=" << self.sh_degree() << ")";
        return out.str();
      });

  py::class_<Gaussian4DModel>(m, "Gaussian4DModel")
      .def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                     at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                     at::Tensor>(),
           py::arg("xyz"), py::arg("f_dc"), py::arg("f_rest"),
           py::arg("opacity_raw"), py::arg("scale_raw"),
           py::arg("rotation_raw"), py::arg("velocity"),
           py::arg("time_center"), py::arg("time_scale_raw"))
      .def_static("from_model", &Gaussian4DModel::from_model,
                  py::arg("model"), py::arg("initial_time_scale") = 1.0)
      .def_static("from_points", &Gaussian4DModel::from_points,
                  py::arg("positions"), py::arg("colors"),
                  py::arg("sh_degree") = 3, py::arg("initial_opacity") = 0.1,
                  py::arg("initial_time_scale") = 0.5)
      .def("__len__", &Gaussian4DModel::size)
      .def_property_readonly("size", &Gaussian4DModel::size)
      .def_property_readonly("sh_degree", &Gaussian4DModel::sh_degree)
      .def_property_readonly("device", &Gaussian4DModel::device)
      .def_property_readonly("xyz", &Gaussian4DModel::xyz)
      .def_property_readonly("f_dc", &Gaussian4DModel::f_dc)
      .def_property_readonly("f_rest", &Gaussian4DModel::f_rest)
      .def_property_readonly("opacity_raw", &Gaussian4DModel::opacity_raw)
      .def_property_readonly("scale_raw", &Gaussian4DModel::scale_raw)
      .def_property_readonly("rotation_raw", &Gaussian4DModel::rotation_raw)
      .def_property_readonly("velocity", &Gaussian4DModel::velocity)
      .def_property_readonly("time_center", &Gaussian4DModel::time_center)
      .def_property_readonly("time_scale_raw", &Gaussian4DModel::time_scale_raw)
      .def("positions", &Gaussian4DModel::positions, py::arg("time"))
      .def("rotations", &Gaussian4DModel::rotations)
      .def("scales", &Gaussian4DModel::scales)
      .def("time_scales", &Gaussian4DModel::time_scales)
      .def("opacities", py::overload_cast<>(&Gaussian4DModel::opacities, py::const_))
      .def("opacities", py::overload_cast<double>(&Gaussian4DModel::opacities, py::const_),
           py::arg("time"))
      .def("sh_coefficients", &Gaussian4DModel::sh_coefficients)
      .def("at_time", &Gaussian4DModel::at_time, py::arg("time"))
      .def("parameters", &Gaussian4DModel::parameters)
      .def("set_parameters", &Gaussian4DModel::set_parameters, py::arg("parameters"))
      .def("enable_gradients", &Gaussian4DModel::enable_gradients)
      .def("to", &Gaussian4DModel::to, py::arg("device"))
      .def("clone", &Gaussian4DModel::clone)
      .def("select", &Gaussian4DModel::select, py::arg("indices"));

  py::class_<TimedView>(m, "TimedView")
      .def_readonly("camera", &TimedView::camera)
      .def_readonly("time", &TimedView::time)
      .def_readonly("image", &TimedView::image)
      .def_readonly("image_path", &TimedView::image_path);
  py::class_<DNerfLoadOptions>(m, "DNerfLoadOptions")
      .def(py::init<>())
      .def_readwrite("stride", &DNerfLoadOptions::stride)
      .def_readwrite("max_frames", &DNerfLoadOptions::max_frames)
      .def_readwrite("downscale", &DNerfLoadOptions::downscale)
      .def_readwrite("background", &DNerfLoadOptions::background);
  m.def("load_dnerf_views", &load_dnerf_views, py::arg("dataset_dir"),
        py::arg("split"), py::arg("options") = DNerfLoadOptions{},
        py::call_guard<py::gil_scoped_release>());

  py::class_<FourDGSExportOptions>(m, "FourDGSExportOptions")
      .def(py::init<>())
      .def_readwrite("time_start", &FourDGSExportOptions::time_start)
      .def_readwrite("time_end", &FourDGSExportOptions::time_end)
      .def_readwrite("seconds_per_time_unit", &FourDGSExportOptions::seconds_per_time_unit)
      .def_readwrite("cutoff", &FourDGSExportOptions::cutoff)
      .def_readwrite("chunk_size", &FourDGSExportOptions::chunk_size);
  py::class_<FourDGSExportResult>(m, "FourDGSExportResult")
      .def_readonly("gaussians", &FourDGSExportResult::gaussians)
      .def_readonly("bytes", &FourDGSExportResult::bytes)
      .def_readonly("clamped_rgb_components", &FourDGSExportResult::clamped_rgb_components)
      .def_readonly("clamped_sh_coefficients", &FourDGSExportResult::clamped_sh_coefficients)
      .def_readonly("max_position_error", &FourDGSExportResult::max_position_error)
      .def_readonly("max_scale_log_error", &FourDGSExportResult::max_scale_log_error)
      .def_readonly("max_rotation_error", &FourDGSExportResult::max_rotation_error)
      .def_readonly("max_rgb_error", &FourDGSExportResult::max_rgb_error)
      .def_readonly("max_opacity_error", &FourDGSExportResult::max_opacity_error)
      .def_readonly("max_velocity_error", &FourDGSExportResult::max_velocity_error)
      .def_readonly("max_time_error", &FourDGSExportResult::max_time_error)
      .def_readonly("max_sigma_log_error", &FourDGSExportResult::max_sigma_log_error)
      .def_readonly("max_sh_error", &FourDGSExportResult::max_sh_error);
  m.def("save_four_dgs", &save_four_dgs, py::arg("model"), py::arg("path"),
        py::arg("options") = FourDGSExportOptions{},
        py::call_guard<py::gil_scoped_release>(),
        "Exports the avala-ai version-1 draft .4dgs gaussian-birth container");

  // --------------------------------------------------------------------
  // PLY input / output
  // --------------------------------------------------------------------
  m.def("load_ply", &load_gaussian_ply, py::arg("path"),
        py::call_guard<py::gil_scoped_release>(),
        "Loads a 3DGS or SuperSplat PLY as an activated GaussianTensor");
  m.def("load_model_ply", &load_gaussian_model_ply, py::arg("path"),
        py::call_guard<py::gil_scoped_release>(),
        "Loads a 3DGS PLY as raw trainable parameters, including SH");
  m.def("save_ply",
        py::overload_cast<const GaussianModel &, const std::string &>(
            &save_gaussian_ply),
        py::arg("model"), py::arg("path"),
        py::call_guard<py::gil_scoped_release>());
  m.def("save_ply",
        py::overload_cast<const GaussianTensor &, const std::string &>(
            &save_gaussian_ply),
        py::arg("gaussians"), py::arg("path"),
        py::call_guard<py::gil_scoped_release>());

  // --------------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------------
  m.def("render_gaussians", &render_gaussians, py::arg("positions"),
        py::arg("rotations"), py::arg("scales"), py::arg("opacities"),
        py::arg("colors"), py::arg("camera"),
        py::arg("options") = RenderOptions{},
        py::call_guard<py::gil_scoped_release>(),
        "Differentiable tile-based rasterizer over raw per-Gaussian tensors");
  m.def("render",
        py::overload_cast<const GaussianModel &, const Camera &,
                          const RenderOptions &>(&render),
        py::arg("model"), py::arg("camera"),
        py::arg("options") = RenderOptions{},
        py::call_guard<py::gil_scoped_release>(),
        "Renders a trainable model with view-dependent SH color");
  m.def("render",
        py::overload_cast<const GaussianTensor &, const Camera &,
                          const RenderOptions &>(&render),
        py::arg("gaussians"), py::arg("camera"),
        py::arg("options") = RenderOptions{},
        py::call_guard<py::gil_scoped_release>());
  m.def("render",
        py::overload_cast<const Gaussian4DModel &, const Camera &, double,
                          const RenderOptions &>(&render),
        py::arg("model"), py::arg("camera"), py::arg("time"),
        py::arg("options") = RenderOptions{},
        py::call_guard<py::gil_scoped_release>());

  // --------------------------------------------------------------------
  // Losses
  // --------------------------------------------------------------------
  m.def("l1_loss", &l1_loss, py::arg("prediction"), py::arg("target"),
        py::call_guard<py::gil_scoped_release>());
  m.def("ssim", &ssim, py::arg("prediction"), py::arg("target"),
        py::arg("window") = 11, py::arg("sigma") = 1.5,
        py::call_guard<py::gil_scoped_release>());

  // --------------------------------------------------------------------
  // Trainer
  // --------------------------------------------------------------------
  py::class_<GaussianTrainerConfig>(m, "GaussianTrainerConfig")
      .def(py::init<>())
      .def_readwrite("lr_position", &GaussianTrainerConfig::lr_position)
      .def_readwrite("lr_position_final",
                     &GaussianTrainerConfig::lr_position_final)
      .def_readwrite("lr_decay_steps", &GaussianTrainerConfig::lr_decay_steps)
      .def_readwrite("lr_dc", &GaussianTrainerConfig::lr_dc)
      .def_readwrite("lr_rest", &GaussianTrainerConfig::lr_rest)
      .def_readwrite("lr_opacity", &GaussianTrainerConfig::lr_opacity)
      .def_readwrite("lr_scale", &GaussianTrainerConfig::lr_scale)
      .def_readwrite("lr_rotation", &GaussianTrainerConfig::lr_rotation)
      .def_readwrite("beta1", &GaussianTrainerConfig::beta1)
      .def_readwrite("beta2", &GaussianTrainerConfig::beta2)
      .def_readwrite("epsilon", &GaussianTrainerConfig::epsilon)
      .def_readwrite("lambda_ssim", &GaussianTrainerConfig::lambda_ssim)
      .def_readwrite("sh_increase_interval",
                     &GaussianTrainerConfig::sh_increase_interval)
      .def_readwrite("densify", &GaussianTrainerConfig::densify)
      .def_readwrite("densify_start", &GaussianTrainerConfig::densify_start)
      .def_readwrite("densify_interval",
                     &GaussianTrainerConfig::densify_interval)
      .def_readwrite("densify_stop", &GaussianTrainerConfig::densify_stop)
      .def_readwrite("densify_grad_threshold",
                     &GaussianTrainerConfig::densify_grad_threshold)
      .def_readwrite("percent_dense", &GaussianTrainerConfig::percent_dense)
      .def_readwrite("min_opacity", &GaussianTrainerConfig::min_opacity)
      .def_readwrite("max_scale_fraction",
                     &GaussianTrainerConfig::max_scale_fraction)
      .def_readwrite("opacity_reset_interval",
                     &GaussianTrainerConfig::opacity_reset_interval)
      .def_readwrite("max_gaussians", &GaussianTrainerConfig::max_gaussians)
      .def_readwrite("scene_extent", &GaussianTrainerConfig::scene_extent)
      .def_readwrite("render", &GaussianTrainerConfig::render);

  py::class_<GaussianTrainer::StepResult>(m, "StepResult")
      .def_readonly("loss", &GaussianTrainer::StepResult::loss)
      .def_readonly("gaussians", &GaussianTrainer::StepResult::gaussians)
      .def_readonly("iteration", &GaussianTrainer::StepResult::iteration)
      .def("__repr__", [](const GaussianTrainer::StepResult &self) {
        std::ostringstream out;
        out << "StepResult(iteration=" << self.iteration
            << ", loss=" << self.loss << ", gaussians=" << self.gaussians
            << ")";
        return out.str();
      });

  py::class_<GaussianTrainer>(m, "GaussianTrainer",
                              "3DGS-style trainer; the model must outlive "
                              "the trainer")
      .def(py::init<GaussianModel &, GaussianTrainerConfig>(),
           py::arg("model"), py::arg("config") = GaussianTrainerConfig{},
           // The trainer keeps a raw pointer to the model.
           py::keep_alive<1, 2>())
      .def("step", &GaussianTrainer::step, py::arg("camera"),
           py::arg("target"),
           // step() runs backward inside C++, which must not hold the GIL.
           py::call_guard<py::gil_scoped_release>())
      .def("render_image", &GaussianTrainer::render_image, py::arg("camera"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("iteration", &GaussianTrainer::iteration)
      .def_property_readonly("active_sh_degree",
                             &GaussianTrainer::active_sh_degree)
      .def_property_readonly("config", &GaussianTrainer::config)
      .def("accumulate_statistics", &GaussianTrainer::accumulate_statistics,
           py::arg("output"), py::arg("camera"),
           py::call_guard<py::gil_scoped_release>())
      .def("densify_and_prune", &GaussianTrainer::densify_and_prune,
           py::call_guard<py::gil_scoped_release>(),
           "Returns (added, removed)")
      .def("reset_opacity", &GaussianTrainer::reset_opacity,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Gaussian4DTrainerConfig, GaussianTrainerConfig>(m, "Gaussian4DTrainerConfig")
      .def(py::init<>())
      .def_readwrite("lr_velocity", &Gaussian4DTrainerConfig::lr_velocity)
      .def_readwrite("lr_time_center", &Gaussian4DTrainerConfig::lr_time_center)
      .def_readwrite("lr_time_scale", &Gaussian4DTrainerConfig::lr_time_scale);
  py::class_<Gaussian4DTrainer>(m, "Gaussian4DTrainer")
      .def(py::init<Gaussian4DModel &, Gaussian4DTrainerConfig>(),
           py::arg("model"), py::arg("config") = Gaussian4DTrainerConfig{},
           py::keep_alive<1, 2>())
      .def("step", &Gaussian4DTrainer::step, py::arg("camera"),
           py::arg("time"), py::arg("target"),
           py::call_guard<py::gil_scoped_release>())
      .def("render_image", &Gaussian4DTrainer::render_image,
           py::arg("camera"), py::arg("time"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("iteration", &Gaussian4DTrainer::iteration)
      .def_property_readonly("active_sh_degree", &Gaussian4DTrainer::active_sh_degree)
      .def_property_readonly("config", &Gaussian4DTrainer::config)
      .def("accumulate_statistics", &Gaussian4DTrainer::accumulate_statistics,
           py::arg("output"), py::arg("camera"),
           py::call_guard<py::gil_scoped_release>())
      .def("densify_and_prune", &Gaussian4DTrainer::densify_and_prune,
           py::call_guard<py::gil_scoped_release>())
      .def("reset_opacity", &Gaussian4DTrainer::reset_opacity,
           py::call_guard<py::gil_scoped_release>())
      .def("save_checkpoint", &Gaussian4DTrainer::save_checkpoint,
           py::arg("path"), py::call_guard<py::gil_scoped_release>())
      .def("load_checkpoint", &Gaussian4DTrainer::load_checkpoint,
           py::arg("path"),
           py::call_guard<py::gil_scoped_release>());
}

} // namespace riemann_py
