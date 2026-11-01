#include "bindings.hpp"

#include "splat/camera.hpp"
#include "splat/model.hpp"
#include "splat/ply.hpp"
#include "splat/render.hpp"
#include "splat/soup.hpp"
#include "splat/trainer.hpp"

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
}

} // namespace riemann_py
