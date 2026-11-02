#include "bindings.hpp"

#include "riemann.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "RiemannEngine: sparse tensors, adaptive mixed-resolution "
            "voxelization and 3D Gaussian splats for PyTorch";

  m.attr("__version__") = riemann::VERSION;
#ifdef CPU_ONLY
  m.attr("cuda_enabled") = false;
#else
  m.attr("cuda_enabled") = true;
#endif

  riemann_py::bind_sparse(m);

  auto nn = m.def_submodule("nn", "Sparse neural network layers");
  riemann_py::bind_nn(nn);

  auto adaptive =
      m.def_submodule("adaptive", "Adaptive mixed-resolution voxelization");
  riemann_py::bind_adaptive(adaptive);

#ifdef RIEMANN_ENABLE_GAUSSIAN
  auto gaussian = m.def_submodule("gaussian", "3D Gaussian splats");
  riemann_py::bind_gaussian(gaussian);
  auto gaussian_voxel =
      m.def_submodule("gaussian_voxel", "Differentiable Gaussian voxelization");
  auto occupancy =
      m.def_submodule("occupancy", "LiDAR occupancy targets and losses");
  riemann_py::bind_gaussian_voxel(gaussian_voxel, occupancy);
#endif
}
