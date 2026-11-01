#include "bindings.hpp"

#include "engine/foundation/types.hpp"
#include "engine/grid/key.hpp"
#include "engine/grid/space.hpp"
#include "engine/tensor/field.hpp"
#include "engine/tensor/sparse_tensor.hpp"
#include "engine/nn/operations.hpp"
#include "helpers/batching.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace riemann;

namespace riemann_py {

void bind_sparse(py::module_ &m) {
  // --------------------------------------------------------------------
  // Enums
  // --------------------------------------------------------------------
  py::enum_<SparseTensorQuantizationMode::Type>(m, "QuantizationMode")
      .value("RANDOM_SUBSAMPLE", SparseTensorQuantizationMode::RANDOM_SUBSAMPLE)
      .value("UNWEIGHTED_AVERAGE",
             SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE)
      .value("UNWEIGHTED_SUM", SparseTensorQuantizationMode::UNWEIGHTED_SUM)
      .value("MAX_POOL", SparseTensorQuantizationMode::MAX_POOL)
      .value("NO_QUANTIZATION", SparseTensorQuantizationMode::NO_QUANTIZATION);

  py::enum_<RiemannAlgorithm::Mode>(m, "RiemannAlgorithm")
      .value("DEFAULT", RiemannAlgorithm::DEFAULT)
      .value("MEMORY_EFFICIENT", RiemannAlgorithm::MEMORY_EFFICIENT)
      .value("SPEED_OPTIMIZED", RiemannAlgorithm::SPEED_OPTIMIZED);

  py::enum_<CoordinateMapBackend::Type>(m, "CoordinateMapBackend")
      .value("CPU", CoordinateMapBackend::CPU)
      .value("CUDA", CoordinateMapBackend::CUDA);

  py::enum_<RegionType::Type>(m, "RegionType")
      .value("HYPER_CUBE", RegionType::HYPER_CUBE)
      .value("HYPER_CROSS", RegionType::HYPER_CROSS)
      .value("CUSTOM", RegionType::CUSTOM);

  py::enum_<ConvolutionMode::Type>(m, "ConvolutionMode")
      .value("DEFAULT", ConvolutionMode::DEFAULT)
      .value("DIRECT_GEMM", ConvolutionMode::DIRECT_GEMM)
      .value("COPY_GEMM", ConvolutionMode::COPY_GEMM);

  py::enum_<PoolingMode::Type>(m, "PoolingMode")
      .value("LOCAL_SUM_POOLING", PoolingMode::LOCAL_SUM_POOLING)
      .value("LOCAL_AVG_POOLING", PoolingMode::LOCAL_AVG_POOLING)
      .value("LOCAL_MAX_POOLING", PoolingMode::LOCAL_MAX_POOLING)
      .value("GLOBAL_SUM_POOLING_DEFAULT",
             PoolingMode::GLOBAL_SUM_POOLING_DEFAULT)
      .value("GLOBAL_AVG_POOLING_DEFAULT",
             PoolingMode::GLOBAL_AVG_POOLING_DEFAULT)
      .value("GLOBAL_MAX_POOLING_DEFAULT",
             PoolingMode::GLOBAL_MAX_POOLING_DEFAULT)
      .value("GLOBAL_SUM_POOLING_KERNEL",
             PoolingMode::GLOBAL_SUM_POOLING_KERNEL)
      .value("GLOBAL_AVG_POOLING_KERNEL",
             PoolingMode::GLOBAL_AVG_POOLING_KERNEL)
      .value("GLOBAL_MAX_POOLING_KERNEL",
             PoolingMode::GLOBAL_MAX_POOLING_KERNEL)
      .value("GLOBAL_SUM_POOLING_PYTORCH_INDEX",
             PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_AVG_POOLING_PYTORCH_INDEX",
             PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_MAX_POOLING_PYTORCH_INDEX",
             PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX);

  // --------------------------------------------------------------------
  // Coordinate map key and manager
  // --------------------------------------------------------------------
  py::class_<CoordinateMapKey>(m, "CoordinateMapKey")
      .def(py::init<stride_type, std::string>(), py::arg("tensor_stride"),
           py::arg("string_id") = "")
      .def_property_readonly("tensor_stride",
                             &CoordinateMapKey::get_tensor_stride)
      .def_property_readonly("coordinate_size",
                             &CoordinateMapKey::get_coordinate_size)
      .def("is_key_set", &CoordinateMapKey::is_key_set)
      .def("__eq__",
           [](const CoordinateMapKey &a, const CoordinateMapKey &b) {
             return a == b;
           })
      .def("__hash__", &CoordinateMapKey::hash)
      .def("__repr__", &CoordinateMapKey::to_string);

  py::class_<CoordinateManager, std::shared_ptr<CoordinateManager>>(
      m, "CoordinateManager")
      .def(py::init([](int dimension, CoordinateMapBackend::Type backend,
                       RiemannAlgorithm::Mode algorithm) {
             return std::make_shared<CoordinateManager>(
                 dimension, backend, GPUMemoryAllocatorBackend::PYTORCH,
                 algorithm);
           }),
           py::arg("dimension"),
           py::arg("backend") = CoordinateMapBackend::CPU,
           py::arg("algorithm") = RiemannAlgorithm::DEFAULT)
      .def_property_readonly("dimension", &CoordinateManager::D)
      .def("size",
           [](const CoordinateManager &self, const CoordinateMapKey &key) {
             return self.size(key);
           },
           py::arg("coordinate_map_key"))
      .def("get_coordinates", &CoordinateManager::get_coordinates,
           py::arg("coordinate_map_key"));

  // --------------------------------------------------------------------
  // SparseTensor
  // --------------------------------------------------------------------
  py::class_<SparseTensor>(m, "SparseTensor")
      .def(py::init([](at::Tensor features, at::Tensor coordinates,
                       std::shared_ptr<CoordinateManager> manager,
                       SparseTensorQuantizationMode::Type quantization_mode,
                       RiemannAlgorithm::Mode algorithm) {
             return SparseTensor(std::move(features), std::move(coordinates),
                                 std::move(manager), quantization_mode,
                                 GPUMemoryAllocatorBackend::PYTORCH,
                                 algorithm);
           }),
           py::arg("features"), py::arg("coordinates"),
           py::arg("coordinate_manager") =
               std::shared_ptr<CoordinateManager>(),
           py::arg("quantization_mode") =
               SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
           py::arg("algorithm") = RiemannAlgorithm::DEFAULT)
      .def(py::init<at::Tensor, CoordinateMapKey,
                    std::shared_ptr<CoordinateManager>>(),
           py::arg("features"), py::arg("coordinate_map_key"),
           py::arg("coordinate_manager"))
      .def_property_readonly("F", &SparseTensor::F)
      .def_property_readonly("C", &SparseTensor::C)
      .def_property_readonly("features", &SparseTensor::F)
      .def_property_readonly("coordinates", &SparseTensor::C)
      .def_property_readonly("coordinate_map_key",
                             &SparseTensor::coordinate_map_key)
      .def_property_readonly("coordinate_manager",
                             &SparseTensor::coordinate_manager)
      .def_property_readonly("tensor_stride", &SparseTensor::tensor_stride)
      .def_property_readonly("dimension", &SparseTensor::dimension)
      .def_property_readonly("device", &SparseTensor::device)
      .def_property_readonly("dtype", &SparseTensor::dtype)
      .def_property_readonly("requires_grad", &SparseTensor::requires_grad)
      .def_property_readonly(
          "shape",
          [](const SparseTensor &self) {
            return py::make_tuple(self.F().size(0), self.F().size(1));
          })
      .def("__len__",
           [](const SparseTensor &self) { return self.F().size(0); })
      .def("decomposed_coordinates", &SparseTensor::decomposed_coordinates)
      .def("decomposed_features", &SparseTensor::decomposed_features)
      .def("decomposed_coordinates_and_features",
           &SparseTensor::decomposed_coordinates_and_features)
      .def("dense", &SparseTensor::dense, py::arg("shape") = py::none(),
           py::arg("min_coordinate") = py::none())
      .def("sparse", &SparseTensor::sparse)
      .def("__add__",
           [](const SparseTensor &a, const SparseTensor &b) { return a + b; })
      .def("__sub__",
           [](const SparseTensor &a, const SparseTensor &b) { return a - b; })
      .def("__mul__",
           [](const SparseTensor &a, const SparseTensor &b) { return a * b; })
      .def("__truediv__",
           [](const SparseTensor &a, const SparseTensor &b) { return a / b; })
      .def("__add__",
           [](const SparseTensor &a, double s) { return a + s; })
      .def("__radd__",
           [](const SparseTensor &a, double s) { return a + s; })
      .def("__mul__",
           [](const SparseTensor &a, double s) { return a * s; })
      .def("__rmul__",
           [](const SparseTensor &a, double s) { return a * s; })
      .def("to",
           [](const SparseTensor &self, at::Device device) {
             return self.to(device);
           },
           py::arg("device"))
      .def("to",
           [](const SparseTensor &self, at::ScalarType dtype) {
             return self.to(dtype);
           },
           py::arg("dtype"))
      .def("float", &SparseTensor::float_)
      .def("double", &SparseTensor::double_)
      .def("detach", &SparseTensor::detach)
      .def("__repr__", [](const SparseTensor &self) {
        std::ostringstream out;
        out << "SparseTensor(N=" << self.F().size(0)
            << ", C=" << self.F().size(1) << ", D=" << self.dimension()
            << ", device=" << self.device() << ")";
        return out.str();
      });

  // --------------------------------------------------------------------
  // TensorField
  // --------------------------------------------------------------------
  py::class_<TensorField>(m, "TensorField")
      .def(py::init<at::Tensor, at::Tensor,
                    std::shared_ptr<CoordinateManager>,
                    SparseTensorQuantizationMode::Type>(),
           py::arg("features"), py::arg("coordinates"),
           py::arg("coordinate_manager") =
               std::shared_ptr<CoordinateManager>(),
           py::arg("quantization_mode") =
               SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE)
      .def("sparse", &TensorField::sparse,
           py::arg("tensor_stride") = stride_type{},
           py::arg("coordinate_map_key") = py::none(),
           py::arg("quantization_mode") =
               SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE)
      .def("splat", &TensorField::splat)
      .def("inverse_mapping", &TensorField::inverse_mapping,
           py::arg("sparse_coordinate_map_key"))
      .def_property_readonly("F", &TensorField::F)
      .def_property_readonly("C", &TensorField::C)
      .def_property_readonly("features", &TensorField::F)
      .def_property_readonly("coordinates", &TensorField::C)
      .def_property_readonly("coordinate_field_map_key",
                             &TensorField::coordinate_field_map_key)
      .def_property_readonly("coordinate_manager",
                             &TensorField::coordinate_manager)
      .def_property_readonly("dimension", &TensorField::dimension)
      .def_property_readonly("device", &TensorField::device)
      .def_property_readonly("dtype", &TensorField::dtype)
      .def("__len__",
           [](const TensorField &self) { return self.F().size(0); });

  // --------------------------------------------------------------------
  // Batching helpers
  // --------------------------------------------------------------------
  m.def("batched_coordinates", &batched_coordinates, py::arg("coordinates"),
        py::arg("dtype") = torch::kInt32, py::arg("device") = py::none(),
        "Prepends batch indices and concatenates per-sample coordinates");
  m.def("sparse_collate", &sparse_collate, py::arg("coordinates"),
        py::arg("features"), py::arg("labels") = std::vector<at::Tensor>{},
        py::arg("dtype") = torch::kInt32, py::arg("device") = py::none());
  m.def("batch_sparse_collate", &batch_sparse_collate, py::arg("data"),
        py::arg("dtype") = torch::kInt32, py::arg("device") = py::none());

  py::class_<SparseCollation>(m, "SparseCollation")
      .def(py::init<int64_t, c10::ScalarType,
                    c10::optional<torch::Device>>(),
           py::arg("limit_numpoints") = -1,
           py::arg("dtype") = torch::kInt32, py::arg("device") = py::none())
      .def("__call__", &SparseCollation::operator(), py::arg("data"));

  // --------------------------------------------------------------------
  // Functional operations
  // --------------------------------------------------------------------
  m.def("cat", &riemann::cat, py::arg("tensors"),
        "Concatenates sparse tensors along the feature dimension");
  m.def("sum", &riemann::sum, py::arg("tensors"));
  m.def("mean", &riemann::mean, py::arg("tensors"));
  m.def("var", &riemann::var, py::arg("tensors"));
  m.def("dense_coordinates", &dense_coordinates, py::arg("shape"));
  m.def("to_sparse", &to_sparse, py::arg("dense_tensor"),
        "Converts a dense tensor to a SparseTensor, dropping zero rows");
  m.def("to_sparse_all", &to_sparse_all, py::arg("dense_tensor"),
        py::arg("coordinates") = py::none(),
        "Converts a dense tensor to a SparseTensor, keeping every cell");
}

} // namespace riemann_py
