#ifndef RIEMANN_PYTHON_BINDINGS_HPP
#define RIEMANN_PYTHON_BINDINGS_HPP

// torch/extension.h supplies pybind11 (the copy bundled with torch) plus the
// type casters mapping at::Tensor, at::Device, at::ScalarType and c10::Scalar
// to their torch Python counterparts.
#include <torch/extension.h>

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace riemann_py {

void bind_sparse(pybind11::module_ &m);
void bind_nn(pybind11::module_ &m);
void bind_adaptive(pybind11::module_ &m);
#ifdef RIEMANN_ENABLE_GAUSSIAN
void bind_gaussian(pybind11::module_ &m);
#endif

} // namespace riemann_py

#endif
