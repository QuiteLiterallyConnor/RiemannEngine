"""RiemannEngine Python bindings.

Sparse tensors and sparse convolutional networks, adaptive mixed-resolution
voxelization, and 3D Gaussian splats, backed by the RiemannEngine C++ core
and interoperating with PyTorch tensors and autograd.
"""

import sys as _sys

# The extension links against the torch shared libraries; importing torch
# first makes them resolvable on every platform.
import torch as _torch  # noqa: F401

from . import _C
from ._C import (  # noqa: F401
    CoordinateManager,
    CoordinateMapKey,
    CoordinateMapBackend,
    ConvolutionMode,
    PoolingMode,
    QuantizationMode,
    RegionType,
    RiemannAlgorithm,
    SparseCollation,
    SparseTensor,
    TensorField,
    batch_sparse_collate,
    batched_coordinates,
    cat,
    cuda_enabled,
    dense_coordinates,
    mean,
    sparse_collate,
    sum,
    to_sparse,
    to_sparse_all,
    var,
)

__version__ = _C.__version__

# Register the extension submodules so `import riemannengine.nn` and
# friends work in addition to attribute access.
nn = _C.nn
adaptive = _C.adaptive
_sys.modules[__name__ + ".nn"] = nn
_sys.modules[__name__ + ".adaptive"] = adaptive

gaussian = getattr(_C, "gaussian", None)
if gaussian is not None:
    _sys.modules[__name__ + ".gaussian"] = gaussian

__all__ = [
    "CoordinateManager",
    "CoordinateMapKey",
    "CoordinateMapBackend",
    "ConvolutionMode",
    "PoolingMode",
    "QuantizationMode",
    "RegionType",
    "RiemannAlgorithm",
    "SparseCollation",
    "SparseTensor",
    "TensorField",
    "adaptive",
    "batch_sparse_collate",
    "batched_coordinates",
    "cat",
    "cuda_enabled",
    "dense_coordinates",
    "gaussian",
    "mean",
    "nn",
    "sparse_collate",
    "sum",
    "to_sparse",
    "to_sparse_all",
    "var",
]
