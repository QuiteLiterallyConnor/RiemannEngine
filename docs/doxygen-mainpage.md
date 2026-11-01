# RiemannEngine API reference {#mainpage}

Generated reference for the RiemannEngine C++ headers. For guides, tutorials and examples, see the
[documentation site](https://quiteliterallyconnor.github.io/RiemannEngine/).

## Where to start

| Area | Namespace | Main types | Umbrella header |
|---|---|---|---|
| Sparse tensors and layers | `riemann` | `riemann::SparseTensor`, `riemann::TensorField`, `riemann::CoordinateManager` | `riemann.hpp` |
| Training helpers | `riemann::training` | `riemann::training::OptimizerConfig`, `riemann::training::SparseBatch` | `riemann.hpp` |
| Adaptive voxelization | `riemann::adaptive` | `riemann::adaptive::AdaptiveSparseTensor`, `riemann::adaptive::Topology` | `octree/adaptive.hpp` |
| Gaussian splats | `riemann::gaussian` | `riemann::gaussian::GaussianModel`, `riemann::gaussian::GaussianTrainer` | `riemann.hpp` (with `RIEMANN_ENABLE_GAUSSIAN`) |

Headers under `layers/`, `helpers/`, `dynamic/`, `normal/`, `core/` and `learning/`, and most
root-level headers, only forward to the canonical headers in `engine/`, `octree/` and `splat/`.
