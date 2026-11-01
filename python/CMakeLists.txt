# Builds the riemannengine._C extension module against the torch package of
# the Python found by the parent project. pybind11 headers come from the
# torch wheel itself (torch/include/pybind11) so the extension and
# torch_python share one pybind11 ABI.

# torch_python provides the at::Tensor <-> torch.Tensor casters used by the
# bindings. The Torch CMake package does not export it as a target.
find_library(TORCH_PYTHON_LIBRARY torch_python
    HINTS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
)
message(STATUS "torch_python library: ${TORCH_PYTHON_LIBRARY}")

add_library(riemann_python MODULE
    src/module.cpp
    src/sparse_bindings.cpp
    src/nn_bindings.cpp
    src/adaptive_bindings.cpp
)

if(RIEMANN_ENABLE_GAUSSIAN)
    target_sources(riemann_python PRIVATE
        src/gaussian_bindings.cpp
    )
endif()

set_target_properties(riemann_python PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
)

if(WIN32)
    set_target_properties(riemann_python PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
    set_target_properties(riemann_python PROPERTIES SUFFIX ".so")
endif()

target_include_directories(riemann_python PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/RiemannEngine"
    "${CMAKE_SOURCE_DIR}/RiemannEngine/vendor"
    "${CMAKE_SOURCE_DIR}/third_party"
)

target_link_libraries(riemann_python PRIVATE
    riemann_cpp
    Python::Module
    "${TORCH_PYTHON_LIBRARY}"
)

if(RIEMANN_TORCH_CXX_FLAGS)
    target_compile_options(riemann_python PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
    )
endif()

if(MSVC)
    # Binding translation units exceed the default COFF section limit.
    target_compile_options(riemann_python PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    )
else()
    # Recommended for pybind11 modules; shrinks the binary and avoids
    # cross-module RTTI clashes.
    target_compile_options(riemann_python PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
    )
endif()

# Picked up by scikit-build-core through install.components = ["python"];
# plain `cmake --install` places it under <prefix>/riemannengine as well.
install(TARGETS riemann_python
    LIBRARY DESTINATION riemannengine COMPONENT python
    RUNTIME DESTINATION riemannengine COMPONENT python
)
