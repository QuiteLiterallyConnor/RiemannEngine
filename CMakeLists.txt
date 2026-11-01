
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(RiemannEngine
    VERSION 1.0.0
    LANGUAGES CXX
)

# ============================================================
# Options
# ============================================================

option(CPU_ONLY "Build without RiemannEngine CUDA kernels" OFF)
option(RIEMANN_BUILD_TESTS "Build test executables" ON)
option(RIEMANN_ENABLE_OPENMP "Enable OpenMP support" ON)
option(RIEMANN_ENABLE_GAUSSIAN
    "Enable native Gaussian representation support" ON)
option(RIEMANN_BUILD_GAUSSIAN_BENCHMARKS
    "Build Gaussian tensor and optimization benchmarks" ON)
option(RIEMANN_BUILD_PYTHON
    "Build the riemannengine Python bindings" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ============================================================
# Compiler cache (speeds up rebuilds after header/unrelated edits)
# ============================================================

option(RIEMANN_USE_CCACHE "Use ccache/sccache when available" ON)

if(RIEMANN_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(RIEMANN_CCACHE_PROGRAM NAMES ccache sccache)
    if(RIEMANN_CCACHE_PROGRAM)
        message(STATUS "Compiler cache: ${RIEMANN_CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${RIEMANN_CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${RIEMANN_CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${RIEMANN_CCACHE_PROGRAM}")
        if(MSVC)
            # Caches cannot handle /Zi PDBs; embed debug info in objects.
            if(POLICY CMP0141)
                cmake_policy(SET CMP0141 NEW)
                set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
            endif()
        endif()
    endif()
endif()

# ============================================================
# Python bindings prerequisites
# ============================================================

# The bindings must compile against the LibTorch bundled with the active
# Python's torch package so the extension and torch share one ABI. Resolve
# that LibTorch before Torch/CUDA discovery below.
if(RIEMANN_BUILD_PYTHON)

    find_package(Python 3.9 REQUIRED
        COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE RIEMANN_TORCH_PYTHON_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RIEMANN_TORCH_PYTHON_RC
    )

    if(NOT RIEMANN_TORCH_PYTHON_RC EQUAL 0)
        message(FATAL_ERROR
            "RIEMANN_BUILD_PYTHON requires the 'torch' package in "
            "${Python_EXECUTABLE}. Install it first (pip install torch), "
            "or build with 'pip install --no-build-isolation .'")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH "${RIEMANN_TORCH_PYTHON_PREFIX}")
    message(STATUS
        "Python torch CMake prefix: ${RIEMANN_TORCH_PYTHON_PREFIX}")

    # A CPU-only torch wheel cannot host the engine's CUDA kernels.
    if(NOT CPU_ONLY)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                "import torch; print(torch.version.cuda or '')"
            OUTPUT_VARIABLE RIEMANN_TORCH_PYTHON_CUDA
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(RIEMANN_TORCH_PYTHON_CUDA STREQUAL "")
            message(STATUS
                "Python torch build has no CUDA; forcing CPU_ONLY "
                "for the bindings build")
            set(CPU_ONLY ON)
        endif()
    endif()

endif()

# ============================================================
# CUDA configuration
# ============================================================

if(NOT CPU_ONLY)

    # Establish the toolkit before Torch/Caffe2 discovery.
    # The Windows script should supply CUDAToolkit_ROOT and
    # CMAKE_CUDA_COMPILER explicitly.
    
    if(MSVC)
        set(CMAKE_CUDA_FLAGS_INIT
            "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler"
        )
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    # Respect -DCMAKE_CUDA_ARCHITECTURES from the caller.
    # Default to Ampere sm_86 for the current build machine.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
            "CUDA architectures" FORCE)
    endif()

    message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Normalize cuDNN paths supplied by build scripts.
    if(DEFINED CUDNN_INCLUDE_DIR AND CUDNN_INCLUDE_DIR)
        set(CUDNN_INCLUDE_PATH
            "${CUDNN_INCLUDE_DIR}"
            CACHE PATH "cuDNN include directory" FORCE)
    endif()

    if(DEFINED CUDNN_LIBRARY AND CUDNN_LIBRARY)
        set(CUDNN_LIBRARY_PATH
            "${CUDNN_LIBRARY}"
            CACHE FILEPATH "cuDNN import library" FORCE)
    endif()

endif()

# ============================================================
# LibTorch
# ============================================================

# LibTorch remains required in both CPU and CUDA builds.
# CPU_ONLY disables Riemann CUDA kernels, not CUDA
# dependencies of a CUDA-enabled LibTorch distribution.

find_package(Torch REQUIRED)

# Older LibTorch configs hardcode the legacy NvToolsExt SDK on Windows. It is not
# shipped with CUDA 13.x (NVTX 3 is header-only in the toolkit), so drop the
# missing paths instead of failing at generate time.
if(MSVC AND NOT EXISTS "C:/Program Files/NVIDIA Corporation/NvToolsExt/include")
    list(FILTER TORCH_INCLUDE_DIRS EXCLUDE REGEX "NvToolsExt")
    list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvToolsExt")
    if(TARGET torch)
        set_target_properties(torch PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")
    endif()
endif()

message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Torch may provide compiler flags needed for ABI
# compatibility. Keep these as separate list elements.
separate_arguments(
    RIEMANN_TORCH_CXX_FLAGS
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

# ============================================================
# OpenMP
# ============================================================

if(RIEMANN_ENABLE_OPENMP)
    if(MSVC)
        # '#pragma omp atomic capture' requires the LLVM OpenMP runtime on MSVC.
        set(OpenMP_RUNTIME_MSVC llvm)
        unset(OpenMP_CXX_FLAGS CACHE)
        unset(OpenMP_CXX_LIB_NAMES CACHE)
    endif()
    find_package(OpenMP COMPONENTS CXX QUIET)

    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING
            "OpenMP was requested but not found. "
            "Continuing without OpenMP.")
    endif()
endif()

# ============================================================
# Source files
# ============================================================

file(GLOB RIEMANN_CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/engine/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/octree/*.cpp"
)

list(FILTER RIEMANN_CPU_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER RIEMANN_CPU_SOURCES EXCLUDE REGEX ".*example.*")

if(RIEMANN_ENABLE_GAUSSIAN)
    file(GLOB RIEMANN_GAUSSIAN_CPU_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/splat/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/gaussian_voxel/*.cpp"
    )
    list(APPEND RIEMANN_CPU_SOURCES ${RIEMANN_GAUSSIAN_CPU_SOURCES})
endif()

set(RIEMANN_GPU_SOURCES)

if(NOT CPU_ONLY)

    file(GLOB RIEMANN_GPU_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/engine/*/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/octree/*.cu"
    )

    list(FILTER RIEMANN_GPU_SOURCES EXCLUDE REGEX ".*test.*")
    list(FILTER RIEMANN_GPU_SOURCES EXCLUDE REGEX ".*example.*")

    if(RIEMANN_ENABLE_GAUSSIAN)
        file(GLOB RIEMANN_GAUSSIAN_GPU_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/splat/*.cu"
        )
        list(APPEND RIEMANN_GPU_SOURCES ${RIEMANN_GAUSSIAN_GPU_SOURCES})
    endif()

endif()

# ============================================================
# Main library
# ============================================================

add_library(riemann_cpp STATIC
    ${RIEMANN_CPU_SOURCES}
    ${RIEMANN_GPU_SOURCES}
)

add_library(RiemannEngine::riemann_cpp
    ALIAS riemann_cpp)

add_library(RiemannEngine::RiemannEngine
    ALIAS riemann_cpp)

set_target_properties(riemann_cpp PROPERTIES
    EXPORT_NAME riemann_cpp
    POSITION_INDEPENDENT_CODE ON
)

if(NOT CPU_ONLY)
    set_target_properties(riemann_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# ============================================================
# Include directories
# ============================================================

target_include_directories(riemann_cpp
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"

    PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/riemann>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/riemann/vendor>
)

# ============================================================
# Compile definitions
# ============================================================

target_compile_definitions(riemann_cpp
    PUBLIC
        RIEMANN_NO_PYBIND
)

if(CPU_ONLY)
    target_compile_definitions(riemann_cpp
        PUBLIC CPU_ONLY)
else()
    target_compile_definitions(riemann_cpp
        PUBLIC USE_CUDA)
endif()

if(RIEMANN_ENABLE_GAUSSIAN)
    target_compile_definitions(riemann_cpp
        PUBLIC RIEMANN_ENABLE_GAUSSIAN)
endif()

if(WIN32)
    target_compile_definitions(riemann_cpp
        PUBLIC
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
endif()

# ============================================================
# Compiler configuration
# ============================================================

if(RIEMANN_TORCH_CXX_FLAGS)
    target_compile_options(riemann_cpp
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
    )
endif()

if(MSVC)

    target_compile_options(riemann_cpp PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    )

    target_compile_options(riemann_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd5105>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
    )

    if(NOT CPU_ONLY)

        set(RIEMANN_CUDA_COMMON
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/engine/cuda/msvc_compat.hpp"
        )

        if(EXISTS "${RIEMANN_CUDA_COMMON}")

            # Forward the forced-include option to MSVC
            # when compiling CUDA translation units.
            target_compile_options(riemann_cpp PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FI${RIEMANN_CUDA_COMMON}>
            )

        else()
            message(WARNING
                "CUDA workaround header not found: "
                "${RIEMANN_CUDA_COMMON}")
        endif()

        target_compile_definitions(riemann_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING>
        )

        # CCCL (Thrust/CUB) requires the conforming MSVC preprocessor.
        target_compile_options(riemann_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd5105>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4267>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4804>
        )

    endif()

endif()

# ============================================================
# Dependencies
# ============================================================

# TORCH_LIBRARIES holds absolute paths into the LibTorch used for the build.
# The installed package resolves LibTorch again on the consumer machine
# (see cmake/RiemannEngineConfig.cmake.in) through RiemannEngine::torch.
target_link_libraries(riemann_cpp
    PUBLIC
        "$<BUILD_INTERFACE:${TORCH_LIBRARIES}>"
        "$<INSTALL_INTERFACE:RiemannEngine::torch>"
)

set(RIEMANN_PKG_USES_OPENMP_TARGET OFF)

if(NOT CPU_ONLY)

    target_link_libraries(riemann_cpp
        PUBLIC
            CUDA::cudart
            CUDA::cusparse
            CUDA::cublas
    )

    target_include_directories(riemann_cpp PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        "${CUDAToolkit_INCLUDE_DIRS}/cccl"
    )

endif()

if(RIEMANN_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    set(RIEMANN_IOMP_DLL "${TORCH_INSTALL_PREFIX}/lib/libiomp5md.dll")
    set(RIEMANN_IOMP_LIB "${CMAKE_BINARY_DIR}/libiomp5md.lib")
    if(MSVC AND TORCH_INSTALL_PREFIX AND EXISTS "${RIEMANN_IOMP_DLL}")
        # LibTorch loads libiomp5md, and a second LLVM-style runtime
        # (libomp140) in the same process aborts with "OMP: Error #15".
        # LibTorch ships no import library, so build one from the DLL and link
        # the engine against the runtime that is already loaded.
        if(NOT EXISTS "${RIEMANN_IOMP_LIB}")
            get_filename_component(_riemann_msvc_bin "${CMAKE_LINKER}" DIRECTORY)
            execute_process(
                COMMAND "${_riemann_msvc_bin}/dumpbin.exe" /nologo /exports "${RIEMANN_IOMP_DLL}"
                OUTPUT_VARIABLE _riemann_exports RESULT_VARIABLE _riemann_dumpbin_rc)
            set(_riemann_def "LIBRARY libiomp5md\nEXPORTS\n")
            string(REGEX MATCHALL "\n +[0-9]+ +[0-9A-F]+ +[0-9A-F]+ +[A-Za-z_0-9@?$]+"
                   _riemann_rows "${_riemann_exports}")
            foreach(_row IN LISTS _riemann_rows)
                string(REGEX REPLACE "^\n +[0-9]+ +[0-9A-F]+ +[0-9A-F]+ +" "" _name "${_row}")
                string(APPEND _riemann_def "    ${_name}\n")
            endforeach()
            file(WRITE "${CMAKE_BINARY_DIR}/libiomp5md.def" "${_riemann_def}")
            execute_process(
                COMMAND "${_riemann_msvc_bin}/lib.exe" /nologo /machine:x64
                        "/def:${CMAKE_BINARY_DIR}/libiomp5md.def" "/out:${RIEMANN_IOMP_LIB}"
                RESULT_VARIABLE _riemann_lib_rc)
        endif()
    endif()

    if(EXISTS "${RIEMANN_IOMP_LIB}")
        message(STATUS "OpenMP runtime: LibTorch libiomp5md")
        target_compile_options(riemann_cpp PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
        target_link_libraries(riemann_cpp PUBLIC
            "$<BUILD_INTERFACE:${RIEMANN_IOMP_LIB}>"
            "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/libiomp5md.lib>")
        target_link_options(riemann_cpp PUBLIC /NODEFAULTLIB:libomp140.x86_64.lib)
        install(FILES "${RIEMANN_IOMP_LIB}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    else()
        target_link_libraries(riemann_cpp
            PUBLIC OpenMP::OpenMP_CXX)
        set(RIEMANN_PKG_USES_OPENMP_TARGET ON)
    endif()
endif()

# ============================================================
# Test executable
# ============================================================

if(RIEMANN_BUILD_TESTS)

    add_executable(train
        tests/train.cpp
    )

    target_link_libraries(train
        PRIVATE riemann_cpp
    )

    target_include_directories(train PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    )

    if(RIEMANN_TORCH_CXX_FLAGS)
        target_compile_options(train PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
        )
    endif()

endif()

# Adaptive voxelization tests and benchmark (see plan.txt).
option(RIEMANN_BUILD_ADAPTIVE_TESTS "Build adaptive voxelization tests and benchmarks" ON)

if(RIEMANN_BUILD_ADAPTIVE_TESTS)

    add_executable(adaptive_tests
        tests/adaptive/main.cpp
        tests/adaptive/test_geometry.cpp
        tests/adaptive/test_topology.cpp
        tests/adaptive/test_numerical.cpp
        tests/adaptive/test_gradients.cpp
        tests/adaptive/test_regression.cpp
        tests/adaptive/test_sphere.cpp
        tests/adaptive/test_voxelize_obj.cpp
    )

    add_executable(adaptive_bench
        tests/adaptive/bench_adaptive.cpp
    )

    add_executable(voxelize_obj_bench
        tests/adaptive/bench_voxelize_obj.cpp
    )

    add_executable(sphere
        tests/sphere.cpp
    )

    target_compile_definitions(adaptive_tests PRIVATE
        RIEMANN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )
    target_compile_definitions(voxelize_obj_bench PRIVATE
        RIEMANN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )

    foreach(tgt adaptive_tests adaptive_bench voxelize_obj_bench sphere)
        target_link_libraries(${tgt} PRIVATE riemann_cpp)
        target_include_directories(${tgt} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/adaptive"
        )
        if(RIEMANN_TORCH_CXX_FLAGS)
            target_compile_options(${tgt} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
            )
        endif()
    endforeach()

    enable_testing()
    add_test(NAME adaptive_tests COMMAND adaptive_tests)

endif()

# ============================================================
# CPU convolution kernel tests
# ============================================================

if(RIEMANN_BUILD_TESTS)
    add_executable(cpu_convolution_tests
        tests/convolution/main.cpp
        tests/convolution/test_cpu_convolution.cpp
    )
    target_link_libraries(cpu_convolution_tests PRIVATE riemann_cpp)
    target_include_directories(cpu_convolution_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    if(RIEMANN_TORCH_CXX_FLAGS)
        target_compile_options(cpu_convolution_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
        )
    endif()
    enable_testing()
    add_test(NAME cpu_convolution_tests COMMAND cpu_convolution_tests)
endif()

# ============================================================
# High-level engine tests (batching, layers, end-to-end pipelines)
# ============================================================

if(RIEMANN_BUILD_TESTS)
    add_executable(engine_tests
        tests/engine/main.cpp
        tests/engine/test_batching.cpp
        tests/engine/test_layers.cpp
        tests/engine/test_pipeline.cpp
    )
    target_link_libraries(engine_tests PRIVATE riemann_cpp)
    target_include_directories(engine_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    if(RIEMANN_TORCH_CXX_FLAGS)
        target_compile_options(engine_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
        )
    endif()
    enable_testing()
    add_test(NAME engine_tests COMMAND engine_tests)
endif()

# ============================================================
# Gaussian tests
# ============================================================

if(RIEMANN_BUILD_TESTS AND RIEMANN_ENABLE_GAUSSIAN)
    add_executable(gaussian_tests
        tests/gaussian/main.cpp
        tests/gaussian/test_gaussian3d.cpp
        tests/gaussian/test_gaussian_tensor.cpp
        tests/gaussian/test_gaussian_ply.cpp
        tests/gaussian/test_gaussian_render.cpp
        tests/gaussian/test_gaussian_voxel.cpp
        tests/gaussian/test_lidar_occupancy.cpp
    )

    target_compile_definitions(gaussian_tests PRIVATE
        RIEMANN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )
    target_link_libraries(gaussian_tests PRIVATE riemann_cpp)
    target_include_directories(gaussian_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/adaptive"
    )
    if(RIEMANN_TORCH_CXX_FLAGS)
        target_compile_options(gaussian_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
        )
    endif()

    # ============================================================
    # Gaussian benchmarks
    # ============================================================

    if(RIEMANN_BUILD_GAUSSIAN_BENCHMARKS AND RIEMANN_ENABLE_GAUSSIAN)
        add_executable(gaussian_bench
            benchmarks/gaussian_bench.cpp
        )
        target_link_libraries(gaussian_bench PRIVATE riemann_cpp)
        target_include_directories(gaussian_bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        )
        target_compile_definitions(gaussian_bench PRIVATE
            RIEMANN_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        )
        if(RIEMANN_TORCH_CXX_FLAGS)
            target_compile_options(gaussian_bench PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
            )
        endif()
    endif()

    if(RIEMANN_BUILD_GAUSSIAN_BENCHMARKS AND RIEMANN_ENABLE_GAUSSIAN)
        add_executable(gaussian_ply_bench
            benchmarks/gaussian_ply_bench.cpp
        )
        target_link_libraries(gaussian_ply_bench PRIVATE riemann_cpp)
        target_include_directories(gaussian_ply_bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine"
        )
        target_compile_definitions(gaussian_ply_bench PRIVATE
            RIEMANN_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            RIEMANN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        )
        if(RIEMANN_TORCH_CXX_FLAGS)
            target_compile_options(gaussian_ply_bench PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${RIEMANN_TORCH_CXX_FLAGS}>
            )
        endif()
    endif()

    enable_testing()
    add_test(NAME gaussian_tests COMMAND gaussian_tests)
endif()

# Windows resolves DLLs next to the executable, so without this the binaries
# exit silently (0xC0000135) unless LibTorch's lib directory is on PATH.
option(RIEMANN_COPY_TORCH_DLLS "Copy LibTorch DLLs next to the built executables (Windows)" ON)
if(WIN32 AND RIEMANN_COPY_TORCH_DLLS AND TORCH_INSTALL_PREFIX)
    file(GLOB RIEMANN_TORCH_RUNTIME_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(RIEMANN_TORCH_RUNTIME_DLLS)
        add_custom_target(riemann_copy_torch_dlls ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${RIEMANN_TORCH_RUNTIME_DLLS} "${CMAKE_BINARY_DIR}"
            COMMENT "Copying LibTorch DLLs to ${CMAKE_BINARY_DIR}"
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

# ============================================================
# Python bindings
# ============================================================

if(RIEMANN_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ============================================================
# Installation
# ============================================================

install(TARGETS riemann_cpp
    EXPORT RiemannEngineTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/riemann"
    FILES_MATCHING
        PATTERN "splat" EXCLUDE
        PATTERN "gaussian_voxel" EXCLUDE
        PATTERN "normal" EXCLUDE
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hxx"
        PATTERN "*.inl"
        PATTERN "*.cuh"
)

if(RIEMANN_ENABLE_GAUSSIAN)
    foreach(gdir normal splat gaussian_voxel)
        install(DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/RiemannEngine/${gdir}/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/riemann/${gdir}"
            FILES_MATCHING
                PATTERN "*.hpp"
                PATTERN "*.h"
                PATTERN "*.hh"
                PATTERN "*.hxx"
                PATTERN "*.inl"
                PATTERN "*.cuh"
        )
    endforeach()
endif()

# ============================================================
# Package configuration
# ============================================================

set(CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/RiemannEngine"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RiemannEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/RiemannEngineConfig.cmake"
    INSTALL_DESTINATION "${CONFIG_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/RiemannEngineConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT RiemannEngineTargets
    FILE RiemannEngineTargets.cmake
    NAMESPACE RiemannEngine::
    DESTINATION "${CONFIG_INSTALL_DIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/RiemannEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/RiemannEngineConfigVersion.cmake"
    DESTINATION "${CONFIG_INSTALL_DIR}"
)

# ============================================================
# Configuration summary
# ============================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "RiemannEngine C++ Configuration")
message(STATUS "========================================")
message(STATUS "Version:       ${PROJECT_VERSION}")
message(STATUS "Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "CPU only:      ${CPU_ONLY}")
message(STATUS "Torch:         ${Torch_VERSION}")
message(STATUS "OpenMP:        ${OpenMP_CXX_FOUND}")
message(STATUS "Build tests:   ${RIEMANN_BUILD_TESTS}")

if(NOT CPU_ONLY)
    message(STATUS "CUDA:          ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

message(STATUS "========================================")
