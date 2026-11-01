#ifndef RIEMANN_ENGINE_CUDA_MSVC_COMPAT_HPP
#define RIEMANN_ENGINE_CUDA_MSVC_COMPAT_HPP

// Forced include for CUDA translation units compiled with the MSVC host
// compiler. Keeps CCCL (Thrust/CUB) quiet about the traditional preprocessor.
#if defined(_MSC_VER) && defined(__CUDACC__)

#ifndef CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#endif

#endif // _MSC_VER && __CUDACC__

#endif // RIEMANN_ENGINE_CUDA_MSVC_COMPAT_HPP
