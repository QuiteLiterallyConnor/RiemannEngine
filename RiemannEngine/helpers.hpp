#ifndef RIEMANN_ENGINE_VENDOR_HELPERS_SHIM_HPP
#define RIEMANN_ENGINE_VENDOR_HELPERS_SHIM_HPP

// Compatibility shim: the vendored robin_hood.h (vendor/robin_hood.h) includes
// "../helpers.hpp" relative to its own directory, which resolves to this file.
// The vendored header is kept byte-identical, so the utility declarations it
// historically pulled in are provided here instead.

#include "engine/foundation/errors.hpp"

#endif // RIEMANN_ENGINE_VENDOR_HELPERS_SHIM_HPP
