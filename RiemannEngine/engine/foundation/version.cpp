#include <cstdint>

namespace riemann {

// Engine version constants live in riemann.hpp; this translation unit anchors
// the foundation module in the static library.
const char* engine_build_identifier() noexcept {
  return "riemann-engine";
}

} // namespace riemann
