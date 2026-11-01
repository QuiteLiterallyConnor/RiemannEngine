#ifndef RIEMANN_ENGINE_GRID_MAPS_HPP
#define RIEMANN_ENGINE_GRID_MAPS_HPP

#include "engine/foundation/errors.hpp"
#include "engine/foundation/types.hpp"

#include <algorithm>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

namespace riemann {

/*
 * Host-side kernel-map containers. For each kernel offset k, first[k] holds
 * input row indices and second[k] the matching output row indices.
 */
using cpu_in_map = default_types::index_vector_type;
using cpu_out_map = default_types::index_vector_type;

using cpu_in_maps = std::vector<cpu_in_map>;
using cpu_out_maps = std::vector<cpu_out_map>;

struct cpu_kernel_map : std::pair<cpu_in_maps, cpu_out_maps> {
  using index_type = default_types::index_type;
  using index_pair =
      std::pair<default_types::index_type, default_types::index_type>;

  cpu_kernel_map() : std::pair<cpu_in_maps, cpu_out_maps>() {}
  cpu_kernel_map(std::pair<cpu_in_maps, cpu_out_maps> const &other)
      : std::pair<cpu_in_maps, cpu_out_maps>(other) {}

  // Build a batch-decomposed map (origin map): rows are grouped per unique
  // batch, ordered by the output (batch row) index.
  cpu_kernel_map(std::vector<index_pair> &in_out,
                 std::vector<default_types::dcoordinate_type> const
                     &unique_batch_indicies) {
    auto comp = [](index_pair const &l, index_pair const &r) {
      return l.second < r.second;
    };
    std::sort(in_out.begin(), in_out.end(), comp);

    auto const kernel_volume = unique_batch_indicies.size();
    this->first.resize(kernel_volume);
    this->second.resize(kernel_volume);

    for (index_type k = 0; k < unique_batch_indicies.size(); ++k) {
      auto const lb = std::lower_bound(in_out.begin(), in_out.end(),
                                       index_pair{0, k}, comp);
      auto const ub = std::upper_bound(in_out.begin(), in_out.end(),
                                       index_pair{0, k}, comp);
      auto const curr_size = ub - lb;
      default_types::index_type start_index = lb - in_out.begin();
      LOG_DEBUG("batch row_index:", k, "curr_size:", curr_size,
                "start_index:", start_index);
      auto &in_map = this->first[k];
      auto &out_map = this->second[k];
      in_map.resize(curr_size);
      out_map.resize(curr_size);
      for (uint32_t i = 0; i < curr_size; ++i) {
        auto const &curr_pair = in_out[i + start_index];
        in_map[i] = curr_pair.first;
        out_map[i] = curr_pair.second;
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  cpu_kernel_map const &kernel_map) {
    uint32_t map_size = 0;
    for (auto const &v : kernel_map.first) {
      map_size += v.size();
    }
    out << "cpu_kernel_map: number of unique maps:" << kernel_map.first.size()
        << ", kernel map size:" << map_size;
    return out;
  }
};

using cpu_kernel_map_reference = std::pair<cpu_in_maps &, cpu_out_maps &>;

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_MAPS_HPP
