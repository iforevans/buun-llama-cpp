#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

inline bool fusion_nmse_ok(double value) {
    return std::isfinite(value) && value <= 1e-4;
}

// A single-model check uses arch|moe| as its scope. A directory check must cover
// the entire baseline, including rows for models/patterns which disappeared.
inline std::vector<std::string> fusion_missing_rows(
        const std::map<std::string, uint64_t> & baseline,
        const std::set<std::string> & observed,
        const std::string & scope = {}) {
    std::vector<std::string> missing;
    for (const auto & entry : baseline) {
        if ((scope.empty() || entry.first.compare(0, scope.size(), scope) == 0) &&
                !observed.count(entry.first)) {
            missing.push_back(entry.first);
        }
    }
    return missing;
}
