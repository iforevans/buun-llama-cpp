#include "fusion-check.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

int main() {
    const std::map<std::string, uint64_t> baseline = {
        {"a|0|any|RMS_NORM+MUL", 2},
        {"b|1|prefill|ADD+ADD", 3},
    };
    std::set<std::string> observed;
    for (const auto & entry : baseline) observed.insert(entry.first);
    require(fusion_missing_rows(baseline, observed).empty(), "complete rows");
    observed.erase("a|0|any|RMS_NORM+MUL");
    require(fusion_missing_rows(baseline, observed).size() == 1, "deleted pattern detected");
    require(fusion_missing_rows(baseline, observed, "b|1|").empty(), "single model scope");
    require(fusion_missing_rows(baseline, observed, "a|0|").size() == 1, "selected model missing");
    observed.clear();
    require(fusion_missing_rows(baseline, observed).size() == 2, "zero observations cannot pass");
    observed.insert("a|0|prefill|RMS_NORM+MUL");
    require(fusion_missing_rows(baseline, observed, "a|0|").size() == 1, "mode mismatch detected");

    require(fusion_nmse_ok(0.0) && fusion_nmse_ok(1e-4), "finite accepted boundary");
    require(!fusion_nmse_ok(1.001e-4), "over threshold rejected");
    require(!fusion_nmse_ok(std::numeric_limits<double>::quiet_NaN()), "NaN rejected");
    require(!fusion_nmse_ok(std::numeric_limits<double>::infinity()), "+infinity rejected");
    require(!fusion_nmse_ok(-std::numeric_limits<double>::infinity()), "-infinity rejected");
    puts("fusion row-set and finite-value checks: PASS");
}
