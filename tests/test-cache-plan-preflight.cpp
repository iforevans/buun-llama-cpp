#include "server-cache-plan-preflight-internal.h"
#include "server-cache-plan-authority.h"
#include "server-http.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
        __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static common_cache_plan_record live_record() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * live = rec.find_or_add(
        common_cache_plan_provider::live_slot, 7,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 7,
        common_cache_plan_selection::similarity);
    CHECK(live != nullptr);
    live->accept();
    live->lcp_tokens = llama_cache_acct_value::measured(96);
    rec.destruction_legacy_plan_candidate = int32_t(live - rec.inventory.data());
    return rec;
}

static void test_snapshot_selection_and_missing_evidence() {
    auto rec = live_record();
    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(view.provider_available);
    CHECK(view.provider == common_cache_plan_provider::live_slot);
    CHECK(view.reuse_tokens.value == 96 && view.replay_tokens.value == 4);
    CHECK(view.restore_bytes.value == 0);
    CHECK(view.cache_hit == server_cache_plan_preflight_cache_hit::partial);
    CHECK(view.target_relation == server_cache_plan_preflight_target_relation::same_as_legacy);

    rec.selection = common_cache_plan_selection::by_id;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(view.target_relation == server_cache_plan_preflight_target_relation::forced_slot);
    CHECK(server_cache_plan_preflight_build_view(rec, 8, view));
    CHECK(!view.provider_available); // never retarget

    rec.inventory[0].lcp_tokens = {};
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(view.reuse_tokens.state == llama_cache_acct_known::unknown);
    CHECK(view.replay_tokens.state == llama_cache_acct_known::unknown);
    rec.destruction_legacy_plan_candidate = -1;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(!view.provider_available);

    rec = live_record();
    for (uint32_t i = 1; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        CHECK(rec.find_or_add(common_cache_plan_provider::cold_replay, int32_t(i),
            COMMON_CACHE_PLAN_PHASE_LRU, int32_t(i), common_cache_plan_selection::lru));
    }
    CHECK(!rec.find_or_add(common_cache_plan_provider::cold_replay, 999,
        COMMON_CACHE_PLAN_PHASE_LRU, 999, common_cache_plan_selection::lru));
    CHECK(rec.inventory_saturated());
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(!view.provider_available);
}

static void test_gcp_dispatch_excludes_preflight() {
    CHECK(!server_http_gcp_predict_dispatch_allowed("/cache/plan"));
    CHECK(server_http_gcp_predict_dispatch_allowed("/completion"));
    CHECK(server_http_gcp_predict_dispatch_allowed("cachePlan"));
}

static void test_as_if_completion_semantics() {
    const auto native = server_cache_plan_preflight_semantics_for(
        false, true, true, true, true);
    const auto literal_preflight = server_cache_plan_preflight_semantics_for(
        false, false, true, true, true);
    const auto as_if_preflight = server_cache_plan_preflight_semantics_for(
        true, false, true, true, true);
    CHECK(native.completion_semantics);
    CHECK(native.host_lookup_enabled);
    CHECK(!literal_preflight.completion_semantics);
    CHECK(!literal_preflight.host_lookup_enabled);
    CHECK(as_if_preflight.completion_semantics ==
          native.completion_semantics);
    CHECK(as_if_preflight.host_lookup_enabled == native.host_lookup_enabled);
}

static void test_local_source_registry() {
    server_cache_plan_local_source_registry registry;
    int32_t first = -1;
    int32_t second = -1;
    int32_t repeated = -1;
    CHECK(registry.get_or_assign(0x1000, first));
    CHECK(registry.get_or_assign(0x2000, second));
    CHECK(registry.get_or_assign(0x1000, repeated));
    CHECK(first == 0);
    CHECK(second == 1);
    CHECK(repeated == first);
    CHECK(registry.size() == 2);
    int32_t found = -1;
    CHECK(registry.find(0x2000, found));
    CHECK(found == second);
    CHECK(!registry.find(0x3000, found));
    CHECK(found == -1);

    int32_t publish_next = 0;
    int32_t published_a = -1;
    int32_t published_b = -1;
    int32_t publish_a = -1;
    int32_t publish_b = -1;
    CHECK(server_cache_plan_assign_source_id(
        published_a, publish_next, publish_a));
    CHECK(server_cache_plan_assign_source_id(
        published_b, publish_next, publish_b));
    CHECK(publish_a == first);
    CHECK(publish_b == second);
}

static void assert_redacted_keys(const nlohmann::ordered_json & value) {
    // Canonical exhaustive  private-key oracle. The contract scan and live
    // driver carry deliberate security-critical subsets and point back here.
    static const std::set<std::string> forbidden = {
        "target_slot_id", "source_id", "candidate_id", "component_ids",
        "checkpoint_ordinal", "artifact_id", "victim_ids",
        "recovery_source", "manifest_digest", "effect_digest",
        "recovery_digest", "accounting_serial", "admission_sequence",
        "memo_key", "lease_holder", "lease_scope", "lease_expiry",
        "main_family", "device_ordinal", "topology_id", "domains",
        "journal_id", "re_request",
    };
    if (value.is_object()) {
        for (const auto & item : value.items()) {
            CHECK(forbidden.count(item.key()) == 0);
            assert_redacted_keys(item.value());
        }
    } else if (value.is_array()) {
        for (const auto & item : value) {
            assert_redacted_keys(item);
        }
    }
}

static void test_wire_serializer_and_golden() {
    auto rec = live_record();
    for (int source : {91, 92}) {
        auto * rejected = rec.find_or_add(common_cache_plan_provider::host_cache_entry, source,
            COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 7, common_cache_plan_selection::similarity);
        CHECK(rejected);
        rejected->note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
    }
    const auto before = common_cache_plan_record_json(rec).dump();
    server_cache_plan_preflight_view view;
    CHECK(server_cache_plan_preflight_build_view(rec, 7, view));
    CHECK(before == common_cache_plan_record_json(rec).dump());
    const auto wire = server_cache_plan_preflight_json(view);
    CHECK(wire["object"] == "cache_plan_preflight");
    CHECK(wire["schema_version"] == 2);
    CHECK(wire["cache_plan_schema_version"] == COMMON_CACHE_PLAN_SCHEMA_VERSION);
    CHECK(wire["authoritative"] == false);
    CHECK(wire["reservation"] == "none");
    CHECK(wire["valid_until"].is_null());
    CHECK(!wire.contains("planner"));
    CHECK(wire["selection"]["reuse_tokens"] == 96);
    CHECK(wire["selection"]["replay_tokens"] == 4);
    CHECK(!wire["selection"].contains("predicted_ttft_us"));
    CHECK(!wire["selection"].contains("cost_terms"));
    CHECK(!wire.contains("destruction"));
    CHECK(wire["miss_reasons"].size() == 1);
    CHECK(wire["miss_reasons"][0]["count"] == 2);
    assert_redacted_keys(wire);
    const std::string encoded = wire.dump(2) + "\n";
    if (std::getenv("CACHE_PLAN_PRINT_PREFLIGHT_GOLDEN")) {
        std::fputs(encoded.c_str(), stdout);
        std::exit(EXIT_SUCCESS);
    }
#ifdef CACHE_PLAN_PREFLIGHT_GOLDEN_PATH
    std::ifstream golden(CACHE_PLAN_PREFLIGHT_GOLDEN_PATH);
    CHECK(golden.good());
    std::ostringstream expected;
    expected << golden.rdbuf();
    CHECK(encoded == expected.str());
#endif
}

static void test_exposure_gate() {
    CHECK(server_cache_plan_preflight_exposure_allowed("127.0.0.1", 0));
    CHECK(server_cache_plan_preflight_exposure_allowed("localhost", 1));
    CHECK(server_cache_plan_preflight_exposure_allowed("::1", 0));
    CHECK(server_cache_plan_preflight_exposure_allowed(
        "/tmp/llama.sock", 1));
    CHECK(!server_cache_plan_preflight_exposure_allowed("0.0.0.0", 0));
    CHECK(!server_cache_plan_preflight_exposure_allowed("127.0.0.1", 2));

    CHECK(server_cache_plan_preflight_request_field_allowed("prompt"));
    CHECK(server_cache_plan_preflight_request_field_allowed("id_slot"));
    CHECK(server_cache_plan_preflight_request_field_allowed("cache_prompt"));
    CHECK(server_cache_plan_preflight_request_field_allowed("lora"));
    CHECK(server_cache_plan_preflight_request_field_allowed(
        "message_delimiters"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("sampling"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("ticket"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("claim"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("preview_id"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("nonce"));
    CHECK(!server_cache_plan_preflight_request_field_allowed(
        "manifest_digest"));
    CHECK(!server_cache_plan_preflight_request_field_allowed("artifact_id"));
}

int main() {
    test_snapshot_selection_and_missing_evidence();
    test_as_if_completion_semantics();
    test_local_source_registry();
    test_wire_serializer_and_golden();
    test_exposure_gate();
    test_gcp_dispatch_excludes_preflight();
    std::puts("test-cache-plan-preflight: PASS");
    return 0;
}
