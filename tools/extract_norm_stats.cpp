// Re-runs the brain-mode data pipeline against data/ohio_t1dm.csv and
// dumps the six normalization stats that normalize_inplace
// (src/cgm_data.cpp:266-301) computes. The four IOB/COB stats are the
// ones the trainer applies but never persists — see the gap identified
// in results/step7_review.md (CRITICAL: IOB/COB feature channels are
// not normalized in the firmware, but the trained model expects them
// normalized).
//
// Outputs:
//   results/norm_stats.json           — full-precision audit record
//   esp32/src/osdn_norm_stats.gen.h   — generated header consumed by
//                                       the firmware (single source-
//                                       of-truth for the six floats).
//
// Flags are pinned to the shipped osdn-brain-ohio-clamped.json
// configuration so the output is reproducible without CLI arguments.
// Re-train of the brain-mode model with different flags must re-run
// this tool; the gen header then updates in lock-step.

#include "../src/brain_config.h"
#include "../src/cgm_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Data pipeline + fold config is sourced from src/brain_config.h so the
// brain-mode locked values live in one place.
using brain_config::LOOKBACK_STEPS;
using brain_config::HORIZON_STEPS;
using brain_config::STEP_MIN;
using brain_config::HYPO_THRESHOLD;
using brain_config::POST_BOLUS_MIN;
using brain_config::WINDOW_STRIDE;
using brain_config::SEED;
using brain_config::VAL_IDS;
using brain_config::TEST_IDS;

constexpr const char* CSV_PATH   = "data/ohio_t1dm.csv";
constexpr const char* OUT_JSON   = "results/norm_stats.json";
constexpr const char* OUT_HEADER = "esp32/src/osdn_norm_stats.gen.h";

}  // namespace

int main() {
    std::cout << "extract_norm_stats: replaying brain-mode pipeline\n"
              << "  csv=" << CSV_PATH << "\n"
              << "  lookback=" << LOOKBACK_STEPS
              << " horizon=" << HORIZON_STEPS
              << " step_min=" << STEP_MIN
              << " hypo=" << HYPO_THRESHOLD
              << " post_bolus=" << POST_BOLUS_MIN
              << " window_stride=" << WINDOW_STRIDE
              << " seed=" << SEED << "\n"
              << "  val_ids=584,588  test_ids=591,596\n";
    std::cout.flush();

    std::vector<CGMRecord> records;
    try {
        records = load_cgm_csv(CSV_PATH);
    } catch (const std::exception& e) {
        std::cerr << "load_cgm_csv failed: " << e.what() << "\n";
        return 1;
    }
    if (records.empty()) {
        std::cerr << "no patients parsed from " << CSV_PATH << "\n";
        return 1;
    }
    std::cout << "  patients=" << records.size() << "\n";

    PatientSplitPolicy policy;
    for (const char* p : VAL_IDS)  policy.val_ids.emplace_back(p);
    for (const char* p : TEST_IDS) policy.test_ids.emplace_back(p);

    CGMDataset ds;
    try {
        ds = make_windows(records, LOOKBACK_STEPS, HORIZON_STEPS, STEP_MIN,
                          HYPO_THRESHOLD, POST_BOLUS_MIN, policy, SEED,
                          WINDOW_STRIDE);
    } catch (const std::exception& e) {
        std::cerr << "make_windows failed: " << e.what() << "\n";
        return 2;
    }
    std::cout << "  windows: train=" << ds.train.size()
              << " val=" << ds.val.size()
              << " test=" << ds.test.size() << "\n";
    std::cout.flush();

    NormStats stats{};
    normalize_inplace(ds, stats);

    std::printf("\nNorm stats (post-normalize_inplace):\n");
    std::printf("  cgm_mean = %.10f\n", stats.cgm_mean);
    std::printf("  cgm_std  = %.10f\n", stats.cgm_std);
    std::printf("  iob_mean = %.10f\n", stats.iob_mean);
    std::printf("  iob_std  = %.10f\n", stats.iob_std);
    std::printf("  cob_mean = %.10f\n", stats.cob_mean);
    std::printf("  cob_std  = %.10f\n", stats.cob_std);
    std::fflush(stdout);

    // Use C FILE* not std::ofstream — MinGW's ofstream destructor
    // interacts badly with -Wl,--stack,268435456 on this toolchain and
    // segfaults on close. C stdio is fine.
    FILE* fh = std::fopen(OUT_JSON, "w");
    if (!fh) {
        std::cerr << "could not open " << OUT_JSON << " for write\n";
        return 3;
    }
    std::fprintf(fh,
        "{\n"
        "  \"csv\": \"%s\",\n"
        "  \"lookback\": %d,\n"
        "  \"horizon\": %d,\n"
        "  \"step_min\": %d,\n"
        "  \"hypo\": %.4f,\n"
        "  \"post_bolus\": %d,\n"
        "  \"window_stride\": %d,\n"
        "  \"seed\": %u,\n"
        "  \"val_ids\": \"584,588\",\n"
        "  \"test_ids\": \"591,596\",\n"
        "  \"cgm_mean\": %.10f,\n"
        "  \"cgm_std\":  %.10f,\n"
        "  \"iob_mean\": %.10f,\n"
        "  \"iob_std\":  %.10f,\n"
        "  \"cob_mean\": %.10f,\n"
        "  \"cob_std\":  %.10f\n"
        "}\n",
        CSV_PATH, LOOKBACK_STEPS, HORIZON_STEPS, STEP_MIN,
        HYPO_THRESHOLD, POST_BOLUS_MIN, WINDOW_STRIDE,
        static_cast<unsigned>(SEED),
        stats.cgm_mean, stats.cgm_std,
        stats.iob_mean, stats.iob_std,
        stats.cob_mean, stats.cob_std);
    std::fclose(fh);
    std::cout << "\nwrote " << OUT_JSON << "\n";

    // Generated header consumed by the firmware via
    // esp32/src/osdn_weights.h. Precision is fixed at %.4f for the CGM
    // stats and %.7f for IOB/COB to preserve the float32 bit patterns
    // shipped in the locked weights blob; emitting at higher precision
    // would shift the CGM constants by ~2.6 ULP and break bit-identity
    // with results/osdn-brain-ohio-clamped.weights.bin.best.
    FILE* hh = std::fopen(OUT_HEADER, "w");
    if (!hh) {
        std::cerr << "could not open " << OUT_HEADER << " for write\n";
        return 4;
    }
    std::fprintf(hh,
        "// Auto-generated by tools/extract_norm_stats from %s.\n"
        "// DO NOT EDIT. Re-run extract_norm_stats to regenerate.\n"
        "//\n"
        "// Precision rationale (see results/known_limitations.md): CGM stats emit at\n"
        "// %%.4f and IOB/COB at %%.7f to preserve the currently-shipped float32 bit\n"
        "// patterns. Emitting at higher precision would alter the CGM constants by\n"
        "// ~2.6 ULP and break bit-identity with the in-tree weights blob.\n"
        "\n"
        "#pragma once\n"
        "\n"
        "namespace osdn_weights {\n"
        "\n"
        "static constexpr float MEAN_MG_DL = %.4ff;\n"
        "static constexpr float STD_MG_DL  = %9.4ff;\n"
        "\n"
        "static constexpr float IOB_MEAN   = %.7ff;\n"
        "static constexpr float IOB_STD    = %.7ff;\n"
        "static constexpr float COB_MEAN   = %.7ff;\n"
        "static constexpr float COB_STD    = %.7ff;\n"
        "\n"
        "}  // namespace osdn_weights\n",
        OUT_JSON,
        stats.cgm_mean, stats.cgm_std,
        stats.iob_mean, stats.iob_std,
        stats.cob_mean, stats.cob_std);
    std::fclose(hh);
    std::cout << "wrote " << OUT_HEADER << "\n";
    return 0;
}
