// Re-runs the brain-mode data pipeline against data/ohio_t1dm.csv and
// dumps the six normalization stats that normalize_inplace
// (src/cgm_data.cpp:266-301) computes. The four IOB/COB stats are the
// ones the trainer applies but never persists — see the gap identified
// in results/step7_review.md (CRITICAL: IOB/COB feature channels are
// not normalized in the firmware, but the trained model expects them
// normalized). The firmware reads the printed values from
// esp32/src/osdn_weights.h.
//
// Flags are pinned to the shipped osdn-brain-ohio-clamped.json
// configuration so the output is reproducible without CLI arguments.
// Re-train of the brain-mode model with different flags must update
// this tool's pinned values AND re-embed the printed constants.

#include "../src/cgm_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* CSV_PATH         = "data/ohio_t1dm.csv";
constexpr int         LOOKBACK_STEPS   = 144;
constexpr int         HORIZON_STEPS    = 12;
constexpr int         STEP_MIN         = 5;
constexpr double      HYPO_THRESHOLD   = 70.0;
constexpr int         POST_BOLUS_MIN   = 240;
constexpr int         WINDOW_STRIDE    = 12;
constexpr std::uint32_t SEED           = 42;
constexpr const char* VAL_IDS[]        = {"584", "588"};
constexpr const char* TEST_IDS[]       = {"591", "596"};
constexpr const char* OUT_JSON         = "results/norm_stats.json";

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
    return 0;
}
