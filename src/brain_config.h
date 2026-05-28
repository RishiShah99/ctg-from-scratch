#pragma once

// Brain-mode locked configuration. Mirrors the CLI invocation that
// produced results/osdn-brain-ohio-clamped.json (and therefore the
// in-tree weights blob and norm-stats header). Consumed by
// tools/extract_norm_stats so the data pipeline's pinned flags live
// in exactly one place rather than being restated in each tool.
//
// The trainer (src/cgm_train.cpp) intentionally does NOT consume this
// header: its CLI defaults are generic (lookback=96, step_min=15,
// window_stride=1, no patient-fold pins) and brain-mode is selected
// per-invocation via flags. Adopting brain-mode defaults in the
// trainer would silently change the behavior of every non-brain
// training run. The residual: a re-train of the brain-mode model
// must keep its CLI flags in sync with this header by convention.

#include <cstddef>
#include <cstdint>

namespace brain_config {

inline constexpr int    LOOKBACK_STEPS = 144;
inline constexpr int    HORIZON_STEPS  = 12;
inline constexpr int    STEP_MIN       = 5;
inline constexpr double HYPO_THRESHOLD = 70.0;
inline constexpr int    POST_BOLUS_MIN = 240;
inline constexpr int    WINDOW_STRIDE  = 12;
inline constexpr std::uint32_t SEED    = 42;

inline constexpr const char* VAL_IDS[]  = {"584", "588"};
inline constexpr const char* TEST_IDS[] = {"591", "596"};
inline constexpr std::size_t N_VAL_IDS  = sizeof(VAL_IDS)  / sizeof(VAL_IDS[0]);
inline constexpr std::size_t N_TEST_IDS = sizeof(TEST_IDS) / sizeof(TEST_IDS[0]);

}  // namespace brain_config
