#pragma once
#include <cstdint>
#include <cstddef>

namespace osdn_weights {

// Compile-time constants the firmware uses outside the OSDN kernel.
// LOOKBACK pinned to the trainer's --lookback. Norm stats lifted from
// results/osdn-brain-ohio-clamped.json (norm_mean / norm_sd fields,
// produced by Step 6 Tier-1 clamped re-train).
static constexpr int   LOOKBACK   = 144;
static constexpr float MEAN_MG_DL = 157.6580f;
static constexpr float STD_MG_DL  =  60.4485f;

// IOB/COB per-feature norm stats. The trainer applies these in
// src/cgm_data.cpp:281-300 (normalize_inplace) to the IOB/COB channels
// before training but never persists them to JSON or blob. Without the
// firmware applying the same z-score, embed channels 5/6 see a
// distribution the model was never trained on. Re-extracted via
// tools/extract_norm_stats.exe against the same brain-mode flags
// (results/norm_stats.json); any re-train must regenerate these.
static constexpr float IOB_MEAN   = 0.6985540f;
static constexpr float IOB_STD    = 0.8231367f;
static constexpr float COB_MEAN   = 0.2245205f;
static constexpr float COB_STD    = 0.4076972f;

// Runtime shape constants emitted by blob_to_header from the blob's
// 32-byte osdn_blob header. Firmware uses these (not compile-time
// duplicates) so the constants and the blob can't disagree silently.
extern const std::uint32_t BLOB_H;
extern const std::uint32_t BLOB_K;
extern const std::uint32_t BLOB_D_IN;
extern const std::uint32_t BLOB_N_LAYERS;
extern const std::size_t   BLOB_LEN;
extern const float         BLOB[];

}
