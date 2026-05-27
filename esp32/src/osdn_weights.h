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
