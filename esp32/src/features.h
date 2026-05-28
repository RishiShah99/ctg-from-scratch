#pragma once
//
// 7-channel feature builder for the on-device OSDN inference path.
//
// Builds the [L * 7] row-major feature matrix consumed by
// osdn_inf::forward, where each row is:
//
//   [g_z, Δg_z, Δ²g_z, sin(tod), cos(tod), iob, cob]
//
// Constants and formulas are byte-identical to the host trainer's
// feature extraction (src/cgm_data.cpp:184–215 for IOB/COB and time-of-
// day; src/cgm_net.hpp:107–111 for Δg/Δ²g and channel order). Any edit
// here must also update those files (and the host-side cross-check in
// src/features_host_test.cpp).
//
// Two ring buffers track bolus and meal events (held in main.cpp); the
// CGM lookback is a third ring (also in main.cpp), already z-scored
// against the trainer's norm_mean / norm_sd before append.
//
#include <cstddef>
#include <cstdint>

namespace features {

// Paper / dataset constants — must match src/cgm_data.cpp.
constexpr double FEAT_TWO_PI     = 6.283185307179586;
constexpr float  TAU_IOB         = 120.0f;   // minutes
constexpr float  TAU_COB         =  90.0f;   // minutes
constexpr float  IOB_SCALE       =   5.0f;
constexpr float  COB_SCALE       =  50.0f;
constexpr float  IOB_HORIZON_MIN = 600.0f;
constexpr float  COB_HORIZON_MIN = 480.0f;

// Shipped Ohio brain-mode config — must match osdn-brain-ohio-clamped.json
// lookback / step_min fields and esp32/src/osdn_weights.h LOOKBACK.
constexpr int    STEP_MIN        =   5;
constexpr int    LOOKBACK_STEPS  = 144;

// Bolus / meal event held in the ring. `t_min` is the device-clock
// minute at which the event occurred; `amount` is units (insulin) for
// bolus events or grams (carbs) for meal events.
struct Event {
    std::uint32_t t_min;
    float         amount;
};

// Circular append to a ring of size N. Updates head and count in place.
// Returns true if the append fit cleanly, false if it overwrote the
// oldest entry (silently — caller may log if desired). Overwrite is
// behaviorally correct under chronological ordering: the dropped entry
// is the oldest and so the smallest exp(-dt/TAU) contributor to the
// current IOB/COB sum.
bool ring_append(Event* ring, int& head, int& count, int N,
                 std::uint32_t t_min, float amount);

// Build the full [lookback_steps * 7] feature matrix from current ring
// state. out_LxD is row-major: out_LxD[t * 7 + c] for channel c at
// lookback step t. Caller in main.cpp must ensure the precondition
// t_now_min >= (lookback_steps - 1) * step_minutes — i.e. the lookback
// window is fully populated — before invoking. compute_window asserts
// the invariant; violating it underflows the unsigned step_t arithmetic.
//
// `g_norm_ring` is the LOOKBACK_STEPS-element ring of *already-z-scored*
// CGM values (the firmware z-scores in push_glucose before append).
// `g_head` and `g_count` follow main.cpp's ring conventions (head points
// to next write slot; count is min(samples_seen, LOOKBACK_STEPS)).
//
// Walks the IOB/COB rings in insertion order (no early break), since
// the host may have inserted events out of strict chronological order
// (see overflow-behavior commentary in results/step7_plan.md §3).
void compute_window(const float*       g_norm_ring,
                    int                g_head,
                    int                g_count,
                    const Event*       iob_ring,
                    int                iob_count,
                    const Event*       meal_ring,
                    int                meal_count,
                    int                step_minutes,
                    int                lookback_steps,
                    std::uint32_t      t_now_min,
                    float*             out_LxD);

}  // namespace features
