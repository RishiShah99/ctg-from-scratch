#include "features.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace features {

namespace {

// Internal helper. Walks the CGM lookback ring in chronological order and
// returns the z-scored glucose value at lookback index t_lb in [0, L).
// Index 0 is the oldest sample currently in the window (received
// (L - 1) * STEP_MIN minutes before t_now_min), index L-1 is newest.
inline float lookback_at(const float* g_ring, int g_head, int g_count,
                         int lookback_steps, int t_lb) {
    // Oldest valid sample lives g_count steps back from g_head, wrapped.
    int start = (g_head - g_count + lookback_steps) % lookback_steps;
    int idx   = (start + t_lb) % lookback_steps;
    return g_ring[idx];
}

// One row of the [L*7] matrix. step_t is the int64_t-signed historic
// timestamp at this lookback position; see compute_window for derivation.
inline void compute_row(const float* g_ring, int g_head, int g_count,
                        const Event* iob_ring, int iob_count,
                        const Event* meal_ring, int meal_count,
                        int          lookback_steps,
                        std::int64_t step_t,
                        int          t_lb,
                        float*       out_row) {
    const float g_t   = lookback_at(g_ring, g_head, g_count, lookback_steps, t_lb);

    // Δg / Δ²g — leading edge of the lookback yields zero, matching the
    // trainer at cgm_net.hpp:107–111. NOTE: t_lb == 0 here means "oldest
    // sample currently in the window," NOT "first sample ever received."
    // The trainer's zero is the same semantic — a freshly-built window
    // has no prior to its leading edge.
    float dg  = 0.0f;
    float d2g = 0.0f;
    if (t_lb >= 1) {
        const float g_prev =
            lookback_at(g_ring, g_head, g_count, lookback_steps, t_lb - 1);
        dg = g_t - g_prev;
        if (t_lb >= 2) {
            const float g_prev2 =
                lookback_at(g_ring, g_head, g_count, lookback_steps, t_lb - 2);
            const float dg_prev = g_prev - g_prev2;
            d2g = dg - dg_prev;
        }
    }

    // Time-of-day on a 24h cycle. The trainer uses signed int64_t
    // arithmetic at cgm_data.cpp:191 because its mod-1440 idiom only
    // produces correct results for negative dividends when the type is
    // signed; we mirror that.
    const std::int64_t tod  = ((step_t % 1440) + 1440) % 1440;
    const float tod_frac    = static_cast<float>(tod) / 1440.0f;
    const float sin_tod     = std::sin(static_cast<float>(FEAT_TWO_PI) * tod_frac);
    const float cos_tod     = std::cos(static_cast<float>(FEAT_TWO_PI) * tod_frac);

    // IOB / COB exponential-decay sums. Insertion-order walk, no early
    // break (ring is typically chronological but not guaranteed; see
    // §3 of results/step7_plan.md). Filter by horizon and future-event.
    float iob_acc = 0.0f;
    for (int j = 0; j < iob_count; ++j) {
        const Event& e = iob_ring[j];
        if (static_cast<std::int64_t>(e.t_min) > step_t) continue;
        const float dt = static_cast<float>(step_t - static_cast<std::int64_t>(e.t_min));
        if (dt > IOB_HORIZON_MIN) continue;
        iob_acc += e.amount * std::exp(-dt / TAU_IOB);
    }
    const float iob = iob_acc / IOB_SCALE;

    float cob_acc = 0.0f;
    for (int j = 0; j < meal_count; ++j) {
        const Event& e = meal_ring[j];
        if (static_cast<std::int64_t>(e.t_min) > step_t) continue;
        const float dt = static_cast<float>(step_t - static_cast<std::int64_t>(e.t_min));
        if (dt > COB_HORIZON_MIN) continue;
        cob_acc += e.amount * std::exp(-dt / TAU_COB);
    }
    const float cob = cob_acc / COB_SCALE;

    out_row[0] = g_t;
    out_row[1] = dg;
    out_row[2] = d2g;
    out_row[3] = sin_tod;
    out_row[4] = cos_tod;
    out_row[5] = iob;
    out_row[6] = cob;
}

}  // namespace

bool ring_append(Event* ring, int& head, int& count, int N,
                 std::uint32_t t_min, float amount) {
    const bool overwrote = (count == N);
    ring[head].t_min  = t_min;
    ring[head].amount = amount;
    head = (head + 1) % N;
    if (count < N) ++count;
    return !overwrote;
}

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
                    float*             out_LxD) {
    // Precondition: the lookback window is fully populated. Without this,
    // the historic-step subtraction below would underflow an unsigned
    // intermediate on the path back to the oldest sample. Caller in
    // main.cpp gates on g_ring_count >= LOOKBACK before invoking.
    assert(g_count == lookback_steps);
    assert(t_now_min >=
           static_cast<std::uint32_t>((lookback_steps - 1) * step_minutes));

    for (int t_lb = 0; t_lb < lookback_steps; ++t_lb) {
        // Historic timestamp at this lookback position. Cast through
        // int64_t (mirroring src/cgm_data.cpp:191) so the modulo-1440
        // idiom below produces correct results even if a future caller
        // somehow gets here with t_now_min smaller than expected.
        const std::int64_t step_t =
            static_cast<std::int64_t>(t_now_min) -
            static_cast<std::int64_t>(lookback_steps - 1 - t_lb) *
                static_cast<std::int64_t>(step_minutes);

        compute_row(g_norm_ring, g_head, g_count,
                    iob_ring,    iob_count,
                    meal_ring,   meal_count,
                    lookback_steps,
                    step_t,
                    t_lb,
                    out_LxD + t_lb * 7);
    }
}

}  // namespace features
