// Host-side cross-check of the ESP32 firmware feature pipeline against a
// double-precision reference that mirrors src/cgm_data.cpp:184–215 and
// src/cgm_net.hpp:107–111. Links esp32/src/features.cpp directly (the
// module is pure C++ stdlib, no Arduino dependency).
//
// Failure here means the firmware feature pipeline drifted from the
// trainer's algorithm — catches IOB/COB constant typos, ring-walk index
// errors, sign-of-time mistakes, and Δg/Δ²g leading-edge wrong-zero
// before any of it reaches the ESP32 hardware.
//
// Tolerance: 1e-4 per channel. Drift sources are float-vs-double
// accumulators and libm sin/cos/exp precision; experimentally the worst
// channel is iob at the steepest decay step, around 5e-6.
//
#include "../esp32/src/features.h"
#include "../esp32/src/osdn_weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// Double-precision reference algorithm. Mirrors src/cgm_data.cpp's
// IOB/COB and tod loops and src/cgm_net.hpp's Δg/Δ²g exactly, only
// substituting `double` for the firmware's `float`.
void reference_window(const double* g_ring_dbl,
                      int           g_head,
                      int           g_count,
                      int           L,
                      const features::Event* iob_ring, int iob_count,
                      const features::Event* meal_ring, int meal_count,
                      int           step_min,
                      std::uint32_t t_now_min,
                      double*       out_LxD) {
    const double FEAT_TWO_PI     = 6.283185307179586;
    const double TAU_IOB         = 120.0;
    const double TAU_COB         =  90.0;
    const double IOB_SCALE       =   5.0;
    const double COB_SCALE       =  50.0;
    const double IOB_HORIZON_MIN = 600.0;
    const double COB_HORIZON_MIN = 480.0;

    auto lookback_at = [&](int t_lb) {
        const int start = (g_head - g_count + L) % L;
        const int idx   = (start + t_lb) % L;
        return g_ring_dbl[idx];
    };

    for (int t_lb = 0; t_lb < L; ++t_lb) {
        const std::int64_t step_t =
            static_cast<std::int64_t>(t_now_min) -
            static_cast<std::int64_t>(L - 1 - t_lb) *
                static_cast<std::int64_t>(step_min);

        const double g_t = lookback_at(t_lb);
        double dg = 0.0, d2g = 0.0;
        if (t_lb >= 1) {
            const double g_prev = lookback_at(t_lb - 1);
            dg = g_t - g_prev;
            if (t_lb >= 2) {
                const double g_prev2 = lookback_at(t_lb - 2);
                const double dg_prev = g_prev - g_prev2;
                d2g = dg - dg_prev;
            }
        }

        const std::int64_t tod  = ((step_t % 1440) + 1440) % 1440;
        const double tod_frac    = static_cast<double>(tod) / 1440.0;
        const double sin_tod     = std::sin(FEAT_TWO_PI * tod_frac);
        const double cos_tod     = std::cos(FEAT_TWO_PI * tod_frac);

        double iob_acc = 0.0;
        for (int j = 0; j < iob_count; ++j) {
            if (static_cast<std::int64_t>(iob_ring[j].t_min) > step_t) continue;
            const double dt = static_cast<double>(
                step_t - static_cast<std::int64_t>(iob_ring[j].t_min));
            if (dt > IOB_HORIZON_MIN) continue;
            iob_acc += static_cast<double>(iob_ring[j].amount) *
                       std::exp(-dt / TAU_IOB);
        }
        // Mirror the firmware's post-/SCALE z-score (esp32/src/features.cpp
        // — see osdn_weights::IOB_MEAN/STD). Trainer source:
        // src/cgm_data.cpp:296. Use the same fp32-rounded constants the
        // firmware uses, cast to double, so the only intentional drift
        // between the two is float-vs-double accumulator precision.
        const double iob = (iob_acc / IOB_SCALE -
                            static_cast<double>(osdn_weights::IOB_MEAN)) /
                           static_cast<double>(osdn_weights::IOB_STD);

        double cob_acc = 0.0;
        for (int j = 0; j < meal_count; ++j) {
            if (static_cast<std::int64_t>(meal_ring[j].t_min) > step_t) continue;
            const double dt = static_cast<double>(
                step_t - static_cast<std::int64_t>(meal_ring[j].t_min));
            if (dt > COB_HORIZON_MIN) continue;
            cob_acc += static_cast<double>(meal_ring[j].amount) *
                       std::exp(-dt / TAU_COB);
        }
        // Mirror src/cgm_data.cpp:297 — trainer z-scores COB the same way.
        const double cob = (cob_acc / COB_SCALE -
                            static_cast<double>(osdn_weights::COB_MEAN)) /
                           static_cast<double>(osdn_weights::COB_STD);

        out_LxD[t_lb * 7 + 0] = g_t;
        out_LxD[t_lb * 7 + 1] = dg;
        out_LxD[t_lb * 7 + 2] = d2g;
        out_LxD[t_lb * 7 + 3] = sin_tod;
        out_LxD[t_lb * 7 + 4] = cos_tod;
        out_LxD[t_lb * 7 + 5] = iob;
        out_LxD[t_lb * 7 + 6] = cob;
    }
}

}  // namespace

int main() {
    constexpr int L    = features::LOOKBACK_STEPS;   // 144
    constexpr int STEP = features::STEP_MIN;         // 5
    const std::uint32_t t_now = 5000;                // well past (L-1)*STEP = 715

    // CGM lookback ring populated as a monotone z-scored sequence.
    // head=0, count=L means the slot at index 0 is the oldest sample.
    static float  g_ring_f[L];
    static double g_ring_d[L];
    for (int i = 0; i < L; ++i) {
        const double v = -0.5 + 0.05 * static_cast<double>(i);
        g_ring_f[i] = static_cast<float>(v);
        g_ring_d[i] = v;
    }
    const int g_head  = 0;
    const int g_count = L;

    // Synthetic events, chosen to exercise:
    //   - boluses both inside and outside the IOB horizon (oldest at
    //     t_min=4000 has dt=1000 at t_now=5000 → outside 600-min horizon
    //     for the last lookback step; inside for the very first step
    //     at step_t=4285 with dt=285).
    //   - meals likewise distributed.
    //   - mix of round and non-round amounts.
    features::Event iob[5] = {
        { 4000u, 2.0f },
        { 4200u, 1.5f },
        { 4400u, 3.5f },
        { 4500u, 4.0f },
        { 4900u, 2.5f },
    };
    const int iob_count = 5;

    features::Event meal[3] = {
        { 4100u, 20.0f },
        { 4500u, 50.0f },
        { 4850u, 30.0f },
    };
    const int meal_count = 3;

    static float  out_f[L * 7];
    static double out_d[L * 7];

    features::compute_window(g_ring_f, g_head, g_count,
                             iob,  iob_count,
                             meal, meal_count,
                             STEP, L, t_now, out_f);
    reference_window(g_ring_d, g_head, g_count, L,
                     iob,  iob_count,
                     meal, meal_count,
                     STEP, t_now, out_d);

    const double tol = 1e-4;
    double max_diff = 0.0;
    int    max_t = -1, max_c = -1;
    double max_diff_per_ch[7] = {0};
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < 7; ++c) {
            const double d = std::abs(
                static_cast<double>(out_f[t * 7 + c]) - out_d[t * 7 + c]);
            if (d > max_diff_per_ch[c]) max_diff_per_ch[c] = d;
            if (d > max_diff) { max_diff = d; max_t = t; max_c = c; }
        }
    }

    std::printf("features_host_test: L=%d, IOB=%d, meals=%d, t_now=%u\n",
                L, iob_count, meal_count,
                static_cast<unsigned>(t_now));
    static const char* ch_names[7] = {"g","dg","d2g","sin_tod","cos_tod","iob","cob"};
    for (int c = 0; c < 7; ++c) {
        std::printf("  ch[%d]=%-8s max |diff|=%.3e\n",
                    c, ch_names[c], max_diff_per_ch[c]);
    }
    std::printf("  overall max |diff|=%.3e at t_lb=%d, channel=%d (%s)  tol=%.0e\n",
                max_diff, max_t, max_c, ch_names[max_c], tol);

    if (max_diff > tol) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
