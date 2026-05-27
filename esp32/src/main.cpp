#include <Arduino.h>
#include "osdn_inference.h"
#include "osdn_weights.h"
#include "features.h"

using osdn_inf::H_MAX;
using osdn_inf::K_MAX;

static osdn_inf::Weights g_weights;
static osdn_inf::State   g_state;

// CGM lookback ring (z-scored mg/dL). Sized to LOOKBACK = 144 samples.
static float g_ring[osdn_weights::LOOKBACK];
static int   g_ring_head  = 0;
static int   g_ring_count = 0;

// Event rings. Sizes per results/step7_plan.md §3:
//   - IOB ring at 128 entries covers the 600-min horizon with ≥6× headroom
//     against Ohio T1DM worst-case bolus density (~20 events/10h).
//   - COB ring at 96 entries covers the 480-min horizon with ≥30× headroom
//     against ≤8 meals/8h.
// RAM cost: (128 + 96) * 8 = 1.75 KB; negligible against 512 KB SRAM.
// Overwrite policy: circular, silent. Correctness argument in plan §3.
static constexpr int IOB_RING_N  = 128;
static constexpr int MEAL_RING_N = 96;
static features::Event g_iob[IOB_RING_N];
static int g_iob_head  = 0;
static int g_iob_count = 0;
static features::Event g_meal[MEAL_RING_N];
static int g_meal_head  = 0;
static int g_meal_count = 0;

// Device clock in minutes. Advances on `t <…>` lines from the host; the
// replay tool pins this explicitly. Live deploy can derive from
// millis() / 60000 once that path lands (out of scope for step 7).
static std::uint32_t g_now_min = 0;

static constexpr int LED_PIN = 48;

// Alarm threshold in logit space. Derived from osdn-brain-ohio-clamped.json
// test_thr_spec80.thr = 0.14 (the val-selected probability threshold that
// achieves 80% specificity on the held-out val patients per the Step 8
// thr-on-val fix). Logit form: logf(0.14 / 0.86) ≈ -1.8153.
//
// At this threshold the trained model achieves val recall 0.7182 and val
// specificity 0.6771 — a deploy-honest predictive setpoint. Defaulting to
// 0.0f (prob > 0.5) was rejected as too conservative for a predictive alarm.
static constexpr float HYPO_LOGIT_THRESHOLD = -1.8153f;

static float normalize(float mg_dl) {
    return (mg_dl - osdn_weights::MEAN_MG_DL) / osdn_weights::STD_MG_DL;
}

static void push_reading(float mg_dl) {
    g_ring[g_ring_head] = normalize(mg_dl);
    g_ring_head = (g_ring_head + 1) % osdn_weights::LOOKBACK;
    if (g_ring_count < osdn_weights::LOOKBACK) g_ring_count++;
}

// Static scratch buffer for the [L * 7] feature matrix. Lives in .bss
// (LOOKBACK * 7 * 4 = 4032 B); no per-call allocation.
static float g_feat[osdn_weights::LOOKBACK * 7];

void setup() {
    Serial.begin(115200);
    delay(500);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    bool ok = osdn_inf::load_blob(
        g_weights,
        static_cast<int>(osdn_weights::BLOB_H),
        static_cast<int>(osdn_weights::BLOB_K),
        static_cast<int>(osdn_weights::BLOB_D_IN),
        static_cast<int>(osdn_weights::BLOB_N_LAYERS),
        osdn_weights::BLOB,
        osdn_weights::BLOB_LEN);

    Serial.printf("OSDN load: %s  H=%u K=%u D_in=%u n_layers=%u L=%d  params=%u\n",
                  ok ? "ok" : "FAIL",
                  static_cast<unsigned>(osdn_weights::BLOB_H),
                  static_cast<unsigned>(osdn_weights::BLOB_K),
                  static_cast<unsigned>(osdn_weights::BLOB_D_IN),
                  static_cast<unsigned>(osdn_weights::BLOB_N_LAYERS),
                  osdn_weights::LOOKBACK,
                  static_cast<unsigned>(osdn_weights::BLOB_LEN));
    osdn_inf::reset(g_state,
                    static_cast<int>(osdn_weights::BLOB_K),
                    static_cast<int>(osdn_weights::BLOB_N_LAYERS));
}

// Parse "<t_min> <amount>" from the tail of a `b` or `m` line. Returns
// false (and emits an err line) on malformed input.
static bool parse_t_amount(const char* p, std::uint32_t& t_min, float& amount) {
    char* endp = nullptr;
    unsigned long t = strtoul(p, &endp, 10);
    if (endp == p) { Serial.println("err parse evt"); return false; }
    const char* q = endp;
    while (*q == ' ' || *q == '\t') ++q;
    float a = strtof(q, &endp);
    if (endp == q) { Serial.println("err parse evt"); return false; }
    t_min = static_cast<std::uint32_t>(t);
    amount = a;
    return true;
}

void loop() {
    if (!Serial.available()) { delay(10); return; }

    String line = Serial.readStringUntil('\n');
    line.trim();   // strips \r, leading/trailing whitespace
    if (line.length() == 0) { delay(10); return; }

    char cmd = line.charAt(0);
    const char* rest = line.c_str() + 1;
    while (*rest == ' ' || *rest == '\t') ++rest;

    switch (cmd) {
    case 't': {
        // Set device clock. No future/past validation — host owns this.
        char* endp = nullptr;
        unsigned long t_min = strtoul(rest, &endp, 10);
        if (endp == rest) { Serial.println("err parse t"); break; }
        g_now_min = static_cast<std::uint32_t>(t_min);
        break;
    }

    case 'b': {
        std::uint32_t t_min; float units;
        if (!parse_t_amount(rest, t_min, units)) break;
        if (t_min > g_now_min)         { Serial.println("err future evt"); break; }
        if (units < 0.0f || units > 30.0f) { Serial.println("err bolus oor"); break; }
        features::ring_append(g_iob, g_iob_head, g_iob_count, IOB_RING_N,
                              t_min, units);
        break;
    }

    case 'm': {
        std::uint32_t t_min; float carbs;
        if (!parse_t_amount(rest, t_min, carbs)) break;
        if (t_min > g_now_min)            { Serial.println("err future evt"); break; }
        if (carbs < 0.0f || carbs > 300.0f) { Serial.println("err meal oor");  break; }
        features::ring_append(g_meal, g_meal_head, g_meal_count, MEAL_RING_N,
                              t_min, carbs);
        break;
    }

    case 'g': {
        float mg_dl = strtof(rest, nullptr);
        if (!(mg_dl > 20.0f && mg_dl < 600.0f)) break;
        push_reading(mg_dl);
        if (g_ring_count < osdn_weights::LOOKBACK) {
            Serial.printf("cgm=%.1f buffering %d/%d iob=0.00 cob=0.00 t=%u\n",
                          mg_dl, g_ring_count, osdn_weights::LOOKBACK, g_now_min);
            break;
        }
        // 7-channel inference. compute_window asserts its warmup
        // precondition (g_count == LOOKBACK_STEPS && t_now_min >=
        // (L-1)*STEP_MIN) — the buffering branch above gates on
        // g_ring_count, and the replay tool/live deploy seed g_now_min
        // before the lookback fills, so both invariants hold here.
        features::compute_window(
            g_ring, g_ring_head, g_ring_count,
            g_iob,  g_iob_count,
            g_meal, g_meal_count,
            features::STEP_MIN, osdn_weights::LOOKBACK,
            g_now_min, g_feat);
        float logit = osdn_inf::forward(g_weights, g_state, g_feat,
                                        osdn_weights::LOOKBACK);
        float prob  = 1.0f / (1.0f + expf(-logit));
        bool  alert = logit > HYPO_LOGIT_THRESHOLD;
        digitalWrite(LED_PIN, alert ? HIGH : LOW);
        // Surface iob/cob at the final lookback step for host-side
        // cross-check against cgm_data.cpp's per-step features.
        const int last = (osdn_weights::LOOKBACK - 1) * 7;
        float iob_now = g_feat[last + 5];
        float cob_now = g_feat[last + 6];
        Serial.printf("cgm=%.1f logit=%.4f prob=%.4f alert=%d iob=%.2f cob=%.2f t=%u\n",
                      mg_dl, logit, prob, alert ? 1 : 0,
                      iob_now, cob_now, g_now_min);
        break;
    }

    default:
        Serial.println("err unknown cmd");
        break;
    }

    delay(10);
}
