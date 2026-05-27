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
static constexpr float HYPO_LOGIT_THRESHOLD = 0.0f;

static float normalize(float mg_dl) {
    return (mg_dl - osdn_weights::MEAN_MG_DL) / osdn_weights::STD_MG_DL;
}

static void push_reading(float mg_dl) {
    g_ring[g_ring_head] = normalize(mg_dl);
    g_ring_head = (g_ring_head + 1) % osdn_weights::LOOKBACK;
    if (g_ring_count < osdn_weights::LOOKBACK) g_ring_count++;
}

static float run_inference() {
    static float sig[osdn_weights::LOOKBACK];
    int n = g_ring_count;
    int start = (g_ring_head - n + osdn_weights::LOOKBACK) % osdn_weights::LOOKBACK;
    for (int t = 0; t < n; ++t) sig[t] = g_ring[(start + t) % osdn_weights::LOOKBACK];
    return osdn_inf::forward(g_weights, g_state, sig, n);
}

void setup() {
    Serial.begin(115200);
    delay(500);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    bool ok = osdn_inf::load_blob(
        g_weights,
        osdn_weights::H,
        osdn_weights::K,
        osdn_weights::BLOB,
        osdn_weights::BLOB_LEN);

    Serial.printf("OSDN load: %s  H=%d K=%d L=%d  params=%u\n",
                  ok ? "ok" : "FAIL",
                  osdn_weights::H, osdn_weights::K,
                  osdn_weights::LOOKBACK,
                  static_cast<unsigned>(osdn_weights::BLOB_LEN));
    osdn_inf::reset(g_state, osdn_weights::K);
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
        // Single-channel inference path retained from baseline; commit 4
        // replaces it with the 7-channel features::compute_window path.
        if (g_ring_count >= osdn_weights::LOOKBACK) {
            float logit = run_inference();
            float prob  = 1.0f / (1.0f + expf(-logit));
            bool alert = logit > HYPO_LOGIT_THRESHOLD;
            digitalWrite(LED_PIN, alert ? HIGH : LOW);
            Serial.printf("cgm=%.1f logit=%.4f prob=%.4f alert=%d t=%u\n",
                          mg_dl, logit, prob, alert ? 1 : 0, g_now_min);
        } else {
            Serial.printf("cgm=%.1f buffering %d/%d t=%u\n",
                          mg_dl, g_ring_count, osdn_weights::LOOKBACK, g_now_min);
        }
        break;
    }

    default:
        Serial.println("err unknown cmd");
        break;
    }

    delay(10);
}
