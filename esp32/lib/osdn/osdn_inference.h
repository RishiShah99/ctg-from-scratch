#pragma once
//
// Header-only FP32 inference for the OSDN classifier produced by
// src/cgm_train.cpp + src/osdn.cpp. Source of truth is the trainer:
// every change to Net::forward / OSDNLayer::forward / Net::parameters()
// MUST be mirrored here. The osdn_inference_test reference forward in
// src/osdn_inference_test.cpp is the third copy of this math and is
// what catches drift.
//
// Param ordering (must match Net::parameters() in src/cgm_train.cpp):
//   embed_w[D_in][H]  (outer D_in, inner H)
//   embed_b[H]
//   per OSDN layer:
//     for i in [0,K), for h in [0,H): Wk[i][h], Wq[i][h], Wv[i][h]
//     wb[H], wb_bias
//     Wo[H][K]  (outer H, inner K)
//     D[H]
//     y_bias[H]
//     ln_gamma[K], ln_beta[K], log_lambda[K]   (K-vector — NOT a scalar)
//   head_w[H], head_b
//   pred_w[H], pred_b                          (regression head; consumed but
//                                               unused by the binary forward)
//
// d-update math is the paper §4.2 multiplicative correction:
//   n_t       = max(‖k‖², ε)
//   gate      = 1 − β · (d ⋅ k²)
//   d[i]     += (η · β · gate / n_t) · k[i]²
//   d[i]      = clamp(d[i], 0.5, 2.0)              ← paper §4.2 Π_D
// with η = 0.003 and ε = 1e-6 as compile-time constants — NOT learnable.
// The legacy EMA d-update from earlier versions has been removed.
//
// Verified envelope (bit-identity to src/osdn.cpp host autograd):
//   • H=16, K=8,  D_in=7, n_layers=1, L≤144  — PASS, |diff| < 1e-6 on
//     trained Ohio-brain weights.
//   • H=16, K=16, D_in=7, n_layers=2, L=144  — FAIL (FP32 NaN). The Π_D
//     clamp from §4.2 was sufficient to stabilise n_layers=1 K=8 but does
//     NOT extend to this larger config. Diagnosis hypothesis: at K=16 the
//     gate-magnitude ceiling β·⟨d,k²⟩ ≤ β·d_max·K = 2K doubles vs K=8, and
//     layer-2's input is the unbounded post-LN-and-projection output of
//     layer-1 with no input-side norm. Fixing this would likely require
//     either (a) key normalisation ‖k‖₂=1 (paper Tier-2 invariant — see
//     results/tier2_plan.md), or (b) a cross-layer input scaling, or
//     (c) a smaller d_max. None are speculatively applied; the shipped
//     brain-mode config (K=8, n_layers=1) is inside the verified envelope.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace osdn_inf {

static constexpr int H_MAX        = 32;
static constexpr int K_MAX        = 16;
static constexpr int D_IN_MAX     = 8;
static constexpr int N_LAYERS_MAX = 4;

// Paper §4.2 constants — must match src/osdn.cpp.
static constexpr float ETA = 0.003f;
static constexpr float OSDN_EPS = 1e-6f;

struct OSDNLayerW {
    float Wk[K_MAX][H_MAX];
    float Wq[K_MAX][H_MAX];
    float Wv[K_MAX][H_MAX];
    float wb[H_MAX];
    float wb_bias;
    float Wo[H_MAX][K_MAX];
    float D[H_MAX];
    float y_bias[H_MAX];
    float ln_gamma[K_MAX];
    float ln_beta[K_MAX];
    float log_lambda[K_MAX];
};

struct Weights {
    int H;
    int K;
    int D_in;
    int n_layers;
    float embed_w[D_IN_MAX][H_MAX];   // [c][h]
    float embed_b[H_MAX];
    OSDNLayerW layers[N_LAYERS_MAX];
    float head_w[H_MAX];
    float head_b;
    float pred_w[H_MAX];
    float pred_b;
};

struct OSDNLayerS {
    float S[K_MAX][K_MAX];
    float d[K_MAX];
};

struct State {
    OSDNLayerS layers[N_LAYERS_MAX];
};

inline std::size_t expected_blob_floats(int H, int K, int D_in, int n_layers) {
    const std::size_t per_layer =
        static_cast<std::size_t>(3 * K * H + H + 1 + H * K + H + H + K + K + K);
    return static_cast<std::size_t>(D_in * H) + static_cast<std::size_t>(H)
         + static_cast<std::size_t>(n_layers) * per_layer
         + static_cast<std::size_t>(H) + 1 + static_cast<std::size_t>(H) + 1;
}

inline bool load_blob(Weights& w, int H, int K, int D_in, int n_layers,
                      const float* blob, std::size_t n_floats) {
    if (H > H_MAX || K > K_MAX || D_in > D_IN_MAX || n_layers > N_LAYERS_MAX) return false;
    if (H <= 0 || K <= 0 || D_in <= 0 || n_layers <= 0) return false;
    if (n_floats != expected_blob_floats(H, K, D_in, n_layers)) return false;
    w.H = H;
    w.K = K;
    w.D_in = D_in;
    w.n_layers = n_layers;
    std::size_t p = 0;
    for (int c = 0; c < D_in; ++c)
        for (int h = 0; h < H; ++h) w.embed_w[c][h] = blob[p++];
    for (int h = 0; h < H; ++h) w.embed_b[h] = blob[p++];
    for (int li = 0; li < n_layers; ++li) {
        OSDNLayerW& lw = w.layers[li];
        for (int i = 0; i < K; ++i)
            for (int h = 0; h < H; ++h) {
                lw.Wk[i][h] = blob[p++];
                lw.Wq[i][h] = blob[p++];
                lw.Wv[i][h] = blob[p++];
            }
        for (int h = 0; h < H; ++h) lw.wb[h] = blob[p++];
        lw.wb_bias = blob[p++];
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < K; ++i) lw.Wo[h][i] = blob[p++];
        for (int h = 0; h < H; ++h) lw.D[h] = blob[p++];
        for (int h = 0; h < H; ++h) lw.y_bias[h] = blob[p++];
        for (int i = 0; i < K; ++i) lw.ln_gamma[i] = blob[p++];
        for (int i = 0; i < K; ++i) lw.ln_beta[i]  = blob[p++];
        for (int i = 0; i < K; ++i) lw.log_lambda[i] = blob[p++];
    }
    for (int h = 0; h < H; ++h) w.head_w[h] = blob[p++];
    w.head_b = blob[p++];
    for (int h = 0; h < H; ++h) w.pred_w[h] = blob[p++];
    w.pred_b = blob[p++];
    return p == n_floats;
}

inline void reset(State& st, int K, int n_layers) {
    for (int li = 0; li < n_layers; ++li) {
        OSDNLayerS& ls = st.layers[li];
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) ls.S[i][j] = 0.0f;
            ls.d[i] = 1.0f;
        }
    }
}

inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// Run one OSDN layer step in-place. Reads x[H], writes y[H], updates ls.
// Scratch buffers k/q/vv/Sk/u/o/on/ktilde/k_sq must be sized to K_MAX.
inline void osdn_step(const OSDNLayerW& lw, OSDNLayerS& ls,
                      int H, int K,
                      const float* x, float* y) {
    float k[K_MAX], q[K_MAX], vv[K_MAX];
    for (int i = 0; i < K; ++i) {
        float ak = 0.0f, aq = 0.0f, av = 0.0f;
        for (int h = 0; h < H; ++h) {
            ak += lw.Wk[i][h] * x[h];
            aq += lw.Wq[i][h] * x[h];
            av += lw.Wv[i][h] * x[h];
        }
        k[i]  = ak;
        q[i]  = aq;
        vv[i] = av;
    }

    float blogit = lw.wb_bias;
    for (int h = 0; h < H; ++h) blogit += lw.wb[h] * x[h];
    const float beta = sigmoidf(blogit);

    float ktilde[K_MAX];
    float lambda[K_MAX];
    for (int i = 0; i < K; ++i) {
        lambda[i] = sigmoidf(lw.log_lambda[i]);
        ktilde[i] = ls.d[i] * k[i];
    }

    float Sk[K_MAX];
    for (int i = 0; i < K; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) acc += ls.S[i][j] * k[j];
        Sk[i] = acc;
    }

    float u[K_MAX];
    for (int i = 0; i < K; ++i) u[i] = vv[i] - Sk[i];

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            ls.S[i][j] = lambda[i] * ls.S[i][j] + beta * u[i] * ktilde[j];

    float o[K_MAX];
    for (int i = 0; i < K; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) acc += ls.S[i][j] * q[j];
        o[i] = acc;
    }

    float mean_o = 0.0f;
    for (int i = 0; i < K; ++i) mean_o += o[i];
    mean_o /= static_cast<float>(K);
    float var_o = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float c = o[i] - mean_o;
        var_o += c * c;
    }
    var_o /= static_cast<float>(K);
    const float inv_std = 1.0f / std::sqrt(var_o + 1e-5f);
    float on[K_MAX];
    for (int i = 0; i < K; ++i)
        on[i] = lw.ln_gamma[i] * ((o[i] - mean_o) * inv_std) + lw.ln_beta[i];

    for (int h = 0; h < H; ++h) {
        float acc = lw.y_bias[h] + lw.D[h] * x[h];
        for (int i = 0; i < K; ++i) acc += lw.Wo[h][i] * on[i];
        y[h] = acc;
    }

    // Paper §4.2 d-update: floor n = max(‖k‖², ε) implemented as relu(x-ε)+ε,
    // gate by β(1 − β · d·k²), scale by η/n, then d[i] += scale·k[i]².
    float k_sq[K_MAX];
    float k_sqsum = 0.0f;
    for (int i = 0; i < K; ++i) {
        k_sq[i] = k[i] * k[i];
        k_sqsum += k_sq[i];
    }
    float dot_dk2 = 0.0f;
    for (int i = 0; i < K; ++i) dot_dk2 += ls.d[i] * k_sq[i];
    const float gate_term = 1.0f - beta * dot_dk2;
    const float n_floor = (k_sqsum > OSDN_EPS) ? (k_sqsum - OSDN_EPS) : 0.0f;
    const float norm = n_floor + OSDN_EPS;
    const float norm_inv = 1.0f / norm;
    const float scale = ETA * beta * gate_term * norm_inv;
    for (int i = 0; i < K; ++i) ls.d[i] += scale * k_sq[i];
    // Mirror of trainer's Π_D clamp (osdn.cpp) — paper §4.2, D = [0.5, 2.0]^K.
    for (int i = 0; i < K; ++i)
        ls.d[i] = std::fmax(0.5f, std::fmin(2.0f, ls.d[i]));
}

// Run the full forward pass. `feat` is [L * D_in] row-major: feat[t*D_in + c].
// Returns the binary classification logit.
//
// Pool semantics: cumulative-mean-per-step. The trainer computes
// run_sum[h] = sum_{t'<=t} y[t'][h] each step, then the FINAL classifier
// reads run_sum[h] / L at the last timestep — same as adding to a per-h
// accumulator each step and dividing once at the end. We implement the
// equivalent: pool[h] += y_post_stack[h] every step; final logit uses
// pool[h] / L.
inline float forward(const Weights& w, State& st, const float* feat, int L) {
    const int H = w.H;
    const int K = w.K;
    const int D_in = w.D_in;
    const int n_layers = w.n_layers;

    reset(st, K, n_layers);

    float pool[H_MAX] = {0.0f};
    float buf_a[H_MAX];
    float buf_b[H_MAX];

    for (int t = 0; t < L; ++t) {
        // Embed: x = tanh(embed_b + sum_c embed_w[c][h] * feat[t,c])
        for (int h = 0; h < H; ++h) {
            float acc = w.embed_b[h];
            for (int c = 0; c < D_in; ++c) {
                acc += w.embed_w[c][h] * feat[t * D_in + c];
            }
            buf_a[h] = std::tanh(acc);
        }

        // Layer stack: ping-pong between buf_a (input) and buf_b (output).
        float* x_ptr = buf_a;
        float* y_ptr = buf_b;
        for (int li = 0; li < n_layers; ++li) {
            osdn_step(w.layers[li], st.layers[li], H, K, x_ptr, y_ptr);
            float* tmp = x_ptr;
            x_ptr = y_ptr;
            y_ptr = tmp;
        }
        // After the loop, x_ptr holds the final layer's output for this t.
        for (int h = 0; h < H; ++h) pool[h] += x_ptr[h];
    }

    const float inv_L = 1.0f / static_cast<float>(L);
    float logit = w.head_b;
    for (int h = 0; h < H; ++h) logit += w.head_w[h] * (pool[h] * inv_L);
    return logit;
}

}  // namespace osdn_inf
