#include "osdn.hpp"
#include <cmath>

OSDNLayer::OSDNLayer(int H_, int K_, std::mt19937& rng) : H(H_), K(K_) {
    std::normal_distribution<double> n01(0.0, 1.0);
    double sH = 1.0 / std::sqrt(static_cast<double>(H_));
    double sK = 0.5 / std::sqrt(static_cast<double>(K_));

    Wk.assign(K_, std::vector<ValuePtr>(H_));
    Wq.assign(K_, std::vector<ValuePtr>(H_));
    Wv.assign(K_, std::vector<ValuePtr>(H_));
    for (int k = 0; k < K_; ++k) {
        for (int h = 0; h < H_; ++h) {
            Wk[k][h] = v(sH * n01(rng));
            Wq[k][h] = v(sH * n01(rng));
            Wv[k][h] = v(sH * n01(rng));
        }
    }

    wb.assign(H_, ValuePtr{});
    for (int h = 0; h < H_; ++h) wb[h] = v(sH * n01(rng) * 0.1);
    wb_bias = v(-std::log(2.0));

    Wo.assign(H_, std::vector<ValuePtr>(K_));
    for (int h = 0; h < H_; ++h)
        for (int k = 0; k < K_; ++k)
            Wo[h][k] = v(sK * n01(rng));

    D.assign(H_, ValuePtr{});
    for (int h = 0; h < H_; ++h) D[h] = v(1.0);

    y_bias.assign(H_, ValuePtr{});
    for (int h = 0; h < H_; ++h) y_bias[h] = v(0.0);

    ln_gamma.assign(K_, ValuePtr{});
    ln_beta.assign(K_, ValuePtr{});
    for (int i = 0; i < K_; ++i) { ln_gamma[i] = v(1.0); ln_beta[i] = v(0.0); }

    log_lambda.assign(K_, ValuePtr{});
    double lo = std::log(0.85 / 0.15);
    double hi = std::log(0.98 / 0.02);
    for (int i = 0; i < K_; ++i) {
        double t = K_ > 1 ? static_cast<double>(i) / (K_ - 1) : 0.5;
        log_lambda[i] = v(lo + t * (hi - lo));
    }
}

std::vector<std::vector<ValuePtr>> OSDNLayer::forward(
    const std::vector<std::vector<ValuePtr>>& x) const
{
    int L = static_cast<int>(x.size());
    std::vector<std::vector<ValuePtr>> y(L, std::vector<ValuePtr>(H));

    // Initial state: distinct ValuePtr per cell. The vector fill-ctor would
    // copy ONE shared_ptr K*K times, so every S[i][j] would alias the same
    // Value and backprop would accumulate K*K gradient contributions onto a
    // throwaway initial node. Subsequent forward writes create fresh nodes
    // so the aliasing dissolves after t=0, but the cleaner construction
    // makes intent obvious and removes a foot-gun if someone reads the
    // initial state before the loop body.
    std::vector<std::vector<ValuePtr>> S(K, std::vector<ValuePtr>(K));
    std::vector<ValuePtr> d(K);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) S[i][j] = v(0.0);
        d[i] = v(1.0);
    }

    std::vector<ValuePtr> lambda(K);
    for (int i = 0; i < K; ++i) lambda[i] = vsigmoid(log_lambda[i]);

    const double eta = 0.003;
    const double eps = 1e-6;

    for (int t = 0; t < L; ++t) {
        const auto& xt = x[t];

        std::vector<ValuePtr> k(K, v(0.0));
        std::vector<ValuePtr> q(K, v(0.0));
        std::vector<ValuePtr> vv(K, v(0.0));
        for (int i = 0; i < K; ++i) {
            ValuePtr ak = v(0.0);
            ValuePtr aq = v(0.0);
            ValuePtr av = v(0.0);
            for (int h = 0; h < H; ++h) {
                ak = ak + Wk[i][h] * xt[h];
                aq = aq + Wq[i][h] * xt[h];
                av = av + Wv[i][h] * xt[h];
            }
            k[i]  = ak;
            q[i]  = aq;
            vv[i] = av;
        }

        ValuePtr blogit = wb_bias;
        for (int h = 0; h < H; ++h) blogit = blogit + wb[h] * xt[h];
        ValuePtr beta = vsigmoid(blogit);

        std::vector<ValuePtr> ktilde(K);
        for (int i = 0; i < K; ++i) ktilde[i] = d[i] * k[i];

        std::vector<ValuePtr> Sk(K, v(0.0));
        for (int i = 0; i < K; ++i) {
            ValuePtr acc = v(0.0);
            for (int j = 0; j < K; ++j) acc = acc + S[i][j] * k[j];
            Sk[i] = acc;
        }

        std::vector<ValuePtr> u(K);
        for (int i = 0; i < K; ++i) u[i] = vv[i] - Sk[i];

        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                S[i][j] = lambda[i] * S[i][j] + beta * u[i] * ktilde[j];

        std::vector<ValuePtr> o(K, v(0.0));
        for (int i = 0; i < K; ++i) {
            ValuePtr acc = v(0.0);
            for (int j = 0; j < K; ++j) acc = acc + S[i][j] * q[j];
            o[i] = acc;
        }

        ValuePtr mean_o = v(0.0);
        for (int i = 0; i < K; ++i) mean_o = mean_o + o[i];
        mean_o = mean_o * (1.0 / static_cast<double>(K));
        ValuePtr var_o = v(0.0);
        for (int i = 0; i < K; ++i) {
            ValuePtr c = o[i] - mean_o;
            var_o = var_o + c * c;
        }
        var_o = var_o * (1.0 / static_cast<double>(K));
        ValuePtr inv_std = vexp(v(-0.5) * vlog(var_o + v(1e-5)));
        std::vector<ValuePtr> on(K);
        for (int i = 0; i < K; ++i) on[i] = ln_gamma[i] * ((o[i] - mean_o) * inv_std) + ln_beta[i];

        for (int h = 0; h < H; ++h) {
            ValuePtr acc = y_bias[h] + D[h] * xt[h];
            for (int i = 0; i < K; ++i) acc = acc + Wo[h][i] * on[i];
            y[t][h] = acc;
        }

        std::vector<ValuePtr> k_sq(K);
        ValuePtr k_sqsum = v(0.0);
        for (int i = 0; i < K; ++i) {
            k_sq[i] = k[i] * k[i];
            k_sqsum = k_sqsum + k_sq[i];
        }
        ValuePtr dot_dk2 = v(0.0);
        for (int i = 0; i < K; ++i) dot_dk2 = dot_dk2 + d[i] * k_sq[i];
        ValuePtr gate_term = v(1.0) - beta * dot_dk2;
        // paper-exact floor: n_t = max(‖k‖², ε)  →  relu(x - ε) + ε
        ValuePtr norm = vrelu(k_sqsum - v(eps)) + v(eps);
        ValuePtr norm_inv = v(1.0) / norm;
        ValuePtr scale = v(eta) * beta * gate_term * norm_inv;
        for (int i = 0; i < K; ++i) {
            d[i] = d[i] + scale * k_sq[i];
        }
    }

    return y;
}

std::vector<ValuePtr> OSDNLayer::parameters() const {
    std::vector<ValuePtr> ps;
    ps.reserve(K * H * 3 + H + 1 + H * K + H + H + 2 * K + K);
    for (int i = 0; i < K; ++i) {
        for (int h = 0; h < H; ++h) {
            ps.push_back(Wk[i][h]);
            ps.push_back(Wq[i][h]);
            ps.push_back(Wv[i][h]);
        }
    }
    for (int h = 0; h < H; ++h) ps.push_back(wb[h]);
    ps.push_back(wb_bias);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < K; ++i)
            ps.push_back(Wo[h][i]);
    for (int h = 0; h < H; ++h) ps.push_back(D[h]);
    for (int h = 0; h < H; ++h) ps.push_back(y_bias[h]);
    for (int i = 0; i < K; ++i) ps.push_back(ln_gamma[i]);
    for (int i = 0; i < K; ++i) ps.push_back(ln_beta[i]);
    for (int i = 0; i < K; ++i) ps.push_back(log_lambda[i]);
    return ps;
}
