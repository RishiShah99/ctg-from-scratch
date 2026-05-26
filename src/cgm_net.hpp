#pragma once
//
// Host-autograd Net wrapper shared by cgm_train.cpp and cgm_demo.cpp.
// Before Step 5 each translation unit defined its own Net (with a
// different number of channels and no regression head in the demo) —
// loaded weights silently misaligned. Sharing this header guarantees
// both binaries see the same parameter layout.
//
// Parameter ordering (must match the on-disk blob payload):
//   embed_w[D_in][H]                          — outer c, inner h
//   embed_b[H]
//   arch-specific layer params:
//     "s4d":   S4DLayer::parameters()
//     "osdn":  per-layer concat of OSDNLayer::parameters()
//   head_w[H]
//   head_b
//   pred_w[H]                                 — regression head
//   pred_b
//
// The fwd path supports D_in in [1,7]: 1=g only, 3=+Δg+Δ²g, 5=+sin/cos(tod),
// 7=+IOB+COB. Higher channels are zero-filled in `forward` if the window
// hasn't populated them — this mirrors the trainer's behavior.
//
#include "value.hpp"
#include "s4d.hpp"
#include "osdn.hpp"
#include "cgm_data.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

struct FwdOut {
    ValuePtr final_logit;
    std::vector<ValuePtr> step_logits;
    std::vector<ValuePtr> step_preds;
};

struct Net {
    int H;
    int D_in;
    std::string arch;
    std::vector<std::vector<ValuePtr>> embed_w;   // [D_in][H]
    std::vector<ValuePtr> embed_b;
    std::unique_ptr<S4DLayer>  s4d;
    std::vector<std::unique_ptr<OSDNLayer>> osdn_stack;
    std::vector<ValuePtr> head_w;
    ValuePtr head_b;
    std::vector<ValuePtr> pred_w;
    ValuePtr pred_b;

    Net(const std::string& arch_, int H_, int N_, double dt_, int n_osdn_layers,
        int D_in_, std::mt19937& rng)
        : H(H_), D_in(D_in_), arch(arch_)
    {
        if (arch_ == "s4d") {
            s4d = std::make_unique<S4DLayer>(H_, N_, dt_, rng);
        } else {
            int nl = n_osdn_layers > 0 ? n_osdn_layers : 1;
            osdn_stack.reserve(nl);
            for (int i = 0; i < nl; ++i) {
                osdn_stack.push_back(std::make_unique<OSDNLayer>(H_, N_, rng));
            }
        }
        std::normal_distribution<double> n01(0.0, 1.0);
        embed_w.assign(D_in_, std::vector<ValuePtr>(H_));
        embed_b.assign(H_, ValuePtr{});
        head_w.assign(H_, ValuePtr{});
        pred_w.assign(H_, ValuePtr{});
        double s  = 1.0 / std::sqrt(static_cast<double>(H_) * static_cast<double>(D_in_));
        double sH = 1.0 / std::sqrt(static_cast<double>(H_));
        for (int c = 0; c < D_in_; ++c) {
            for (int h = 0; h < H_; ++h) embed_w[c][h] = v(s * n01(rng));
        }
        for (int h = 0; h < H_; ++h) {
            embed_b[h] = v(0.0);
            head_w[h]  = v(sH * n01(rng));
            pred_w[h]  = v(sH * n01(rng));
        }
        head_b = v(0.0);
        pred_b = v(0.0);
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& row : embed_w) for (auto& q : row) p.push_back(q);
        for (auto& q : embed_b) p.push_back(q);
        if (arch == "s4d") {
            for (auto& q : s4d->parameters()) p.push_back(q);
        } else {
            for (auto& layer : osdn_stack) {
                for (auto& q : layer->parameters()) p.push_back(q);
            }
        }
        for (auto& q : head_w) p.push_back(q);
        p.push_back(head_b);
        for (auto& q : pred_w) p.push_back(q);
        p.push_back(pred_b);
        return p;
    }

    FwdOut forward(const CGMWindow& w) {
        int L = static_cast<int>(w.lookback.size());
        std::vector<std::vector<double>> ch(D_in, std::vector<double>(L, 0.0));
        for (int t = 0; t < L; ++t) ch[0][t] = w.lookback[t];
        if (D_in >= 3) {
            for (int t = 1; t < L; ++t) ch[1][t] = w.lookback[t] - w.lookback[t-1];
            for (int t = 2; t < L; ++t) ch[2][t] = ch[1][t] - ch[1][t-1];
        }
        if (D_in >= 5) {
            int n_tod = std::min<int>(L, static_cast<int>(w.tod_sin.size()));
            for (int t = 0; t < n_tod; ++t) {
                ch[3][t] = w.tod_sin[t];
                ch[4][t] = w.tod_cos[t];
            }
        }
        if (D_in >= 7) {
            int n_iob = std::min<int>(L, static_cast<int>(w.iob.size()));
            int n_cob = std::min<int>(L, static_cast<int>(w.cob.size()));
            for (int t = 0; t < n_iob; ++t) ch[5][t] = w.iob[t];
            for (int t = 0; t < n_cob; ++t) ch[6][t] = w.cob[t];
        }

        std::vector<std::vector<ValuePtr>> x(L, std::vector<ValuePtr>(H));
        for (int t = 0; t < L; ++t) {
            for (int h = 0; h < H; ++h) {
                ValuePtr acc = embed_b[h];
                for (int c = 0; c < D_in; ++c) acc = acc + embed_w[c][h] * ch[c][t];
                x[t][h] = vtanh(acc);
            }
        }

        std::vector<std::vector<ValuePtr>> y;
        if (arch == "s4d") {
            y = s4d->forward(x);
        } else {
            y = osdn_stack[0]->forward(x);
            for (size_t i = 1; i < osdn_stack.size(); ++i) {
                y = osdn_stack[i]->forward(y);
            }
        }

        FwdOut out;
        out.step_logits.reserve(L);
        out.step_preds.reserve(L);

        std::vector<ValuePtr> run_sum(H, v(0.0));
        for (int t = 0; t < L; ++t) {
            for (int h = 0; h < H; ++h) run_sum[h] = run_sum[h] + y[t][h];
            double inv_n = 1.0 / static_cast<double>(t + 1);
            ValuePtr lg = head_b;
            for (int h = 0; h < H; ++h) lg = lg + head_w[h] * (run_sum[h] * inv_n);
            out.step_logits.push_back(lg);
            ValuePtr pr = pred_b;
            for (int h = 0; h < H; ++h) pr = pr + pred_w[h] * (run_sum[h] * inv_n);
            out.step_preds.push_back(pr);
        }
        out.final_logit = out.step_logits.back();
        return out;
    }
};
