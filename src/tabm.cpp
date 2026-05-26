#include "tabm.hpp"
#include <cmath>

BatchEnsembleLayer::BatchEnsembleLayer(int in_features, int out_features, int k, std::mt19937& rng)
    : in_(in_features), out_(out_features), k_(k) {
    double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
    std::uniform_real_distribution<double> dist(-limit, limit);

    W_.assign(out_features, std::vector<ValuePtr>(in_features));
    for (int o = 0; o < out_features; ++o)
        for (int j = 0; j < in_features; ++j)
            W_[o][j] = v(dist(rng));

    std::uniform_int_distribution<int> sign(0, 1);
    r_.assign(k, std::vector<ValuePtr>(in_features));
    s_.assign(k, std::vector<ValuePtr>(out_features));
    b_.assign(k, std::vector<ValuePtr>(out_features));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < in_features; ++j) r_[i][j] = v(sign(rng) ? 1.0 : -1.0);
        for (int o = 0; o < out_features; ++o) {
            s_[i][o] = v(1.0);
            b_[i][o] = v(0.0);
        }
    }
}

std::vector<std::vector<ValuePtr>> BatchEnsembleLayer::forward(const std::vector<ValuePtr>& x) const {
    std::vector<std::vector<ValuePtr>> result;
    result.reserve(k_);
    for (int i = 0; i < k_; ++i) {
        std::vector<ValuePtr> m_x(in_);
        for (int j = 0; j < in_; ++j) m_x[j] = r_[i][j] * x[j];

        std::vector<ValuePtr> out(out_);
        for (int o = 0; o < out_; ++o) {
            ValuePtr pre = W_[o][0] * m_x[0];
            for (int j = 1; j < in_; ++j) pre = pre + W_[o][j] * m_x[j];
            out[o] = s_[i][o] * pre + b_[i][o];
        }
        result.push_back(std::move(out));
    }
    return result;
}

std::vector<ValuePtr> BatchEnsembleLayer::parameters() const {
    // r_ holds the rank-1 sign vectors that the paper initializes to ±1
    // and freezes thereafter. Including them in parameters() would let
    // Adam push them off ±1, defeating the BatchEnsemble formulation.
    // s_ and b_ remain learnable as the per-ensemble scale and bias.
    std::vector<ValuePtr> params;
    for (const auto& row : W_) for (const auto& w : row) params.push_back(w);
    for (const auto& row : s_) for (const auto& s : row) params.push_back(s);
    for (const auto& row : b_) for (const auto& b : row) params.push_back(b);
    return params;
}

TabM::TabM(const std::vector<int>& widths, int k, std::mt19937& rng)
    : k_(k), be_(widths[0], widths[1], k, rng) {
    for (size_t i = 1; i + 1 < widths.size(); ++i) {
        SharedLinear lin;
        lin.in       = widths[i];
        lin.out      = widths[i + 1];
        lin.use_gelu = (i + 2 < widths.size());

        double limit = std::sqrt(6.0 / static_cast<double>(lin.in + lin.out));
        std::uniform_real_distribution<double> dist(-limit, limit);

        lin.W.assign(lin.out, std::vector<ValuePtr>(lin.in));
        for (int o = 0; o < lin.out; ++o)
            for (int j = 0; j < lin.in; ++j)
                lin.W[o][j] = v(dist(rng));

        lin.b.assign(lin.out, nullptr);
        for (int o = 0; o < lin.out; ++o) lin.b[o] = v(0.0);

        shared_.push_back(std::move(lin));
    }
}

std::vector<std::vector<ValuePtr>> TabM::forward(const std::vector<ValuePtr>& x) const {
    auto streams = be_.forward(x);
    for (const auto& lin : shared_) {
        for (int i = 0; i < k_; ++i) {
            std::vector<ValuePtr> out(lin.out);
            for (int o = 0; o < lin.out; ++o) {
                ValuePtr acc = lin.b[o];
                for (int j = 0; j < lin.in; ++j) acc = acc + lin.W[o][j] * streams[i][j];
                out[o] = lin.use_gelu ? vgelu(acc) : acc;
            }
            streams[i] = std::move(out);
        }
    }
    return streams;
}

std::vector<ValuePtr> TabM::parameters() const {
    auto params = be_.parameters();
    for (const auto& lin : shared_) {
        for (const auto& row : lin.W) for (const auto& w : row) params.push_back(w);
        for (const auto& b : lin.b) params.push_back(b);
    }
    return params;
}
