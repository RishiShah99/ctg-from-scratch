#include "tx.hpp"
#include "loss.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

static double xavier_limit(int fan_in, int fan_out) {
    return std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
}

LayerNorm::LayerNorm(int dim, std::mt19937& /*rng*/)
    : dim_(dim), gamma_(dim), beta_(dim), eps_(1e-5) {
    for (int i = 0; i < dim; ++i) {
        gamma_[i] = v(1.0);
        beta_[i]  = v(0.0);
    }
}

std::vector<ValuePtr> LayerNorm::forward(const std::vector<ValuePtr>& x) const {
    int d = dim_;
    double inv_d = 1.0 / static_cast<double>(d);

    ValuePtr m = x[0];
    for (int i = 1; i < d; ++i) m = m + x[i];
    m = inv_d * m;

    std::vector<ValuePtr> diff(d);
    for (int i = 0; i < d; ++i) diff[i] = x[i] - m;

    ValuePtr var = vpow(diff[0], 2.0);
    for (int i = 1; i < d; ++i) var = var + vpow(diff[i], 2.0);
    var = inv_d * var;

    auto inv_std = vpow(var + eps_, -0.5);

    std::vector<ValuePtr> out(d);
    for (int i = 0; i < d; ++i) out[i] = gamma_[i] * (diff[i] * inv_std) + beta_[i];
    return out;
}

std::vector<ValuePtr> LayerNorm::parameters() const {
    std::vector<ValuePtr> p = gamma_;
    p.insert(p.end(), beta_.begin(), beta_.end());
    return p;
}

static int mha_check_dh(int d_model, int n_heads) {
    if (n_heads <= 0 || d_model <= 0 || d_model % n_heads != 0) {
        throw std::invalid_argument(
            "MultiHeadAttention: d_model (" + std::to_string(d_model) +
            ") must be a positive multiple of n_heads (" +
            std::to_string(n_heads) + "); otherwise per-head dim would "
            "silently truncate and drop dimensions");
    }
    return d_model / n_heads;
}

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads, std::mt19937& rng)
    : d_(d_model), h_(n_heads), dh_(mha_check_dh(d_model, n_heads)),
      Wq_(d_model, std::vector<ValuePtr>(d_model)),
      Wk_(d_model, std::vector<ValuePtr>(d_model)),
      Wv_(d_model, std::vector<ValuePtr>(d_model)),
      Wo_(d_model, std::vector<ValuePtr>(d_model)),
      bq_(d_model), bk_(d_model), bv_(d_model), bo_(d_model) {
    double limit = xavier_limit(d_model, d_model);
    std::uniform_real_distribution<double> dist(-limit, limit);
    auto init_W = [&](std::vector<std::vector<ValuePtr>>& W) {
        for (int i = 0; i < d_model; ++i)
            for (int j = 0; j < d_model; ++j)
                W[i][j] = v(dist(rng));
    };
    auto init_b = [&](std::vector<ValuePtr>& b) {
        for (int i = 0; i < d_model; ++i) b[i] = v(0.0);
    };
    init_W(Wq_); init_W(Wk_); init_W(Wv_); init_W(Wo_);
    init_b(bq_); init_b(bk_); init_b(bv_); init_b(bo_);
}

std::vector<std::vector<ValuePtr>> MultiHeadAttention::forward(
    const std::vector<std::vector<ValuePtr>>& X) const {
    int n = static_cast<int>(X.size());
    int d = d_, h = h_, dh = dh_;
    double scale = 1.0 / std::sqrt(static_cast<double>(dh));

    auto project = [&](const std::vector<std::vector<ValuePtr>>& W,
                       const std::vector<ValuePtr>& b) {
        std::vector<std::vector<ValuePtr>> Y(n, std::vector<ValuePtr>(d));
        for (int t = 0; t < n; ++t) {
            for (int k = 0; k < d; ++k) {
                ValuePtr acc = b[k];
                for (int j = 0; j < d; ++j) acc = acc + W[j][k] * X[t][j];
                Y[t][k] = acc;
            }
        }
        return Y;
    };
    auto Q = project(Wq_, bq_);
    auto K = project(Wk_, bk_);
    auto V = project(Wv_, bv_);

    std::vector<std::vector<ValuePtr>> heads_out(n, std::vector<ValuePtr>(d));

    for (int head = 0; head < h; ++head) {
        std::vector<std::vector<ValuePtr>> scores(n, std::vector<ValuePtr>(n));
        for (int q = 0; q < n; ++q) {
            for (int k = 0; k < n; ++k) {
                ValuePtr s = Q[q][head * dh] * K[k][head * dh];
                for (int i = 1; i < dh; ++i)
                    s = s + Q[q][head * dh + i] * K[k][head * dh + i];
                scores[q][k] = scale * s;
            }
        }
        for (int q = 0; q < n; ++q) scores[q] = softmax(scores[q]);
        for (int q = 0; q < n; ++q) {
            for (int i = 0; i < dh; ++i) {
                ValuePtr acc = scores[q][0] * V[0][head * dh + i];
                for (int k = 1; k < n; ++k)
                    acc = acc + scores[q][k] * V[k][head * dh + i];
                heads_out[q][head * dh + i] = acc;
            }
        }
    }

    std::vector<std::vector<ValuePtr>> final_out(n, std::vector<ValuePtr>(d));
    for (int t = 0; t < n; ++t) {
        for (int k = 0; k < d; ++k) {
            ValuePtr acc = bo_[k];
            for (int i = 0; i < d; ++i) acc = acc + Wo_[i][k] * heads_out[t][i];
            final_out[t][k] = acc;
        }
    }
    return final_out;
}

std::vector<ValuePtr> MultiHeadAttention::parameters() const {
    std::vector<ValuePtr> p;
    auto append_W = [&](const std::vector<std::vector<ValuePtr>>& W) {
        for (const auto& row : W) for (const auto& w : row) p.push_back(w);
    };
    auto append_b = [&](const std::vector<ValuePtr>& b) {
        for (const auto& w : b) p.push_back(w);
    };
    append_W(Wq_); append_W(Wk_); append_W(Wv_); append_W(Wo_);
    append_b(bq_); append_b(bk_); append_b(bv_); append_b(bo_);
    return p;
}

FFN::FFN(int d_model, int d_hidden, std::mt19937& rng)
    : d_(d_model), dh_(d_hidden),
      W1_(d_model, std::vector<ValuePtr>(d_hidden)),
      W2_(d_hidden, std::vector<ValuePtr>(d_model)),
      b1_(d_hidden), b2_(d_model) {
    double l1 = xavier_limit(d_model, d_hidden);
    std::uniform_real_distribution<double> d1(-l1, l1);
    for (int i = 0; i < d_model; ++i)
        for (int j = 0; j < d_hidden; ++j)
            W1_[i][j] = v(d1(rng));
    double l2 = xavier_limit(d_hidden, d_model);
    std::uniform_real_distribution<double> d2(-l2, l2);
    for (int i = 0; i < d_hidden; ++i)
        for (int j = 0; j < d_model; ++j)
            W2_[i][j] = v(d2(rng));
    for (int j = 0; j < d_hidden; ++j) b1_[j] = v(0.0);
    for (int j = 0; j < d_model; ++j) b2_[j] = v(0.0);
}

std::vector<ValuePtr> FFN::forward(const std::vector<ValuePtr>& x) const {
    std::vector<ValuePtr> h(dh_);
    for (int j = 0; j < dh_; ++j) {
        ValuePtr acc = b1_[j];
        for (int i = 0; i < d_; ++i) acc = acc + W1_[i][j] * x[i];
        h[j] = vgelu(acc);
    }
    std::vector<ValuePtr> out(d_);
    for (int j = 0; j < d_; ++j) {
        ValuePtr acc = b2_[j];
        for (int i = 0; i < dh_; ++i) acc = acc + W2_[i][j] * h[i];
        out[j] = acc;
    }
    return out;
}

std::vector<ValuePtr> FFN::parameters() const {
    std::vector<ValuePtr> p;
    for (const auto& row : W1_) for (const auto& w : row) p.push_back(w);
    for (const auto& w : b1_) p.push_back(w);
    for (const auto& row : W2_) for (const auto& w : row) p.push_back(w);
    for (const auto& w : b2_) p.push_back(w);
    return p;
}

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ffn, std::mt19937& rng)
    : ln1_(d_model, rng), attn_(d_model, n_heads, rng),
      ln2_(d_model, rng), ffn_(d_model, d_ffn, rng) {}

std::vector<std::vector<ValuePtr>> TransformerBlock::forward(
    const std::vector<std::vector<ValuePtr>>& X) const {
    int n = static_cast<int>(X.size());
    int d = static_cast<int>(X[0].size());

    std::vector<std::vector<ValuePtr>> X_norm(n);
    for (int t = 0; t < n; ++t) X_norm[t] = ln1_.forward(X[t]);
    auto attn_out = attn_.forward(X_norm);
    std::vector<std::vector<ValuePtr>> X2(n, std::vector<ValuePtr>(d));
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < d; ++i) X2[t][i] = X[t][i] + attn_out[t][i];

    std::vector<std::vector<ValuePtr>> X3(n, std::vector<ValuePtr>(d));
    for (int t = 0; t < n; ++t) {
        auto norm_x  = ln2_.forward(X2[t]);
        auto ffn_out = ffn_.forward(norm_x);
        for (int i = 0; i < d; ++i) X3[t][i] = X2[t][i] + ffn_out[i];
    }
    return X3;
}

std::vector<ValuePtr> TransformerBlock::parameters() const {
    std::vector<ValuePtr> p = ln1_.parameters();
    auto pa = attn_.parameters(); p.insert(p.end(), pa.begin(), pa.end());
    auto pn = ln2_.parameters();  p.insert(p.end(), pn.begin(), pn.end());
    auto pf = ffn_.parameters();  p.insert(p.end(), pf.begin(), pf.end());
    return p;
}

FTTransformer::FTTransformer(int n_features, int n_classes, int d_model,
                             int n_heads, int d_ffn, int n_blocks, std::mt19937& rng)
    : n_features_(n_features), d_model_(d_model), n_classes_(n_classes),
      feat_W_(n_features, std::vector<ValuePtr>(d_model)),
      feat_b_(n_features, std::vector<ValuePtr>(d_model)),
      cls_(d_model),
      final_ln_(d_model, rng),
      head_W_(n_classes, std::vector<ValuePtr>(d_model)),
      head_b_(n_classes) {
    double limit_feat = xavier_limit(1, d_model);
    std::uniform_real_distribution<double> df(-limit_feat, limit_feat);
    for (int i = 0; i < n_features; ++i)
        for (int j = 0; j < d_model; ++j) {
            feat_W_[i][j] = v(df(rng));
            feat_b_[i][j] = v(df(rng));
        }
    std::uniform_real_distribution<double> dc(-limit_feat, limit_feat);
    for (int j = 0; j < d_model; ++j) cls_[j] = v(dc(rng));

    blocks_.reserve(n_blocks);
    for (int b = 0; b < n_blocks; ++b) blocks_.emplace_back(d_model, n_heads, d_ffn, rng);

    double limit_h = xavier_limit(d_model, n_classes);
    std::uniform_real_distribution<double> dh(-limit_h, limit_h);
    for (int i = 0; i < n_classes; ++i) {
        for (int j = 0; j < d_model; ++j) head_W_[i][j] = v(dh(rng));
        head_b_[i] = v(0.0);
    }
}

std::vector<ValuePtr> FTTransformer::forward(const std::vector<ValuePtr>& x) const {
    int n = n_features_;
    int d = d_model_;

    std::vector<std::vector<ValuePtr>> X(n + 1, std::vector<ValuePtr>(d));
    X[0] = cls_;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            X[i + 1][j] = feat_W_[i][j] * x[i] + feat_b_[i][j];

    for (const auto& block : blocks_) X = block.forward(X);

    auto cls_final = final_ln_.forward(X[0]);

    std::vector<ValuePtr> logits(n_classes_);
    for (int c = 0; c < n_classes_; ++c) {
        ValuePtr acc = head_b_[c];
        for (int j = 0; j < d; ++j) acc = acc + head_W_[c][j] * cls_final[j];
        logits[c] = acc;
    }
    return logits;
}

std::vector<ValuePtr> FTTransformer::parameters() const {
    std::vector<ValuePtr> p;
    for (const auto& row : feat_W_) for (const auto& w : row) p.push_back(w);
    for (const auto& row : feat_b_) for (const auto& w : row) p.push_back(w);
    for (const auto& w : cls_) p.push_back(w);
    for (const auto& block : blocks_) {
        auto bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto lp = final_ln_.parameters();
    p.insert(p.end(), lp.begin(), lp.end());
    for (const auto& row : head_W_) for (const auto& w : row) p.push_back(w);
    for (const auto& w : head_b_) p.push_back(w);
    return p;
}
