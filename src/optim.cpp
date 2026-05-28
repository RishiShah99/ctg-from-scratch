#include "optim.hpp"
#include <cmath>
#include <iostream>

Adam::Adam(std::vector<ValuePtr> params, double lr, double beta1, double beta2,
           double eps, double weight_decay)
    : params_(std::move(params)),
      m_(params_.size(), 0.0),
      v_(params_.size(), 0.0),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      t_(0) {}

void Adam::set_lr(double lr) { lr_ = lr; }

void Adam::zero_grad() {
    for (auto& p : params_) p->grad = 0.0;
}

void Adam::step() {
    ++t_;
    double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
    double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));
    for (size_t i = 0; i < params_.size(); ++i) {
        double g = params_[i]->grad;
        if (weight_decay_ != 0.0) g += weight_decay_ * params_[i]->data;
        // Reject NaN/Inf gradients before they touch the moment buffers.
        // A single non-finite g permanently poisons m_[i] / v_[i] (the
        // EMAs absorb the NaN and never decay back), so subsequent steps
        // on this parameter produce NaN updates regardless of how clean
        // later gradients are. Skip-without-zeroing preserves the prior
        // moment estimates; the next finite gradient updates from those.
        if (!std::isfinite(g)) {
            ++nonfinite_skip_count_;
            if (!warned_nonfinite_) {
                std::cerr << "  WARN: Adam saw non-finite gradient at param "
                          << i << " step " << t_
                          << " — skipping this param's update. Further "
                             "skips will be counted silently; check "
                             "nonfinite_grad_skips() at end of run.\n";
                warned_nonfinite_ = true;
            }
            continue;
        }
        m_[i] = beta1_ * m_[i] + (1.0 - beta1_) * g;
        v_[i] = beta2_ * v_[i] + (1.0 - beta2_) * g * g;
        double m_hat = m_[i] / bc1;
        double v_hat = v_[i] / bc2;
        params_[i]->data -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
}
