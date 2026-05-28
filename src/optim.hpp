#pragma once
#include "value.hpp"
#include <vector>

class Adam {
public:
    Adam(std::vector<ValuePtr> params,
         double lr,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0);

    void set_lr(double lr);
    void zero_grad();
    void step();

    // Diagnostic accessor: number of (param, step) pairs whose update was
    // skipped because the gradient was non-finite. Stays at 0 on healthy
    // training runs; a non-zero value means at least one gradient blew up
    // and Adam refused to poison m_/v_ with NaN/Inf.
    long long nonfinite_grad_skips() const { return nonfinite_skip_count_; }

private:
    std::vector<ValuePtr> params_;
    std::vector<double>   m_;
    std::vector<double>   v_;
    double lr_;
    double beta1_, beta2_, eps_, weight_decay_;
    long long t_;
    long long nonfinite_skip_count_ = 0;
    bool      warned_nonfinite_     = false;
};
