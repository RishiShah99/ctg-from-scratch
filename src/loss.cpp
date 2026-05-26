#include "loss.hpp"

std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits) {
    // Numerical-stability shift: softmax(l - max) = softmax(l). The K
    // subtractions used to allocate K transient `v(-m)` Values per
    // softmax call via operator-(ValuePtr, double) → `a + v(b)`. Hoist
    // the negated-max Value once and reuse it so we allocate one
    // constant per call, not K.
    double m = logits[0]->data;
    for (const auto& l : logits) if (l->data > m) m = l->data;
    ValuePtr neg_m = v(-m);

    std::vector<ValuePtr> exps;
    exps.reserve(logits.size());
    for (const auto& l : logits) exps.push_back(vexp(l + neg_m));

    ValuePtr s = exps[0];
    for (size_t i = 1; i < exps.size(); ++i) s = s + exps[i];

    std::vector<ValuePtr> probs;
    probs.reserve(exps.size());
    for (const auto& e : exps) probs.push_back(e / s);
    return probs;
}

ValuePtr cross_entropy(const std::vector<ValuePtr>& probs, int label) {
    return -vlog(probs[static_cast<size_t>(label)]);
}

ValuePtr cross_entropy_weighted(const std::vector<ValuePtr>& probs,
                                int label,
                                const std::vector<double>& class_weights) {
    return class_weights[static_cast<size_t>(label)]
         * (-vlog(probs[static_cast<size_t>(label)]));
}
