#include "nn.hpp"
#include <cmath>

Neuron::Neuron(int in_features, int fan_out_hint, bool nonlin_, std::mt19937& rng)
    : nonlin(nonlin_) {
    double limit = std::sqrt(6.0 / static_cast<double>(in_features + fan_out_hint));
    std::uniform_real_distribution<double> dist(-limit, limit);
    weights.reserve(in_features);
    for (int i = 0; i < in_features; ++i) weights.push_back(v(dist(rng)));
    bias = v(0.0);
}

ValuePtr Neuron::forward(const std::vector<ValuePtr>& x) const {
    ValuePtr acc = bias;
    for (size_t i = 0; i < weights.size(); ++i) acc = acc + weights[i] * x[i];
    return nonlin ? vtanh(acc) : acc;
}

std::vector<ValuePtr> Neuron::parameters() const {
    std::vector<ValuePtr> params = weights;
    params.push_back(bias);
    return params;
}

Layer::Layer(int in_features, int out_features, bool nonlin, std::mt19937& rng) {
    neurons.reserve(out_features);
    for (int i = 0; i < out_features; ++i)
        neurons.emplace_back(in_features, out_features, nonlin, rng);
}

std::vector<ValuePtr> Layer::forward(const std::vector<ValuePtr>& x) const {
    std::vector<ValuePtr> out;
    out.reserve(neurons.size());
    for (const auto& n : neurons) out.push_back(n.forward(x));
    return out;
}

std::vector<ValuePtr> Layer::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto& n : neurons) {
        auto p = n.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

MLP::MLP(const std::vector<int>& sizes, std::mt19937& rng) {
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bool nonlin = (i + 2 < sizes.size());
        layers.emplace_back(static_cast<int>(sizes[i]),
                            static_cast<int>(sizes[i + 1]),
                            nonlin, rng);
    }
}

std::vector<ValuePtr> MLP::forward(const std::vector<ValuePtr>& x) const {
    std::vector<ValuePtr> h = x;
    for (const auto& layer : layers) h = layer.forward(h);
    return h;
}

std::vector<ValuePtr> MLP::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto& layer : layers) {
        auto p = layer.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

void MLP::zero_grad() {
    for (auto& p : parameters()) p->grad = 0.0;
}
