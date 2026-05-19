#pragma once
#include "value.hpp"
#include <vector>
#include <random>

class Neuron {
public:
    std::vector<ValuePtr> weights;
    ValuePtr bias;
    bool nonlin;

    Neuron(int in_features, int fan_out_hint, bool nonlin, std::mt19937& rng);
    ValuePtr forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int in_features, int out_features, bool nonlin, std::mt19937& rng);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& sizes, std::mt19937& rng);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
    void zero_grad();
};
