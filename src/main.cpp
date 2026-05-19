#include "value.hpp"
#include "nn.hpp"
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    std::mt19937 rng(42);
    MLP net({22, 16, 16, 3}, rng);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<ValuePtr> x;
    x.reserve(22);
    for (int i = 0; i < 22; ++i) x.push_back(v(dist(rng)));

    auto logits = net.forward(x);
    auto params = net.parameters();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "params = " << params.size() << "\n";
    std::cout << "logits = ";
    for (const auto& l : logits) std::cout << l->data << " ";
    std::cout << "\n";

    auto loss = logits[0] * logits[0] + logits[1] * logits[1] + logits[2] * logits[2];
    net.zero_grad();
    loss->backward();

    std::cout << "loss = " << loss->data << "\n";
    std::cout << "w0.grad = " << params[0]->grad << "\n";
    std::cout << "b_last.grad = " << params.back()->grad << "\n";
    return 0;
}
