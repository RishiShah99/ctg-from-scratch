#include "value.hpp"
#include <iostream>
#include <iomanip>

int main() {
    auto a = v(-4.0);
    auto b = v(2.0);
    auto c = a + b;
    auto d = a * b + vpow(b, 3.0);
    c = c + c + v(1.0);
    c = c + v(1.0) + c + (-a);
    d = d + d * v(2.0) + vrelu(b + a);
    d = d + v(3.0) * d + vrelu(b - a);
    auto e = c - d;
    auto f = vpow(e, 2.0);
    auto g = f / v(2.0);
    g = g + v(10.0) / f;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "g.data = " << g->data << "\n";
    g->backward();
    std::cout << "a.grad = " << a->grad << "\n";
    std::cout << "b.grad = " << b->grad << "\n";
    return 0;
}
