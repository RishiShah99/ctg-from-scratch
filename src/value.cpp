#include "value.hpp"
#include <cmath>
#include <unordered_set>

Value::Value(double d) : data(d) {}

Value::Value(double d, std::vector<ValuePtr> children, char o)
    : data(d), prev(std::move(children)), op(o) {}

ValuePtr v(double d) { return std::make_shared<Value>(d); }

void Value::backward() {
    std::vector<ValuePtr> topo;
    std::unordered_set<Value*> seen;

    struct Frame { ValuePtr node; size_t idx; };
    std::vector<Frame> stack;
    stack.push_back({shared_from_this(), 0});

    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.idx == 0) {
            if (seen.count(f.node.get())) { stack.pop_back(); continue; }
            seen.insert(f.node.get());
        }
        if (f.idx < f.node->prev.size()) {
            ValuePtr next = f.node->prev[f.idx];
            f.idx++;
            stack.push_back({next, 0});
        } else {
            topo.push_back(f.node);
            stack.pop_back();
        }
    }

    grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }

    for (auto& node : topo) node->prev.clear();
}

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data + b->data, std::vector<ValuePtr>{a, b}, '+');
    Value* ap = a.get();
    Value* bp = b.get();
    Value* op = out.get();
    out->backward_fn = [ap, bp, op]() {
        ap->grad += op->grad;
        bp->grad += op->grad;
    };
    return out;
}

ValuePtr operator+(const ValuePtr& a, double b) { return a + v(b); }
ValuePtr operator+(double a, const ValuePtr& b) { return v(a) + b; }

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data * b->data, std::vector<ValuePtr>{a, b}, '*');
    Value* ap = a.get();
    Value* bp = b.get();
    Value* op = out.get();
    out->backward_fn = [ap, bp, op]() {
        ap->grad += bp->data * op->grad;
        bp->grad += ap->data * op->grad;
    };
    return out;
}

ValuePtr operator*(const ValuePtr& a, double b) { return a * v(b); }
ValuePtr operator*(double a, const ValuePtr& b) { return v(a) * b; }

ValuePtr operator-(const ValuePtr& a) { return a * -1.0; }
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) { return a + (-b); }
ValuePtr operator-(const ValuePtr& a, double b) { return a + (-b); }
ValuePtr operator-(double a, const ValuePtr& b) { return v(a) + (-b); }

ValuePtr vpow(const ValuePtr& a, double exponent) {
    auto out = std::make_shared<Value>(std::pow(a->data, exponent),
                                       std::vector<ValuePtr>{a}, '^');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op, exponent]() {
        // For fractional exponents the derivative exponent*a^(exponent-1)
        // blows up at a→0 (it's pow(0, negative) = inf). Clamp |a| away
        // from zero with a tiny eps to keep gradients finite. For integer
        // exponents the original expression is well-defined at zero, but
        // the clamp doesn't change values when |a| ≥ eps so it's safe to
        // apply uniformly.
        const double eps = 1e-12;
        double a_safe = ap->data;
        if (std::fabs(a_safe) < eps) a_safe = (a_safe < 0.0 ? -eps : eps);
        ap->grad += exponent * std::pow(a_safe, exponent - 1.0) * op->grad;
    };
    return out;
}

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) { return a * vpow(b, -1.0); }
ValuePtr operator/(const ValuePtr& a, double b) { return a * (1.0 / b); }
ValuePtr operator/(double a, const ValuePtr& b) { return v(a) * vpow(b, -1.0); }

ValuePtr vexp(const ValuePtr& a) {
    double e = std::exp(a->data);
    auto out = std::make_shared<Value>(e, std::vector<ValuePtr>{a}, 'e');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op]() {
        ap->grad += op->data * op->grad;
    };
    return out;
}

ValuePtr vlog(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::log(a->data),
                                       std::vector<ValuePtr>{a}, 'l');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op]() {
        ap->grad += (1.0 / ap->data) * op->grad;
    };
    return out;
}

ValuePtr vtanh(const ValuePtr& a) {
    double t = std::tanh(a->data);
    auto out = std::make_shared<Value>(t, std::vector<ValuePtr>{a}, 't');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op, t]() {
        ap->grad += (1.0 - t * t) * op->grad;
    };
    return out;
}

ValuePtr vrelu(const ValuePtr& a) {
    double r = a->data > 0.0 ? a->data : 0.0;
    auto out = std::make_shared<Value>(r, std::vector<ValuePtr>{a}, 'r');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op]() {
        ap->grad += (op->data > 0.0 ? 1.0 : 0.0) * op->grad;
    };
    return out;
}

ValuePtr vgelu(const ValuePtr& a) {
    const double k0 = 0.7978845608028654;
    const double k1 = 0.044715;
    double x = a->data;
    double x2 = x * x;
    double inner = k0 * (x + k1 * x * x2);
    double t = std::tanh(inner);
    double out_val = 0.5 * x * (1.0 + t);
    auto out = std::make_shared<Value>(out_val, std::vector<ValuePtr>{a}, 'g');
    Value* ap = a.get();
    Value* op = out.get();
    double dt_dx = (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * x2);
    double dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx;
    out->backward_fn = [ap, op, dgelu_dx]() {
        ap->grad += dgelu_dx * op->grad;
    };
    return out;
}

ValuePtr vsigmoid(const ValuePtr& a) {
    double x = a->data;
    double s = x >= 0.0
        ? 1.0 / (1.0 + std::exp(-x))
        : std::exp(x) / (1.0 + std::exp(x));
    auto out = std::make_shared<Value>(s, std::vector<ValuePtr>{a}, 'S');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op, s]() {
        ap->grad += s * (1.0 - s) * op->grad;
    };
    return out;
}

ValuePtr vabs(const ValuePtr& a) {
    double x = a->data;
    double sgn = x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    auto out = std::make_shared<Value>(std::fabs(x), std::vector<ValuePtr>{a}, 'A');
    Value* ap = a.get();
    Value* op = out.get();
    out->backward_fn = [ap, op, sgn]() {
        ap->grad += sgn * op->grad;
    };
    return out;
}

// clamp(x, lo, hi) built from vrelu only — inherits vrelu's tested backward,
// so the subgradient at the box boundaries is whatever vrelu already does.
//   clamp(x, lo, hi) = hi - vrelu((hi - lo) - vrelu(x - lo))
// Verify by cases: x<lo → lo; lo≤x≤hi → x; x>hi → hi.
ValuePtr vclamp(const ValuePtr& x, double lo, double hi) {
    return hi - vrelu((hi - lo) - vrelu(x - lo));
}
