#include "value.hpp"
#include <cmath>
#include <unordered_set>

Value::Value(double d) : data(d) {}

Value::Value(double d, std::vector<ValuePtr> children, char o)
    : data(d), prev(std::move(children)), op(o) {}

ValuePtr v(double d) { return std::make_shared<Value>(d); }

static void build_topo(const ValuePtr& node,
                       std::unordered_set<Value*>& seen,
                       std::vector<ValuePtr>& order) {
    if (seen.count(node.get())) return;
    seen.insert(node.get());
    for (const auto& child : node->prev) build_topo(child, seen, order);
    order.push_back(node);
}

void Value::backward() {
    std::vector<ValuePtr> topo;
    std::unordered_set<Value*> seen;
    build_topo(shared_from_this(), seen, topo);
    grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
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
        ap->grad += exponent * std::pow(ap->data, exponent - 1.0) * op->grad;
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
