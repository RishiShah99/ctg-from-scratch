#pragma once
#include <vector>
#include <memory>
#include <functional>

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad = 0.0;
    std::vector<ValuePtr> prev;
    std::function<void()> backward_fn;
    char op = ' ';

    explicit Value(double d);
    Value(double d, std::vector<ValuePtr> children, char o);

    void backward();
};

ValuePtr v(double d);

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator+(const ValuePtr& a, double b);
ValuePtr operator+(double a, const ValuePtr& b);

ValuePtr operator-(const ValuePtr& a);
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, double b);
ValuePtr operator-(double a, const ValuePtr& b);

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, double b);
ValuePtr operator*(double a, const ValuePtr& b);

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, double b);
ValuePtr operator/(double a, const ValuePtr& b);

ValuePtr vpow(const ValuePtr& a, double exponent);
ValuePtr vexp(const ValuePtr& a);
ValuePtr vlog(const ValuePtr& a);
ValuePtr vtanh(const ValuePtr& a);
ValuePtr vrelu(const ValuePtr& a);
