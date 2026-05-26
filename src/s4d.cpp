#include "s4d.hpp"
#include <cmath>

static constexpr double PI = 3.14159265358979323846;

S4DLayer::S4DLayer(int H_, int N_, double dt_, std::mt19937& rng)
    : H(H_), N(N_), dt(dt_),
      A_bar(H_, std::vector<std::complex<double>>(N_)),
      B(H_, std::vector<ValuePtr>(N_)),
      C(H_, std::vector<ComplexValue>(N_)),
      D(H_)
{
    std::normal_distribution<double> n01(0.0, 1.0);
    double bscale = 1.0 / std::sqrt(static_cast<double>(N_));
    double cscale = 1.0 / std::sqrt(static_cast<double>(N_));

    for (int h = 0; h < H_; ++h) {
        for (int n = 0; n < N_; ++n) {
            std::complex<double> A_n(-0.5, PI * static_cast<double>(n));
            A_bar[h][n] = std::exp(dt_ * A_n);

            B[h][n] = v(bscale * n01(rng));
            C[h][n] = ComplexValue(cscale * n01(rng), cscale * n01(rng));
        }
        D[h] = v(1.0);
    }
}

std::vector<std::vector<ValuePtr>> S4DLayer::forward(
    const std::vector<std::vector<ValuePtr>>& x) const
{
    // Discretization is HYBRID, on purpose:
    //   A_bar = exp(dt · A)                — Zero-Order Hold on A (set in ctor)
    //   B is used as `dt · B` per step      — Explicit Euler on the input term
    // Strict ZOH would use B_bar = A^{-1}(A_bar - I)B, but the Euler form is
    // simpler, numerically well-behaved for the small dt used here (0.05),
    // and is what the trainer was producing — switching now would invalidate
    // every existing checkpoint. Documented; not changed.
    //
    // `dt * B[ch][n]` used to allocate a new Value inside every inner-loop
    // iteration: L * H * N transient nodes per forward. Lift it: B is a
    // learnable parameter, so Bdt must be recomputed at the START of each
    // forward (Adam updates B between calls), but only ONCE per forward
    // instead of L times.
    int L = static_cast<int>(x.size());
    std::vector<std::vector<ValuePtr>> y(L, std::vector<ValuePtr>(H));

    std::vector<std::vector<ValuePtr>> Bdt(H, std::vector<ValuePtr>(N));
    for (int ch = 0; ch < H; ++ch)
        for (int n = 0; n < N; ++n) Bdt[ch][n] = dt * B[ch][n];

    std::vector<std::vector<ComplexValue>> h_state(
        H, std::vector<ComplexValue>(N, ComplexValue(0.0, 0.0)));

    for (int t = 0; t < L; ++t) {
        for (int ch = 0; ch < H; ++ch) {
            ValuePtr u = x[t][ch];
            ValuePtr acc = v(0.0);
            for (int n = 0; n < N; ++n) {
                h_state[ch][n] = A_bar[ch][n] * h_state[ch][n]
                               + Bdt[ch][n] * u;
                ComplexValue Ch = C[ch][n] * h_state[ch][n];
                acc = acc + Ch.re;
            }
            y[t][ch] = 2.0 * acc + D[ch] * u;
        }
    }
    return y;
}

std::vector<ValuePtr> S4DLayer::parameters() const {
    std::vector<ValuePtr> ps;
    ps.reserve(H * (2 * N + N + 1));
    for (int h = 0; h < H; ++h) {
        for (int n = 0; n < N; ++n) {
            ps.push_back(B[h][n]);
            ps.push_back(C[h][n].re);
            ps.push_back(C[h][n].im);
        }
        ps.push_back(D[h]);
    }
    return ps;
}
