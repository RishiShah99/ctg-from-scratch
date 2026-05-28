#include "value.hpp"
#include "nn.hpp"
#include "data.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <vector>
#include <cstdio>
#ifdef _WIN32
  #include <io.h>
  #define ISATTY_STDOUT() (_isatty(_fileno(stdout)) != 0)
#else
  #include <unistd.h>
  #define ISATTY_STDOUT() (isatty(fileno(stdout)) != 0)
#endif

// Mutable so disable_colors() can blank them out when stdout is not a
// TTY (piped to a file, captured by CI). With the previous const
// literals, piping demo output to a log produced unreadable raw
// escape sequences.
namespace ansi {
    std::string reset  = "\033[0m";
    std::string bold   = "\033[1m";
    std::string dim    = "\033[2m";
    std::string red    = "\033[38;5;203m";
    std::string green  = "\033[38;5;78m";
    std::string yellow = "\033[38;5;221m";
    std::string blue   = "\033[38;5;75m";
    std::string mag    = "\033[38;5;213m";
    std::string cyan   = "\033[38;5;87m";
    std::string gray   = "\033[38;5;243m";
    std::string white  = "\033[38;5;255m";
}

static void maybe_disable_colors() {
    if (ISATTY_STDOUT()) return;
    ansi::reset = ansi::bold = ansi::dim = ansi::red = ansi::green =
        ansi::yellow = ansi::blue = ansi::mag = ansi::cyan =
        ansi::gray = ansi::white = "";
}

static std::vector<ValuePtr> wrap_input(const std::vector<double>& x) {
    std::vector<ValuePtr> out;
    out.reserve(x.size());
    for (double d : x) out.push_back(v(d));
    return out;
}

static std::vector<double> synthesize_fhr(double LB, double MSTV, double MLTV,
                                          int n_accel, int n_decel_light,
                                          int n_decel_severe, int n_decel_prolonged,
                                          uint32_t seed, int n_samples = 200) {
    std::mt19937 rng(seed);
    std::vector<double> fhr(n_samples, LB);

    std::uniform_real_distribution<double> phase_dist(0.0, 6.2831853);
    double phase = phase_dist(rng);
    double ltv_amp = std::max(2.0, MLTV * 0.6);
    for (int i = 0; i < n_samples; ++i)
        fhr[i] += ltv_amp * std::sin(2.0 * 3.14159265 * i / 45.0 + phase);

    std::normal_distribution<double> noise(0.0, std::max(0.2, MSTV * 0.6));
    for (int i = 0; i < n_samples; ++i) fhr[i] += noise(rng);

    auto place = [&](double amp, int w) {
        if (n_samples - w <= 0) return;
        std::uniform_int_distribution<int> pos(0, n_samples - w);
        int p = pos(rng);
        for (int i = 0; i < w; ++i) {
            double t = (i - w * 0.5) / (w * 0.25);
            double bump = amp * std::exp(-t * t);
            if (p + i >= 0 && p + i < n_samples) fhr[p + i] += bump;
        }
    };
    for (int i = 0; i < n_accel; ++i)            place(18.0,  18);
    for (int i = 0; i < n_decel_light; ++i)      place(-12.0, 16);
    for (int i = 0; i < n_decel_severe; ++i)     place(-26.0, 22);
    for (int i = 0; i < n_decel_prolonged; ++i)  place(-32.0, 34);

    return fhr;
}

static void plot_signal(const std::vector<double>& y, int width, int height,
                        double y_min, double y_max) {
    int n = static_cast<int>(y.size());
    std::vector<double> bin(width, 0.0);
    std::vector<int> cnt(width, 0);
    for (int i = 0; i < n; ++i) {
        int c = i * width / n;
        if (c >= width) c = width - 1;
        bin[c] += y[i];
        cnt[c]++;
    }
    for (int c = 0; c < width; ++c) if (cnt[c] > 0) bin[c] /= cnt[c];

    for (int row = height - 1; row >= 0; --row) {
        double row_y_top = y_min + (y_max - y_min) * (row + 1) / height;
        double row_y_bot = y_min + (y_max - y_min) * row / height;

        std::string ylabel = "     ";
        if (row == height - 1)        ylabel = "  170";
        else if (row == 0)            ylabel = "   90";
        else if (row == height / 2)   ylabel = "  130";
        else if (row == height * 3/4) ylabel = "  150";
        else if (row == height / 4)   ylabel = "  110";
        std::cout << ansi::gray << ylabel << " " << "│" << ansi::reset;

        for (int c = 0; c < width; ++c) {
            double v_ = bin[c];
            if (v_ >= row_y_bot && v_ < row_y_top) {
                std::string col;
                if (v_ < 110.0 || v_ > 160.0) col = ansi::red;
                else if (v_ < 120.0 || v_ > 150.0) col = ansi::yellow;
                else col = ansi::green;
                std::cout << col << "●" << ansi::reset;
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << ansi::gray << "      └";
    for (int c = 0; c < width; ++c) std::cout << "─";
    std::cout << "\n      0 min";
    for (int c = 0; c < width - 12; ++c) std::cout << " ";
    std::cout << "30 min" << ansi::reset << "\n";
}

static void plot_activations(const std::vector<double>& vals,
                             const std::string& label,
                             const std::string& color) {
    double mn = *std::min_element(vals.begin(), vals.end());
    double mx = *std::max_element(vals.begin(), vals.end());
    double range = mx - mn;
    if (range < 1e-9) range = 1.0;

    static const char* blocks[] = {" ", "▁", "▂", "▃", "▄",
                                   "▅", "▆", "▇", "█"};

    std::cout << "   " << color << std::left << std::setw(8) << label << ansi::reset
              << ansi::gray << " [" << std::right << std::setw(2) << vals.size() << "] " << ansi::reset;
    for (double val : vals) {
        double norm = (val - mn) / range;
        int level = static_cast<int>(norm * 8.0 + 0.5);
        if (level < 0) level = 0;
        if (level > 8) level = 8;
        std::cout << color << blocks[level] << ansi::reset;
    }
    std::cout << ansi::gray << "  [" << std::fixed << std::setprecision(2)
              << mn << ", " << mx << "]" << ansi::reset << "\n";
}

static double cosine_lr(double lr_max, double lr_min, int t, int T) {
    if (T <= 1) return lr_max;
    double phase = 3.14159265358979323846 * static_cast<double>(t)
                                          / static_cast<double>(T - 1);
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + std::cos(phase));
}

static void clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    double sq = 0.0;
    for (auto& p : params) sq += p->grad * p->grad;
    double norm = std::sqrt(sq);
    if (norm > max_norm && norm > 0.0) {
        double scale = max_norm / norm;
        for (auto& p : params) p->grad *= scale;
    }
}

static void train_mlp(MLP& net, std::vector<ValuePtr>& params,
                      const Dataset& ds, uint32_t seed) {
    const int epochs     = 40;
    const int batch_size = 16;
    const double lr_max  = 0.003;
    const double lr_min  = 1e-5;
    const double wd      = 1e-4;

    std::mt19937 rng(seed);
    Adam optim(params, lr_max, 0.9, 0.999, 1e-8, wd);
    std::vector<size_t> order(ds.X_train.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> cw(3, 1.0);

    std::cout << ansi::gray << "  training Adam-MLP (40 epochs, ~80s)" << ansi::reset << "\n";
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optim.set_lr(cosine_lr(lr_max, lr_min, epoch - 1, epochs));
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t bi = 0; bi < order.size(); bi += batch_size) {
            size_t bsz = std::min(static_cast<size_t>(batch_size), order.size() - bi);
            optim.zero_grad();
            for (size_t j = 0; j < bsz; ++j) {
                size_t k = order[bi + j];
                auto logits = net.forward(wrap_input(ds.X_train[k]));
                auto probs  = softmax(logits);
                auto loss   = cross_entropy_weighted(probs, ds.y_train[k], cw);
                loss->backward();
            }
            double scale = 1.0 / static_cast<double>(bsz);
            for (auto& p : params) p->grad *= scale;
            clip_grad_norm(params, 1.0);
            optim.step();
        }
        if (epoch % 5 == 0) std::cout << "  " << ansi::gray << "epoch " << epoch
                                       << "/" << epochs << ansi::reset << "\n";
    }
}

static int argmax_v(const std::vector<ValuePtr>& xs) {
    int best = 0;
    for (size_t i = 1; i < xs.size(); ++i)
        if (xs[i]->data > xs[best]->data) best = static_cast<int>(i);
    return best;
}

static void render_example(int example_idx, const Dataset& ds, MLP& net) {
    const auto& feat_names = ds.feat_names;
    const auto& x_norm = ds.X_test[example_idx];
    int y_true = ds.y_test[example_idx];
    const char* names[] = {"Normal", "Suspect", "Pathological"};
    std::string class_color[] = {ansi::green, ansi::yellow, ansi::red};

    std::vector<double> x_raw(x_norm.size());
    for (size_t j = 0; j < x_norm.size(); ++j)
        x_raw[j] = x_norm[j] * ds.feat_std[j] + ds.feat_mean[j];

    auto idx_of = [&](const std::string& s) -> int {
        for (size_t i = 0; i < feat_names.size(); ++i) if (feat_names[i] == s) return static_cast<int>(i);
        return -1;
    };
    double LB   = x_raw[idx_of("LB")];
    double AC   = x_raw[idx_of("AC")];
    double DL   = x_raw[idx_of("DL")];
    double DS   = x_raw[idx_of("DS")];
    double DP   = x_raw[idx_of("DP")];
    double MSTV = x_raw[idx_of("MSTV")];
    double MLTV = x_raw[idx_of("MLTV")];

    int n_acc = std::max(0, static_cast<int>(AC * 1500.0));
    int n_dl  = std::max(0, static_cast<int>(DL * 1500.0));
    int n_ds  = std::max(0, static_cast<int>(DS * 1500.0));
    int n_dp  = std::max(0, static_cast<int>(DP * 1500.0));

    auto fhr = synthesize_fhr(LB, MSTV, MLTV, n_acc, n_dl, n_ds, n_dp,
                              static_cast<uint32_t>(42 + example_idx));

    std::cout << "\n" << ansi::bold << ansi::white
              << "  test patient #" << std::setw(3) << example_idx
              << ansi::reset << "    true label: "
              << class_color[y_true] << ansi::bold << names[y_true] << ansi::reset << "\n\n";

    std::cout << ansi::gray << "  synthesized FHR trace (bpm vs time):" << ansi::reset << "\n";
    plot_signal(fhr, 70, 12, 90.0, 175.0);

    std::cout << "\n" << ansi::gray << "  22 extracted features:" << ansi::reset << "\n";
    auto print_feat = [&](const std::string& n) {
        int i = idx_of(n);
        if (i < 0) return;
        std::cout << "    " << ansi::cyan << std::left << std::setw(8) << n << ansi::reset
                  << "= " << ansi::white << std::right << std::setw(7)
                  << std::fixed << std::setprecision(2) << x_raw[i] << ansi::reset << "  ";
    };
    print_feat("LB");   print_feat("AC");   print_feat("FM");    std::cout << "\n";
    print_feat("UC");   print_feat("DL");   print_feat("DS");    std::cout << "\n";
    print_feat("DP");   print_feat("ASTV"); print_feat("MSTV");  std::cout << "\n";
    print_feat("ALTV"); print_feat("MLTV"); print_feat("Width"); std::cout << "\n";
    print_feat("Min");  print_feat("Max");  print_feat("Mode");  std::cout << "\n";
    print_feat("Mean"); print_feat("Median"); print_feat("Variance"); std::cout << "\n";

    std::cout << "\n" << ansi::gray << "  MLP forward pass (showing intermediate activations):" << ansi::reset << "\n\n";

    auto x_vals = wrap_input(x_norm);
    std::vector<double> x_d(x_vals.size());
    for (size_t i = 0; i < x_vals.size(); ++i) x_d[i] = x_vals[i]->data;
    plot_activations(x_d, "input", ansi::cyan);

    auto h = x_vals;
    int layer_idx = 0;
    for (const auto& layer : net.layers) {
        h = layer.forward(h);
        std::vector<double> hd(h.size());
        for (size_t i = 0; i < h.size(); ++i) hd[i] = h[i]->data;
        std::string label = "layer" + std::to_string(++layer_idx);
        std::string col   = (layer_idx == static_cast<int>(net.layers.size())) ? ansi::mag : ansi::blue;
        plot_activations(hd, label, col);
    }

    auto probs = softmax(h);
    std::vector<double> p_d(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) p_d[i] = probs[i]->data;
    int pred = argmax_v(probs);

    std::cout << "\n" << ansi::gray << "  softmax probabilities:" << ansi::reset << "\n";
    for (int c = 0; c < 3; ++c) {
        std::cout << "    " << class_color[c] << std::left << std::setw(14) << names[c] << ansi::reset;
        int bar_len = static_cast<int>(p_d[c] * 40.0 + 0.5);
        std::cout << class_color[c];
        for (int b = 0; b < bar_len; ++b) std::cout << "█";
        std::cout << ansi::reset << " " << std::fixed << std::setprecision(3) << p_d[c];
        if (c == pred) std::cout << "  " << ansi::bold << "← predicted" << ansi::reset;
        std::cout << "\n";
    }

    bool correct = (pred == y_true);
    std::cout << "\n  "
              << (correct ? ansi::green + std::string("✓  correct")
                          : ansi::red   + std::string("✗  wrong"))
              << ansi::reset << "  (predicted "
              << class_color[pred] << names[pred] << ansi::reset
              << ", true " << class_color[y_true] << names[y_true] << ansi::reset << ")\n";
}

int main(int argc, char** argv) {
    const std::string csv_path = (argc > 1) ? argv[1] : "data/CTG.csv";
    const uint32_t seed = 42;
    const std::vector<int> arch = {22, 32, 16, 3};

    maybe_disable_colors();

    std::cout << "\n" << ansi::bold << ansi::white
              << "  CTG-from-scratch  ·  pure C++ neural network demo"
              << ansi::reset << "\n";
    std::cout << ansi::gray
              << "  ============================================================"
              << ansi::reset << "\n";

    auto ds = load_ctg(csv_path, 0.15, 0.20, seed);
    std::mt19937 rng(seed);
    MLP net(arch, rng);
    auto params = net.parameters();

    train_mlp(net, params, ds, seed);

    int idx_n = -1, idx_s = -1, idx_p = -1;
    for (size_t i = 0; i < ds.y_test.size(); ++i) {
        if (idx_n < 0 && ds.y_test[i] == 0) idx_n = static_cast<int>(i);
        if (idx_s < 0 && ds.y_test[i] == 1) idx_s = static_cast<int>(i);
        if (idx_p < 0 && ds.y_test[i] == 2) idx_p = static_cast<int>(i);
    }

    render_example(idx_n, ds, net);
    std::cout << "\n" << ansi::gray
              << "  ------------------------------------------------------------"
              << ansi::reset << "\n";
    render_example(idx_s, ds, net);
    std::cout << "\n" << ansi::gray
              << "  ------------------------------------------------------------"
              << ansi::reset << "\n";
    render_example(idx_p, ds, net);

    std::cout << "\n";
    return 0;
}
