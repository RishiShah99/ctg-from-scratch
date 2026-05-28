#include "value.hpp"
#include "cgm_data.hpp"
#include "cgm_net.hpp"
#include "osdn_blob.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <string>

namespace c {
    const std::string reset  = "\033[0m";
    const std::string bold   = "\033[1m";
    const std::string dim    = "\033[2m";
    const std::string red    = "\033[38;5;203m";
    const std::string green  = "\033[38;5;78m";
    const std::string amber  = "\033[38;5;221m";
    const std::string blue   = "\033[38;5;75m";
    const std::string cyan   = "\033[38;5;87m";
    const std::string gray   = "\033[38;5;243m";
    const std::string white  = "\033[38;5;255m";
    const std::string teal   = "\033[38;5;80m";
    const std::string mag    = "\033[38;5;177m";
}

struct Args {
    std::string arch = "osdn";   // brain-mode story is OSDN-only by default
    std::string csv = "data/ohio_t1dm.csv";
    std::string s4d_weights;     // optional; only loaded when --arch s4d|both
    std::string osdn_weights = "results/osdn-brain-ohio-disjoint.weights.bin.best";
    int H = 16, N = 8;
    int lookback = 144, horizon = 12, step_min = 5;
    int post_bolus = 240;
    double hypo = 70.0;
    double dt = 0.05;
    int window_stride = 12;
    int n_show = 4;
    int score_max = 200;
    int osdn_layers = 1;
    int input_channels = 7;
    uint32_t seed = 42;
    std::string val_ids_csv  = "584,588";
    std::string test_ids_csv = "591,596";
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--arch") a.arch = next();
        else if (k == "--csv") a.csv = next();
        else if (k == "--load-weights-s4d") a.s4d_weights = next();
        else if (k == "--load-weights-osdn") a.osdn_weights = next();
        else if (k == "--H") a.H = std::stoi(next());
        else if (k == "--N") a.N = std::stoi(next());
        else if (k == "--lookback") a.lookback = std::stoi(next());
        else if (k == "--horizon") a.horizon = std::stoi(next());
        else if (k == "--step-min") a.step_min = std::stoi(next());
        else if (k == "--post-bolus") a.post_bolus = std::stoi(next());
        else if (k == "--hypo") a.hypo = std::stod(next());
        else if (k == "--dt") a.dt = std::stod(next());
        else if (k == "--window-stride") a.window_stride = std::stoi(next());
        else if (k == "--n-show") a.n_show = std::stoi(next());
        else if (k == "--score-max") a.score_max = std::stoi(next());
        else if (k == "--osdn-layers") a.osdn_layers = std::stoi(next());
        else if (k == "--input-channels") a.input_channels = std::stoi(next());
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (k == "--val-ids") a.val_ids_csv = next();
        else if (k == "--test-ids") a.test_ids_csv = next();
        else { std::cerr << "unknown arg: " << k << "\n"; std::exit(2); }
    }
    return a;
}

// Strict header-validating loader. Returns true on success, false on any
// shape / version / magic / size mismatch — with the mismatch table printed
// to stderr. The caller decides whether to abort or continue without this
// model.
static bool load_weights_strict(Net& net, const std::string& path,
                                uint32_t H, uint32_t K, uint32_t D_in,
                                uint32_t n_layers, const std::string& tag)
{
    std::ifstream wf(path, std::ios::binary);
    if (!wf.is_open()) {
        std::cerr << "  [" << tag << "] cannot open " << path << "\n";
        return false;
    }
    osdn_blob::Header h{};
    std::string err;
    if (!osdn_blob::read_header(wf, h, err)) {
        std::cerr << "  [" << tag << "] " << path << ": " << err << "\n";
        return false;
    }
    std::string shape_err;
    if (!osdn_blob::shape_matches(h, H, K, D_in, n_layers, shape_err)) {
        std::cerr << "  [" << tag << "] " << path
                  << ": shape mismatch\n" << shape_err;
        return false;
    }
    auto params = net.parameters();
    if (h.param_count != params.size()) {
        std::cerr << "  [" << tag << "] " << path
                  << ": header param_count " << h.param_count
                  << " ≠ build expected " << params.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < params.size(); ++i) {
        float fv = 0.0f;
        if (!wf.read(reinterpret_cast<char*>(&fv), sizeof(float))) {
            std::cerr << "  [" << tag << "] " << path
                      << ": short read at param " << i << "/" << params.size() << "\n";
            return false;
        }
        params[i]->data = static_cast<double>(fv);
    }
    return true;
}

static double sig(double z) { return 1.0 / (1.0 + std::exp(-z)); }

static std::vector<std::string> split_ids(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else if (!std::isspace(static_cast<unsigned char>(c))) cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static void plot_cgm(const std::vector<double>& bg, double hypo_thr) {
    const int W = 72, Hh = 8;
    const double y_min = 40.0, y_max = 280.0;
    std::vector<int> col_top(W, -1), col_bot(W, -1);
    int N_pts = static_cast<int>(bg.size());
    for (int i = 0; i < N_pts; ++i) {
        int c = i * W / N_pts;
        if (c >= W) c = W - 1;
        double y = bg[i];
        int row = static_cast<int>((y_max - y) / (y_max - y_min) * (Hh * 2));
        row = std::clamp(row, 0, Hh * 2 - 1);
        if (col_top[c] < 0 || row < col_top[c]) col_top[c] = row;
        if (col_bot[c] < 0 || row > col_bot[c]) col_bot[c] = row;
    }
    int hypo_row = static_cast<int>((y_max - hypo_thr) / (y_max - y_min) * (Hh * 2));
    int hyper_row = static_cast<int>((y_max - 180.0) / (y_max - y_min) * (Hh * 2));

    for (int row = 0; row < Hh; ++row) {
        std::cout << "  ";
        int rr1 = row * 2, rr2 = row * 2 + 1;
        double y_label = y_max - (rr1 + 0.5) / (Hh * 2) * (y_max - y_min);
        std::cout << c::gray << std::setw(3) << static_cast<int>(y_label) << " │" << c::reset;

        for (int x = 0; x < W; ++x) {
            int rt = col_top[x], rb = col_bot[x];
            bool hit_top = (rt >= rr1 && rt <= rr2);
            bool hit_bot = (rb >= rr1 && rb <= rr2);
            bool hypo_band = (rr1 <= hypo_row && hypo_row <= rr2);
            bool hyper_band = (rr1 <= hyper_row && hyper_row <= rr2);

            double y_at = y_max - ((rr1 + rr2) * 0.5) / (Hh * 2) * (y_max - y_min);
            std::string color = c::cyan;
            if (y_at < hypo_thr)      color = c::red;
            else if (y_at > 180.0)    color = c::amber;

            if (hit_top || hit_bot)         std::cout << color << "●" << c::reset;
            else if (hypo_band || hyper_band) std::cout << c::gray << "·" << c::reset;
            else                              std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout << "      " << c::gray;
    for (int x = 0; x < W; ++x) std::cout << "─";
    std::cout << c::reset << "\n";
    std::cout << "      " << c::gray << "-12h" << std::string(W - 7, ' ') << "now" << c::reset << "\n";
}

static void plot_prob_row(const std::string& label, const std::string& color_tag,
                          double prob, int true_label, double inf_ms)
{
    const int bar_w = 36;
    int filled = static_cast<int>(prob * bar_w);
    std::string col = prob >= 0.5 ? c::red : c::green;
    std::cout << "  " << color_tag << std::setw(5) << std::left << label << c::reset << " ";
    for (int i = 0; i < bar_w; ++i) std::cout << (i < filled ? col + "█" : c::gray + "·");
    std::cout << c::reset << "  " << col << std::fixed << std::setprecision(3) << prob << c::reset;
    bool right = (prob >= 0.5) == (true_label == 1);
    std::string verdict_col = right ? c::green : c::red;
    std::string mark = right ? "✓" : "✗";
    std::cout << "  " << verdict_col << mark << c::reset
              << c::gray << "   " << std::setprecision(1) << inf_ms << " ms" << c::reset << "\n";
}

struct ScoredWindow {
    int idx;
    double s4d_logit;
    double osdn_logit;
};

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::cout << "\n" << c::bold << c::white
              << "  CGM-DEMO  ·  pure-C++ hypoglycemia prediction on real Ohio T1DM"
              << c::reset << "\n";
    std::cout << "  " << c::gray
              << "============================================================"
              << c::reset << "\n";

    auto records = load_cgm_csv(a.csv);
    PatientSplitPolicy policy;
    policy.val_ids  = split_ids(a.val_ids_csv);
    policy.test_ids = split_ids(a.test_ids_csv);
    CGMDataset ds;
    try {
        ds = make_windows(records, a.lookback, a.horizon, a.step_min,
                          a.hypo, a.post_bolus, policy, a.seed, a.window_stride);
    } catch (const std::exception& e) {
        std::cerr << "split error: " << e.what() << "\n";
        return 2;
    }
    std::vector<std::vector<double>> raw_lookback;
    raw_lookback.reserve(ds.test.size());
    for (const auto& w : ds.test) raw_lookback.push_back(w.lookback);
    NormStats norm_stats{};
    normalize_inplace(ds, norm_stats);
    const double mean = norm_stats.cgm_mean;
    const double std_ = norm_stats.cgm_std;
    std::cout << "  " << c::gray << "loaded " << records.size() << " patients · "
              << ds.test.size() << " test windows · z=(" << std::fixed << std::setprecision(2)
              << mean << "," << std_ << ")" << c::reset << "\n";

    bool want_s4d  = (a.arch == "s4d"  || a.arch == "both");
    bool want_osdn = (a.arch == "osdn" || a.arch == "both");

    std::mt19937 rng_s(123);
    std::mt19937 rng_o(456);
    std::unique_ptr<Net> s4d_net, osdn_net;
    if (want_s4d) {
        if (a.s4d_weights.empty()) {
            std::cerr << "  [S4D]  no --load-weights-s4d given; skipping S4D\n";
            want_s4d = false;
        } else {
            s4d_net = std::make_unique<Net>("s4d", a.H, a.N, a.dt, 1,
                                            a.input_channels, rng_s);
            if (!load_weights_strict(*s4d_net, a.s4d_weights,
                                     a.H, a.N, a.input_channels, 1, "S4D")) {
                want_s4d = false;
            } else {
                std::cout << "  " << c::blue << "S4D" << c::reset << c::gray
                          << "  loaded " << s4d_net->parameters().size()
                          << " params from " << a.s4d_weights << c::reset << "\n";
            }
        }
    }
    if (want_osdn) {
        osdn_net = std::make_unique<Net>("osdn", a.H, a.N, a.dt, a.osdn_layers,
                                         a.input_channels, rng_o);
        if (!load_weights_strict(*osdn_net, a.osdn_weights,
                                 a.H, a.N, a.input_channels, a.osdn_layers, "OSDN")) {
            want_osdn = false;
        } else {
            std::cout << "  " << c::mag << "OSDN" << c::reset << c::gray
                      << " loaded " << osdn_net->parameters().size()
                      << " params from " << a.osdn_weights
                      << " (" << a.osdn_layers << " layer" << (a.osdn_layers > 1 ? "s)" : ")")
                      << c::reset << "\n";
        }
    }
    if (!want_s4d && !want_osdn) {
        std::cerr << "  no models loaded — abort\n";
        return 1;
    }

    int n_score = std::min<int>(a.score_max, static_cast<int>(ds.test.size()));
    std::cout << "  " << c::gray << "scoring " << n_score << " of " << ds.test.size()
              << " test windows to curate examples..." << c::reset << "\n";
    std::vector<ScoredWindow> scored;
    scored.reserve(n_score);
    for (int i = 0; i < n_score; ++i) {
        ScoredWindow s;
        s.idx = i;
        s.s4d_logit  = want_s4d  ? s4d_net->forward(ds.test[i]).final_logit->data  : 0.0;
        s.osdn_logit = want_osdn ? osdn_net->forward(ds.test[i]).final_logit->data : 0.0;
        scored.push_back(s);
    }

    auto pick = [&](bool want_hypo, bool want_both_agree) -> int {
        int best_idx = -1;
        double best_score = want_hypo ? -1e9 : 1e9;
        for (const auto& s : scored) {
            int lab = ds.test[s.idx].label;
            if (want_hypo && lab != 1) continue;
            if (!want_hypo && lab != 0) continue;
            // "Both agree" pick: when hunting a HYPO window we want both p_hypo
            // high, so min(p_s4d, p_osdn) maximises = both are high. When hunting
            // a SAFE window we want both p_hypo low, so max(p_s4d, p_osdn)
            // minimises = both are low. Using min in both branches would let
            // a single-model dissenter satisfy the "both agree safe" claim.
            double both = want_s4d && want_osdn
                ? (want_hypo ? std::min(sig(s.s4d_logit), sig(s.osdn_logit))
                             : std::max(sig(s.s4d_logit), sig(s.osdn_logit)))
                : sig(want_s4d ? s.s4d_logit : s.osdn_logit);
            double avg = want_s4d && want_osdn
                ? 0.5 * (sig(s.s4d_logit) + sig(s.osdn_logit))
                : sig(want_s4d ? s.s4d_logit : s.osdn_logit);
            double score = want_both_agree ? both : avg;
            if (want_hypo  && score > best_score) { best_score = score; best_idx = s.idx; }
            if (!want_hypo && score < best_score) { best_score = score; best_idx = s.idx; }
        }
        return best_idx;
    };

    std::vector<int> show_idx;
    int idx_hypo = pick(true,  true);   if (idx_hypo  >= 0) show_idx.push_back(idx_hypo);
    int idx_safe = pick(false, true);   if (idx_safe  >= 0) show_idx.push_back(idx_safe);
    int idx_post_bolus = -1;
    for (const auto& s : scored) {
        if (ds.test[s.idx].label == 1 && ds.test[s.idx].in_post_bolus_window) {
            idx_post_bolus = s.idx; break;
        }
    }
    if (idx_post_bolus >= 0) show_idx.push_back(idx_post_bolus);
    if ((int)show_idx.size() < a.n_show) {
        for (int j = 0; j < (int)scored.size() && (int)show_idx.size() < a.n_show; ++j) {
            if (std::find(show_idx.begin(), show_idx.end(), j) == show_idx.end())
                show_idx.push_back(j);
        }
    }
    if ((int)show_idx.size() > a.n_show) show_idx.resize(a.n_show);

    const char* labels[] = {"clear hypo (both models agree)",
                            "clear safe (both models agree)",
                            "post-bolus hypo (the hard case)",
                            "additional window"};

    for (size_t k = 0; k < show_idx.size(); ++k) {
        int i = show_idx[k];
        const auto& w = ds.test[i];
        std::cout << "\n" << c::bold << c::white
                  << "  window #" << (k + 1) << "  ·  " << labels[std::min(k, sizeof(labels)/sizeof(*labels) - 1)]
                  << c::reset << "\n";
        std::cout << "  " << c::gray << "patient " << w.patient_id
                  << "   t_anchor=" << w.t_anchor_min << " min"
                  << "   future_min_glucose=" << std::fixed << std::setprecision(1) << w.future_min_glucose
                  << " mg/dL   true=" << (w.label == 1 ? "hypo" : "safe")
                  << (w.in_post_bolus_window ? "   [post-bolus window]" : "")
                  << c::reset << "\n\n";

        std::cout << "  " << c::gray << "12-hour CGM trace (mg/dL):" << c::reset << "\n";
        plot_cgm(raw_lookback[i], a.hypo);
        std::cout << "\n";

        if (want_s4d) {
            auto t0 = std::chrono::steady_clock::now();
            double z = s4d_net->forward(w).final_logit->data;
            auto t1 = std::chrono::steady_clock::now();
            double ms = 1000.0 * std::chrono::duration<double>(t1 - t0).count();
            plot_prob_row("S4D", c::blue, sig(z), w.label, ms);
        }
        if (want_osdn) {
            auto t0 = std::chrono::steady_clock::now();
            double z = osdn_net->forward(w).final_logit->data;
            auto t1 = std::chrono::steady_clock::now();
            double ms = 1000.0 * std::chrono::duration<double>(t1 - t0).count();
            plot_prob_row("OSDN", c::mag, sig(z), w.label, ms);
        }
        std::cout << "  " << c::gray
                  << "------------------------------------------------------------"
                  << c::reset << "\n";
    }

    std::cout << "\n  " << c::bold << c::white
              << "summary" << c::reset << c::gray
              << "   pure-C++ scalar autograd · brain-mode 7-channel features"
              << c::reset << "\n";
    std::cout << "  " << c::gray
              << "        weights loaded via header-validated osdn_blob format —"
              << c::reset << "\n";
    std::cout << "  " << c::gray
              << "        any shape mismatch fails loudly, no silent garbled inference."
              << c::reset << "\n\n";
    return 0;
}
