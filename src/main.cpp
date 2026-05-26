#include "value.hpp"
#include "tx.hpp"
#include "data.hpp"
#include "loss.hpp"
#include "optim.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <vector>

static std::vector<ValuePtr> wrap(const std::vector<double>& x) {
    std::vector<ValuePtr> out;
    out.reserve(x.size());
    for (double d : x) out.push_back(v(d));
    return out;
}

static int argmax_v(const std::vector<ValuePtr>& xs) {
    int best = 0;
    for (size_t i = 1; i < xs.size(); ++i)
        if (xs[i]->data > xs[best]->data) best = static_cast<int>(i);
    return best;
}

static double evaluate(const FTTransformer& net,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       int cm[3][3]) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) cm[i][j] = 0;
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto logits = net.forward(wrap(X[i]));
        int pred = argmax_v(logits);
        cm[y[i]][pred]++;
        if (pred == y[i]) correct++;
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

static double balanced_accuracy(int cm[3][3]) {
    double sum = 0.0;
    int classes = 0;
    for (int c = 0; c < 3; ++c) {
        int total = cm[c][0] + cm[c][1] + cm[c][2];
        if (total > 0) {
            sum += static_cast<double>(cm[c][c]) / static_cast<double>(total);
            classes++;
        }
    }
    return classes > 0 ? sum / classes : 0.0;
}

static double cosine_lr(double lr_max, double lr_min, int t, int T) {
    if (T <= 1) return lr_max;
    double phase = 3.14159265358979323846 * static_cast<double>(t)
                                          / static_cast<double>(T - 1);
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + std::cos(phase));
}

static double linear_warmup(double lr_target, int t, int warmup) {
    if (warmup <= 0 || t >= warmup) return lr_target;
    return lr_target * static_cast<double>(t + 1) / static_cast<double>(warmup);
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

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/CTG.csv";

    const uint32_t seed        = 42;
    const int    n_features    = 22;
    const int    n_classes     = 3;
    const int    d_model       = 8;
    const int    n_heads       = 2;
    const int    d_ffn         = 32;
    const int    n_blocks      = 1;

    const int    epochs        = 40;
    const int    batch_size    = 16;
    const int    warmup_epochs = 3;
    const double lr_max        = 0.001;
    const double lr_min        = 1e-5;
    const double weight_decay  = 1e-4;
    const double max_grad_norm = 1.0;
    const int    patience      = 12;

    auto ds = load_ctg(path, 0.15, 0.20, seed);
    std::cout << "train = " << ds.X_train.size()
              << "   val = " << ds.X_val.size()
              << "   test = " << ds.X_test.size() << "\n";

    std::mt19937 rng(seed);
    FTTransformer net(n_features, n_classes, d_model, n_heads, d_ffn, n_blocks, rng);
    auto params = net.parameters();
    Adam optim(params, lr_max, 0.9, 0.999, 1e-8, weight_decay);

    std::cout << std::fixed << std::setprecision(4)
              << "FT-Transformer  d_model=" << d_model
              << "  n_heads=" << n_heads
              << "  d_ffn=" << d_ffn
              << "  n_blocks=" << n_blocks
              << "  params=" << params.size()
              << "  batch=" << batch_size
              << "  lr_max=" << lr_max
              << "  wd=" << weight_decay
              << "  warmup=" << warmup_epochs << "\n\n";

    // Inverse-frequency class weights from the training labels.
    // The previous hard-coded {1.0, 1.0, 1.0} made cross_entropy_weighted
    // a no-op disguised as a weighted loss — labels are imbalanced in
    // CTG (class 2 is rare), so the model would learn to ignore it. We
    // normalize so the average weight is 1.0, which keeps the loss scale
    // comparable to the unweighted version.
    std::vector<double> class_weights(3, 1.0);
    {
        std::vector<size_t> count(3, 0);
        for (int y : ds.y_train) {
            if (y >= 0 && y < 3) count[static_cast<size_t>(y)]++;
        }
        double total = 0.0;
        for (size_t i = 0; i < 3; ++i) total += static_cast<double>(count[i]);
        if (total > 0.0) {
            double mean = 0.0;
            for (size_t i = 0; i < 3; ++i) {
                if (count[i] == 0) class_weights[i] = 1.0;
                else class_weights[i] = total / (3.0 * static_cast<double>(count[i]));
                mean += class_weights[i];
            }
            mean /= 3.0;
            // Renormalize so the average weight is 1.0 — preserves loss scale.
            if (mean > 0.0) for (auto& w : class_weights) w /= mean;
        }
        std::cout << "  class_weights = ["
                  << class_weights[0] << ", "
                  << class_weights[1] << ", "
                  << class_weights[2] << "]"
                  << "  (counts " << count[0] << "/" << count[1] << "/" << count[2] << ")\n";
    }
    std::vector<size_t> order(ds.X_train.size());
    std::iota(order.begin(), order.end(), 0);

    std::vector<double> best_weights(params.size(), 0.0);
    double best_val   = -1.0;
    int    best_epoch = 0;
    int    no_improve = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double base_lr    = cosine_lr(lr_max, lr_min, epoch - 1, epochs);
        double current_lr = linear_warmup(base_lr, epoch - 1, warmup_epochs);
        optim.set_lr(current_lr);

        std::shuffle(order.begin(), order.end(), rng);
        double loss_sum = 0.0;
        int    batches  = 0;

        for (size_t bi = 0; bi < order.size(); bi += batch_size) {
            size_t bsz = std::min(static_cast<size_t>(batch_size), order.size() - bi);
            optim.zero_grad();
            double batch_loss = 0.0;
            for (size_t j = 0; j < bsz; ++j) {
                size_t idx = order[bi + j];
                auto logits = net.forward(wrap(ds.X_train[idx]));
                auto probs  = softmax(logits);
                auto loss   = cross_entropy_weighted(probs, ds.y_train[idx], class_weights);
                batch_loss += loss->data;
                loss->backward();
            }
            double scale = 1.0 / static_cast<double>(bsz);
            for (auto& p : params) p->grad *= scale;
            clip_grad_norm(params, max_grad_norm);
            optim.step();
            loss_sum += batch_loss * scale;
            batches++;
        }
        double avg_loss = loss_sum / static_cast<double>(batches);

        int cm_val[3][3];
        double val_acc  = evaluate(net, ds.X_val, ds.y_val, cm_val);
        double val_bacc = balanced_accuracy(cm_val);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        bool improved = val_acc > best_val + 1e-6;
        if (improved) {
            best_val   = val_acc;
            best_epoch = epoch;
            for (size_t i = 0; i < params.size(); ++i) best_weights[i] = params[i]->data;
            no_improve = 0;
        } else {
            no_improve++;
        }

        std::cout << "ep " << std::setw(3) << epoch
                  << "  lr=" << current_lr
                  << "  loss=" << avg_loss
                  << "  val_acc=" << val_acc
                  << "  val_bacc=" << val_bacc
                  << (improved ? "  *" : "   ")
                  << "  " << std::setprecision(1) << elapsed << "s\n";
        std::cout << std::setprecision(4);

        if (no_improve >= patience) {
            std::cout << "early stop at epoch " << epoch
                      << "   best val_acc=" << best_val
                      << " (ep " << best_epoch << ")\n";
            break;
        }
    }

    for (size_t i = 0; i < params.size(); ++i) params[i]->data = best_weights[i];
    std::cout << "\nrestored best weights from epoch " << best_epoch
              << "   val_acc=" << best_val << "\n";

    int cm[3][3];
    double test_acc  = evaluate(net, ds.X_test, ds.y_test, cm);
    double test_bacc = balanced_accuracy(cm);

    std::cout << "\n=== FT-Transformer test results ===\n";
    std::cout << "test accuracy           = " << test_acc << "\n";
    std::cout << "test balanced accuracy  = " << test_bacc << "\n";

    std::cout << "\nconfusion matrix (rows=true, cols=pred):\n";
    std::cout << "             N       S       P\n";
    const char* names[] = {"N", "S", "P"};
    for (int i = 0; i < 3; ++i) {
        std::cout << "    true " << names[i];
        for (int j = 0; j < 3; ++j) std::cout << std::setw(8) << cm[i][j];
        std::cout << "\n";
    }
    std::cout << "\nper-class metrics:\n";
    for (int c = 0; c < 3; ++c) {
        int tp = cm[c][c];
        int fp = 0, fn = 0;
        for (int i = 0; i < 3; ++i) {
            if (i != c) { fp += cm[i][c]; fn += cm[c][i]; }
        }
        double prec = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double rec  = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1   = (prec + rec) > 0.0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
        std::cout << "  " << names[c]
                  << "   precision=" << prec
                  << "   recall=" << rec
                  << "   f1=" << f1 << "\n";
    }
    return 0;
}
