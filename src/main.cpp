#include "value.hpp"
#include "nn.hpp"
#include "data.hpp"
#include "loss.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

static std::vector<ValuePtr> wrap(const std::vector<double>& x) {
    std::vector<ValuePtr> out;
    out.reserve(x.size());
    for (double d : x) out.push_back(v(d));
    return out;
}

static int argmax(const std::vector<ValuePtr>& logits) {
    int best = 0;
    double bv = logits[0]->data;
    for (size_t i = 1; i < logits.size(); ++i)
        if (logits[i]->data > bv) { bv = logits[i]->data; best = static_cast<int>(i); }
    return best;
}

static double evaluate(const MLP& net,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       int cm[3][3]) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) cm[i][j] = 0;
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto logits = net.forward(wrap(X[i]));
        int pred = argmax(logits);
        cm[y[i]][pred]++;
        if (pred == y[i]) correct++;
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/CTG.csv";
    const uint32_t seed       = 42;
    const double   lr         = 0.05;
    const int      epochs     = 50;
    const std::vector<int> arch = {22, 16, 16, 3};
    const int      eval_every = 5;

    auto ds = load_ctg(path, 0.2, seed);
    std::cout << "train = " << ds.X_train.size()
              << "   test = " << ds.X_test.size() << "\n";

    std::mt19937 rng(seed);
    MLP net(arch, rng);
    auto params = net.parameters();
    std::cout << "params = " << params.size() << "\n";
    std::cout << "lr = " << lr << "   epochs = " << epochs << "\n\n";

    std::vector<size_t> order(ds.X_train.size());
    std::iota(order.begin(), order.end(), 0);

    std::cout << std::fixed << std::setprecision(4);
    auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double loss_sum = 0.0;

        for (size_t k : order) {
            auto logits = net.forward(wrap(ds.X_train[k]));
            auto probs  = softmax(logits);
            auto loss   = cross_entropy(probs, ds.y_train[k]);
            loss_sum += loss->data;

            for (auto& p : params) p->grad = 0.0;
            loss->backward();
            for (auto& p : params) p->data -= lr * p->grad;
        }

        double avg_loss = loss_sum / static_cast<double>(order.size());
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        if (epoch % eval_every == 0 || epoch == 1 || epoch == epochs) {
            int cm[3][3];
            double acc = evaluate(net, ds.X_test, ds.y_test, cm);
            std::cout << "epoch " << std::setw(3) << epoch
                      << "   train_loss=" << avg_loss
                      << "   test_acc=" << acc
                      << "   elapsed=" << elapsed << "s\n";
        } else {
            std::cout << "epoch " << std::setw(3) << epoch
                      << "   train_loss=" << avg_loss
                      << "                    elapsed=" << elapsed << "s\n";
        }
    }

    int cm[3][3];
    double final_acc = evaluate(net, ds.X_test, ds.y_test, cm);
    std::cout << "\nfinal test accuracy = " << final_acc << "\n";
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
        std::cout << "  " << names[c]
                  << "   precision=" << prec
                  << "   recall=" << rec << "\n";
    }
    return 0;
}
