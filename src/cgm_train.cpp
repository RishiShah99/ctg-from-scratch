#include "value.hpp"
#include "s4d.hpp"
#include "osdn.hpp"
#include "optim.hpp"
#include "cgm_data.hpp"
#include "osdn_blob.hpp"
#include "cgm_net.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_set>
#include <cctype>

struct Args {
    std::string csv = "data/ohio_t1dm.csv";
    std::string results = "results/cgm.json";
    std::string val_ids_csv;   // comma-separated patient IDs for val fold (required)
    std::string test_ids_csv;  // comma-separated patient IDs for test fold (required)
    std::string arch = "s4d";
    int lookback = 96;
    int horizon  = 12;
    int step_min = 15;
    double hypo  = 70.0;
    int post_bolus = 240;
    int H = 32;
    int N = 16;
    double dt = 0.05;
    int epochs = 30;
    int batch = 8;
    double lr = 0.005;
    double wd = 1e-5;
    uint32_t seed = 42;
    int window_stride = 1;
    double grad_clip = 5.0;
    std::string pos_weight = "auto";
    double pos_weight_cap = 50.0;
    std::string select_by = "auroc";
    int threads = 0;
    std::string save_weights;
    std::string load_weights;
    std::string resume_from;   // prefix; loads <prefix>.last + <prefix>.state.json
    bool eval_only = false;
    int osdn_layers = 1;
    int input_channels = 7;
    double pred_loss_weight = 0.1;
    double aux_cls_weight = 0.3;
    std::string dump_test_probs;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto v_str = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--csv") a.csv = v_str();
        else if (k == "--results") a.results = v_str();
        else if (k == "--arch") a.arch = v_str();
        else if (k == "--lookback") a.lookback = std::stoi(v_str());
        else if (k == "--horizon") a.horizon = std::stoi(v_str());
        else if (k == "--step-min") a.step_min = std::stoi(v_str());
        else if (k == "--hypo") a.hypo = std::stod(v_str());
        else if (k == "--post-bolus") a.post_bolus = std::stoi(v_str());
        else if (k == "--H") a.H = std::stoi(v_str());
        else if (k == "--N") a.N = std::stoi(v_str());
        else if (k == "--dt") a.dt = std::stod(v_str());
        else if (k == "--epochs") a.epochs = std::stoi(v_str());
        else if (k == "--batch") a.batch = std::stoi(v_str());
        else if (k == "--lr") a.lr = std::stod(v_str());
        else if (k == "--wd") a.wd = std::stod(v_str());
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::stoul(v_str()));
        else if (k == "--window-stride") a.window_stride = std::stoi(v_str());
        else if (k == "--grad-clip") a.grad_clip = std::stod(v_str());
        else if (k == "--pos-weight") a.pos_weight = v_str();
        else if (k == "--pos-weight-cap") a.pos_weight_cap = std::stod(v_str());
        else if (k == "--select-by") a.select_by = v_str();
        else if (k == "--threads") a.threads = std::stoi(v_str());
        else if (k == "--save-weights") a.save_weights = v_str();
        else if (k == "--load-weights") a.load_weights = v_str();
        else if (k == "--eval-only") a.eval_only = true;
        else if (k == "--osdn-layers") a.osdn_layers = std::stoi(v_str());
        else if (k == "--input-channels") a.input_channels = std::stoi(v_str());
        else if (k == "--pred-loss-weight") a.pred_loss_weight = std::stod(v_str());
        else if (k == "--aux-cls-weight") a.aux_cls_weight = std::stod(v_str());
        else if (k == "--dump-test-probs") a.dump_test_probs = v_str();
        else if (k == "--val-ids") a.val_ids_csv = v_str();
        else if (k == "--test-ids") a.test_ids_csv = v_str();
        else if (k == "--resume") a.resume_from = v_str();
        else { std::cerr << "unknown arg: " << k << "\n"; std::exit(2); }
    }
    return a;
}

// Net + FwdOut are now defined in src/cgm_net.hpp and shared with cgm_demo.cpp.

static double sig(double z) { return 1.0 / (1.0 + std::exp(-z)); }

static ValuePtr bce(ValuePtr logit, int y, double pos_w) {
    ValuePtr softplus_neg = vlog(v(1.0) + vexp(v(-1.0) * logit));
    if (y == 1) return v(pos_w) * softplus_neg;
    return logit + softplus_neg;
}

static double bce_scalar(double z, int y, double pos_w) {
    double softplus_neg = std::log(1.0 + std::exp(-z));
    if (y == 1) return pos_w * softplus_neg;
    return z + softplus_neg;
}

class WorkerPool {
public:
    using Task = std::function<void(int)>;

    explicit WorkerPool(int n) : nworkers(n), pending(n, 0), barrier_count(0) {
        threads.reserve(n);
        for (int i = 0; i < n; ++i) threads.emplace_back([this, i]{ worker_loop(i); });
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            for (int i = 0; i < nworkers; ++i) pending[i] = 1;
        }
        cv_work.notify_all();
        for (auto& t : threads) t.join();
    }

    void run(Task task) {
        {
            std::lock_guard<std::mutex> lk(mu);
            current_task = std::move(task);
            barrier_count = 0;
            for (int i = 0; i < nworkers; ++i) pending[i] = 1;
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&]{ return barrier_count == nworkers; });
    }

    int size() const { return nworkers; }

private:
    void worker_loop(int id) {
        while (true) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&]{ return pending[id] != 0; });
                pending[id] = 0;
                if (stop) return;
                t = current_task;
            }
            t(id);
            {
                std::lock_guard<std::mutex> lk(mu);
                barrier_count++;
            }
            cv_done.notify_one();
        }
    }

    int nworkers;
    std::vector<std::thread> threads;
    std::vector<int> pending;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    Task current_task;
    int barrier_count;
    bool stop = false;
};

struct EvalReport {
    double loss;
    double acc;
    double recall;
    double recall_post_bolus;
    double specificity;
    double auroc_approx;
    int tp, fp, tn, fn;
};

struct ScoreRec {
    double logit;
    double prob;
    int label;
    bool in_pb;
};

static std::vector<ScoreRec> compute_scores(Net& net, const std::vector<CGMWindow>& set) {
    std::vector<ScoreRec> out;
    out.reserve(set.size());
    for (const auto& w : set) {
        FwdOut fo = net.forward(w);
        double p = sig(fo.final_logit->data);
        out.push_back({ fo.final_logit->data, p, w.label, w.in_post_bolus_window });
    }
    return out;
}

static EvalReport evaluate_from_scores(const std::vector<ScoreRec>& scored,
                                       double pos_w,
                                       double threshold)
{
    EvalReport r{};
    double loss_sum = 0.0;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    int tp_pb = 0, fn_pb = 0;
    std::vector<std::pair<double,int>> ranked;
    ranked.reserve(scored.size());
    for (const auto& s : scored) {
        loss_sum += bce_scalar(s.logit, s.label, pos_w);
        ranked.push_back({ s.prob, s.label });
        int pred = s.prob >= threshold ? 1 : 0;
        if (pred == 1 && s.label == 1) { tp++; if (s.in_pb) tp_pb++; }
        else if (pred == 1 && s.label == 0) fp++;
        else if (pred == 0 && s.label == 0) tn++;
        else                                 { fn++; if (s.in_pb) fn_pb++; }
    }
    r.loss = loss_sum / std::max<size_t>(1, scored.size());
    r.tp = tp; r.fp = fp; r.tn = tn; r.fn = fn;
    r.acc    = (tp + tn) / std::max(1.0, static_cast<double>(scored.size()));
    r.recall = tp / std::max(1.0, static_cast<double>(tp + fn));
    r.recall_post_bolus = (tp_pb + fn_pb) > 0
        ? static_cast<double>(tp_pb) / (tp_pb + fn_pb) : 0.0;
    r.specificity = tn / std::max(1.0, static_cast<double>(tn + fp));

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });
    int n_pos = tp + fn, n_neg = tn + fp;
    double cum_pos = 0.0, auc = 0.0;
    for (const auto& s : ranked) {
        if (s.second == 1) cum_pos += 1.0;
        else if (n_pos > 0) auc += cum_pos / n_pos;
    }
    r.auroc_approx = n_neg > 0 ? auc / n_neg : 0.0;
    return r;
}

static EvalReport evaluate(Net& net,
                           const std::vector<CGMWindow>& set,
                           double pos_w,
                           double threshold = 0.5)
{
    auto scored = compute_scores(net, set);
    return evaluate_from_scores(scored, pos_w, threshold);
}

static void print_eval(const std::string& tag, const EvalReport& r) {
    std::cout << "  " << tag
              << "  loss=" << std::fixed << std::setprecision(4) << r.loss
              << "  acc=" << r.acc
              << "  recall=" << r.recall
              << "  spec=" << r.specificity
              << "  pb_recall=" << r.recall_post_bolus
              << "  auroc=" << r.auroc_approx
              << "  cm=[tp=" << r.tp << " fp=" << r.fp
              << " tn=" << r.tn << " fn=" << r.fn << "]\n";
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.arch != "s4d" && a.arch != "osdn") {
        std::cerr << "unknown --arch: " << a.arch << " (s4d|osdn)\n";
        return 2;
    }
    if (a.input_channels < 1 || a.input_channels > 7) {
        std::cerr << "--input-channels must be in [1,7] (1=g, 3=+derivatives, 5=+tod, 7=+iob/cob)\n";
        return 2;
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::unitbuf;
    std::cout << a.arch << "-CGM real-data trainer  brain-mode\n";
    std::cout << "  csv=" << a.csv << "  arch=" << a.arch
              << "  H=" << a.H << "  N=" << a.N
              << "  dt=" << a.dt << "  lookback=" << a.lookback
              << "  horizon=" << a.horizon << "  step=" << a.step_min << "min"
              << "  stride=" << a.window_stride
              << "  in_ch=" << a.input_channels
              << "  pred_w=" << a.pred_loss_weight
              << "  aux_cls_w=" << a.aux_cls_weight << "\n";

    std::vector<CGMRecord> records;
    try {
        records = load_cgm_csv(a.csv);
    } catch (const std::exception& e) {
        std::cerr << "data load failed: " << e.what() << "\n";
        std::cerr << "expected long-format CSV with columns: patient_id,t_min,glucose,event,amount\n";
        return 1;
    }
    if (records.empty()) {
        std::cerr << "no patients parsed from " << a.csv << "\n";
        return 1;
    }
    std::cout << "  patients=" << records.size() << "\n";

    if (a.val_ids_csv.empty() || a.test_ids_csv.empty()) {
        std::cerr << "step-2 split requires explicit --val-ids and --test-ids "
                     "(comma-separated patient IDs).\n"
                     "  Ohio T1DM 8/2/2 split: --val-ids 584,588 --test-ids 591,596\n";
        return 2;
    }
    auto split_ids = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : s) {
            if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
            else if (!std::isspace(static_cast<unsigned char>(c))) cur.push_back(c);
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    };
    PatientSplitPolicy policy;
    policy.val_ids  = split_ids(a.val_ids_csv);
    policy.test_ids = split_ids(a.test_ids_csv);

    CGMDataset ds;
    try {
        ds = make_windows(records, a.lookback, a.horizon, a.step_min,
                          a.hypo, a.post_bolus, policy, a.seed,
                          a.window_stride);
    } catch (const std::exception& e) {
        std::cerr << "split error: " << e.what() << "\n";
        return 2;
    }
    auto fold_ids = [](const std::vector<CGMWindow>& v) {
        std::vector<std::string> ids;
        std::unordered_set<std::string> seen;
        for (const auto& w : v) if (seen.insert(w.patient_id).second) ids.push_back(w.patient_id);
        std::sort(ids.begin(), ids.end());
        return ids;
    };
    auto join_ids = [](const std::vector<std::string>& ids) {
        std::string out;
        for (size_t i = 0; i < ids.size(); ++i) { if (i) out += ","; out += ids[i]; }
        return out;
    };
    std::cout << "  split (patient-disjoint):"
              << "  train=[" << join_ids(fold_ids(ds.train)) << "]"
              << "  val=["   << join_ids(fold_ids(ds.val))   << "]"
              << "  test=["  << join_ids(fold_ids(ds.test))  << "]\n";
    double mean = 0.0, sd = 1.0;
    normalize_inplace(ds, mean, sd);

    int n_pos = 0, n_neg = 0;
    for (const auto& w : ds.train) { (w.label == 1 ? n_pos : n_neg)++; }
    double pos_w = 1.0;
    if (a.pos_weight == "auto") {
        pos_w = (n_pos > 0) ? static_cast<double>(n_neg) / static_cast<double>(n_pos) : 1.0;
        if (pos_w > a.pos_weight_cap) pos_w = a.pos_weight_cap;
    } else {
        pos_w = std::stod(a.pos_weight);
    }

    std::cout << "  windows: train=" << ds.train.size()
              << "  val=" << ds.val.size() << "  test=" << ds.test.size()
              << "  z=(" << mean << "," << sd << ")\n"
              << "  train_balance: pos=" << n_pos << " neg=" << n_neg
              << " (pos_rate=" << (ds.train.empty() ? 0.0 :
                    static_cast<double>(n_pos) / ds.train.size()) << ")"
              << "  pos_weight=" << pos_w
              << "  select_by=" << a.select_by << "\n";

    std::vector<double> future_mean(a.horizon, 0.0);
    std::vector<double> future_std(a.horizon, 1.0);
    {
        std::vector<double> sum(a.horizon, 0.0), sqsum(a.horizon, 0.0);
        int cnt = 0;
        for (const auto& w : ds.train) {
            if (static_cast<int>(w.future_glucose.size()) < a.horizon) continue;
            for (int i = 0; i < a.horizon; ++i) {
                double fg = (w.future_glucose[i] - mean) / sd;
                sum[i] += fg;
                sqsum[i] += fg * fg;
            }
            ++cnt;
        }
        for (int i = 0; i < a.horizon; ++i) {
            double m = cnt > 0 ? sum[i] / cnt : 0.0;
            double v = cnt > 0 ? sqsum[i] / cnt - m * m : 1.0;
            future_mean[i] = m;
            future_std[i] = std::sqrt(std::max(1e-6, v));
        }
    }

    auto next_norm_target = [&](const CGMWindow& w, int t) -> double {
        int L = static_cast<int>(w.lookback.size());
        if (t + 1 < L) return w.lookback[t + 1];
        int idx_future = t + 1 - L;
        if (idx_future >= 0 && idx_future < static_cast<int>(w.future_glucose.size())) {
            return (w.future_glucose[idx_future] - mean) / sd;
        }
        return w.lookback[L - 1];
    };

    std::cout << "\n";

    std::mt19937 rng(a.seed);
    Net net(a.arch, a.H, a.N, a.dt, a.osdn_layers, a.input_channels, rng);
    auto params = net.parameters();
    Adam optim(params, a.lr, 0.9, 0.999, 1e-8, a.wd);

    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 4;
    int n_threads = a.threads > 0 ? a.threads : std::min(hw, a.batch);
    if (n_threads > a.batch) n_threads = a.batch;
    if (n_threads < 1) n_threads = 1;
    std::cout << "  params=" << params.size()
              << "  threads=" << n_threads
              << " (hw_concurrency=" << hw << ")\n";

    if (!a.load_weights.empty() && !a.resume_from.empty()) {
        std::cerr << "--load-weights and --resume are mutually exclusive\n";
        return 2;
    }
    if (!a.load_weights.empty()) {
        std::ifstream wf(a.load_weights, std::ios::binary);
        if (!wf.is_open()) {
            std::cerr << "failed to open --load-weights file: " << a.load_weights << "\n";
            return 4;
        }
        osdn_blob::Header h{};
        std::string err;
        if (!osdn_blob::read_header(wf, h, err)) {
            std::cerr << "--load-weights: " << err << "\n";
            return 4;
        }
        std::string shape_err;
        if (!osdn_blob::shape_matches(h,
                static_cast<uint32_t>(a.H),
                static_cast<uint32_t>(a.N),
                static_cast<uint32_t>(a.input_channels),
                static_cast<uint32_t>(a.osdn_layers),
                shape_err)) {
            std::cerr << "--load-weights: shape mismatch\n" << shape_err;
            return 4;
        }
        if (h.param_count != params.size()) {
            std::cerr << "--load-weights: header param_count " << h.param_count
                      << " ≠ build expected " << params.size() << "\n";
            return 4;
        }
        for (size_t i = 0; i < params.size(); ++i) {
            float fv = 0.0f;
            wf.read(reinterpret_cast<char*>(&fv), sizeof(float));
            params[i]->data = static_cast<double>(fv);
        }
        std::cout << "  loaded weights from " << a.load_weights
                  << "  (" << osdn_blob::format_header(h) << ")\n";
    }
    std::cout << "\n";

    bool select_by_auroc = (a.select_by == "auroc");

    // --- write config JSON BEFORE training starts ---
    // Anyone who reads results/<run>.json mid-training sees what was being trained.
    // End-of-training overwrites the same file with full results.
    auto write_results_json = [&](int best_ep, double best_score, int last_completed_ep,
                                  const char* status,
                                  const EvalReport* td05, double thr80,
                                  const EvalReport* td80, double ms_per_window) {
        std::ofstream rf(a.results);
        if (!rf.is_open()) return;
        rf << std::fixed << std::setprecision(4);
        rf << "{\n"
           << "  \"status\": \"" << status << "\",\n"
           << "  \"csv\": \"" << a.csv << "\",\n"
           << "  \"arch\": \"" << a.arch << "\",\n"
           << "  \"osdn_layers\": " << a.osdn_layers << ",\n"
           << "  \"input_channels\": " << a.input_channels << ",\n"
           << "  \"pred_loss_weight\": " << a.pred_loss_weight << ",\n"
           << "  \"aux_cls_weight\": " << a.aux_cls_weight << ",\n"
           << "  \"H\": " << a.H << ", \"N\": " << a.N << ", \"dt\": " << a.dt << ",\n"
           << "  \"lookback\": " << a.lookback << ", \"horizon\": " << a.horizon
           << ", \"step_min\": " << a.step_min << ", \"hypo\": " << a.hypo << ",\n"
           << "  \"window_stride\": " << a.window_stride << ",\n"
           << "  \"epochs\": " << a.epochs << ", \"batch\": " << a.batch
           << ", \"lr\": " << a.lr << ", \"wd\": " << a.wd << ",\n"
           << "  \"seed\": " << a.seed << ", \"grad_clip\": " << a.grad_clip
           << ", \"post_bolus\": " << a.post_bolus << ",\n"
           << "  \"val_ids\": \"" << a.val_ids_csv << "\",\n"
           << "  \"test_ids\": \"" << a.test_ids_csv << "\",\n"
           << "  \"params\": " << params.size() << ",\n"
           << "  \"threads\": " << n_threads << ",\n"
           << "  \"train_pos\": " << n_pos << ", \"train_neg\": " << n_neg << ",\n"
           << "  \"pos_weight\": " << pos_w << ", \"select_by\": \"" << a.select_by << "\",\n"
           << "  \"norm_mean\": " << mean << ", \"norm_sd\": " << sd << ",\n"
           << "  \"last_completed_epoch\": " << last_completed_ep << ",\n"
           << "  \"best_epoch\": " << best_ep
           << ", \"best_val_" << (select_by_auroc ? "auroc" : "loss") << "\": " << best_score;
        if (td05 && td80) {
            rf << ",\n"
               << "  \"test_thr05\": { \"acc\": " << td05->acc
               << ", \"recall\": " << td05->recall
               << ", \"specificity\": " << td05->specificity
               << ", \"pb_recall\": " << td05->recall_post_bolus
               << ", \"auroc_approx\": " << td05->auroc_approx << " },\n"
               << "  \"test_thr_spec80\": { \"thr\": " << thr80
               << ", \"acc\": " << td80->acc
               << ", \"recall\": " << td80->recall
               << ", \"specificity\": " << td80->specificity
               << ", \"pb_recall\": " << td80->recall_post_bolus
               << ", \"auroc_approx\": " << td80->auroc_approx << " },\n"
               << "  \"inference_ms_per_window\": " << ms_per_window << "\n";
        } else {
            rf << "\n";
        }
        rf << "}\n";
    };
    write_results_json(/*best_ep=*/-1, /*best_score=*/-1.0, /*last_completed_ep=*/0,
                       "in_progress", nullptr, 0.0, nullptr, 0.0);

    std::vector<std::unique_ptr<Net>> worker_nets;
    worker_nets.reserve(n_threads);
    for (int w = 0; w < n_threads; ++w) {
        std::mt19937 wrng(a.seed + 1000u + static_cast<uint32_t>(w));
        worker_nets.push_back(std::make_unique<Net>(a.arch, a.H, a.N, a.dt, a.osdn_layers, a.input_channels, wrng));
    }
    std::vector<std::vector<ValuePtr>> worker_params(n_threads);
    for (int w = 0; w < n_threads; ++w) {
        worker_params[w] = worker_nets[w]->parameters();
        if (worker_params[w].size() != params.size()) {
            std::cerr << "worker " << w << " param-count mismatch: "
                      << worker_params[w].size() << " vs master " << params.size() << "\n";
            return 3;
        }
    }
    WorkerPool pool(n_threads);
    std::vector<double> master_weights(params.size(), 0.0);
    std::vector<std::vector<double>> worker_grads(
        n_threads, std::vector<double>(params.size(), 0.0));
    std::vector<double> worker_loss(n_threads, 0.0);
    std::vector<std::vector<size_t>> worker_assigns(n_threads);

    std::vector<size_t> order(ds.train.size());
    std::iota(order.begin(), order.end(), 0);

    double best_score = select_by_auroc ? -1.0 : 1e9;
    int    best_ep  = 0;
    int    start_epoch = 1;
    std::vector<double> best_params(params.size());

    auto read_blob_into = [&](const std::string& path, std::vector<double>& out) -> bool {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        osdn_blob::Header h{};
        std::string err;
        if (!osdn_blob::read_header(f, h, err)) {
            std::cerr << "  --resume: " << path << ": " << err << "\n";
            return false;
        }
        std::string shape_err;
        if (!osdn_blob::shape_matches(h,
                static_cast<uint32_t>(a.H),
                static_cast<uint32_t>(a.N),
                static_cast<uint32_t>(a.input_channels),
                static_cast<uint32_t>(a.osdn_layers),
                shape_err)) {
            std::cerr << "  --resume: " << path << ": shape mismatch\n" << shape_err;
            return false;
        }
        if (h.param_count != out.size()) {
            std::cerr << "  --resume: " << path << ": header param_count "
                      << h.param_count << " ≠ build expected " << out.size() << "\n";
            return false;
        }
        for (size_t i = 0; i < out.size(); ++i) {
            float fv = 0.0f;
            if (!f.read(reinterpret_cast<char*>(&fv), sizeof(float))) return false;
            out[i] = static_cast<double>(fv);
        }
        return true;
    };

    if (!a.resume_from.empty()) {
        const std::string last_path  = a.resume_from + ".last";
        const std::string best_path  = a.resume_from + ".best";
        const std::string state_path = a.resume_from + ".state.json";
        std::vector<double> last_w(params.size());
        if (!read_blob_into(last_path, last_w)) {
            std::cerr << "--resume: cannot read " << last_path << "\n";
            return 4;
        }
        for (size_t i = 0; i < params.size(); ++i) params[i]->data = last_w[i];

        if (!read_blob_into(best_path, best_params)) {
            std::cerr << "--resume: cannot read " << best_path << "\n";
            return 4;
        }

        std::ifstream sf(state_path);
        if (!sf.is_open()) {
            std::cerr << "--resume: cannot read " << state_path << "\n";
            return 4;
        }
        // tiny ad-hoc JSON reader: find "key": value pairs we care about.
        std::string blob((std::istreambuf_iterator<char>(sf)),
                          std::istreambuf_iterator<char>());
        auto find_num = [&](const std::string& key) -> double {
            std::string needle = "\"" + key + "\"";
            auto p = blob.find(needle);
            if (p == std::string::npos) return std::nan("");
            p = blob.find(':', p);
            if (p == std::string::npos) return std::nan("");
            return std::stod(blob.substr(p + 1));
        };
        double v_pc  = find_num("param_count");
        double v_lce = find_num("last_completed_epoch");
        double v_be  = find_num("best_epoch");
        double v_bs  = find_num("best_score");
        if (std::isnan(v_pc) || static_cast<size_t>(v_pc) != params.size()) {
            std::cerr << "--resume: param_count in state.json (" << v_pc
                      << ") does not match current build (" << params.size() << ")\n";
            return 4;
        }
        if (std::isnan(v_lce) || std::isnan(v_be) || std::isnan(v_bs)) {
            std::cerr << "--resume: state.json missing required fields\n";
            return 4;
        }
        start_epoch = static_cast<int>(v_lce) + 1;
        best_ep    = static_cast<int>(v_be);
        best_score = v_bs;
        std::cout << "  resumed from " << a.resume_from
                  << "  start_epoch=" << start_epoch
                  << "  best_ep=" << best_ep
                  << "  best_score=" << best_score << "\n"
                  << "  NOTE: Adam moment state is not persisted; moments reset on resume.\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    if (a.eval_only) {
        std::cout << "  --eval-only set: skipping training\n";
        for (size_t i = 0; i < params.size(); ++i) best_params[i] = params[i]->data;
        best_ep = 0;
        best_score = 0.0;
    }
    for (int ep = start_epoch; !a.eval_only && ep <= a.epochs; ++ep) {
        std::cout << "ep " << std::setw(3) << ep << "  [training " << order.size() << " windows...]\n";
        std::shuffle(order.begin(), order.end(), rng);
        double loss_sum = 0.0;
        int batches = 0;
        for (size_t bi = 0; bi < order.size(); bi += a.batch) {
            size_t bsz = std::min(static_cast<size_t>(a.batch), order.size() - bi);

            for (size_t i = 0; i < params.size(); ++i) master_weights[i] = params[i]->data;
            for (int w = 0; w < n_threads; ++w) {
                worker_assigns[w].clear();
                worker_loss[w] = 0.0;
            }
            for (size_t j = 0; j < bsz; ++j) {
                worker_assigns[j % n_threads].push_back(order[bi + j]);
            }

            const auto& train_set = ds.train;
            const double pw = pos_w;
            const double aux_w = a.aux_cls_weight;
            const double pred_w_coef = a.pred_loss_weight;
            pool.run([&, bsz](int wid) {
                (void)bsz;
                auto& wparams = worker_params[wid];
                for (size_t i = 0; i < wparams.size(); ++i) {
                    wparams[i]->data = master_weights[i];
                    wparams[i]->grad = 0.0;
                }
                double lsum = 0.0;
                auto* wnet = worker_nets[wid].get();
                for (size_t idx : worker_assigns[wid]) {
                    const auto& s = train_set[idx];
                    FwdOut fo = wnet->forward(s);
                    int L = static_cast<int>(fo.step_logits.size());
                    ValuePtr loss = bce(fo.final_logit, s.label, pw);

                    if (aux_w > 0.0 && L > 1) {
                        ValuePtr aux_sum = v(0.0);
                        int aux_n = 0;
                        for (int t = L / 2; t < L - 1; ++t) {
                            aux_sum = aux_sum + bce(fo.step_logits[t], s.label, pw);
                            aux_n++;
                        }
                        if (aux_n > 0) loss = loss + v(aux_w / aux_n) * aux_sum;
                    }

                    if (pred_w_coef > 0.0 && L > 1) {
                        ValuePtr pred_sum = v(0.0);
                        int pred_n = 0;
                        for (int t = 0; t < L - 1; ++t) {
                            double target = next_norm_target(s, t);
                            ValuePtr err = fo.step_preds[t] - v(target);
                            pred_sum = pred_sum + err * err;
                            pred_n++;
                        }
                        if (pred_n > 0) loss = loss + v(pred_w_coef / pred_n) * pred_sum;
                    }

                    lsum += loss->data;
                    loss->backward();
                }
                auto& wg = worker_grads[wid];
                for (size_t i = 0; i < wparams.size(); ++i) wg[i] = wparams[i]->grad;
                worker_loss[wid] = lsum;
            });

            for (auto& p : params) p->grad = 0.0;
            for (int w = 0; w < n_threads; ++w) {
                for (size_t i = 0; i < params.size(); ++i) {
                    params[i]->grad += worker_grads[w][i];
                }
                loss_sum += worker_loss[w];
            }
            double scale = 1.0 / static_cast<double>(bsz);
            for (auto& p : params) p->grad *= scale;
            if (a.grad_clip > 0.0) {
                double gn2 = 0.0;
                for (auto& p : params) gn2 += p->grad * p->grad;
                double gn = std::sqrt(gn2);
                if (!std::isfinite(gn) || gn > a.grad_clip) {
                    double clip_scale = std::isfinite(gn) ? (a.grad_clip / (gn + 1e-12)) : 0.0;
                    for (auto& p : params) p->grad *= clip_scale;
                }
            }
            optim.step();
            batches++;
        }
        double train_avg = loss_sum / std::max(1.0, static_cast<double>(batches * a.batch));

        EvalReport vr = evaluate(net, ds.val, pos_w);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "ep " << std::setw(3) << ep
                  << "  train=" << train_avg
                  << "  val_loss=" << vr.loss
                  << "  val_acc=" << vr.acc
                  << "  val_recall=" << vr.recall
                  << "  val_pb_recall=" << vr.recall_post_bolus
                  << "  val_auroc=" << vr.auroc_approx
                  << "  " << std::setprecision(1) << elapsed << "s\n"
                  << std::setprecision(4);

        bool improved = select_by_auroc
            ? (vr.auroc_approx > best_score + 1e-6)
            : (vr.loss        < best_score - 1e-6);
        if (improved) {
            best_score = select_by_auroc ? vr.auroc_approx : vr.loss;
            best_ep  = ep;
            for (size_t i = 0; i < params.size(); ++i) best_params[i] = params[i]->data;
        }

        if (!a.save_weights.empty()) {
            auto write_blob = [&](const std::string& p, const std::vector<double>& vals) {
                std::ofstream wf(p, std::ios::binary);
                if (!wf.is_open()) {
                    std::cerr << "  WARN: failed to open checkpoint " << p << "\n";
                    return;
                }
                osdn_blob::write_header(wf,
                    static_cast<uint32_t>(a.H),
                    static_cast<uint32_t>(a.N),
                    static_cast<uint32_t>(a.input_channels),
                    static_cast<uint32_t>(a.osdn_layers),
                    static_cast<uint32_t>(vals.size()));
                for (size_t i = 0; i < vals.size(); ++i) {
                    float fv = static_cast<float>(vals[i]);
                    wf.write(reinterpret_cast<const char*>(&fv), sizeof(float));
                }
            };
            std::vector<double> cur(params.size());
            for (size_t i = 0; i < params.size(); ++i) cur[i] = params[i]->data;
            write_blob(a.save_weights + ".last", cur);
            if (improved) write_blob(a.save_weights + ".best", best_params);

            // Per-epoch resume state — written AFTER both .last and (possibly) .best
            // so a crash mid-write can't leave state pointing at non-existent weights.
            std::ofstream sf(a.save_weights + ".state.json");
            if (sf.is_open()) {
                sf << std::fixed << std::setprecision(6) << "{\n"
                   << "  \"param_count\": " << params.size() << ",\n"
                   << "  \"last_completed_epoch\": " << ep << ",\n"
                   << "  \"best_epoch\": " << best_ep << ",\n"
                   << "  \"best_score\": " << best_score << ",\n"
                   << "  \"select_by\": \"" << a.select_by << "\",\n"
                   << "  \"pos_w\": " << pos_w << ",\n"
                   << "  \"norm_mean\": " << mean << ", \"norm_sd\": " << sd << "\n"
                   << "}\n";
            }
        }
        write_results_json(best_ep, best_score, ep, "in_progress",
                           nullptr, 0.0, nullptr, 0.0);
    }

    for (size_t i = 0; i < params.size(); ++i) params[i]->data = best_params[i];
    if (!a.eval_only) {
        std::cout << "\nrestored best weights from epoch " << best_ep
                  << "   val_" << (select_by_auroc ? "auroc" : "loss")
                  << "=" << best_score << "\n\n";
    } else {
        std::cout << "\n";
    }

    if (!a.save_weights.empty()) {
        std::ofstream wf(a.save_weights, std::ios::binary);
        if (!wf.is_open()) {
            std::cerr << "failed to open --save-weights file: " << a.save_weights << "\n";
            return 5;
        }
        osdn_blob::write_header(wf,
            static_cast<uint32_t>(a.H),
            static_cast<uint32_t>(a.N),
            static_cast<uint32_t>(a.input_channels),
            static_cast<uint32_t>(a.osdn_layers),
            static_cast<uint32_t>(params.size()));
        for (size_t i = 0; i < params.size(); ++i) {
            float fv = static_cast<float>(params[i]->data);
            wf.write(reinterpret_cast<const char*>(&fv), sizeof(float));
        }
        std::cout << "  saved weights -> " << a.save_weights
                  << " (" << osdn_blob::HEADER_BYTES
                  << " header bytes + " << (params.size() * sizeof(float))
                  << " param bytes)\n\n";
    }

    auto test_scored = compute_scores(net, ds.test);
    EvalReport test_default = evaluate_from_scores(test_scored, pos_w, 0.5);
    print_eval("test (thr=0.5):", test_default);

    if (!a.dump_test_probs.empty()) {
        std::ofstream df(a.dump_test_probs);
        if (df.is_open()) {
            df << "patient_id,t_anchor_min,label,in_pb,future_min_glucose,prob,logit\n";
            for (size_t i = 0; i < ds.test.size(); ++i) {
                const auto& w = ds.test[i];
                const auto& s = test_scored[i];
                df << w.patient_id << "," << w.t_anchor_min << "," << w.label
                   << "," << (w.in_post_bolus_window ? 1 : 0)
                   << "," << w.future_min_glucose
                   << "," << std::setprecision(6) << s.prob
                   << "," << std::setprecision(6) << s.logit << "\n";
            }
            std::cout << "  dumped " << ds.test.size()
                      << " test probs -> " << a.dump_test_probs << "\n";
        }
    }

    // Threshold selection: scan on VAL to pick thr@spec, then report metrics
    // on TEST at that frozen threshold. Scanning on test would leak test
    // labels into the threshold choice — dishonest for a deploy framing.
    auto val_scored = compute_scores(net, ds.val);
    auto pick_thr_on_val = [&](double target_spec) -> double {
        double best_thr = -1.0;
        double best_recall = -1.0;
        for (double thr = 0.01; thr < 1.0; thr += 0.01) {
            EvalReport r = evaluate_from_scores(val_scored, pos_w, thr);
            if (r.specificity >= target_spec && r.recall > best_recall) {
                best_thr = thr;
                best_recall = r.recall;
            }
        }
        return best_thr >= 0.0 ? best_thr : 0.5;
    };
    double thr80 = pick_thr_on_val(0.80);
    EvalReport test_at_80 = evaluate_from_scores(test_scored, pos_w, thr80);
    std::cout << "  threshold @ specificity=0.80 (picked on val) -> " << thr80 << "\n";
    print_eval("test (thr@spec80):", test_at_80);

    auto t_inf0 = std::chrono::steady_clock::now();
    int n_inf = std::min<int>(50, static_cast<int>(ds.test.size()));
    for (int i = 0; i < n_inf; ++i) net.forward(ds.test[i]);
    auto t_inf1 = std::chrono::steady_clock::now();
    double ms_per_window = 1000.0 * std::chrono::duration<double>(t_inf1 - t_inf0).count() / std::max(1, n_inf);
    std::cout << "  inference: " << std::setprecision(2) << ms_per_window << " ms / window\n";

    write_results_json(best_ep, best_score, a.epochs, "complete",
                       &test_default, thr80, &test_at_80, ms_per_window);
    std::cout << "\n  results -> " << a.results << "\n";
    return 0;
}
