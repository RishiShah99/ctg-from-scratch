#include "cgm_data.hpp"
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdint>
#include <unordered_map>

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    for (char c : line) {
        if (c == '"') { in_quote = !in_quote; continue; }
        if (c == ',' && !in_quote) { out.push_back(cur); cur.clear(); continue; }
        cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

std::vector<CGMRecord> load_cgm_csv(const std::string& path) {
    std::ifstream f(path.c_str());
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(f, line)) throw std::runtime_error("empty file " + path);
    auto header = split_csv(line);

    auto col = [&](const std::string& name) -> int {
        for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return static_cast<int>(i);
        return -1;
    };
    int c_pid    = col("patient_id");
    int c_t      = col("t_min");
    int c_g      = col("glucose");
    int c_event  = col("event");
    int c_amount = col("amount");
    if (c_pid < 0 || c_t < 0) throw std::runtime_error("missing required columns patient_id, t_min");

    std::unordered_map<std::string, CGMRecord> by_pid;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = split_csv(line);
        if (static_cast<int>(cells.size()) <= std::max({c_pid, c_t})) continue;
        const std::string& pid = cells[c_pid];
        int64_t t = std::stoll(cells[c_t]);
        auto& rec = by_pid[pid];
        if (rec.patient_id.empty()) rec.patient_id = pid;

        bool has_g = c_g >= 0 && static_cast<int>(cells.size()) > c_g && !cells[c_g].empty();
        bool has_e = c_event >= 0 && static_cast<int>(cells.size()) > c_event && !cells[c_event].empty();
        bool has_a = c_amount >= 0 && static_cast<int>(cells.size()) > c_amount && !cells[c_amount].empty();

        if (has_g) {
            rec.t_min.push_back(t);
            rec.glucose.push_back(std::stod(cells[c_g]));
        }
        if (has_e) {
            const std::string& e = cells[c_event];
            double amount = has_a ? std::stod(cells[c_amount]) : 0.0;
            if (e == "bolus") { rec.bolus_t_min.push_back(t); rec.bolus_units.push_back(amount); }
            else if (e == "meal") { rec.meal_t_min.push_back(t); rec.meal_carbs_g.push_back(amount); }
        }
    }

    std::vector<CGMRecord> out;
    out.reserve(by_pid.size());
    for (auto& kv : by_pid) {
        auto& r = kv.second;
        std::vector<size_t> idx(r.t_min.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return r.t_min[a] < r.t_min[b]; });
        std::vector<int64_t> ts; std::vector<double> gs;
        ts.reserve(idx.size()); gs.reserve(idx.size());
        for (size_t i : idx) { ts.push_back(r.t_min[i]); gs.push_back(r.glucose[i]); }
        r.t_min = std::move(ts); r.glucose = std::move(gs);

        auto sort_events = [](std::vector<int64_t>& times, std::vector<double>& vals) {
            std::vector<size_t> e_idx(times.size());
            std::iota(e_idx.begin(), e_idx.end(), 0);
            std::sort(e_idx.begin(), e_idx.end(), [&](size_t a, size_t b){ return times[a] < times[b]; });
            std::vector<int64_t> ts2; std::vector<double> vs2;
            ts2.reserve(e_idx.size()); vs2.reserve(e_idx.size());
            for (size_t i : e_idx) { ts2.push_back(times[i]); vs2.push_back(vals[i]); }
            times = std::move(ts2); vals = std::move(vs2);
        };
        sort_events(r.bolus_t_min, r.bolus_units);
        sort_events(r.meal_t_min, r.meal_carbs_g);

        out.push_back(std::move(r));
    }
    return out;
}

static bool any_recent(const std::vector<int64_t>& times, int64_t now, int max_age_min) {
    for (auto it = times.rbegin(); it != times.rend(); ++it) {
        int64_t age = now - *it;
        if (age < 0) continue;
        if (age <= max_age_min) return true;
        return false;  // events sorted ascending; once age > max_age_min,
                       // every earlier event is even older.
    }
    return false;
}

CGMDataset make_windows(const std::vector<CGMRecord>& records,
                        int lookback_steps,
                        int horizon_steps,
                        int step_minutes,
                        double hypo_threshold,
                        int post_bolus_max_min,
                        const PatientSplitPolicy& policy,
                        uint32_t seed,
                        int window_stride)
{
    if (window_stride < 1) window_stride = 1;
    CGMDataset ds;
    ds.lookback_steps = lookback_steps;
    ds.horizon_steps = horizon_steps;
    ds.step_minutes = step_minutes;
    ds.hypo_threshold = hypo_threshold;
    ds.post_bolus_max_min = post_bolus_max_min;

    // --- patient-disjoint split validation ---
    std::unordered_map<std::string, int> fold;  // 0=train, 1=val, 2=test
    for (const auto& rec : records) fold.emplace(rec.patient_id, 0);

    auto assign = [&](const std::vector<std::string>& ids, int code, const char* name) {
        for (const auto& id : ids) {
            auto it = fold.find(id);
            if (it == fold.end()) {
                throw std::runtime_error("split policy references unknown patient_id '"
                                         + id + "' (not present in CSV) for fold " + name);
            }
            if (it->second != 0) {
                throw std::runtime_error("patient_id '" + id
                                         + "' appears in more than one fold");
            }
            it->second = code;
        }
    };
    assign(policy.val_ids, 1, "val");
    assign(policy.test_ids, 2, "test");

    for (const auto& rec : records) {
        if (rec.t_min.size() < static_cast<size_t>(lookback_steps + horizon_steps + 2)) continue;
        for (size_t anchor = lookback_steps; anchor + horizon_steps < rec.t_min.size(); anchor += window_stride) {
            int64_t t_now = rec.t_min[anchor];
            // Regular-sampling validation: every adjacent pair in
            // [anchor - lookback_steps, anchor + horizon_steps] must be
            // exactly step_minutes apart. The endpoint-only check we used
            // before missed a dropped-then-duplicated sample in the
            // interior: total span still equaled lookback*step but the
            // sequence was wrong.
            bool reg = true;
            const int64_t want_dt = static_cast<int64_t>(step_minutes);
            // Inclusive bound: the last pair checked is
            // (t[anchor+horizon-1], t[anchor+horizon]). The horizon target
            // sample at t[anchor+horizon] is consumed downstream by
            // future_min_*; rejecting windows where that final gap is
            // irregular is intentional.
            for (size_t k = anchor - lookback_steps;
                 k < anchor + horizon_steps && reg; ++k) {
                if ((rec.t_min[k + 1] - rec.t_min[k]) != want_dt) reg = false;
            }
            if (!reg) continue;

            CGMWindow w;
            w.patient_id = rec.patient_id;
            w.t_anchor_min = t_now;
            w.lookback.reserve(lookback_steps);
            for (int i = lookback_steps; i > 0; --i)
                w.lookback.push_back(rec.glucose[anchor - i + 1]);

            w.tod_sin.reserve(lookback_steps);
            w.tod_cos.reserve(lookback_steps);
            w.iob.reserve(lookback_steps);
            w.cob.reserve(lookback_steps);
            constexpr double TWO_PI = 6.283185307179586;
            constexpr double TAU_IOB = 120.0;
            constexpr double TAU_COB = 90.0;
            constexpr double IOB_SCALE = 5.0;
            constexpr double COB_SCALE = 50.0;
            constexpr double IOB_HORIZON_MIN = 600.0;
            constexpr double COB_HORIZON_MIN = 480.0;
            for (int i = 0; i < lookback_steps; ++i) {
                int64_t step_t = t_now - static_cast<int64_t>(lookback_steps - 1 - i) * step_minutes;
                int64_t tod = ((step_t % 1440) + 1440) % 1440;
                double tod_frac = static_cast<double>(tod) / 1440.0;
                w.tod_sin.push_back(std::sin(TWO_PI * tod_frac));
                w.tod_cos.push_back(std::cos(TWO_PI * tod_frac));

                double iob_acc = 0.0;
                for (size_t b = 0; b < rec.bolus_t_min.size(); ++b) {
                    int64_t tb = rec.bolus_t_min[b];
                    if (tb > step_t) break;
                    double dt = static_cast<double>(step_t - tb);
                    if (dt > IOB_HORIZON_MIN) continue;
                    iob_acc += rec.bolus_units[b] * std::exp(-dt / TAU_IOB);
                }
                w.iob.push_back(iob_acc / IOB_SCALE);

                double cob_acc = 0.0;
                for (size_t m = 0; m < rec.meal_t_min.size(); ++m) {
                    int64_t tm = rec.meal_t_min[m];
                    if (tm > step_t) break;
                    double dt = static_cast<double>(step_t - tm);
                    if (dt > COB_HORIZON_MIN) continue;
                    cob_acc += rec.meal_carbs_g[m] * std::exp(-dt / TAU_COB);
                }
                w.cob.push_back(cob_acc / COB_SCALE);
            }

            w.future_glucose.reserve(horizon_steps);
            double mn = rec.glucose[anchor + 1];
            for (int i = 1; i <= horizon_steps; ++i) {
                double g = rec.glucose[anchor + i];
                w.future_glucose.push_back(g);
                if (g < mn) mn = g;
            }
            w.future_min_glucose = mn;
            w.label = mn < hypo_threshold ? 1 : 0;
            w.in_post_bolus_window = any_recent(rec.bolus_t_min, t_now, post_bolus_max_min);
            int code = fold[rec.patient_id];
            if      (code == 1) ds.val.push_back(std::move(w));
            else if (code == 2) ds.test.push_back(std::move(w));
            else                ds.train.push_back(std::move(w));
        }
    }

    std::mt19937 rng(seed);
    std::shuffle(ds.train.begin(), ds.train.end(), rng);
    std::shuffle(ds.val.begin(),   ds.val.end(),   rng);
    std::shuffle(ds.test.begin(),  ds.test.end(),  rng);

    // Assert pairwise-disjoint patient_id sets across folds. Cheap belt-and-braces:
    // the fold[] code above already guarantees each patient lives in one bucket,
    // but verifying the actual windows catches future regressions in the assign logic.
    auto ids_in = [](const std::vector<CGMWindow>& v) {
        std::unordered_map<std::string, int> s;
        for (const auto& w : v) s[w.patient_id] = 1;
        return s;
    };
    auto train_ids = ids_in(ds.train);
    auto val_ids   = ids_in(ds.val);
    auto test_ids  = ids_in(ds.test);
    auto overlap = [](const std::unordered_map<std::string, int>& a,
                      const std::unordered_map<std::string, int>& b,
                      const char* an, const char* bn) {
        for (const auto& kv : a)
            if (b.count(kv.first))
                throw std::runtime_error(std::string("patient '") + kv.first
                                         + "' present in both " + an + " and " + bn);
    };
    overlap(train_ids, val_ids,  "train", "val");
    overlap(train_ids, test_ids, "train", "test");
    overlap(val_ids,   test_ids, "val",   "test");

    return ds;
}

void normalize_inplace(CGMDataset& ds, NormStats& out_stats) {
    double sum = 0.0, sqsum = 0.0; size_t cnt = 0;
    for (const auto& w : ds.train) {
        for (double g : w.lookback) { sum += g; sqsum += g * g; cnt++; }
    }
    double mean = cnt > 0 ? sum / static_cast<double>(cnt) : 0.0;
    double var  = cnt > 0 ? sqsum / static_cast<double>(cnt) - mean * mean : 1.0;
    double std_dev = std::sqrt(std::max(1e-12, var));
    out_stats.cgm_mean = mean;
    out_stats.cgm_std  = std_dev;

    auto apply = [&](std::vector<CGMWindow>& set) {
        for (auto& w : set) for (auto& g : w.lookback) g = (g - mean) / std_dev;
    };
    apply(ds.train); apply(ds.val); apply(ds.test);

    double iob_sum = 0.0, iob_sqsum = 0.0; size_t iob_cnt = 0;
    double cob_sum = 0.0, cob_sqsum = 0.0; size_t cob_cnt = 0;
    for (const auto& w : ds.train) {
        for (double x : w.iob) { iob_sum += x; iob_sqsum += x * x; iob_cnt++; }
        for (double x : w.cob) { cob_sum += x; cob_sqsum += x * x; cob_cnt++; }
    }
    double iob_mean = iob_cnt > 0 ? iob_sum / iob_cnt : 0.0;
    double iob_var  = iob_cnt > 0 ? iob_sqsum / iob_cnt - iob_mean * iob_mean : 1.0;
    double iob_std  = std::sqrt(std::max(1e-6, iob_var));
    double cob_mean = cob_cnt > 0 ? cob_sum / cob_cnt : 0.0;
    double cob_var  = cob_cnt > 0 ? cob_sqsum / cob_cnt - cob_mean * cob_mean : 1.0;
    double cob_std  = std::sqrt(std::max(1e-6, cob_var));
    out_stats.iob_mean = iob_mean;
    out_stats.iob_std  = iob_std;
    out_stats.cob_mean = cob_mean;
    out_stats.cob_std  = cob_std;

    auto apply_ic = [&](std::vector<CGMWindow>& set) {
        for (auto& w : set) {
            for (auto& x : w.iob) x = (x - iob_mean) / iob_std;
            for (auto& x : w.cob) x = (x - cob_mean) / cob_std;
        }
    };
    apply_ic(ds.train); apply_ic(ds.val); apply_ic(ds.test);
}
