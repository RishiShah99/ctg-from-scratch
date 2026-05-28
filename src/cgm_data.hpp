#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct CGMRecord {
    std::string patient_id;
    std::vector<int64_t> t_min;
    std::vector<double> glucose;
    std::vector<int64_t> bolus_t_min;
    std::vector<double> bolus_units;
    std::vector<int64_t> meal_t_min;
    std::vector<double> meal_carbs_g;
};

struct CGMWindow {
    std::string patient_id;
    int64_t t_anchor_min;
    std::vector<double> lookback;
    std::vector<double> tod_sin;
    std::vector<double> tod_cos;
    std::vector<double> iob;
    std::vector<double> cob;
    std::vector<double> future_glucose;
    double future_min_glucose;
    int label;
    bool in_post_bolus_window;
};

struct CGMDataset {
    int lookback_steps;
    int horizon_steps;
    int step_minutes;
    double hypo_threshold;
    int post_bolus_max_min;
    std::vector<CGMWindow> train;
    std::vector<CGMWindow> val;
    std::vector<CGMWindow> test;
};

std::vector<CGMRecord> load_cgm_csv(const std::string& path);

struct PatientSplitPolicy {
    std::vector<std::string> val_ids;
    std::vector<std::string> test_ids;
};

CGMDataset make_windows(const std::vector<CGMRecord>& records,
                        int lookback_steps,
                        int horizon_steps,
                        int step_minutes,
                        double hypo_threshold,
                        int post_bolus_max_min,
                        const PatientSplitPolicy& policy,
                        uint32_t seed,
                        int window_stride = 1);

struct NormStats {
    double cgm_mean;
    double cgm_std;
    double iob_mean;
    double iob_std;
    double cob_mean;
    double cob_std;
};

void normalize_inplace(CGMDataset& ds, NormStats& out_stats);
