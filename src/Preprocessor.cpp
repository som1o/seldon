#include "Preprocessor.h"
#include "SeldonExceptions.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace {
using NumVec = std::vector<double>;
using StrVec = std::vector<std::string>;
using TimeVec = std::vector<int64_t>;

bool isBinarySeries(const NumVec& values) {
    bool seenZero = false;
    bool seenOne = false;
    for (double v : values) {
        if (std::abs(v) <= 1e-9) {
            seenZero = true;
        } else if (std::abs(v - 1.0) <= 1e-9) {
            seenOne = true;
        } else {
            return false;
        }
    }
    return seenZero || seenOne;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

double medianOf(NumVec values) {
    if (values.empty()) return 0.0;
    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double upper = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + (mid - 1), values.begin() + mid);
        return (values[mid - 1] + upper) / 2.0;
    }
    return upper;
}

void imputeNumeric(NumVec& values, std::vector<bool>& missing, const std::string& method) {
    NumVec valid;
    valid.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && std::isfinite(values[i])) valid.push_back(values[i]);

    double fill = 0.0;
    std::string m = toLower(method);
    if (m == "median") fill = medianOf(valid);
    else if (m == "zero") fill = 0.0;
    else {
        double sum = 0.0;
        for (double v : valid) sum += v;
        fill = valid.empty() ? 0.0 : (sum / valid.size());
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) {
            values[i] = fill;
            missing[i] = false;
        }
    }
}

void interpolateSeries(NumVec& values, std::vector<bool>& missing) {
    int lastKnown = -1;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!missing[i]) {
            if (lastKnown >= 0 && static_cast<size_t>(lastKnown + 1) < i) {
                double left = values[lastKnown];
                double right = values[i];
                size_t span = i - static_cast<size_t>(lastKnown);
                for (size_t k = 1; k < span; ++k) {
                    double t = static_cast<double>(k) / static_cast<double>(span);
                    values[lastKnown + static_cast<int>(k)] = left + t * (right - left);
                    missing[lastKnown + static_cast<int>(k)] = false;
                }
            }
            lastKnown = static_cast<int>(i);
        }
    }

    // forward/back fill edges
    for (size_t i = 1; i < values.size(); ++i) if (missing[i] && !missing[i - 1]) { values[i] = values[i - 1]; missing[i] = false; }
    for (size_t i = values.size(); i-- > 1;) {
        if (missing[i - 1] && !missing[i]) {
            values[i - 1] = values[i];
            missing[i - 1] = false;
        }
    }
}

void imputeCategorical(StrVec& values, std::vector<bool>& missing, const std::string& method) {
    std::unordered_map<std::string, size_t> freq;
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && !values[i].empty()) freq[values[i]]++;

    std::string mode = "Unknown";
    size_t best = 0;
    for (const auto& kv : freq) {
        if (kv.second > best) { best = kv.second; mode = kv.first; }
    }

    std::string m = toLower(method);
    std::string fill = (m == "mode" || m == "auto") ? mode : mode;
    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || values[i].empty()) {
            values[i] = fill;
            missing[i] = false;
        }
    }
}

void imputeDatetime(TimeVec& values, std::vector<bool>& missing, const std::string& method) {
    std::string m = toLower(method);
    if (m == "interpolate" || m == "auto") {
        NumVec temp(values.begin(), values.end());
        interpolateSeries(temp, missing);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<int64_t>(temp[i]);
    } else {
        int64_t fill = 0;
        for (size_t i = 0; i < values.size(); ++i) if (!missing[i]) { fill = values[i]; break; }
        for (size_t i = 0; i < values.size(); ++i) if (missing[i]) { values[i] = fill; missing[i] = false; }
    }
}

std::vector<bool> detectOutliersIQR(const NumVec& values) {
    std::vector<bool> flags(values.size(), false);
    if (values.size() < 4) return flags;

    NumVec sorted = values;
    std::sort(sorted.begin(), sorted.end());
    double q1 = sorted[sorted.size() / 4];
    double q3 = sorted[(sorted.size() * 3) / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr;
    double hi = q3 + 1.5 * iqr;

    for (size_t i = 0; i < values.size(); ++i) flags[i] = (values[i] < lo || values[i] > hi);
    return flags;
}

std::vector<bool> detectOutliersZ(const NumVec& values) {
    std::vector<bool> flags(values.size(), false);
    if (values.size() < 3) return flags;

    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= values.size();

    double var = 0.0;
    for (double v : values) { double d = v - mean; var += d * d; }
    var /= std::max<size_t>(1, values.size() - 1);
    double sd = std::sqrt(var);
    if (sd <= 1e-12) return flags;

    for (size_t i = 0; i < values.size(); ++i) {
        double z = std::abs((values[i] - mean) / sd);
        flags[i] = z > 3.0;
    }
    return flags;
}

void capOutliers(NumVec& values, const std::vector<bool>& flags) {
    NumVec inliers;
    for (size_t i = 0; i < values.size(); ++i) if (!flags[i]) inliers.push_back(values[i]);
    if (inliers.empty()) return;
    auto mm = std::minmax_element(inliers.begin(), inliers.end());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!flags[i]) continue;
        if (values[i] < *mm.first) values[i] = *mm.first;
        if (values[i] > *mm.second) values[i] = *mm.second;
    }
}

}

PreprocessReport Preprocessor::run(TypedDataset& data, const AutoConfig& config) {
    PreprocessReport report;
    report.originalRowCount = data.rowCount();

    // missing counts + imputation
    for (auto& col : data.columns()) {
        size_t missingCount = std::count(col.missing.begin(), col.missing.end(), true);
        report.missingCounts[col.name] = missingCount;

        std::string strategy = "auto";
        auto it = config.columnImputation.find(col.name);
        if (it != config.columnImputation.end()) strategy = it->second;

        if (col.type == ColumnType::NUMERIC) {
            auto& values = std::get<NumVec>(col.values);
            if (toLower(strategy) == "interpolate") interpolateSeries(values, col.missing);
            else imputeNumeric(values, col.missing, strategy);
        } else if (col.type == ColumnType::CATEGORICAL) {
            auto& values = std::get<StrVec>(col.values);
            imputeCategorical(values, col.missing, strategy.empty() ? "mode" : strategy);
        } else {
            auto& values = std::get<TimeVec>(col.values);
            imputeDatetime(values, col.missing, strategy.empty() ? "interpolate" : strategy);
        }
    }

    // outliers on numeric columns
    std::vector<bool> keep(data.rowCount(), true);
    for (auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        auto& values = std::get<NumVec>(col.values);

        std::vector<bool> flags = (toLower(config.outlierMethod) == "zscore")
            ? detectOutliersZ(values)
            : detectOutliersIQR(values);

        report.outlierFlags[col.name] = flags;
        size_t c = std::count(flags.begin(), flags.end(), true);
        report.outlierCounts[col.name] = c;

        std::string action = toLower(config.outlierAction);
        if (action == "remove") {
            for (size_t r = 0; r < flags.size(); ++r) if (flags[r]) keep[r] = false;
        } else if (action == "cap") {
            capOutliers(values, flags);
        }
    }

    if (toLower(config.outlierAction) == "remove") data.removeRows(keep);

    // scaling
    for (auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        auto& values = std::get<NumVec>(col.values);
        if (values.empty()) continue;

        if (!config.targetColumn.empty() && col.name == config.targetColumn && isBinarySeries(values)) {
            ScalingParams params;
            params.method = ScalingMethod::NONE;
            auto mm = std::minmax_element(values.begin(), values.end());
            params.min = *mm.first;
            params.max = *mm.second;
            params.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
            report.scaling[col.name] = params;
            continue;
        }

        auto mm = std::minmax_element(values.begin(), values.end());
        double minv = *mm.first;
        double maxv = *mm.second;
        double mean = 0.0;
        for (double v : values) mean += v;
        mean /= values.size();
        double var = 0.0;
        for (double v : values) { double d = v - mean; var += d * d; }
        double stdv = std::sqrt(var / std::max<size_t>(1, values.size() - 1));

        ScalingMethod method = ScalingMethod::NONE;
        std::string pref = toLower(config.scalingMethod);
        if (pref == "zscore") method = ScalingMethod::ZSCORE;
        else if (pref == "minmax") method = ScalingMethod::MINMAX;
        else if (pref == "none") method = ScalingMethod::NONE;
        else {
            double range = maxv - minv;
            method = (range > 100.0 || std::abs(mean) > 100.0) ? ScalingMethod::MINMAX : ScalingMethod::ZSCORE;
        }

        ScalingParams params;
        params.method = method;
        params.mean = mean;
        params.stddev = (stdv <= 1e-12 ? 1.0 : stdv);
        params.min = minv;
        params.max = maxv;

        if (method == ScalingMethod::ZSCORE) {
            for (double& v : values) v = (v - params.mean) / params.stddev;
        } else if (method == ScalingMethod::MINMAX) {
            double range = (params.max - params.min);
            if (std::abs(range) < 1e-12) range = 1.0;
            for (double& v : values) v = (v - params.min) / range;
        }

        report.scaling[col.name] = params;
    }

    return report;
}
