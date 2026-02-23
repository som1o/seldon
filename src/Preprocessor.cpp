#include "Preprocessor.h"
#include "CommonUtils.h"
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
using MissingMask = ::MissingMask;

void validateImputationStrategy(const std::string& strategy, ColumnType type, const std::string& columnName) {
    const std::string m = CommonUtils::toLower(strategy);
    bool valid = false;
    if (type == ColumnType::NUMERIC) {
        valid = (m == "auto" || m == "mean" || m == "median" || m == "zero" || m == "interpolate");
    } else if (type == ColumnType::CATEGORICAL) {
        valid = (m == "auto" || m == "mode");
    } else {
        valid = (m == "auto" || m == "interpolate");
    }
    if (!valid) {
        throw Seldon::ConfigurationException("Invalid imputation strategy for column '" + columnName + "': " + strategy);
    }
}

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

void imputeNumeric(NumVec& values, MissingMask& missing, const std::string& method) {
    NumVec valid;
    valid.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && std::isfinite(values[i])) valid.push_back(values[i]);

    double fill = 0.0;
    std::string m = CommonUtils::toLower(method);
    if (m == "median") fill = CommonUtils::medianByNth(valid);
    else if (m == "zero") fill = 0.0;
    else {
        double sum = 0.0;
        for (double v : valid) sum += v;
        fill = valid.empty() ? 0.0 : (sum / valid.size());
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) {
            values[i] = fill;
            missing[i] = static_cast<uint8_t>(0);
        }
    }
}

void interpolateSeries(NumVec& values, MissingMask& missing) {
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
                    missing[lastKnown + static_cast<int>(k)] = static_cast<uint8_t>(0);
                }
            }
            lastKnown = static_cast<int>(i);
        }
    }

    // forward/back fill edges
    for (size_t i = 1; i < values.size(); ++i) if (missing[i] && !missing[i - 1]) { values[i] = values[i - 1]; missing[i] = static_cast<uint8_t>(0); }
    for (size_t i = values.size(); i-- > 1;) {
        if (missing[i - 1] && !missing[i]) {
            values[i - 1] = values[i];
            missing[i - 1] = static_cast<uint8_t>(0);
        }
    }
}

void imputeCategorical(StrVec& values, MissingMask& missing, const std::string& method) {
    std::unordered_map<std::string, size_t> freq;
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && !values[i].empty()) freq[values[i]]++;

    std::string mode = "Unknown";
    size_t best = 0;
    for (const auto& kv : freq) {
        if (kv.second > best) { best = kv.second; mode = kv.first; }
    }

    std::string m = CommonUtils::toLower(method);
    std::string fill = (m == "mode" || m == "auto") ? mode : mode;
    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || values[i].empty()) {
            values[i] = fill;
            missing[i] = static_cast<uint8_t>(0);
        }
    }
}

void imputeDatetime(TimeVec& values, MissingMask& missing, const std::string& method) {
    std::string m = CommonUtils::toLower(method);
    if (m == "interpolate" || m == "auto") {
        NumVec temp(values.begin(), values.end());
        interpolateSeries(temp, missing);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<int64_t>(temp[i]);
    } else {
        int64_t fill = 0;
        for (size_t i = 0; i < values.size(); ++i) if (!missing[i]) { fill = values[i]; break; }
        for (size_t i = 0; i < values.size(); ++i) if (missing[i]) { values[i] = fill; missing[i] = static_cast<uint8_t>(0); }
    }
}

std::vector<bool> detectOutliersIQR(const NumVec& values, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    if (values.size() < 4) return flags;

    const double q1 = CommonUtils::quantileByNth(values, 0.25);
    const double q3 = CommonUtils::quantileByNth(values, 0.75);
    double iqr = q3 - q1;
    double lo = q1 - iqrMultiplier * iqr;
    double hi = q3 + iqrMultiplier * iqr;

    for (size_t i = 0; i < values.size(); ++i) flags[i] = (values[i] < lo || values[i] > hi);
    return flags;
}

std::vector<bool> detectOutliersIQRObserved(const NumVec& values, const MissingMask& missing, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    observed.reserve(values.size());
    std::vector<size_t> observedIdx;
    observedIdx.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i]) continue;
        if (!std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        observedIdx.push_back(i);
    }
    if (observed.size() < 4) return flags;

    std::vector<bool> obsFlags = detectOutliersIQR(observed, iqrMultiplier);
    for (size_t i = 0; i < obsFlags.size() && i < observedIdx.size(); ++i) {
        flags[observedIdx[i]] = obsFlags[i];
    }
    return flags;
}

std::vector<bool> detectOutliersZ(const NumVec& values, double zThreshold) {
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
        flags[i] = z > zThreshold;
    }
    return flags;
}

std::vector<bool> detectOutliersZObserved(const NumVec& values, const MissingMask& missing, double zThreshold) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    observed.reserve(values.size());
    std::vector<size_t> observedIdx;
    observedIdx.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i]) continue;
        if (!std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        observedIdx.push_back(i);
    }
    if (observed.size() < 3) return flags;

    std::vector<bool> obsFlags = detectOutliersZ(observed, zThreshold);
    for (size_t i = 0; i < obsFlags.size() && i < observedIdx.size(); ++i) {
        flags[observedIdx[i]] = obsFlags[i];
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

    // Outlier flags are computed from observed raw numeric values before imputation.
    std::unordered_map<std::string, std::vector<bool>> detectedFlags;
    for (const auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        const auto& values = std::get<NumVec>(col.values);

        std::vector<bool> flags = (CommonUtils::toLower(config.outlierMethod) == "zscore")
            ? detectOutliersZObserved(values, col.missing, config.tuning.outlierZThreshold)
            : detectOutliersIQRObserved(values, col.missing, config.tuning.outlierIqrMultiplier);

        detectedFlags[col.name] = flags;
        report.outlierFlags[col.name] = flags;
        report.outlierCounts[col.name] = std::count(flags.begin(), flags.end(), true);
    }

    // missing counts + imputation
    for (auto& col : data.columns()) {
        size_t missingCount = std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1));
        report.missingCounts[col.name] = missingCount;

        std::string strategy = "auto";
        auto it = config.columnImputation.find(col.name);
        if (it != config.columnImputation.end()) strategy = it->second;
        validateImputationStrategy(strategy, col.type, col.name);

        if (col.type == ColumnType::NUMERIC) {
            auto& values = std::get<NumVec>(col.values);
            if (CommonUtils::toLower(strategy) == "interpolate") interpolateSeries(values, col.missing);
            else imputeNumeric(values, col.missing, strategy);
        } else if (col.type == ColumnType::CATEGORICAL) {
            auto& values = std::get<StrVec>(col.values);
            imputeCategorical(values, col.missing, strategy.empty() ? "mode" : strategy);
        } else {
            auto& values = std::get<TimeVec>(col.values);
            imputeDatetime(values, col.missing, strategy.empty() ? "interpolate" : strategy);
        }
    }

    // Outlier flow remains two-phase: detect first, then apply one action.
    MissingMask keep(data.rowCount(), static_cast<uint8_t>(1));
    std::string action = CommonUtils::toLower(config.outlierAction);
    if (action == "remove") {
        for (const auto& col : data.columns()) {
            if (col.type != ColumnType::NUMERIC) continue;
            const auto fit = detectedFlags.find(col.name);
            if (fit == detectedFlags.end()) continue;
            const auto& flags = fit->second;
            for (size_t r = 0; r < flags.size(); ++r) {
                if (flags[r]) keep[r] = static_cast<uint8_t>(0);
            }
        }
        data.removeRows(keep);
    } else if (action == "cap") {
        for (auto& col : data.columns()) {
            if (col.type != ColumnType::NUMERIC) continue;
            auto fit = detectedFlags.find(col.name);
            if (fit == detectedFlags.end()) continue;
            auto& values = std::get<NumVec>(col.values);
            capOutliers(values, fit->second);
        }
    }

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
        std::string pref = CommonUtils::toLower(config.scalingMethod);
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
