#include "Preprocessor.h"
#include "CommonUtils.h"
#include "SeldonExceptions.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <unordered_map>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
using NumVec = std::vector<double>;
using StrVec = std::vector<std::string>;
using TimeVec = std::vector<int64_t>;
using MissingMask = ::MissingMask;

int64_t floorDiv(int64_t value, int64_t divisor) {
    int64_t q = value / divisor;
    int64_t r = value % divisor;
    if (r != 0 && ((r > 0) != (divisor > 0))) {
        --q;
    }
    return q;
}

std::tuple<int, unsigned, unsigned> civilFromDays(int64_t z) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int y = static_cast<int>(yoe) + static_cast<int>(era) * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
    return {y, m, d};
}

int dayOfWeekFromUnixSeconds(int64_t unixSeconds) {
    const int64_t days = floorDiv(unixSeconds, 86400);
    int weekday = static_cast<int>((days + 4) % 7);
    if (weekday < 0) weekday += 7;
    return weekday; // 0=Sunday, 6=Saturday
}

std::string uniqueColumnName(const TypedDataset& data, const std::string& base) {
    if (data.findColumnIndex(base) < 0) return base;
    int suffix = 2;
    while (true) {
        std::string candidate = base + "_" + std::to_string(suffix);
        if (data.findColumnIndex(candidate) < 0) return candidate;
        ++suffix;
    }
}

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

void addTemporalDateFeatures(TypedDataset& data, const AutoConfig& config) {
    const auto datetimeIndices = data.datetimeColumnIndices();
    if (datetimeIndices.empty()) return;

    std::vector<TypedColumn> engineered;
    engineered.reserve(datetimeIndices.size() * 5);

    for (size_t idx : datetimeIndices) {
        const auto& srcCol = data.columns()[idx];
        const auto& rawValues = std::get<TimeVec>(srcCol.values);
        const MissingMask& rawMissing = srcCol.missing;

        std::string strategy = "auto";
        auto it = config.columnImputation.find(srcCol.name);
        if (it != config.columnImputation.end()) strategy = it->second;
        validateImputationStrategy(strategy, ColumnType::DATETIME, srcCol.name);

        TimeVec imputedValues = rawValues;
        MissingMask imputedMissing = rawMissing;
        imputeDatetime(imputedValues, imputedMissing, strategy);

        std::vector<double> dayOfWeek(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> month(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> dayOfYear(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> quarter(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> isWeekend(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        MissingMask derivedMissing(imputedValues.size(), static_cast<uint8_t>(0));

        for (size_t r = 0; r < imputedValues.size(); ++r) {
            if (r < imputedMissing.size() && imputedMissing[r]) {
                derivedMissing[r] = static_cast<uint8_t>(1);
                continue;
            }

            const int64_t ts = imputedValues[r];
            const int weekday = dayOfWeekFromUnixSeconds(ts);
            const int64_t days = floorDiv(ts, 86400);
            const auto [year, monthValue, day] = civilFromDays(days);
            const bool leap = ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0));
            static const int cumDaysNorm[12] = {0,31,59,90,120,151,181,212,243,273,304,334};
            static const int cumDaysLeap[12] = {0,31,60,91,121,152,182,213,244,274,305,335};
            const int* cum = leap ? cumDaysLeap : cumDaysNorm;

            dayOfWeek[r] = static_cast<double>(weekday);
            month[r] = static_cast<double>(monthValue);
            dayOfYear[r] = static_cast<double>(cum[std::max(1u, monthValue) - 1] + static_cast<int>(day));
            quarter[r] = static_cast<double>(((std::max(1u, monthValue) - 1) / 3) + 1);
            isWeekend[r] = (weekday == 0 || weekday == 6) ? 1.0 : 0.0;
        }

        TypedColumn dowCol;
        dowCol.name = uniqueColumnName(data, srcCol.name + "_DayOfWeek");
        dowCol.type = ColumnType::NUMERIC;
        dowCol.values = std::move(dayOfWeek);
        dowCol.missing = derivedMissing;
        engineered.push_back(std::move(dowCol));

        TypedColumn monthCol;
        monthCol.name = uniqueColumnName(data, srcCol.name + "_Month");
        monthCol.type = ColumnType::NUMERIC;
        monthCol.values = std::move(month);
        monthCol.missing = derivedMissing;
        engineered.push_back(std::move(monthCol));

        TypedColumn weekendCol;
        weekendCol.name = uniqueColumnName(data, srcCol.name + "_IsWeekend");
        weekendCol.type = ColumnType::NUMERIC;
        weekendCol.values = std::move(isWeekend);
        weekendCol.missing = derivedMissing;
        engineered.push_back(std::move(weekendCol));

        TypedColumn doyCol;
        doyCol.name = uniqueColumnName(data, srcCol.name + "_DayOfYear");
        doyCol.type = ColumnType::NUMERIC;
        doyCol.values = std::move(dayOfYear);
        doyCol.missing = derivedMissing;
        engineered.push_back(std::move(doyCol));

        TypedColumn quarterCol;
        quarterCol.name = uniqueColumnName(data, srcCol.name + "_Quarter");
        quarterCol.type = ColumnType::NUMERIC;
        quarterCol.values = std::move(quarter);
        quarterCol.missing = std::move(derivedMissing);
        engineered.push_back(std::move(quarterCol));
    }

    auto& columns = data.columns();
    columns.insert(columns.end(), std::make_move_iterator(engineered.begin()), std::make_move_iterator(engineered.end()));
}

void addAutoNumericFeatureEngineering(TypedDataset& data, const AutoConfig& config) {
    auto numericIdx = data.numericColumnIndices();
    if (numericIdx.empty()) return;

    const size_t maxBase = std::min<size_t>(numericIdx.size(), std::max<size_t>(2, config.featureEngineeringMaxBase));
    std::vector<size_t> base(numericIdx.begin(), numericIdx.begin() + maxBase);

    std::vector<TypedColumn> engineered;
    engineered.reserve(maxBase * (1 + static_cast<size_t>(std::max(1, config.featureEngineeringDegree))));
    const size_t maxGenerated = std::max<size_t>(16, config.featureEngineeringMaxGeneratedColumns);
    bool capReached = false;

    auto canAddMore = [&]() {
        return engineered.size() < maxGenerated;
    };

    const int maxDegree = std::clamp(config.featureEngineeringDegree, 1, 4);

    for (size_t idx : base) {
        if (!canAddMore()) {
            capReached = true;
            break;
        }
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<NumVec>(col.values);

        if (config.featureEngineeringEnablePoly) {
            for (int p = 2; p <= maxDegree; ++p) {
                if (!canAddMore()) {
                    capReached = true;
                    break;
                }
                std::vector<double> poly(vals.size(), 0.0);
                for (size_t i = 0; i < vals.size(); ++i) {
                    poly[i] = std::pow(vals[i], static_cast<double>(p));
                }
                TypedColumn c;
                c.name = uniqueColumnName(data, col.name + "_pow" + std::to_string(p));
                c.type = ColumnType::NUMERIC;
                c.values = std::move(poly);
                c.missing = col.missing;
                engineered.push_back(std::move(c));
            }
        }

        if (config.featureEngineeringEnableLog) {
            if (!canAddMore()) {
                capReached = true;
                break;
            }
            std::vector<double> logSigned(vals.size(), 0.0);
            for (size_t i = 0; i < vals.size(); ++i) {
                const double v = vals[i];
                const double mag = std::log1p(std::abs(v));
                logSigned[i] = (v < 0.0) ? -mag : mag;
            }
            TypedColumn c;
            c.name = uniqueColumnName(data, col.name + "_log1p_abs");
            c.type = ColumnType::NUMERIC;
            c.values = std::move(logSigned);
            c.missing = col.missing;
            engineered.push_back(std::move(c));
        }
    }

    if (config.featureEngineeringEnablePoly && !capReached) {
        for (size_t i = 0; i < base.size(); ++i) {
            if (!canAddMore()) {
                capReached = true;
                break;
            }
            for (size_t j = i + 1; j < base.size(); ++j) {
                if (!canAddMore()) {
                    capReached = true;
                    break;
                }
                const auto& a = data.columns()[base[i]];
                const auto& b = data.columns()[base[j]];
                const auto& av = std::get<NumVec>(a.values);
                const auto& bv = std::get<NumVec>(b.values);
                const size_t n = std::min(av.size(), bv.size());
                std::vector<double> inter(n, 0.0);
                MissingMask miss(n, static_cast<uint8_t>(0));
                for (size_t r = 0; r < n; ++r) {
                    if (a.missing[r] || b.missing[r]) {
                        miss[r] = static_cast<uint8_t>(1);
                        continue;
                    }
                    inter[r] = av[r] * bv[r];
                }

                TypedColumn c;
                c.name = uniqueColumnName(data, a.name + "_x_" + b.name);
                c.type = ColumnType::NUMERIC;
                c.values = std::move(inter);
                c.missing = std::move(miss);
                engineered.push_back(std::move(c));
            }
            if (capReached) break;
        }
    }

    auto& cols = data.columns();
    cols.insert(cols.end(), std::make_move_iterator(engineered.begin()), std::make_move_iterator(engineered.end()));
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

std::vector<bool> detectOutliersModifiedZObserved(const NumVec& values, const MissingMask& missing, double zThreshold) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    std::vector<size_t> obsIdx;
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }
    if (observed.size() < 4) return flags;

    const double med = CommonUtils::medianByNth(observed);
    NumVec absDev(observed.size(), 0.0);
    for (size_t i = 0; i < observed.size(); ++i) absDev[i] = std::abs(observed[i] - med);
    const double mad = CommonUtils::medianByNth(absDev);
    if (mad <= 1e-12) return flags;

    for (size_t i = 0; i < observed.size(); ++i) {
        const double mz = 0.6745 * (observed[i] - med) / mad;
        if (std::abs(mz) > zThreshold) flags[obsIdx[i]] = true;
    }
    return flags;
}

std::vector<bool> detectOutliersAdjustedBoxplotObserved(const NumVec& values, const MissingMask& missing, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    std::vector<size_t> obsIdx;
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }
    if (observed.size() < 8) return flags;

    const double q1 = CommonUtils::quantileByNth(observed, 0.25);
    const double med = CommonUtils::quantileByNth(observed, 0.50);
    const double q3 = CommonUtils::quantileByNth(observed, 0.75);
    const double iqr = q3 - q1;
    if (iqr <= 1e-12) return flags;

    const double bowleySkew = (q3 + q1 - 2.0 * med) / iqr;
    const double loFactor = iqrMultiplier * std::exp(-4.0 * bowleySkew);
    const double hiFactor = iqrMultiplier * std::exp(3.0 * bowleySkew);
    const double lo = q1 - loFactor * iqr;
    const double hi = q3 + hiFactor * iqr;

    for (size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] < lo || observed[i] > hi) flags[obsIdx[i]] = true;
    }
    return flags;
}

std::vector<bool> detectOutliersLOFObserved(const NumVec& values, const MissingMask& missing) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    std::vector<size_t> obsIdx;
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }
    if (observed.size() < 12) return flags;

    const size_t n = observed.size();
    constexpr size_t kLofMaxRows = 120000;
    if (n > kLofMaxRows) {
        return detectOutliersModifiedZObserved(values, missing, 3.5);
    }

    const size_t k = std::min<size_t>(10, std::max<size_t>(3, n / 12));

    std::vector<std::pair<double, size_t>> sorted;
    sorted.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        sorted.push_back({observed[i], i});
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first < b.first;
    });

    std::vector<size_t> rankByIndex(n, 0);
    for (size_t rank = 0; rank < n; ++rank) {
        rankByIndex[sorted[rank].second] = rank;
    }

    std::vector<std::vector<size_t>> nbr(n);
    std::vector<double> kdist(n, 0.0);
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        const size_t rank = rankByIndex[i];
        size_t left = rank;
        size_t right = rank + 1;
        nbr[i].reserve(k);

        while (nbr[i].size() < k && (left > 0 || right < n)) {
            const bool hasLeft = left > 0;
            const bool hasRight = right < n;

            if (hasLeft && hasRight) {
                const double dl = std::abs(observed[i] - sorted[left - 1].first);
                const double dr = std::abs(observed[i] - sorted[right].first);
                if (dl <= dr) {
                    --left;
                    nbr[i].push_back(sorted[left].second);
                } else {
                    nbr[i].push_back(sorted[right].second);
                    ++right;
                }
            } else if (hasLeft) {
                --left;
                nbr[i].push_back(sorted[left].second);
            } else {
                nbr[i].push_back(sorted[right].second);
                ++right;
            }
        }

        double kd = 0.0;
        for (size_t nb : nbr[i]) {
            kd = std::max(kd, std::abs(observed[i] - observed[nb]));
        }
        kdist[i] = kd;
    }

    std::vector<double> lrd(n, 0.0);
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        double reach = 0.0;
        for (size_t nb : nbr[i]) {
            reach += std::max(kdist[nb], std::abs(observed[i] - observed[nb]));
        }
        lrd[i] = (reach <= 1e-12) ? 0.0 : (static_cast<double>(k) / reach);
    }

    std::vector<double> lof(n, 1.0);
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        if (lrd[i] <= 1e-12) {
            lof[i] = 1.0;
            continue;
        }
        double ratioSum = 0.0;
        for (size_t nb : nbr[i]) ratioSum += (lrd[nb] / lrd[i]);
        lof[i] = ratioSum / static_cast<double>(k);
    }

    const double q3 = CommonUtils::quantileByNth(lof, 0.75);
    const double q1 = CommonUtils::quantileByNth(lof, 0.25);
    const double threshold = std::max(1.5, q3 + 1.5 * (q3 - q1));
    for (size_t i = 0; i < n; ++i) {
        if (lof[i] > threshold) flags[obsIdx[i]] = true;
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

    addTemporalDateFeatures(data, config);
    addAutoNumericFeatureEngineering(data, config);

    // Outlier flags are computed from observed raw numeric values before imputation.
    std::unordered_map<std::string, std::vector<bool>> detectedFlags;
    for (const auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        const auto& values = std::get<NumVec>(col.values);

        const std::string method = CommonUtils::toLower(config.outlierMethod);
        std::vector<bool> flags;
        if (method == "zscore") {
            flags = detectOutliersZObserved(values, col.missing, config.tuning.outlierZThreshold);
        } else if (method == "modified_zscore") {
            flags = detectOutliersModifiedZObserved(values, col.missing, config.tuning.outlierZThreshold);
        } else if (method == "adjusted_boxplot") {
            flags = detectOutliersAdjustedBoxplotObserved(values, col.missing, config.tuning.outlierIqrMultiplier);
        } else if (method == "lof") {
            flags = detectOutliersLOFObserved(values, col.missing);
        } else {
            flags = detectOutliersIQRObserved(values, col.missing, config.tuning.outlierIqrMultiplier);
        }

        detectedFlags[col.name] = flags;
        if (config.storeOutlierFlagsInReport) {
            report.outlierFlags[col.name] = flags;
        }
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
