#include "DeterministicHeuristics.h"

#include "CommonUtils.h"
#include "MathUtils.h"
#include "Statistics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <regex>
#include <set>
#include <unordered_map>

namespace DeterministicHeuristics {
namespace {

std::string formatDouble(double v, int prec = 2) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(prec);
    os << v;
    return os.str();
}

bool isIntegerLikeValue(double v) {
    if (!std::isfinite(v)) return false;
    return std::abs(v - std::round(v)) <= 1e-9;
}

bool isIntegerLikeColumn(const std::vector<double>& values, const MissingMask& missing) {
    size_t seen = 0;
    size_t integerLike = 0;
    const size_t n = std::min(values.size(), missing.size());
    for (size_t i = 0; i < n; ++i) {
        if (missing[i]) continue;
        if (!std::isfinite(values[i])) continue;
        ++seen;
        if (isIntegerLikeValue(values[i])) ++integerLike;
    }
    if (seen == 0) return false;
    return static_cast<double>(integerLike) / static_cast<double>(seen) >= 0.98;
}

size_t uniqueCountForColumn(const TypedColumn& col) {
    if (col.type == ColumnType::NUMERIC) {
        const auto& vals = std::get<std::vector<double>>(col.values);
        std::set<long long> uniq;
        const size_t n = std::min(vals.size(), col.missing.size());
        for (size_t i = 0; i < n; ++i) {
            if (col.missing[i]) continue;
            if (!std::isfinite(vals[i])) continue;
            uniq.insert(static_cast<long long>(std::llround(vals[i] * 100000.0)));
        }
        return uniq.size();
    }
    if (col.type == ColumnType::DATETIME) {
        const auto& vals = std::get<std::vector<int64_t>>(col.values);
        std::set<int64_t> uniq;
        const size_t n = std::min(vals.size(), col.missing.size());
        for (size_t i = 0; i < n; ++i) {
            if (col.missing[i]) continue;
            uniq.insert(vals[i]);
        }
        return uniq.size();
    }

    const auto& vals = std::get<std::vector<std::string>>(col.values);
    std::set<std::string> uniq;
    const size_t n = std::min(vals.size(), col.missing.size());
    for (size_t i = 0; i < n; ++i) {
        if (col.missing[i]) continue;
        uniq.insert(vals[i]);
    }
    return uniq.size();
}

bool isMetadataName(const std::string& name) {
    static const std::regex re("(id|date|time|coord|name|timestamp|lat|lon|lng)", std::regex::icase);
    return std::regex_search(name, re);
}

bool isTargetCandidateName(const std::string& name) {
    static const std::regex re("(score|target|label|total|class)", std::regex::icase);
    return std::regex_search(name, re);
}

double safeAbsCorr(const std::vector<double>& x, const std::vector<double>& y) {
    const ColumnStats sx = Statistics::calculateStats(x);
    const ColumnStats sy = Statistics::calculateStats(y);
    return std::abs(MathUtils::calculatePearson(x, y, sx, sy).value_or(0.0));
}

double softThreshold(double z, double gamma) {
    if (z > gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}

std::vector<int> lassoTopK(const TypedDataset& data,
                           int targetIdx,
                           const std::vector<int>& featureIdx,
                           size_t keepK) {
    std::vector<int> out;
    if (targetIdx < 0 || static_cast<size_t>(targetIdx) >= data.columns().size()) return out;
    if (featureIdx.empty()) return out;

    const auto& yRaw = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);

    std::vector<int> validFeatureIdx;
    for (int idx : featureIdx) {
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        validFeatureIdx.push_back(idx);
    }
    if (validFeatureIdx.empty()) return out;

    std::vector<size_t> rows;
    rows.reserve(data.rowCount());
    for (size_t r = 0; r < data.rowCount(); ++r) {
        if (r >= yRaw.size() || !std::isfinite(yRaw[r])) continue;
        if (data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
        bool ok = true;
        for (int idx : validFeatureIdx) {
            const auto& col = data.columns()[static_cast<size_t>(idx)];
            const auto& vals = std::get<std::vector<double>>(col.values);
            if (r >= vals.size() || r >= col.missing.size() || col.missing[r] || !std::isfinite(vals[r])) {
                ok = false;
                break;
            }
        }
        if (ok) rows.push_back(r);
    }

    const size_t n = rows.size();
    const size_t p = validFeatureIdx.size();
    if (n < 20 || p == 0) return out;

    std::vector<std::vector<double>> X(n, std::vector<double>(p, 0.0));
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const size_t r = rows[i];
        y[i] = yRaw[r];
        for (size_t j = 0; j < p; ++j) {
            const auto& vals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(validFeatureIdx[j])].values);
            X[i][j] = vals[r];
        }
    }

    double yMean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    for (double& v : y) v -= yMean;

    std::vector<double> xMean(p, 0.0), xStd(p, 1.0);
    for (size_t j = 0; j < p; ++j) {
        for (size_t i = 0; i < n; ++i) xMean[j] += X[i][j];
        xMean[j] /= static_cast<double>(n);
        double s2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            X[i][j] -= xMean[j];
            s2 += X[i][j] * X[i][j];
        }
        xStd[j] = std::sqrt(std::max(1e-12, s2 / static_cast<double>(std::max<size_t>(1, n - 1))));
        for (size_t i = 0; i < n; ++i) X[i][j] /= xStd[j];
    }

    std::vector<double> beta(p, 0.0);
    std::vector<double> pred(n, 0.0);
    const double lambda = (n < 200) ? 0.12 : 0.08;

    for (size_t iter = 0; iter < 80; ++iter) {
        for (size_t j = 0; j < p; ++j) {
            double rho = 0.0;
            double z = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double r = y[i] - (pred[i] - X[i][j] * beta[j]);
                rho += X[i][j] * r;
                z += X[i][j] * X[i][j];
            }
            const double newBeta = softThreshold(rho / static_cast<double>(n), lambda) /
                                   std::max(1e-12, z / static_cast<double>(n));
            const double delta = newBeta - beta[j];
            if (std::abs(delta) > 0.0) {
                beta[j] = newBeta;
                for (size_t i = 0; i < n; ++i) pred[i] += X[i][j] * delta;
            }
        }
    }

    std::vector<std::pair<int, double>> ranked;
    ranked.reserve(p);
    for (size_t j = 0; j < p; ++j) ranked.push_back({validFeatureIdx[j], std::abs(beta[j])});
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    for (const auto& kv : ranked) {
        if (out.size() >= keepK) break;
        if (kv.second > 1e-9) out.push_back(kv.first);
    }

    if (out.size() < std::min(keepK, ranked.size())) {
        for (const auto& kv : ranked) {
            if (out.size() >= keepK) break;
            if (std::find(out.begin(), out.end(), kv.first) == out.end()) out.push_back(kv.first);
        }
    }

    return out;
}

} // namespace

Outcome runAllPhases(const TypedDataset& data,
                     const PreprocessReport& prep,
                     int targetIdx,
                     const std::vector<int>& candidateFeatureIdx) {
    Outcome out;
    out.filteredFeatures.reserve(candidateFeatureIdx.size());

    const size_t rows = std::max<size_t>(1, data.rowCount());

    for (size_t c = 0; c < data.columns().size(); ++c) {
        const auto& col = data.columns()[c];
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name)
                                                                   : static_cast<size_t>(std::count(col.missing.begin(), col.missing.end(), true));
        const double nullRatio = static_cast<double>(missing) / static_cast<double>(rows);
        const size_t uniq = uniqueCountForColumn(col);

        const bool metadataName = isMetadataName(col.name);
        const bool targetName = isTargetCandidateName(col.name);

        bool integerLike = false;
        if (col.type == ColumnType::NUMERIC) {
            integerLike = isIntegerLikeColumn(std::get<std::vector<double>>(col.values), col.missing);
        }

        bool constant = false;
        if (col.type == ColumnType::NUMERIC) {
            const auto st = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
            constant = std::isfinite(st.variance) && st.variance <= 1e-12;
        }

        const bool idLikeType = (col.type == ColumnType::CATEGORICAL) || (col.type == ColumnType::NUMERIC && integerLike);
        const bool admin = (uniq == data.rowCount()) && idLikeType;

        std::string role = "FEATURE";
        if (admin) role = "ADMIN";
        else if (constant) role = "CONSTANT";
        else if (nullRatio > 0.40) role = "LOW_SIGNAL";
        else if (targetName) role = "TARGET_CANDIDATE";
        else if (metadataName) role = "METADATA";

        out.roleTagRows.push_back({
            col.name,
            role,
            std::to_string(uniq),
            std::to_string(missing),
            formatDouble(100.0 * nullRatio, 2) + "%"
        });
    }

    for (int idx : candidateFeatureIdx) {
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        const auto& col = data.columns()[static_cast<size_t>(idx)];

        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name)
                                                                   : static_cast<size_t>(std::count(col.missing.begin(), col.missing.end(), true));
        const double nullRatio = static_cast<double>(missing) / static_cast<double>(rows);
        const size_t uniq = uniqueCountForColumn(col);

        bool integerLike = false;
        if (col.type == ColumnType::NUMERIC) {
            integerLike = isIntegerLikeColumn(std::get<std::vector<double>>(col.values), col.missing);
        }

        bool constant = false;
        if (col.type == ColumnType::NUMERIC) {
            const auto st = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
            constant = std::isfinite(st.variance) && st.variance <= 1e-12;
        }

        const bool idLikeType = (col.type == ColumnType::CATEGORICAL) || (col.type == ColumnType::NUMERIC && integerLike);
        const bool admin = (uniq == data.rowCount()) && idLikeType;

        if (admin) {
            out.excludedReasonLines.push_back(col.name + " [!!] CRITICAL ADMIN signal dropped");
            continue;
        }
        if (constant) {
            out.excludedReasonLines.push_back(col.name + " [!!] CRITICAL constant variance=0 dropped");
            continue;
        }
        if (nullRatio > 0.40) {
            out.excludedReasonLines.push_back(col.name + " [*] STABLE low-signal (>40% missing) excluded from neural lattice");
            continue;
        }

        out.filteredFeatures.push_back(idx);
    }

    if (out.filteredFeatures.empty()) {
        out.filteredFeatures = candidateFeatureIdx;
    }

    out.rowsToFeatures = static_cast<double>(data.rowCount()) /
                         static_cast<double>(std::max<size_t>(1, out.filteredFeatures.size()));
    out.lowRatioMode = out.rowsToFeatures < 10.0;
    out.highRatioMode = out.rowsToFeatures > 100.0;

    if (out.filteredFeatures.size() > 30) {
        std::vector<int> selected = lassoTopK(data, targetIdx, out.filteredFeatures, 15);
        if (!selected.empty()) {
            out.filteredFeatures = std::move(selected);
            out.lassoGateApplied = true;
            out.lassoSelectedCount = out.filteredFeatures.size();
        }
    }

    if (targetIdx >= 0 && static_cast<size_t>(targetIdx) < data.columns().size() &&
        data.columns()[static_cast<size_t>(targetIdx)].type == ColumnType::NUMERIC &&
        out.filteredFeatures.size() >= 2) {
        const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);

        int bestFeature = -1;
        double bestCorr = 0.0;
        for (int idx : out.filteredFeatures) {
            if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
            const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
            const double c = safeAbsCorr(x, y);
            if (std::isfinite(c) && c > bestCorr) {
                bestCorr = c;
                bestFeature = idx;
            }
        }

        if (bestFeature >= 0) {
            const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(bestFeature)].values);
            const ColumnStats sx = Statistics::calculateStats(x);
            const ColumnStats sy = Statistics::calculateStats(y);
            const double r = MathUtils::calculatePearson(x, y, sx, sy).value_or(0.0);
            const auto fit = MathUtils::simpleLinearRegression(x, y, sx, sy, r);

            std::vector<double> residual;
            residual.reserve(y.size());
            for (size_t i = 0; i < y.size() && i < x.size(); ++i) {
                if (!std::isfinite(y[i]) || !std::isfinite(x[i])) {
                    residual.push_back(std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                residual.push_back(y[i] - (fit.first * x[i] + fit.second));
            }

            int hiddenDriver = -1;
            double hiddenCorr = 0.0;
            for (int idx : out.filteredFeatures) {
                if (idx == bestFeature) continue;
                if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
                const auto& xv = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
                const double c = safeAbsCorr(xv, residual);
                if (std::isfinite(c) && c > hiddenCorr) {
                    hiddenCorr = c;
                    hiddenDriver = idx;
                }
            }

            if (hiddenDriver >= 0 && hiddenCorr >= 0.25) {
                out.residualNarrative = "[*] STABLE Hidden Driver: " +
                    data.columns()[static_cast<size_t>(hiddenDriver)].name +
                    " explains residual error after primary predictor " +
                    data.columns()[static_cast<size_t>(bestFeature)].name +
                    " (|r_residual|=" + formatDouble(hiddenCorr, 3) + ").";
            }
        }
    }

    if (out.lowRatioMode) {
        out.badgeNarratives.push_back("[!!] CRITICAL Overfit Guardrail: Rows:Features=" +
                                      formatDouble(out.rowsToFeatures, 2) +
                                      " (<10), forcing shallow lattice profile.");
    } else if (out.highRatioMode) {
        out.badgeNarratives.push_back("[*] STABLE Capacity Mode: Rows:Features=" +
                                      formatDouble(out.rowsToFeatures, 2) +
                                      " (>100), expressive lattice profile allowed.");
    }

    if (out.lassoGateApplied) {
        out.badgeNarratives.push_back("[*] STABLE Lasso Gate: features pruned to top " + std::to_string(out.lassoSelectedCount) + " weighted predictors.");
    }

    if (!out.residualNarrative.empty()) {
        out.badgeNarratives.push_back(out.residualNarrative);
    }

    return out;
}

} // namespace DeterministicHeuristics
